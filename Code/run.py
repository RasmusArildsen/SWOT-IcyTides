"""Run one restartable step of the November 190/191-acquisition analysis.

Use ``plan`` to see the order without opening data. For an indexed step, choose
one ``--index`` (for an HPC job) or ``--all-parts`` (a sequential local run).
"""

from pathlib import Path
import argparse
import json
import os
import sys

CODE_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(CODE_DIR / "processing"), str(CODE_DIR.parent / "src")]
from layout import (
    DEFAULT_ACQUISITIONS,
    acquisition_counts,
    band_indices,
    fit_part_indices,
)

STAGES = (
    "plan",
    "grid",
    "geometry",
    "row",
    "band",
    "independent",
    "collect",
    "gls-probe",
    "gls-prepare",
    "gls-node",
    "gls-edge",
    "gls-assemble",
    "gls-solve",
    "figure-products",
)
PART_STAGES = {"row", "band", "independent", "gls-node", "gls-edge"}


def resources(path):
    """Read local input paths, resolving them relative to the settings file.

    This public helper is also used by the native, reference and plot scripts.
    Keys beginning with an underscore are explanatory notes, not paths.
    """
    settings = json.loads(path.read_text())
    return {
        key: str((path.parent / Path(value)).expanduser().resolve())
        for key, value in settings.items()
        if not key.startswith("_")
    }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument(
        "--resources",
        type=Path,
        default=CODE_DIR / "aux_files/resources.json",
        help="Local raw/model/reference paths; see resources.example.json",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("work/november191"))
    parser.add_argument(
        "--acquisitions",
        type=int,
        choices=acquisition_counts(),
        default=DEFAULT_ACQUISITIONS,
    )
    parts = parser.add_mutually_exclusive_group()
    parts.add_argument("--index", type=int, help="Run one numbered part")
    parts.add_argument(
        "--all-parts",
        action="store_true",
        help="Run every part sequentially",
    )
    args = parser.parse_args(argv)
    if args.stage in PART_STAGES and args.index is None and not args.all_parts:
        parser.error("Choose --index or --all-parts for " + args.stage)
    if args.stage not in PART_STAGES and (args.index is not None or args.all_parts):
        parser.error("This step has no numbered parts: " + args.stage)
    return args


def configure_runtime(work_dir, acquisitions):
    """Set run paths and date count before importing scientific modules."""
    independent_dir = work_dir / f"independent{acquisitions}"
    os.environ["S4IS_INDEPENDENT_ROOT"] = str(independent_dir)
    os.environ["S4IS_ACQUISITIONS"] = str(acquisitions)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return independent_dir


def print_plan():
    """Describe the workflow without reading inputs or submitting jobs."""
    print(
        "Raw inputs and reference preparation (see Code/README.md)\n"
        "  1. Download pinned Raster/PIXC files; build the sparse raw cube.\n"
        "  2. Prepare BedMachine/REMA; run native region and residual repairs.\n"
        "Analysis steps\n"
        "  3. geometry -> row -> band: shared grid and observation bands from the geometry.\n"
        "  4. independent -> collect: UTide-Huber fit parts from the geometry.\n"
        "  5. gls-probe -> gls-prepare -> gls-node -> gls-edge.\n"
        "  6. gls-assemble -> gls-solve: verified all-free connected fit.\n"
        "  7. figure-products -> Figure1/2/3 plotting scripts.\n\n"
        "Indexed steps accept --index N or --all-parts.\n"
        "Settings: Code/aux_files/analysis.json\n"
        "Input paths: Code/aux_files/resources.example.json\n"
        "No data processing or downloads performed by plan."
    )


def part_indices(stage, acquisitions, gls_dir, index=None, geometry=None):
    """Select every recorded part, or validate a single requested index."""
    if stage == "row":
        indices = range(acquisitions)
    elif stage == "band":
        indices = band_indices(geometry)
    elif stage == "independent":
        indices = fit_part_indices(geometry)
    else:
        prepared = json.loads((gls_dir / "prepared.json").read_text())
        indices = range(prepared["node_parts" if stage == "gls-node" else "edge_parts"])
    if index is None:
        return indices
    if index not in indices:
        raise ValueError(f"{stage} index must be one of {indices}")
    return [index]


def freeze_analysis_inputs(analysis, independent_dir, gls_dir=None):
    """Freeze completed inputs before fitting or combining any observations."""
    files = [independent_dir / "geometry.h5"]
    files += [
        independent_dir / "inputs" / f"{part:03d}.h5"
        for part in band_indices(independent_dir / "geometry.h5")
    ]
    # The numbered filtering step saves these masks. Freeze them with the
    # observations so a later mask edit cannot change the fitted support.
    admission = independent_dir / "admission"
    if admission.is_dir():
        files += [
            admission / f"{part:03d}.h5"
            for part in band_indices(independent_dir / "geometry.h5")
        ]
    receipt_dir = independent_dir
    if gls_dir is not None:
        files += [independent_dir / "independent.h5"]
        files += [
            independent_dir / "fits" / f"{part:03d}.h5"
            for part in fit_part_indices(independent_dir / "geometry.h5")
        ]
        receipt_dir = gls_dir
    analysis.freeze_inputs(receipt_dir / "source_inventory.json", files)


def run_stage(args, input_paths, work_dir, independent_dir):
    """Call the scientific step explicitly; each function writes its own result."""
    stage = args.stage
    gls_dir = work_dir / "gls"

    if stage == "grid":
        import numpy as np
        import prepare

        work_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            work_dir / "grid.npz",
            x=prepare.axis("x100"),
            y=prepare.axis("y100"),
        )
        return

    if stage == "geometry":
        import prepare

        independent_dir.mkdir(parents=True, exist_ok=True)
        prepare.geometry(
            independent_dir,
            input_paths["analysis_bedmachine_mask"],
            input_paths["cats_model"],
            input_paths["native_bedmachine_mask"],
            nt=args.acquisitions,
        )
        return

    if stage in {"row", "band"}:
        import prepare

        for index in part_indices(
            stage,
            args.acquisitions,
            gls_dir,
            args.index,
            independent_dir / "geometry.h5",
        ):
            if stage == "row":
                prepare.row(index, input_paths, independent_dir)
            else:
                prepare.band(index, input_paths, independent_dir, args.acquisitions)
        return

    import analysis

    if stage in {"independent", "collect"}:
        freeze_analysis_inputs(analysis, independent_dir)
        if stage == "independent":
            for index in part_indices(
                stage,
                args.acquisitions,
                gls_dir,
                args.index,
                independent_dir / "geometry.h5",
            ):
                analysis.fit_part(index)
        else:
            analysis.collect()
        return

    if stage == "figure-products":
        import figure_products

        figure_products.build(
            gls_dir / "native_tides.h5",
            independent_dir,
            input_paths,
            work_dir / "figure_products",
        )
        return

    import gls

    gls.ROOT = gls_dir
    gls.NT = args.acquisitions
    if stage != "gls-probe":
        freeze_analysis_inputs(analysis, independent_dir, gls_dir)

    if stage == "gls-probe":
        gls_dir.mkdir(parents=True, exist_ok=True)
        gls.probe()
    elif stage == "gls-prepare":
        gls.prepare()
    elif stage == "gls-node":
        for index in part_indices(
            stage,
            args.acquisitions,
            gls_dir,
            args.index,
            independent_dir / "geometry.h5",
        ):
            gls.node_part(index)
    elif stage == "gls-edge":
        for index in part_indices(
            stage,
            args.acquisitions,
            gls_dir,
            args.index,
            independent_dir / "geometry.h5",
        ):
            gls.edge_part(index)
    elif stage == "gls-assemble":
        gls.assemble()
    elif stage == "gls-solve":
        gls.solve()


def main(argv=None):
    args = parse_arguments(argv)
    work_dir = args.work_dir.resolve()
    independent_dir = configure_runtime(work_dir, args.acquisitions)
    if args.stage == "plan":
        print_plan()
        return
    run_stage(args, resources(args.resources), work_dir, independent_dir)


if __name__ == "__main__":
    main()

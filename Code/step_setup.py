"""Shared paths and command-line options for the six numbered scripts."""

from pathlib import Path
import argparse
import json
import os
import sys

CODE = Path(__file__).resolve().parent
REPOSITORY = CODE.parent
sys.path[:0] = [str(CODE / "processing"), str(REPOSITORY / "src")]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from layout import DEFAULT_ACQUISITIONS, acquisition_counts


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--inputs", type=Path, default=CODE / "input_paths.json")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--acquisitions",
        type=int,
        choices=acquisition_counts(),
        default=DEFAULT_ACQUISITIONS,
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Describe the steps without reading data"
    )
    return parser


def setup(args):
    """Read external input paths and put all generated files in one work folder."""
    args.inputs = args.inputs.resolve()
    external = json.loads(args.inputs.read_text())
    required = {
        "raster_raw",
        "native_raw",
        "atl06_raw",
        "bedmachine_source",
        "rema_source",
        "cats_model",
        "aviso_cache",
        "legacy_reference_surface",
        "optical_mosaic",
        "land_file",
        "lima_mosaic",
    }
    missing = sorted(required - external.keys())
    if missing:
        raise ValueError("Missing input paths: " + ", ".join(missing))
    args.paths = {}
    for name, value in external.items():
        if name.startswith("_"):
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = args.inputs.parent / path
        args.paths[name] = path.resolve()

    args.work_dir = (
        args.work_dir or REPOSITORY / f"work/november{args.acquisitions}"
    ).resolve()
    args.independent = args.work_dir / f"independent{args.acquisitions}"
    args.reference = args.work_dir / "reference"
    args.native = args.work_dir / "native_phase"
    args.ancestor = args.work_dir / "phase_reference"
    args.paths.update(
        raw_cube=args.work_dir / "raw_cube.h5",
        region_layer=args.work_dir / "phase_region190.h5",
        residual_layer=args.work_dir / "phase_residual190.h5",
        reference_bands=args.work_dir / "reference_bands",
        analysis_bedmachine_mask=args.reference / "bedmachine_analysis.npz",
        native_bedmachine_mask=args.reference / "bedmachine_native.npz",
        native_rema=args.reference / "rema_100m.nc",
        analysis_geometry=args.independent / "geometry.h5",
        native_queue=CODE / "aux_files/native_inventory190.json",
        legacy_coverage=CODE / "aux_files/legacy_coverage190.json",
        legacy_reference_geometry=args.ancestor / "geometry.h5",
        legacy_native_archives=args.ancestor / "native_archives",
        phase2_reference_bands=args.ancestor / "bands",
        first_native_archives=args.native / "shelf_native_passes",
        atl06_reference=args.reference / "atl06_100m/atl06_dac_100m.nc",
    )
    args.resources = args.work_dir / "resources.json"
    if not args.dry_run:
        contents = {name: str(path) for name, path in args.paths.items()}
        if args.resources.exists():
            if json.loads(args.resources.read_text()) != contents:
                raise ValueError("Input paths changed. Choose a new --work-dir.")
        else:
            args.work_dir.mkdir(parents=True, exist_ok=True)
            temporary = args.resources.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(contents, indent=2) + "\n")
            temporary.replace(args.resources)
    return args


def existing_file(path):
    """Reuse supplied/generated reference files; never overwrite them silently."""
    if path.is_file():
        print(f"Using {path}", flush=True)
        return True
    return False

"""Shared paths and command-line options for the processing scripts."""

from pathlib import Path
import argparse
import json
import os
import sys

CODE = Path(__file__).resolve().parent
REPOSITORY = CODE.parent
sys.path.insert(0, str(CODE / "processing"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from layout import (
    DEFAULT_ACQUISITIONS,
    geometry_acquisition_ids,
    selected_acquisitions,
)


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--inputs", type=Path, default=CODE / "input_paths.json")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--acquisitions",
        type=int,
        metavar="N",
        help="Use the first N acquisitions in the catalogue (default: all)",
        default=DEFAULT_ACQUISITIONS,
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Describe the steps without reading data"
    )
    return parser


def read_input_paths(filename, required):
    """Read the requested paths, relative to the input file when necessary."""
    filename = Path(filename).resolve()
    external = json.loads(filename.read_text())
    missing = sorted(required - external.keys())
    if missing:
        raise ValueError("Missing input paths: " + ", ".join(missing))
    paths = {}
    for name, value in external.items():
        if not name.startswith("_"):
            path = Path(value).expanduser()
            paths[name] = (filename.parent / path).resolve()
    return paths


def setup(args):
    """Read external input paths and put all generated files in one work folder."""
    catalogue = selected_acquisitions(args.acquisitions)
    args.inputs = args.inputs.resolve()
    required = {
        "raster_raw",
        "atl06_raw",
        "bedmachine_source",
        "rema_source",
        "cats_model",
        "aviso_cache",
        "optical_mosaic",
        "lima_mosaic",
    }
    args.paths = read_input_paths(args.inputs, required)

    args.work_dir = (args.work_dir or REPOSITORY / "work").resolve()
    args.independent = args.work_dir / "independent"
    args.reference = args.work_dir / "reference"
    repair_dir = args.paths.pop("phase_repairs", args.work_dir)
    args.paths.update(
        raw_cube=args.work_dir / "raw_cube.h5",
        finished_cube=args.work_dir / "finished_cube.h5",
        region_layer=repair_dir / "phase_region.h5",
        residual_layer=repair_dir / "phase_residual.h5",
        reference_bands=args.work_dir / "reference_bands",
        analysis_bedmachine_mask=args.reference / "bedmachine_analysis.npz",
        native_bedmachine_mask=args.reference / "bedmachine_native.npz",
        native_rema=args.reference / "rema_100m.nc",
        analysis_geometry=args.independent / "geometry.h5",
        atl06_reference=args.reference / "atl06_100m/atl06_dac_100m.nc",
    )
    record = args.work_dir / ".input_paths.json"
    if not args.dry_run:
        geometry = args.paths["analysis_geometry"]
        if geometry.is_file():
            if geometry_acquisition_ids(geometry) != [date["id"] for date in catalogue]:
                raise ValueError(
                    "Acquisition selection changed. Choose a new --work-dir."
                )
        contents = {name: str(path) for name, path in args.paths.items()}
        if record.exists():
            previous = json.loads(record.read_text())
            if any(previous.get(key) != value for key, value in contents.items()):
                raise ValueError("Input paths changed. Choose a new --work-dir.")
        else:
            args.work_dir.mkdir(parents=True, exist_ok=True)
            temporary = record.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(contents, indent=2) + "\n")
            temporary.replace(record)
    return args


def existing_file(path):
    """Reuse supplied/generated reference files; never overwrite them silently."""
    if path.is_file():
        print(f"Using {path}", flush=True)
        return True
    return False

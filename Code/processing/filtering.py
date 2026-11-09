"""Write the fixed REMA admission mask without changing heights or clocks."""

from pathlib import Path
import argparse
import json
import h5py
import numpy as np
from layout import band_indices, DEFAULT_ACQUISITIONS, acquisition_counts

SETTINGS = json.loads(
    (Path(__file__).resolve().parents[1] / "aux_files/analysis.json").read_text()
)
SCREENING = SETTINGS["screening"]


def admission_mask(height, reference):
    """Use the same finite/REMA rule as the November fit reader."""
    return np.isfinite(height) & (
        ~np.isfinite(reference)[None]
        | (
            abs(height - reference[None] - SCREENING["centre_m"])
            <= SCREENING["multiplier"] * SCREENING["NMAD_m"]
        )
    )


def check_file(handle, source, acquisitions):
    """Reject incomplete masks, different source files or changed thresholds."""
    stat = source.stat()
    assert handle.attrs["complete"] == 1
    assert handle.attrs["acquisitions"] == acquisitions
    assert handle.attrs["source_bytes"] == stat.st_size
    assert handle.attrs["source_mtime_ns"] == stat.st_mtime_ns
    assert json.loads(handle.attrs["screening_json"]) == SCREENING


def build_band(root, band, acquisitions):
    source = root / "inputs" / f"{band:03d}.h5"
    target = root / "admission" / f"{band:03d}.h5"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with h5py.File(target) as handle:
            check_file(handle, source, acquisitions)
        return

    stat = source.stat()
    temporary = target.with_suffix(".tmp.h5")
    admitted_count = finite_count = 0
    with h5py.File(source) as inputs, h5py.File(temporary, "w") as output:
        shape = inputs["height"].shape
        assert shape[0] == acquisitions and shape[2] == 9
        output.create_dataset("cell_id", data=inputs["cell_id"][:])
        mask = output.create_dataset(
            "admitted",
            shape,
            dtype="?",
            chunks=(1, min(shape[1], 1024), 9),
            compression="gzip",
        )
        for start in range(0, shape[1], 256):
            stop = min(start + 256, shape[1])
            height = inputs["height"][:, start:stop]
            reference = inputs["reference_geoid"][start:stop]
            selected = admission_mask(height, reference)
            mask[:, start:stop] = selected
            finite_count += int(np.isfinite(height).sum())
            admitted_count += int(selected.sum())
        output.attrs.update(
            complete=1,
            acquisitions=acquisitions,
            source_bytes=stat.st_size,
            source_mtime_ns=stat.st_mtime_ns,
            screening_json=json.dumps(SCREENING, sort_keys=True),
            finite_observations=finite_count,
            admitted_observations=admitted_count,
            rejected_finite_observations=finite_count - admitted_count,
            method="Finite height and fixed REMA residual gate; missing REMA retained; heights/clocks unchanged",
        )
    temporary.replace(target)
    print(
        f"Band {band:02d}: {admitted_count} admitted; {finite_count-admitted_count} finite observations rejected",
        flush=True,
    )


def check_complete(root, acquisitions):
    """Require the masks for all geometry bands before starting the numbered fit step."""
    for band in band_indices(root / "geometry.h5"):
        source = root / "inputs" / f"{band:03d}.h5"
        with h5py.File(root / "admission" / f"{band:03d}.h5") as handle:
            check_file(handle, source, acquisitions)
    print("All geometry-band admission masks are complete", flush=True)


def checked_mask(root, nodes, geometry, expected):
    """Read saved selections and require exact agreement with the original rule.

    Older run directories without saved masks retain the original read-time
    calculation. A saved mask can never relax or replace that calculation.
    """
    folder = root / "admission"
    if not folder.is_dir():
        return expected
    saved = np.zeros_like(expected)
    for band in np.unique(geometry["source_band"][nodes]):
        destination = np.flatnonzero(geometry["source_band"][nodes] == band)
        local = geometry["source_local"][nodes[destination]]
        source = root / "inputs" / f"{int(band):03d}.h5"
        with h5py.File(folder / f"{int(band):03d}.h5") as handle:
            check_file(handle, source, expected.shape[0])
            np.testing.assert_array_equal(
                handle["cell_id"][local], geometry["cell_id"][nodes[destination]]
            )
            saved[:, destination] = handle["admitted"][:, local]
    np.testing.assert_array_equal(
        saved,
        expected,
        err_msg="Saved mask differs from the fixed November admission rule",
    )
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--acquisitions",
        type=int,
        choices=acquisition_counts(),
        default=DEFAULT_ACQUISITIONS,
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--all-parts", action="store_true")
    selection.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        check_complete(args.input_dir, args.acquisitions)
        return
    bands = band_indices(args.input_dir / "geometry.h5")
    if args.index is not None and args.index not in bands:
        parser.error(f"--index must be one of {bands}")
    for band in bands if args.all_parts else [args.index]:
        build_band(args.input_dir, band, args.acquisitions)


if __name__ == "__main__":
    main()

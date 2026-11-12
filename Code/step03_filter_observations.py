"""Screen corrected heights against REMA and save the finished cube.

Keep finite heights with abs(height - REMA - centre) <= 3.5 * NMAD.
Finite heights without a REMA value also pass. Retained heights and clocks
keep their original values.
"""

import json

from step_setup import argument_parser, setup

import h5py
import numpy as np

from filtering import SCREENING, check_file
from layout import band_indices, geometry_acquisition_ids


def admission_mask(height, rema):
    """Apply the configured REMA residual threshold in metres."""
    residual = height - rema - SCREENING["centre_m"]
    limit = SCREENING["multiplier"] * SCREENING["NMAD_m"]
    return np.isfinite(height) & (~np.isfinite(rema) | (np.abs(residual) <= limit))


def filter_band(root, band, output, dates):
    """Read each chunk once, save its mask and write its screened heights."""
    source = root / "inputs" / f"{band:03d}.h5"
    target = root / "admission" / f"{band:03d}.h5"
    target.parent.mkdir(parents=True, exist_ok=True)
    reuse_mask = target.is_file()
    temporary = target.with_suffix(".tmp.h5")
    mask_path = target if reuse_mask else temporary
    mode = "r" if reuse_mask else "w"

    stat = source.stat()
    finite_count = admitted_count = 0
    with h5py.File(source) as inputs, h5py.File(mask_path, mode) as selection:
        assert inputs.attrs.get("complete") == 1
        shape = inputs["height"].shape
        assert shape[0] == len(dates) and shape[2] == 9
        np.testing.assert_array_equal(inputs["time_ns"][:], dates)
        cell_ids = inputs["cell_id"][:]
        output["cell_id"] = cell_ids
        output["fine_id"] = inputs["fine_id"][:]

        chunk = min(1024, shape[1])
        chunks = (1, chunk, shape[2])
        height = output.create_dataset(
            "height",
            shape,
            dtype="f4",
            chunks=chunks,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        height.attrs["units"] = "m above mean-tide geoid"
        clock = output.create_dataset(
            "time_seconds",
            shape[:2],
            dtype="f8",
            chunks=chunks[:2],
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        clock.attrs["units"] = "seconds since 2000-01-01"

        if reuse_mask:
            check_file(selection, source, len(dates))
            np.testing.assert_array_equal(selection["cell_id"][:], cell_ids)
            mask = selection["admitted"]
            assert mask.shape == shape
        else:
            selection["cell_id"] = cell_ids
            mask = selection.create_dataset(
                "admitted",
                shape,
                dtype="?",
                chunks=chunks,
                compression="gzip",
            )

        for start in range(0, shape[1], chunk):
            cells = slice(start, start + chunk)
            values = inputs["height"][:, cells]
            rema = inputs["reference_geoid"][cells]
            keep = admission_mask(values, rema)

            if reuse_mask:
                np.testing.assert_array_equal(mask[:, cells], keep)
            else:
                mask[:, cells] = keep
            finite_count += int(np.isfinite(values).sum())
            admitted_count += int(keep.sum())
            values[~keep] = np.nan
            height[:, cells] = values
            clock[:, cells] = inputs["time_seconds"][:, cells]

        if not reuse_mask:
            selection.attrs.update(
                complete=1,
                acquisitions=len(dates),
                source_bytes=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns,
                screening_json=json.dumps(SCREENING, sort_keys=True),
                finite_observations=finite_count,
                admitted_observations=admitted_count,
                rejected_finite_observations=finite_count - admitted_count,
                method="Finite height and fixed REMA residual gate; missing REMA retained; heights/clocks unchanged",
            )
        check_file(selection, source, len(dates))

    if not reuse_mask:
        temporary.replace(target)
    print(
        f"Band {band:02d}: {admitted_count} admitted; {finite_count - admitted_count} finite observations rejected",
        flush=True,
    )


def filter_observations(independent_dir, output, acquisitions):
    geometry = independent_dir / "geometry.h5"
    acquisition_ids = geometry_acquisition_ids(geometry)
    assert len(acquisition_ids) == acquisitions
    temporary = output.with_suffix(".tmp.h5")
    with h5py.File(geometry) as grid, h5py.File(temporary, "w") as cube:
        cube.attrs["crs"] = "EPSG:3031"
        cube.create_dataset(
            "acquisition_id", data=acquisition_ids, dtype=h5py.string_dtype()
        )
        dates = grid["time_ns"][:]
        cube["time_ns"] = dates
        for name in ("reference_seconds", "passes"):
            cube[name] = grid[name][:]
        for source, name in (
            ("x100", "x"),
            ("y100", "y"),
            ("global_x", "coarse_x"),
            ("global_y", "coarse_y"),
        ):
            cube[f"grid/{name}"] = grid[source][:]

        for band in band_indices(geometry):
            filter_band(
                independent_dir, band, cube.create_group(f"bands/{band:03d}"), dates
            )
        cube.attrs["complete"] = 1
    temporary.replace(output)


def main(argv=None):
    args = argument_parser(__doc__).parse_args(argv)
    setup(args)
    if args.dry_run:
        print(f"Work folder: {args.work_dir}")
        print("Apply the REMA screening rule and save finished_cube.h5.")
        return
    filter_observations(
        args.independent, args.paths["finished_cube"], args.acquisitions
    )


if __name__ == "__main__":
    main()

"""Rebuild the frozen REMA/geoid selection reference from raw Raster metadata.

The reference selects observations only. Its offset is never removed from
heights, used as a fitted mean or introduced as a tide coefficient prior.
"""

from pathlib import Path
import argparse

import h5py
import numpy as np
from analysis import ds, SETTINGS
from prepare import text
from layout import band_indices

BOXES = [(-65.15, -63.70, -67.52, -66.78), (-61.86, -60.90, -66.56, -66.08)]


def geoid(raw, fine, ordered_titles, overwrite):
    """Sample source geoids in the frozen tile order and overwrite convention."""
    values = np.full(len(fine), np.nan)
    if not len(fine):
        return values
    with h5py.File(raw) as handle:
        titles = text(handle["tiles/title"][:])
        lookup = {geoid_values: i for i, geoid_values in enumerate(titles)}
        starts = handle["tiles/observation_start"][:]
        observations = handle["observations"]
        for name in ordered_titles:
            if np.isfinite(values).all():
                break
            tile = lookup[name]
            tile_start, tile_stop = map(int, starts[tile : tile + 2])
            tile_start = bound_range(
                observations["cell_id"], int(fine[0]), tile_start, tile_stop
            )
            tile_stop = bound_range(
                observations["cell_id"], int(fine[-1]) + 1, tile_start, tile_stop
            )
            for start in range(tile_start, tile_stop, 65536):
                stop = min(start + 65536, tile_stop)
                ids = observations["cell_id"][start:stop]
                positions = np.searchsorted(fine, ids)
                hit = positions < len(fine)
                hit[hit] &= fine[positions[hit]] == ids[hit]
                geoid_values = observations["geoid"][start:stop][hit]
                destination = positions[hit]
                ok = np.isfinite(geoid_values) & (abs(geoid_values) < 1e20)
                if not overwrite:
                    ok &= ~np.isfinite(values[destination])
                values[destination[ok]] = geoid_values[ok]
    return values


def bound_range(ds, value, lo, hi):
    """Binary-search a sorted cell index inside one tile range."""
    while lo < hi:
        middle = (lo + hi) // 2
        if int(ds[middle]) < value:
            lo = middle + 1
        else:
            hi = middle
    return lo


def rema(path, ids, x, y):
    """Read matched ellipsoidal REMA values in bounded row blocks."""
    elevation = np.full(len(ids), np.nan)
    column, row = ids % len(x), ids // len(x)
    with h5py.File(path) as handle:
        np.testing.assert_array_equal(handle["x"][:], x)
        np.testing.assert_array_equal(handle["y"][:], y)
        for row_start in range(int(row.min()) // 64 * 64, int(row.max()) + 1, 64):
            selected = (row >= row_start) & (row < row_start + 64)
            if not selected.any():
                continue
            left, right = int(column[selected].min()), int(column[selected].max()) + 1
            block = handle["ellipsoid"][
                row_start : min(row_start + 64, len(y)), left:right
            ]
            sampled = block[row[selected] - row_start, column[selected] - left].astype(
                float
            )
            sampled[abs(sampled) > 1e20] = np.nan
            fill = handle["ellipsoid"].attrs.get("_FillValue")
            if fill is not None:
                sampled[sampled == np.asarray(fill).ravel()[0]] = np.nan
            elevation[selected] = sampled
    return elevation


def build(raw, geometry, rema_path, destination, band):
    """Write one REMA-minus-geoid reference band used for admission only."""
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{band:03d}.h5"
    if target.exists():
        raise FileExistsError(target)
    order = SETTINGS["reference_tile_order"]
    date_order = [
        name for q in SETTINGS["acquisitions"] for name in order if name in q["tiles"]
    ]
    with h5py.File(geometry) as geometry_file:
        x, y = geometry_file["x100"][:], geometry_file["y100"][:]
        positions = np.flatnonzero(geometry_file["source_band"][:] == band)
        fine_mapping = geometry_file["fine_id"][positions]
        cell_ids = geometry_file["cell_id"][positions]
        flat = np.flatnonzero(fine_mapping.ravel() >= 0)
        sort = np.argsort(fine_mapping.ravel()[flat])
        fine = fine_mapping.ravel()[flat][sort]
        fine_destination = flat[sort]
        geoid_values = geoid(raw, fine, date_order, overwrite=False)
        reference = np.full(fine_mapping.size, np.nan)
        reference[fine_destination] = rema(rema_path, fine, x, y) - geoid_values
        # Preserve the original regional reference-read convention, including
        # its tile-order overwrite rule; no silent geoid datum replacement.
        lon, lat = geometry_file["lon"][:], geometry_file["lat"][:]
        for west, east, south, north in BOXES:
            loc = np.flatnonzero(
                (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
            )
            regional_fine_mapping = geometry_file["fine_id"][loc]
            regional_fine_ids = np.sort(
                regional_fine_mapping[regional_fine_mapping >= 0]
            )
            regional_geoid = geoid(raw, regional_fine_ids, order, overwrite=True)
            regional_reference = (
                rema(rema_path, regional_fine_ids, x, y) - regional_geoid
            )
            pos = np.searchsorted(regional_fine_ids, fine)
            hit = pos < len(regional_fine_ids)
            hit[hit] &= regional_fine_ids[pos[hit]] == fine[hit]
            reference[fine_destination[hit]] = regional_reference[pos[hit]]
    with h5py.File(target, "w") as output:
        ds(output, "cell_id", cell_ids)
        ds(output, "reference_geoid", reference.reshape(fine_mapping.shape))
        output.attrs.update(
            complete=1,
            raw_source=str(raw),
            REMA_source=str(rema_path),
            reference_selection_only=1,
        )


def main(argv=None):
    """Rebuild one selection-reference band or all bands in the geometry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-cube", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--rema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parts = parser.add_mutually_exclusive_group()
    parts.add_argument("--band", type=int, help="One spatial band ID from the geometry")
    parts.add_argument(
        "--all-parts", action="store_true", help="All geometry bands, sequentially"
    )
    args = parser.parse_args(argv)
    bands = band_indices(args.geometry)
    if not args.all_parts and args.band not in bands:
        parser.error(f"Choose a band from {bands} or --all-parts")
    bands = bands if args.all_parts else [args.band]
    for band in bands:
        build(args.raw_cube, args.geometry, args.rema, args.output, band)


if __name__ == "__main__":
    main()

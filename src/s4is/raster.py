"""Turn native SWOT UTM Raster tiles into sparse EPSG:3031 tile rows.

The source file is handled as a raster, not as an unordered point cloud.  That
distinction prevents transformed pixel centres from colliding and leaving an
artificial checkerboard of holes in the target grid.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .earthdata import parse_granule_title


@dataclass(frozen=True)
class TargetGrid:
    x: np.ndarray
    y: np.ndarray
    study_area_mask: np.ndarray

    def __post_init__(self):
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("Target x and y must be one-dimensional")
        if self.study_area_mask.shape != (len(self.y), len(self.x)):
            raise ValueError("study_area_mask must have shape (y, x)")
        if len(self.x) < 2 or len(self.y) < 2:
            raise ValueError("Target grid needs at least two coordinates per axis")
        if not (np.all(np.diff(self.x) > 0) and np.all(np.diff(self.y) > 0)):
            raise ValueError("Target x and y must be strictly ascending")

    @classmethod
    def from_npz(cls, path: Path) -> "TargetGrid":
        with np.load(path) as source:
            return cls(
                np.asarray(source["x"], dtype=float),
                np.asarray(source["y"], dtype=float),
                np.asarray(source["study_area_mask"], dtype=bool),
            )


def quality_mask(dataset, mode: str) -> np.ndarray:
    """Apply the explicit source-pixel tests used before reprojection."""

    height = dataset.wse.values
    geoid = dataset.geoid.values
    pixels = (
        dataset.n_wse_pix.values if "n_wse_pix" in dataset else np.ones(height.shape)
    )
    valid = np.isfinite(height) & np.isfinite(geoid) & (pixels > 0)
    if mode == "tide":
        return valid
    if mode != "mean":
        raise ValueError("mode must be 'mean' or 'tide'")
    uncertainty = dataset.wse_uncert.values
    quality = dataset.wse_qual.values
    valid &= np.isfinite(uncertainty) & (uncertainty > 0) & np.isin(quality, [0, 1])
    meanings = dataset.wse_qual_bitwise.attrs["flag_meanings"].split()
    masks = dataset.wse_qual_bitwise.attrs["flag_masks"]
    rejected_bits = sum(
        int(bit)
        for meaning, bit in zip(meanings, masks)
        if meaning in {"geolocation_qual_suspect", "large_uncert_suspect"}
    )
    bit_values = np.nan_to_num(dataset.wse_qual_bitwise.values, nan=4294967295).astype(
        "uint32"
    )
    valid &= (bit_values & rejected_bits) == 0
    return valid


def _source_crs(dataset):
    from rasterio.crs import CRS

    mapping_name = dataset.wse.attrs.get("grid_mapping", "crs")
    mapping = dataset[mapping_name]
    wkt = mapping.attrs.get("spatial_ref") or mapping.attrs.get("crs_wkt")
    if not wkt:
        raise ValueError("SWOT Raster tile has no CRS WKT")
    return CRS.from_wkt(wkt)


def _target_window(bounds, source_crs, grid: TargetGrid) -> tuple[int, int, int, int]:
    from rasterio.warp import transform_bounds

    left, bottom, right, top = transform_bounds(
        source_crs, "EPSG:3031", *bounds, densify_pts=41
    )
    dx, dy = float(np.diff(grid.x).mean()), float(np.diff(grid.y).mean())
    x_edge, y_edge = grid.x[0] - dx / 2, grid.y[0] - dy / 2
    ix0 = max(0, int(np.floor((left - x_edge) / dx)))
    ix1 = min(len(grid.x), int(np.ceil((right - x_edge) / dx)))
    iy0 = max(0, int(np.floor((bottom - y_edge) / dy)))
    iy1 = min(len(grid.y), int(np.ceil((top - y_edge) / dy)))
    return ix0, ix1, iy0, iy1


def read_tile(
    path: Path, grid: TargetGrid, *, mode: str
) -> dict[str, np.ndarray | str]:
    """Read, quality-screen and reproject one downloaded SWOT tile."""

    import xarray as xr
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    with xr.open_dataset(path, engine="netcdf4") as source:
        sx, sy = np.asarray(source.x, float), np.asarray(source.y, float)
        if sx.ndim != 1 or sy.ndim != 1 or len(sx) < 2 or len(sy) < 2:
            raise ValueError(f"Invalid Raster axes in {path}")
        sdx, sdy = float(np.median(np.diff(sx))), float(np.median(np.diff(sy)))
        if sdx <= 0 or sdy <= 0:
            raise ValueError(f"Expected ascending Raster axes in {path}")
        source_crs = _source_crs(source)
        source_transform = from_origin(sx[0] - sdx / 2, sy[-1] + sdy / 2, sdx, sdy)
        bounds = (sx[0] - sdx / 2, sy[0] - sdy / 2, sx[-1] + sdx / 2, sy[-1] + sdy / 2)
        ix0, ix1, iy0, iy1 = _target_window(bounds, source_crs, grid)
        if ix0 >= ix1 or iy0 >= iy1:
            empty = np.array([], dtype="float64")
            return {
                "cell_id": np.array([], dtype="int64"),
                "x_m": empty,
                "y_m": empty,
                "longitude": empty,
                "latitude": empty,
                "height_m": empty,
                "geoid_m": empty,
                "uncertainty_m": empty,
                "source_crs": source_crs.to_string(),
            }
        valid = quality_mask(source, mode)
        arrays = {
            "height_m": np.where(valid, source.wse.values, np.nan).astype("float64"),
            "geoid_m": np.where(valid, source.geoid.values, np.nan).astype("float64"),
            "uncertainty_m": np.where(
                valid,
                source.wse_uncert.values if mode == "mean" else 1.0,
                np.nan,
            ).astype("float64"),
        }

    dx, dy = float(np.diff(grid.x).mean()), float(np.diff(grid.y).mean())
    destination_transform = from_origin(
        grid.x[ix0] - dx / 2, grid.y[iy1 - 1] + dy / 2, dx, dy
    )
    destination_shape = (iy1 - iy0, ix1 - ix0)
    warped: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        destination = np.full(destination_shape, np.nan, dtype="float64")
        reproject(
            source=values[::-1],
            destination=destination,
            src_transform=source_transform,
            src_crs=source_crs,
            src_nodata=np.nan,
            dst_transform=destination_transform,
            dst_crs="EPSG:3031",
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
        )
        warped[name] = destination[::-1]

    xx, yy = np.meshgrid(grid.x[ix0:ix1], grid.y[iy0:iy1])
    keep = grid.study_area_mask[iy0:iy1, ix0:ix1].copy()
    for values in warped.values():
        keep &= np.isfinite(values)
    row, column = np.nonzero(keep)
    x, y = xx[keep], yy[keep]
    lon, lat = Transformer.from_crs(3031, 4326, always_xy=True).transform(x, y)
    return {
        "cell_id": ((iy0 + row) * len(grid.x) + ix0 + column).astype("int64"),
        "x_m": x.astype("float64"),
        "y_m": y.astype("float64"),
        "longitude": np.asarray(lon, dtype="float64"),
        "latitude": np.asarray(lat, dtype="float64"),
        **{name: values[keep] for name, values in warped.items()},
        "source_crs": source_crs.to_string(),
    }


def merge_acquisition(paths: Sequence[Path], grid: TargetGrid, *, mode: str) -> dict:
    """Merge overlapping tiles so one overpass contributes once per cell."""

    rows = [read_tile(Path(path), grid, mode=mode) for path in paths]
    rows = [row for row in rows if len(row["cell_id"])]
    if not rows:
        return {
            name: np.array([], dtype="int64" if name == "cell_id" else float)
            for name in (
                "cell_id",
                "x_m",
                "y_m",
                "longitude",
                "latitude",
                "height_m",
                "geoid_m",
                "uncertainty_m",
            )
        }
    fields = (
        "cell_id",
        "x_m",
        "y_m",
        "longitude",
        "latitude",
        "height_m",
        "geoid_m",
        "uncertainty_m",
    )
    joined = {name: np.concatenate([row[name] for row in rows]) for name in fields}
    order = np.argsort(joined["cell_id"], kind="stable")
    joined = {name: values[order] for name, values in joined.items()}
    cells, starts, counts = np.unique(
        joined["cell_id"], return_index=True, return_counts=True
    )

    def medians(values):
        sorted_values = np.empty_like(values)
        for start, count in zip(starts, counts):
            sorted_values[start : start + count] = np.sort(
                values[start : start + count]
            )
        return (
            sorted_values[starts + (counts - 1) // 2]
            + sorted_values[starts + counts // 2]
        ) / 2

    result = {"cell_id": cells.astype("int64")}
    for name in fields[1:]:
        result[name] = medians(joined[name]).astype("float32")
    return result


def save_tile_row(
    output: Path,
    values: dict,
    *,
    tile_path: Path,
    mode: str,
) -> None:
    """Write one sparse reprojected tile checkpoint before corrections."""

    parsed = parse_granule_title(Path(tile_path).name)
    start = datetime.fromisoformat(parsed.start_utc.removesuffix("Z"))
    stop = datetime.fromisoformat(parsed.stop_utc.removesuffix("Z"))
    midpoint = start + (stop - start) / 2
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **values,
        acquisition=np.array(parsed.acquisition),
        tile=np.array(Path(tile_path).name),
        time_ns=np.array(np.datetime64(midpoint, "ns").astype("int64")),
        correction_status=np.array("raw_raster_wse"),
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "acquisition": parsed.acquisition,
                "tile": Path(tile_path).name,
                "mode": mode,
                "cells": int(len(values["cell_id"])),
                "projection": "affine nearest-neighbour UTM raster to EPSG:3031",
                "correction_status": "raw_raster_wse",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        type=Path,
        required=True,
        help="NPZ containing x, y, and study_area_mask",
    )
    parser.add_argument("--mode", choices=("mean", "tide"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("tiles", nargs="+", type=Path)
    args = parser.parse_args(argv)
    grid = TargetGrid.from_npz(args.grid)
    for tile in args.tiles:
        output = args.output_dir / f"{tile.stem}.npz"
        save_tile_row(
            output,
            read_tile(tile, grid, mode=args.mode),
            tile_path=tile,
            mode=args.mode,
        )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

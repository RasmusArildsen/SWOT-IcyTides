"""Build an unfiltered sparse LCIS cube from official SWOT Raster fields.

The only transformation is nearest-neighbour reprojection from each official
regular UTM tile to the shared 100 m EPSG:3031 grid.  No quality flag, AOI
mask, geophysical correction, DAC, tide model, or statistical filter is
applied.  Tiles remain distinct and are linked to acquisitions by metadata.
"""

from __future__ import annotations

import argparse

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .earthdata import read_manifest, temporary_acquisition
from .raster import _source_crs, _target_window

FLOAT_FIELDS = {
    # Preserve the source precision of coordinates and mission times.  The
    # other floating-point Raster fields are float32 in the official product.
    "longitude": "f8",
    "latitude": "f8",
    "illumination_time": "f8",
    "illumination_time_tai": "f8",
    "wse": "f4",
    "wse_uncert": "f4",
    "water_area": "f4",
    "water_area_uncert": "f4",
    "water_frac": "f4",
    "water_frac_uncert": "f4",
    "sig0": "f4",
    "sig0_uncert": "f4",
    "inc": "f4",
    "cross_track": "f4",
    "dark_frac": "f4",
    "layover_impact": "f4",
    "sig0_cor_atmos_model": "f4",
    "height_cor_xover": "f4",
    "geoid": "f4",
    "solid_earth_tide": "f4",
    "load_tide_fes": "f4",
    "load_tide_got": "f4",
    "pole_tide": "f4",
    "model_dry_tropo_cor": "f4",
    "model_wet_tropo_cor": "f4",
    "iono_cor_gim_ka": "f4",
}
INTEGER_FIELDS = {
    "wse_qual": "u1",
    "wse_qual_bitwise": "u4",
    "water_area_qual": "u1",
    "water_area_qual_bitwise": "u4",
    "sig0_qual": "u1",
    "sig0_qual_bitwise": "u4",
    "n_wse_pix": "u4",
    "n_water_area_pix": "u4",
    "n_sig0_pix": "u4",
    "n_other_pix": "u4",
    "ice_clim_flag": "u1",
    "ice_dyn_flag": "u1",
}


def load_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as source:
        x = np.asarray(source["x"], dtype="f8")
        y = np.asarray(source["y"], dtype="f8")
    if (
        len(x) < 2
        or len(y) < 2
        or not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0))
    ):
        raise ValueError("Grid coordinates must be ascending one-dimensional axes")
    if not (np.allclose(np.diff(x), 100) and np.allclose(np.diff(y), 100)):
        raise ValueError("Raw LCIS cube requires an exact 100 m grid")
    return x, y


def _float_values(variable, dtype: str) -> np.ndarray:
    values = np.asarray(variable.values, dtype=dtype)
    fill = variable.attrs.get("_FillValue")
    invalid = ~np.isfinite(values) | (np.abs(values) >= 1e20)
    if fill is not None:
        invalid |= values == np.asarray(fill, dtype=dtype)
    values[invalid] = np.nan
    return values


def _integer_values(variable, dtype: str) -> tuple[np.ndarray, int]:
    target_dtype = np.dtype(dtype)
    fill = int(variable.attrs.get("_FillValue", np.iinfo(target_dtype).max))
    return np.asarray(variable.values, dtype=target_dtype), fill


def _json_value(value):
    """Convert NetCDF attributes to plain JSON without losing their values."""

    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _source_attributes(source) -> str:
    names = ["x", "y", "crs", *FLOAT_FIELDS, *INTEGER_FIELDS]
    payload = {
        "global": {key: _json_value(value) for key, value in source.attrs.items()},
        "variables": {
            name: {
                "dtype": str(source[name].dtype),
                "attributes": {
                    key: _json_value(value) for key, value in source[name].attrs.items()
                },
            }
            for name in names
            if name in source
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def reproject_raw_tile(path: Path, x: np.ndarray, y: np.ndarray) -> dict:
    """Reproject every retained raw field without screening source pixels."""

    import xarray as xr
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    # Preserve numeric mission times exactly as delivered.  Xarray's datetime
    # decoder cannot represent the SWOT floating-point fill value (9.969e36),
    # and decoding here would also violate the raw-cube contract.
    with xr.open_dataset(
        path,
        engine="netcdf4",
        mask_and_scale=False,
        decode_times=False,
    ) as source:
        sx, sy = np.asarray(source.x, float), np.asarray(source.y, float)
        sdx, sdy = float(np.median(np.diff(sx))), float(np.median(np.diff(sy)))
        if sdx <= 0 or sdy <= 0:
            raise ValueError(f"Expected ascending source axes in {path}")
        source_crs = _source_crs(source)
        source_transform = from_origin(sx[0] - sdx / 2, sy[-1] + sdy / 2, sdx, sdy)
        bounds = (
            sx[0] - sdx / 2,
            sy[0] - sdy / 2,
            sx[-1] + sdx / 2,
            sy[-1] + sdy / 2,
        )
        grid = type("Grid", (), {"x": x, "y": y})()
        ix0, ix1, iy0, iy1 = _target_window(bounds, source_crs, grid)
        if ix0 >= ix1 or iy0 >= iy1:
            return {
                "cell_id": np.array([], dtype="i8"),
                "source_crs": source_crs.to_string(),
            }
        dx, dy = float(np.diff(x).mean()), float(np.diff(y).mean())
        destination_transform = from_origin(
            x[ix0] - dx / 2, y[iy1 - 1] + dy / 2, dx, dy
        )
        shape = (iy1 - iy0, ix1 - ix0)

        def warp_float(name, dtype):
            destination = np.full(shape, np.nan, dtype=dtype)
            if name in source:
                reproject(
                    _float_values(source[name], dtype)[::-1],
                    destination,
                    src_transform=source_transform,
                    src_crs=source_crs,
                    src_nodata=np.nan,
                    dst_transform=destination_transform,
                    dst_crs="EPSG:3031",
                    dst_nodata=np.nan,
                    resampling=Resampling.nearest,
                    init_dest_nodata=True,
                )
            return destination[::-1]

        warped_float = {
            name: warp_float(name, dtype) for name, dtype in FLOAT_FIELDS.items()
        }
        keep = np.isfinite(warped_float["wse"])
        row, column = np.nonzero(keep)
        result = {
            "cell_id": ((iy0 + row) * len(x) + ix0 + column).astype("i8"),
            **{name: values[keep] for name, values in warped_float.items()},
            "source_crs": source_crs.to_string(),
        }
        for name, dtype in INTEGER_FIELDS.items():
            target_dtype = np.dtype(dtype)
            target_fill = np.iinfo(target_dtype).max
            destination = np.full(shape, target_fill, dtype=target_dtype)
            if name in source:
                values, source_fill = _integer_values(source[name], dtype)
                reproject(
                    values[::-1],
                    destination,
                    src_transform=source_transform,
                    src_crs=source_crs,
                    src_nodata=source_fill,
                    dst_transform=destination_transform,
                    dst_crs="EPSG:3031",
                    dst_nodata=target_fill,
                    resampling=Resampling.nearest,
                    init_dest_nodata=True,
                )
            result[name] = destination[::-1][keep]
        result["product_version"] = str(source.attrs.get("product_version", ""))
        result["source_attributes_json"] = _source_attributes(source)
    return result


def _create_cube(path: Path, manifest: dict, grid_path: Path) -> None:
    x, y = load_grid(grid_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    with h5py.File(path, "w") as target:
        target.attrs.update(
            schema="s4is-raw-swot-sparse-cube-v1",
            processing_level="official_l2_product_fields_reprojected_only",
            crs="EPSG:3031",
            resolution_m=100,
            corrections_applied="none",
            quality_filter_applied="none",
            aoi_mask_applied="none; fixed LCIS rectangular grid extent only",
            overlap_policy="none; every source tile remains separate",
            manifest_sha256=hashlib.sha256(manifest_text.encode()).hexdigest(),
        )
        grid = target.create_group("grid")
        grid.create_dataset("x", data=x)
        grid.create_dataset("y", data=y)
        observations = target.create_group("observations")
        observations.create_dataset(
            "cell_id",
            shape=(0,),
            maxshape=(None,),
            dtype="i8",
            chunks=(262_144,),
            compression="gzip",
            compression_opts=1,
        )
        for name, dtype in FLOAT_FIELDS.items():
            observations.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(262_144,),
                compression="gzip",
                compression_opts=1,
                fillvalue=np.nan,
            )
        for name, dtype in INTEGER_FIELDS.items():
            observations.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(262_144,),
                compression="gzip",
                compression_opts=1,
                fillvalue=np.iinfo(np.dtype(dtype)).max,
            )
        tiles = target.create_group("tiles")
        tiles.create_dataset(
            "observation_start", data=np.array([0], dtype="i8"), maxshape=(None,)
        )
        string = h5py.string_dtype("utf-8")
        for name in (
            "title",
            "acquisition",
            "product",
            "concept_id",
            "revision_date",
            "source_crs",
            "product_version",
            "source_attributes_json",
        ):
            tiles.create_dataset(name, shape=(0,), maxshape=(None,), dtype=string)
        tiles.create_dataset("revision_id", shape=(0,), maxshape=(None,), dtype="i4")
        acquisitions = target.create_group("acquisitions")
        rows = manifest["acquisitions"]
        for name, values in {
            "id": [row["id"] for row in rows],
            "start_utc": [row["start_utc"] for row in rows],
            "stop_utc": [row["stop_utc"] for row in rows],
        }.items():
            acquisitions.create_dataset(name, data=np.asarray(values, dtype=string))
        acquisitions.create_dataset(
            "cycle", data=np.asarray([row["cycle"] for row in rows], dtype="i2")
        )
        acquisitions.create_dataset(
            "pass", data=np.asarray([row["pass"] for row in rows], dtype="i2")
        )
        target.create_dataset(
            "manifest_json", data=np.asarray(manifest_text, dtype=string)
        )


def _append(target, metadata: dict, values: dict) -> None:
    observations = target["observations"]
    tiles = target["tiles"]
    start = len(observations["cell_id"])
    stop = start + len(values["cell_id"])
    for name, dataset in observations.items():
        dataset.resize((stop,))
        if name == "cell_id":
            dataset[start:stop] = values[name]
        elif name in values:
            dataset[start:stop] = values[name]
    tile_index = len(tiles["title"])
    for name in ("title", "acquisition", "product", "concept_id", "revision_date"):
        tiles[name].resize((tile_index + 1,))
        tiles[name][tile_index] = str(metadata.get(name, ""))
    for name in ("source_crs", "product_version", "source_attributes_json"):
        tiles[name].resize((tile_index + 1,))
        tiles[name][tile_index] = str(values.get(name, ""))
    tiles["revision_id"].resize((tile_index + 1,))
    tiles["revision_id"][tile_index] = int(metadata.get("revision_id", 0))
    pointer = tiles["observation_start"]
    pointer.resize((len(pointer) + 1,))
    pointer[-1] = stop
    target.flush()


def build_raw_cube(
    manifest_path: Path,
    grid_path: Path,
    output: Path,
    cache_dir: Path,
    *,
    limit_acquisitions: int | None = None,
    keep_raw_tiles: bool = False,
) -> dict:
    manifest = read_manifest(manifest_path)
    if not output.exists():
        _create_cube(output, manifest, grid_path)
    x, y = load_grid(grid_path)
    with h5py.File(output, "r+") as target:
        manifest_text = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        expected_hash = hashlib.sha256(manifest_text.encode()).hexdigest()
        if target.attrs.get("manifest_sha256") != expected_hash:
            raise ValueError("Existing cube was created from a different manifest")
        safe_length = int(target["tiles/observation_start"][-1])
        for dataset in target["observations"].values():
            if len(dataset) != safe_length:
                dataset.resize((safe_length,))
        safe_tiles = len(target["tiles/observation_start"]) - 1
        for name, dataset in target["tiles"].items():
            if name != "observation_start" and len(dataset) != safe_tiles:
                dataset.resize((safe_tiles,))
        completed = {
            value.decode() if isinstance(value, bytes) else str(value)
            for value in target["tiles/title"][:]
        }
        acquisitions = manifest["acquisitions"]
        if limit_acquisitions is not None:
            acquisitions = acquisitions[:limit_acquisitions]
        granules = {row["title"]: row for row in manifest["granules"]}
        for number, acquisition in enumerate(acquisitions, 1):
            pending = [
                title for title in acquisition["tiles"] if title not in completed
            ]
            if not pending:
                continue
            with temporary_acquisition(
                manifest,
                acquisition["id"],
                cache_dir,
                keep=keep_raw_tiles,
            ) as paths:
                path_by_name = {path.name: path for path in paths}
                for title in pending:
                    values = reproject_raw_tile(path_by_name[title], x, y)
                    _append(target, granules[title], values)
                    completed.add(title)
                    print(
                        f"RAW_TILE {len(completed)}/{manifest['granule_count']} "
                        f"{acquisition['id']} {title} cells={len(values['cell_id'])}",
                        flush=True,
                    )
    with h5py.File(output) as source:
        report = {
            "output": str(output),
            "tiles_complete": int(len(source["tiles/title"])),
            "tiles_expected": int(manifest["granule_count"]),
            "acquisitions_expected": int(manifest["acquisition_count"]),
            "raw_observations": int(len(source["observations/cell_id"])),
            "bytes": int(output.stat().st_size),
            "complete": len(source["tiles/title"]) == manifest["granule_count"],
        }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--limit-acquisitions", type=int)
    parser.add_argument("--keep-raw-tiles", action="store_true")
    args = parser.parse_args(argv)
    report = build_raw_cube(
        args.manifest,
        args.grid,
        args.output,
        args.cache_dir,
        limit_acquisitions=args.limit_acquisitions,
        keep_raw_tiles=args.keep_raw_tiles,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

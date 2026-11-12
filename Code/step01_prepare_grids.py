"""Reproject the local SWOT Raster files and prepare the analysis grids.

Build the unfiltered 100 m cube, crop BedMachine, resample REMA and
record the analysis grid, acquisition clocks and CATS comparison fields.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from step_setup import CODE, argument_parser, existing_file, setup

import h5py
import numpy as np
import rasterio
from layout import (
    SETTINGS,
    axis,
    ds,
    DEFAULT_ACQUISITIONS,
    selected_acquisitions,
    PARTITIONS,
    geometry_acquisition_ids,
)
from scipy.ndimage import distance_transform_edt
from study_area import STUDY_AREA, TO_LL, StudyArea, study_area_mask
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from cats import load_cats, cats_constants
from download_data import resolve_raster_files

# Prepare the analysis grids


def prepare_grids(paths, independent_dir, acquisitions):
    dates = selected_acquisitions(acquisitions)
    files = resolve_raster_files(dates, paths["raster_raw"])

    # Regrid the Raster fields used in the analysis.
    build_raw_cube(
        [date["id"] for date in dates],
        files,
        axis("x100"),
        axis("y100"),
        paths["raw_cube"],
    )

    # Crop BedMachine and independently resample REMA onto the 100 m grid.
    if not existing_file(paths["native_bedmachine_mask"]):
        prepare_bedmachine_mask(
            paths["bedmachine_source"], paths["native_bedmachine_mask"]
        )
    if not existing_file(paths["analysis_bedmachine_mask"]):
        prepare_bedmachine_mask(
            paths["bedmachine_source"], paths["analysis_bedmachine_mask"], expanded=True
        )
    if not existing_file(paths["native_rema"]):
        prepare_rema(paths["rema_source"], paths["native_rema"])

    # Assign fine cells to coarse cells and prepare clocks and CATS comparisons.
    independent_dir.mkdir(parents=True, exist_ok=True)
    prepare_geometry(
        independent_dir,
        paths["analysis_bedmachine_mask"],
        paths["cats_model"],
        nt=acquisitions,
    )


def main(argv=None):
    args = setup(argument_parser(__doc__).parse_args(argv))
    if args.dry_run:
        print(f"Work folder: {args.work_dir}")
        print("Reproject the local SWOT Raster files and prepare the reference grids.")
        return
    prepare_grids(args.paths, args.independent, args.acquisitions)


# Raster reprojection and raw cube


RASTER_FIELDS = {
    "wse": "f4",
    "geoid": "f4",
    "illumination_time": "f8",
    "cross_track": "f4",
}


def _source_crs(dataset):
    from rasterio.crs import CRS

    mapping_name = dataset.wse.attrs.get("grid_mapping", "crs")
    mapping = dataset[mapping_name]
    wkt = mapping.attrs.get("spatial_ref") or mapping.attrs.get("crs_wkt")
    if not wkt:
        raise ValueError("SWOT Raster tile has no CRS WKT")
    return CRS.from_wkt(wkt)


def _target_window(bounds, source_crs, x, y) -> tuple[int, int, int, int]:
    from rasterio.warp import transform_bounds

    left, bottom, right, top = transform_bounds(
        source_crs, "EPSG:3031", *bounds, densify_pts=41
    )
    dx, dy = float(np.diff(x).mean()), float(np.diff(y).mean())
    x_edge, y_edge = x[0] - dx / 2, y[0] - dy / 2
    ix0 = max(0, int(np.floor((left - x_edge) / dx)))
    ix1 = min(len(x), int(np.ceil((right - x_edge) / dx)))
    iy0 = max(0, int(np.floor((bottom - y_edge) / dy)))
    iy1 = min(len(y), int(np.ceil((top - y_edge) / dy)))
    return ix0, ix1, iy0, iy1


def _float_values(variable, dtype: str) -> np.ndarray:
    values = np.asarray(variable.values, dtype=dtype)
    fill = variable.attrs.get("_FillValue")
    invalid = ~np.isfinite(values) | (np.abs(values) >= 1e20)
    if fill is not None:
        invalid |= values == np.asarray(fill, dtype=dtype)
    values[invalid] = np.nan
    return values


def reproject_raw_tile(path: Path, x: np.ndarray, y: np.ndarray) -> dict:
    """Reproject WSE and the fields needed for corrections and screening."""

    import xarray as xr

    # Keep the numeric mission clock in double precision.
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
        ix0, ix1, iy0, iy1 = _target_window(bounds, source_crs, x, y)
        if ix0 >= ix1 or iy0 >= iy1:
            return {
                "cell_id": np.empty(0, dtype="i8"),
                **{
                    name: np.empty(0, dtype=dtype)
                    for name, dtype in RASTER_FIELDS.items()
                },
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
            name: warp_float(name, dtype) for name, dtype in RASTER_FIELDS.items()
        }
        keep = np.isfinite(warped_float["wse"])
        row, column = np.nonzero(keep)
        return {
            "cell_id": ((iy0 + row) * len(x) + ix0 + column).astype("i8"),
            **{name: values[keep] for name, values in warped_float.items()},
        }


def create_cube(output, x, y, acquisition_ids):
    """Create the sparse cube, keeping overlapping Raster tiles separate."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.h5")
    string = h5py.string_dtype()
    with h5py.File(temporary, "w") as cube:
        cube.attrs["crs"] = "EPSG:3031"
        cube["grid/x"] = x
        cube["grid/y"] = y
        cube.create_dataset("acquisitions/id", data=acquisition_ids, dtype=string)
        tiles = cube.create_group("tiles")
        tiles.attrs["completed"] = 0
        tiles.create_dataset("title", shape=(0,), maxshape=(None,), dtype=string)
        tiles.create_dataset("acquisition", shape=(0,), maxshape=(None,), dtype=string)
        tiles.create_dataset(
            "observation_start", data=[0], maxshape=(None,), dtype="i8"
        )
        observations = cube.create_group("observations")
        for field, dtype in {"cell_id": "i8", **RASTER_FIELDS}.items():
            observations.create_dataset(
                field,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(262144,),
                compression="gzip",
                compression_opts=1,
            )
    temporary.replace(output)


def build_raw_cube(acquisition_ids, files, x, y, output):
    """Append the listed tiles. Resume from the last saved tile offset."""
    output = Path(output)
    if not output.exists():
        create_cube(output, x, y, acquisition_ids)

    with h5py.File(output, "r+") as cube:
        tiles = cube["tiles"]
        starts = tiles["observation_start"]
        count = int(tiles.attrs.get("completed", len(starts) - 1))
        titles = tiles["title"]
        dates = tiles["acquisition"]
        expected = [(date, path.name) for date, path in files]
        recorded = list(zip(dates.asstr()[:count], titles.asstr()[:count]))
        if (
            cube["acquisitions/id"].asstr()[:].tolist() != acquisition_ids
            or len(recorded) != count
            or recorded != expected[:count]
            or not np.array_equal(cube["grid/x"][:], x)
            or not np.array_equal(cube["grid/y"][:], y)
        ):
            raise ValueError(
                "Raw cube uses different tiles or grid. Choose a new --work-dir."
            )

        observations = cube["observations"]
        end = int(starts[count])
        if any(len(field) < end for field in observations.values()):
            raise ValueError(f"Incomplete data in {output}. Choose a new --work-dir.")
        if count == len(files):
            return
        if set(observations) != {"cell_id", *RASTER_FIELDS}:
            raise ValueError(
                "This partial cube uses an older layout. Choose a new --work-dir."
            )

        for index in range(count, len(files)):
            date, path = files[index]
            values = reproject_raw_tile(path, x, y)
            stop = end + len(values["cell_id"])
            for name, dataset in observations.items():
                dataset.resize((stop,))
                dataset[end:stop] = values[name]
            for dataset, value in ((titles, path.name), (dates, date)):
                dataset.resize((index + 1,))
                dataset[index] = value
            starts.resize((index + 2,))
            starts[index + 1] = stop
            cube.flush()
            # A retry replaces any tile that was interrupted before this point.
            tiles.attrs["completed"] = index + 1
            cube.flush()
            end = stop
            print(f"Raster tile {index + 1}/{len(files)}: {path.name}", flush=True)


# BedMachine masks


def prepare_bedmachine_mask(source, destination, expanded=False):
    """Crop BedMachine and optionally extend the floating component inland."""
    window = json.loads((CODE / "aux_files/bedmachine_windows.json").read_text())
    axes = {
        axis: window[axis]["start"]
        + window[axis]["step"] * np.arange(window[axis]["size"])
        for axis in ["x", "y"]
    }
    with h5py.File(source) as handle:
        indices = {}
        for coordinate in ["x", "y"]:
            source_axis = handle[coordinate][:]
            order = np.argsort(source_axis)
            positions = np.searchsorted(source_axis[order], axes[coordinate])
            np.testing.assert_array_equal(
                source_axis[order[positions]], axes[coordinate]
            )
            indices[coordinate] = order[positions]
        columns, rows = indices["x"], indices["y"]
        crop_window = (
            slice(rows.min(), rows.max() + 1),
            slice(columns.min(), columns.max() + 1),
        )
        mask_values = handle["mask"][crop_window]
        mask_values = mask_values[
            (rows - rows.min())[:, None], (columns - columns.min())[None, :]
        ]
    data = {**axes, "mask": mask_values}
    inland_margin = STUDY_AREA["analysis_inland_margin_m"] if expanded else 0
    if expanded:
        original = StudyArea(data).floating
        spacing = (abs(window["y"]["step"]), abs(window["x"]["step"]))
        change = (mask_values == 2) & (
            distance_transform_edt(~original, sampling=spacing) <= inland_margin
        )
        mask_values[change] = 3
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    np.savez_compressed(
        destination,
        **data,
        source=str(source),
        metadata=json.dumps(
            dict(
                base="BedMachineAntarcticaV04.1",
                inland_margin_m=inland_margin,
                margin_source_class=2,
                component_seed_lonlat=STUDY_AREA["floating_component_seed_lonlat"],
            )
        ),
    )


# REMA resampling


def source_window(src, x, y):
    """Bound a source read to the destination footprint, with a two-pixel margin."""
    w = from_bounds(
        float(x[0] - 50),
        float(y[0] - 50),
        float(x[-1] + 50),
        float(y[-1] + 50),
        src.transform,
    )
    left = max(0, int(np.floor(w.col_off)) - 2)
    top = max(0, int(np.floor(w.row_off)) - 2)
    right = min(src.width, int(np.ceil(w.col_off + w.width)) + 2)
    bottom = min(src.height, int(np.ceil(w.row_off + w.height)) + 2)
    return Window(left, top, max(0, right - left), max(0, bottom - top))


def resample_rema(source, x, y):
    """Resample sorted REMA tiles with the original count gate and averaging rules."""
    assert len(x) > 1 and len(y) > 1
    np.testing.assert_array_equal(np.diff(x), 100)
    np.testing.assert_array_equal(np.diff(y), 100)
    shape = (len(y), len(x))
    transform = from_origin(x[0] - 50, y[-1] + 50, 100, 100)
    elevation = np.full(shape, np.nan, "f4")
    tiles = 0
    for path in sorted(Path(source).glob("*_dem.tif")):
        with rasterio.open(path) as dem:
            if dem.crs != rasterio.crs.CRS.from_epsg(3031):
                raise ValueError(f"Expected native EPSG:3031: {path.name}")
            if (
                dem.bounds.right < x.min()
                or dem.bounds.left > x.max()
                or dem.bounds.top < y.min()
                or dem.bounds.bottom > y.max()
            ):
                continue
            window = source_window(dem, x, y)
            if not window.width or not window.height:
                continue
            expected_transform = dem.transform
            expected_shape = dem.shape
        count_path = path.with_name(path.name.replace("_dem.tif", "_count.tif"))
        with rasterio.open(count_path) as count:
            assert (
                count.transform == expected_transform and count.shape == expected_shape
            )
            source_count = count.read(1, window=window, masked=True).filled(0)
        with rasterio.open(path) as dem:
            values = dem.read(1, window=window, masked=True).astype("f4").filled(np.nan)
            values[source_count <= 0] = np.nan
            reproject(
                values,
                elevation,
                src_transform=dem.window_transform(window),
                src_crs=dem.crs,
                src_nodata=np.nan,
                dst_transform=transform,
                dst_crs="EPSG:3031",
                dst_nodata=np.nan,
                resampling=Resampling.average,
                init_dest_nodata=False,
                num_threads=1,
            )
        tiles += 1
    if not tiles:
        raise FileNotFoundError("No overlapping REMA *_dem.tif tiles")
    return elevation[::-1]


def prepare_rema(source, output):
    """Write the independently resampled 100 m REMA reference and source metadata."""
    import xarray as xr

    axes = {"x": axis("x100"), "y": axis("y100")}
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    elevation = resample_rema(source, axes["x"], axes["y"])
    result = xr.Dataset(
        {"ellipsoid": (("y", "x"), elevation)},
        coords=axes,
        attrs=dict(
            title="REMA v2 independently resampled to 100 m",
            elevation="Original WGS84 ellipsoidal mosaic; common geoid conversion performed downstream",
            resampling="Area-weighted average of finite 32 m elevations with source count>0; no comparison-residual or elevation-range filter",
            source=str(source),
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.nc")
    result.to_netcdf(
        temporary, encoding={"ellipsoid": dict(zlib=True, complevel=4)}
    )
    temporary.replace(output)


# Observation geometry and corrected observation bands


def domain_mapping(x, y, cx, cy, mask):
    """Admit 250 m centres, then assign accepted 100 m cells to those centres.

    The output has one row per coarse centre and up to nine fine-cell IDs per row.
    Missing fine positions stay -1; fine cells never expand the coarse domain.
    """
    admitted = []
    for start in range(0, len(cy), 96):
        xx, yy = np.meshgrid(cx, cy[start : start + 96])
        admitted.append(np.flatnonzero(mask(xx, yy)) + start * len(cx))
    coarse_ids = np.concatenate(admitted)
    parts = []
    owners = []
    for start in range(0, len(y), 96):
        xx, yy = np.meshgrid(x, y[start : start + 96])
        inside = mask(xx, yy)
        row_fine_ids = np.flatnonzero(inside) + start * len(x)
        owner = np.rint((yy[inside] - cy[0]) / 250).astype("i8") * len(cx) + np.rint(
            (xx[inside] - cx[0]) / 250
        ).astype("i8")
        keep = np.isin(owner, coarse_ids)
        parts.append(row_fine_ids[keep])
        owners.append(owner[keep])
    fine_ids = np.concatenate(parts)
    owner = np.concatenate(owners)
    order = np.lexsort((fine_ids, owner))
    fine_ids, owner = fine_ids[order], owner[order]
    unique_owners, starts, counts = np.unique(
        owner, return_index=True, return_counts=True
    )
    assert counts.max() <= 9
    coarse_row = np.repeat(np.searchsorted(coarse_ids, unique_owners), counts)
    fine_slot = np.arange(len(fine_ids)) - np.repeat(starts, counts)
    fine_mapping = np.full((len(coarse_ids), 9), -1, "i8")
    fine_mapping[coarse_row, fine_slot] = fine_ids
    return coarse_ids, fine_mapping


def acquisition_clocks(acquisitions):
    """Derive nominal times and pass numbers from the Raster file list."""
    earliest = set(SETTINGS["earliest_tile_midpoint_acquisitions"])
    time_ns, passes = [], []
    for row in acquisitions:
        acquisition = row["id"]
        midpoints = []
        for filename in row["tiles"]:
            start_text, stop_text = filename.split("_")[-4:-2]
            start = datetime.strptime(start_text, "%Y%m%dT%H%M%S")
            stop = datetime.strptime(stop_text, "%Y%m%dT%H%M%S")
            midpoints.append(start + (stop - start) / 2)
        midpoint = min(midpoints)
        if acquisition not in earliest:
            offsets = sum((time - midpoint for time in midpoints), timedelta())
            midpoint += offsets / len(midpoints)
        time_ns.append(np.datetime64(midpoint, "ns").astype("i8"))
        passes.append(int(acquisition.rsplit("_", 1)[-1]))
    return time_ns, passes


def prepare_geometry(out, bedmachine, cats, nt=DEFAULT_ACQUISITIONS):
    """Write geometry.h5: grid, fine-cell membership, clocks and comparison CATS."""
    acquisitions = selected_acquisitions(nt)
    dates = [row["id"] for row in acquisitions]
    target = Path(out) / "geometry.h5"
    if target.exists():
        if geometry_acquisition_ids(target) != dates:
            raise ValueError("Existing geometry uses a different acquisition selection")
        with h5py.File(target) as handle:
            assert handle.attrs["complete"] == 1
        return
    x, y, cx, cy = [axis(name) for name in ["x100", "y100", "global_x", "global_y"]]
    cell_ids, fine_mapping = domain_mapping(
        x, y, cx, cy, lambda xx, yy: study_area_mask(xx, yy, bedmachine)
    )
    expected_cells = SETTINGS["domain_cells"]
    if expected_cells is not None and len(cell_ids) != expected_cells:
        raise ValueError(f"Study area has {len(cell_ids)} coarse cells; expected {expected_cells}")
    cell_x, cell_y = cx[cell_ids % len(cx)], cy[cell_ids // len(cx)]
    lon, lat = TO_LL.transform(cell_x, cell_y)
    cats_coefficients = np.empty((8, len(cell_ids)))
    model = load_cats(cats)
    for start in range(0, len(cell_ids), 8192):
        chunk = slice(start, start + 8192)
        amplitude, phase = cats_constants(cell_x[chunk], cell_y[chunk], model)
        cats_coefficients[0::2, chunk] = amplitude * np.cos(np.deg2rad(phase))
        cats_coefficients[1::2, chunk] = amplitude * np.sin(np.deg2rad(phase))
    band_count = PARTITIONS["observation_bands"]
    band_sizes = [len(values) for values in np.array_split(cell_ids, band_count)]
    bands = np.repeat(np.arange(band_count), band_sizes).astype("i1")
    local_index = np.concatenate([np.arange(size, dtype="i4") for size in band_sizes])
    time_ns, passes = acquisition_clocks(acquisitions)
    with h5py.File(target.with_suffix(".tmp.h5"), "w") as handle:
        for name, values in dict(
            cell_id=cell_ids,
            fine_id=fine_mapping,
            lat=lat,
            lon=lon,
            global_x=cx,
            global_y=cy,
            x100=x,
            y100=y,
            cats_beta=cats_coefficients,
            source_band=bands,
            source_local=local_index,
            time_ns=time_ns,
            passes=passes,
            acquisition_id=np.asarray(dates, dtype="S"),
        ).items():
            ds(handle, name, values)
        handle.attrs.update(
            complete=1,
            band_ids=np.arange(band_count),
            fit_parts_per_band=PARTITIONS["fit_parts_per_band"],
            reference_epoch_seconds_2000=SETTINGS["time_center_seconds_2000"],
            time_center_seconds_2000=SETTINGS["time_center_seconds_2000"],
            time_span_seconds=SETTINGS["time_span_seconds"],
            crs="EPSG:3031",
        )
    target.with_suffix(".tmp.h5").replace(target)


if __name__ == "__main__":
    main()

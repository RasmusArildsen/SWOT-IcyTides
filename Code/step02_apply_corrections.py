"""Apply the saved phase repairs and build the corrected observation bands.

Raster WSE keeps its supplied geophysical convention. The residual phase
layer already has DAC removed. Ocean tide remains in the heights for fitting."""

from __future__ import annotations

from step_setup import argument_parser, setup

import numpy as np
import xarray as xr
import h5py
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from layout import (
    AUX,
    SETTINGS,
    ds,
    band_indices,
    DEFAULT_ACQUISITIONS,
    geometry_acquisition_ids,
    selected_acquisitions,
)
from collections import OrderedDict
from datetime import datetime, timezone
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from study_area import STUDY_AREA, TO_LL, TO_XY, nearest_grid_indices


def apply_corrections(paths, independent_dir, acquisitions):
    names = geometry_acquisition_ids(paths["analysis_geometry"])
    for key, dac_applied in (("region_layer", False), ("residual_layer", True)):
        if not paths[key].is_file():
            raise FileNotFoundError(
                f"Missing phase repair layer {paths[key]}. Download the saved layers listed in the README."
            )
        check_layer(paths[key], names, dac_applied)

    # Prepare the REMA/geoid screening reference.
    for band in band_indices(paths["analysis_geometry"]):
        build_screening_reference(
            paths["raw_cube"],
            paths["analysis_geometry"],
            paths["native_rema"],
            paths["reference_bands"],
            band,
        )

    # Combine Raster and native repairs, enforce the range guard and remove DAC.
    for date_index in range(acquisitions):
        assemble_corrected_acquisition(date_index, paths, independent_dir)
    write_reference_clocks(paths["analysis_geometry"], independent_dir / "rows")

    # Arrange the corrected rows into the spatial bands defined by the geometry.
    for band in band_indices(paths["analysis_geometry"]):
        assemble_observation_band(band, paths, independent_dir, acquisitions)


def main(argv=None):
    args = argument_parser(__doc__).parse_args(argv)
    setup(args)
    if args.dry_run:
        print(f"Work folder: {args.work_dir}")
        print(
            "Apply the saved phase repairs and AVISO DAC and assemble corrected observations."
        )
        return
    apply_corrections(args.paths, args.independent, args.acquisitions)


# AVISO dynamic atmospheric correction


DAC_EPOCH = datetime(1950, 1, 1)


def _utc_naive(when: datetime) -> datetime:
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    return when


def _filename(when: datetime) -> str:
    when = _utc_naive(when)
    cnes_day = int((when - DAC_EPOCH).total_seconds() // 86400)
    return f"dac_dif_{cnes_day}_{when.hour:02d}.nc"


class AVISODAC:
    """Read and sample the local six-hour AVISO fields."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self._datasets: OrderedDict[Path, xr.Dataset] = OrderedDict()
        self._grid_axes = None
        self._triangulation = None
        self._grid_indices = None

    def cache_path(self, when: datetime) -> Path:
        when = _utc_naive(when)
        return self.cache_dir / str(when.year) / _filename(when)

    @staticmethod
    def _read(path: Path) -> xr.Dataset:
        with xr.open_dataset(path) as source:
            for name in ("dac", "latitude", "longitude"):
                if name not in source:
                    raise ValueError(f"Missing {name} in AVISO field {path}")
            if source.latitude.dims != ("latitude",) or source.longitude.dims != (
                "longitude",
            ):
                raise ValueError(
                    "AVISO coordinates must be one-dimensional latitude/longitude"
                )
            data = source["dac"]
            for dim in set(data.dims) - {"latitude", "longitude"}:
                if data.sizes[dim] != 1:
                    raise ValueError(f"Expected one DAC time slice in {path}")
                data = data.isel({dim: 0}, drop=True)
            units = str(data.attrs.get("units", "")).strip().lower()
            if units not in ("m", "meter", "meters", "metre", "metres"):
                raise ValueError(
                    f"Expected DAC in metres, got units={units!r} in {path}"
                )
            latitude = source.latitude.values.astype(float)
            longitude = source.longitude.values.astype(float)
            if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
                raise ValueError(f"Nonfinite AVISO grid coordinates in {path}")
            if np.any(np.abs(latitude) > 90):
                raise ValueError(f"Invalid AVISO latitude in {path}")
            rows = np.flatnonzero(latitude < -39.0)
            if rows.size < 2 or longitude.size < 2:
                raise ValueError(f"Insufficient Antarctic DAC grid support in {path}")
            data = data.isel(latitude=rows).transpose("latitude", "longitude").load()
        return data.to_dataset(name="dac")

    def _open(self, path: Path) -> xr.Dataset:
        ds = self._datasets.pop(path, None)
        if ds is None:
            ds = self._read(path)
        self._datasets[path] = ds
        while len(self._datasets) > 4:
            self._datasets.popitem(last=False)
        return ds

    def _prepare_geometry(self, ds: xr.Dataset):
        axes = (ds.longitude.values, ds.latitude.values)
        if self._grid_axes is not None and all(
            np.array_equal(a, b) for a, b in zip(axes, self._grid_axes)
        ):
            return
        lon, lat = np.meshgrid(*axes)
        x, y = TO_XY.transform(lon.ravel(), lat.ravel())
        points = np.column_stack((x, y))
        # The geographic grid can contain coincident nodes at the pole.
        _, indices = np.unique(points, axis=0, return_index=True)
        self._grid_indices = np.sort(indices)
        self._triangulation = Delaunay(points[self._grid_indices])
        self._grid_axes = tuple(a.copy() for a in axes)

    def _spatial_sample(self, ds: xr.Dataset, lon, lat) -> np.ndarray:
        lon, lat = np.broadcast_arrays(
            np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        )
        result = np.full(lon.shape, np.nan)
        valid = np.isfinite(lon) & np.isfinite(lat) & (lat >= -90.0) & (lat < -39.0)
        if not np.any(valid):
            return result
        self._prepare_geometry(ds)
        # Keep masked DAC vertices: removing them would bridge unsupported areas.
        values = ds.dac.values.astype(float).ravel()[self._grid_indices]
        interpolator = LinearNDInterpolator(
            self._triangulation, values, fill_value=np.nan
        )
        x, y = TO_XY.transform(lon[valid], lat[valid])
        result[valid] = interpolator(x, y)
        return result

    def sample(self, when: datetime, lon, lat) -> np.ndarray:
        path = self.cache_path(when)
        if not path.is_file():
            raise FileNotFoundError(f"Missing six-hour AVISO DAC field: {path}")
        return self._spatial_sample(self._open(path), lon, lat)


# Observation geometry and corrected observation bands


def bound(dataset, value, upper=False, start=0, stop=None):
    """Binary-search a sorted on-disk cell index without reading it in full."""
    low, high = start, len(dataset) if stop is None else stop
    while low < high:
        middle = (low + high) // 2
        at_middle = int(dataset[middle])
        if at_middle < value or (upper and at_middle == value):
            low = middle + 1
        else:
            high = middle
    return low


def sorted_fine_cells(mapping):
    flat = mapping.ravel()
    positions = np.flatnonzero(flat >= 0)
    positions = positions[np.argsort(flat[positions])]
    return flat[positions], positions


def match_band_cells(source, fine, destination):
    """Read only row IDs that could fall inside this band."""
    begin = bound(source, int(fine[0]))
    end = bound(source, int(fine[-1]), upper=True)
    window = slice(begin, end)
    row_ids = source[window]
    positions = np.searchsorted(fine, row_ids)
    matched = positions < len(fine)
    matched[matched] &= fine[positions[matched]] == row_ids[matched]
    return window, matched, destination[positions[matched]]


def decode_text(a):
    """Decode HDF5 text values."""
    return [value.decode() if isinstance(value, bytes) else str(value) for value in a]


def check_layer(path, names, dac_applied):
    """Check the repair layer's acquisition order, offsets and DAC convention."""
    with h5py.File(path) as layer:
        dates = layer["acquisition_id"].asstr()
        offsets = layer["observation_start"][:]
        if (
            layer.attrs.get("complete") != 1
            or dates[: len(names)].tolist() != names
            or bool(layer.attrs.get("DAC_already_applied", False)) != dac_applied
            or offsets.shape != (len(dates) + 1,)
            or offsets[0] != 0
            or (np.diff(offsets) < 0).any()
            or any(layer[name].shape != (offsets[-1],) for name in ("cell_id", "after_wse"))
        ):
            raise ValueError(f"Invalid or mismatched saved phase repairs: {path}")


def repair_layer(layer, index, acquisition_id):
    """Read one acquisition from a saved native-phase layer."""
    if layer["acquisition_id"].asstr()[index] != acquisition_id:
        raise ValueError(f"Phase layer {layer.filename} does not match {acquisition_id}")
    start, stop = map(int, layer["observation_start"][index : index + 2])
    return layer["cell_id"][start:stop], layer["after_wse"][start:stop]


def sample_dac(provider, seconds, lon, lat):
    """Interpolate six-hour AVISO fields at each observation time and location."""
    dac_values = np.full(len(seconds), np.nan)
    six_hour_bins = np.floor(seconds / 21600).astype("i8")
    for time_bin in np.unique(six_hour_bins):
        selected = six_hour_bins == time_bin
        low = np.datetime64("2000-01-01") + np.timedelta64(int(time_bin) * 21600, "s")
        high = low + np.timedelta64(21600, "s")
        lower_dac = provider.sample(
            low.astype("datetime64[us]").astype(object), lon[selected], lat[selected]
        )
        upper_dac = provider.sample(
            high.astype("datetime64[us]").astype(object), lon[selected], lat[selected]
        )
        time_fraction = (seconds[selected] - time_bin * 21600) / 21600
        dac_values[selected] = (
            1 - time_fraction
        ) * lower_dac + time_fraction * upper_dac
    return dac_values


def collect_raster_clocks(
    cells, tile_cells, valid, times, cross_track, count, total, minimum
):
    """Accumulate clocks and the smallest absolute cross-track distance per cell."""
    positions = np.searchsorted(cells, tile_cells)
    hit = positions < len(cells)
    hit[hit] &= cells[positions[hit]] == tile_cells[hit]
    hit &= valid
    np.add.at(count, positions[hit], 1)
    np.add.at(total, positions[hit], times[hit])
    np.minimum.at(minimum, positions[hit], abs(cross_track[hit]))


def assemble_corrected_acquisition(index, resources, out):
    """Prepare one acquisition: native repairs, source-range guard and DAC once.

    The region layer is still raw WSE; the residual layer already has DAC removed.
    Save base values and residual updates separately to preserve clock arithmetic.
    """
    dates = geometry_acquisition_ids(Path(out) / "geometry.h5")
    if not 0 <= index < len(dates):
        raise ValueError("Row index is outside the selected acquisition list")
    acquisition_id = dates[index]
    output_path = Path(out) / "rows" / f"{index:03d}.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with h5py.File(output_path) as handle:
            if (
                handle.attrs.get("complete") != 1
                or handle.attrs.get("acquisition") != acquisition_id
            ):
                raise ValueError(f"Stale or incomplete row: {output_path}")
        return
    with h5py.File(Path(out) / "geometry.h5") as geometry_info:
        x, y = geometry_info["x100"][:], geometry_info["y100"][:]
        mask = np.zeros(len(x) * len(y), bool)
        for start in range(0, len(geometry_info["cell_id"]), 8192):
            fine_mapping = geometry_info["fine_id"][start : start + 8192]
            mask[fine_mapping[fine_mapping >= 0]] = True
    base_ids = np.array([], dtype="i8")
    base_height = np.array([], dtype="f4")
    base_when = np.array([], dtype="f8")
    update_ids = np.array([], dtype="i8")
    update_height = np.array([], dtype="f4")
    update_when = np.array([], dtype="f8")
    cross_track_min = SETTINGS["screening"]["minimum_abs_source_cross_track_m"]
    with h5py.File(resources["raw_cube"]) as handle:
        names = decode_text(handle["acquisitions/id"][:])
        assert acquisition_id in names
        tiles = np.asarray(decode_text(handle["tiles/acquisition"][:]))
        starts = handle["tiles/observation_start"][:]
        observations = handle["observations"]
        with h5py.File(resources["region_layer"]) as phase_file:
            affected, region_height = repair_layer(
                phase_file, index, acquisition_id
            )
        with h5py.File(resources["residual_layer"]) as phase_file:
            newids, repaired_height = repair_layer(
                phase_file, index, acquisition_id
            )
        # Preserve explicit source exclusions by identity, regardless of catalogue size.
        if acquisition_id in SETTINGS["excluded_acquisitions"]:
            cell_ids = np.array([], dtype="i8")
            height = when = np.array([], dtype="f4")
        else:
            good = (
                mask[affected]
                & np.isfinite(region_height)
                & (abs(region_height) < 1e20)
            )
            native_ids = affected[good]
            native_height = region_height[good]
            native_count = np.zeros(len(native_ids), "u2")
            native_time_sum = np.zeros(len(native_ids))
            minimum = np.full(len(native_ids), np.inf)
            keep = mask[newids]
            newids, repaired_height = newids[keep], repaired_height[keep]
            repair_count = np.zeros(len(newids), "u2")
            repair_time_sum = np.zeros(len(newids))
            minimum_cross_track = np.full(len(newids), np.inf)
            pieces = {k: [] for k in ["cell_id", "wse", "time"]}
            for tile in np.flatnonzero(tiles == acquisition_id):
                for start in range(int(starts[tile]), int(starts[tile + 1]), 250000):
                    stop = min(start + 250000, int(starts[tile + 1]))
                    fine_ids = observations["cell_id"][start:stop]
                    selected = mask[fine_ids]
                    if not selected.any():
                        continue
                    fine_ids = fine_ids[selected]
                    height_values = observations["wse"][start:stop][selected]
                    time_values = observations["illumination_time"][start:stop][
                        selected
                    ]
                    cross_track = observations["cross_track"][start:stop][selected]
                    metadata_valid = (
                        np.isfinite(cross_track + time_values)
                        & (abs(cross_track) < 1e20)
                        & (abs(time_values) < 1e20)
                    )
                    if len(native_ids):
                        collect_raster_clocks(
                            native_ids,
                            fine_ids,
                            metadata_valid,
                            time_values,
                            cross_track,
                            native_count,
                            native_time_sum,
                            minimum,
                        )
                    if len(newids):
                        collect_raster_clocks(
                            newids,
                            fine_ids,
                            metadata_valid,
                            time_values,
                            cross_track,
                            repair_count,
                            repair_time_sum,
                            minimum_cross_track,
                        )
                    moved = np.isin(fine_ids, affected, assume_unique=False)
                    keep = (
                        metadata_valid
                        & np.isfinite(height_values)
                        & (abs(height_values) < 1e20)
                        & (abs(cross_track) >= cross_track_min)
                        & ~moved
                    )
                    for k, v in [
                        ("cell_id", fine_ids),
                        ("wse", height_values),
                        ("time", time_values),
                    ]:
                        pieces[k].append(v[keep])
            selected = (native_count > 0) & (minimum >= cross_track_min)
            for k, v in [
                ("cell_id", native_ids[selected]),
                ("wse", native_height[selected]),
                ("time", native_time_sum[selected] / native_count[selected]),
            ]:
                pieces[k].append(v)
            arrays = {k: np.concatenate(v) for k, v in pieces.items()}
            cell_ids, inverse, count = np.unique(
                arrays["cell_id"], return_inverse=True, return_counts=True
            )
            wse = (np.bincount(inverse, weights=arrays["wse"]) / count).astype("f4")
            when = np.bincount(inverse, weights=arrays["time"]) / count
            lon, lat = TO_LL.transform(x[cell_ids % len(x)], y[cell_ids // len(x)])
            dac = sample_dac(
                AVISODAC(resources["aviso_cache"]), when, lon, lat
            ).astype("f4")
            # The calibrated extra cross-track correction is zero.
            height = (wse.astype(float) - dac.astype(float)).astype("f4")
            base_ids, base_height, base_when = cell_ids, height, when
            # A residual repair still needs Raster support, including at vacated cells.
            known = (repair_count > 0) & (minimum_cross_track >= cross_track_min)
            value = np.where(
                known & np.isfinite(repaired_height) & (abs(repaired_height) < 1e20),
                repaired_height,
                np.nan,
            ).astype("f4")
            newtime = np.divide(
                repair_time_sum,
                repair_count,
                out=np.full(len(repair_count), np.nan),
                where=repair_count > 0,
            )
            update_ids, update_height, update_when = newids, value, newtime
            union = np.union1d(cell_ids, newids)
            merged_height = np.full(len(union), np.nan, "f4")
            merged_time = np.full(len(union), np.nan)
            positions_in_union = np.searchsorted(union, cell_ids)
            merged_height[positions_in_union] = height
            merged_time[positions_in_union] = when
            positions_in_union = np.searchsorted(union, newids)
            merged_height[positions_in_union] = value
            merged_time[positions_in_union] = newtime
            cell_ids, height, when = union, merged_height, merged_time
    with h5py.File(output_path.with_suffix(".tmp.h5"), "w") as handle:
        for k, v in dict(
            cell_id=cell_ids,
            height=height,
            time_seconds=when,
            base_cell_id=base_ids,
            base_height=base_height,
            base_time_seconds=base_when,
            update_cell_id=update_ids,
            update_height=update_height,
            update_time_seconds=update_when,
        ).items():
            ds(handle, k, v)
        handle.attrs.update(
            complete=1,
            acquisition=acquisition_id,
            DAC_applied_once=1,
            datum="WSE above mean-tide geoid, ocean tide retained",
        )
    output_path.with_suffix(".tmp.h5").replace(output_path)


def write_reference_clocks(geometry, rows):
    """Set fallback clocks from the Raster times before residual phase repairs."""
    with h5py.File(geometry) as grid:
        if "reference_seconds" in grid:
            return
        dates = grid["acquisition_id"].asstr()[:]
        reference = (
            grid["time_ns"][:].astype("datetime64[ns]") - np.datetime64("2000-01-01")
        ) / np.timedelta64(1, "s")
        x, y = grid["x100"][:], grid["y100"][:]

    # The October pass used the original interior calibration sample.
    sample_dates = SETTINGS["reference_clock_sample_acquisitions"]
    if set(dates).intersection(sample_dates):
        with np.load(AUX / "swot_reference_sample.npz") as sample:
            sample_x, sample_y = sample["x"], sample["y"]
        column, row, inside = nearest_grid_indices(sample_x, sample_y, x, y)
        column, row = column[inside], row[inside]
        aligned = (x[column] == sample_x[inside]) & (y[row] == sample_y[inside])
        sample_cells = (row * len(x) + column)[aligned]

    for index, date in enumerate(dates):
        with h5py.File(rows / f"{index:03d}.h5") as observations:
            times = observations["base_time_seconds"][:]
            if date in sample_dates:
                cells = observations["base_cell_id"][:]
                times = times[np.isin(cells, sample_cells)]
            if len(times):
                reference[index] = np.median(times)

    with TemporaryDirectory(dir=geometry.parent) as directory:
        temporary = Path(directory) / geometry.name
        copyfile(geometry, temporary)
        with h5py.File(temporary, "r+") as grid:
            ds(grid, "reference_seconds", reference)
        temporary.replace(geometry)


def assemble_observation_band(index, resources, out, nt=DEFAULT_ACQUISITIONS):
    """Pack one geometry band, preserving base/subtract/add clock order.

    Outputs height[date, cell, fine] and time_seconds[date, cell]. No new screen
    or fit is applied in this step.
    """
    output_path = Path(out) / "inputs" / f"{index:03d}.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with h5py.File(output_path) as handle:
            assert handle.attrs["complete"] == 1
        return
    dates = geometry_acquisition_ids(Path(out) / "geometry.h5")
    if len(dates) != nt:
        raise ValueError("Band size does not match the selected acquisitions")
    with h5py.File(Path(out) / "geometry.h5") as geometry_info:
        band_positions = np.flatnonzero(geometry_info["source_band"][:] == index)
        cell_ids = geometry_info["cell_id"][band_positions]
        fine_mapping = geometry_info["fine_id"][band_positions]
        meta = {
            name: geometry_info[name][:]
            for name in ["time_ns", "passes", "reference_seconds"]
        }
    fine, dest = sorted_fine_cells(fine_mapping)
    cell_count = len(fine_mapping)
    heights = np.full((nt, cell_count, 9), np.nan, "f4")
    cell_times = np.full((nt, cell_count), np.nan)
    for date in range(nt):
        with h5py.File(Path(out) / "rows" / f"{date:03d}.h5") as handle:
            if (
                handle.attrs.get("complete") != 1
                or handle.attrs.get("acquisition") != dates[date]
            ):
                raise ValueError(f"Stale or incomplete row for {dates[date]}")
            window, matched, destination = match_band_cells(
                handle["base_cell_id"], fine, dest
            )
            height_values = handle["base_height"][window][matched]
            time_values = handle["base_time_seconds"][window][matched]
            heights[date].ravel()[destination] = height_values
            base_clock = np.full(fine_mapping.size, np.nan)
            base_clock[destination] = time_values
            good = np.isfinite(height_values)
            observation_count = np.zeros(cell_count, "u1")
            total = np.zeros(cell_count)
            np.add.at(observation_count, destination[good] // 9, 1)
            np.add.at(total, destination[good] // 9, time_values[good])
            clock = np.divide(
                total,
                observation_count,
                out=np.full(cell_count, np.nan),
                where=observation_count > 0,
            )
            window, matched, destination = match_band_cells(
                handle["update_cell_id"], fine, dest
            )
            if len(destination):
                coarse_row = destination // 9
                value = handle["update_height"][window][matched]
                time_values = handle["update_time_seconds"][window][matched]
                old_valid = np.isfinite(heights[date].ravel()[destination])
                old_time = base_clock[destination]
                assert np.isfinite(old_time[old_valid]).all()
                number = np.isfinite(heights[date]).sum(1)
                total = np.where(number > 0, clock * number, 0)
                np.add.at(total, coarse_row[old_valid], -old_time[old_valid])
                okay = np.isfinite(value)
                np.add.at(total, coarse_row[okay], time_values[okay])
                heights[date].ravel()[destination] = value
                number = np.isfinite(heights[date]).sum(1)
                clock = np.divide(
                    total, number, out=np.full(cell_count, np.nan), where=number > 0
                )
            cell_times[date] = clock
    # The selection reference is fixed before fitting. It is
    # never subtracted from heights or used as a coefficient prior.
    with h5py.File(Path(resources["reference_bands"]) / f"{index:03d}.h5") as handle:
        np.testing.assert_array_equal(handle["cell_id"][:], cell_ids)
        reference = handle["reference_geoid"][:]
    with h5py.File(output_path.with_suffix(".tmp.h5"), "w") as handle:
        for name, values in dict(
            cell_id=cell_ids,
            fine_id=fine_mapping,
            reference_geoid=reference,
            time_seconds=cell_times,
            **{name: values[:nt] for name, values in meta.items()},
        ).items():
            ds(handle, name, values)
        handle.create_dataset(
            "height",
            data=heights,
            chunks=(1, min(cell_count, 1024), 9),
            compression="gzip",
            compression_opts=2,
            shuffle=True,
        )
        handle.attrs.update(
            complete=1,
            reference_selection_only=1,
            initial_weights="p=1; reported Raster sigma unused",
        )
    output_path.with_suffix(".tmp.h5").replace(output_path)


# REMA and geoid reference for screening


def selection_geoid(raw, fine, ordered_titles, overwrite):
    """Sample source geoids in the frozen tile order and overwrite convention."""
    values = np.full(len(fine), np.nan)
    if not len(fine):
        return values
    with h5py.File(raw) as handle:
        titles = decode_text(handle["tiles/title"][:])
        lookup = {geoid_values: i for i, geoid_values in enumerate(titles)}
        starts = handle["tiles/observation_start"][:]
        observations = handle["observations"]
        for name in ordered_titles:
            if np.isfinite(values).all():
                break
            if name not in lookup:
                continue
            tile = lookup[name]
            tile_start, tile_stop = map(int, starts[tile : tile + 2])
            tile_start = bound(
                observations["cell_id"], int(fine[0]), start=tile_start, stop=tile_stop
            )
            tile_stop = bound(
                observations["cell_id"],
                int(fine[-1]),
                upper=True,
                start=tile_start,
                stop=tile_stop,
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


def selection_rema(path, ids, x, y):
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


def build_screening_reference(raw, geometry, rema_path, destination, band):
    """Write one REMA-minus-geoid reference band used for admission only."""
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{band:03d}.h5"
    if target.exists():
        raise FileExistsError(target)
    end_date = SETTINGS["geoid_reference_end_date"]
    order = [
        filename
        for acquisition in selected_acquisitions()
        if acquisition["id"].split("_", 1)[0] <= end_date
        for filename in acquisition["tiles"]
    ]
    with h5py.File(geometry) as geometry_file:
        x, y = geometry_file["x100"][:], geometry_file["y100"][:]
        positions = np.flatnonzero(geometry_file["source_band"][:] == band)
        fine_mapping = geometry_file["fine_id"][positions]
        cell_ids = geometry_file["cell_id"][positions]
        fine, fine_destination = sorted_fine_cells(fine_mapping)
        geoid_values = selection_geoid(raw, fine, order, overwrite=False)
        reference = np.full(fine_mapping.size, np.nan)
        reference[fine_destination] = (
            selection_rema(rema_path, fine, x, y) - geoid_values
        )
        # Preserve the original regional reference-read convention, including
        # its tile-order overwrite rule; no silent geoid datum replacement.
        lon, lat = geometry_file["lon"][:], geometry_file["lat"][:]
        boxes = STUDY_AREA["geoid_overwrite_boxes_west_east_south_north"]
        for west, east, south, north in boxes:
            loc = np.flatnonzero(
                (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
            )
            regional_fine_mapping = geometry_file["fine_id"][loc]
            regional_fine_ids = np.sort(
                regional_fine_mapping[regional_fine_mapping >= 0]
            )
            regional_geoid = selection_geoid(
                raw, regional_fine_ids, order, overwrite=True
            )
            regional_reference = (
                selection_rema(rema_path, regional_fine_ids, x, y) - regional_geoid
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


if __name__ == "__main__":
    main()

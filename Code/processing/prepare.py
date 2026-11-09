"""Prepare analysis cubes from raw Raster observations and native repair layers.

geometry() -> shared grid and clocks
row()      -> one acquisition, after native repairs and DAC exactly once
band()     -> all acquisition rows for one spatial band defined by the geometry

The stored base/update clocks preserve the original arithmetic order.
"""

from pathlib import Path


import h5py
import numpy as np
from pyproj import Transformer
from study_area import study_area_mask
from analysis import ds, SETTINGS
from layout import DEFAULT_ACQUISITIONS, PARTITIONS, selected_acquisitions
from aviso_dac import AVISODAC
from cats import load_cats, cats_constants

TO_LL = Transformer.from_crs(3031, 4326, always_xy=True)


def axis(name):
    """Construct one fixed grid axis from analysis.json."""
    definition = SETTINGS["grid"][name]
    return definition["start"] + definition["step"] * np.arange(definition["size"])


def bound(dataset, value, upper=False):
    """Binary-search a sorted on-disk cell index without reading it in full."""
    low, high = 0, len(dataset)
    while low < high:
        middle = (low + high) // 2
        if int(dataset[middle]) < value or (upper and int(dataset[middle]) == value):
            low = middle + 1
        else:
            high = middle
    return low


def text(a):
    """Decode HDF5 text values."""
    return [value.decode() if isinstance(value, bytes) else str(value) for value in a]


def domain_mapping(x, y, cx, cy, mask):
    # Admission is evaluated at the original 250 m centres first. A fine
    # cell cannot introduce a new coarse cell across the shelf boundary.
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


def geometry(out, bedmachine, cats, native_bedmachine, nt=DEFAULT_ACQUISITIONS):
    """Write geometry.h5: grid, fine-cell membership, clocks and comparison CATS."""
    target = Path(out) / "geometry.h5"
    if target.exists():
        with h5py.File(target) as handle:
            assert handle.attrs["complete"] == 1
        return
    x, y, cx, cy = [axis(name) for name in ["x100", "y100", "global_x", "global_y"]]
    cell_ids, fine_mapping = domain_mapping(
        x, y, cx, cy, lambda xx, yy: study_area_mask(xx, yy, bedmachine)
    )
    assert len(cell_ids) == SETTINGS["domain_cells"]
    lon, lat = TO_LL.transform(cx[cell_ids % len(cx)], cy[cell_ids // len(cx)])
    cats_coefficients = np.empty((8, len(cell_ids)))
    model = load_cats(cats)
    for start in range(0, len(cell_ids), 8192):
        chunk = slice(start, start + 8192)
        amplitude, phase = cats_constants(
            cx[cell_ids[chunk] % len(cx)], cy[cell_ids[chunk] // len(cx)], model
        )
        cats_coefficients[0::2, chunk] = amplitude * np.cos(np.deg2rad(phase))
        cats_coefficients[1::2, chunk] = amplitude * np.sin(np.deg2rad(phase))
    # Preserve the actual historical BedMachine2km annotation. Every one
    # of these cells remains a free unknown in the all-free GLS equations.
    from scipy.ndimage import distance_transform_edt

    with np.load(native_bedmachine) as bedmachine_data:
        bed_x, bed_y = bedmachine_data["x"], bedmachine_data["y"]
        ground = (bedmachine_data["mask"] == 1) | (bedmachine_data["mask"] == 2)
        distance = distance_transform_edt(
            ~ground, sampling=(abs(bed_y[1] - bed_y[0]), abs(bed_x[1] - bed_x[0]))
        )
        bed_column = np.rint(
            (cx[cell_ids % len(cx)] - bed_x[0]) / (bed_x[1] - bed_x[0])
        ).astype(int)
        bed_row = np.rint(
            (cy[cell_ids // len(cx)] - bed_y[0]) / (bed_y[1] - bed_y[0])
        ).astype(int)
        ground_distance = distance[bed_row, bed_column]
        protected = ground_distance <= 2000
    band_count = PARTITIONS["observation_bands"]
    bands = np.repeat(
        np.arange(band_count),
        [len(values) for values in np.array_split(cell_ids, band_count)],
    ).astype("i1")
    local_index = np.concatenate(
        [
            np.arange(len(values), dtype="i4")
            for values in np.array_split(cell_ids, band_count)
        ]
    )
    acquisitions = selected_acquisitions(nt, SETTINGS)
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
            protected=protected,
            distance_to_ground_m=ground_distance,
            time_ns=[acquisition["time_ns"] for acquisition in acquisitions],
            passes=[acquisition["pass"] for acquisition in acquisitions],
            reference_seconds=[
                acquisition.get(
                    "reference_seconds_2000", acquisition.get("reference_seconds")
                )
                for acquisition in acquisitions
            ],
            time_reference_seconds190=[
                acquisition["reference_seconds_2000"]
                for acquisition in acquisitions[: len(SETTINGS["acquisitions"])]
            ],
        ).items():
            ds(handle, name, values)
        handle.attrs.update(
            complete=1,
            band_ids=np.arange(band_count),
            fit_parts_per_band=PARTITIONS["fit_parts_per_band"],
            reference_epoch_seconds_2000=SETTINGS[
                "original190_time_center_seconds_2000"
            ],
            time_center_seconds_2000=SETTINGS["original190_time_center_seconds_2000"],
            time_span_seconds=SETTINGS["original190_time_span_seconds"],
            protected_definition="BedMachine1/2 nearest500m EDT distance<=2000m; diagnostic only. All eligible cells free in GLS.",
            crs="EPSG:3031",
        )
    target.with_suffix(".tmp.h5").replace(target)


def layer(f, index, name):
    """Read one acquisition from a packed native-phase repair layer."""
    group = f[name] if name in f else f
    start, stop = map(int, group["observation_start"][index : index + 2])
    return group["cell_id"][start:stop], group["after_wse"][start:stop]


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


def row(index, resources, out):
    """Prepare one acquisition: native repairs, source-range guard and DAC once.

    The region layer is still raw WSE; the residual layer already has DAC removed.
    Save base values and residual updates separately to preserve clock arithmetic.
    """
    output_path = Path(out) / "rows" / f"{index:03d}.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with h5py.File(output_path) as handle:
            assert handle.attrs["complete"] == 1
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
    acquisition_id = (
        SETTINGS["acquisitions"][index]["id"]
        if index < len(SETTINGS["acquisitions"])
        else SETTINGS["added_acquisition"]["acquisition_id"]
    )
    with h5py.File(resources["raw_cube"]) as handle:
        names = text(handle["acquisitions/id"][:])
        assert acquisition_id in names
        tiles = np.asarray(text(handle["tiles/acquisition"][:]))
        starts = handle["tiles/observation_start"][:]
        observations = handle["observations"]
        # Original190 row189 had no admitted analysis observations. Appending
        # Oct1 must not backfill that row from another source catalogue.
        if index in SETTINGS["empty_original190_analysis_rows"]:
            cell_ids = np.array([], dtype="i8")
            height = when = np.array([], dtype="f4")
        else:
            if index < len(SETTINGS["acquisitions"]):
                with h5py.File(resources["region_layer"]) as phase_file:
                    affected, region_height = layer(phase_file, index, "phase_region")
            else:
                affected = np.array([], dtype="i8")
                region_height = np.array([])
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
                        positions = np.searchsorted(native_ids, fine_ids)
                        hit = positions < len(native_ids)
                        hit[hit] &= native_ids[positions[hit]] == fine_ids[hit]
                        hit &= metadata_valid
                        np.add.at(native_count, positions[hit], 1)
                        np.add.at(native_time_sum, positions[hit], time_values[hit])
                        np.minimum.at(minimum, positions[hit], abs(cross_track[hit]))
                    moved = np.isin(fine_ids, affected, assume_unique=False)
                    keep = (
                        metadata_valid
                        & np.isfinite(height_values)
                        & (abs(height_values) < 1e20)
                        & (abs(cross_track) >= 10000)
                        & ~moved
                    )
                    for k, v in [
                        ("cell_id", fine_ids),
                        ("wse", height_values),
                        ("time", time_values),
                    ]:
                        pieces[k].append(v[keep])
            selected = (native_count > 0) & (minimum >= 10000)
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
                AVISODAC(resources["aviso_cache"], allow_download=False), when, lon, lat
            ).astype("f4")
            # Frozen extra-cross-track theta is exactly zero for all190 rows.
            height = (wse.astype(float) - dac.astype(float)).astype("f4")
            base_ids, base_height, base_when = (
                cell_ids.copy(),
                height.copy(),
                when.copy(),
            )
            if index < len(SETTINGS["acquisitions"]):
                with h5py.File(resources["residual_layer"]) as phase_file:
                    newids, repaired_height = layer(phase_file, index, "phase_residual")
                keep = mask[newids]
                newids, repaired_height = newids[keep], repaired_height[keep]
                # Conservative actual Raster metadata at moved100m cells,
                # including locations vacated by a residual native repair.
                repair_count = np.zeros(len(newids), "u2")
                repair_time_sum = np.zeros(len(newids))
                minimum_cross_track = np.full(len(newids), np.inf)
                for tile in np.flatnonzero(tiles == acquisition_id):
                    for start in range(
                        int(starts[tile]), int(starts[tile + 1]), 250000
                    ):
                        stop = min(start + 250000, int(starts[tile + 1]))
                        fine_ids = observations["cell_id"][start:stop]
                        positions = np.searchsorted(newids, fine_ids)
                        hit = positions < len(newids)
                        hit[hit] &= newids[positions[hit]] == fine_ids[hit]
                        if not hit.any():
                            continue
                        cross_track = observations["cross_track"][start:stop][hit]
                        time_values = observations["illumination_time"][start:stop][hit]
                        positions = positions[hit]
                        valid = (
                            np.isfinite(cross_track + time_values)
                            & (abs(cross_track) < 1e20)
                            & (abs(time_values) < 1e20)
                        )
                        positions = positions[valid]
                        np.add.at(repair_count, positions, 1)
                        np.add.at(repair_time_sum, positions, time_values[valid])
                        np.minimum.at(
                            minimum_cross_track, positions, abs(cross_track[valid])
                        )
                known = (
                    (repair_count > 0) & (minimum_cross_track >= 10000) & mask[newids]
                )
                value = np.where(
                    known
                    & np.isfinite(repaired_height)
                    & (abs(repaired_height) < 1e20),
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


def band(index, resources, out, nt=DEFAULT_ACQUISITIONS):
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
    with h5py.File(Path(out) / "geometry.h5") as geometry_info:
        band_positions = np.flatnonzero(geometry_info["source_band"][:] == index)
        cell_ids = geometry_info["cell_id"][band_positions]
        fine_mapping = geometry_info["fine_id"][band_positions]
        meta = {
            name: geometry_info[name][:]
            for name in ["time_ns", "passes", "reference_seconds"]
        }
    flat = np.flatnonzero(fine_mapping.ravel() >= 0)
    order = np.argsort(fine_mapping.ravel()[flat])
    fine = fine_mapping.ravel()[flat][order]
    dest = flat[order]
    cell_count = len(fine_mapping)
    heights = np.full((nt, cell_count, 9), np.nan, "f4")
    cell_times = np.full((nt, cell_count), np.nan)
    for date in range(nt):
        with h5py.File(Path(out) / "rows" / f"{date:03d}.h5") as handle:
            source = handle["base_cell_id"]
            begin = bound(source, int(fine[0]))
            end = bound(source, int(fine[-1]), upper=True)
            row_ids = source[begin:end]
            positions = np.searchsorted(fine, row_ids)
            matched = positions < len(fine)
            matched[matched] &= fine[positions[matched]] == row_ids[matched]
            destination = dest[positions[matched]]
            height_values = handle["base_height"][begin:end][matched]
            time_values = handle["base_time_seconds"][begin:end][matched]
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
            source = handle["update_cell_id"]
            begin = bound(source, int(fine[0]))
            end = bound(source, int(fine[-1]), upper=True)
            row_ids = source[begin:end]
            positions = np.searchsorted(fine, row_ids)
            matched = positions < len(fine)
            matched[matched] &= fine[positions[matched]] == row_ids[matched]
            if matched.any():
                destination = dest[positions[matched]]
                coarse_row = destination // 9
                value = handle["update_height"][begin:end][matched]
                time_values = handle["update_time_seconds"][begin:end][matched]
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
    # Selection reference is frozen before omission/191/GLS trials. It is
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

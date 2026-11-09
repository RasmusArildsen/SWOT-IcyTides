"""Assess residual phase shifts and regrid the corrected native PIXC points.

Call native_phase.setup() first. regrid_phase2() writes heights with DAC
already removed."""

import numpy as np
import h5py
from pyproj import Transformer
from aviso_dac import tile_midpoint

TO_XY = Transformer.from_crs(4326, 3031, always_xy=True)

GRID = None
FIELDS = (
    "height",
    "longitude",
    "latitude",
    "cross_track",
    "dheight_dphase",
    "dlongitude_dphase",
    "dlatitude_dphase",
    "phase_unwrapping_region",
    "classification",
    "geolocation_qual",
    "geoid",
    "solid_earth_tide",
    "pole_tide",
    "load_tide_fes",
)


def loo_median(count, lo, mid, hi, current):
    """Exact temporal median after deleting this acquisition, including ties."""
    current = np.asarray(current)
    available = np.isfinite(current)
    result = np.where((count % 2) == 0, (lo + mid) / 2, mid).astype("f4")
    odd = (count % 2) == 1
    result = np.where(available & odd & (current < mid), (mid + hi) / 2, result)
    result = np.where(available & odd & (current > mid), (lo + mid) / 2, result)
    result = np.where(available & odd & (current == mid), (lo + hi) / 2, result)
    result = np.where(available & ~odd & (current <= lo), mid, result)
    result = np.where(available & ~odd & (current > lo), lo, result)
    result = np.where((count - available.astype("i4")) >= 8, result, np.nan)
    return result


def chunk_values(group, start, stop, fields):
    result = {}
    for key in fields:
        dataset = group[key]
        values = np.asarray(dataset[start:stop])
        if values.dtype.kind == "f":
            values = values.astype("f8")
            fill = dataset.attrs.get("_FillValue")
            if fill is not None:
                values[values == np.asarray(fill).ravel()[0]] = np.nan
            values[np.abs(values) > 1e20] = np.nan
        result[key] = values
    return result


def geolocation_reject_bits(group):
    """Build the original missing/bad geolocation quality-bit mask."""
    quality = group["geolocation_qual"]
    meanings = quality.attrs["flag_meanings"]
    meanings = meanings.decode() if isinstance(meanings, bytes) else str(meanings)
    reject = 0
    for name, bit in zip(meanings.split(), quality.attrs["flag_masks"]):
        if (
            name.endswith("_missing")
            or name.endswith("_bad")
            or name
            in (
                "geolocation_is_from_refloc",
                "layover_significant",
                "phase_noise_suspect",
                "medium_phase_suspect",
            )
        ):
            reject |= int(bit)
    return reject


def assess_source(path, item, prior, sample, bulk, dac, output):
    """Evaluate supported phase candidates against the frozen leave-out reference."""
    candidates = []
    report = dict(
        file=item["file"],
        source_points=0,
        quality_points=0,
        reference_points=0,
        over_5m=0,
        eligible_points=0,
        supported_shifted_points=0,
        unresolved_over_5m=0,
        accepted_blocks=0,
        stage1_shifted_quality_points=0,
    )
    with h5py.File(path) as source:
        group = source["pixel_cloud"]
        report["source_points"] = len(group["height"])
        reject = geolocation_reject_bits(group)
        accepted = {
            int(region["region"]): int(region["candidate"])
            for region in prior.get("regions", [])
            if region.get("accepted")
        }
        for start in range(0, len(group["height"]), 131072):
            stop = min(start + 131072, len(group["height"]))
            points = chunk_values(group, start, stop, FIELDS)
            good = (
                np.isfinite(
                    points["height"]
                    + points["longitude"]
                    + points["latitude"]
                    + points["dheight_dphase"]
                    + points["dlongitude_dphase"]
                    + points["dlatitude_dphase"]
                )
                & np.isin(points["classification"], [3, 4])
                & ((points["geolocation_qual"].astype("u4") & reject) == 0)
                & (np.abs(points["cross_track"]) >= 10000)
            )
            if not good.any():
                continue
            quality_indices = np.flatnonzero(good)
            points = {key: values[good] for key, values in points.items()}
            report["quality_points"] += len(quality_indices)
            region_cycles = np.zeros(len(quality_indices), "i1")
            for region, cycle in accepted.items():
                region_cycles[points["phase_unwrapping_region"] == region] = cycle
            report["stage1_shifted_quality_points"] += int(
                np.count_nonzero(region_cycles)
            )
            phase = 2 * np.pi * region_cycles
            lon = points["longitude"] + phase * points["dlongitude_dphase"]
            lat = points["latitude"] + phase * points["dlatitude_dphase"]
            x, y = TO_XY.transform(lon, lat)
            height = points["height"] + phase * points["dheight_dphase"]
            geophysical_correction = (
                points["geoid"]
                + points["solid_earth_tide"]
                + points["pole_tide"]
                + points["load_tide_fes"]
            )
            current = (
                height
                - geophysical_correction
                - dac.sample(
                    tile_midpoint(item["file"]),
                    lon,
                    lat,
                    source_path=item["file"],
                )
            )
            baseline = np.abs(current - sample(x, y) - bulk)
            valid = np.isfinite(baseline)
            report["reference_points"] += int(valid.sum())
            trigger = valid & (baseline > 5)
            report["over_5m"] += int(trigger.sum())
            if not trigger.any():
                continue
            target = np.flatnonzero(trigger)
            best = np.full(len(target), np.inf)
            best_shift = np.zeros(len(target), "i1")
            for shift in (-2, -1, 1, 2):
                candidate_phase = 2 * np.pi * (region_cycles[target] + shift)
                candidate_longitude = (
                    points["longitude"][target]
                    + candidate_phase * points["dlongitude_dphase"][target]
                )
                candidate_latitude = (
                    points["latitude"][target]
                    + candidate_phase * points["dlatitude_dphase"][target]
                )
                candidate_x, candidate_y = TO_XY.transform(
                    candidate_longitude, candidate_latitude
                )
                candidate_height = (
                    points["height"][target]
                    + candidate_phase * points["dheight_dphase"][target]
                )
                candidate_datum_height = (
                    candidate_height
                    - geophysical_correction[target]
                    - dac.sample(
                        tile_midpoint(item["file"]),
                        candidate_longitude,
                        candidate_latitude,
                        source_path=item["file"],
                    )
                )
                score = np.abs(
                    candidate_datum_height - sample(candidate_x, candidate_y) - bulk
                )
                improve = np.isfinite(score) & (score < best)
                best[improve] = score[improve]
                best_shift[improve] = shift
            before = baseline[target]
            eligible = (best <= 5) & (best + 1 < before) & (best <= 0.65 * before)
            report["eligible_points"] += int(eligible.sum())
            if not eligible.any():
                continue
            selected = target[eligible]
            candidates.append(
                dict(
                    pixel_index=(start + quality_indices[selected]).astype("i4"),
                    region=points["phase_unwrapping_region"][selected].astype("i4"),
                    before_x=x[selected].astype("f4"),
                    before_y=y[selected].astype("f4"),
                    old_cycles=region_cycles[selected].astype("i1"),
                    candidate_shift=best_shift[eligible].astype("i1"),
                    before_abs_residual=before[eligible].astype("f4"),
                    after_abs_residual=best[eligible].astype("f4"),
                    height=points["height"][selected].astype("f4"),
                    longitude=points["longitude"][selected].astype("f8"),
                    latitude=points["latitude"][selected].astype("f8"),
                    dheight=points["dheight_dphase"][selected].astype("f4"),
                    dlon=points["dlongitude_dphase"][selected].astype("f4"),
                    dlat=points["dlatitude_dphase"][selected].astype("f4"),
                )
            )
    if not candidates:
        report["unresolved_over_5m"] = report["over_5m"]
        return report
    candidates_by_point = {
        key: np.concatenate([points[key] for points in candidates])
        for key in candidates[0]
    }
    # Pool the entire native tile before testing local support; chunk boundaries
    # must never split a coherent 1 km phase block.
    block_column = np.floor(candidates_by_point["before_x"] / 1000).astype("i4")
    block_row = np.floor(candidates_by_point["before_y"] / 1000).astype("i4")
    keys = np.column_stack(
        (
            candidates_by_point["region"],
            block_column,
            block_row,
            candidates_by_point["candidate_shift"],
        )
    )
    unique, inverse, count = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    accepted = np.zeros(len(candidates_by_point["pixel_index"]), bool)
    for group, key in enumerate(unique):
        members = np.flatnonzero(inverse == group)
        if len(members) < 12:
            continue
        same_region_block = (
            (candidates_by_point["region"] == key[0])
            & (block_column == key[1])
            & (block_row == key[2])
        )
        if len(members) / same_region_block.sum() < 0.7:
            continue
        parity = (
            np.floor(candidates_by_point["before_x"][members] / 250).astype(int)
            + np.floor(candidates_by_point["before_y"][members] / 250).astype(int)
        ) % 2
        split = []
        for candidate_phase in (0, 1):
            q = members[parity == candidate_phase]
            if len(q):
                split.append(
                    (
                        len(q),
                        float(np.median(candidates_by_point["before_abs_residual"][q])),
                        float(np.median(candidates_by_point["after_abs_residual"][q])),
                    )
                )
        if len(split) != 2 or any(n < 3 or a > 5 or b - a < 1 for n, b, a in split):
            continue
        accepted[members] = True
        report["accepted_blocks"] += 1
    report["supported_shifted_points"] = int(accepted.sum())
    report["unresolved_over_5m"] = (
        report["over_5m"] - report["supported_shifted_points"]
    )
    if accepted.any():
        candidates_by_point = {
            key: value[accepted] for key, value in candidates_by_point.items()
        }
        candidate_phase = (
            2
            * np.pi
            * (
                candidates_by_point["old_cycles"]
                + candidates_by_point["candidate_shift"]
            )
        )
        candidates_by_point["after_height"] = (
            candidates_by_point["height"]
            + candidate_phase * candidates_by_point["dheight"]
        ).astype("f4")
        candidates_by_point["after_longitude"] = (
            candidates_by_point["longitude"]
            + candidate_phase * candidates_by_point["dlon"]
        ).astype("f8")
        candidates_by_point["after_latitude"] = (
            candidates_by_point["latitude"]
            + candidate_phase * candidates_by_point["dlat"]
        ).astype("f8")
        candidates_by_point["after_x"], candidates_by_point["after_y"] = (
            TO_XY.transform(
                candidates_by_point["after_longitude"],
                candidates_by_point["after_latitude"],
            )
        )
        candidates_by_point["after_x"] = candidates_by_point["after_x"].astype("f4")
        candidates_by_point["after_y"] = candidates_by_point["after_y"].astype("f4")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.with_suffix(".partial").open("wb") as stream:
            np.savez_compressed(stream, **candidates_by_point)
        output.with_suffix(".partial").replace(output)
    return report


def grid_ids(x, y, gx, gy):
    """Map native points to valid nearest 100 m cells."""
    col = np.rint((x - gx[0]) / 100).astype("i8")
    row = np.rint((y - gy[0]) / 100).astype("i8")
    good = (
        np.isfinite(x + y) & (col >= 0) & (col < len(gx)) & (row >= 0) & (row < len(gy))
    )
    return np.where(good, row * len(gx) + col, -1)


def sparse_median(ids, values):
    """Reduce finite native heights to an ordered sparse cell median."""
    valid = (ids >= 0) & np.isfinite(values)
    ids = ids[valid]
    values = values[valid]
    if not len(ids):
        return (
            np.array([], dtype="i8"),
            np.array([], dtype="f4"),
            np.array([], dtype="i4"),
        )
    order = np.lexsort((values, ids))
    ids = ids[order]
    values = values[order]
    unique, first, count = np.unique(ids, return_index=True, return_counts=True)
    median = (values[first + (count - 1) // 2] + values[first + count // 2]) / 2
    return unique, median.astype("f4"), count.astype("i4")


def regrid_phase2(row, source_reports, paths, dac, directory):
    """Regrid all nearby quality native points after both correction stages.

    A candidate ledger alone is insufficient: unchanged neighbors and other
    PIXC sources can contribute to the moved 100 m cell. This second pass
    reads each already-downloaded source in bounded PIXC chunks.
    """
    with h5py.File(GRID) as geometry:
        grid_x = geometry["x100"][:]
        grid_y = geometry["y100"][:]
    ledgers = {}
    affected = []
    for item in row["sources"]:
        path = directory / "corrections" / f"{item['file']}.npz"
        if not path.exists():
            continue
        with np.load(path) as ledger:
            pixel_indices = ledger["pixel_index"].astype("i4")
            shifts = ledger["candidate_shift"].astype("i1")
            before = grid_ids(ledger["before_x"], ledger["before_y"], grid_x, grid_y)
            after = grid_ids(ledger["after_x"], ledger["after_y"], grid_x, grid_y)
        if not len(pixel_indices):
            continue
        if np.any(np.diff(pixel_indices) <= 0):
            raise ValueError(
                f'Native correction indices not strictly increasing: {item["file"]}'
            )
        ledgers[item["file"]] = (pixel_indices, shifts)
        affected.extend((before[before >= 0], after[after >= 0]))
    if not affected:
        return dict(affected_cells=0, grid_corrected_cells=0, native_points_regridded=0)
    union = np.unique(np.concatenate(affected)).astype("i8")
    before_ids = []
    before_values = []
    after_ids = []
    after_values = []
    points_used = 0
    for item in row["sources"]:
        if not source_reports[item["file"]].get("shelf_quality_points", 0):
            continue
        path = paths[item["file"]]
        pixel_indices, shifts = ledgers.get(
            item["file"], (np.array([], dtype="i4"), np.array([], dtype="i1"))
        )
        accepted = {
            int(region["region"]): int(region["candidate"])
            for region in source_reports[item["file"]].get("regions", [])
            if region.get("accepted")
        }
        with h5py.File(path) as ledger:
            group = ledger["pixel_cloud"]
            reject = geolocation_reject_bits(group)
            for start in range(0, len(group["height"]), 131072):
                stop = min(start + 131072, len(group["height"]))
                points = chunk_values(group, start, stop, FIELDS)
                good = (
                    np.isfinite(
                        points["height"]
                        + points["longitude"]
                        + points["latitude"]
                        + points["dheight_dphase"]
                        + points["dlongitude_dphase"]
                        + points["dlatitude_dphase"]
                    )
                    & np.isin(points["classification"], [3, 4])
                    & ((points["geolocation_qual"].astype("u4") & reject) == 0)
                )
                if not good.any():
                    continue
                native_index = start + np.flatnonzero(good)
                points = {key: values[good] for key, values in points.items()}
                base = np.zeros(len(native_index), "i1")
                for region, cycle in accepted.items():
                    base[points["phase_unwrapping_region"] == region] = cycle
                before_phase = 2 * np.pi * base
                lon0 = points["longitude"] + before_phase * points["dlongitude_dphase"]
                lat0 = points["latitude"] + before_phase * points["dlatitude_dphase"]
                before_x, before_y = TO_XY.transform(lon0, lat0)
                cell0 = grid_ids(before_x, before_y, grid_x, grid_y)
                extra = np.zeros(len(native_index), "i1")
                if len(pixel_indices):
                    positions = np.searchsorted(pixel_indices, native_index)
                    matching = (positions < len(pixel_indices)) & (
                        pixel_indices[np.minimum(positions, len(pixel_indices) - 1)]
                        == native_index
                    )
                    extra[matching] = shifts[positions[matching]]
                if extra.any():
                    after_phase = 2 * np.pi * (base + extra)
                    lon1 = (
                        points["longitude"] + after_phase * points["dlongitude_dphase"]
                    )
                    lat1 = points["latitude"] + after_phase * points["dlatitude_dphase"]
                    after_x, after_y = TO_XY.transform(lon1, lat1)
                    cell1 = grid_ids(after_x, after_y, grid_x, grid_y)
                else:
                    after_phase = before_phase
                    lon1 = lon0
                    lat1 = lat0
                    cell1 = cell0
                relevant = np.isin(cell0, union) | np.isin(cell1, union)
                if not relevant.any():
                    continue
                points_used += int(relevant.sum())
                geophysical_correction = (
                    points["geoid"]
                    + points["solid_earth_tide"]
                    + points["pole_tide"]
                    + points["load_tide_fes"]
                )[relevant]
                name = item["file"]
                midpoint = tile_midpoint(name)
                wse0 = (
                    points["height"][relevant]
                    + before_phase[relevant] * points["dheight_dphase"][relevant]
                    - geophysical_correction
                    - dac.sample(
                        midpoint, lon0[relevant], lat0[relevant], source_path=name
                    )
                )
                wse1 = (
                    points["height"][relevant]
                    + after_phase[relevant] * points["dheight_dphase"][relevant]
                    - geophysical_correction
                    - dac.sample(
                        midpoint, lon1[relevant], lat1[relevant], source_path=name
                    )
                )
                before_ids.append(cell0[relevant])
                before_values.append(wse0.astype("f4"))
                after_ids.append(cell1[relevant])
                after_values.append(wse1.astype("f4"))
    b_ids, b_values, b_count = (
        sparse_median(np.concatenate(before_ids), np.concatenate(before_values))
        if before_ids
        else sparse_median(np.array([], dtype="i8"), np.array([]))
    )
    a_ids, a_values, a_count = (
        sparse_median(np.concatenate(after_ids), np.concatenate(after_values))
        if after_ids
        else sparse_median(np.array([], dtype="i8"), np.array([]))
    )
    before = np.full(len(union), np.nan, "f4")
    after = np.full(len(union), np.nan, "f4")
    before_count = np.zeros(len(union), "i4")
    after_count = np.zeros(len(union), "i4")
    for ids, values, count, target, target_count in (
        (b_ids, b_values, b_count, before, before_count),
        (a_ids, a_values, a_count, after, after_count),
    ):
        if len(ids):
            positions = np.searchsorted(union, ids)
            good = (positions < len(union)) & (
                union[np.minimum(positions, len(union) - 1)] == ids
            )
            target[positions[good]] = values[good]
            target_count[positions[good]] = count[good]
    output = directory / "grid_phase2.npz"
    with output.with_suffix(".partial").open("wb") as stream:
        np.savez_compressed(
            stream,
            cell_id=union,
            before_wse=before,
            after_wse=after,
            before_count=before_count,
            after_count=after_count,
        )
    output.with_suffix(".partial").replace(output)
    return dict(
        affected_cells=len(union),
        grid_corrected_cells=int(np.isfinite(after).sum()),
        native_points_regridded=points_used,
        median_abs_grid_change_m=(
            float(np.nanmedian(np.abs(after - before)))
            if np.isfinite(after - before).any()
            else None
        ),
        grid_archive=str(output),
    )

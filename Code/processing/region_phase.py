"""First native region pass; configure explicit resources before process()."""

from pathlib import Path
import os
import json
import tempfile
import shutil
import requests
import h5py
import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter
from scipy.spatial import cKDTree
from pyproj import Transformer
from aviso_dac import tile_midpoint

TR = Transformer.from_crs(4326, 3031, always_xy=True)
VERSION = "whole_shelf_native_v1"
EPOCH = np.datetime64("2018-01-01", "ns")
SESSION = requests.Session()
PROC = None


def phase_adjust(height, lon, lat, dh, dlon, dlat, cycles):
    """First-order correction; n=0 is an exact no-op even for missing slopes."""
    height, lon, lat = [np.array(a, dtype=float, copy=True) for a in (height, lon, lat)]
    cycles = np.asarray(cycles)
    use = cycles != 0
    if not np.all(
        np.isfinite(np.asarray(dh)[use] + np.asarray(dlon)[use] + np.asarray(dlat)[use])
    ):
        raise ValueError(
            "Missing phase sensitivities on points selected for correction"
        )
    phase = 2 * np.pi * cycles[use]
    height[use] += phase * np.asarray(dh)[use]
    lon[use] += phase * np.asarray(dlon)[use]
    lat[use] += phase * np.asarray(dlat)[use]
    return height, lon, lat


def cell(x, y, pad=0):
    """Map projected coordinates to the nearest 100 m grid cell."""
    finite = np.isfinite(x + y)
    column = np.rint(np.where(finite, (x - X[0]) / 100, 0)).astype("int64") + pad
    row = np.rint(np.where(finite, (y - Y[0]) / 100, 0)).astype("int64") + pad
    inside = (
        finite
        & (column >= 0)
        & (column < NX + 2 * pad)
        & (row >= 0)
        & (row < NY + 2 * pad)
    )
    return np.where(inside, row * (NX + 2 * pad) + column, -1)


def xy(ids, pad=0):
    """Return projected cell-centre coordinates, including optional context padding."""
    row, column = np.divmod(ids, NX + 2 * pad)
    return X[0] + 100 * (column - pad), Y[0] + 100 * (row - pad)


def reference(x, y):
    """Interpolate the frozen REMA reference at native points."""
    return map_coordinates(
        REMA,
        [(y - Y[0]) / 100, (x - X[0]) / 100],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def sparse_median(c, z):
    """Reduce finite points to a median and count per occupied cell."""
    finite = (c >= 0) & np.isfinite(z)
    c = c[finite]
    z = z[finite]
    order = np.lexsort((z, c))
    c = c[order]
    z = z[order]
    ids, start, counts = np.unique(c, return_index=True, return_counts=True)
    return ids, (z[start + (counts - 1) // 2] + z[start + counts // 2]) / 2, counts


def stage_source(item, folder, rawfolder):
    """Read and screen one pinned native source into a reusable repair archive."""
    dest = folder / (item["file"] + ".npz")
    meta = dest.with_suffix(".json")
    if dest.exists() and meta.exists():
        report = json.loads(meta.read_text())
        assert report["version"] == VERSION
        return report
    source_path, disposable = download(item, rawfolder)
    with h5py.File(source_path) as source:
        cloud = source["pixel_cloud"]
        longitude = cloud["longitude"][:]
        latitude = cloud["latitude"][:]
        x, y = TR.transform(longitude, latitude)
        context_cells = cell(x, y, 30)
        keep = (context_cells >= 0) & CONTEXT[
            np.clip(context_cells, 0, len(CONTEXT) - 1)
        ]
        source_points = len(longitude)
        points = {}
        for k in [
            "height",
            "longitude",
            "latitude",
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
        ]:
            a = cloud[k][:][keep]
            if a.dtype.kind == "f":
                fill = cloud[k].attrs.get("_FillValue")
                if fill is not None:
                    a[a == np.asarray(fill).ravel()[0]] = np.nan
                a[abs(a) > 1e20] = np.nan
            points[k] = a
        x, y, context_cells = x[keep], y[keep], context_cells[keep]
        meanings = cloud["geolocation_qual"].attrs["flag_meanings"]
        meanings = meanings.decode() if isinstance(meanings, bytes) else str(meanings)
        reject = 0
        for name, bit in zip(
            meanings.split(), cloud["geolocation_qual"].attrs["flag_masks"]
        ):
            if (
                name.endswith("_missing")
                or name.endswith("_bad")
                or name
                in [
                    "geolocation_is_from_refloc",
                    "layover_significant",
                    "phase_noise_suspect",
                    "medium_phase_suspect",
                ]
            ):
                reject |= int(bit)
        stamp = source.attrs["time_granule_start"]
        stamp = stamp.decode() if isinstance(stamp, bytes) else str(stamp)
        when = np.datetime64(stamp.rstrip("Z"), "ns")
    del longitude, latitude, keep
    models = {"model_cell": context_cells.astype("i4")}
    for k in ["geoid", "solid_earth_tide", "pole_tide", "load_tide_fes"]:
        models["model_" + k] = points[k]
    good = (
        np.isfinite(points["height"] + points["longitude"] + points["latitude"])
        & np.isin(points["classification"], [3, 4])
        & ((points["geolocation_qual"].astype("u4") & reject) == 0)
    )
    points = {
        k: v[good]
        for k, v in points.items()
        if k not in ["geoid", "solid_earth_tide", "pole_tide", "load_tide_fes"]
    }
    x, y = x[good], y[good]
    cell_ids = cell(x, y)
    study = (cell_ids >= 0) & GEO.ravel()[np.clip(cell_ids, 0, N - 1)]
    finite = np.isfinite(
        points["dheight_dphase"]
        + points["dlongitude_dphase"]
        + points["dlatitude_dphase"]
    )
    assessed = np.flatnonzero(study & finite)
    train = (
        np.floor((x + 2330000) / 1000).astype(int)
        + np.floor((y - 1195000) / 1000).astype(int)
    ) % 2 == 0
    scores = []
    for k in [-2, -1, 0, 1, 2]:
        candidate_x, candidate_y = TR.transform(
            points["longitude"][assessed]
            + 2 * np.pi * k * points["dlongitude_dphase"][assessed],
            points["latitude"][assessed]
            + 2 * np.pi * k * points["dlatitude_dphase"][assessed],
        )
        scores.append(
            points["height"][assessed]
            + 2 * np.pi * k * points["dheight_dphase"][assessed]
            - reference(candidate_x, candidate_y)
        )
    scores = np.asarray(scores)
    paired = np.all(np.isfinite(scores), axis=0)
    cycles = np.zeros(len(x), "i1")
    regions = []
    for region in np.unique(points["phase_unwrapping_region"][assessed]):
        if region < 0 or region > 1e9:
            continue
        use = (points["phase_unwrapping_region"][assessed] == region) & paired
        training = use & train[assessed]
        held_out = use & ~train[assessed]
        if training.sum() < 100 or held_out.sum() < 100:
            regions.append(
                dict(
                    region=int(region),
                    n_train=int(training.sum()),
                    n_test=int(held_out.sum()),
                    accepted=False,
                    reason="insufficient_matched_support",
                )
            )
            continue
        costs = np.median(abs(scores[:, training]), axis=1)
        test = np.median(abs(scores[:, held_out]), axis=1)
        choice = int(np.argmin(costs))
        k = choice - 2
        accepted = (
            abs(k) == 1 and costs[2] > 2 * costs[choice] and test[2] > 2 * test[choice]
        )
        if accepted:
            cycles[(points["phase_unwrapping_region"] == region) & finite] = k
        regions.append(
            dict(
                region=int(region),
                n_train=int(training.sum()),
                n_test=int(held_out.sum()),
                candidate=k,
                accepted=bool(accepted),
                training_medabs=costs.tolist(),
                heldout_medabs=test.tolist(),
            )
        )
    changed = np.flatnonzero(cycles)
    corrected_height, corrected_longitude, corrected_latitude = phase_adjust(
        points["height"][changed],
        points["longitude"][changed],
        points["latitude"][changed],
        points["dheight_dphase"][changed],
        points["dlongitude_dphase"][changed],
        points["dlatitude_dphase"][changed],
        cycles[changed],
    )
    corrected_x, corrected_y = TR.transform(corrected_longitude, corrected_latitude)
    affected = np.unique(np.r_[cell_ids[changed], cell(corrected_x, corrected_y)])
    affected = affected[(affected >= 0) & GEO.ravel()[np.clip(affected, 0, N - 1)]]
    report = dict(
        version=VERSION,
        file=item["file"],
        source_points=source_points,
        quality_points=len(x),
        shelf_quality_points=int(study.sum()),
        phase_assessable_points=int((study & finite).sum()),
        changed_points=len(changed),
        affected_cells=len(affected),
        regions=regions,
        reject_bits=reject,
    )
    payload = dict(
        h=points["height"],
        lon=points["longitude"],
        lat=points["latitude"],
        changed_index=changed.astype("i4"),
        cycles=cycles[changed],
        hc=corrected_height,
        lonc=corrected_longitude,
        latc=corrected_latitude,
        affected_cell=affected.astype("i4"),
        time_ns=when.astype("i8"),
        **models,
    )
    with dest.with_suffix(".partial").open("wb") as source:
        np.savez_compressed(source, **payload)
    dest.with_suffix(".partial").replace(dest)
    meta.write_text(json.dumps(report, indent=2))
    if disposable:
        source_path.unlink()
    print(
        "SOURCE_ASSESSED",
        item["file"],
        report["shelf_quality_points"],
        len(changed),
        flush=True,
    )
    return report


def process(row):
    """Choose region-wide phase integers, move native points and regrid one overpass."""
    global PROC
    acquisition_id = row["overpass"]
    folder = R / "shelf_native_work" / acquisition_id
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="lcis_shelf_", dir=os.environ.get("TMPDIR", "/tmp")
    ) as temporary_dir:
        reports = [
            stage_source(item, folder, Path(temporary_dir)) for item in row["sources"]
        ]
    target = np.zeros(N, bool)
    affected = np.zeros(N, bool)
    north = (
        (Y[:, None] >= 1195000)
        & (Y[:, None] <= 1235000)
        & (X[None, :] >= -2330000)
        & (X[None, :] <= -2270000)
        & GEO
    ).ravel()
    target |= north
    for item in row["sources"]:
        with np.load(folder / (item["file"] + ".npz")) as archive:
            affected[archive["affected_cell"]] = True
    target |= affected
    model_near = maximum_filter(np.pad(target.reshape(NY, NX), 30), size=41).ravel()
    collected = {
        key: []
        for key in [
            "h",
            "lon",
            "lat",
            "hc",
            "lonc",
            "latc",
            "hdac",
            "hcdac",
            "mean_height",
            "time",
        ]
    }
    models = {
        key: []
        for key in ["cell", "geoid", "solid_earth_tide", "pole_tide", "load_tide_fes"]
    }
    for item in row["sources"]:
        with np.load(folder / (item["file"] + ".npz")) as archive:
            context_cells = archive["model_cell"]
            use = model_near[context_cells]
            models["cell"].append(context_cells[use])
            for key in models:
                if key != "cell":
                    models[key].append(archive["model_" + key][use])
            height = archive["h"].astype(float)
            longitude = archive["lon"]
            latitude = archive["lat"]
            corrected_height = height.copy()
            corrected_longitude = longitude.copy()
            corrected_latitude = latitude.copy()
            changed = archive["changed_index"]
            corrected_height[changed] = archive["hc"]
            corrected_longitude[changed] = archive["lonc"]
            corrected_latitude[changed] = archive["latc"]
            x, y = TR.transform(longitude, latitude)
            corrected_x, corrected_y = TR.transform(
                corrected_longitude, corrected_latitude
            )
            before = cell(x, y)
            after = cell(corrected_x, corrected_y)
            use = ((before >= 0) & target[np.clip(before, 0, N - 1)]) | (
                (after >= 0) & target[np.clip(after, 0, N - 1)]
            )
            if not use.any():
                continue
            (
                height,
                longitude,
                latitude,
                corrected_height,
                corrected_longitude,
                corrected_latitude,
                x,
                y,
                corrected_x,
                corrected_y,
            ) = [
                values_at_cells[use]
                for values_at_cells in [
                    height,
                    longitude,
                    latitude,
                    corrected_height,
                    corrected_longitude,
                    corrected_latitude,
                    x,
                    y,
                    corrected_x,
                    corrected_y,
                ]
            ]
            when = archive["time_ns"].astype("datetime64[ns]")
            dac0 = PROVIDER.sample(
                tile_midpoint(item["file"]),
                longitude,
                latitude,
                source_path=item["file"],
            )
            dac1 = PROVIDER.sample(
                tile_midpoint(item["file"]),
                corrected_longitude,
                corrected_latitude,
                source_path=item["file"],
            )
            if PROC is None:
                PROC = Processor()
            tide = PROC.tide(corrected_x, corrected_y, when)
            for key, value in dict(
                h=height,
                lon=longitude,
                lat=latitude,
                hc=corrected_height,
                lonc=corrected_longitude,
                latc=corrected_latitude,
                hdac=height - dac0,
                hcdac=corrected_height - dac1,
                mean_height=corrected_height - dac1 - tide,
                time=np.full(
                    len(height), float((when - EPOCH) / np.timedelta64(1, "D"))
                ),
            ).items():
                collected[key].append(value)
    points = {
        key: np.concatenate(values_at_cells) if values_at_cells else np.array([])
        for key, values_at_cells in collected.items()
    }
    if len(points["h"]):
        order = np.lexsort((points["h"], points["lat"], points["lon"]))
        unique = np.r_[
            True,
            (np.diff(points["lon"][order]) != 0)
            | (np.diff(points["lat"][order]) != 0)
            | (np.diff(points["h"][order]) != 0),
        ]
        order = order[unique]
        points = {key: value[order] for key, value in points.items()}
        x, y = TR.transform(points["lon"], points["lat"])
        corrected_x, corrected_y = TR.transform(points["lonc"], points["latc"])
        before = cell(x, y)
        after = cell(corrected_x, corrected_y)
        ids = np.unique(np.r_[before, after, np.flatnonzero(affected)])
        ids = ids[
            (ids >= 0)
            & target[np.clip(ids, 0, N - 1)]
            & GEO.ravel()[np.clip(ids, 0, N - 1)]
        ]
    else:
        ids = np.flatnonzero(affected)
        before = after = np.array([], dtype=int)
    values = {}
    for key, source, c in [
        ("raw", "h", before),
        ("corrected", "hc", after),
        ("raw_minus_dac", "hdac", before),
        ("corrected_minus_dac", "hcdac", after),
        ("mean_height", "mean_height", after),
        ("time", "time", after),
    ]:
        occupied_cells, medians, counts = sparse_median(c, points[source])
        values_at_cells = np.full(len(ids), np.nan)
        changed = np.searchsorted(occupied_cells, ids)
        matched = changed < len(occupied_cells)
        if len(occupied_cells):
            matched &= (
                occupied_cells[np.clip(changed, 0, len(occupied_cells) - 1)] == ids
            )
        values_at_cells[matched] = medians[changed[matched]]
        values[key] = values_at_cells
    geoid = np.full(len(ids), np.nan)
    if len(ids):
        context_cells = np.concatenate(models["cell"])
        fields = []
        model_ids = None
        for key in ["geoid", "solid_earth_tide", "pole_tide", "load_tide_fes"]:
            occupied_cells, medians, _ = sparse_median(
                context_cells, np.concatenate(models[key])
            )
            fields.append((occupied_cells, medians))
            model_ids = (
                occupied_cells
                if model_ids is None
                else np.intersect1d(model_ids, occupied_cells)
            )
        if len(model_ids):
            model_fields = np.array(
                [
                    medians[np.searchsorted(occupied_cells, model_ids)]
                    for occupied_cells, medians in fields
                ]
            )
            model_x, model_y = xy(model_ids, 30)
            target_x, target_y = xy(ids)
            neighbour_count = min(4, len(model_ids))
            distance, neighbours = cKDTree(np.column_stack((model_x, model_y))).query(
                np.column_stack((target_x, target_y)), k=neighbour_count, workers=1
            )
            if neighbour_count == 1:
                distance, neighbours = distance[:, None], neighbours[:, None]
            weights = np.where(distance <= 2000, 1 / np.maximum(distance, 1) ** 2, 0)
            total_weight = weights.sum(1)
            interpolated_fields = np.array(
                [
                    np.divide(
                        np.sum(weights * values_at_cells[neighbours], axis=1),
                        total_weight,
                        out=np.full(len(ids), np.nan),
                        where=total_weight > 0,
                    )
                    for values_at_cells in model_fields
                ]
            )
            geoid = interpolated_fields[0]
            for key in [
                "raw",
                "corrected",
                "raw_minus_dac",
                "corrected_minus_dac",
                "mean_height",
            ]:
                values[key] -= interpolated_fields.sum(0)
        else:
            for key in [
                "raw",
                "corrected",
                "raw_minus_dac",
                "corrected_minus_dac",
                "mean_height",
            ]:
                values[key][:] = np.nan
    dest = OUT / (acquisition_id + ".npz")
    with dest.with_suffix(".partial").open("wb") as archive:
        np.savez_compressed(
            archive,
            cell=ids.astype("i4"),
            geoid=geoid.astype("f4"),
            affected=affected[ids],
            north=north[ids],
            **{
                key: values_at_cells.astype("f4" if key != "time" else "f8")
                for key, values_at_cells in values.items()
            },
        )
    dest.with_suffix(".partial").replace(dest)
    report = dict(
        overpass=acquisition_id,
        status="assessed",
        version=VERSION,
        scope="Entire canonical LCIS shelf",
        source_count=len(reports),
        sources=reports,
        changed_points=sum(
            values_at_cells["changed_points"] for values_at_cells in reports
        ),
        affected_cells=int(affected.sum()),
        native_context_cells=len(ids),
        native_mean_cells=int(np.isfinite(values["mean_height"]).sum()),
        policy="Assess all native sources. Preserve Raster outside accepted phase-shift union; retain established all-native northern mean patch. No northern restriction on assessment or corrections.",
    )
    dest.with_suffix(".json").write_text(json.dumps(report, indent=2))
    shutil.rmtree(folder)
    print("SHELF_PASS", acquisition_id, report["changed_points"], len(ids), flush=True)

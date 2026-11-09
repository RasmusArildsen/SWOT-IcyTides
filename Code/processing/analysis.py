"""Read prepared observations and fit four tides, TIME and fine-cell elevations.

heights and weights: [date, coarse cell, fine position]
design:              [coarse cell, date, 9]
beta:                [9, coarse cell]
terrain:             [coarse cell, 9]

Coarse cells are 250 m; each has up to nine 100 m positions."""

from pathlib import Path
import os
import json

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
import astronomy
import huber
from filtering import checked_mask
from layout import (
    DEFAULT_ACQUISITIONS,
    band_indices,
    fit_part_indices,
    fit_parts_per_band,
)

ROOT = Path(os.environ.get("S4IS_INDEPENDENT_ROOT", "work/independent191")).resolve()
OLD = ROOT
BASE = Path(__file__).resolve().parents[1]
SETTINGS = json.loads((BASE / "aux_files/analysis.json").read_text())
NT = int(os.environ.get("S4IS_ACQUISITIONS", DEFAULT_ACQUISITIONS))


def dump(path, value):
    """Write a small JSON receipt atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def ds(f, key, value):
    """Write an HDF5 array with the standard lossless compression."""
    array = np.asarray(value)
    return (
        f.create_dataset(
            key, data=array, compression="gzip", compression_opts=2, shuffle=True
        )
        if array.ndim and array.size
        else f.create_dataset(key, data=array)
    )


def geometry():
    """Read the shared grid and the locations of completed independent fits."""
    with h5py.File(OLD / "geometry.h5") as handle:
        assert handle.attrs["complete"] == 1
        geometry_info = {
            name: (
                handle[name][:NT]
                if name in ["reference_seconds", "time_ns", "passes"]
                else handle[name][:]
            )
            for name in [
                "cell_id",
                "lat",
                "lon",
                "global_x",
                "global_y",
                "source_band",
                "source_local",
                "protected",
                "reference_seconds",
                "time_ns",
                "passes",
            ]
        }
    grid_width = len(geometry_info["global_x"])
    grid_height = len(geometry_info["global_y"])
    geometry_info["nx"] = grid_width
    geometry_info["xy"] = np.c_[
        geometry_info["global_x"][geometry_info["cell_id"] % grid_width],
        geometry_info["global_y"][geometry_info["cell_id"] // grid_width],
    ]
    geometry_info["free_grid"] = np.zeros(grid_width * grid_height, bool)
    geometry_info["free_grid"][
        geometry_info["cell_id"][~geometry_info["protected"]]
    ] = True
    geometry_info["margin"] = distance_transform_edt(
        geometry_info["free_grid"].reshape(grid_height, grid_width), sampling=250
    ).ravel()[geometry_info["cell_id"]]
    if (OLD / "independent.h5").exists():
        with h5py.File(OLD / "independent.h5") as handle:
            cell_count = len(geometry_info["cell_id"])
            geometry_info["valid"] = np.empty(cell_count, bool)
            for start in range(0, cell_count, 16384):
                geometry_info["valid"][start : start + 16384] = np.isfinite(
                    handle["beta"][:, start : start + 16384]
                ).all(0)
            geometry_info["part_id"] = handle["part_id"][:]
            geometry_info["part_local"] = handle["part_local"][:]
    return geometry_info


def basis(seconds, lat, ref):
    """Build eight tide columns and one TIME column at the observation clocks."""
    seconds = np.where(np.isfinite(seconds), seconds, np.asarray(ref)[None])
    unique_seconds, time_lookup = np.unique(
        np.rint(seconds).astype("i8"), return_inverse=True
    )
    dates = np.datetime64("2000-01-01", "ns") + unique_seconds.astype("timedelta64[s]")
    astronomical_values = astronomy.astronomical_bases(dates)
    latitude_weights = astronomy.latitude_weights(lat)
    tidal_design = np.einsum(
        "nk,kntp->ntp",
        latitude_weights,
        astronomical_values[:, time_lookup.reshape(seconds.shape), :],
    )
    # November191 deliberately retains the ORIGINAL190 TIME normalization.
    with h5py.File(ROOT / "geometry.h5") as handle:
        centre = float(handle.attrs["time_center_seconds_2000"])
        span = float(handle.attrs["time_span_seconds"])
    return np.concatenate(
        [tidal_design, ((seconds - centre) / span)[:, :, None]], axis=2
    )


def admitted(nodes, g):
    """Read fine heights and apply the frozen REMA admission gate.

    Returns heights[date, cell, fine], design[cell, date, 9] and an admission mask.
    The reference selects observations; it is never subtracted from heights.
    """
    cell_count = len(nodes)
    heights = np.full((NT, cell_count, 9), np.nan, "f4")
    reference = np.full((cell_count, 9), np.nan)
    seconds = np.full((NT, cell_count), np.nan)
    for band in np.unique(g["source_band"][nodes]):
        destination = np.flatnonzero(g["source_band"][nodes] == band)
        local_index = g["source_local"][nodes[destination]]
        assert np.all(np.diff(local_index) > 0)
        with h5py.File(ROOT / "inputs" / f"{int(band):03d}.h5") as handle:
            np.testing.assert_array_equal(
                handle["cell_id"][local_index], g["cell_id"][nodes[destination]]
            )
            heights[:, destination] = handle["height"][:NT, local_index]
            reference[destination] = handle["reference_geoid"][local_index]
            seconds[:, destination] = handle["time_seconds"][:NT, local_index]
    screening = SETTINGS["screening"]
    mask = np.isfinite(heights) & (
        ~np.isfinite(reference)[None]
        | (
            abs(heights - reference[None] - screening["centre_m"])
            <= screening["multiplier"] * screening["NMAD_m"]
        )
    )
    mask = checked_mask(ROOT, nodes, g, mask)
    return heights, basis(seconds.T, g["lat"][nodes], g["reference_seconds"]), mask


def load(nodes, g, weights=False):
    """Read heights, design and independent fits; optionally read final robust weights."""
    nodes = np.asarray(nodes, int)
    heights, design, mask = admitted(nodes, g)
    with h5py.File(OLD / "independent.h5") as handle:
        fit = {
            name: (handle[name][:, nodes] if name == "beta" else handle[name][nodes])
            for name in ["beta", "terrain", "scale"]
        }
    if weights:
        normal_weight = np.zeros(heights.shape)
        for part in np.unique(g["part_id"][nodes]):
            destination = np.flatnonzero(g["part_id"][nodes] == part)
            local_index = g["part_local"][nodes[destination]]
            assert np.all(np.diff(local_index) > 0)
            with h5py.File(OLD / "fits" / f"{int(part):03d}.h5") as handle:
                for start in range(0, len(destination), 512):
                    normal_weight[:, destination[start : start + 512]] = handle[
                        "normal_weight"
                    ][:NT, local_index[start : start + 512]]
        fit["normal_weight"] = normal_weight
    return heights.astype(float), design, mask, fit


def fit_part(index):
    """Fit one cell group from the geometry layout, preserving observation order."""
    geometry_path = ROOT / "geometry.h5"
    parts = fit_part_indices(geometry_path)
    if index not in parts:
        raise ValueError(f"Part index must be in {parts}")
    target = ROOT / "fits" / f"{index:03d}.h5"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with h5py.File(target) as handle:
            assert handle.attrs["complete"] == 1
        return
    geometry_info = geometry()
    parts_per_band = fit_parts_per_band(geometry_path)
    band_offset, quarter = divmod(index, parts_per_band)
    band = band_indices(geometry_path)[band_offset]
    band_cells = np.flatnonzero(geometry_info["source_band"] == band)
    nodes = band_cells[
        len(band_cells)
        * quarter
        // parts_per_band : len(band_cells)
        * (quarter + 1)
        // parts_per_band
    ]
    cell_count = len(nodes)
    shapes = dict(
        beta=(9, cell_count),
        terrain=(cell_count, 9),
        count=(cell_count,),
        effective_n=(cell_count,),
        condition=(cell_count,),
        scale=(cell_count,),
        whitened_scale=(cell_count,),
        iterations=(cell_count,),
        fine_count=(cell_count,),
    )
    temporary = target.with_suffix(".tmp.h5")
    with h5py.File(temporary, "w") as handle:
        ds(handle, "global_index", nodes)
        for name, shape in shapes.items():
            handle.create_dataset(
                name, shape, dtype="f8", compression="gzip", compression_opts=2
            )
        handle.create_dataset(
            "normal_weight",
            (NT, cell_count, 9),
            dtype="f8",
            chunks=(1, min(128, cell_count), 9),
            compression="gzip",
            compression_opts=2,
            shuffle=True,
        )
        handle.create_dataset("status", (cell_count,), dtype="S48", compression="gzip")
        for start in range(0, cell_count, 128):
            stop = min(start + 128, cell_count)
            heights, design, mask = admitted(nodes[start:stop], geometry_info)
            fit = huber.fit(heights, design, mask.astype(float))
            for name in shapes:
                if name == "beta":
                    handle[name][:, start:stop] = fit[name]
                else:
                    handle[name][start:stop] = fit[name]
            normal_weight = fit["normal_weight"]
            normal_weight[:, ~np.isfinite(fit["beta"]).all(0)] = 0
            handle["normal_weight"][:, start:stop] = normal_weight
            handle["status"][start:stop] = fit["status"].astype("S48")
        handle.attrs.update(
            complete=1, estimator_json=json.dumps(huber.estimator_metadata())
        )
    temporary.replace(target)


def collect():
    """Combine all fit parts defined by the geometry into independent.h5."""
    target = ROOT / "independent.h5"
    if target.exists():
        with h5py.File(target) as output:
            assert output.attrs["complete"] == 1
        return
    cell_count = len(geometry()["cell_id"])
    temporary = target.with_suffix(".tmp.h5")
    with h5py.File(temporary, "w") as output:
        seen = np.zeros(cell_count, bool)
        part_id = np.full(cell_count, -1, "i2")
        local = np.full(cell_count, -1, "i4")
        for part in fit_part_indices(ROOT / "geometry.h5"):
            with h5py.File(ROOT / "fits" / f"{part:03d}.h5") as source:
                assert source.attrs["complete"] == 1
                positions = source["global_index"][:]
                assert not seen[positions].any()
                seen[positions] = True
                part_id[positions] = part
                local[positions] = np.arange(len(positions))
                for name in [
                    "beta",
                    "terrain",
                    "count",
                    "effective_n",
                    "condition",
                    "scale",
                    "whitened_scale",
                    "iterations",
                    "fine_count",
                ]:
                    shape = (
                        (9, cell_count)
                        if name == "beta"
                        else (cell_count, 9) if name == "terrain" else (cell_count,)
                    )
                    if name not in output:
                        output.create_dataset(
                            name,
                            shape,
                            dtype="f8",
                            compression="gzip",
                            compression_opts=2,
                        )
                    if name == "beta":
                        output[name][:, positions] = source[name][:]
                    else:
                        output[name][positions] = source[name][:]
        assert seen.all()
        ds(output, "part_id", part_id)
        ds(output, "part_local", local)
        output.attrs["complete"] = 1
    temporary.replace(target)


def freeze_inputs(path, files):
    """Freeze completed inputs/code by small hashes and file metadata."""
    import hashlib

    code_dir = Path(__file__).resolve().parent
    aux_dir = code_dir.parent / "aux_files"
    watched = [
        *code_dir.glob("*.py"),
        *aux_dir.glob("*.json"),
        aux_dir / "figure3_sampling.h5",
    ]
    watched = [p for p in watched if p.name != "resources.json"]
    hashes = {
        str(p.relative_to(code_dir.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in watched
    }
    records = []
    pending = {Path(p).resolve() for p in files}
    seen = set()
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        stat = item.stat()
        records.append(
            dict(path=str(item), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        )
        if item.suffix == ".h5":
            with h5py.File(item) as handle:
                for key in [
                    "height",
                    "time_seconds",
                    "sigma",
                    "beta",
                    "terrain",
                    "fine_id",
                    "cell_id",
                ]:
                    link = handle.get(key, getlink=True)
                    if isinstance(link, h5py.ExternalLink):
                        pending.add((item.parent / link.filename).resolve())
                        continue
                    if key not in handle:
                        continue
                    dataset = handle[key]
                    if isinstance(dataset, h5py.Dataset) and dataset.is_virtual:
                        for source in dataset.virtual_sources():
                            name = (
                                source.file_name.decode()
                                if isinstance(source.file_name, bytes)
                                else source.file_name
                            )
                            if name != ".":
                                pending.add((item.parent / name).resolve())
        pending -= seen
    records.sort(key=lambda r: r["path"])
    snapshot = dict(
        source_snapshots=records, source_code_sha256=hashes, acquisitions=NT
    )
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != snapshot:
            raise RuntimeError(
                "Frozen source inputs or code changed; use a new run directory."
            )
    else:
        import uuid

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if json.loads(path.read_text()) != snapshot:
                    raise RuntimeError("Concurrent source freeze differs.")
        finally:
            temporary.unlink(missing_ok=True)

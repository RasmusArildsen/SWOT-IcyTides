"""Rebuild the earlier observation histories used by residual phase correction.

This preparation retains the screening and CATS conventions of that reference."""

from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import argparse
import json
import re
import sys
import h5py
import numpy as np
import pyTMD
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer
from rasterio.warp import transform_bounds
from aviso_dac import tile_midpoint, AVISODAC
from analysis import ds
from layout import band_indices

T0 = datetime(1992, 1, 1)
CONSTITUENTS = ("m2", "s2", "k1", "o1")
LL = Transformer.from_crs(3031, 4326, always_xy=True)
ARGS = None


def configure(settings):
    """Load the recorded settings for rebuilding the phase-selection reference."""
    global ARGS
    ARGS = SimpleNamespace(**settings)


def configuration():
    if ARGS is None:
        raise RuntimeError("Call configure(settings) before rebuilding the reference")
    return ARGS


def parse_time_from_name(path: Path) -> datetime:
    match = re.search(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_", path.name)
    if not match:
        raise ValueError(f"Could not parse SWOT timestamps from {path.name}")
    start = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    end = datetime.strptime(match.group(2), "%Y%m%dT%H%M%S")
    return start + (end - start) / 2


def robust_median_mad(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    median = float(np.nanmedian(values))
    median_deviation = float(np.nanmedian(np.abs(values - median)))
    return median, median_deviation


def rema_pass_filter(
    wse: np.ndarray,
    tide: np.ndarray,
    geoid: np.ndarray,
    rema_dem: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    rema_geoid = rema_dem[iy, ix].astype(np.float64)
    if not getattr(args, "reference_dem_is_geoid", False):
        rema_geoid = rema_geoid - geoid
    rema_resid = (wse - tide) - rema_geoid
    finite = np.isfinite(rema_resid)
    diagnostic = rema_resid[finite]
    if diagnostic.size > args.pass_diagnostic_subsample:
        sample_indices = np.random.default_rng(20260814).choice(
            diagnostic.size, size=args.pass_diagnostic_subsample, replace=False
        )
        diagnostic = diagnostic[sample_indices]
    median, median_deviation = robust_median_mad(diagnostic)
    sigma = (
        1.4826 * median_deviation
        if np.isfinite(median_deviation) and median_deviation > 0
        else np.nan
    )
    relative_clip = args.residual_mad_sigma * sigma if np.isfinite(sigma) else np.nan
    if not np.isfinite(relative_clip):
        relative_clip = args.rema_prefilter_max - args.rema_prefilter_min
    relative_clip = max(args.residual_min_clip, min(6.0, relative_clip))

    pass_ok = (
        diagnostic.size >= args.min_pass_samples
        and np.isfinite(median)
        and np.isfinite(median_deviation)
        and args.pass_rema_min <= median <= args.pass_rema_max
        and median_deviation <= args.pass_rema_mad_max
    )
    keep = (
        finite
        & (rema_resid >= args.rema_prefilter_min)
        & (rema_resid <= args.rema_prefilter_max)
        & (np.abs(rema_resid - median) <= relative_clip)
    )
    info = {
        "pass_rema_samples": int(diagnostic.size),
        "pass_rema_median_m": median,
        "pass_rema_mad_m": median_deviation,
        "pass_rema_relative_clip_m": float(relative_clip),
        "pass_rema_status": "accepted" if pass_ok else "rejected",
        "pass_rema_local_keep_samples": int(keep.sum()),
    }
    if not pass_ok:
        keep[:] = False
    return keep, info


def cats_tide_at_cells(
    re_grid, im_grid, omega, phase, iy, ix, obs_time: datetime
) -> np.ndarray:
    seconds = (obs_time - T0).total_seconds()
    mjd = 48622.0 + seconds / 86400.0
    nodal_phase, nodal_factor, _ = pyTMD.constituents.arguments(
        np.atleast_1d(mjd), list(CONSTITUENTS), corrections="TMD3"
    )
    theta = omega * seconds + phase + nodal_phase[0]
    cos_t = np.cos(theta).astype(np.float32)
    sin_t = np.sin(theta).astype(np.float32)
    tide = np.zeros(ix.shape, dtype=np.float32)
    for constituent_index in range(len(CONSTITUENTS)):
        real_part = re_grid[constituent_index, iy, ix]
        imaginary_part = im_grid[constituent_index, iy, ix]
        # OTIS/TMD3 stores the complex coefficient as hRe - 1j*hIm.
        tide += nodal_factor[0, constituent_index] * (
            real_part * cos_t[constituent_index]
            + imaginary_part * sin_t[constituent_index]
        )
    return tide


def cats_at_points(path, x, y):
    """Exact frozen complex interpolation/float32/flexure operations, cropped IO."""
    with xr.open_dataset(path, engine="netcdf4") as source:
        order = source.constituents.attrs.get(
            "constituent_order", "m2 s2 n2 k2 k1 o1 p1 q1 mf mm"
        ).split()
        indices = [order.index(k) for k in ["m2", "s2", "k1", "o1"]]
        projected_x, projected_y = Transformer.from_crs(
            3031, source.mapping.attrs["spatial_proj4"], always_xy=True
        ).transform(x, y)
        model_x, model_y = source.x.values, source.y.values
        columns = np.flatnonzero(
            (model_x >= np.min(projected_x) - 3 * abs(model_x[1] - model_x[0]))
            & (model_x <= np.max(projected_x) + 3 * abs(model_x[1] - model_x[0]))
        )
        rows = np.flatnonzero(
            (model_y >= np.min(projected_y) - 3 * abs(model_y[1] - model_y[0]))
            & (model_y <= np.max(projected_y) + 3 * abs(model_y[1] - model_y[0]))
        )
        section = source.isel(
            x=slice(columns[0], columns[-1] + 1),
            y=slice(rows[0], rows[-1] + 1),
            constituents=indices,
        )
        real_part = section.hRe.values.astype(float)
        imaginary_part = section.hIm.values.astype(float)
        flex = section.flexure.values.astype(float)
        model_x, model_y = model_x[columns], model_y[rows]
        omega = section.omega.values.astype(float)
        phase = section.phase.values.astype(float)
    if model_x[0] > model_x[-1]:
        model_x = model_x[::-1]
        real_part = real_part[:, :, ::-1]
        imaginary_part = imaginary_part[:, :, ::-1]
        flex = flex[:, ::-1]
    if model_y[0] > model_y[-1]:
        model_y = model_y[::-1]
        real_part = real_part[:, ::-1, :]
        imaginary_part = imaginary_part[:, ::-1, :]
        flex = flex[::-1, :]
    points = np.column_stack([projected_y, projected_x])
    factor = np.clip(
        RegularGridInterpolator(
            (model_y, model_x),
            flex,
            method="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )(points)
        / 100,
        0,
        1,
    ).astype("f4")
    result = []
    for values in [real_part, imaginary_part]:
        result.append(
            np.array(
                [
                    RegularGridInterpolator(
                        (model_y, model_x), v, bounds_error=False, fill_value=np.nan
                    )(points).astype("f4")
                    * factor
                    for v in values
                ]
            )
        )
    return *result, omega, phase


def point_field(ds, ix, iy, inside):
    """Read target-centre values from bounded Raster row blocks."""
    values = np.full(len(ix), np.nan)
    for block in np.unique(iy[inside] // 128):
        selected = inside & (iy // 128 == block)
        start = int(block) * 128
        stop = min(start + 128, ds.shape[0])
        left = int(ix[selected].min())
        right = int(ix[selected].max()) + 1
        chunk = ds[start:stop, left:right]
        sampled = chunk[iy[selected] - start, ix[selected] - left].astype(float)
        fill = ds.attrs.get("_FillValue")
        sampled[
            (abs(sampled) > 1e20) | (sampled == fill if fill is not None else False)
        ] = np.nan
        values[selected] = sampled
    return values


def tile_input(path, acq, geo, dac, archive):
    """Inverse-nearest UTM sampling at exact target centres, bounded Raster reads."""
    with h5py.File(path) as handle:
        source_x, source_y = handle["x"][:], handle["y"][:]
        dx, dy = float(source_x[1] - source_x[0]), float(source_y[1] - source_y[0])
        wkt = handle["crs"].attrs.get("crs_wkt", handle["crs"].attrs.get("spatial_ref"))
        wkt = wkt.decode() if isinstance(wkt, bytes) else wkt
        bounds = transform_bounds(
            wkt,
            "EPSG:3031",
            source_x[0] - dx / 2,
            source_y[0] - dy / 2,
            source_x[-1] + dx / 2,
            source_y[-1] + dy / 2,
            densify_pts=41,
        )
        fine_y = geo["fine_y"]
        start = int(np.searchsorted(fine_y, bounds[1] - 100))
        stop = int(np.searchsorted(fine_y, bounds[3] + 100, side="right"))
        if start == stop:
            return None
        xx = geo["fine_x"][start:stop]
        yy = fine_y[start:stop]
        window = (xx >= bounds[0] - 100) & (xx <= bounds[2] + 100)
        positions = np.arange(start, stop)[window]
        x, y = xx[window], yy[window]
        raster_x, raster_y = Transformer.from_crs(3031, wkt, always_xy=True).transform(
            x, y
        )
        column = np.floor((raster_x - source_x[0]) / dx + 0.5).astype(int)
        row = np.floor((raster_y - source_y[0]) / dy + 0.5).astype(int)
        inside = (
            (column >= 0)
            & (row >= 0)
            & (column < len(source_x))
            & (row < len(source_y))
        )
        if not inside.any():
            return None
        height = point_field(handle["wse"], column, row, inside)
        geoid = point_field(handle["geoid"], column, row, inside)
        pixel_count = (
            point_field(handle["n_wse_pix"], column, row, inside)
            if "n_wse_pix" in handle
            else np.ones(len(x))
        )
        cross = point_field(handle["cross_track"], column, row, inside)
    valid = inside & np.isfinite(height + geoid) & (pixel_count > 0)
    before = int(valid.sum())
    valid &= np.isfinite(cross) & (abs(cross) >= 10000)
    args = configuration()
    valid &= (height > args.min_wse) & (height < args.max_wse)
    positions, x, y, height, geoid, cross = [
        v[valid] for v in [positions, x, y, height, geoid, cross]
    ]
    if not len(x):
        return None
    ids = geo["fine_id"][start:stop][window][valid]
    phase = native_subset(archive, ids)
    native_positions = np.searchsorted(ids, phase["cell_id"])
    native = np.isfinite(phase["after_wse_dac"] + phase["geoid"])
    bad = (np.isfinite(phase["before_wse"]) & ~np.isfinite(phase["after_wse"])) | (
        np.isfinite(phase["after_wse"])
        & ~np.isfinite(phase["after_wse_dac"] + phase["geoid"])
    )
    bad_at = native_positions[bad]
    kept = np.ones(len(x), bool)
    kept[bad_at] = False
    lon, lat = LL.transform(x, y)
    west, south, east, north = map(float, args.bbox.split(","))
    kept &= (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    replaced = np.zeros(len(x), bool)
    replaced[native_positions[native]] = True
    adjusted = np.full(len(x), np.nan)
    fallback = kept & ~replaced
    if fallback.any():
        adjusted[fallback] = height[fallback] - dac.sample(
            tile_midpoint(path), lon[fallback], lat[fallback], source_path=path
        )
    adjusted[native_positions[native]] = phase["after_wse_dac"][native]
    geoid[native_positions[native]] = phase["geoid"][native]
    kept &= np.isfinite(adjusted + geoid)
    positions, x, y, adjusted, geoid, cross, ids = [
        v[kept] for v in [positions, x, y, adjusted, geoid, cross, ids]
    ]
    # HDF5 fancy indices are ascending; reference reads are limited to this tile window.
    reference_fields = {
        name: geo[name][start:stop][window][valid][kept] for name in ["dem", "rema"]
    }
    cats_real = geo["cats_re"][:, start:stop][:, window][:, valid][:, kept]
    cats_imag = geo["cats_im"][:, start:stop][:, window][:, valid][:, kept]
    tide = cats_tide_at_cells(
        cats_real[:, None, :],
        cats_imag[:, None, :],
        geo["omega"][:],
        geo["phase"][:],
        np.zeros(len(x), int),
        np.arange(len(x)),
        parse_time_from_name(path),
    ).astype(float)
    return dict(
        x=x,
        y=y,
        h=adjusted,
        geoid=geoid,
        cross=cross,
        id=ids,
        tide=tide,
        **reference_fields,
    ), dict(
        before_range=before,
        after_input=len(x),
        native_replaced=int((replaced & kept).sum()),
        quarantined=int(bad.sum()),
        minimum_abs_cross_m=float(abs(cross).min()) if len(cross) else None,
    )


def relative_keep(residual, k):
    """Apply the original median/MAD selection without changing retained heights."""
    centre = float(np.median(residual))
    scale = 1.4826 * float(np.median(abs(residual - centre)))
    tolerance = np.finfo(float).eps * max(1.0, abs(centre)) * 16
    return abs(residual - centre) <= max(k * scale, tolerance), dict(
        centre_NOT_SUBTRACTED_m=centre, NMAD_m=scale, multiplier=k
    )


def admission(d):
    """Apply the ancestor reference screens used only to select residual-phase repairs."""
    args = configuration()
    point_count = len(d["h"])
    keep, info = rema_pass_filter(
        d["h"],
        d["tide"],
        d["geoid"],
        d["rema"][None, :],
        np.zeros(point_count, int),
        np.arange(point_count),
        args,
    )
    retained = {key: values[keep] for key, values in d.items()}
    finite = np.isfinite(retained["h"] + retained["dem"] + retained["tide"])
    retained = {key: values[finite] for key, values in retained.items()}
    residual = retained["h"] - retained["dem"] - retained["tide"]
    median_residual, mad_residual = robust_median_mad(residual)
    okay = (
        len(residual) >= args.min_pass_samples
        and np.isfinite(median_residual + mad_residual)
        and abs(median_residual) <= args.final_pass_residual_abs_max
        and mad_residual <= args.final_pass_residual_mad_max
    )
    if not okay:
        return None, dict(**info, status="early_rejected", samples=len(residual))
    keep, stats = relative_keep(residual, 3.5)
    return {key: values[keep] for key, values in retained.items()}, dict(
        **info,
        status="accepted",
        samples=len(residual),
        retained=int(keep.sum()),
        relative_screen=stats,
    )


def native_subset(path, wanted):
    # Input is the exact frozen compatible-native ledger, never a coefficient
    # field. It is regenerated by the first native stage and archive adapter.
    """Read only matching native-ledger points in small archive chunks."""
    import zipfile

    keys = [
        "cell_id",
        "before_wse_dac",
        "after_wse_dac",
        "before_wse",
        "after_wse",
        "geoid",
    ]
    pieces = {key: [] for key in keys}
    with zipfile.ZipFile(path) as archive:
        handles = {key: archive.open(key + ".npy") for key in keys}
        info = {}
        try:
            for key, stream in handles.items():
                version = np.lib.format.read_magic(stream)
                shape, order, dtype = (
                    np.lib.format.read_array_header_1_0(stream)
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0(stream)
                )
                assert not order and len(shape) == 1
                info[key] = (shape[0], dtype)
            point_count = info["cell_id"][0]
            assert all(chunk[0] == point_count for chunk in info.values())
            for start in range(0, point_count, 65536):
                size = min(65536, point_count - start)
                chunk = {
                    key: np.frombuffer(
                        stream.read(size * info[key][1].itemsize), dtype=info[key][1]
                    )
                    for key, stream in handles.items()
                }
                positions = np.searchsorted(wanted, chunk["cell_id"])
                matched = positions < len(wanted)
                matched[matched] &= (
                    wanted[positions[matched]] == chunk["cell_id"][matched]
                )
                for key in keys:
                    pieces[key].append(chunk[key][matched])
        finally:
            for stream in handles.values():
                stream.close()
    return {
        key: np.concatenate(chunk) if chunk else np.array([], dtype=info[key][1])
        for key, chunk in pieces.items()
    }


def raster_path(root, title, acquisition):
    root = Path(root)
    for candidate in (root / title, root / acquisition / title):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Pinned Raster file missing: {title}")


def build_rows(paths, out, index, settings):
    """Rebuild one frozen ancestor acquisition history from its pinned raw tiles."""
    out = Path(out)
    folder = out / "rows"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{index:03d}.h5"
    if target.exists():
        with h5py.File(target) as output:
            assert output.attrs["complete"] == 1
        return
    item = settings["acquisitions"][index]
    name = item["id"]
    parts = []
    coverage = json.loads(Path(paths["legacy_coverage"]).read_text())
    eligible = {
        retained["acquisition"]: retained["coverage_eligible"]
        for retained in coverage["acquisitions"]
    }
    dac = AVISODAC(paths["aviso_cache"], allow_download=False)
    with h5py.File(paths["legacy_reference_geometry"]) as geometry:
        grid_x, grid_y = geometry["x"][:], geometry["y"][:]
        if eligible[name]:
            for title in item["tiles"]:
                result = tile_input(
                    raster_path(paths["raster_raw"], title, name),
                    name,
                    geometry,
                    dac,
                    Path(paths["legacy_native_archives"]) / (name + ".npz"),
                )
                if result is None:
                    continue
                retained, _ = admission(result[0])
                if retained is None or not len(retained["h"]):
                    continue
                coarse_id = np.rint((retained["y"] - grid_y[0]) / 250).astype(
                    "i8"
                ) * len(grid_x) + np.rint((retained["x"] - grid_x[0]) / 250).astype(
                    "i8"
                )
                unique_coarse, inverse, count = np.unique(
                    coarse_id, return_inverse=True, return_counts=True
                )
                parts.append(
                    dict(
                        id=retained["id"],
                        cid=coarse_id,
                        h=retained["h"],
                        weight=1 / count[inverse],
                        unique=unique_coarse,
                    )
                )
    if parts:
        fine_ids = np.concatenate([retained["id"] for retained in parts])
        coarse_ids = np.concatenate([retained["cid"] for retained in parts])
        heights = np.concatenate([retained["h"] for retained in parts])
        weight = np.concatenate([retained["weight"] for retained in parts])
        unique_coarse, votes = np.unique(
            np.concatenate([retained["unique"] for retained in parts]),
            return_counts=True,
        )
        weight /= votes[np.searchsorted(unique_coarse, coarse_ids)]
        ids, inverse = np.unique(fine_ids, return_inverse=True)
        total = np.bincount(inverse, weights=weight)
        height = np.bincount(inverse, weights=weight * heights) / total
    else:
        ids = np.array([], dtype="i8")
        height = total = np.array([], dtype=float)
    with h5py.File(target, "w") as output:
        for key, retained in dict(fine_id=ids, height=height, weight=total).items():
            ds(output, key, retained)
        output.attrs["complete"] = 1


def pack_band(paths, out, index, settings):
    """Pack the reference acquisition histories for one spatial band."""
    out = Path(out)
    dest = out / "bands"
    dest.mkdir(exist_ok=True)
    target = dest / f"{index:03d}.h5"
    if target.exists():
        raise FileExistsError(target)
    with h5py.File(paths["analysis_geometry"]) as geometry_file:
        positions = np.flatnonzero(geometry_file["source_band"][:] == index)
        fine_mapping = geometry_file["fine_id"][positions]
        ids = geometry_file["cell_id"][positions]
    flat = np.flatnonzero(fine_mapping.ravel() >= 0)
    order = np.argsort(fine_mapping.ravel()[flat])
    fine = fine_mapping.ravel()[flat][order]
    destination = flat[order]
    with h5py.File(target, "w") as output:
        ds(output, "cell_id", ids)
        ds(output, "fine_id", fine_mapping)
        heights = output.create_dataset(
            "height",
            (len(settings["acquisitions"]), len(ids), fine_mapping.shape[1]),
            dtype="f8",
            chunks=(1, min(len(ids), 1024), fine_mapping.shape[1]),
            compression="gzip",
            fillvalue=np.nan,
        )
        weights = output.create_dataset(
            "weight",
            heights.shape,
            dtype="f8",
            chunks=heights.chunks,
            compression="gzip",
        )
        for date in range(len(settings["acquisitions"])):
            with h5py.File(out / "rows" / f"{date:03d}.h5") as source:
                row_ids = source["fine_id"][:]
                positions = np.searchsorted(fine, row_ids)
                keep = positions < len(fine)
                keep[keep] &= fine[positions[keep]] == row_ids[keep]
                dest = destination[positions[keep]]
                height = np.full(fine_mapping.size, np.nan)
                weight = np.zeros(fine_mapping.size)
                height[dest] = source["height"][:][keep]
                weight[dest] = source["weight"][:][keep]
                heights[date] = height.reshape(fine_mapping.shape)
                weights[date] = weight.reshape(fine_mapping.shape)
        output.attrs.update(
            complete=1,
            purpose="Frozen phase2 reference history only; these selection masks do not become November fitting masks",
        )


def main(argv=None):
    """Run ancestor geometry, archive conversion, row or band preparation."""
    global ARGS
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run import resources

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["geometry", "archives", "row", "band"])
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parts = parser.add_mutually_exclusive_group()
    parts.add_argument("--index", type=int, help="One acquisition or spatial band")
    parts.add_argument(
        "--all-parts", action="store_true", help="All rows or bands, sequentially"
    )
    args = parser.parse_args(argv)
    input_paths = resources(args.resources)
    settings = json.loads(
        (Path(__file__).resolve().parents[1] / "aux_files/analysis.json").read_text()
    )
    ARGS = SimpleNamespace(
        **json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "aux_files/reference_settings.json"
            ).read_text()
        )
    )
    if args.stage == "geometry":
        prepare_geometry(
            input_paths["analysis_geometry"],
            input_paths["legacy_reference_surface"],
            input_paths["cats_model"],
            input_paths["legacy_reference_geometry"],
        )
        return
    if args.stage == "archives":
        archive_native(
            input_paths["first_native_archives"],
            input_paths["legacy_native_archives"],
            settings,
        )
        return
    indices = (
        range(len(settings["acquisitions"]))
        if args.stage == "row"
        else band_indices(input_paths["analysis_geometry"])
    )
    count = len(indices)
    if not args.all_parts and (args.index is None or args.index not in indices):
        parser.error(f"Choose --index 0..{count - 1} or --all-parts")
    indices = indices if args.all_parts else [args.index]
    for index in indices:
        if args.stage == "row":
            build_rows(input_paths, args.work_dir, index, settings)
        else:
            pack_band(input_paths, args.work_dir, index, settings)


def prepare_geometry(analysis_geometry, reference_surface, cats_model, output):
    """Build the ancestor selection geometry from its frozen reference surface.

    The reference surface is an ancillary calibration input. It must be the
    pre-November surface used for native decision selection, never the new
    fitted mean being evaluated.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with (
        h5py.File(analysis_geometry) as geometry_file,
        h5py.File(reference_surface) as source,
        h5py.File(output, "w") as output_file,
    ):
        x, y = geometry_file["x100"][:], geometry_file["y100"][:]
        for k, values in [
            ("x100", x),
            ("y100", y),
            ("x", geometry_file["global_x"][:]),
            ("y", geometry_file["global_y"][:]),
        ]:
            ds(output_file, k, values)
        for key, axis in [("x", x), ("y", y)]:
            np.testing.assert_array_equal(source[key][:], axis)
        pieces = []
        for start in range(0, len(geometry_file["cell_id"]), 8192):
            fine_mapping = geometry_file["fine_id"][start : start + 8192]
            pieces.append(fine_mapping[fine_mapping >= 0])
        ids = np.sort(np.concatenate(pieces))
        ds(output_file, "fine_id", ids)
        for key, value in [("fine_x", x[ids % len(x)]), ("fine_y", y[ids // len(x)])]:
            ds(output_file, key, value)
        point_count = len(ids)
        for key in ["dem", "rema"]:
            output_file.create_dataset(
                key, (point_count,), dtype="f4", compression="gzip"
            )
        for key in ["cats_re", "cats_im"]:
            output_file.create_dataset(
                key, (4, point_count), dtype="f4", compression="gzip"
            )
        for start in range(0, point_count, 65536):
            fine_ids = ids[start : start + 65536]
            column, row_indices = fine_ids % len(x), fine_ids // len(x)
            for key, variable in [
                ("dem", "swot_wse_geoid_cats_corrected_mean"),
                ("rema", "rema_dem_ellipsoid"),
            ]:
                value = np.full(len(fine_ids), np.nan)
                for row in np.unique(row_indices):
                    selected = row_indices == row
                    left, right = (
                        int(column[selected].min()),
                        int(column[selected].max()) + 1,
                    )
                    value[selected] = source[variable][int(row), left:right][
                        column[selected] - left
                    ]
                output_file[key][start : start + len(fine_ids)] = value
            cats_real, cats_imag, omega, phase = cats_at_points(
                cats_model, x[column], y[row_indices]
            )
            output_file["cats_re"][:, start : start + len(fine_ids)] = cats_real
            output_file["cats_im"][:, start : start + len(fine_ids)] = cats_imag
        ds(output_file, "omega", omega)
        ds(output_file, "phase", phase)
        output_file.attrs.update(
            complete=1,
            reference_surface=str(reference_surface),
            purpose="Ancestor phase2 selection reference; frozen legacy CATS interpolation/float32/flexure convention",
        )


def archive_native(source_folder, destination, settings):
    """Adapt native first-stage archives without changing any stored heights."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    mapping = dict(
        cell_id="cell",
        before_wse="raw",
        after_wse="corrected",
        before_wse_dac="raw_minus_dac",
        after_wse_dac="corrected_minus_dac",
        geoid="geoid",
    )
    for row in settings["acquisitions"]:
        name = row["id"]
        output_path = destination / (name + ".npz")
        if output_path.exists():
            raise FileExistsError(output_path)
        with np.load(Path(source_folder) / (name + ".npz")) as archive:
            data = {key: archive[value] for key, value in mapping.items()}
        np.savez_compressed(output_path, **data)


if __name__ == "__main__":
    main()

"""Fit tides, linear TIME and fine-cell elevations independently in each cell.

UTide Huber fits M2, S2, K1 and O1. The connected fit uses the resulting
coefficients, terrain levels and observation weights."""

from __future__ import annotations

from step_setup import CODE, argument_parser, setup

import json
import numpy as np
import hashlib
import importlib
import importlib.util
import os
import types
import warnings
import h5py
from pathlib import Path
from functools import cache
from layout import (
    SETTINGS,
    ds,
    band_indices,
    fit_part_indices,
    FILE_LISTS,
    fit_parts_per_band,
)
from utide._time_conversion import _normalize_time
from utide._ut_constants import constit_index_dict, ut_constants
from utide.harmonics import ut_E
from filtering import check_complete, checked_mask
from step03_filter_observations import admission_mask

def fit_independent_tides(independent_dir, acquisitions):
    check_complete(independent_dir, acquisitions)
    freeze_analysis_inputs(independent_dir, acquisitions)
    for part in fit_part_indices(independent_dir / "geometry.h5"):
        fit_cells(part, independent_dir, acquisitions)
    freeze_analysis_inputs(independent_dir, acquisitions)
    collect_cell_fits(independent_dir, acquisitions)


def main(argv=None):
    args = argument_parser(__doc__).parse_args(argv)
    setup(args)
    if args.dry_run:
        print(f"Work folder: {args.work_dir}")
        print(
            "Fit four tidal constituents, TIME and fine-cell elevations independently in each cell."
        )
        return
    fit_independent_tides(args.independent, args.acquisitions)


# Check the weighted design


def fit_condition(design, weights):
    """Check rank and condition for the weighted tide and fine-cell columns."""
    cell_count, _, fine_count = weights.shape
    parameter_count = design.shape[-1]
    fine_weight_sum = weights.sum(1)
    date_weight_sum = weights.sum(2)
    norm = np.sqrt(np.einsum("ntp,ntp,nt->np", design, design, date_weight_sum))
    norm = np.maximum(norm, 1e-30)
    normalized_design = design / norm[:, None, :]
    static_cross = np.einsum("ntf,ntp->nfp", weights, normalized_design)
    inverse_fine_weight = np.divide(
        1.0,
        fine_weight_sum,
        out=np.zeros_like(fine_weight_sum),
        where=fine_weight_sum > 0,
    )
    normal_block = np.einsum(
        "ntp,ntq,nt->npq", normalized_design, normalized_design, date_weight_sum
    )
    profiled_block = normal_block - np.einsum(
        "nfp,nfq,nf->npq", static_cross, static_cross, inverse_fine_weight
    )
    eigenvalues = np.linalg.eigvalsh(profiled_block)
    full_rank = eigenvalues[:, 0] > np.finfo(float).eps * 100
    # Unobserved nuisance columns are excluded by an artificial orthogonal
    # unit diagonal. Their added eigenvalue=1 cannot change the extrema of
    # an observed correlation Gram matrix (whose trace equals dimension).
    full = np.zeros(
        (cell_count, parameter_count + fine_count, parameter_count + fine_count)
    )
    full[:, :parameter_count, :parameter_count] = normal_block
    cross = static_cross * np.sqrt(inverse_fine_weight)[:, :, None]
    full[:, parameter_count:, :parameter_count] = cross
    full[:, :parameter_count, parameter_count:] = np.swapaxes(cross, 1, 2)
    static_columns = np.arange(parameter_count, parameter_count + fine_count)
    full[:, static_columns, static_columns] = 1.0
    joint_eigenvalues = np.linalg.eigvalsh(full)
    positive = joint_eigenvalues[:, 0] > np.finfo(float).eps * 100
    condition = np.sqrt(
        np.divide(
            joint_eigenvalues[:, -1],
            joint_eigenvalues[:, 0],
            out=np.full(cell_count, np.inf),
            where=positive,
        )
    )
    return condition, full_rank


# Astronomical tidal design


CONSTITUENTS = ("M2", "S2", "K1", "O1")


REFERENCE_LATITUDES = np.array([-30.0, -50.0, -75.0])


def latitude_factors(latitude):
    """Compute the two latitude factors used by these primary constituents."""
    sin_latitude = np.sin(np.deg2rad(np.asarray(latitude)))
    return np.stack(
        [
            np.ones_like(sin_latitude),
            0.36309 * (1 - 5 * sin_latitude * sin_latitude) / sin_latitude,
            2.59808 * sin_latitude,
        ],
        axis=-1,
    )


def astronomical_latitude_weights(latitude):
    """Find weights for the three precomputed reference-latitude bases."""
    return latitude_factors(latitude) @ np.linalg.inv(
        latitude_factors(REFERENCE_LATITUDES)
    )


def astronomical_bases(datetimes):
    """Exact latitude interpolation for these four non-shallow constituents.

    UTide's complex nodal factor is affine in its two latitude factors.
    Precomputing three bases avoids repeating astronomy for every shelf cell.
    This identity is tested against ut_E at independent shelf latitudes.
    """
    days = _normalize_time(np.asarray(datetimes, dtype="datetime64[ns]"))
    indices = [constit_index_dict[name] for name in CONSTITUENTS]
    if np.any(np.isfinite(ut_constants.const.nshallow[indices])):
        raise ValueError("Latitude interpolation requires primary constituents")
    result = []
    for lat in REFERENCE_LATITUDES:
        complex_basis = ut_E(
            days,
            np.mean(days),
            ut_constants.const.freq[indices],
            indices,
            lat,
            [False] * 4,
            [],
        )
        result.append(
            np.stack([complex_basis.real, complex_basis.imag], axis=-1).reshape(
                len(days), 8
            )
        )
    return np.asarray(result)


# UTide Huber fitting


FIT_SETTINGS = SETTINGS["independent_fit"]
TUNE = FIT_SETTINGS["huber_tune"]
TOL = FIT_SETTINGS["huber_tolerance"]
MAXIT = FIT_SETTINGS["huber_max_iterations"]
MINIMUM_DATES = FIT_SETTINGS["minimum_dates"]
MAXIMUM_CONDITION = FIT_SETTINGS["maximum_joint_condition"]


CALLBACK_SOURCE = "def huber_array_safe(r):\n    return 1 / np.maximum(1, np.abs(r))\n"


EXPECTED_ENGINE_SHA256 = (
    "b6484a022f25148bcba137621e86cf9f69cddbe467c4332b6192bcca86bc2e9c"
)


@cache
def utide_engine():
    """Return the pinned UTide solver with an array-safe Huber callback."""
    override = os.environ.get("LCIS_UTIDE_ROBUSTFIT_SOURCE")
    if override:
        path = Path(override).expanduser().resolve()
        spec = importlib.util.spec_from_file_location(
            "lcis_trial_utide_robustfit_0_4_0", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                "Cannot load specified UTide0.4.0 robustfit source: " + str(path)
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module("utide.robustfit")
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if digest != EXPECTED_ENGINE_SHA256:
        raise RuntimeError(
            "UTide robustfit source differs from the frozen1024-cell trial. "
            "Set LCIS_UTIDE_ROBUSTFIT_SOURCE to the exact copied0.4.0 source. "
            "ObservedSHA256=" + digest
        )

    def huber_array_safe(r):
        return 1 / np.maximum(1, np.abs(r))

    namespace = dict(module.robustfit.__globals__)
    namespace["wfuncdict"] = dict(module.wfuncdict, huber=huber_array_safe)
    fn = types.FunctionType(
        module.robustfit.__code__,
        namespace,
        "utide_robustfit_array_safe",
        module.robustfit.__defaults__,
    )
    fn.__kwdefaults__ = module.robustfit.__kwdefaults__
    return module, fn


def huber_metadata():
    module, fn = utide_engine()
    package = importlib.import_module("utide")
    return dict(
        engine="utide.robustfit.robustfit",
        engine_path=str(Path(module.__file__).resolve()),
        engine_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        required_engine_sha256=EXPECTED_ENGINE_SHA256,
        engine_source_override=os.environ.get("LCIS_UTIDE_ROBUSTFIT_SOURCE"),
        installed_utide_package_version=str(package.__version__),
        engine_source="Exact frozenPythonUTide0.4.0 robustfit.py used by the1024-cell comparison; sourceSHA enforced independently of installedpackage version.",
        identical_engine_bytecode=fn.__code__ is module.robustfit.__code__,
        private_callback_source=CALLBACK_SOURCE,
        installed_module_unchanged=True,
        tune=TUNE,
        rcond=None,
        tol=TOL,
        maxit=MAXIT,
        model="Exactly nine FOUR/TIME coefficients plus one static elevation per active repeated fine location; no extra intercept.",
        precision="Use caller sample_weight exactly, then constant per-cell max normalization by default. No precision floor is estimated here.",
        row_multiplier="Actual UTide full rf.w on both X and y; normal_weight=normalized_base_precision*rf.w**2.",
        scale="Returned scale=unweighted physical residual MAD/0.6745 in metres. whitened_scale=MAD of returned-fit precision-whitened residual/0.6745. Neither is a fitted concomitant scale.",
        acceptance=(
            f"At least {MINIMUM_DATES} unique/effective dates, full rank and normalized "
            f"joint-design condition <= {MAXIMUM_CONDITION}; maxit {MAXIT} or numerical failures remain missing."
        ),
        qualification="Precision-prewhitened UTide IRWLS, not the current concomitant-Huber objective and not stock public utide.solve with a single intercept.",
    )


def residual_mad(residual):
    return np.median(abs(residual - np.median(residual))) / 0.6745


def fit_huber(height, design, sample_weight, train=None, normalize_dates=False):
    """Drop-in FOUR/TIME fine-static alternative to pinning_fine_solver.fit.

    Input shapes: height/sample_weight=(dates,cells,fine); design=(cells,dates,9).
    ``sample_weight`` already includes the caller's frozen source admission.
    Failed candidates retain NaN beta/terrain/prediction and explicit status.
    Diagnostic weights can exist for failed candidates; callers MUST also
    require finite beta/accepted status before using their normal equations.
    """
    heights = np.asarray(height)
    design_values = np.asarray(design)
    supplied_weights = np.asarray(sample_weight)
    if heights.ndim != 3 or supplied_weights.shape != heights.shape:
        raise ValueError(
            "height and sample_weight must have identical(date,cell,fine) shapes"
        )
    date_count, cell_count, fine_count = heights.shape
    if design_values.shape != (cell_count, date_count, 9):
        raise ValueError("Adapter supports exactly nine FOUR+TIME design columns")
    train = np.ones(date_count, bool) if train is None else np.asarray(train, bool)
    if train.shape != (date_count,):
        raise ValueError("train must be a one-dimensional date mask")
    result = dict(
        beta=np.full((9, cell_count), np.nan),
        terrain=np.full((cell_count, fine_count), np.nan),
        prediction=np.full(heights.shape, np.nan),
        count=np.zeros(cell_count, "u2"),
        condition=np.full(cell_count, np.inf),
        effective_n=np.zeros(cell_count),
        scale=np.full(cell_count, np.nan),
        whitened_scale=np.full(cell_count, np.nan),
        iterations=np.zeros(cell_count, "u2"),
        rank=np.zeros(cell_count, "u1"),
        fine_count=np.zeros(cell_count, "u1"),
        converged=np.zeros(cell_count, bool),
        status=np.full(cell_count, "not_attempted", dtype="U48"),
        row_multiplier=np.zeros(heights.shape),
        normal_weight=np.zeros(heights.shape),
        base_weight_normalization=np.full(cell_count, np.nan),
        weighted_refit_prediction_defect_m=np.full(cell_count, np.nan),
    )
    _, robustfit = utide_engine()
    tiny = np.finfo(float).tiny
    for cell in range(cell_count):
        cell_height = heights[:, cell].astype(float)
        cell_design = design_values[cell].astype(float)
        supplied = supplied_weights[:, cell].astype(float)
        observed = np.isfinite(cell_height) & (supplied > 0) & train[:, None]
        valid = observed & (observed.sum(0) >= 2)[None]
        result["count"][cell] = int(valid.any(1).sum())
        fine = np.flatnonzero(valid.any(0))
        result["fine_count"][cell] = len(fine)
        if result["count"][cell] < MINIMUM_DATES:
            result["status"][cell] = f"fewer_than{MINIMUM_DATES}_dates"
            continue
        if not np.isfinite(supplied[valid]).all():
            result["status"][cell] = "nonfinite_sample_weight"
            continue
        raw = np.where(valid, supplied, 0)
        if normalize_dates:
            date_totals = raw.sum(1)[:, None]
            base_full = np.divide(
                raw, date_totals, out=np.zeros_like(raw), where=date_totals > 0
            )
            result["base_weight_normalization"][cell] = 1.0
        else:
            maximum = max(float(raw.max()), tiny)
            base_full = raw / maximum
            result["base_weight_normalization"][cell] = maximum
        row, col = np.where(valid)
        joint_design = np.column_stack(
            [cell_design[row], np.eye(len(fine))[np.searchsorted(fine, col)]]
        )
        if not np.isfinite(joint_design).all():
            result["status"][cell] = "nonfinite_design"
            continue
        if len(row) <= joint_design.shape[1]:
            result["status"][cell] = "insufficient_residual_degrees_freedom"
            continue
        y = cell_height[valid]
        base = base_full[valid]
        norm = np.sqrt(np.sum(base[:, None] * joint_design * joint_design, axis=0))
        if not (np.isfinite(norm).all() and np.all(norm > 0)):
            result["status"][cell] = "invalid_column_norm"
            continue
        normalized_design = joint_design / norm
        root_base = np.sqrt(base)
        weighted_design, weighted_height = (
            normalized_design * root_base[:, None],
            y * root_base,
        )
        try:
            singular = np.linalg.svd(weighted_design, compute_uv=False)
            initial_condition = singular[0] / singular[-1]
            if not np.isfinite(initial_condition) or initial_condition > MAXIMUM_CONDITION:
                result["status"][cell] = "initial_condition_above20"
                continue
            # UTide assumes an overdetermined, full-rank design and a usable
            # residual MAD. Detect unsupported inputs without editing rows.
            initial, _, initial_rank, _ = np.linalg.lstsq(
                weighted_design, weighted_height, rcond=None
            )
            if initial_rank != joint_design.shape[1]:
                result["status"][cell] = "initial_rank_failure"
                continue
            initial_residual = weighted_height - weighted_design @ initial
            initial_mad = residual_mad(initial_residual)
            if not np.isfinite(initial_mad) or initial_mad <= 0:
                result["status"][cell] = "zero_or_nonfinite_initial_mad"
                continue
            with (
                warnings.catch_warnings(),
                np.errstate(divide="raise", invalid="raise", over="raise"),
            ):
                warnings.simplefilter("error", RuntimeWarning)
                robust_fit = robustfit(
                    weighted_design,
                    weighted_height,
                    weight_function="huber",
                    tune=TUNE,
                    rcond=None,
                    tol=TOL,
                    maxit=MAXIT,
                )
        except (
            ValueError,
            IndexError,
            np.linalg.LinAlgError,
            FloatingPointError,
            RuntimeWarning,
        ) as exc:
            result["status"][cell] = "utide_exception_" + type(exc).__name__
            continue
        result["iterations"][cell] = int(robust_fit.iterations)
        result["rank"][cell] = int(robust_fit.rank)
        robust_row = np.asarray(robust_fit.w, float)
        if (
            robust_row.shape != y.shape
            or not np.isfinite(robust_row).all()
            or np.any(robust_row < 0)
            or not np.isfinite(robust_fit.b).all()
        ):
            result["status"][cell] = "nonfinite_returned_fit_or_weights"
            continue
        normal_weight = np.zeros_like(cell_height)
        normal_weight[valid] = base * robust_row**2
        result["row_multiplier"][:, cell][valid] = robust_row
        result["normal_weight"][:, cell] = normal_weight
        condition, final_rank = fit_condition(cell_design[None], normal_weight[None])
        date_weight = normal_weight.sum(1)
        effective_dates = date_weight.sum() ** 2 / max(
            float(np.sum(date_weight**2)), tiny
        )
        result["condition"][cell] = condition[0]
        result["effective_n"][cell] = effective_dates
        if robust_fit.iterations >= MAXIT:
            result["status"][cell] = f"utide_maxit{MAXIT}"
            continue
        if robust_fit.rank != joint_design.shape[1] or not final_rank[0]:
            result["status"][cell] = "rank_failure"
            continue
        if (
            not np.isfinite(condition[0])
            or condition[0] > MAXIMUM_CONDITION
            or effective_dates < MINIMUM_DATES - 1e-10
        ):
            result["status"][cell] = "final_condition_or_effective_gate"
            continue
        # This is a diagnostic identity, not a replacement of UTide's answer.
        refit = np.linalg.lstsq(
            weighted_design * robust_row[:, None],
            weighted_height * robust_row,
            rcond=None,
        )[0]
        defect = float(np.max(abs(normalized_design @ (refit - robust_fit.b))))
        result["weighted_refit_prediction_defect_m"][cell] = defect
        if defect > 1e-7:
            result["status"][cell] = "returned_weights_do_not_match_fit"
            continue
        coefficients = np.asarray(robust_fit.b, float) / norm
        static_levels = np.full(fine_count, np.nan)
        static_levels[fine] = coefficients[9:]
        prediction = (cell_design @ coefficients[:9])[:, None] + static_levels
        physical = (cell_height - prediction)[valid]
        whitened = weighted_height - weighted_design @ robust_fit.b
        physical_mad = residual_mad(physical)
        whitened_mad = residual_mad(whitened)
        if not (
            np.isfinite(physical_mad)
            and np.isfinite(whitened_mad)
            and physical_mad > 0
            and whitened_mad > 0
        ):
            result["status"][cell] = "zero_or_nonfinite_final_mad"
            continue
        result["beta"][:, cell] = coefficients[:9]
        result["terrain"][cell] = static_levels
        result["prediction"][:, cell] = prediction
        result["scale"][cell] = physical_mad
        result["whitened_scale"][cell] = whitened_mad
        result["converged"][cell] = True
        result["status"][cell] = "accepted"
    return result


# Read the observations and fit independent cells


def fit_geometry(independent_dir, acquisitions):
    """Read the shared grid and the locations of completed independent fits."""
    with h5py.File(independent_dir / "geometry.h5") as handle:
        assert handle.attrs["complete"] == 1
        geometry_info = {
            name: (
                handle[name][:acquisitions]
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
                "reference_seconds",
                "time_ns",
                "passes",
            ]
        }
    grid_width = len(geometry_info["global_x"])
    geometry_info["nx"] = grid_width
    geometry_info["xy"] = np.c_[
        geometry_info["global_x"][geometry_info["cell_id"] % grid_width],
        geometry_info["global_y"][geometry_info["cell_id"] // grid_width],
    ]
    if (independent_dir / "independent.h5").exists():
        with h5py.File(independent_dir / "independent.h5") as handle:
            cell_count = len(geometry_info["cell_id"])
            geometry_info["valid"] = np.empty(cell_count, bool)
            for start in range(0, cell_count, 16384):
                geometry_info["valid"][start : start + 16384] = np.isfinite(
                    handle["beta"][:, start : start + 16384]
                ).all(0)
            geometry_info["part_id"] = handle["part_id"][:]
            geometry_info["part_local"] = handle["part_local"][:]
    return geometry_info


def tide_and_trend_basis(seconds, lat, ref, geometry_path):
    """Build eight tide columns and one TIME column at the observation clocks."""
    seconds = np.where(np.isfinite(seconds), seconds, np.asarray(ref)[None])
    unique_seconds, time_lookup = np.unique(
        np.rint(seconds).astype("i8"), return_inverse=True
    )
    dates = np.datetime64("2000-01-01", "ns") + unique_seconds.astype("timedelta64[s]")
    astronomical_values = astronomical_bases(dates)
    latitude_weights = astronomical_latitude_weights(lat)
    tidal_design = np.einsum(
        "nk,kntp->ntp",
        latitude_weights,
        astronomical_values[:, time_lookup.reshape(seconds.shape), :],
    )
    # The reference epoch and TIME scaling are fixed calibration inputs.
    with h5py.File(geometry_path) as handle:
        centre = float(handle.attrs["time_center_seconds_2000"])
        span = float(handle.attrs["time_span_seconds"])
    return np.concatenate(
        [tidal_design, ((seconds - centre) / span)[:, :, None]], axis=2
    )


def read_admitted_observations(nodes, g, independent_dir, acquisitions):
    """Read fine heights and apply the frozen REMA admission gate.

    Returns heights[date, cell, fine], design[cell, date, 9] and an admission mask.
    The reference selects observations; it is never subtracted from heights.
    """
    cell_count = len(nodes)
    heights = np.full((acquisitions, cell_count, 9), np.nan, "f4")
    reference = np.full((cell_count, 9), np.nan)
    seconds = np.full((acquisitions, cell_count), np.nan)
    for band in np.unique(g["source_band"][nodes]):
        destination = np.flatnonzero(g["source_band"][nodes] == band)
        local_index = g["source_local"][nodes[destination]]
        assert np.all(np.diff(local_index) > 0)
        with h5py.File(independent_dir / "inputs" / f"{int(band):03d}.h5") as handle:
            np.testing.assert_array_equal(
                handle["cell_id"][local_index], g["cell_id"][nodes[destination]]
            )
            heights[:, destination] = handle["height"][:acquisitions, local_index]
            reference[destination] = handle["reference_geoid"][local_index]
            seconds[:, destination] = handle["time_seconds"][:acquisitions, local_index]
    mask = checked_mask(independent_dir, nodes, g, admission_mask(heights, reference))
    return (
        heights,
        tide_and_trend_basis(
            seconds.T,
            g["lat"][nodes],
            g["reference_seconds"],
            independent_dir / "geometry.h5",
        ),
        mask,
    )


def fit_cells(index, independent_dir, acquisitions):
    """Fit one cell group from the geometry layout, preserving observation order."""
    geometry_path = independent_dir / "geometry.h5"
    parts = fit_part_indices(geometry_path)
    if index not in parts:
        raise ValueError(f"Part index must be in {parts}")
    target = independent_dir / "fits" / f"{index:03d}.h5"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with h5py.File(target) as handle:
            assert handle.attrs["complete"] == 1
        return
    geometry_info = fit_geometry(independent_dir, acquisitions)
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
            (acquisitions, cell_count, 9),
            dtype="f8",
            chunks=(1, min(128, cell_count), 9),
            compression="gzip",
            compression_opts=2,
            shuffle=True,
        )
        handle.create_dataset("status", (cell_count,), dtype="S48", compression="gzip")
        for start in range(0, cell_count, 128):
            stop = min(start + 128, cell_count)
            heights, design, mask = read_admitted_observations(
                nodes[start:stop], geometry_info, independent_dir, acquisitions
            )
            fit = fit_huber(heights, design, mask.astype(float))
            for name in shapes:
                if name == "beta":
                    handle[name][:, start:stop] = fit[name]
                else:
                    handle[name][start:stop] = fit[name]
            normal_weight = fit["normal_weight"]
            normal_weight[:, ~np.isfinite(fit["beta"]).all(0)] = 0
            handle["normal_weight"][:, start:stop] = normal_weight
            handle["status"][start:stop] = fit["status"].astype("S48")
        handle.attrs.update(complete=1, estimator_json=json.dumps(huber_metadata()))
    temporary.replace(target)


def collect_cell_fits(independent_dir, acquisitions):
    """Combine all fit parts defined by the geometry into independent.h5."""
    target = independent_dir / "independent.h5"
    if target.exists():
        with h5py.File(target) as output:
            assert output.attrs["complete"] == 1
        return
    cell_count = len(fit_geometry(independent_dir, acquisitions)["cell_id"])
    temporary = target.with_suffix(".tmp.h5")
    with h5py.File(temporary, "w") as output:
        seen = np.zeros(cell_count, bool)
        part_id = np.full(cell_count, -1, "i2")
        local = np.full(cell_count, -1, "i4")
        for part in fit_part_indices(independent_dir / "geometry.h5"):
            with h5py.File(independent_dir / "fits" / f"{part:03d}.h5") as source:
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


def freeze_analysis_inputs(independent_dir, acquisitions, gls_dir=None):
    """Freeze completed inputs before fitting or combining any observations."""
    geometry = independent_dir / "geometry.h5"
    bands = band_indices(geometry)
    files = [
        geometry,
        *(independent_dir / "inputs" / f"{band:03d}.h5" for band in bands),
    ]
    # The numbered filtering step saves these masks. Freeze them with the
    # observations so a later mask edit cannot change the fitted support.
    admission = independent_dir / "admission"
    if admission.is_dir():
        files += [admission / f"{band:03d}.h5" for band in bands]
    receipt_dir = independent_dir
    if gls_dir is not None:
        files += [independent_dir / "independent.h5"]
        files += [
            independent_dir / "fits" / f"{part:03d}.h5"
            for part in fit_part_indices(geometry)
        ]
        receipt_dir = gls_dir
    freeze_inputs(receipt_dir / "source_inventory.json", files, acquisitions)


def freeze_inputs(path, files, acquisitions):
    """Freeze completed inputs/code by small hashes and file metadata."""
    code_dir = CODE
    aux_dir = CODE / "aux_files"
    watched = [
        *code_dir.glob("*.py"),
        *(CODE / "processing").glob("*.py"),
        *aux_dir.glob("*.json"),
        *FILE_LISTS.glob("*.tsv"),
        aux_dir / "figure3_sampling.h5",
    ]
    hashes = {
        str(p.relative_to(CODE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
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
        source_snapshots=records, source_code_sha256=hashes, acquisitions=acquisitions
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


if __name__ == "__main__":
    main()

"""Fit four tides, TIME and repeated fine-cell elevations with UTide Huber.

Inputs:
    height, sample_weight: [date, coarse cell, fine position]
    design:                [coarse cell, date, 9]

Each accepted cell has eight M2/S2/K1/O1 sine/cosine coefficients, one TIME
coefficient and one static level per observed fine position. The frozen UTide
0.4.0 engine supplies the full row multipliers; their squares enter the normal
weights. Only its private scalar Huber callback is made array-safe.

Returned scale is physical residual MAD/0.6745 in metres. Network stages use
normal_weight directly, rather than reconstructing weights from that scale.
Unsupported or failed fits remain NaN, with an explicit status.
"""

from pathlib import Path

import hashlib
import importlib
import importlib.util

import os
import types
import warnings

import numpy as np
from terrain import profile

TUNE = 1.345
TOL = 0.001
MAXIT = 50
CALLBACK_SOURCE = "def huber_array_safe(r):\n    return 1 / np.maximum(1, np.abs(r))\n"
EXPECTED_ENGINE_SHA256 = (
    "b6484a022f25148bcba137621e86cf9f69cddbe467c4332b6192bcca86bc2e9c"
)
_ENGINE = None


def engine():
    """Return unmodified robustfit bytecode with one private safe callback."""
    global _ENGINE
    if _ENGINE is None:
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
        _ENGINE = module, fn
    return _ENGINE


def estimator_metadata():
    module, fn = engine()
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
        acceptance="At least12 unique/effective dates, full rank and normalized full joint-design condition<=20; maxit50 or numerical failures remain missing, no fallback.",
        qualification="Precision-prewhitened UTide IRWLS, not the current concomitant-Huber objective and not stock public utide.solve with a single intercept.",
    )


def fit(height, design, sample_weight, train=None, normalize_dates=False):
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
    _, robustfit = engine()
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
        if result["count"][cell] < 12:
            result["status"][cell] = "fewer_than12_dates"
            continue
        if not np.isfinite(supplied[valid]).all():
            result["status"][cell] = "nonfinite_sample_weight"
            continue
        raw = np.where(valid, supplied, 0)
        if normalize_dates:
            base_full = np.divide(
                raw,
                raw.sum(1)[:, None],
                out=np.zeros_like(raw),
                where=raw.sum(1)[:, None] > 0,
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
            if not np.isfinite(initial_condition) or initial_condition > 20:
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
            initial_mad = (
                np.median(abs(initial_residual - np.median(initial_residual))) / 0.6745
            )
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
        _, _, _, condition, final_rank = profile(
            np.where(valid, cell_height, 0)[None],
            cell_design[None],
            normal_weight[None],
            True,
        )
        date_weight = normal_weight.sum(1)
        effective_dates = date_weight.sum() ** 2 / max(
            float(np.sum(date_weight**2)), tiny
        )
        result["condition"][cell] = condition[0]
        result["effective_n"][cell] = effective_dates
        if robust_fit.iterations >= MAXIT:
            result["status"][cell] = "utide_maxit50"
            continue
        if robust_fit.rank != joint_design.shape[1] or not final_rank[0]:
            result["status"][cell] = "rank_failure"
            continue
        if (
            not np.isfinite(condition[0])
            or condition[0] > 20
            or effective_dates < 12 - 1e-10
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
        physical_mad = np.median(abs(physical - np.median(physical))) / 0.6745
        whitened_mad = np.median(abs(whitened - np.median(whitened))) / 0.6745
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

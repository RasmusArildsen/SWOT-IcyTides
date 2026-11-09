"""Four-constituent recovery using the production UTide astronomical basis."""

import numpy as np
from utide._time_conversion import _normalize_time
from utide._ut_constants import constit_index_dict, ut_constants
from utide.harmonics import ut_E

NAMES = ("M2", "S2", "K1", "O1")
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


def latitude_weights(latitude):
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
    indices = [constit_index_dict[name] for name in NAMES]
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


def design(bases, days, latitude):
    tidal = np.einsum("k,ktp->tp", latitude_weights(latitude), bases)
    span = np.ptp(days)
    return np.column_stack(
        [tidal, np.ones(len(days)), (days - (days[0] + days[-1]) / 2) / span]
    )


def ols_operator(matrix):
    """Use scaled SVD, avoiding squared condition numbers from normal equations."""
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms == 0) or len(matrix) <= matrix.shape[1]:
        return None, np.inf
    scaled = matrix / norms
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        scaled, full_matrices=False
    )
    condition = (
        singular_values[0] / singular_values[-1] if singular_values[-1] > 0 else np.inf
    )
    if (
        singular_values[-1]
        <= np.finfo(float).eps * max(scaled.shape) * singular_values[0]
    ):
        return None, condition
    return ((right_vectors.T / singular_values) @ left_vectors.T) / norms[
        :, None
    ], condition


def coefficient_draws(beta, operator, covariance, members, rng):
    coefficient_covariance = operator @ covariance @ operator.T
    eigenvalues, eigenvectors = np.linalg.eigh(
        (coefficient_covariance + coefficient_covariance.T) / 2
    )
    return beta[:, None] + (
        eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    ) @ rng.standard_normal((len(beta), members))


def amplitude_phase(beta):
    return (
        np.hypot(beta[:8:2], beta[1:8:2]),
        np.rad2deg(np.arctan2(beta[1:8:2], beta[:8:2])) % 360,
    )


def circular_difference(a, b):
    return (a - b + 180) % 360 - 180


def dense_tide_rmse(draws, truth, weights, gram):
    # Mean and trend are fitted nuisance terms, excluded from tidal predictions.
    coefficients = np.vstack(
        [
            (weights[:, None, None] * draws[None, :8]).reshape(24, -1),
            -np.broadcast_to(truth[:, None], (8, draws.shape[1])),
        ]
    )
    squared = np.sum(coefficients * (gram @ coefficients), axis=0)
    return np.sqrt(np.maximum(squared, 0))

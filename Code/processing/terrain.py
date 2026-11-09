"""Profile fine-cell static elevations out of a weighted least-squares fit."""

import numpy as np


def profile(heights, design, weights, diagnostics=False):
    """Weighted least squares, profiling static pixel means by Schur complement."""
    cell_count, date_count, fine_count = heights.shape
    parameter_count = design.shape[-1]
    fine_weight_sum = weights.sum(1)
    date_weight_sum = weights.sum(2)
    norm = np.sqrt(np.einsum("ntp,ntp,nt->np", design, design, date_weight_sum))
    norm = np.maximum(norm, 1e-30)
    normalized_design = design / norm[:, None, :]
    static_cross = np.einsum("ntf,ntp->nfp", weights, normalized_design)
    weighted_height_sum = np.sum(weights * heights, axis=1)
    inverse_fine_weight = np.divide(
        1.0,
        fine_weight_sum,
        out=np.zeros_like(fine_weight_sum),
        where=fine_weight_sum > 0,
    )
    normal_block = np.einsum(
        "ntp,ntq,nt->npq", normalized_design, normalized_design, date_weight_sum
    )
    rhs = np.einsum("ntp,nt->np", normalized_design, np.sum(weights * heights, axis=2))
    profiled_block = normal_block - np.einsum(
        "nfp,nfq,nf->npq", static_cross, static_cross, inverse_fine_weight
    )
    profiled_rhs = rhs - np.einsum(
        "nfp,nf,nf->np", static_cross, weighted_height_sum, inverse_fine_weight
    )
    eigenvalues = np.linalg.eigvalsh(profiled_block)
    good = eigenvalues[:, 0] > np.finfo(float).eps * 100
    gamma = np.full((cell_count, parameter_count), np.nan)
    if good.any():
        gamma[good] = np.linalg.solve(
            profiled_block[good], profiled_rhs[good, :, None]
        )[..., 0]
    beta = gamma / norm
    terrain = np.divide(
        weighted_height_sum - np.einsum("nfp,np->nf", static_cross, gamma),
        fine_weight_sum,
        out=np.full(fine_weight_sum.shape, np.nan),
        where=fine_weight_sum > 0,
    )
    prediction = np.einsum("ntp,np->nt", design, beta)[:, :, None] + terrain[:, None, :]
    condition = np.full(cell_count, np.inf)
    if diagnostics:
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
    return beta, terrain, prediction, condition, good

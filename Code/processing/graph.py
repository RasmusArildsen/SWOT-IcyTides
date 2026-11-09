"""Build cell and neighbour blocks for the connected GLS fit."""

import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import LinearOperator, cg
from scipy.optimize import minimize_scalar

P = 18


def blocks(heights, design, normal_weight, residual_scale, rho, seed):
    """Build cell blocks and shared-clock terms.

    H and W have shape [date, cell, fine]; T is [cell, date, 9].
    Each block has nine tide/TIME coefficients and nine fine static levels.
    """
    date_count, cell_count, fine_count = heights.shape
    precision = normal_weight / (
        residual_scale[None, :, None] ** 2 * (1 - rho * normal_weight)
    )
    date_precision = precision.sum(2)
    loading = np.sqrt(rho) * residual_scale
    residual = np.where(
        normal_weight > 0,
        heights
        - np.einsum("ntp,np->tn", design, seed[:, :9])[:, :, None]
        - seed[None, :, 9:],
        0.0,
    )
    clock_loading = (
        np.concatenate(
            [date_precision[:, :, None] * design.transpose(1, 0, 2), precision], 2
        ).transpose(1, 2, 0)
        * loading[:, None, None]
    )
    normal_blocks = np.zeros((cell_count, P, P))
    block_rhs = np.zeros((cell_count, P))
    inverse_blocks = np.zeros_like(normal_blocks)
    for cell in range(cell_count):
        cell_precision = precision[:, cell]
        cell_design = design[cell]
        cross = np.einsum("tf,tp->fp", cell_precision, cell_design)
        normal_blocks[cell] = np.block(
            [
                [
                    cell_design.T @ (cell_precision.sum(1)[:, None] * cell_design),
                    cross.T,
                ],
                [cross, np.diag(cell_precision.sum(0))],
            ]
        )
        block_rhs[cell, :9] = cell_design.T @ (cell_precision * residual[:, cell]).sum(
            1
        )
        block_rhs[cell, 9:] = (cell_precision * residual[:, cell]).sum(0)
        use = np.diag(normal_blocks[cell]) > 1e-14
        assert use[:9].all()
        active_parameters = np.flatnonzero(use)
        active_block = normal_blocks[cell][np.ix_(active_parameters, active_parameters)]
        norm = np.sqrt(np.diag(active_block))
        scaled_block = active_block / norm[:, None] / norm[None, :]
        active_inverse = (
            cho_solve(
                cho_factor((scaled_block + scaled_block.T) / 2, lower=True),
                np.eye(len(active_parameters)),
            )
            / norm[:, None]
            / norm[None, :]
        )
        inverse_blocks[cell][
            np.ix_(active_parameters, active_parameters)
        ] = active_inverse
    sumq = np.sum(date_precision * loading[None, :] ** 2, 1)
    sumy = np.sum(precision * residual * loading[None, :, None], axis=(1, 2))
    return normal_blocks, inverse_blocks, clock_loading, block_rhs, sumq, sumy


def pairs(ids, grid_width):
    """Find neighbours at 250 m, 1 km and 2 km along each grid direction."""
    links = []
    for step in [1, 4, 8, grid_width, 4 * grid_width, 8 * grid_width]:
        right = np.searchsorted(ids, ids + step)
        matched = right < len(ids)
        matched[matched] &= ids[right[matched]] == ids[matched] + step
        if step < grid_width:
            matched &= ids % grid_width < grid_width - step
        left = np.flatnonzero(matched)
        links.append(np.c_[left, right[left]])
    return np.concatenate(links)


def invert_blocks(normal_blocks):
    """Invert each positive normal block using its original diagonal scaling."""
    inverse_blocks = np.zeros_like(normal_blocks)
    for cell, block in enumerate(normal_blocks):
        norm = np.sqrt(np.diag(block))
        scaled_block = block / norm[:, None] / norm[None, :]
        inverse_blocks[cell] = (
            cho_solve(
                cho_factor((scaled_block + scaled_block.T) / 2, lower=True), np.eye(P)
            )
            / norm[:, None]
            / norm[None, :]
        )
    return inverse_blocks


def system(H0, F, r0, sumq, sumy, ij, diag, off, grhs, lam):
    """Construct the small reference normal operator used by the GLS control."""
    n = len(H0)
    nt = F.shape[2]
    hd = H0 + lam * diag
    # Unobserved fine levels are absent parameters, represented by decoupled
    # unit coordinates with zero RHS. They are restored to NaN in exports.
    inactive = np.diagonal(H0, axis1=1, axis2=2) <= 1e-14
    for i, k in zip(*np.where(inactive)):
        hd[i, k, k] = 1
    rows = np.r_[np.arange(n), ij[:, 0], ij[:, 1]]
    cols = np.r_[np.arange(n), ij[:, 1], ij[:, 0]]
    data = np.concatenate([hd, lam * off, lam * off.transpose(0, 2, 1)])
    order = np.argsort(rows, kind="stable")
    counts = np.bincount(rows, minlength=n)
    sparse_matrix = bsr_matrix(
        (data[order], cols[order], np.r_[0, np.cumsum(counts)]), shape=(n * P, n * P)
    )
    sparse_matrix.sort_indices()
    clock_loading_flat = F.reshape(n * P, nt)
    den = 1 + sumq

    def mv(x):
        return sparse_matrix @ x - clock_loading_flat @ (
            (clock_loading_flat.T @ x) / den
        )

    rhs = (r0 + lam * grhs).ravel() - clock_loading_flat @ (sumy / den)
    inv = invert_blocks(hd)
    inverse_clock_loading = np.einsum("nij,njt->nit", inv, F, optimize=True).reshape(
        n * P, nt
    )
    clock_system = np.diag(den) - clock_loading_flat.T @ inverse_clock_loading
    clock_system = (clock_system + clock_system.T) / 2
    norm = np.sqrt(np.diag(clock_system))
    factor = cho_factor(clock_system / norm[:, None] / norm[None, :], lower=True)

    def small(x):
        return cho_solve(factor, x / norm) / norm

    def pre(x):
        return np.einsum(
            "nij,nj->ni", inv, x.reshape(n, P), optimize=True
        ).ravel() + inverse_clock_loading @ small(inverse_clock_loading.T @ x)

    operator = LinearOperator(sparse_matrix.shape, matvec=mv, dtype=float)
    preconditioner = LinearOperator(sparse_matrix.shape, matvec=pre, dtype=float)

    def solve(b, x0=None, verbose=False):
        iterations = [0]

        def callback(x):
            iterations[0] += 1
            if verbose and iterations[0] % 100 == 0:
                print("ERROR_GRAPH_CG", iterations[0], flush=True)

        x, status = cg(
            operator,
            b,
            x0=x0,
            M=preconditioner,
            rtol=1e-9,
            atol=0,
            maxiter=1500,
            callback=callback,
        )
        defect = float(np.linalg.norm(mv(x) - b) / max(np.linalg.norm(b), 1e-12))
        assert status == 0 and defect < 1e-7, (status, defect)
        return x, dict(iterations=iterations[0], normal_relative_residual=defect)

    return (
        operator,
        rhs,
        solve,
        dict(
            sparse_bytes=int(
                sparse_matrix.data.nbytes
                + sparse_matrix.indices.nbytes
                + sparse_matrix.indptr.nbytes
            ),
            inactive_coordinates=int(inactive.sum()),
            fixed_pixels=0,
        ),
    )


def fine_blocks(design, pairs, pair_weight, residual):
    """W is time,edge,fine; one shared static per endpoint fine location."""
    left, right = pairs.T
    left_design = design[left]
    right_design = design[right]
    w = pair_weight.transpose(1, 0, 2)
    q = w.sum(2)
    fine_count = pair_weight.shape[2]
    left_block = np.zeros((len(pairs), P, P))
    right_block = np.zeros_like(left_block)
    off = np.zeros_like(left_block)
    left_block[:, :9, :9] = np.einsum(
        "etp,etq,et->epq", left_design, left_design, q, optimize=True
    )
    right_block[:, :9, :9] = np.einsum(
        "etp,etq,et->epq", right_design, right_design, q, optimize=True
    )
    off[:, :9, :9] = -np.einsum(
        "etp,etq,et->epq", left_design, right_design, q, optimize=True
    )
    left_cross = np.einsum("etp,etf->epf", left_design, w, optimize=True)
    right_cross = np.einsum("etp,etf->epf", right_design, w, optimize=True)
    left_block[:, :9, 9:] = left_cross
    left_block[:, 9:, :9] = left_cross.transpose(0, 2, 1)
    right_block[:, :9, 9:] = right_cross
    right_block[:, 9:, :9] = right_cross.transpose(0, 2, 1)
    off[:, :9, 9:] = -left_cross
    off[:, 9:, :9] = -right_cross.transpose(0, 2, 1)
    for f in range(fine_count):
        left_block[:, 9 + f, 9 + f] = w[:, :, f].sum(1)
        right_block[:, 9 + f, 9 + f] = w[:, :, f].sum(1)
        off[:, 9 + f, 9 + f] = -w[:, :, f].sum(1)
    delta = (residual[:, left] - residual[:, right]).transpose(1, 0, 2)
    weighted = w * delta
    left_rhs = np.concatenate(
        [np.einsum("etp,et->ep", left_design, weighted.sum(2)), weighted.sum(1)], 1
    )
    right_rhs = -np.concatenate(
        [np.einsum("etp,et->ep", right_design, weighted.sum(2)), weighted.sum(1)], 1
    )
    return left_block, right_block, off, left_rhs, right_rhs


def calibration(
    absolute_precision, absolute_gradient, pair_precision, pair_gradient, admitted
):
    """Choose pair strength using the frozen conditional residual objective."""
    rows_per_date = np.maximum(admitted.sum((1, 2)), 1)
    weights = admitted / rows_per_date[:, None, None]
    weights /= weights.sum()

    def objective(lam, detail=False):
        row_precision = np.maximum(absolute_precision + lam * pair_precision, 1e-30)
        row_gradient = absolute_gradient + lam * pair_gradient
        profile_variance = float(
            np.sum(weights * row_gradient * row_gradient / row_precision)
        )
        value = float(
            np.log(max(profile_variance, 1e-30))
            - np.sum(weights * np.log(row_precision))
        )
        return (value, profile_variance) if detail else value

    optimum = minimize_scalar(
        lambda x: objective(np.expm1(x)),
        bounds=(0, np.log1p(10000)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    lam = float(np.expm1(optimum.x))
    if objective(0) <= objective(lam):
        lam = 0.0
    value, profile_variance = objective(lam, True)
    return dict(
        lambda_local=lam,
        profile_scale_squared=profile_variance,
        objective=value,
        lambda0_objective=objective(0),
        at_upper_bound=lam > 9900,
        optimization_success=bool(optimum.success),
        calibration_fine_rows=int(admitted.sum()),
        qualification="Conditional pseudolikelihood of fitted training residuals; not isolated instrumental errors.",
    )


def quietmedian(x, axis):
    """Compute a NaN-aware median without empty-support warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(x, axis=axis)

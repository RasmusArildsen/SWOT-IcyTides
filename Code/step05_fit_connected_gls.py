"""Combine absolute observations and neighbour differences in a connected GLS fit.

The independent fits provide the starting field and robust weights. This
step builds the normal equations, solves for all free coefficients and
fine-cell levels, checks convergence and exports the fitted maps."""

from __future__ import annotations

from step_setup import CODE, argument_parser, setup

import json
import numpy as np
import os
import warnings
import h5py
import time
import pandas as pd
from pathlib import Path
from layout import (
    SETTINGS,
    ds,
    dump,
    band_indices,
    DEFAULT_ACQUISITIONS,
    PARTITIONS,
)
from filtering import check_complete
from step03_filter_observations import admission_mask
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import LinearOperator, cg
from scipy.optimize import minimize_scalar

from step04_fit_independent_tides import (
    fit_geometry,
    fit_huber,
    freeze_analysis_inputs,
    read_admitted_observations,
    tide_and_trend_basis,
)


def fit_connected_gls(work_dir, acquisitions):
    global INDEPENDENT_DIR, GLS_ROOT, NT

    independent_dir = work_dir / "independent"
    gls_dir = work_dir / "gls"
    products_dir = work_dir / "figure_products"
    INDEPENDENT_DIR, GLS_ROOT, NT = independent_dir, gls_dir, acquisitions

    independent_fit = independent_dir / "independent.h5"
    if not independent_fit.is_file():
        raise FileNotFoundError(
            f"Missing {independent_fit}. Run step04_fit_independent_tides.py first."
        )
    with h5py.File(independent_fit) as fit:
        if fit.attrs.get("complete") != 1:
            raise ValueError(f"Incomplete independent fit: {independent_fit}")
    check_complete(independent_dir, acquisitions)
    freeze_analysis_inputs(independent_dir, acquisitions)

    # Prepare the connected system.
    freeze_analysis_inputs(independent_dir, acquisitions, gls_dir)
    prepare_connected_system()

    # Build absolute-observation blocks and neighbour-difference blocks.
    freeze_analysis_inputs(independent_dir, acquisitions, gls_dir)
    prepared = json.loads((gls_dir / "prepared.json").read_text())
    for part in range(prepared["node_parts"]):
        build_node_part(part)
    edge_parts = prepared["edge_parts"]
    freeze_analysis_inputs(independent_dir, acquisitions, gls_dir)
    for part in range(edge_parts):
        build_edge_part(part)

    # Assemble and solve. The solve checks its residual and injected signal.
    freeze_analysis_inputs(independent_dir, acquisitions, gls_dir)
    assemble_connected_system()
    freeze_analysis_inputs(independent_dir, acquisitions, gls_dir)
    solve_connected_system()

    # Extract the fitted DEM, tidal maps and transect for plotting.
    extract_maps_and_profiles(
        gls_dir / "native_tides.h5", independent_dir, products_dir
    )


def main(argv=None):
    args = argument_parser(__doc__).parse_args(argv)
    setup(args)
    if args.dry_run:
        print(f"Work folder: {args.work_dir}")
        print("Build and solve the connected GLS system and export the fitted maps.")
        return
    fit_connected_gls(args.work_dir, args.acquisitions)


# Neighbour differences and connected normal equations


P = 18
GLS_SETTINGS = SETTINGS["gls"]
NORMAL_RESIDUAL_MAX = GLS_SETTINGS["normal_residual_max"]
SIGNAL_ERROR_MAX_M = GLS_SETTINGS["signal_translation_error_max_m"]


def observation_residual(heights, design, normal_weight, seed, *, missing=0.0):
    tide_and_trend = np.einsum("ntp,np->tn", design, seed[:, :9])
    residual = heights - tide_and_trend[:, :, None] - seed[None, :, 9:]
    return np.where(normal_weight > 0, residual, missing)


def observation_blocks(heights, design, normal_weight, residual_scale, rho, seed):
    """Build cell blocks and shared-clock terms.

    H and W have shape [date, cell, fine]; T is [cell, date, 9].
    Each block has nine tide/TIME coefficients and nine fine static levels.
    """
    cell_count = heights.shape[1]
    precision = normal_weight / (
        residual_scale[None, :, None] ** 2 * (1 - rho * normal_weight)
    )
    date_precision = precision.sum(2)
    loading = np.sqrt(rho) * residual_scale
    residual = observation_residual(heights, design, normal_weight, seed)
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
        factor = cho_factor((scaled_block + scaled_block.T) / 2, lower=True)
        active_inverse = (
            cho_solve(factor, np.eye(len(active_parameters)))
            / norm[:, None]
            / norm[None, :]
        )
        inverse_blocks[cell][
            np.ix_(active_parameters, active_parameters)
        ] = active_inverse
    sumq = np.sum(date_precision * loading[None, :] ** 2, 1)
    sumy = np.sum(precision * residual * loading[None, :, None], axis=(1, 2))
    return normal_blocks, inverse_blocks, clock_loading, block_rhs, sumq, sumy


def neighbour_pairs(ids, grid_width):
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


def calibrate_neighbour_weight(
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


def quiet_median(x, axis):
    """Compute a NaN-aware median without empty-support warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(x, axis=axis)


# Assemble, solve and verify the connected system


INDEPENDENT_DIR = Path("work/independent").resolve()
NT = DEFAULT_ACQUISITIONS


GLS_ROOT = Path(
    os.environ.get(
        "LCIS_CONNECTED_FULL_ROOT",
        INDEPENDENT_DIR.parent / "gls",
    )
)


NODE_PARTS = PARTITIONS["gls_node_parts"]


EDGE_BATCH = 8192


def node_part_count():
    """Use the part count recorded when this connected system was prepared."""
    with h5py.File(GLS_ROOT / "metadata.h5") as source:
        count = int(source.attrs["node_parts"])
    if count < 1:
        raise ValueError("Prepared node_parts must be positive")
    return count


def assert_sources():
    """Refuse a fit if any frozen prepared source has changed."""
    path = GLS_ROOT / "source_inventory.json"
    if not path.exists():
        return
    records = json.loads(path.read_text())["source_snapshots"]
    assert all(
        Path(v["path"]).stat().st_size == v["bytes"]
        and Path(v["path"]).stat().st_mtime_ns == v["mtime_ns"]
        for v in records
    ), "Frozen prepared source or baseline changed."


def connected_geometry():
    """Read the shared geometry and select finite independent fits."""
    geometry_info = fit_geometry(INDEPENDENT_DIR, NT)
    geometry_info["nodes"] = np.flatnonzero(geometry_info["valid"])
    return geometry_info


def load_connected_observations(nodes, geometry):
    """Read fine observations and the results of the independent fit."""
    nodes = np.asarray(nodes, int)
    heights, design, _ = read_admitted_observations(
        nodes, geometry, INDEPENDENT_DIR, NT
    )
    with h5py.File(INDEPENDENT_DIR / "independent.h5") as fit:
        seed = np.c_[fit["beta"][:, nodes].T, fit["terrain"][nodes]]
        residual_scale = np.maximum(fit["scale"][nodes], 1e-8)

    normal_weight = np.zeros(heights.shape)
    for part in np.unique(geometry["part_id"][nodes]):
        destination = np.flatnonzero(geometry["part_id"][nodes] == part)
        local_index = geometry["part_local"][nodes[destination]]
        assert np.all(np.diff(local_index) > 0)
        with h5py.File(INDEPENDENT_DIR / "fits" / f"{int(part):03d}.h5") as fit:
            for start in range(0, len(destination), 512):
                cells = slice(start, start + 512)
                normal_weight[:, destination[cells]] = fit["normal_weight"][
                    :NT, local_index[cells]
                ]
    return heights.astype(float), design, normal_weight, residual_scale, seed


def rho_calibration(g):
    """Estimate the working shared covariance from the frozen residual-pair rule."""
    sample_cells = GLS_SETTINGS["rho_sample_cells"]
    min_distance, max_distance = GLS_SETTINGS["rho_pair_distance_m"]
    minimum_dates = GLS_SETTINGS["rho_minimum_paired_dates"]
    nodes = g["nodes"][np.linspace(0, len(g["nodes"]) - 1, sample_cells).round().astype(int)]
    heights, design, normal_weight, residual_scale, seed = load_connected_observations(
        nodes, g
    )
    residual = quiet_median(
        observation_residual(heights, design, normal_weight, seed, missing=np.nan),
        2,
    )
    values = []
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            distance_m = np.linalg.norm(g["xy"][nodes[a]] - g["xy"][nodes[b]])
            if not min_distance <= distance_m <= max_distance:
                continue
            paired_dates = np.isfinite(residual[:, a] + residual[:, b])
            if paired_dates.sum() < minimum_dates:
                continue

            def standard(x):
                x = x - np.median(x)
                return np.clip(x / max(np.median(abs(x)) / 0.6745, 1e-12), -3, 3)

            x = standard(residual[paired_dates, a])
            y = standard(residual[paired_dates, b])
            if min(np.std(x), np.std(y)) < 1e-8:
                continue
            values.append(float(np.corrcoef(x, y)[0, 1]))
    assert values, "No supported covariance calibration pairs."
    return dict(
        rho=float(np.clip(np.median(values), 0, 0.8)),
        pairs=len(values),
        cell_ids=g["cell_id"][nodes].tolist(),
        correlation_p05_p50_p95=np.percentile(values, [5, 50, 95]).tolist(),
        qualification=(
            f"Working covariance from {sample_cells} geometry-selected cells, "
            f"{min_distance / 1000:g}–{max_distance / 1000:g} km separation and "
            f"at least {minimum_dates} paired residual dates."
        ),
    )


def prepare_connected_system():
    """Write eligible nodes, candidate links and the frozen calibration sample."""
    GLS_ROOT.mkdir(exist_ok=True, parents=True)
    for name in ["nodes", "edges", "logs", "receipts"]:
        (GLS_ROOT / name).mkdir(exist_ok=True)
    path = GLS_ROOT / "metadata.h5"
    if (
        path.exists()
        and (GLS_ROOT / "rho.json").exists()
        and (GLS_ROOT / "prepared.json").exists()
    ):
        with h5py.File(path) as handle:
            assert handle.attrs["complete"] == 1
        return
    geometry_info = connected_geometry()
    nodes = geometry_info["nodes"]
    pairs = neighbour_pairs(
        geometry_info["cell_id"][nodes], geometry_info["nx"]
    ).astype("i4")
    sample_cells = min(GLS_SETTINGS["lambda_sample_cells"], len(nodes))
    calibration_nodes = np.linspace(0, len(nodes) - 1, sample_cells).round().astype("i4")
    with h5py.File(path.with_suffix(".tmp.h5"), "w") as handle:
        ds(handle, "global_index", nodes)
        ds(handle, "cell_id", geometry_info["cell_id"][nodes])
        ds(handle, "pairs", pairs)
        ds(handle, "calibration_index", calibration_nodes)
        handle.attrs.update(complete=1, node_parts=NODE_PARTS, edge_batch=EDGE_BATCH)
    path.with_suffix(".tmp.h5").replace(path)
    rho = rho_calibration(geometry_info)
    dump(GLS_ROOT / "rho.json", rho)
    dump(
        GLS_ROOT / "prepared.json",
        dict(
            domain=len(geometry_info["cell_id"]),
            eligible=len(nodes),
            fixed_cells=0,
            candidate_edges=len(pairs),
            edge_parts=(len(pairs) + EDGE_BATCH - 1) // EDGE_BATCH,
            node_parts=NODE_PARTS,
            rho=rho["rho"],
            calibration_cells=len(calibration_nodes),
            source=str(INDEPENDENT_DIR),
            model="FOUR/TIME/own100mstatic; all-free connected original-height error likelihood.",
        ),
    )
    print(
        "CONNECTED_FULL_PREPARED",
        json.dumps(json.loads((GLS_ROOT / "prepared.json").read_text())),
        flush=True,
    )


def build_node_part(index):
    """Build one independent normal-equation block per selected cell."""
    path = GLS_ROOT / "nodes" / f"{index:03d}.h5"
    if path.exists():
        with h5py.File(path) as output:
            assert output.attrs["complete"] == 1
        return
    geometry_info = connected_geometry()
    rho = json.loads((GLS_ROOT / "rho.json").read_text())["rho"]
    with h5py.File(GLS_ROOT / "metadata.h5") as output:
        nodes = output["global_index"][:]
    part_count = node_part_count()
    start_node = len(nodes) * index // part_count
    stop_node = len(nodes) * (index + 1) // part_count
    chosen = nodes[start_node:stop_node]
    node_count = len(chosen)
    sums = {k: np.zeros(NT) for k in ["sumq", "sumy", "sumy_clipped"]}
    start = time.monotonic()
    with h5py.File(path.with_suffix(".tmp.h5"), "w") as output:
        ds(output, "index", np.arange(start_node, stop_node))
        ds(output, "global_index", chosen)
        for name, shape in [
            ("H0", (node_count, P, P)),
            ("F", (node_count, P, NT)),
            ("r0", (node_count, P)),
            ("seed", (node_count, P)),
        ]:
            output.create_dataset(
                name,
                shape,
                dtype="f8",
                compression="gzip",
                compression_opts=2,
                shuffle=True,
            )
        for chunk_start in range(0, node_count, 128):
            node_ids = chosen[chunk_start : chunk_start + 128]
            heights, design, normal_weight, residual_scale, seed = (
                load_connected_observations(node_ids, geometry_info)
            )
            (
                normal_block,
                _,
                clock_loading,
                block_rhs,
                clock_precision_sum,
                clock_residual_sum,
            ) = observation_blocks(
                heights, design, normal_weight, residual_scale, rho, seed
            )
            output["H0"][chunk_start : chunk_start + len(node_ids)] = normal_block
            output["F"][chunk_start : chunk_start + len(node_ids)] = clock_loading
            output["r0"][chunk_start : chunk_start + len(node_ids)] = block_rhs
            output["seed"][chunk_start : chunk_start + len(node_ids)] = seed
            residual = observation_residual(heights, design, normal_weight, seed)
            precision = normal_weight / (
                residual_scale[None, :, None] ** 2 * (1 - rho * normal_weight)
            )
            clipped = np.clip(
                residual,
                -4 * residual_scale[None, :, None],
                4 * residual_scale[None, :, None],
            )
            sums["sumq"] += clock_precision_sum
            sums["sumy"] += clock_residual_sum
            sums["sumy_clipped"] += (
                precision * clipped * np.sqrt(rho) * residual_scale[None, :, None]
            ).sum((1, 2))
        for k, values in sums.items():
            ds(output, k, values)
        output.attrs.update(
            complete=1, seconds=time.monotonic() - start, independent_refits=0
        )
    path.with_suffix(".tmp.h5").replace(path)
    print(
        "CONNECTED_FULL_NODE_DONE",
        index,
        node_count,
        round(time.monotonic() - start, 1),
        flush=True,
    )


def build_edge_part(index):
    """Fit fine-observation differences and save their pair equation blocks."""
    path = GLS_ROOT / "edges" / f"{index:04d}.h5"
    if path.exists():
        with h5py.File(path) as handle:
            assert handle.attrs["complete"] == 1
        return
    geometry_info = connected_geometry()
    with h5py.File(GLS_ROOT / "metadata.h5") as handle:
        a = index * EDGE_BATCH
        b = min(a + EDGE_BATCH, len(handle["pairs"]))
        if a >= b:
            return
        pairs = handle["pairs"][a:b]
        nodes = handle["global_index"][:]
        cal = handle["calibration_index"][:]
    n = len(pairs)
    start = time.monotonic()
    cal_records = []
    with h5py.File(path.with_suffix(".tmp.h5"), "w") as handle:
        ds(handle, "pairs", pairs)
        handle.create_dataset("eligible", (n,), dtype="?")
        for name, shape in [
            ("Hi", (n, P, P)),
            ("Hj", (n, P, P)),
            ("off", (n, P, P)),
            ("ri", (n, P)),
            ("rj", (n, P)),
        ]:
            handle.create_dataset(
                name,
                shape,
                dtype="f8",
                compression="gzip",
                compression_opts=2,
                shuffle=True,
            )
        for chunk_start in range(0, n, 64):
            edge_pairs = pairs[chunk_start : chunk_start + 64]
            union = np.unique(edge_pairs)
            local_pairs = np.searchsorted(union, edge_pairs)
            i, j = local_pairs.T
            heights, design, normal_weight, residual_scale, seed = (
                load_connected_observations(nodes[union], geometry_info)
            )
            observed = (
                (normal_weight[:, i] > 0)
                & (normal_weight[:, j] > 0)
                & np.isfinite(heights[:, i] + heights[:, j])
            )
            observed &= (observed.sum(0) >= 2)[None]
            height_difference = heights[:, i] - heights[:, j]
            average_design = (design[i] + design[j]) / 2
            clock_difference = np.einsum(
                "etp,ep->te", design[i] - design[j], (seed[i, :9] + seed[j, :9]) / 2
            )
            pair_fit = fit_huber(
                height_difference - clock_difference[:, :, None],
                average_design,
                observed.astype(float),
            )
            accepted = np.isfinite(pair_fit["beta"]).all(0)
            scale = np.maximum(pair_fit["scale"], GLS_SETTINGS["pair_scatter_floor_m"])
            weight = np.nan_to_num(
                np.where(
                    observed & accepted[None, :, None],
                    pair_fit["normal_weight"] / scale[None, :, None] ** 2,
                    0,
                )
            )
            residual = observation_residual(heights, design, normal_weight, seed)
            hi, hj, off, ri, rj = fine_blocks(design, local_pairs, weight, residual)
            for name, value in [
                ("Hi", hi),
                ("Hj", hj),
                ("off", off),
                ("ri", ri),
                ("rj", rj),
            ]:
                handle[name][chunk_start : chunk_start + len(edge_pairs)] = value
            handle["eligible"][chunk_start : chunk_start + len(edge_pairs)] = accepted
            selected = np.flatnonzero(
                np.isin(edge_pairs[:, 0], cal) | np.isin(edge_pairs[:, 1], cal)
            )
            if len(selected):
                clip = np.clip(
                    residual,
                    -4 * residual_scale[None, :, None],
                    4 * residual_scale[None, :, None],
                )
                delta = clip[:, i[selected]] - clip[:, j[selected]]
                cal_records.append(
                    (
                        chunk_start + selected,
                        weight[:, selected].transpose(1, 0, 2),
                        (weight[:, selected] * delta).transpose(1, 0, 2),
                    )
                )
            if chunk_start % 2048 == 0:
                print(
                    "CONNECTED_FULL_EDGE_PROGRESS",
                    index,
                    chunk_start,
                    n,
                    round(time.monotonic() - start, 1),
                    flush=True,
                )
        for k, column in [("cal_edge", 0), ("cal_diagonal", 1), ("cal_gradient", 2)]:
            value = (
                np.concatenate([r[column] for r in cal_records])
                if cal_records
                else np.empty((0,) if column == 0 else (0, NT, 9))
            )
            ds(handle, k, value)
        handle.attrs.update(
            complete=1, start=a, stop=b, seconds=time.monotonic() - start
        )
    path.with_suffix(".tmp.h5").replace(path)
    print(
        "CONNECTED_FULL_EDGE_DONE",
        index,
        n,
        round(time.monotonic() - start, 1),
        flush=True,
    )


def assemble_connected_system():
    """Balance accepted links by degree and calibrate their common strength."""
    path = GLS_ROOT / "assembly.h5"
    if path.exists():
        with h5py.File(path) as handle:
            assert handle.attrs["complete"] == 1
        return
    geometry_info = connected_geometry()
    rho = json.loads((GLS_ROOT / "rho.json").read_text())["rho"]
    with h5py.File(GLS_ROOT / "metadata.h5") as handle:
        nodes = handle["global_index"][:]
        pairs = handle["pairs"][:]
        calibration_nodes = handle["calibration_index"][:]
    sums = {k: np.zeros(NT) for k in ["sumq", "sumy", "sumy_clipped"]}
    seen = np.zeros(len(nodes), bool)
    for k in range(node_part_count()):
        with h5py.File(GLS_ROOT / "nodes" / f"{k:03d}.h5") as handle:
            assert handle.attrs["complete"] == 1
            idx = handle["index"][:]
            assert not seen[idx].any()
            seen[idx] = True
            for name in sums:
                sums[name] += handle[name][:]
    assert seen.all()
    eligible = np.zeros(len(pairs), bool)
    for k in range((len(pairs) + EDGE_BATCH - 1) // EDGE_BATCH):
        with h5py.File(GLS_ROOT / "edges" / f"{k:04d}.h5") as handle:
            assert handle.attrs["complete"] == 1
            eligible[int(handle.attrs["start"]) : int(handle.attrs["stop"])] = handle[
                "eligible"
            ][:]
    degree = np.bincount(pairs[eligible].ravel(), minlength=len(nodes))
    calibration_diagonal = np.zeros((NT, len(calibration_nodes), 9))
    calibration_gradient = np.zeros_like(calibration_diagonal)
    calibration_lookup = np.full(len(nodes), -1, int)
    calibration_lookup[calibration_nodes] = np.arange(len(calibration_nodes))
    for k in range((len(pairs) + EDGE_BATCH - 1) // EDGE_BATCH):
        with h5py.File(GLS_ROOT / "edges" / f"{k:04d}.h5") as handle:
            edge_pairs = handle["pairs"][:]
            pick = handle["cal_edge"][:].astype(int)
            if not len(pick):
                continue
            edge_pairs = edge_pairs[pick]
            i, j = edge_pairs.T
            balance = 1 / np.sqrt(np.maximum(degree[i] * degree[j], 1))
            edge_diagonal = handle["cal_diagonal"][:] * balance[:, None, None]
            edge_gradient = handle["cal_gradient"][:] * balance[:, None, None]
            for endpoint, sign in [(i, 1), (j, -1)]:
                good = calibration_lookup[endpoint] >= 0
                for ix, diag, grad in zip(
                    calibration_lookup[endpoint[good]],
                    edge_diagonal[good],
                    edge_gradient[good],
                ):
                    calibration_diagonal[:, ix] += diag
                    calibration_gradient[:, ix] += sign * grad
    heights, design, normal_weight, residual_scale, seed = load_connected_observations(
        nodes[calibration_nodes], geometry_info
    )
    residual = observation_residual(heights, design, normal_weight, seed)
    precision = normal_weight / (
        residual_scale[None, :, None] ** 2 * (1 - rho * normal_weight)
    )
    loadings = np.sqrt(rho) * residual_scale
    den = 1 + sums["sumq"]
    clip = np.clip(
        residual, -4 * residual_scale[None, :, None], 4 * residual_scale[None, :, None]
    )
    a = precision - (precision * loadings[None, :, None]) ** 2 / den[:, None, None]
    b = (
        precision * clip
        - precision
        * loadings[None, :, None]
        * sums["sumy_clipped"][:, None, None]
        / den[:, None, None]
    )
    calibration = calibrate_neighbour_weight(
        a, b, calibration_diagonal, calibration_gradient, precision > 0
    )
    assert (
        calibration["optimization_success"] and not calibration["at_upper_bound"]
    ), calibration
    dump(
        GLS_ROOT / "calibration.json",
        dict(
            calibration,
            rho_global=rho,
            geometry_calibration_cell_ids=geometry_info["cell_id"][
                nodes[calibration_nodes]
            ].tolist(),
            qualification=(
                f"Conditional fine-row objective on {len(calibration_nodes)} geometry-selected "
                "nodes and their accepted neighbours. The absolute covariance "
                "denominators use all eligible cells."
            ),
        ),
    )
    with h5py.File(path.with_suffix(".tmp.h5"), "w") as handle:
        for k, v in dict(degree=degree, eligible=eligible, **sums).items():
            ds(handle, k, v)
        handle.attrs.update(complete=1, lambda_local=calibration["lambda_local"])
    path.with_suffix(".tmp.h5").replace(path)
    print(
        "CONNECTED_FULL_ASSEMBLED",
        json.dumps(calibration),
        int(eligible.sum()),
        flush=True,
    )


def block_system(hd, F, rhs, sumq, pairs, off, checkpoint=False):
    """Same normal equations in diagonally equilibrated parameter coordinates.

    x = S z, with positive diagonal S, gives (S A S) z = S b.
    This is a congruence transformation, not a change to the likelihood,
    observation weights, parameter space, or absolute tidal information.
    In particular, a few tiny residual scales must not dominate CG stopping.
    Checkpoints and the public operator/solve interface use physical x.
    """
    # hd: diagonal cell blocks [cell, 18, 18].
    # F: shared-clock loading [cell, 18, date]; off: accepted edge blocks.
    # rhs and exported/checkpoint vectors use physical parameter coordinates.
    cell_count = len(hd)
    date_count = F.shape[2]
    edge_count = len(pairs)
    rows = np.r_[np.arange(cell_count), pairs[:, 0], pairs[:, 1]]
    cols = np.r_[np.arange(cell_count), pairs[:, 1], pairs[:, 0]]
    scale = 1 / np.sqrt(np.diagonal(hd, axis1=1, axis2=2))
    assert np.isfinite(scale).all() and (scale > 0).all()
    for start in range(0, cell_count, 4096):
        chunk_scale = scale[start : start + 4096]
        hd[start : start + len(chunk_scale)] *= (
            chunk_scale[:, :, None] * chunk_scale[:, None, :]
        )
        F[start : start + len(chunk_scale)] *= chunk_scale[:, :, None]
    order = np.argsort(rows, kind="stable")
    destination = np.empty(len(rows), int)
    destination[order] = np.arange(len(rows))
    row_pointer = np.r_[0, np.cumsum(np.bincount(rows, minlength=cell_count))].astype(
        "i4"
    )
    data = np.empty((len(rows), P, P))
    data[destination[:cell_count]] = hd
    for start in range(0, edge_count, 4096):
        i, j = pairs[start : start + 4096].T
        z = off[start : start + 4096] * scale[i, :, None] * scale[j, None, :]
        data[destination[cell_count + start : cell_count + start + len(z)]] = z
        data[
            destination[
                cell_count
                + edge_count
                + start : cell_count
                + edge_count
                + start
                + len(z)
            ]
        ] = z.transpose(0, 2, 1)
    sparse_matrix = bsr_matrix(
        (data, cols[order].astype("i4"), row_pointer),
        shape=(cell_count * P, cell_count * P),
        copy=False,
    )
    sparse_matrix.sort_indices()
    assert np.shares_memory(sparse_matrix.data, data)
    del data, order, destination, rows, cols, off
    inverse_blocks = np.empty_like(hd)
    inverse_clock_loading = np.empty_like(F)
    clock_system = np.diag(1 + sumq)
    for start in range(0, cell_count, 4096):
        block_chunk = hd[start : start + 4096]
        norm = np.sqrt(np.diagonal(block_chunk, axis1=1, axis2=2))
        scaled_block = block_chunk / norm[:, :, None] / norm[:, None, :]
        np.linalg.cholesky(scaled_block)
        inverse_blocks[start : start + len(block_chunk)] = (
            np.linalg.inv(scaled_block) / norm[:, :, None] / norm[:, None, :]
        )
        inverse_loading_chunk = np.einsum(
            "nij,njt->nit",
            inverse_blocks[start : start + len(block_chunk)],
            F[start : start + len(block_chunk)],
            optimize=True,
        )
        inverse_clock_loading[start : start + len(block_chunk)] = inverse_loading_chunk
        clock_system -= F[start : start + len(block_chunk)].reshape(
            -1, date_count
        ).T @ inverse_loading_chunk.reshape(-1, date_count)
    del hd
    clock_system = (clock_system + clock_system.T) / 2
    norm = np.sqrt(np.diag(clock_system))
    factor = cho_factor(clock_system / norm[:, None] / norm[None, :], lower=True)
    clock_loading_flat = F.reshape(-1, date_count)
    inverse_loading_flat = inverse_clock_loading.reshape(-1, date_count)
    den = 1 + sumq

    def small(x):
        return cho_solve(factor, x / norm) / norm

    def mv(x):
        return sparse_matrix @ x - clock_loading_flat @ (
            (clock_loading_flat.T @ x) / den
        )

    def pre(x):
        return np.einsum(
            "nij,nj->ni", inverse_blocks, x.reshape(cell_count, P), optimize=True
        ).ravel() + inverse_loading_flat @ small(inverse_loading_flat.T @ x)

    scaled_operator = LinearOperator(sparse_matrix.shape, matvec=mv, dtype=float)
    preconditioner = LinearOperator(sparse_matrix.shape, matvec=pre, dtype=float)
    flat_scale = scale.ravel()
    physical_operator = LinearOperator(
        sparse_matrix.shape,
        matvec=lambda x: mv(x / flat_scale) / flat_scale,
        dtype=float,
    )

    def save_checkpoint(x, iteration, tolerance, label):
        temporary = GLS_ROOT / "checkpoint.tmp.npy"
        with temporary.open("wb") as checkpoint_file:
            np.save(checkpoint_file, x)
        temporary.replace(GLS_ROOT / "checkpoint.npy")
        dump(
            GLS_ROOT / "checkpoint.json",
            dict(
                iteration=iteration,
                tolerance=tolerance,
                parameters=len(x),
                coordinates="physical",
                label=label,
                solver="diagonally equilibrated unchanged normal equations",
            ),
        )

    def solve(b, x0=None, label="main", tolerance=1e-11):
        iteration = [0]
        started = time.monotonic()
        scaled_b = b * flat_scale
        initial = None if x0 is None else x0 / flat_scale

        def callback(x):
            iteration[0] += 1
            if iteration[0] % 25 == 0:
                relative = float(
                    np.linalg.norm(mv(x) - scaled_b)
                    / max(np.linalg.norm(scaled_b), 1e-30)
                )
                print(
                    "CONNECTED_FULL_CG_SCALED",
                    label,
                    iteration[0],
                    round(time.monotonic() - started, 1),
                    "scaled_residual",
                    relative,
                    flush=True,
                )
                if checkpoint and label == "main":
                    save_checkpoint(x * flat_scale, iteration[0], tolerance, label)

        z, status = cg(
            scaled_operator,
            scaled_b,
            x0=initial,
            M=preconditioner,
            rtol=tolerance,
            atol=0,
            maxiter=3000,
            callback=callback,
        )
        x = z * flat_scale
        residual = mv(z) - scaled_b
        scaled_defect = float(
            np.linalg.norm(residual) / max(np.linalg.norm(scaled_b), 1e-30)
        )
        defect = float(
            np.linalg.norm(residual / flat_scale) / max(np.linalg.norm(b), 1e-12)
        )
        assert (
            status == 0 and defect < NORMAL_RESIDUAL_MAX and scaled_defect < max(2 * tolerance, 1e-14)
        ), (status, defect, scaled_defect)
        if checkpoint and label.startswith("main"):
            save_checkpoint(x, iteration[0], tolerance, label)
        return x, dict(
            iterations=iteration[0],
            normal_relative_residual=defect,
            scaled_normal_relative_residual=scaled_defect,
            tolerance=tolerance,
            seconds=time.monotonic() - started,
        )

    def refine(b, x, label):
        # Solve the freshly recomputed residual equation, so the large original
        # RHS does not set the stopping scale for the small remaining error.
        remaining = (b * flat_scale - mv(x / flat_scale)) / flat_scale
        correction, details = solve(remaining, label=label, tolerance=1e-11)
        value = x + correction
        residual = mv(value / flat_scale) - b * flat_scale
        details.update(
            normal_relative_residual=float(
                np.linalg.norm(residual / flat_scale) / max(np.linalg.norm(b), 1e-12)
            ),
            scaled_normal_relative_residual=float(
                np.linalg.norm(residual) / max(np.linalg.norm(b * flat_scale), 1e-30)
            ),
            maximum_correction_m=float(np.max(abs(correction))),
            residual_equation_refinement=True,
        )
        assert details["normal_relative_residual"] < NORMAL_RESIDUAL_MAX, details
        if checkpoint and label.startswith("main"):
            save_checkpoint(value, details["iterations"], details["tolerance"], label)
        return value, details

    solve.refine = refine
    return (
        physical_operator,
        solve,
        dict(
            sparse_bytes=int(
                sparse_matrix.data.nbytes
                + sparse_matrix.indices.nbytes
                + sparse_matrix.indptr.nbytes
            ),
            coarse_condition=float(
                np.linalg.cond(clock_system / norm[:, None] / norm[None, :])
            ),
            numerical_coordinates="x=S z, S=diag(Hblock)^(-1/2); physical checkpoints; unchanged observation model",
            coordinate_scale_min_max=[float(scale.min()), float(scale.max())],
        ),
    )


def solve_connected_system():
    """Solve the assembled system and verify a full-parameter injected signal.

    Export only after both the physical residual and signal-preservation checks pass.
    No tide coefficient, mean, grounded cell or unsupported hole is filled or fixed.
    """
    path = GLS_ROOT / "native_tides.h5"
    if path.exists():
        with h5py.File(path) as handle:
            assert handle.attrs["complete"] == 1
        return
    started = time.monotonic()
    geometry_info = connected_geometry()
    print("CONNECTED_FULL_PHASE loading_prepared_node_blocks", flush=True)
    with h5py.File(GLS_ROOT / "metadata.h5") as handle:
        nodes = handle["global_index"][:]
        allpairs = handle["pairs"][:]
    with h5py.File(GLS_ROOT / "assembly.h5") as handle:
        assert handle.attrs["complete"] == 1
        degree = handle["degree"][:]
        eligible = handle["eligible"][:]
        sumq = handle["sumq"][:]
        sumy = handle["sumy"][:]
        pair_strength = float(handle.attrs["lambda_local"])
    n = len(nodes)
    normal_blocks = np.empty((n, P, P))
    clock_loading = np.empty((n, P, NT))
    rhs = np.empty((n, P))
    seed = np.empty((n, P))
    for k in range(node_part_count()):
        with h5py.File(GLS_ROOT / "nodes" / f"{k:03d}.h5") as handle:
            at = handle["index"][:]
            for a in range(0, len(at), 1024):
                q = at[a : a + 1024]
                sl = slice(a, a + len(q))
                normal_blocks[q] = handle["H0"][sl]
                clock_loading[q] = handle["F"][sl]
                rhs[q] = handle["r0"][sl]
                seed[q] = handle["seed"][sl]
    inactive = np.diagonal(normal_blocks, axis1=1, axis2=2) <= 1e-14
    for i, p in zip(*np.where(inactive)):
        normal_blocks[i, p, p] = 1
    pairs = allpairs[eligible]
    edge_count = len(pairs)
    off = np.empty((edge_count, P, P))
    at = 0
    print("CONNECTED_FULL_PHASE loading_prepared_edge_blocks", flush=True)
    for k in range((len(allpairs) + EDGE_BATCH - 1) // EDGE_BATCH):
        with h5py.File(GLS_ROOT / "edges" / f"{k:04d}.h5") as handle:
            ep = handle["pairs"][:]
            valid = handle["eligible"][:]
            positions = np.flatnonzero(valid)
            for a in range(0, len(positions), 256):
                pick = positions[a : a + 256]
                i, j = ep[pick].T
                scale = pair_strength / np.sqrt(degree[i] * degree[j])
                hi = handle["Hi"][pick] * scale[:, None, None]
                hj = handle["Hj"][pick] * scale[:, None, None]
                np.add.at(normal_blocks, i, hi)
                np.add.at(normal_blocks, j, hj)
                np.add.at(rhs, i, handle["ri"][pick] * scale[:, None])
                np.add.at(rhs, j, handle["rj"][pick] * scale[:, None])
                off[at : at + len(pick)] = handle["off"][pick] * scale[:, None, None]
                np.testing.assert_array_equal(pairs[at : at + len(pick)], ep[pick])
                at += len(pick)
    assert at == edge_count
    rhs = rhs.ravel() - clock_loading.reshape(n * P, NT) @ (sumy / (1 + sumq))
    print(
        "CONNECTED_FULL_PHASE building_scaled_operator_and_preconditioner", flush=True
    )
    operator, solve_system, info = block_system(
        normal_blocks, clock_loading, rhs, sumq, pairs, off, True
    )
    del normal_blocks, off, clock_loading
    initial = None
    if (GLS_ROOT / "checkpoint.npy").exists():
        with (GLS_ROOT / "checkpoint.npy").open("rb") as handle:
            initial = np.load(handle)
        assert initial.shape == rhs.shape
    delta, main = solve_system(rhs, initial)
    info.update(main)
    dump(
        GLS_ROOT / "main_numerical_convergence.json",
        dict(
            **info,
            status="MAIN_NUMERICAL_ONLY_SIGNAL_CONTROL_PENDING_NOT_A_TIDAL_PRODUCT",
        ),
    )
    injected_signal = np.zeros((n, P))
    distance_from_centre = np.linalg.norm(
        geometry_info["xy"][nodes] - np.median(geometry_info["xy"][nodes], axis=0),
        axis=1,
    )
    injected_signal[:, 0] = 0.15 + 0.2 * np.exp(
        -0.5 * (distance_from_centre / 750) ** 2
    )
    injected_signal[:, 1] = -0.08
    injected_signal[:, 2] = 0.1 * np.exp(-0.5 * (distance_from_centre / 1500) ** 2)
    injected_signal[:, 3] = 0.06
    injected_signal[:, 4] = 0.08 + 0.1 * np.exp(
        -0.5 * (distance_from_centre / 2000) ** 2
    )
    injected_signal[:, 5] = -0.03
    injected_signal[:, 6] = -0.06 + 0.12 * np.exp(
        -0.5 * (distance_from_centre / 1000) ** 2
    )
    injected_signal[:, 7] = 0.04
    injected_signal[:, 8] = 0.05
    injected_signal[:, 9:] = 0.1 * np.sin(
        np.arange(n)[:, None] * 0.7 + np.arange(9)[None]
    )
    injected_signal[np.isnan(seed)] = 0
    translated, signal_check = solve_system(
        rhs + operator @ injected_signal.ravel(), label="signal_control"
    )
    signal_error = float(np.max(abs(translated - delta - injected_signal.ravel())))
    if signal_error >= SIGNAL_ERROR_MAX_M:
        delta, main = solve_system(rhs, delta, label="main_refine", tolerance=1e-14)
        translated, signal_check = solve_system(
            rhs + operator @ injected_signal.ravel(),
            translated,
            label="signal_refine",
            tolerance=1e-14,
        )
        signal_error = float(np.max(abs(translated - delta - injected_signal.ravel())))
        info.update(main)
    refinements = []
    for iteration in range(2):
        if signal_error < SIGNAL_ERROR_MAX_M:
            break
        delta, main = solve_system.refine(
            rhs, delta, label="main_residual_refine_" + str(iteration)
        )
        translated, signal_check = solve_system.refine(
            rhs + operator @ injected_signal.ravel(),
            translated,
            label="signal_residual_refine_" + str(iteration),
        )
        signal_error = float(np.max(abs(translated - delta - injected_signal.ravel())))
        refinements.append(dict(error_m=signal_error, main=main, signal=signal_check))
        info.update(main)
    control = dict(
        passed=signal_error < SIGNAL_ERROR_MAX_M,
        maximum_error_m=signal_error,
        required_maximum_error_m=SIGNAL_ERROR_MAX_M,
        main=main,
        signal=signal_check,
        residual_refinements=refinements,
    )
    if signal_error >= SIGNAL_ERROR_MAX_M:
        difference = (translated - delta - injected_signal.ravel()).reshape(n, P)
        at = np.unravel_index(np.argmax(abs(difference)), difference.shape)
        control.update(
            worst_global_index=int(nodes[at[0]]),
            worst_parameter=int(at[1]),
            row_error_p50_p95_p99_p100=np.percentile(
                np.max(abs(difference), axis=1), [50, 95, 99, 100]
            ).tolist(),
        )
    dump(GLS_ROOT / "full_signal_control.json", control)
    assert signal_error < SIGNAL_ERROR_MAX_M, control
    assert_sources()
    value = seed + delta.reshape(n, P)
    value[np.isnan(seed)] = np.nan
    info.update(
        fixed_cells=0,
        eligible_cells=n,
        accepted_links=edge_count,
        lambda_local=pair_strength,
        true_four_TIME_static_translation_error_m=signal_error,
        all_date_experimental=True,
        accuracy_promotion=False,
        source_acquisitions=NT,
        marginal_condition="Not assigned from original independent model; field is numerical experimental output, not new gate-accepted product.",
        total_seconds=time.monotonic() - started,
    )
    with h5py.File(path.with_suffix(".tmp.h5"), "w") as handle:
        for k in [
            "cell_id",
            "global_x",
            "global_y",
            "lat",
            "lon",
            "time_ns",
            "reference_seconds",
            "passes",
        ]:
            ds(handle, k, geometry_info[k])
        with h5py.File(INDEPENDENT_DIR / "geometry.h5") as old:
            for k in ["x100", "y100"]:
                ds(handle, k, old[k][:])
            handle.create_dataset(
                "fine_id",
                old["fine_id"].shape,
                dtype=old["fine_id"].dtype,
                compression="gzip",
                compression_opts=2,
            )
            for a in range(0, len(geometry_info["cell_id"]), 8192):
                handle["fine_id"][a : a + 8192] = old["fine_id"][a : a + 8192]
        for k, shape in [
            ("beta", (9, len(geometry_info["cell_id"]))),
            ("terrain", (len(geometry_info["cell_id"]), 9)),
        ]:
            handle.create_dataset(
                k,
                shape,
                dtype="f8",
                fillvalue=np.nan,
                compression="gzip",
                compression_opts=2,
            )
        for a in range(0, n, 8192):
            q = nodes[a : a + 8192]
            handle["beta"][:, q] = value[a : a + len(q), :9].T
            handle["terrain"][q] = value[a : a + len(q), 9:]
        handle.attrs.update(
            complete=1,
            constituents="M2 S2 K1 O1",
            fixed_cells=0,
            solver_json=json.dumps(info),
            qualification="User-requested full-LCIS experimental no-fixed-pixel original-height likelihood. Prior central predictive gate remains unresolved. No new covariance-conditioning acceptance or calibrated uncertainty claimed.",
        )
    path.with_suffix(".tmp.h5").replace(path)
    dump(GLS_ROOT / "solver.json", info)
    print("CONNECTED_FULL_SOLVED", json.dumps(info), flush=True)


# Extract fitted maps and transects


AUX = CODE / "aux_files"


def extract_transect(native, independent_dir, output_dir):
    """Build the frozen Figure 3 corridor with the same cells and dates.

    Subtract each fitted fine-cell static level, then add the centreline level.
    """
    line = json.loads((AUX / "figure3_line.json").read_text())
    with h5py.File(AUX / "figure3_sampling.h5") as handle:
        sampling = {
            k: handle[k][:]
            for k in [
                "fine_id",
                "native_cell_id",
                "bin",
                "accepted",
                "acquisition_index",
            ]
        }
    if np.any(sampling["acquisition_index"] >= NT):
        print(
            "Skipping Figure 3: the selected catalogue omits required transect dates.",
            flush=True,
        )
        return
    cells = np.unique(np.r_[sampling["native_cell_id"], line["nearest_cell_id"]])
    cells = cells[cells >= 0]
    with h5py.File(independent_dir / "geometry.h5") as handle:
        all_cell_ids = handle["cell_id"][:]
        positions = np.searchsorted(all_cell_ids, cells)
        np.testing.assert_array_equal(all_cell_ids[positions], cells)
        fine_mapping = handle["fine_id"][positions]
        cats = handle["cats_beta"][:, positions]
        lat = handle["lat"][positions]
        reference_seconds = handle["reference_seconds"][:NT]
        bands = handle["source_band"][positions]
        local = handle["source_local"][positions]
        time_ns = handle["time_ns"][:NT]
    with h5py.File(native) as handle:
        assert handle.attrs["complete"] == 1
        np.testing.assert_array_equal(handle["cell_id"][positions], cells)
        coefficients = handle["beta"][:, positions]
        static_levels = handle["terrain"][positions]
    dates = sampling["acquisition_index"]
    heights = np.full((len(dates), len(cells), 9), np.nan, "f4")
    when = np.full((len(reference_seconds), len(cells)), np.nan)
    for band in np.unique(bands):
        destination = np.flatnonzero(bands == band)
        local_index = local[destination]
        with h5py.File(independent_dir / "inputs" / f"{int(band):03d}.h5") as handle:
            reference = handle["reference_geoid"][local_index]
            for k, t in enumerate(dates):
                height = handle["height"][int(t), local_index]
                admitted = admission_mask(height[None], reference)[0]
                heights[k, destination] = np.where(admitted, height, np.nan)
            when[:, destination] = handle["time_seconds"][:, local_index]
    time_design = tide_and_trend_basis(
        when.T, lat, reference_seconds, independent_dir / "geometry.h5"
    )
    swot_prediction = np.einsum("ntp,pn->tn", time_design[:, :, :8], coefficients[:8])[
        dates
    ]
    cats_prediction = np.einsum("ntp,pn->tn", time_design[:, :, :8], cats)[dates]
    flat = np.flatnonzero(fine_mapping.ravel() >= 0)
    flat = flat[np.argsort(fine_mapping.ravel()[flat])]
    positions_in_fine = np.searchsorted(fine_mapping.ravel()[flat], sampling["fine_id"])
    np.testing.assert_array_equal(
        fine_mapping.ravel()[flat][positions_in_fine], sampling["fine_id"]
    )
    rows, slots = np.divmod(flat[positions_in_fine], 9)
    centre = np.searchsorted(cells, line["nearest_cell_id"])
    hit = np.asarray(line["nearest_cell_id"]) >= 0
    np.testing.assert_array_equal(
        cells[centre[hit]], np.asarray(line["nearest_cell_id"])[hit]
    )
    centre = np.clip(centre, 0, len(cells) - 1)
    level = quiet_median(static_levels[centre], 1)
    level[~hit] = np.nan
    shape = (len(dates), len(line["distance_km"]))
    observed = np.full(shape, np.nan)
    swot_profile = observed.copy()
    cats_profile = observed.copy()
    for date in range(len(dates)):
        for bin_index in range(shape[1]):
            selected = np.flatnonzero(
                (sampling["bin"] == bin_index) & sampling["accepted"][date]
            )
            if len(selected) < SETTINGS["minimum_transect_cells"]:
                continue
            selected_heights = heights[date, rows[selected], slots[selected]]
            if not np.isfinite(selected_heights).all():
                continue
            normalized_height = (
                selected_heights
                - static_levels[rows[selected], slots[selected]]
                + level[bin_index]
            )
            swot_tide = swot_prediction[date, rows[selected]]
            cats_tide = cats_prediction[date, rows[selected]]
            if np.isfinite(cats_tide).all():
                cats_profile[date, bin_index] = np.median(cats_tide)
            unavailable = ~np.isfinite(normalized_height + swot_tide + cats_tide)
            if unavailable.any():
                continue
            observed[date, bin_index] = np.median(normalized_height)
            swot_profile[date, bin_index] = np.median(swot_tide)
    distance = np.asarray(line["distance_km"])
    figure_settings = json.loads((AUX / "figure_settings.json").read_text())["figure3"]
    intervals = figure_settings["unsmoothed_prediction_intervals_km"]
    protect = np.zeros(len(distance), dtype=bool)
    for start, end in intervals:
        protect |= (distance >= start) & (distance <= end)

    def display(value):
        result = np.array(
            [
                pd.Series(v).rolling(5, center=True, min_periods=3).median().to_numpy()
                for v in value
            ]
        )
        result[~np.isfinite(value)] = np.nan
        result[:, protect] = value[:, protect]
        return result

    with h5py.File(output_dir / "figure3.h5", "w") as handle:
        for k, v in dict(
            time_ns=time_ns[dates],
            observed=observed,
            swot_tide=display(swot_profile),
            cats_tide=display(cats_profile),
            **{
                k: line[k]
                for k in ["distance_km", "longitude", "latitude"]
            },
        ).items():
            ds(handle, k, v)
        handle.attrs.update(
            complete=1,
            source=str(native),
            support="Pinned corridor; bins withheld when required run observations or model values are missing",
            display=f"Median5 of predicted tides only; unsmoothed within {intervals} km; no observed/residual smoothing",
            qualification="In-sample selected transect, not independent validation",
        )


def extract_maps_and_profiles(native, independent_dir, output_dir):
    """Extract maps, own-static elevations and transects from a completed native fit."""
    native, independent_dir, output_dir = (
        Path(native),
        Path(independent_dir),
        Path(output_dir),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (
        h5py.File(native) as native_file,
        h5py.File(independent_dir / "geometry.h5") as geometry_file,
    ):
        assert native_file.attrs["complete"] == 1
        np.testing.assert_array_equal(
            native_file["cell_id"][:], geometry_file["cell_id"][:]
        )
        x, y = native_file["global_x"][:], native_file["global_y"][:]
        nx, ny = len(x), len(y)
        with h5py.File(output_dir / "maps.h5", "w") as output:
            ds(output, "x", x)
            ds(output, "y", y)
            for k in ("m2", "s2"):
                for suffix in ["amp", "cats_amp"]:
                    output.create_dataset(
                        k + "_" + suffix,
                        (ny, nx),
                        dtype="f4",
                        fillvalue=np.nan,
                        chunks=(96, min(1024, nx)),
                        compression="gzip",
                    )
            for start in range(0, len(geometry_file["cell_id"]), 8192):
                ids = geometry_file["cell_id"][start : start + 8192]
                coefficients = native_file["beta"][:, start : start + 8192]
                cats_coefficients = geometry_file["cats_beta"][:, start : start + 8192]
                valid = np.isfinite(coefficients).all(0)
                for j, k in enumerate(("m2", "s2")):
                    amp = np.where(
                        valid,
                        np.hypot(coefficients[2 * j], coefficients[2 * j + 1]),
                        np.nan,
                    )
                    cats_amplitude = np.hypot(
                        cats_coefficients[2 * j], cats_coefficients[2 * j + 1]
                    )
                    # Bounded row-wise writes preserve all unsmoothed finite fits.
                    for row in np.unique(ids // nx):
                        input_band = ids // nx == row
                        cols = ids[input_band] % nx
                        output[k + "_amp"][int(row), cols] = amp[input_band]
                        output[k + "_cats_amp"][int(row), cols] = cats_amplitude[
                            input_band
                        ]
            output.attrs.update(
                complete=1,
                source=str(native),
                units="m",
                support="Finite numerical field; no new marginal conditioning or accuracy claim",
                smoothing="none",
            )
        x, y = geometry_file["x100"][:], geometry_file["y100"][:]
        nx, ny = len(x), len(y)
        with h5py.File(output_dir / "figure1.h5", "w") as output:
            ds(output, "x", x)
            ds(output, "y", y)
            for k in ["mean", "rema"]:
                output.create_dataset(
                    k,
                    (ny, nx),
                    dtype="f4",
                    fillvalue=np.nan,
                    chunks=(96, min(nx, 1024)),
                    compression="gzip",
                )
            for band in band_indices(independent_dir / "geometry.h5"):
                band_positions = np.flatnonzero(geometry_file["source_band"][:] == band)
                with h5py.File(
                    independent_dir / "inputs" / f"{band:03d}.h5"
                ) as input_band:
                    for start in range(0, len(band_positions), 4096):
                        positions = band_positions[start : start + 4096]
                        fine_mapping = geometry_file["fine_id"][positions]
                        static_levels = native_file["terrain"][positions]
                        reference = input_band["reference_geoid"][start : start + 4096]
                        valid = np.isfinite(native_file["beta"][:, positions]).all(0)
                        use = (fine_mapping >= 0) & valid[:, None]
                        ids = fine_mapping[use]
                        mean_values = static_levels[use]
                        reference_values = reference[use]
                        order = np.argsort(ids)
                        ids, mean_values, reference_values = (
                            ids[order],
                            mean_values[order],
                            reference_values[order],
                        )
                        for row in np.unique(ids // nx):
                            mask = ids // nx == row
                            cols = ids[mask] % nx
                            output["mean"][int(row), cols] = mean_values[mask]
                            output["rema"][int(row), cols] = reference_values[mask]
            output.attrs.update(
                complete=1,
                source=str(native),
                epoch_seconds_2000=SETTINGS["time_center_seconds_2000"],
                units="m above mean-tide geoid",
                method="Matching final own100m statics; TIME evaluated at the fixed reference epoch. REMA is upstream selection, not independent validation.",
            )
    extract_transect(native, independent_dir, output_dir)


if __name__ == "__main__":
    main()

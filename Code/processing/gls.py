"""Combine independent cell fits using the original connected GLS equations.

Stages: prepare -> node_part / edge_part -> assemble -> solve.
Each cell has 18 unknowns: eight tide coefficients, TIME and nine fine levels.
The cell blocks describe absolute observations; edge blocks describe observed
neighbour differences. The solve also includes the shared clock covariance.
All eligible coefficients and fine levels remain free unknowns.
"""

from pathlib import Path
import argparse
import json
import os
import time

import h5py
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import LinearOperator, cg

import analysis
import graph
from graph import quietmedian
from layout import PARTITIONS

ROOT = Path(
    os.environ.get(
        "LCIS_CONNECTED_FULL_ROOT",
        Path(__file__).resolve().parent / "CONNECTED_FINE_ERROR_FULL_LCIS_20261103",
    )
)
NODE_PARTS = PARTITIONS["gls_node_parts"]
EDGE_BATCH = 8192
P = 18
NT = analysis.NT
dump = analysis.dump
ds = analysis.ds


def node_part_count():
    """Use the part count recorded when this connected system was prepared."""
    with h5py.File(ROOT / "metadata.h5") as source:
        count = int(source.attrs["node_parts"])
    if count < 1:
        raise ValueError("Prepared node_parts must be positive")
    return count


def assert_sources():
    """Refuse a fit if any frozen prepared source has changed."""
    path = ROOT / "source_inventory.json"
    if not path.exists():
        return
    records = json.loads(path.read_text())["source_snapshots"]
    assert all(
        Path(v["path"]).stat().st_size == v["bytes"]
        and Path(v["path"]).stat().st_mtime_ns == v["mtime_ns"]
        for v in records
    ), "Frozen prepared source or baseline changed."


def geometry():
    """Read the shared geometry and select finite independent fits."""
    geometry_info = analysis.geometry()
    geometry_info["nodes"] = np.flatnonzero(geometry_info["valid"])
    return geometry_info


def load(nodes, g):
    """Read fine observations, fixed robust weights, scales and independent seeds."""
    heights, design, _, fit = analysis.load(np.asarray(nodes, int), g, weights=True)
    normal_weight = fit["normal_weight"]
    residual_scale = np.maximum(fit["scale"], 1e-8)
    seed = np.c_[fit["beta"].T, fit["terrain"]]
    return heights, design, normal_weight, residual_scale, seed


def rho_calibration(g):
    """Estimate the working shared covariance from the frozen residual-pair rule."""
    nodes = g["nodes"][np.linspace(0, len(g["nodes"]) - 1, 128).round().astype(int)]
    heights, design, normal_weight, residual_scale, seed = load(nodes, g)
    residual = quietmedian(
        np.where(
            normal_weight > 0,
            heights
            - np.einsum("ntp,np->tn", design, seed[:, :9])[:, :, None]
            - seed[None, :, 9:],
            np.nan,
        ),
        2,
    )
    values = []
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            distance_m = np.linalg.norm(g["xy"][nodes[a]] - g["xy"][nodes[b]])
            if not 5000 <= distance_m <= 40000:
                continue
            paired_dates = np.isfinite(residual[:, a] + residual[:, b])
            if paired_dates.sum() < 12:
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
        qualification="All-date working covariance from same128geometry/5-40km/12date robust-residual rule as pilot; no CATS or map tuning.",
    )


def prepare():
    """Write eligible nodes, candidate links and the frozen calibration sample."""
    ROOT.mkdir(exist_ok=True, parents=True)
    for name in ["nodes", "edges", "logs", "receipts"]:
        (ROOT / name).mkdir(exist_ok=True)
    path = ROOT / "metadata.h5"
    if (
        path.exists()
        and (ROOT / "rho.json").exists()
        and (ROOT / "prepared.json").exists()
    ):
        with h5py.File(path) as handle:
            assert handle.attrs["complete"] == 1
        return
    geometry_info = geometry()
    nodes = geometry_info["nodes"]
    pairs = graph.pairs(geometry_info["cell_id"][nodes], geometry_info["nx"]).astype(
        "i4"
    )
    calibration_nodes = (
        np.linspace(0, len(nodes) - 1, min(512, len(nodes))).round().astype("i4")
    )
    with h5py.File(path.with_suffix(".tmp.h5"), "w") as handle:
        ds(handle, "global_index", nodes)
        ds(handle, "cell_id", geometry_info["cell_id"][nodes])
        ds(handle, "pairs", pairs)
        ds(handle, "calibration_index", calibration_nodes)
        handle.attrs.update(complete=1, node_parts=NODE_PARTS, edge_batch=EDGE_BATCH)
    path.with_suffix(".tmp.h5").replace(path)
    rho = rho_calibration(geometry_info)
    dump(ROOT / "rho.json", rho)
    dump(
        ROOT / "prepared.json",
        dict(
            domain=len(geometry_info["cell_id"]),
            eligible=len(nodes),
            formerly_fixed_but_free=int(geometry_info["protected"][nodes].sum()),
            fixed_cells=0,
            candidate_edges=len(pairs),
            edge_parts=(len(pairs) + EDGE_BATCH - 1) // EDGE_BATCH,
            node_parts=NODE_PARTS,
            rho=rho["rho"],
            calibration_cells=len(calibration_nodes),
            source=str(analysis.OLD),
            model="FOUR/TIME/own100mstatic; all-free connected original-height error likelihood.",
        ),
    )
    print(
        "CONNECTED_FULL_PREPARED",
        json.dumps(json.loads((ROOT / "prepared.json").read_text())),
        flush=True,
    )


def node_part(index):
    """Build one independent normal-equation block per selected cell."""
    path = ROOT / "nodes" / f"{index:03d}.h5"
    if path.exists():
        with h5py.File(path) as output:
            assert output.attrs["complete"] == 1
        return
    geometry_info = geometry()
    rho = json.loads((ROOT / "rho.json").read_text())["rho"]
    with h5py.File(ROOT / "metadata.h5") as output:
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
            heights, design, normal_weight, residual_scale, seed = load(
                node_ids, geometry_info
            )
            (
                normal_block,
                _,
                clock_loading,
                block_rhs,
                clock_precision_sum,
                clock_residual_sum,
            ) = graph.blocks(heights, design, normal_weight, residual_scale, rho, seed)
            output["H0"][chunk_start : chunk_start + len(node_ids)] = normal_block
            output["F"][chunk_start : chunk_start + len(node_ids)] = clock_loading
            output["r0"][chunk_start : chunk_start + len(node_ids)] = block_rhs
            output["seed"][chunk_start : chunk_start + len(node_ids)] = seed
            residual = np.where(
                normal_weight > 0,
                heights
                - np.einsum("ntp,np->tn", design, seed[:, :9])[:, :, None]
                - seed[None, :, 9:],
                0.0,
            )
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


def edge_part(index):
    """Fit fine-observation differences and save their pair equation blocks."""
    path = ROOT / "edges" / f"{index:04d}.h5"
    if path.exists():
        with h5py.File(path) as handle:
            assert handle.attrs["complete"] == 1
        return
    geometry_info = geometry()
    with h5py.File(ROOT / "metadata.h5") as handle:
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
            heights, design, normal_weight, residual_scale, seed = load(
                nodes[union], geometry_info
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
            pair_fit = graph.estimator.fit(
                height_difference - clock_difference[:, :, None],
                average_design,
                observed.astype(float),
            )
            accepted = np.isfinite(pair_fit["beta"]).all(0)
            scale = np.maximum(pair_fit["scale"], 0.05)
            weight = np.nan_to_num(
                np.where(
                    observed & accepted[None, :, None],
                    pair_fit["normal_weight"] / scale[None, :, None] ** 2,
                    0,
                )
            )
            residual = np.where(
                normal_weight > 0,
                heights
                - np.einsum("ntp,np->tn", design, seed[:, :9])[:, :, None]
                - seed[None, :, 9:],
                0.0,
            )
            hi, hj, off, ri, rj = graph.fine_blocks(
                design, local_pairs, weight, residual
            )
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


def assemble():
    """Balance accepted links by degree and calibrate their common strength."""
    path = ROOT / "assembly.h5"
    if path.exists():
        with h5py.File(path) as handle:
            assert handle.attrs["complete"] == 1
        return
    geometry_info = geometry()
    rho = json.loads((ROOT / "rho.json").read_text())["rho"]
    with h5py.File(ROOT / "metadata.h5") as handle:
        nodes = handle["global_index"][:]
        pairs = handle["pairs"][:]
        calibration_nodes = handle["calibration_index"][:]
    sums = {k: np.zeros(NT) for k in ["sumq", "sumy", "sumy_clipped"]}
    seen = np.zeros(len(nodes), bool)
    for k in range(node_part_count()):
        with h5py.File(ROOT / "nodes" / f"{k:03d}.h5") as handle:
            assert handle.attrs["complete"] == 1
            idx = handle["index"][:]
            assert not seen[idx].any()
            seen[idx] = True
            for name in sums:
                sums[name] += handle[name][:]
    assert seen.all()
    eligible = np.zeros(len(pairs), bool)
    for k in range((len(pairs) + EDGE_BATCH - 1) // EDGE_BATCH):
        with h5py.File(ROOT / "edges" / f"{k:04d}.h5") as handle:
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
        with h5py.File(ROOT / "edges" / f"{k:04d}.h5") as handle:
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
    heights, design, normal_weight, residual_scale, seed = load(
        nodes[calibration_nodes], geometry_info
    )
    residual = np.where(
        normal_weight > 0,
        heights
        - np.einsum("ntp,np->tn", design, seed[:, :9])[:, :, None]
        - seed[None, :, 9:],
        0.0,
    )
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
    calibration = graph.calibration(
        a, b, calibration_diagonal, calibration_gradient, precision > 0
    )
    assert (
        calibration["optimization_success"] and not calibration["at_upper_bound"]
    ), calibration
    dump(
        ROOT / "calibration.json",
        dict(
            calibration,
            rho_global=rho,
            geometry_calibration_cell_ids=geometry_info["cell_id"][
                nodes[calibration_nodes]
            ].tolist(),
            qualification="Same conditional fine-row objective on512geometry-selected nodes, including ALL their accepted neighbours; global absolute covariance denominators use all eligible LCIS cells. Scale-up sampling, not CATS tuning.",
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
        temporary = ROOT / "checkpoint.tmp.npy"
        with temporary.open("wb") as checkpoint_file:
            np.save(checkpoint_file, x)
        temporary.replace(ROOT / "checkpoint.npy")
        dump(
            ROOT / "checkpoint.json",
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
            status == 0 and defect < 1e-7 and scaled_defect < max(2 * tolerance, 1e-14)
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
        assert details["normal_relative_residual"] < 1e-7, details
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


def solve():
    """Solve the assembled system and verify a full-parameter injected signal.

    Export only after both the physical residual and signal-preservation checks pass.
    No tide coefficient, mean, grounded cell or unsupported hole is filled or fixed.
    """
    path = ROOT / "native_tides.h5"
    if path.exists():
        with h5py.File(path) as handle:
            assert handle.attrs["complete"] == 1
        return
    started = time.monotonic()
    geometry_info = geometry()
    print("CONNECTED_FULL_PHASE loading_prepared_node_blocks", flush=True)
    with h5py.File(ROOT / "metadata.h5") as handle:
        nodes = handle["global_index"][:]
        allpairs = handle["pairs"][:]
    with h5py.File(ROOT / "assembly.h5") as handle:
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
        with h5py.File(ROOT / "nodes" / f"{k:03d}.h5") as handle:
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
        with h5py.File(ROOT / "edges" / f"{k:04d}.h5") as handle:
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
    if (ROOT / "checkpoint.npy").exists():
        with (ROOT / "checkpoint.npy").open("rb") as handle:
            initial = np.load(handle)
        assert initial.shape == rhs.shape
    delta, main = solve_system(rhs, initial)
    info.update(main)
    dump(
        ROOT / "main_numerical_convergence.json",
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
    if signal_error >= 1e-5:
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
        if signal_error < 1e-5:
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
        passed=signal_error < 1e-5,
        maximum_error_m=signal_error,
        required_maximum_error_m=1e-5,
        main=main,
        signal=signal_check,
        residual_refinements=refinements,
    )
    if signal_error >= 1e-5:
        difference = (translated - delta - injected_signal.ravel()).reshape(n, P)
        at = np.unravel_index(np.argmax(abs(difference)), difference.shape)
        control.update(
            worst_global_index=int(nodes[at[0]]),
            worst_parameter=int(at[1]),
            row_error_p50_p95_p99_p100=np.percentile(
                np.max(abs(difference), axis=1), [50, 95, 99, 100]
            ).tolist(),
        )
    dump(ROOT / "full_signal_control.json", control)
    assert signal_error < 1e-5, control
    assert_sources()
    value = seed + delta.reshape(n, P)
    value[np.isnan(seed)] = np.nan
    info.update(
        fixed_cells=0,
        eligible_cells=n,
        formerly_fixed_but_free=int(geometry_info["protected"][nodes].sum()),
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
            "protected",
            "time_ns",
            "reference_seconds",
            "passes",
        ]:
            ds(handle, k, geometry_info[k])
        with h5py.File(analysis.OLD / "geometry.h5") as old:
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
            protected_definition="Mapped2kmgrounding/pinning vicinity for diagnostics ONLY; all eligible cells including these have free coefficients.",
            qualification="User-requested full-LCIS experimental no-fixed-pixel original-height likelihood. Prior central predictive gate remains unresolved. No new covariance-conditioning acceptance or calibrated uncertainty claimed.",
        )
    path.with_suffix(".tmp.h5").replace(path)
    dump(ROOT / "solver.json", info)
    print("CONNECTED_FULL_SOLVED", json.dumps(info), flush=True)


def probe():
    """Compare the scaled operator and solution with the small reference solver."""
    ROOT.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(71103)
    date_count = 36
    cell_count = 4
    fine_count = 9
    design = rng.normal(size=(cell_count, date_count, 9))
    normal_weight = rng.uniform(0.3, 1, (date_count, cell_count, fine_count))
    residual_scale = rng.uniform(0.2, 0.6, cell_count)
    seed = rng.normal(size=(cell_count, P))
    heights = (
        np.einsum("ntp,np->tn", design, seed[:, :9])[:, :, None]
        + seed[None, :, 9:]
        + rng.normal(0, 0.2, (date_count, cell_count, fine_count))
    )
    rho = 0.5
    lam = 3.0
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    hh, _, ff, rr, qq, yy = graph.blocks(
        heights, design, normal_weight, residual_scale, rho, seed
    )
    residual = (
        heights
        - np.einsum("ntp,np->tn", design, seed[:, :9])[:, :, None]
        - seed[None, :, 9:]
    )
    pair_weight = rng.uniform(0.2, 2, (date_count, len(pairs), fine_count))
    hi, hj, off, ri, rj = graph.fine_blocks(design, pairs, pair_weight, residual)
    diag = np.zeros_like(hh)
    grhs = np.zeros((cell_count, P))
    np.add.at(diag, pairs[:, 0], hi)
    np.add.at(diag, pairs[:, 1], hj)
    np.add.at(grhs, pairs[:, 0], ri)
    np.add.at(grhs, pairs[:, 1], rj)
    oldA, rhs, oldrun, _ = graph.system(hh, ff, rr, qq, yy, pairs, diag, off, grhs, lam)
    A, run, _ = block_system(hh + lam * diag, ff.copy(), rhs, qq, pairs, lam * off)
    expected, _ = oldrun(rhs)
    actual, _ = run(rhs)
    error = float(np.max(abs(expected - actual)))
    assert error < 1e-7, error
    vector = rng.normal(size=cell_count * P)
    matvec = float(np.max(abs(A @ vector - oldA @ vector)))
    assert matvec < 1e-7
    dump(
        ROOT / "execution_probe.json",
        dict(
            passed=True,
            pilot_solver_error_m=error,
            normal_matvec_error=matvec,
            fixed_cells=0,
            source_acquisitions=NT,
        ),
    )
    expected = ROOT / "source_probe_expected.h5"
    if expected.exists():
        geometry_info = geometry()
        with h5py.File(expected) as expected_file:
            ids = expected_file["cell_id"][:]
            positions = np.searchsorted(geometry_info["cell_id"], ids)
            np.testing.assert_array_equal(geometry_info["cell_id"][positions], ids)
            heights, design, normal_weight, residual_scale, seed = load(
                positions, geometry_info
            )
            errors = {}
            for k, actual, atol in [
                ("height", heights, 1e-7),
                ("design", design, 1e-8),
                ("normal_weight", normal_weight, 1e-7),
                ("scale", residual_scale, 1e-8),
                ("seed", seed, 1e-7),
            ]:
                old = expected_file[k][:]
                np.testing.assert_allclose(
                    actual, old, atol=atol, rtol=1e-7, equal_nan=True
                )
                finite = np.isfinite(actual + old)
                errors[k] = (
                    float(np.max(abs(actual[finite] - old[finite])))
                    if finite.any()
                    else 0.0
                )
        dump(
            ROOT / "source_probe.json",
            dict(
                passed=True,
                cells=len(ids),
                maximum_absolute_errors=errors,
                qualification="Actual prepared heights/astronomy/frozen normal weights/scales/independent seeds match completed local pilot sources.",
            ),
        )
    print("CONNECTED_FULL_EXECUTION_PROBE_PASS", error, matvec, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["probe", "prepare", "node", "edge", "assemble", "solve"]
    )
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args(argv)
    assert_sources()
    if args.mode == "probe":
        probe()
    elif args.mode == "prepare":
        prepare()
    elif args.mode == "node":
        node_part(args.index)
    elif args.mode == "edge":
        edge_part(args.index)
    elif args.mode == "assemble":
        assemble()
    elif args.mode == "solve":
        solve()


if __name__ == "__main__":
    main()

"""Check catalogue sizes and file dimensions on small, synthetic inputs."""

from pathlib import Path
import json
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "processing"))
from layout import (
    acquisition_counts,
    selected_acquisitions,
    band_indices,
    fit_part_indices,
)
from leaveout_reference import one
from residual_phase import loo_median


def test_selection_uses_the_catalogue_and_its_recorded_addition():
    base = [{"id": f"date_{i}"} for i in range(3)]
    settings = {"acquisitions": base, "added_acquisition": {"acquisition_id": "extra"}}
    assert acquisition_counts(settings) == (3, 4)
    assert selected_acquisitions(3, settings) == base
    assert [date["id"] for date in selected_acquisitions(4, settings)] == [
        "date_0",
        "date_1",
        "date_2",
        "extra",
    ]
    assert len(settings["acquisitions"]) == 3
    assert acquisition_counts({"acquisitions": base}) == (3,)
    with pytest.raises(ValueError, match="catalogue"):
        selected_acquisitions(190, settings)


def test_band_metadata_and_older_geometry_agree(tmp_path):
    path = tmp_path / "geometry.h5"
    with h5py.File(path, "w") as handle:
        # Cross the bounded-read boundary; include a non-contiguous band ID.
        handle["source_band"] = np.r_[np.zeros(65537, dtype="i1"), [5, 9]]
    assert band_indices(path) == (0, 5, 9)
    with h5py.File(path, "r+") as handle:
        handle.attrs.update(band_ids=[0, 5, 9], fit_parts_per_band=2)
    assert band_indices(path) == (0, 5, 9)
    assert list(fit_part_indices(path)) == list(range(6))
    with h5py.File(path, "r+") as handle:
        handle.attrs["band_ids"] = [0, 0]
    with pytest.raises(ValueError, match="Invalid band IDs"):
        band_indices(path)


@pytest.mark.parametrize("date_count", [11, 15, 190])
def test_leaveout_history_matches_deleting_each_date(tmp_path, date_count):
    rng = np.random.default_rng(41)
    height = rng.integers(-5, 6, size=(date_count, 2, 9)).astype("f8")
    weight = np.ones_like(height)
    height[0, 0, 0] = np.nan
    weight[:3, 0, 1] = 0
    weight[:, 0, 2] = 0
    weight[7:, 0, 3] = 0
    ids = np.arange(18).reshape(2, 9)
    ids[-1, -1] = -1
    source = tmp_path / "history.h5"
    with h5py.File(source, "w") as handle:
        handle["height"] = height
        handle["weight"] = weight
        handle["fine_id"] = ids
    destination = tmp_path / "statistics.h5"
    one(source, destination, coarse_chunk=1)
    admitted = height.reshape(date_count, -1).copy()
    admitted[~(weight.reshape(date_count, -1) > 0)] = np.nan
    admitted[:, ids.ravel() < 0] = np.nan
    with h5py.File(destination) as handle:
        assert handle.attrs["complete"]
        count, lo, mid, hi = [handle[key][:] for key in ("count", "lo", "mid", "hi")]
        np.testing.assert_array_equal(count, np.isfinite(admitted).sum(axis=0))
        for date in range(date_count):
            actual = loo_median(count, lo, mid, hi, admitted[date])
            other = np.delete(admitted, date, axis=0)
            expected = np.array(
                [
                    (
                        np.median(column[np.isfinite(column)])
                        if np.isfinite(column).sum() >= 8
                        else np.nan
                    )
                    for column in other.T
                ],
                dtype="f4",
            )
            np.testing.assert_array_equal(actual, expected)


def test_sparse_layers_follow_native_manifest_and_reject_incomplete_repairs(tmp_path):
    import native_phase

    rows = [{"overpass": name} for name in ("first", "second", "third")]
    reports = tmp_path / "shelf_native_passes"
    reports.mkdir()
    for index, row in enumerate(rows):
        name = row["overpass"]
        (reports / f"{name}.json").write_text(
            json.dumps({"status": "assessed", "affected_cells": index})
        )
        np.savez(
            reports / f"{name}.npz",
            cell=np.arange(index + 1),
            affected=np.arange(index + 1) < index,
            raw=np.arange(index + 1, dtype="f4"),
            corrected=np.arange(index + 1, dtype="f4") + 1,
        )
        residual = tmp_path / "residual" / name
        residual.mkdir(parents=True)
        (residual / "report.json").write_text(json.dumps({"complete": True}))
        if index:
            np.savez(
                residual / "grid_phase2.npz",
                cell_id=np.arange(index),
                before_wse=np.zeros(index),
                after_wse=np.ones(index),
            )
    output = tmp_path / "packed"
    output.mkdir()
    native_phase.pack(tmp_path, output, rows)
    for filename in ("phase_region190.h5", "phase_residual190.h5"):
        with h5py.File(output / filename) as handle:
            assert handle.attrs["acquisitions"] == len(rows)
            np.testing.assert_array_equal(handle["observation_start"][:], [0, 0, 1, 3])
            assert list(handle["acquisition_id"].asstr()[:]) == [
                row["overpass"] for row in rows
            ]
    from pack_region_phase import build

    (reports / "second.json").write_text(
        json.dumps({"status": "pending", "affected_cells": 1})
    )
    with pytest.raises(ValueError, match="Incomplete first-stage"):
        build(tmp_path / "selection.json", [tmp_path], output / "incomplete.h5")


def test_reference_packer_uses_all_catalogue_dates(tmp_path):
    from reference_replay import pack_band

    geometry = tmp_path / "geometry.h5"
    with h5py.File(geometry, "w") as handle:
        handle["source_band"] = [2, 5]
        handle["fine_id"] = np.arange(18).reshape(2, 9)
        handle["cell_id"] = [10, 20]
    (tmp_path / "rows").mkdir()
    dates = [{"id": str(i)} for i in range(3)]
    for date in range(len(dates)):
        with h5py.File(tmp_path / "rows" / f"{date:03d}.h5", "w") as handle:
            handle["fine_id"] = [0, 9, 17]
            handle["height"] = np.array([1, 2, 3]) + date
            handle["weight"] = [1, 1, 1]
    pack_band({"analysis_geometry": geometry}, tmp_path, 5, {"acquisitions": dates})
    with h5py.File(tmp_path / "bands/005.h5") as handle:
        assert handle["height"].shape == (3, 1, 9)
        np.testing.assert_array_equal(handle["height"][:, 0, 0], [2, 3, 4])
        np.testing.assert_array_equal(handle["height"][:, 0, 8], [3, 4, 5])
        assert np.isnan(handle["height"][:, 0, 1:8]).all()


@pytest.mark.parametrize(
    "bands,parts_per_band", [(tuple(range(16)), 4), ((2, 5, 9), 3)]
)
def test_independent_fit_groups_cover_each_cell_once(
    tmp_path, monkeypatch, bands, parts_per_band
):
    import analysis

    geometry = tmp_path / "geometry.h5"
    source_band = np.repeat(bands, 13)
    with h5py.File(geometry, "w") as handle:
        handle.attrs.update(band_ids=bands, fit_parts_per_band=parts_per_band)
        handle["source_band"] = source_band
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    monkeypatch.setattr(analysis, "geometry", lambda: {"source_band": source_band})
    captured = []

    def admitted(nodes, geometry_info):
        captured.append(nodes.copy())
        raise StopIteration

    monkeypatch.setattr(analysis, "admitted", admitted)
    for part in fit_part_indices(geometry):
        with pytest.raises(StopIteration):
            analysis.fit_part(part)
    np.testing.assert_array_equal(
        np.sort(np.concatenate(captured)), np.arange(len(source_band))
    )
    expected = []
    for band in bands:
        cells = np.flatnonzero(source_band == band)
        for quarter in range(parts_per_band):
            expected.append(
                cells[
                    len(cells)
                    * quarter
                    // parts_per_band : len(cells)
                    * (quarter + 1)
                    // parts_per_band
                ]
            )
    for actual, previous in zip(captured, expected):
        np.testing.assert_array_equal(actual, previous)
    with pytest.raises(ValueError, match="Part index"):
        analysis.fit_part(len(bands) * parts_per_band)


def test_solver_part_count_uses_prepared_metadata(tmp_path, monkeypatch):
    import gls

    monkeypatch.setattr(gls, "ROOT", tmp_path)
    monkeypatch.setattr(gls, "NODE_PARTS", 99)
    with h5py.File(tmp_path / "metadata.h5", "w") as handle:
        handle.attrs["node_parts"] = 7
    assert gls.node_part_count() == 7
    with h5py.File(tmp_path / "metadata.h5", "r+") as handle:
        handle.attrs["node_parts"] = 0
    with pytest.raises(ValueError, match="positive"):
        gls.node_part_count()

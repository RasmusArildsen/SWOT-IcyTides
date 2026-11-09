"""Check comparison support and constituent statistics without a shelf run."""

from pathlib import Path
import sys
import h5py
import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE), str(CODE / "processing")]
from Figure3 import plot_figure3
from paper_statistics import constituent_table, circular_median


def test_fixed_observations_require_the_original_raw_footprint(tmp_path, monkeypatch):
    reference = dict(
        time_ns=np.array([1, 2]),
        distance_km=np.array([2.0, 3.0]),
        longitude=np.array([-62.0, -61.0]),
        latitude=np.array([-67.0, -66.0]),
        corridor_support=np.array([[5, 6], [6, 5]]),
        original_height=np.array([[20.0, 21.0], [22.0, 23.0]]),
        cats_tide=np.array([[1.0, 2.0], [3.0, 4.0]]),
        observed=np.array([[19.0, 20.0], [21.0, 22.0]]),
    )
    np.savez(tmp_path / "figure3_reference_profiles.npz", **reference)
    monkeypatch.setattr(plot_figure3, "AUX", tmp_path)
    swot = np.array([[0.2, 0.3], [0.4, 0.5]])
    with h5py.File(tmp_path / "figure3.h5", "w") as handle:
        for name, value in reference.items():
            handle[name] = value + 100 if name == "observed" else value
        handle["swot_tide"] = swot
    result = plot_figure3.comparison_profiles(tmp_path)
    np.testing.assert_array_equal(result["observed"], reference["observed"])
    np.testing.assert_array_equal(result["swot_tide"], swot)
    with h5py.File(tmp_path / "figure3.h5", "r+") as handle:
        handle["original_height"][0, 0] += 0.01
    with pytest.raises(ValueError, match="original_height"):
        plot_figure3.comparison_profiles(tmp_path)


def test_table_uses_complex_differences_and_keeps_low_amplitude_cells(tmp_path):
    angles = np.deg2rad([359.0, 1.0, 180.0, 40.0])
    swot = np.array([1.0, 1.0, 0.01, 1.0]) * np.exp(1j * angles)
    cats = np.array([1.0, 1.0, 0.02, 1.0])
    beta = np.zeros((9, 4))
    model = np.zeros((8, 4))
    for index in range(4):
        beta[2 * index] = swot.real
        beta[2 * index + 1] = swot.imag
        model[2 * index] = cats
    beta[:, -1] = np.nan
    native, geometry = tmp_path / "native.h5", tmp_path / "geometry.h5"
    with h5py.File(native, "w") as handle:
        handle.attrs["complete"] = 1
        handle["cell_id"] = [4, 6, 9, 10]
        handle["beta"] = beta
    with h5py.File(geometry, "w") as handle:
        handle["cell_id"] = [4, 6, 9, 10]
        handle["cats_beta"] = model
    rows = constituent_table(native, geometry, tmp_path / "figures")
    assert all(row["amplitude_vector_cells"] == 3 for row in rows)
    assert all(row["phase_cells"] == 2 for row in rows)
    np.testing.assert_allclose(
        rows[0]["median_vector_difference_cm"],
        np.median(abs(swot[:3] - cats[:3])) * 100,
    )
    assert (
        min(
            rows[0]["swot_circular_median_phase_deg"],
            360 - rows[0]["swot_circular_median_phase_deg"],
        )
        < 1e-10
    )
    assert circular_median([]) is None
    with h5py.File(geometry, "r+") as handle:
        handle["cell_id"][0] = 5
    with pytest.raises(AssertionError):
        constituent_table(native, geometry, tmp_path / "other")

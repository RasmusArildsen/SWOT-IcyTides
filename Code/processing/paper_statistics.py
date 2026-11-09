"""Summarise the native SWOT and CATS coefficients for Table 1."""

import json
import h5py
import numpy as np

CONSTITUENTS = ("M2", "S2", "K1", "O1")


def circular_median(degrees):
    """Use the paper's circular-mean branch, then take a wrapped median."""
    if not len(degrees):
        return None
    centre = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(degrees)))))
    return float((centre + np.median((degrees - centre + 180) % 360 - 180)) % 360)


def constituent_table(native, geometry, output_dir):
    """Read bounded coefficient blocks on the prepared study-area cells.

    Amplitudes and vector differences use every finite paired cell. Phase
    summaries require both amplitudes to be at least 5 cm. No amplitude cap.
    """
    columns = [
        {
            name: []
            for name in ("swot_amp", "cats_amp", "swot_phase", "cats_phase", "vector")
        }
        for _ in CONSTITUENTS
    ]
    with h5py.File(native) as source, h5py.File(geometry) as reference:
        if source.attrs.get("complete") != 1:
            raise ValueError("Native solution is incomplete")
        if source["cell_id"].shape != reference["cell_id"].shape:
            raise ValueError("Native solution and CATS geometry use different cells")
        for first in range(0, len(source["cell_id"]), 8192):
            window = np.s_[first : first + 8192]
            np.testing.assert_array_equal(
                source["cell_id"][window], reference["cell_id"][window]
            )
            beta = source["beta"][:, window].astype(float)
            cats = reference["cats_beta"][:, window].astype(float)
            valid = np.isfinite(beta).all(axis=0)
            for index, values in enumerate(columns):
                swot = beta[2 * index] + 1j * beta[2 * index + 1]
                model = cats[2 * index] + 1j * cats[2 * index + 1]
                paired = valid & np.isfinite(model)
                phase = paired & (abs(swot) >= 0.05) & (abs(model) >= 0.05)
                for name, data in {
                    "swot_amp": abs(swot[paired]) * 100,
                    "cats_amp": abs(model[paired]) * 100,
                    "swot_phase": np.angle(swot[phase], deg=True) % 360,
                    "cats_phase": np.angle(model[phase], deg=True) % 360,
                    "vector": abs(swot[paired] - model[paired]) * 100,
                }.items():
                    values[name].append(data)
    rows = []
    latex = []
    for name, blocks in zip(CONSTITUENTS, columns):
        values = {key: np.concatenate(parts) for key, parts in blocks.items()}
        row = dict(
            constituent=name,
            amplitude_vector_cells=len(values["vector"]),
            phase_cells=len(values["swot_phase"]),
            swot_median_amplitude_cm=float(np.median(values["swot_amp"])),
            cats_median_amplitude_cm=float(np.median(values["cats_amp"])),
            swot_circular_median_phase_deg=circular_median(values["swot_phase"]),
            cats_circular_median_phase_deg=circular_median(values["cats_phase"]),
            median_vector_difference_cm=float(np.median(values["vector"])),
        )
        rows.append(row)
        numbers = [
            row[key]
            for key in (
                "swot_median_amplitude_cm",
                "cats_median_amplitude_cm",
                "swot_circular_median_phase_deg",
                "cats_circular_median_phase_deg",
            )
        ]
        formatted = ["--" if value is None else f"{value:.1f}" for value in numbers]
        latex.append(
            name[0]
            + "$_"
            + name[1]
            + "$ & "
            + " & ".join(formatted)
            + f" & {row['median_vector_difference_cm']:.2f}"
            + r" \\"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "constituent_statistics.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n"
    )
    (output_dir / "Table1_rows.tex").write_text("\n".join(latex) + "\n")
    return rows

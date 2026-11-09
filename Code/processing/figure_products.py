"""Extract maps and transects from a completed November fit.

maps.h5:    native and CATS constituent amplitudes.
figure1.h5: fitted fine-cell elevations and REMA reference.
figure3.h5: observations and predictions on the selected transect."""

from pathlib import Path
import json
import warnings
import h5py
import numpy as np
import pandas as pd
import analysis
from analysis import ds, dump
from layout import band_indices

AUX = Path(__file__).resolve().parents[1] / "aux_files"


def quiet_median(x, axis=None):
    """Compute a NaN-aware median without warnings for empty support."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(x, axis=axis)


def profiles(native, independent_dir, output_dir):
    """Build the frozen Figure 3 corridor with the same cells and dates.

    Subtract each fitted fine-cell static level, then add the centreline level.
    Retain raw predictions alongside the inherited prediction-display median.
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
    cells = np.unique(np.r_[sampling["native_cell_id"], line["nearest_cell_id"]])
    cells = cells[cells >= 0]
    with h5py.File(independent_dir / "geometry.h5") as handle:
        all_cell_ids = handle["cell_id"][:]
        positions = np.searchsorted(all_cell_ids, cells)
        np.testing.assert_array_equal(all_cell_ids[positions], cells)
        fine_mapping = handle["fine_id"][positions]
        cats = handle["cats_beta"][:, positions]
        lat = handle["lat"][positions]
        reference_seconds = handle["reference_seconds"][: analysis.NT]
        bands = handle["source_band"][positions]
        local = handle["source_local"][positions]
        time_ns = handle["time_ns"][: analysis.NT]
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
            for k, t in enumerate(dates):
                heights[k, destination] = handle["height"][int(t), local_index]
            when[:, destination] = handle["time_seconds"][:, local_index]
    time_design = analysis.basis(when.T, lat, reference_seconds)
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
    raw_height = observed.copy()
    swot_profile = observed.copy()
    cats_profile = observed.copy()
    counts = np.zeros(shape, "i2")
    missing = np.zeros(shape, "i2")
    for date in range(len(dates)):
        for bin_index in range(shape[1]):
            selected = np.flatnonzero(
                (sampling["bin"] == bin_index) & sampling["accepted"][date]
            )
            counts[date, bin_index] = len(selected)
            if len(selected) < 5:
                continue
            selected_heights = heights[date, rows[selected], slots[selected]]
            assert np.isfinite(
                selected_heights
            ).all(), "Frozen observation support changed"
            normalized_height = (
                selected_heights
                - static_levels[rows[selected], slots[selected]]
                + level[bin_index]
            )
            swot_tide = swot_prediction[date, rows[selected]]
            cats_tide = cats_prediction[date, rows[selected]]
            raw_height[date, bin_index] = np.median(selected_heights)
            if np.isfinite(cats_tide).all():
                cats_profile[date, bin_index] = np.median(cats_tide)
            unavailable = ~np.isfinite(normalized_height + swot_tide + cats_tide)
            missing[date, bin_index] = int(unavailable.sum())
            if unavailable.any():
                continue
            observed[date, bin_index] = np.median(normalized_height)
            swot_profile[date, bin_index] = np.median(swot_tide)
    distance = np.asarray(line["distance_km"])
    protect = ((distance >= 4.6) & (distance <= 8.6)) | (
        (distance >= 38.5) & (distance <= 42.9)
    )

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
            original_height=raw_height,
            swot_tide=display(swot_profile),
            cats_tide=display(cats_profile),
            swot_tide_unsmoothed=swot_profile,
            cats_tide_unsmoothed=cats_profile,
            corridor_support=counts,
            unavailable=missing,
            **{
                k: line[k]
                for k in ["distance_km", "longitude", "latitude", "x3031", "y3031"]
            },
        ).items():
            ds(handle, k, v)
        handle.attrs.update(
            complete=1,
            source=str(native),
            support="Frozen8720fine cells/12dates; entire bin withheld if any model point unavailable",
            display="Median5 of predicted tides only; unsmoothed within4.6–8.6km and38.5–42.9km; no observed/residual smoothing",
            qualification="In-sample selected transect, not independent validation",
        )


def build(native, independent_dir, resources, output_dir):
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
            for k in ["m2", "s2", "k1", "o1"]:
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
                for j, k in enumerate(["m2", "s2", "k1", "o1"]):
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
                epoch_seconds_2000=analysis.SETTINGS[
                    "original190_time_center_seconds_2000"
                ],
                units="m above mean-tide geoid",
                method="Matching final own100m statics; TIME evaluated at frozen original190 centre. REMA is upstream selection, not independent validation.",
            )
    profiles(native, independent_dir, output_dir)
    dump(
        output_dir / "sources.json",
        dict(
            native=str(native.resolve()),
            geometry=str((independent_dir / "geometry.h5").resolve()),
            preparation=str(independent_dir.resolve()),
            products=["maps.h5", "figure1.h5", "figure3.h5"],
            analysis_version=analysis.SETTINGS["version"],
        ),
    )

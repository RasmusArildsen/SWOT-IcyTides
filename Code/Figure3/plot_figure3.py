"""Figure 3: the fixed observation profiles and current four-constituent tides."""

from pathlib import Path
import sys
import json
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paper_maps import (
    AUX,
    NORTH,
    set_style,
    tidal_fields,
    optical,
    map_panel,
    inventory,
    save_figure,
    plot_arguments,
    plt,
)


def comparison_profiles(products):
    """Keep the paper's observed heights fixed when comparing a new tide fit.

    Changing the fitted static surface must not change the CATS benchmark.
    Check dates, footprint and raw heights before using the saved observations.
    """
    with h5py.File(products / "figure3.h5") as source:
        values = {name: source[name][:] for name in source.keys()}
    with np.load(AUX / "figure3_reference_profiles.npz") as reference:
        for name in (
            "time_ns",
            "distance_km",
            "longitude",
            "latitude",
            "corridor_support",
        ):
            if not np.array_equal(values[name], reference[name], equal_nan=True):
                raise ValueError(f"Figure 3 reference does not match {name}")
        for name in ("original_height", "cats_tide"):
            if not np.allclose(
                values[name], reference[name], rtol=0, atol=1e-7, equal_nan=True
            ):
                raise ValueError(f"Figure 3 reference does not match {name}")
        values["observed"] = reference["observed"].copy()
    return values


def figure3(products, resources, output_dir, formats=("png", "pdf")):
    set_style()
    values = comparison_profiles(products)
    distance = values["distance_km"]
    days = values["time_ns"].astype("datetime64[ns]").astype("datetime64[D]")
    chosen = [
        int(np.flatnonzero(days == np.datetime64(day))[0])
        for day in ("2025-09-05", "2025-03-22")
    ]
    later, earlier = chosen
    h1, h0 = values["observed"][[later, earlier]]
    c1, c0 = values["cats_tide"][[later, earlier]]
    t1, t0 = values["swot_tide"][[later, earlier]]
    paired = (distance >= 2) & np.isfinite(h1 + h0 + c1 + c0 + t1 + t0)
    cats_residual = abs((h1 - h0) - (c1 - c0))
    swot_residual = abs((h1 - h0) - (t1 - t0))
    report = {
        "paired_positions": int(paired.sum()),
        "CATS_MAE_m": float(cats_residual[paired].mean()),
        "SWOT_MAE_m": float(swot_residual[paired].mean()),
        "observations": "Original paper profiles, checked against dates, raw heights and footprint.",
        "comparison": "In sample; FOUR only; common finite positions from 2 km onward.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transect_statistics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    fig = plt.figure(figsize=(7.2, 8.6), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.35, 1.35])
    maps = [fig.add_subplot(grid[0, column]) for column in range(2)]
    fields = tidal_fields(products, NORTH, (1100, 1200))
    background = optical(resources["lima_mosaic"], NORTH, (1100, 1200))
    for column, ax in enumerate(maps):
        field = fields["m2_amp" if column == 0 else "m2_cats_amp"]
        title = ["(a) SWOT M$_2$ amplitude", "(b) CATS M$_2$ amplitude"][column]
        image = map_panel(ax, NORTH, title, field, background)
        inventory(ax, thin=True)
        ax.plot(
            values["longitude"],
            values["latitude"],
            color="#d73027",
            lw=1.1,
            label="Selected transect",
        )
        ax.scatter(
            values["longitude"][0],
            values["latitude"][0],
            s=21,
            c="black",
            ec="white",
            lw=0.8,
            zorder=8,
        )
        ax.text(
            -61.18,
            -66.12,
            "Transect start",
            fontsize=7,
            bbox=dict(fc="white", ec="none", alpha=0.85),
        )
        for position, label in [
            ((-61.56, -66.14), "NE\n5.8 km"),
            ((-61.82, -66.41), "SW\n40.6 km"),
        ]:
            ax.text(
                *position,
                label,
                color="#d73027",
                fontsize=7,
                va="center",
                bbox=dict(fc="white", ec="none", alpha=0.86),
            )
        ax.legend(loc="lower right", fontsize=7, framealpha=0.93)
        if column:
            ax.set_ylabel("")
    colourbar = fig.colorbar(
        image, ax=maps, orientation="vertical", pad=0.025, fraction=0.035, aspect=25
    )
    colourbar.set_label("M$_2$ amplitude (cm)")

    observed = fig.add_subplot(grid[1, :])
    for index in range(len(days)):
        if index not in chosen:
            observed.plot(
                distance,
                values["observed"][index],
                color="#777777",
                lw=0.45,
                alpha=0.32,
                zorder=1,
            )
    observed.plot(distance, h1, color="#d73027", lw=1.2, label="Cycle 38 (2025-09-05)")
    observed.plot(distance, h0, color="#2166ac", lw=1.2, label="Cycle 30 (2025-03-22)")
    observed.plot([], [], color="#777777", lw=0.6, label="Other cycles")
    observed.set(
        title="(c) Observed elevations before ocean tide correction",
        ylabel="Elevation (m)",
    )
    observed.legend(loc="upper right", fontsize=7, framealpha=0.94)

    residual = fig.add_subplot(grid[2, :])
    curves = [
        (abs(h1 - h0), "Observed", "black", "-"),
        (abs(c1 - c0), "CATS predicted difference", "#f39c12", "-"),
        (abs(t1 - t0), "SWOT predicted difference", "#7570b3", "-"),
        (cats_residual, "CATS residual", "#f39c12", "--"),
        (swot_residual, "SWOT residual", "#7570b3", "--"),
    ]
    for curve, label, colour, line in curves:
        residual.plot(
            distance,
            np.where(paired, curve, np.nan),
            label=label,
            color=colour,
            ls=line,
            lw=1.1,
        )
    residual.set(
        title="(d) Absolute two-acquisition differences",
        ylabel="Absolute difference (m)",
        xlabel="Distance along transect (km)",
        ylim=(-0.04, None),
    )
    residual.legend(
        loc="center",
        bbox_to_anchor=(0.53, 0.35),
        fontsize=7,
        ncol=2,
        framealpha=0.94,
        handlelength=1.7,
        columnspacing=1,
    )
    for ax in (observed, residual):
        ax.set_xlim(2, float(distance.max()))
        ax.grid(alpha=0.23, lw=0.45)
        ax.axvspan(4, 10.3, color="#377eb8", alpha=0.15, zorder=0)
        ax.axvspan(38.9, 42.5, color="#e41a1c", alpha=0.15, zorder=0)
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.06, wspace=0.04, hspace=0.05)
    save_figure(fig, output_dir, "figure3", dpi=500, formats=formats)
    return report


if __name__ == "__main__":
    args, resources, formats = plot_arguments(3)
    figure3(args.products, resources, args.output_dir, formats)

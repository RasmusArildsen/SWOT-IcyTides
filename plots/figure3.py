"""Figure 3. Northern transect and tide predictions."""

from .style import (
    FIGURE_SETTINGS, NORTH, OUTPUT, inventory, load_inputs, map_panel,
    save_figure, set_style,
)
import matplotlib.pyplot as plt
import numpy as np


def draw_figure3(values, output_dir, formats=("png", "pdf")):
    set_style()
    distance = values["distance_km"]
    days = values["time_ns"].astype("datetime64[ns]").astype("datetime64[D]")
    settings = FIGURE_SETTINGS["figure3"]
    profiles = settings["profiles"]
    chosen = []
    for profile in profiles:
        matches = np.flatnonzero(days == np.datetime64(profile["date"]))
        if not len(matches):
            raise ValueError(f"No transect profile for {profile['date']}")
        chosen.append(int(matches[0]))
    later, earlier = chosen
    minimum_distance = settings["minimum_distance_km"]
    labels = [f"Cycle {profile['cycle']} ({profile['date']})" for profile in profiles]
    h1, h0 = values["observed"][[later, earlier]]
    c1, c0 = values["cats_tide"][[later, earlier]]
    t1, t0 = values["swot_tide"][[later, earlier]]
    paired = (distance >= minimum_distance) & np.isfinite(h1 + h0 + c1 + c0 + t1 + t0)
    cats_residual = abs((h1 - h0) - (c1 - c0))
    swot_residual = abs((h1 - h0) - (t1 - t0))
    fig = plt.figure(figsize=(7.2, 8.6), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.35, 1.35])
    maps = [fig.add_subplot(grid[0, column]) for column in range(2)]
    background = values["background"]
    for column, ax in enumerate(maps):
        field = values["m2_amp" if column == 0 else "m2_cats_amp"]
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
            *settings["start_label_lonlat"],
            "Transect start",
            fontsize=7,
            bbox=dict(fc="white", ec="none", alpha=0.85),
        )
        for feature in settings["feature_labels"]:
            ax.text(
                *feature["lonlat"],
                f"{feature['name']}\n{feature['distance_km']:g} km",
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
    observed.plot(distance, h1, color="#d73027", lw=1.2, label=labels[0])
    observed.plot(distance, h0, color="#2166ac", lw=1.2, label=labels[1])
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
        ax.set_xlim(minimum_distance, float(distance.max()))
        ax.grid(alpha=0.23, lw=0.45)
        for interval in settings["highlighted_intervals"]:
            ax.axvspan(*interval["bounds_km"], color=interval["color"], alpha=0.15, zorder=0)
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.06, wspace=0.04, hspace=0.05)
    save_figure(fig, output_dir, "figure3", dpi=500, formats=formats)


if __name__ == "__main__":
    draw_figure3(load_inputs(3), OUTPUT)

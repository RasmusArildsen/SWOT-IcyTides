"""Figure 1. SWOT mean elevation compared with REMA and ICESat-2."""

import json

from .style import (
    AUX, FIGURE_SETTINGS, OUTPUT, load_inputs, map_panel, save_figure, set_style,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator
import numpy as np


def draw_locator(ax, definition):
    """Draw the paper's coastline, study outline and reference-orbit boundaries."""
    for patch in definition["patches"]:
        ax.add_patch(
            Polygon(
                patch["xy"],
                closed=True,
                facecolor=patch["facecolor"],
                edgecolor=patch["edgecolor"],
                linewidth=patch["linewidth"],
                linestyle=patch["linestyle"],
                alpha=patch["alpha"],
                zorder=patch["zorder"],
            )
        )
    for line in definition["lines"]:
        ax.plot(
            line["x"],
            line["y"],
            color=line["color"],
            linewidth=line["linewidth"],
            linestyle=line["linestyle"],
            alpha=line["alpha"],
            zorder=line["zorder"],
            marker=line["marker"],
            markersize=line["markersize"],
        )
    for text in definition["texts"]:
        transform = ax.transAxes if text["axes_coordinates"] else ax.transData
        ax.text(
            *text["position"],
            text["text"],
            transform=transform,
            color=text["color"],
            fontsize=text["fontsize"],
            rotation=text["rotation"],
            ha=text["ha"],
            va=text["va"],
        )
    ax.set(
        xlim=definition["xlim"],
        ylim=definition["ylim"],
        aspect=definition["aspect"],
        xticks=[],
        yticks=[],
    )


def draw_figure1(data, output_dir, formats=("png", "pdf")):
    set_style()
    definition = json.loads((AUX / "figure1_locator.json").read_text())
    bounds = definition["comparison_bounds"]
    fig, axes = plt.subplots(
        1, 4, figsize=(7.2, 3.4), layout="constrained", width_ratios=[1, 1, 1, 1]
    )
    draw_locator(axes[0], definition["locator"])
    axes[0].set(
        title="(a)",
        xlabel="Longitude",
        ylabel="Latitude",
        xticks=[-68, -64, -60],
        yticks=list(range(-72, -62)),
    )
    axes[0].grid(alpha=0.15, lw=0.5)
    inset = axes[0].inset_axes([0.65, 0.795, 0.345, 0.205])
    draw_locator(inset, definition["inset"])
    axes[0].legend(
        handles=[
            Line2D([], [], color="#006f92", lw=2, label="Pass 421 swath"),
            Line2D([], [], color="#888888", lw=1, label="Other SWOT passes"),
            Line2D([], [], color="red", ls="--", lw=0.8, label="Comparison extent"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        fontsize=6.2,
        framealpha=0.94,
        handlelength=1.7,
        borderpad=0.35,
    )

    elevation_bar = map_panel(
        axes[1], bounds, "(b) SWOT mean", data["mean"], maximum=100, alpha=1
    )
    difference_bar = map_panel(
        axes[2], bounds, "(c) SWOT − REMA", data["rema_difference"],
        difference=True, maximum=5, alpha=1,
    )
    map_panel(axes[3], bounds, "(d) SWOT − ICESat-2")
    # Keep the source row order for overlapping scatter markers.
    for start, stop in zip(data["scatter_offsets"][:-1], data["scatter_offsets"][1:]):
        points = slice(start, stop)
        axes[3].scatter(
            data["icesat2_longitude"][points], data["icesat2_latitude"][points],
            c=data["icesat2_difference"][points], s=1.7, marker="s",
            cmap=difference_bar.cmap, norm=difference_bar.norm,
            linewidths=0, rasterized=True,
        )
    settings = FIGURE_SETTINGS["figure1"]
    for ax in axes[1:]:
        ax.set_aspect(1 / np.cos(np.deg2rad(settings["aspect_latitude_deg"])))
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.set_yticks(settings["latitude_ticks_deg"])
        for ring in definition["study_outline"]:
            outline = np.asarray(ring)
            ax.plot(outline[:, 0], outline[:, 1], color="#5a5047", lw=0.65)
        if ax is not axes[1]:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
    bar = fig.colorbar(
        elevation_bar,
        ax=axes[1],
        orientation="horizontal",
        pad=0.025,
        fraction=0.045,
        aspect=23,
        extend="both",
    )
    bar.set_label("Elevation (m)")
    bar.set_ticks([0, 25, 50, 75, 100])
    bar = fig.colorbar(
        difference_bar,
        ax=list(axes[2:]),
        orientation="horizontal",
        pad=0.025,
        fraction=0.045,
        aspect=35,
        extend="both",
    )
    bar.set_label("SWOT − reference (m)")
    bar.set_ticks([-4, -2, 0, 2, 4])
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.045, wspace=0.04, hspace=0.05)
    save_figure(fig, output_dir, "figure1", formats=formats)


if __name__ == "__main__":
    draw_figure1(load_inputs(1), OUTPUT)

"""Figure 1: study area, fitted mean surface and reference comparisons."""

from pathlib import Path
import sys
import json
import h5py
import numpy as np
from pyproj import Transformer
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paper_maps import (
    AUX,
    set_style,
    sample_grid,
    map_panel,
    draw_locator,
    save_figure,
    plot_arguments,
    plt,
)

TO_LL = Transformer.from_crs(3031, 4326, always_xy=True)


def figure1(products, resources, output_dir, formats=("png", "pdf")):
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

    differences = {"swot_rema": [], "swot_atl06": []}
    with (
        h5py.File(products / "figure1.h5") as source,
        h5py.File(resources["atl06_reference"]) as laser,
    ):
        x, y = source["x"][:], source["y"][:]
        np.testing.assert_array_equal(x, laser["x"][:])
        np.testing.assert_array_equal(y, laser["y"][:])
        mean = sample_grid(
            x, y, lambda window: source["mean"][window], bounds, (2000, 1000)
        )

        def rema_difference(window):
            height, reference = source["mean"][window], source["rema"][window]
            return np.where(
                (reference >= 0) & (reference <= 100), height - reference, np.nan
            )

        rema = sample_grid(x, y, rema_difference, bounds, (2000, 1000))
        elevation_bar = map_panel(
            axes[1], bounds, "(b) SWOT mean", mean, maximum=100, alpha=1
        )
        difference_bar = map_panel(
            axes[2],
            bounds,
            "(c) SWOT − REMA",
            rema,
            difference=True,
            maximum=5,
            alpha=1,
        )
        map_panel(axes[3], bounds, "(d) SWOT − ICESat-2")
        for first in range(0, len(y), 96):
            window = np.s_[first : first + 96, :]
            height = source["mean"][window]
            longitude, latitude = TO_LL.transform(
                *np.meshgrid(x, y[first : first + 96])
            )
            inside = (
                (longitude >= bounds[0])
                & (longitude <= bounds[1])
                & (latitude >= bounds[2])
                & (latitude <= bounds[3])
            )
            rema_ref = source["rema"][window]
            laser_ref = laser["mean"][window]
            count = laser["count"][window]
            valid_rema = (
                inside
                & np.isfinite(height + rema_ref)
                & (rema_ref >= 0)
                & (rema_ref <= 100)
            )
            valid_laser = (
                inside
                & np.isfinite(height + laser_ref)
                & (laser_ref >= 0)
                & (laser_ref <= 100)
                & (count > 0)
                & (count < 65535)
            )
            differences["swot_rema"].append((height - rema_ref)[valid_rema])
            differences["swot_atl06"].append((height - laser_ref)[valid_laser])
            axes[3].scatter(
                longitude[valid_laser],
                latitude[valid_laser],
                c=(height - laser_ref)[valid_laser],
                s=1.7,
                marker="s",
                cmap=difference_bar.cmap,
                norm=difference_bar.norm,
                linewidths=0,
                rasterized=True,
            )
    for ax in axes[1:]:
        ax.set_aspect(1 / np.cos(np.deg2rad(-67.5)))
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.set_yticks(np.arange(-69.5, -65.9, 0.5))
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
    statistics = {}
    for name, blocks in differences.items():
        values = np.concatenate(blocks).astype(float)
        statistics[name] = dict(
            n=len(values),
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            rmsd=float(np.sqrt(np.mean(values**2))),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figure1_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n"
    )
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.045, wspace=0.04, hspace=0.05)
    save_figure(fig, output_dir, "figure1", formats=formats)
    return statistics


if __name__ == "__main__":
    args, resources, formats = plot_arguments(1)
    figure1(args.products, resources, args.output_dir, formats)

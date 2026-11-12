"""Map drawing and file paths shared by the three figures."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
AUX = ROOT / "Code" / "aux_files"
OUTPUT = ROOT / "figures" / "reproduced"

FIGURE_SETTINGS = json.loads((AUX / "figure_settings.json").read_text())
map_bounds = FIGURE_SETTINGS["map_bounds_west_east_south_north"]
SHELF = tuple(map_bounds["shelf"])
WEST = tuple(map_bounds["west"])
NORTH = tuple(map_bounds["north"])


def set_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
        }
    )


def map_panel(
    ax,
    bounds,
    title,
    values=None,
    background=None,
    difference=False,
    maximum=120,
    alpha=0.85,
    aspect="auto",
):
    colourmap = cm.vik if difference else "viridis"
    minimum = -maximum if difference else 0
    if background is not None:
        ax.imshow(background, extent=bounds, origin="upper", aspect=aspect)
    if values is not None:
        ax.imshow(
            values,
            extent=bounds,
            origin="lower",
            interpolation="none",
            cmap=colourmap,
            vmin=minimum,
            vmax=maximum,
            alpha=alpha,
            aspect=aspect,
        )
    ax.set(
        xlim=bounds[:2],
        ylim=bounds[2:],
        xlabel="Longitude",
        ylabel="Latitude",
        title=title,
        facecolor="#dedede",
    )
    ax.grid(alpha=0.2, linewidth=0.4)
    ax.tick_params(length=2.5, pad=2)
    return ScalarMappable(norm=Normalize(minimum, maximum), cmap=colourmap)


def inventory(ax, numbered=False, thin=False):
    geometry = json.loads((AUX / "paper_map_geometry.json").read_text())
    for feature in geometry["ice_rises"]:
        for ring in feature["rings"]:
            ring = np.asarray(ring)
            ax.plot(
                ring[:, 0],
                ring[:, 1],
                color="black",
                linewidth=0.25 if thin else 0.45,
                zorder=5,
            )
    if numbered:
        for point in geometry["pin_labels"]:
            ax.annotate(
                point["label"],
                point["position"],
                xytext=point["offset"],
                textcoords="offset points",
                ha=point["ha"],
                va=point["va"],
                fontsize=7,
                bbox=dict(boxstyle="circle,pad=.12", fc="white", ec="black", lw=0.4),
                zorder=6,
            )


def save_figure(fig, output_dir, name, dpi=600, formats=("png", "pdf")):
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(output_dir / f"{name}.{extension}", dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved {name} in {output_dir}", flush=True)


def load_inputs(number):
    path = ROOT / "figure_inputs" / f"figure{number}.npz"
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Download and extract the figure inputs from the README.")
    with np.load(path, allow_pickle=False) as inputs:
        return dict(inputs)

"""Map sampling and drawing shared by the three paper figures."""

from pathlib import Path
import argparse
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon
from pyproj import Transformer
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT

AUX = Path(__file__).resolve().parents[1] / "aux_files"
TO_XY = Transformer.from_crs(4326, 3031, always_xy=True)
SHELF = (-65.9, -60.25, -70.15, -65.8)
WEST = (-65.15, -63.70, -67.52, -66.78)
NORTH = (-61.86, -60.90, -66.56, -66.08)


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


def sample_grid(x, y, read, bounds, shape):
    """Sample geographic display pixels from the nearest native grid cells.

    Only the row and column windows needed by the display are read. The
    scientific grid is neither smoothed nor interpolated to fill gaps.
    """
    height, width = shape
    longitude = bounds[0] + (np.arange(width) + 0.5) * (bounds[1] - bounds[0]) / width
    latitude = bounds[2] + (np.arange(height) + 0.5) * (bounds[3] - bounds[2]) / height
    xx, yy = TO_XY.transform(*np.meshgrid(longitude, latitude))
    columns = np.rint((xx - x[0]) / (x[1] - x[0])).astype(int)
    rows = np.rint((yy - y[0]) / (y[1] - y[0])).astype(int)
    inside = (columns >= 0) & (columns < len(x)) & (rows >= 0) & (rows < len(y))
    values = np.full(shape, np.nan, "f4")
    for first in range(max(0, int(rows.min())), min(len(y), int(rows.max()) + 1), 128):
        last = min(first + 128, len(y))
        selected = inside & (rows >= first) & (rows < last)
        if not selected.any():
            continue
        left, right = int(columns[selected].min()), int(columns[selected].max()) + 1
        block = read(np.s_[first:last, left:right])
        values[selected] = block[rows[selected] - first, columns[selected] - left]
    return values


def tidal_fields(products, bounds, shape):
    result = {}
    with h5py.File(products / "maps.h5") as source:
        x, y = source["x"][:], source["y"][:]
        for name in ("m2", "s2"):
            for suffix in ("amp", "cats_amp"):
                key = name + "_" + suffix
                result[key] = sample_grid(
                    x, y, lambda window: source[key][window] * 100, bounds, shape
                )
    return result


def optical(path, bounds, shape, stretch=False):
    height, width = shape
    transform = from_bounds(bounds[0], bounds[2], bounds[1], bounds[3], width, height)
    with rasterio.open(path) as source:
        with WarpedVRT(
            source, crs="EPSG:4326", transform=transform, width=width, height=height
        ) as view:
            rgb = np.moveaxis(view.read([1, 2, 3]), 0, -1)
    if stretch:
        rgb = rgb.astype(float)
        valid = np.all(rgb > 0, axis=-1)
        for channel in range(3):
            lower, upper = np.percentile(rgb[:, :, channel][valid], [0.5, 99.5])
            rgb[:, :, channel] = (
                np.clip((rgb[:, :, channel] - lower) / (upper - lower), 0, 1) ** 0.88
            )
        rgb[~valid] = 0.85
    return rgb


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
    from cmcrameri import cm

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


def save_figure(fig, output_dir, name, dpi=600, formats=("png", "pdf")):
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(output_dir / f"{name}.{extension}", dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved {name} in {output_dir}", flush=True)


def plot_arguments(number):
    parser = argparse.ArgumentParser(description=f"Draw paper Figure {number}.")
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--png-only", action="store_true")
    args = parser.parse_args()
    resources = json.loads(args.resources.read_text())
    resources = {
        key: (args.resources.parent / value).expanduser().resolve()
        for key, value in resources.items()
        if not key.startswith("_")
    }
    return args, resources, ("png",) if args.png_only else ("png", "pdf")

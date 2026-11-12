"""Figure 2. M2 and S2 amplitudes from SWOT and CATS."""

from .style import (
    FIGURE_SETTINGS, NORTH, OUTPUT, SHELF, WEST, inventory, load_inputs, map_panel, save_figure, set_style,
)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def draw_figure2(data, output_dir, formats=("png", "pdf")):
    set_style()
    fig = plt.figure(figsize=(7.2, 7.7), layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.05])
    settings = FIGURE_SETTINGS["figure2"]
    background = data["shelf_background"]
    for row, (constituent, maximum) in enumerate((("m2", 120), ("s2", 90))):
        panels = grid[row].subgridspec(1, 3)
        swot, cats = data[f"shelf_{constituent}_amp"], data[f"shelf_{constituent}_cats_amp"]
        for column, (values, label) in enumerate(
            zip((swot, cats, swot - cats), ("SWOT", "CATS", "SWOT − CATS"))
        ):
            ax = fig.add_subplot(panels[0, column])
            difference = column == 2
            letter = chr(ord("a") + row * 3 + column)
            title = f"({letter}) {label} {constituent[0].upper()}$_2$"
            image = map_panel(
                ax,
                SHELF,
                title,
                values,
                background,
                difference=difference,
                maximum=60 if difference else maximum,
            )
            inventory(ax, thin=True)
            for bounds, text in ((WEST, "A"), (NORTH, "B")):
                ax.add_patch(
                    Rectangle(
                        (bounds[0], bounds[2]),
                        bounds[1] - bounds[0],
                        bounds[3] - bounds[2],
                        fill=False,
                        ec="red",
                        lw=0.7,
                    )
                )
                ax.text(
                    bounds[0], bounds[3], text, color="red", fontsize=7, va="bottom"
                )
            ax.set_xticks(settings["shelf_longitude_ticks_deg"])
            ax.set_yticks(settings["shelf_latitude_ticks_deg"])
            if column:
                ax.set_ylabel("")
            if row == 0:
                ax.set_xlabel("")
            bar = fig.colorbar(
                image,
                ax=ax,
                orientation="horizontal",
                pad=0.035,
                fraction=0.05,
                aspect=28,
                extend="both" if difference else "max",
            )
            bar.set_label(
                "Amplitude difference (cm)" if difference else "Amplitude (cm)"
            )
            bar.set_ticks(
                [-60, -30, 0, 30, 60]
                if difference
                else [0, 30, 60, 90] if row else [0, 40, 80, 120]
            )

    detail = grid[2].subgridspec(1, 6, width_ratios=[1, 1, 1, 0.055, 1, 0.055])
    background = data["west_background"]
    swot, cats = data["west_m2_amp"], data["west_m2_cats_amp"]
    axes = []
    colourbars = []
    for column, values, title in [
        (0, None, "(g) Sentinel-2"),
        (1, swot, "(h) SWOT M$_2$"),
        (2, cats, "(i) CATS M$_2$"),
        (4, swot - cats, "(j) SWOT − CATS"),
    ]:
        ax = fig.add_subplot(detail[0, column])
        axes.append(ax)
        image = map_panel(
            ax,
            WEST,
            title,
            values,
            background,
            difference=column == 4,
            maximum=60 if column == 4 else 120,
            alpha=0.78 if column == 4 else 0.68,
            aspect=1 / np.cos(np.deg2rad(np.mean(WEST[2:]))),
        )
        inventory(ax, numbered=True)
        if column:
            ax.set_ylabel("")
        if column == 2:
            bar = fig.colorbar(image, cax=fig.add_subplot(detail[0, 3]), extend="max")
            bar.set_label("Amplitude (cm)")
            colourbars.append(bar.ax)
        if column == 4:
            bar = fig.colorbar(image, cax=fig.add_subplot(detail[0, 5]))
            bar.set_label("Difference (cm)")
            colourbars.append(bar.ax)
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.045, wspace=0.04, hspace=0.05)
    fig.canvas.draw()
    fig.set_layout_engine(None)
    reference = axes[1].get_position()
    position = axes[0].get_position()
    axes[0].set_position([position.x0, reference.y0, position.width, reference.height])
    for ax in colourbars:
        position = ax.get_position()
        ax.set_position([position.x0, reference.y0, position.width, reference.height])
    save_figure(fig, output_dir, "figure2", formats=formats)


if __name__ == "__main__":
    draw_figure2(load_inputs(2), OUTPUT)

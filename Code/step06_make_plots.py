"""Prepare the ATL06 comparison and draw Figures 1–3 from the completed fit.

The ICESat-2 reference keeps its pinned source window and native ATL06 DAC.
Plotting does not refit the observations or fill unsupported cells."""

from __future__ import annotations

from step_setup import CODE, REPOSITORY, argument_parser, setup, existing_file

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from layout import FILE_LISTS, grouped_files
from study_area import MASK_ID, TO_LL, TO_XY, StudyArea, nearest_grid_indices
from cats import load_cats, prepare_cats_tide, cats_tide_prepared

sys.path.insert(0, str(REPOSITORY))
from plots.figure1 import draw_figure1
from plots.figure2 import draw_figure2
from plots.figure3 import draw_figure3
from plots.style import FIGURE_SETTINGS, NORTH, SHELF, WEST


# Build the ICESat-2 reference and draw the paper figures


def make_figures(paths, work_dir):
    # 1. Build the comparison surface from the pinned ICESat-2 segments, if needed.
    if not existing_file(paths["atl06_reference"]):
        build_icesat2_reference(paths, paths["atl06_reference"].parent)

    # 2. Check the completed map and transect products.
    products_dir = work_dir / "figure_products"
    output_dir = work_dir / "figures"
    has_transect = (products_dir / "figure3.h5").is_file()
    files = ["maps.h5", "figure1.h5"]
    if has_transect:
        files.append("figure3.h5")
    for filename in files:
        with h5py.File(products_dir / filename) as product:
            assert product.attrs.get("complete") == 1
    output_dir.mkdir(parents=True, exist_ok=True)
    # 3. Draw each paper figure using the same explicit inputs.
    plot_figure1(products_dir, paths, output_dir)
    plot_figure2(products_dir, paths, output_dir)
    if has_transect:
        plot_figure3(products_dir, paths, output_dir)
    else:
        print("Skipping Figure 3: no completed transect product.", flush=True)



def main(argv=None):
    args = setup(argument_parser(__doc__).parse_args(argv))
    if args.dry_run:
        print(f"Work folder: {args.work_dir}")
        print("Build the ATL06 reference if needed; check products; draw Figures 1–3.")
        return
    make_figures(args.paths, args.work_dir)


# Sample the maps on the plotting grids


AUX = CODE / "aux_files"


def sample_grid(x, y, read, bounds, shape):
    """Sample geographic display pixels from the nearest native grid cells.

    Only the row and column windows needed by the display are read. The
    scientific grid is neither smoothed nor interpolated to fill gaps.
    """
    height, width = shape
    longitude = bounds[0] + (np.arange(width) + 0.5) * (bounds[1] - bounds[0]) / width
    latitude = bounds[2] + (np.arange(height) + 0.5) * (bounds[3] - bounds[2]) / height
    xx, yy = TO_XY.transform(*np.meshgrid(longitude, latitude))
    columns, rows, inside = nearest_grid_indices(xx, yy, x, y)
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


def tidal_fields(products, bounds, shape, constituents=("m2", "s2")):
    result = {}
    with h5py.File(products / "maps.h5") as source:
        x, y = source["x"][:], source["y"][:]
        for name in constituents:
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


# ICESat-2 corrections and comparison surface


EPOCH = np.datetime64("2018-01-01T00:00:00", "ns")


def finite(d):
    """Read a segment field and replace declared/magnitude fill values with NaN."""
    values = d[:].astype("float64")
    fill = d.attrs.get("_FillValue")
    if fill is not None:
        values[values == np.asarray(fill).ravel()[0]] = np.nan
    values[np.abs(values) > 1e20] = np.nan
    return values


def medians(h, starts, counts):
    """Get grouped medians from values already sorted by cell then height."""
    return (h[starts + (counts - 1) // 2] + h[starts + counts // 2]) / 2


def reduce_overpass(parts):
    """Combine same-overpass samples using median height and RMS uncertainty."""
    if not parts or not sum(len(part[0]) for part in parts):
        return None
    cell_ids, heights, uncertainty = [
        np.concatenate([part[field] for part in parts]) for field in range(3)
    ]
    order = np.lexsort((heights, cell_ids))
    cell_ids, heights, uncertainty = cell_ids[order], heights[order], uncertainty[order]
    cells, starts, counts = np.unique(cell_ids, return_index=True, return_counts=True)
    # Median across same-overpass footprint samples limits segment/raster blunders.
    height = medians(heights, starts, counts)
    # RMS reported uncertainty, not divided by sqrt(n): same-pass errors correlate.
    sigma = np.sqrt(np.add.reduceat(uncertainty * uncertainty, starts) / counts)
    return dict(
        cell=cells.astype("int32"),
        height=height.astype("float32"),
        sigma=sigma.astype("float32"),
    )


def save_icesat2_reference(overpass_dir, processor, output):
    """Apply the frozen temporal screen and write the overpass-balanced ICESat-2 reference."""
    paths = sorted(overpass_dir.glob("overpass_*.npz"))
    parts = []
    for path in paths:
        with np.load(path) as archive:
            parts.append((archive["cell"], archive["height"], archive["sigma"]))
    cell_ids, heights, uncertainty = [np.concatenate(field) for field in zip(*parts)]
    order = np.lexsort((heights, cell_ids))
    cell_ids, heights, uncertainty = cell_ids[order], heights[order], uncertainty[order]
    cells, starts, counts = np.unique(cell_ids, return_index=True, return_counts=True)
    median_height = medians(heights, starts, counts)
    deviation = np.abs(heights - np.repeat(median_height, counts))
    deviation_order = np.lexsort((deviation, cell_ids))
    median_deviation = medians(deviation[deviation_order], starts, counts)
    uncertainty_order = np.lexsort((uncertainty, cell_ids))
    median_uncertainty = medians(uncertainty[uncertainty_order], starts, counts)
    # Symmetric standardized within-mission screen, no fixed metre cutoff.
    threshold = 6 * np.maximum(1.4826 * median_deviation, median_uncertainty)
    keep = (np.repeat(counts, counts) < 5) | (deviation <= np.repeat(threshold, counts))
    cell_ids, heights = cell_ids[keep], heights[keep]
    order = np.lexsort((heights, cell_ids))
    cell_ids, heights = cell_ids[order], heights[order]
    cells, starts, counts = np.unique(cell_ids, return_index=True, return_counts=True)
    mean = np.add.reduceat(heights.astype("float64"), starts) / counts
    fields = {}
    values = {"mean": mean, "count": counts}
    for name, v in values.items():
        full = np.full(
            np.prod(processor.shape),
            0 if name == "count" else np.nan,
            dtype="uint16" if name == "count" else "float32",
        )
        full[cells] = v
        fields[name] = (("y", "x"), full.reshape(processor.shape))
    result = xr.Dataset(
        fields,
        coords=dict(x=processor.x, y=processor.y),
        attrs=dict(
            title="Reference-independent ATL06 overpass-balanced surface",
            crs="EPSG:3031",
            estimator="Arithmetic mean of accepted overpass cell medians",
            display_support="At least one quality-screened ATL06 overpass; repeat count is diagnostic.",
            reference_residual_filter="none",
            absolute_height_filter="none",
            empirical_plane_or_bias_correction="none",
            geographic_mask=MASK_ID
            + "; connected floating LCIS with existing east/south exclusions; no height cutoff or grounded buffer",
            temporal_screen="For >=5 overpasses: abs(height-median)<=6*max(1.4826*MAD,median reported uncertainty); no fixed metre cutoff",
            atmosphere="Native ATL06 DAC subtracted once.",
            ocean_tide="CATS2008-v2023 full available constituents and minor inference, native flexure before interpolation, removed once",
            tide_free_to_mean_tide="ATL06 h_li + tide_earth_free2mean; mean-tide geoid geoid_h+geoid_free2mean",
            atl06_quality="atl06_quality_summary 0, 0<h_li_sigma<=1 m",
        ),
    )
    result["mean"].attrs["units"] = "m"
    result.to_netcdf(
        output, encoding={name: dict(zlib=True, complevel=4) for name in fields}
    )


class ICESat2Reference:
    def __init__(self, geometry, bedmachine, cats):
        with h5py.File(geometry) as geometry_file:
            self.x, self.y = geometry_file["x100"][:], geometry_file["y100"][:]
        self.shape = (len(self.y), len(self.x))
        with np.load(bedmachine) as bedmachine_data:
            self.study_area = StudyArea(bedmachine_data)
        self.cats = prepare_cats_tide(
            np.array([self.x.min(), self.x.max()]),
            np.array([self.y.min(), self.y.max()]),
            load_cats(cats),
        )

    def geography(self, lon, lat):
        projected_x, projected_y = TO_XY.transform(lon, lat)
        column, row, valid = nearest_grid_indices(projected_x, projected_y, self.x, self.y)
        valid &= self.study_area.contains(projected_x, projected_y)
        return projected_x, projected_y, row * len(self.x) + column, valid

    def tide(self, x, y, t):
        prediction = np.full(len(x), np.nan)
        if np.ndim(t) == 0:
            t = np.full(len(x), t, dtype="datetime64[ns]")
        for start in range(0, len(x), 50000):
            stop = start + 50000
            prediction[start:stop] = cats_tide_prepared(
                x[start:stop], y[start:stop], t[start:stop], self.cats
            )
        return prediction

    def atl(self, files, start, end):
        parts = []
        for path in files:
            with h5py.File(path) as source:
                for beam in ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"):
                    segments = source.get(beam + "/land_ice_segments")
                    if segments is None:
                        continue
                    height, longitude, latitude, uncertainty, delta = [
                        finite(segments[field])
                        for field in (
                            "h_li",
                            "longitude",
                            "latitude",
                            "h_li_sigma",
                            "delta_time",
                        )
                    ]
                    geoid = finite(segments["dem/geoid_h"]) + finite(
                        segments["dem/geoid_free2mean"]
                    )
                    mean_tide_adjustment = finite(
                        segments["geophysical/tide_earth_free2mean"]
                    )
                    dac = finite(segments["geophysical/dac"])
                    day = delta / 86400
                    valid = (
                        np.isfinite(
                            height
                            + longitude
                            + latitude
                            + uncertainty
                            + day
                            + geoid
                            + mean_tide_adjustment
                            + dac
                        )
                        & (segments["atl06_quality_summary"][:] == 0)
                        & (uncertainty > 0)
                        & (uncertainty <= 1)
                        & (day >= start)
                        & (day <= end)
                    )
                    (
                        height,
                        longitude,
                        latitude,
                        uncertainty,
                        day,
                        geoid,
                        mean_tide_adjustment,
                        dac,
                    ) = [
                        values[valid]
                        for values in (
                            height,
                            longitude,
                            latitude,
                            uncertainty,
                            day,
                            geoid,
                            mean_tide_adjustment,
                            dac,
                        )
                    ]
                    projected_x, projected_y, cell, inside = self.geography(
                        longitude, latitude
                    )
                    (
                        projected_x,
                        projected_y,
                        cell,
                        height,
                        uncertainty,
                        day,
                        geoid,
                        mean_tide_adjustment,
                        dac,
                    ) = [
                        values[inside]
                        for values in (
                            projected_x,
                            projected_y,
                            cell,
                            height,
                            uncertainty,
                            day,
                            geoid,
                            mean_tide_adjustment,
                            dac,
                        )
                    ]
                    times = EPOCH + (day * 86400e9).astype("timedelta64[ns]")
                    height = (
                        height
                        - geoid
                        + mean_tide_adjustment
                        - dac
                        - self.tide(projected_x, projected_y, times)
                    )
                    finite_height = np.isfinite(height)
                    parts.append(
                        (
                            cell[finite_height],
                            height[finite_height],
                            uncertainty[finite_height],
                        )
                    )
        return parts


def build_icesat2_reference(input_paths, output):
    """Rebuild the pinned ATL06 reference from local segment files."""
    output = Path(output)
    overpass_dir = output / "atl06"
    overpass_dir.mkdir(parents=True, exist_ok=True)
    files_by_pass = grouped_files(FILE_LISTS / "icesat2_atl06.tsv")
    processor = ICESat2Reference(
        input_paths["analysis_geometry"],
        input_paths["native_bedmachine_mask"],
        input_paths["cats_model"],
    )
    first, last = np.asarray(FIGURE_SETTINGS["icesat2_window_utc"], dtype="datetime64[s]")
    start = float((first - EPOCH) / np.timedelta64(1, "D"))
    end = float((last - EPOCH) / np.timedelta64(1, "D"))
    for key, files in sorted(files_by_pass.items()):
        path = overpass_dir / f"overpass_{key}.npz"
        if path.exists():
            continue
        parts = processor.atl(
            [Path(input_paths["atl06_raw"]) / Path(f).name for f in files], start, end
        )
        reduced = reduce_overpass(parts)
        if reduced is not None:
            np.savez_compressed(path, **reduced)
    save_icesat2_reference(overpass_dir, processor, output / "atl06_dac_100m.nc")



# Figure 1


def prepare_figure1(products, resources):
    """Sample the display grids and collect native-grid comparison points."""
    bounds = json.loads((AUX / "figure1_locator.json").read_text())["comparison_bounds"]
    longitudes, latitudes, scatter_offsets = [], [], [0]
    differences = []
    with (
        h5py.File(products / "figure1.h5") as source,
        h5py.File(resources["atl06_reference"]) as icesat2,
    ):
        x, y = source["x"][:], source["y"][:]
        np.testing.assert_array_equal(x, icesat2["x"][:])
        np.testing.assert_array_equal(y, icesat2["y"][:])
        mean = sample_grid(
            x, y, lambda window: source["mean"][window], bounds, (2000, 1000)
        )

        def rema_difference(window):
            height, reference = source["mean"][window], source["rema"][window]
            return np.where(
                (reference >= 0) & (reference <= 100), height - reference, np.nan
            )

        rema = sample_grid(x, y, rema_difference, bounds, (2000, 1000))
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
            icesat2_ref = icesat2["mean"][window]
            count = icesat2["count"][window]
            valid_icesat2 = (
                inside
                & np.isfinite(height + icesat2_ref)
                & (icesat2_ref >= 0)
                & (icesat2_ref <= 100)
                & (count > 0)
                & (count < 65535)
            )
            differences.append((height - icesat2_ref)[valid_icesat2])
            longitudes.append(longitude[valid_icesat2])
            latitudes.append(latitude[valid_icesat2])
            scatter_offsets.append(scatter_offsets[-1] + int(valid_icesat2.sum()))
    return dict(
        mean=mean,
        rema_difference=rema,
        icesat2_longitude=np.concatenate(longitudes),
        icesat2_latitude=np.concatenate(latitudes),
        icesat2_difference=np.concatenate(differences),
        scatter_offsets=np.asarray(scatter_offsets),
    )


def plot_figure1(products, resources, output_dir, formats=("png", "pdf")):
    return draw_figure1(prepare_figure1(products, resources), output_dir, formats)


def prepare_figure2(products, resources):
    """Sample the shelf maps and the western detail panels in centimetres."""
    shelf = tidal_fields(products, SHELF, (1100, 1000))
    west = tidal_fields(products, WEST, (1500, 900), constituents=("m2",))
    return {
        **{f"shelf_{key}": values for key, values in shelf.items()},
        **{f"west_{key}": values for key, values in west.items()},
        "shelf_background": optical(resources["lima_mosaic"], SHELF, (1100, 1000)),
        "west_background": optical(resources["optical_mosaic"], WEST, (1500, 900), stretch=True),
    }


def plot_figure2(products, resources, output_dir, formats=("png", "pdf")):
    draw_figure2(prepare_figure2(products, resources), output_dir, formats)


def plot_figure3(products, resources, output_dir, formats=("png", "pdf")):
    names = ("distance_km", "time_ns", "longitude", "latitude", "observed", "cats_tide", "swot_tide")
    with h5py.File(products / "figure3.h5") as source:
        values = {name: source[name][:] for name in names}
    values.update(tidal_fields(products, NORTH, (1100, 1200), constituents=("m2",)))
    values["background"] = optical(resources["lima_mosaic"], NORTH, (1100, 1200))
    draw_figure3(values, output_dir, formats)


if __name__ == "__main__":
    main()

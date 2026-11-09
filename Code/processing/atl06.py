"""Build the ATL06 comparison surface from its original source window.

Subtract native DAC and the full CATS prediction, including minor inference."""

from pathlib import Path
import argparse
import json
import sys
import h5py
import numpy as np
import xarray as xr
from pyproj import Transformer
from study_area import StudyArea, MASK_ID
from cats import load_cats, prepare_cats_tide, cats_tide_prepared

EPOCH = np.datetime64("2018-01-01T00:00:00", "ns")
WORK = None


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
    cell_ids, heights, uncertainty, time_days, geoid = [
        np.concatenate([part[first] for part in parts]) for first in range(5)
    ]
    order = np.lexsort((heights, cell_ids))
    cell_ids, heights, uncertainty, time_days, geoid = [
        values[order] for values in (cell_ids, heights, uncertainty, time_days, geoid)
    ]
    cells, starts, counts = np.unique(cell_ids, return_index=True, return_counts=True)
    # Median across same-overpass footprint samples limits segment/raster blunders.
    height = medians(heights, starts, counts)
    # RMS reported uncertainty, not divided by sqrt(n): same-pass errors correlate.
    sigma = np.sqrt(np.add.reduceat(uncertainty * uncertainty, starts) / counts)
    return dict(
        cell=cells.astype("int32"),
        height=height.astype("float32"),
        sigma=sigma.astype("float32"),
        time=(np.add.reduceat(time_days, starts) / counts).astype("float64"),
        geoid=(np.add.reduceat(geoid, starts) / counts).astype("float32"),
        raw_count=counts.astype("uint32"),
    )


def finish(mission, processor, metadata=None, filename=None):
    """Apply the frozen temporal screen and write the overpass-balanced laser reference."""
    paths = sorted((WORK / mission).glob("overpass_*.npz"))
    archives = [np.load(path) for path in paths]
    cell_ids, heights, uncertainty, time_days, geoid = [
        np.concatenate([values_at_cells[name] for values_at_cells in archives])
        for name in ("cell", "height", "sigma", "time", "geoid")
    ]
    del archives
    order = np.lexsort((heights, cell_ids))
    cell_ids, heights, uncertainty, time_days, geoid = [
        v[order] for v in (cell_ids, heights, uncertainty, time_days, geoid)
    ]
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
    counters = dict(
        overpasses=len(paths),
        observations_before=int(len(cell_ids)),
        outliers=int((~keep).sum()),
    )
    cell_ids, heights, uncertainty, time_days, geoid = [
        v[keep] for v in (cell_ids, heights, uncertainty, time_days, geoid)
    ]
    order = np.lexsort((heights, cell_ids))
    cell_ids, heights, uncertainty, time_days, geoid = [
        v[order] for v in (cell_ids, heights, uncertainty, time_days, geoid)
    ]
    cells, starts, counts = np.unique(cell_ids, return_index=True, return_counts=True)
    median_height = medians(heights, starts, counts)
    mean = np.add.reduceat(heights.astype("float64"), starts) / counts
    standard_deviation = np.sqrt(
        np.add.reduceat((heights - np.repeat(mean, counts)) ** 2, starts)
        / np.maximum(counts - 1, 1)
    )
    standard_deviation[counts < 2] = np.nan
    deviation = np.abs(heights - np.repeat(median_height, counts))
    deviation_order = np.lexsort((deviation, cell_ids))
    median_deviation = medians(deviation[deviation_order], starts, counts)
    fields = {}
    values = dict(
        mean=mean,
        median=median_height,
        std=standard_deviation,
        mad=median_deviation,
        count=counts,
        time_min=np.minimum.reduceat(time_days, starts),
        time_max=np.maximum.reduceat(time_days, starts),
        geoid=np.add.reduceat(geoid.astype("float64"), starts) / counts,
    )
    for name, v in values.items():
        full = np.full(
            np.prod(processor.shape),
            0 if name == "count" else np.nan,
            dtype="uint16" if name == "count" else "float32",
        )
        full[cells] = v
        fields[name] = (("y", "x"), full.reshape(processor.shape))
    counters.update(
        retained_overpass_cells=int(counts.sum()),
        cells=int(len(cells)),
        cells_at_least_3=int((counts >= 3).sum()),
        start=str(EPOCH + np.timedelta64(int(time_days.min() * 86400), "s")),
        end=str(EPOCH + np.timedelta64(int(time_days.max() * 86400), "s")),
    )
    result = xr.Dataset(
        fields,
        coords=dict(x=processor.x, y=processor.y),
        attrs=dict(
            title=f"Reference-independent {mission} overpass-balanced surface",
            crs="EPSG:3031",
            estimator="Arithmetic mean of accepted overpass cell medians; median also provided",
            display_support="SWOT means: at least 3 overpasses. ATL06 comparison: at least 1 quality-screened overpass; repeat count is diagnostic.",
            reference_residual_filter="none",
            absolute_height_filter="none",
            empirical_plane_or_bias_correction="none",
            geographic_mask=MASK_ID
            + "; connected floating LCIS with existing east/south exclusions; no height cutoff or grounded buffer",
            temporal_screen="For >=5 overpasses: abs(height-median)<=6*max(1.4826*MAD,median reported uncertainty); no fixed metre cutoff",
            atmosphere="DAC not applied to either mission; atmospheric elevation variability remains in scatter and possible sampling bias",
            ocean_tide="CATS2008-v2023 full available constituents and minor inference, native flexure before interpolation, removed once",
            tide_free_to_mean_tide="ATL06 h_li + tide_earth_free2mean; mean-tide geoid geoid_h+geoid_free2mean",
            swot_quality="wse_qual 0 or 1; exclude geolocation_qual_suspect and large_uncert_suspect; finite positive uncertainty and n_wse_pix",
            atl06_quality="atl06_quality_summary 0, 0<h_li_sigma<=1 m",
            counters=json.dumps(counters),
        ),
    )
    for name in ("mean", "median", "std", "mad", "geoid"):
        result[name].attrs["units"] = "m"
    for name in ("time_min", "time_max"):
        result[name].attrs["units"] = "days since 2018-01-01"
    if metadata:
        result.attrs.update(metadata)
    output_path = WORK / (filename or f"{mission}_independent_100m.nc")
    result.to_netcdf(
        output_path, encoding={name: dict(zlib=True, complevel=4) for name in fields}
    )
    (WORK / f"{mission}_summary.json").write_text(json.dumps(counters, indent=2))
    print(counters, flush=True)


class Processor:
    def __init__(self, geometry, bedmachine, cats):
        with h5py.File(geometry) as geometry_file:
            self.x, self.y = geometry_file["x100"][:], geometry_file["y100"][:]
        self.shape = (len(self.y), len(self.x))
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        with np.load(bedmachine) as bedmachine_data:
            self.study_area = StudyArea(bedmachine_data)
        self.tr = Transformer.from_crs(4326, 3031, always_xy=True)
        self.cats = prepare_cats_tide(
            np.array([self.x.min(), self.x.max()]),
            np.array([self.y.min(), self.y.max()]),
            load_cats(cats),
        )

    def geography(self, lon, lat):
        projected_x, projected_y = self.tr.transform(lon, lat)
        valid = self.study_area.contains(projected_x, projected_y)
        column = np.rint((projected_x - self.x[0]) / self.dx).astype(int)
        row = np.rint((projected_y - self.y[0]) / self.dy).astype(int)
        valid &= (
            (column >= 0) & (column < len(self.x)) & (row >= 0) & (row < len(self.y))
        )
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
        audit = []
        for path in files:
            accepted_count = 0
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
                    accepted_count += int(finite_height.sum())
                    parts.append(
                        (
                            cell[finite_height],
                            height[finite_height],
                            uncertainty[finite_height],
                            day[finite_height],
                            geoid[finite_height],
                        )
                    )
            audit.append(dict(file=str(path), accepted=accepted_count))
        return parts, audit


def build_reference(input_paths, output):
    """Rebuild the pinned ATL06 reference from local segment files."""
    global WORK
    WORK = Path(output).resolve()
    (WORK / "atl06").mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1] / "aux_files/atl06_inventory.json"
        ).read_text()
    )
    processor = Processor(
        input_paths["analysis_geometry"],
        input_paths["native_bedmachine_mask"],
        input_paths["cats_model"],
    )
    start = float((np.datetime64(manifest["start"]) - EPOCH) / np.timedelta64(1, "D"))
    end = float(
        (np.datetime64(manifest["end_exclusive"]) - EPOCH) / np.timedelta64(1, "D")
    )
    for key, files in sorted(manifest["groups"].items()):
        path = WORK / "atl06" / f"overpass_{key}.npz"
        if path.exists():
            continue
        parts, audit = processor.atl(
            [Path(input_paths["atl06_raw"]) / Path(f).name for f in files], start, end
        )
        reduced = reduce_overpass(parts)
        if reduced is not None:
            np.savez_compressed(path, **reduced)
        (WORK / "atl06" / f"audit_{key}.json").write_text(json.dumps(audit, indent=2))
    finish(
        "atl06",
        processor,
        metadata=dict(
            atmosphere="Native ATL06 geophysical/dac subtracted exactly once; no extra IB or Howard correction",
            reference_residual_filter="none",
            qualification="Original archived source-window comparison reference; not fresh independent GLS accuracy validation",
        ),
        filename="atl06_dac_100m.nc",
    )


def main():
    """Rebuild the original ATL06 comparison surface from its pinned raw segments."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run import resources

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_paths = resources(args.resources)
    build_reference(input_paths, args.output)


if __name__ == "__main__":
    main()

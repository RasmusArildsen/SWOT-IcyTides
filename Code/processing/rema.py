"""Area-average the original REMA v2 32 m mosaic onto the analysis 100 m grid.

Only source pixels overlapping the destination footprint (plus two pixels) are
read. Count<=0 is excluded for every field; no SWOT-agreement filter is used.
"""

from pathlib import Path
import argparse
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds

FIELDS = (
    ("ellipsoid", "dem", Resampling.average),
    ("count", "count", Resampling.nearest),
    ("mad", "mad", Resampling.average),
    ("mindate", "mindate", Resampling.nearest),
    ("maxdate", "maxdate", Resampling.nearest),
)


def source_window(src, x, y):
    """Bound a source read to the destination footprint, with a two-pixel margin."""
    w = from_bounds(
        float(x[0] - 50),
        float(y[0] - 50),
        float(x[-1] + 50),
        float(y[-1] + 50),
        src.transform,
    )
    left = max(0, int(np.floor(w.col_off)) - 2)
    top = max(0, int(np.floor(w.row_off)) - 2)
    right = min(src.width, int(np.ceil(w.col_off + w.width)) + 2)
    bottom = min(src.height, int(np.ceil(w.row_off + w.height)) + 2)
    return Window(left, top, max(0, right - left), max(0, bottom - top))


def resample(source, x, y):
    """Resample sorted REMA tiles with the original count gate and averaging rules."""
    assert len(x) > 1 and len(y) > 1
    np.testing.assert_array_equal(np.diff(x), 100)
    np.testing.assert_array_equal(np.diff(y), 100)
    shape = (len(y), len(x))
    transform = from_origin(x[0] - 50, y[-1] + 50, 100, 100)
    fields = {key: np.full(shape, np.nan, "f4") for key, _, _ in FIELDS}
    log = []
    tiles = 0
    for path in sorted(Path(source).glob("*_dem.tif")):
        with rasterio.open(path) as dem:
            if dem.crs != rasterio.crs.CRS.from_epsg(3031):
                raise ValueError(f"Expected native EPSG:3031: {path.name}")
            if (
                dem.bounds.right < x.min()
                or dem.bounds.left > x.max()
                or dem.bounds.top < y.min()
                or dem.bounds.bottom > y.max()
            ):
                continue
            window = source_window(dem, x, y)
            if not window.width or not window.height:
                continue
            expected_transform = dem.transform
            expected_shape = dem.shape
        count_path = path.with_name(path.name.replace("_dem.tif", "_count.tif"))
        with rasterio.open(count_path) as count:
            assert (
                count.transform == expected_transform and count.shape == expected_shape
            )
            source_count = count.read(1, window=window, masked=True).filled(0)
        for key, suffix, method in FIELDS:
            field_path = path.with_name(path.name.replace("_dem.tif", f"_{suffix}.tif"))
            if not field_path.exists():
                continue
            with rasterio.open(field_path) as source_raster:
                assert (
                    source_raster.transform == expected_transform
                    and source_raster.shape == expected_shape
                )
                values = (
                    source_raster.read(1, window=window, masked=True)
                    .astype("f4")
                    .filled(np.nan)
                )
                values[source_count <= 0] = np.nan
                reproject(
                    values,
                    fields[key],
                    src_transform=source_raster.window_transform(window),
                    src_crs=source_raster.crs,
                    src_nodata=np.nan,
                    dst_transform=transform,
                    dst_crs="EPSG:3031",
                    dst_nodata=np.nan,
                    resampling=method,
                    init_dest_nodata=False,
                    num_threads=1,
                )
                if key in ("mindate", "maxdate"):
                    log.append(
                        dict(tile=path.name, variable=key, tags=source_raster.tags())
                    )
        tiles += 1
    if not tiles:
        raise FileNotFoundError("No overlapping REMA *_dem.tif tiles")
    return {key: v[::-1] for key, v in fields.items()}, log


def build(source, output):
    """Write the independently resampled 100 m REMA reference and source metadata."""
    import xarray as xr

    config = json.loads(
        (Path(__file__).resolve().parents[1] / "aux_files/analysis.json").read_text()
    )["grid"]
    axes = {
        key: config[key + "100"]["start"]
        + config[key + "100"]["step"] * np.arange(config[key + "100"]["size"])
        for key in ("x", "y")
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    fields, log = resample(source, axes["x"], axes["y"])
    result = xr.Dataset(
        {key: (("y", "x"), v) for key, v in fields.items()},
        coords=axes,
        attrs=dict(
            title="REMA v2 independently resampled to 100 m",
            elevation="Original WGS84 ellipsoidal mosaic; common geoid conversion performed downstream",
            resampling="Area-weighted average of finite 32 m elevations with source count>0; no comparison-residual or elevation-range filter",
            dates="Nearest-neighbour source minimum/maximum date rasters; not a single epoch",
            source=str(source),
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.nc")
    result.to_netcdf(
        temporary, encoding={key: dict(zlib=True, complevel=4) for key in fields}
    )
    temporary.replace(output)
    output.with_name(output.stem + "_source_metadata.json").write_text(
        json.dumps(log, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)

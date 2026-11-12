"""Read a CATS subset and apply its native flexure before interpolation.

Cropping limits memory use. The 12 km crop padding is our implementation
choice; it is not a flexure length or a CATS model parameter.
"""

import json
from pathlib import Path

import numpy as np
import pyTMD
import timescale
import xarray as xr
from pyproj import CRS, Transformer

ANALYSIS_SETTINGS = Path(__file__).resolve().parents[1] / "aux_files" / "analysis.json"
CROP_PADDING_KM = json.loads(ANALYSIS_SETTINGS.read_text())["cats_crop_padding_km"]


def load_cats(path):
    """Open the native CATS model and retain its declared constituent order."""
    return Path(path), pyTMD.io.load_database().get("CATS2008-v2023")


def _local_flexure_dataset(x3031, y3031, cats, constituents=None, buffer_km=CROP_PADDING_KM):
    """Load a small native CATS subset and apply its supplied flexure field."""
    path, model = cats
    x3031, y3031 = np.asarray(x3031), np.asarray(y3031)
    crs = CRS.from_user_input(model["projection"])
    model_x, model_y = Transformer.from_crs(3031, crs, always_xy=True).transform(
        x3031, y3031
    )
    with xr.open_dataset(path, engine="netcdf4", mask_and_scale=True) as source:
        names = source.constituents.attrs["constituent_order"].split()
        units = source.hRe.attrs.get("units", "m")
        source["constituents"] = names
        source = source.reindex(y=source.y[::-1])
        subset = source.sel(
            x=slice(
                float(np.nanmin(model_x)) - buffer_km,
                float(np.nanmax(model_x)) + buffer_km,
            ),
            y=slice(
                float(np.nanmin(model_y)) - buffer_km,
                float(np.nanmax(model_y)) + buffer_km,
            ),
        )
        if constituents is not None:
            subset = subset.sel(constituents=[str(c).lower() for c in constituents])
        # This is the operation used by pyTMD.compute.tide_elevations when
        # apply_flexure=True: scale every native-grid constituent by flexure,
        # preserving the model's permitted values above 100 percent.
        flexure = subset.flexure.astype("f") / 100.0
        tide = ((subset.hRe + -1j * subset.hIm) * flexure).to_dataset(
            dim="constituents"
        )
        tide["flexure"] = flexure
        tide = tide.load()
    for name in (str(c).lower() for c in tide.tmd.constituents):
        tide[name].attrs["units"] = units
    tide.attrs.update(
        crs=model["projection"],
        group="z",
        lineage=path.name,
        variable=model["z"]["variable"],
    )
    return tide, np.asarray(model_x), np.asarray(model_y), model


def cats_constants(
    x3031, y3031, cats, constituents=("m2", "s2", "k1", "o1"), method="linear",
    *, buffer_km=CROP_PADDING_KM,
):
    """Return officially flexure-scaled amplitude and Greenwich phase lag."""
    x3031, y3031 = np.asarray(x3031), np.asarray(y3031)
    tide, _, _, _ = _local_flexure_dataset(
        x3031, y3031, cats, constituents=constituents, buffer_km=buffer_km
    )
    xx, yy = tide.tmd.coords_as(x3031, y3031, type="drift", crs=3031)
    local = tide.tmd.interp(xx, yy, method=method, extrapolate=False)
    complex_coefficients = np.vstack(
        [np.asarray(local[str(name).lower()]).squeeze() for name in constituents]
    )
    amplitude = np.abs(complex_coefficients)
    # pyTMD stores hRe - i*hIm; phase lag is atan2(hIm,hRe).
    phase_lag = (-np.rad2deg(np.angle(complex_coefficients))) % 360.0
    return amplitude, phase_lag


def prepare_cats_tide(
    x3031_bounds, y3031_bounds, cats, constituents=None, buffer_km=CROP_PADDING_KM
):
    """Load and flexure-scale one reusable CATS subset for a fixed domain.

    Reusing the subset avoids reopening and reslicing the 1.8 GB native model
    for every ATL06 beam or acquisition. The returned object is intended for
    repeated calls to :func:`cats_tide_prepared` within one process.
    """
    tide, _, _, model = _local_flexure_dataset(
        np.asarray(x3031_bounds),
        np.asarray(y3031_bounds),
        cats,
        constituents=constituents,
        buffer_km=buffer_km,
    )
    return tide, model


def cats_tide_prepared(
    x3031, y3031, datetimes, prepared, infer_minor=True, method="linear"
):
    """Predict CATS tide using an in-memory subset from ``prepare_cats_tide``."""
    tide, model = prepared
    x3031, y3031 = np.asarray(x3031), np.asarray(y3031)
    tide_time = timescale.from_datetime(
        np.asarray(datetimes, dtype="datetime64[ns]")
    ).tide
    xx, yy = tide.tmd.coords_as(x3031, y3031, type="drift", crs=3031)
    local = tide.tmd.interp(xx, yy, method=method, extrapolate=False)
    prediction = local.tmd.predict(tide_time, corrections=model["format"])
    if infer_minor:
        prediction += local.tmd.infer(tide_time, corrections=model["format"])
    return np.asarray(prediction.tmd.to_units("m")).squeeze()

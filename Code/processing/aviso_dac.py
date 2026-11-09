#!/usr/bin/env python3
"""AVISO dynamic atmospheric correction matching Howard et al. (2026).

The AVISO/LEGOS DAC product is provided every six hours. Interpolate in time
to the tile midpoint, then use piecewise-linear interpolation in Antarctic
polar stereographic coordinates, as in Howard's MATLAB griddata call. There
is no nearest-ocean extrapolation or removal of valid zero DAC values.
Corrected height is ``wse - dac``. This does not remove ocean tides.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import netrc
import os
from pathlib import Path
import re

import numpy as np
import pyproj
import requests
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
import xarray as xr

EPOCH = datetime(1950, 1, 1)
BASE_URL = (
    "https://tds-odatis.aviso.altimetry.fr/thredds/fileServer/"
    "dataset-auxiliary-dynamic-atmospheric-correction"
)
AVISO_HOST = "tds-odatis.aviso.altimetry.fr"
METHOD_ID = "aviso_howard2026_epsg3031_linear_v1"
METHOD_DESCRIPTION = (
    "AVISO six-hourly dac_dif grids; linear time interpolation at SWOT tile "
    "midpoint; piecewise-linear Delaunay interpolation in EPSG:3031 using "
    "source latitudes south of 39 S; no spatial extrapolation; valid zeros retained"
)
_TIME_PATTERN = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")


class DACAuthenticationError(RuntimeError):
    pass


def _utc_naive(when: datetime) -> datetime:
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    return when


def tile_midpoint(source_path: str | Path) -> datetime:
    """Read the native tile interval, including intervals crossing midnight."""
    match = _TIME_PATTERN.search(Path(source_path).name)
    if match is None:
        raise ValueError(f"Cannot parse SWOT tile times from {source_path}")
    start, end = [datetime.strptime(v, "%Y%m%dT%H%M%S") for v in match.groups()]
    if end < start:
        raise ValueError(f"SWOT tile end precedes its start: {source_path}")
    return start + (end - start) / 2


def _six_hour_brackets(when: datetime) -> tuple[datetime, datetime]:
    when = _utc_naive(when)
    lower = when.replace(hour=(when.hour // 6) * 6, minute=0, second=0, microsecond=0)
    upper = lower if when == lower else lower + timedelta(hours=6)
    return lower, upper


def _filename(when: datetime) -> str:
    when = _utc_naive(when)
    cnes_day = int((when - EPOCH).total_seconds() // 86400)
    return f"dac_dif_{cnes_day}_{when.hour:02d}.nc"


def _session() -> requests.Session:
    session = requests.Session()
    user = os.environ.get("AVISO_USER")
    password = os.environ.get("AVISO_PASSWORD")
    if user and password:
        session.auth = (user.strip(), password.strip())
        return session
    auth = None
    configured = os.environ.get("AVISO_NETRC")
    candidates = ([Path(configured).expanduser()] if configured else []) + [
        Path.home() / ".netrc_aviso",
        Path.home() / ".netrc",
    ]
    for candidate in candidates:
        try:
            auth = netrc.netrc(candidate).authenticators(AVISO_HOST)
        except (FileNotFoundError, netrc.NetrcParseError):
            continue
        if auth:
            break
    if auth:
        # Copy/paste from password managers and web pages can introduce NBSP
        # or line-edge whitespace; neither is part of AVISO credentials.
        session.auth = (auth[0].strip(), auth[2].strip())
    return session


class AVISODAC:
    def __init__(
        self,
        cache_dir: str | Path,
        base_url: str = BASE_URL,
        *,
        allow_download: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.allow_download = allow_download
        self.session = _session()
        self._datasets: OrderedDict[Path, xr.Dataset] = OrderedDict()
        self._grid_axes = None
        self._triangulation = None
        self._grid_indices = None
        self._to_xy = pyproj.Transformer.from_crs(4326, 3031, always_xy=True)

    def cache_path(self, when: datetime) -> Path:
        when = _utc_naive(when)
        return self.cache_dir / str(when.year) / _filename(when)

    def download(self, when: datetime) -> Path:
        when = _utc_naive(when)
        target = self.cache_path(when)
        if target.exists():
            # Invalid existing files must fail validation, not be overwritten.
            self._open(target)
            return target
        if not self.allow_download:
            raise FileNotFoundError(f"AVISO DAC field not cached: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{when.year}/{target.name}"
        response = self.session.get(url, timeout=180)
        if response.status_code in (401, 403):
            raise DACAuthenticationError(
                "AVISO DAC authentication failed. Add a netrc entry for "
                f"{AVISO_HOST}, set AVISO_NETRC, or set AVISO_USER and "
                "AVISO_PASSWORD."
            )
        response.raise_for_status()
        if len(response.content) < 1000:
            raise IOError(f"AVISO returned an unexpectedly small file for {url}")
        temporary = target.with_suffix(".nc.part")
        temporary.write_bytes(response.content)
        try:
            self._read(temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def ensure_time(self, when: datetime) -> tuple[Path, Path]:
        lower, upper = _six_hour_brackets(when)
        first = self.download(lower)
        return first, first if upper == lower else self.download(upper)

    @staticmethod
    def _read(path: Path) -> xr.Dataset:
        with xr.open_dataset(path) as source:
            for name in ("dac", "latitude", "longitude"):
                if name not in source:
                    raise ValueError(f"Missing {name} in AVISO field {path}")
            if source.latitude.dims != ("latitude",) or source.longitude.dims != (
                "longitude",
            ):
                raise ValueError(
                    "AVISO coordinates must be one-dimensional latitude/longitude"
                )
            data = source["dac"]
            for dim in set(data.dims) - {"latitude", "longitude"}:
                if data.sizes[dim] != 1:
                    raise ValueError(f"Expected one DAC time slice in {path}")
                data = data.isel({dim: 0}, drop=True)
            units = str(data.attrs.get("units", "")).strip().lower()
            if units not in ("m", "meter", "meters", "metre", "metres"):
                raise ValueError(
                    f"Expected DAC in metres, got units={units!r} in {path}"
                )
            latitude = source.latitude.values.astype(float)
            longitude = source.longitude.values.astype(float)
            if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
                raise ValueError(f"Nonfinite AVISO grid coordinates in {path}")
            if np.any(np.abs(latitude) > 90):
                raise ValueError(f"Invalid AVISO latitude in {path}")
            rows = np.flatnonzero(latitude < -39.0)
            if rows.size < 2 or longitude.size < 2:
                raise ValueError(f"Insufficient Antarctic DAC grid support in {path}")
            data = data.isel(latitude=rows).transpose("latitude", "longitude").load()
        return data.to_dataset(name="dac")

    def _open(self, path: Path) -> xr.Dataset:
        ds = self._datasets.pop(path, None)
        if ds is None:
            ds = self._read(path)
        self._datasets[path] = ds
        while len(self._datasets) > 4:
            self._datasets.popitem(last=False)
        return ds

    def _prepare_geometry(self, ds: xr.Dataset):
        axes = (ds.longitude.values, ds.latitude.values)
        if self._grid_axes is not None and all(
            np.array_equal(a, b) for a, b in zip(axes, self._grid_axes)
        ):
            return
        lon, lat = np.meshgrid(*axes)
        x, y = self._to_xy.transform(lon.ravel(), lat.ravel())
        points = np.column_stack((x, y))
        # The geographic grid can contain coincident nodes at the pole.
        _, indices = np.unique(points, axis=0, return_index=True)
        self._grid_indices = np.sort(indices)
        self._triangulation = Delaunay(points[self._grid_indices])
        self._grid_axes = tuple(a.copy() for a in axes)

    def _spatial_sample(self, ds: xr.Dataset, lon, lat) -> np.ndarray:
        lon, lat = np.broadcast_arrays(
            np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        )
        result = np.full(lon.shape, np.nan)
        valid = np.isfinite(lon) & np.isfinite(lat) & (lat >= -90.0) & (lat < -39.0)
        if not np.any(valid):
            return result
        self._prepare_geometry(ds)
        # Keep masked DAC vertices: removing them would bridge unsupported areas.
        values = ds.dac.values.astype(float).ravel()[self._grid_indices]
        interpolator = LinearNDInterpolator(
            self._triangulation, values, fill_value=np.nan
        )
        x, y = self._to_xy.transform(lon[valid], lat[valid])
        result[valid] = interpolator(x, y)
        return result

    def provenance(self, when: datetime, source_path=None) -> dict:
        when = (
            tile_midpoint(source_path) if source_path is not None else _utc_naive(when)
        )
        lower, upper = _six_hour_brackets(when)
        weight = 0.0 if lower == upper else (when - lower) / (upper - lower)
        return dict(
            method=METHOD_ID,
            description=METHOD_DESCRIPTION,
            time_utc=when.isoformat() + "Z",
            time_source="tile_midpoint" if source_path is not None else "explicit_utc",
            lower_time_utc=lower.isoformat() + "Z",
            upper_time_utc=upper.isoformat() + "Z",
            upper_weight=weight,
            source_files=list(
                dict.fromkeys(str(self.cache_path(t).resolve()) for t in (lower, upper))
            ),
            sign="corrected_height = input_height - dac",
        )

    def sample(self, when: datetime, lon, lat, source_path=None) -> np.ndarray:
        when = (
            tile_midpoint(source_path) if source_path is not None else _utc_naive(when)
        )
        lower, upper = _six_hour_brackets(when)
        lower_path, upper_path = self.ensure_time(when)
        a = self._open(lower_path)
        if lower == upper:
            return self._spatial_sample(a, lon, lat)
        b = self._open(upper_path)
        if not all(
            np.array_equal(a[name], b[name]) for name in ("longitude", "latitude")
        ):
            raise ValueError("Bracketing AVISO fields have different grids")
        weight = (when - lower).total_seconds() / (upper - lower).total_seconds()
        field = a.copy(deep=False)
        field["dac"] = a.dac + weight * (b.dac - a.dac)
        return self._spatial_sample(field, lon, lat)


def apply_dac(wse, dac):
    """Howard et al. convention: subtract the modelled ocean-height DAC."""
    return np.asarray(wse, dtype=float) - np.asarray(dac, dtype=float)

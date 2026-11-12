"""Canonical LCIS geographic study area, adopted 2026-09-18.

Floating BedMachine LCIS component only; existing east/south exclusions retained.
No elevation cutoff, grounded buffer, or dependence on observation availability.
"""

from pathlib import Path
from functools import lru_cache
import json
import os
import numpy as np
from scipy import ndimage
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent
DEFAULT_BEDMACHINE = (
    ROOT / "output/independent_surfaces/source_metadata/larsen_bedmachine_v4_mask.npz"
)
STUDY_AREA = json.loads((ROOT.parent / "aux_files/study_area.json").read_text())
MASK_ID = STUDY_AREA["mask_id"]
TO_LL = Transformer.from_crs(3031, 4326, always_xy=True)
TO_XY = Transformer.from_crs(4326, 3031, always_xy=True)


def nearest_grid_indices(x, y, grid_x, grid_y, *, pad=0, spacing_m=None):
    """Return nearest column, row and in-grid mask on regularly spaced axes."""
    x, y = np.asarray(x), np.asarray(y)
    if spacing_m is None:
        dx, dy = grid_x[1] - grid_x[0], grid_y[1] - grid_y[0]
    else:
        dx = dy = spacing_m
    finite = np.isfinite(x) & np.isfinite(y)
    column = np.rint(np.where(finite, (x - grid_x[0]) / dx, 0)).astype("i8") + pad
    row = np.rint(np.where(finite, (y - grid_y[0]) / dy, 0)).astype("i8") + pad
    inside = (
        finite
        & (column >= 0) & (column < len(grid_x) + 2 * pad)
        & (row >= 0) & (row < len(grid_y) + 2 * pad)
    )
    return column, row, inside


def nearest_grid_cells(x, y, grid_x, grid_y, *, pad=0, spacing_m=None):
    """Return row-major cell IDs, with -1 for points outside the grid."""
    column, row, inside = nearest_grid_indices(
        x, y, grid_x, grid_y, pad=pad, spacing_m=spacing_m
    )
    return np.where(inside, row * (len(grid_x) + 2 * pad) + column, -1)


class StudyArea:
    def __init__(self, bed, definition=None):
        definition = STUDY_AREA if definition is None else definition
        self.bounds = definition["bounds_west_east_south_north"]
        front = np.asarray(definition["front_lat_lon"])
        self.front = front[np.argsort(front[:, 0])]
        self.south = np.asarray(definition["south_lon_lat"])
        self.x = np.asarray(bed["x"])
        self.y = np.asarray(bed["y"])
        classes = np.asarray(bed["mask"])
        labels, _ = ndimage.label(classes == 3, np.ones((3, 3)))
        sx, sy = TO_XY.transform(*definition["floating_component_seed_lonlat"])
        i, j, inside = nearest_grid_indices(sx, sy, self.x, self.y)
        if not inside:
            raise ValueError("Floating-component seed is outside the BedMachine crop")
        label = labels[j, i]
        if label == 0:
            raise ValueError("Floating-component seed is not on mapped floating ice")
        self.floating = labels == label

    def contains(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        i, j, inside = nearest_grid_indices(x, y, self.x, self.y)
        inside &= self.floating[
            np.clip(j, 0, len(self.y) - 1), np.clip(i, 0, len(self.x) - 1)
        ]
        lon, lat = TO_LL.transform(x, y)
        west, east, south, north = self.bounds
        inside &= (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
        inside &= lon <= np.interp(lat, self.front[:, 0], self.front[:, 1])
        inside &= lat >= np.interp(lon, self.south[:, 0], self.south[:, 1])
        return inside


@lru_cache(maxsize=4)
def load_area(path=None):
    path = Path(path or os.environ.get("LCIS_BEDMACHINE_MASK", DEFAULT_BEDMACHINE))
    with np.load(path) as bed:
        return StudyArea(bed)


def study_area_mask(x, y, bedmachine_path=None):
    """Mask broadcastable EPSG:3031 point coordinates."""
    return load_area(bedmachine_path).contains(x, y)


def study_area_grid(x, y, bedmachine_path=None):
    result = np.empty((len(y), len(x)), bool)
    for row in range(0, len(y), 128):
        result[row : row + 128] = study_area_mask(
            np.asarray(x)[None, :],
            np.asarray(y)[row : row + 128, None],
            bedmachine_path,
        )
    return result

"""Canonical LCIS geographic study area, adopted 2026-09-18.

Floating BedMachine LCIS component only; existing east/south exclusions retained.
No elevation cutoff, grounded buffer, or dependence on observation availability.
"""

from pathlib import Path
from functools import lru_cache
import os
import numpy as np
from scipy import ndimage
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent
DEFAULT_BEDMACHINE = (
    ROOT / "output/independent_surfaces/source_metadata/larsen_bedmachine_v4_mask.npz"
)
MASK_ID = "lcis_bedmachine_v4_floating_20260918"
FRONT = np.array(
    [
        (-65.80, -60.25),
        (-66.65, -60.25),
        (-66.72, -60.70),
        (-66.84, -61.05),
        (-67.02, -61.32),
        (-67.18, -61.50),
        (-67.45, -61.55),
        (-67.82, -61.55),
        (-68.20, -61.53),
        (-68.42, -61.42),
        (-68.58, -61.18),
        (-68.60, -61.03),
        (-68.68, -60.79),
        (-68.78, -60.78),
        (-68.88, -61.06),
        (-68.98, -61.08),
        (-69.08, -60.91),
        (-69.14, -60.73),
        (-69.20, -60.47),
        (-69.28, -60.25),
        (-70.15, -60.25),
    ]
)
SOUTH = np.array(
    [
        (-65.75, -70.12),
        (-62.90, -70.12),
        (-62.78, -69.73),
        (-62.55, -69.48),
        (-62.15, -69.42),
        (-61.65, -69.40),
        (-61.20, -69.28),
        (-60.80, -69.42),
        (-60.15, -69.20),
    ]
)
LL = Transformer.from_crs(3031, 4326, always_xy=True)
POLAR = Transformer.from_crs(4326, 3031, always_xy=True)


class StudyArea:
    def __init__(self, bed):
        self.x = np.asarray(bed["x"])
        self.y = np.asarray(bed["y"])
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        classes = np.asarray(bed["mask"])
        labels, _ = ndimage.label(classes == 3, np.ones((3, 3)))
        sx, sy = POLAR.transform(-62.5, -67.5)
        i = int(np.rint((sx - self.x[0]) / self.dx))
        j = int(np.rint((sy - self.y[0]) / self.dy))
        label = labels[j, i]
        if label == 0:
            raise ValueError("Central LCIS seed is not mapped floating ice")
        self.floating = labels == label

    def contains(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        finite = np.isfinite(x) & np.isfinite(y)
        i = np.rint(np.where(finite, (x - self.x[0]) / self.dx, 0)).astype(int)
        j = np.rint(np.where(finite, (y - self.y[0]) / self.dy, 0)).astype(int)
        inside = finite & (i >= 0) & (i < len(self.x)) & (j >= 0) & (j < len(self.y))
        inside &= self.floating[
            np.clip(j, 0, len(self.y) - 1), np.clip(i, 0, len(self.x) - 1)
        ]
        lon, lat = LL.transform(x, y)
        inside &= (lon >= -65.9) & (lon <= -60.25) & (lat >= -70.15) & (lat <= -65.8)
        order = np.argsort(FRONT[:, 0])
        inside &= lon <= np.interp(lat, FRONT[order, 0], FRONT[order, 1])
        inside &= lat >= np.interp(lon, SOUTH[:, 0], SOUTH[:, 1])
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

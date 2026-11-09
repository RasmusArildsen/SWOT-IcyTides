"""Crop BedMachine and prepare the analysis mask, including its 1 km extension."""

from pathlib import Path
import argparse
import json
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
from study_area import StudyArea


def mask(source, destination, expanded=False):
    """Crop BedMachine and optionally reproduce the fixed 1 km inland extension."""
    window = json.loads(
        (
            Path(__file__).resolve().parents[1] / "aux_files/bedmachine_windows.json"
        ).read_text()
    )["native"]
    axes = {
        axis: window[axis]["start"]
        + window[axis]["step"] * np.arange(window[axis]["size"])
        for axis in ["x", "y"]
    }
    with h5py.File(source) as handle:
        indices = {}
        for axis in ["x", "y"]:
            source_axis = handle[axis][:]
            order = np.argsort(source_axis)
            positions = np.searchsorted(source_axis[order], axes[axis])
            np.testing.assert_array_equal(source_axis[order[positions]], axes[axis])
            indices[axis] = order[positions]
        columns, rows = indices["x"], indices["y"]
        crop_window = (
            slice(rows.min(), rows.max() + 1),
            slice(columns.min(), columns.max() + 1),
        )
        mask_values = handle["mask"][crop_window]
        mask_values = mask_values[
            (rows - rows.min())[:, None], (columns - columns.min())[None, :]
        ]
    data = {**axes, "mask": mask_values}
    if expanded:
        original = StudyArea(data).floating
        change = (mask_values == 2) & (
            distance_transform_edt(~original, sampling=500) <= 1000
        )
        mask_values[change] = 3
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    np.savez_compressed(
        destination,
        **data,
        source=str(source),
        metadata=json.dumps(
            dict(
                base="BedMachineAntarcticaV04.1",
                inland_margin_m=1000 if expanded else 0,
                margin_source_class=2,
                component_seed_lonlat=[-62.5, -67.5],
            )
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bedmachine", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expanded", action="store_true")
    a = p.parse_args()
    mask(a.bedmachine, a.output, a.expanded)

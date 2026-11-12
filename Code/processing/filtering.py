"""Shared REMA screening rule and saved-mask checks."""

import json
import h5py
import numpy as np
from layout import SETTINGS, band_indices

SCREENING = SETTINGS["screening"]


def check_file(handle, source, acquisitions):
    """Reject incomplete masks, different source files or changed thresholds."""
    stat = source.stat()
    assert handle.attrs["complete"] == 1
    assert handle.attrs["acquisitions"] == acquisitions
    assert handle.attrs["source_bytes"] == stat.st_size
    assert handle.attrs["source_mtime_ns"] == stat.st_mtime_ns
    assert json.loads(handle.attrs["screening_json"]) == SCREENING


def check_complete(root, acquisitions):
    """Require the masks for all geometry bands before starting the numbered fit step."""
    for band in band_indices(root / "geometry.h5"):
        source = root / "inputs" / f"{band:03d}.h5"
        with h5py.File(root / "admission" / f"{band:03d}.h5") as handle:
            check_file(handle, source, acquisitions)
    print("All geometry-band admission masks are complete", flush=True)


def checked_mask(root, nodes, geometry, expected):
    """Read saved selections and require exact agreement with the original rule.

    Older run directories without saved masks retain the original read-time
    calculation. A saved mask can never relax or replace that calculation.
    """
    folder = root / "admission"
    if not folder.is_dir():
        return expected
    saved = np.zeros_like(expected)
    for band in np.unique(geometry["source_band"][nodes]):
        destination = np.flatnonzero(geometry["source_band"][nodes] == band)
        local = geometry["source_local"][nodes[destination]]
        source = root / "inputs" / f"{int(band):03d}.h5"
        with h5py.File(folder / f"{int(band):03d}.h5") as handle:
            check_file(handle, source, expected.shape[0])
            np.testing.assert_array_equal(
                handle["cell_id"][local], geometry["cell_id"][nodes[destination]]
            )
            saved[:, destination] = handle["admitted"][:, local]
    np.testing.assert_array_equal(
        saved,
        expected,
        err_msg="Saved mask differs from the REMA screening rule",
    )
    return saved

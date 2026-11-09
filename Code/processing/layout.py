"""Acquisition selection and the partition layout used by the processing scripts."""

from pathlib import Path
import json

AUX = Path(__file__).resolve().parents[1] / "aux_files"
SETTINGS = json.loads((AUX / "analysis.json").read_text())
PARTITIONS = json.loads((AUX / "processing_layout.json").read_text())
DEFAULT_ACQUISITIONS = SETTINGS["default_acquisitions"]


def acquisition_counts(settings=None):
    settings = SETTINGS if settings is None else settings
    base_count = len(settings["acquisitions"])
    return (
        (base_count, base_count + 1)
        if settings.get("added_acquisition")
        else (base_count,)
    )


def selected_acquisitions(count, settings=None):
    """Select the original catalogue, optionally including its recorded addition."""
    settings = SETTINGS if settings is None else settings
    if count not in acquisition_counts(settings):
        raise ValueError(f"Acquisition count {count} is not in the supplied catalogue")
    dates = list(settings["acquisitions"])
    if count > len(dates):
        added = settings["added_acquisition"]
        dates.append(dict(added, id=added["acquisition_id"]))
    return dates


def band_indices(geometry):
    """Read band IDs from the geometry, using bounded reads for older files."""
    import h5py
    import numpy as np

    with h5py.File(geometry) as source:
        if "band_ids" in source.attrs:
            bands = tuple(int(value) for value in source.attrs["band_ids"])
        else:
            bands = set()
            values = source["source_band"]
            for start in range(0, len(values), 65536):
                bands.update(
                    int(value) for value in np.unique(values[start : start + 65536])
                )
            bands = tuple(sorted(bands))
    if not bands or len(set(bands)) != len(bands) or any(band < 0 for band in bands):
        raise ValueError(f"Invalid band IDs in {geometry}")
    return bands


def fit_parts_per_band(geometry):
    import h5py

    with h5py.File(geometry) as source:
        count = int(
            source.attrs.get("fit_parts_per_band", PARTITIONS["fit_parts_per_band"])
        )
    if count < 1:
        raise ValueError("fit_parts_per_band must be positive")
    return count


def fit_part_indices(geometry):
    """Return the complete set of independent fit parts for this geometry."""
    return range(len(band_indices(geometry)) * fit_parts_per_band(geometry))

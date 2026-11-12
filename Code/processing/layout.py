"""Acquisition selection and the partition layout used by the processing scripts."""

from pathlib import Path
import csv
import json

import numpy as np

CODE = Path(__file__).resolve().parents[1]
AUX = CODE / "aux_files"
FILE_LISTS = CODE.parent / "file_lists"


def grouped_files(path):
    """Read the acquisition and filename columns in a source list."""
    groups = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            groups.setdefault(row["acquisition"], []).append(row["filename"])
    return groups


SETTINGS = json.loads((AUX / "analysis.json").read_text())
PARTITIONS = json.loads((AUX / "processing_layout.json").read_text())
RASTER_FILES = grouped_files(FILE_LISTS / "swot_raster.tsv")
DEFAULT_ACQUISITIONS = len(RASTER_FILES)


def selected_acquisitions(count=None):
    """Read the selected acquisitions in Raster file-list order."""
    count = DEFAULT_ACQUISITIONS if count is None else count
    if not 1 <= count <= DEFAULT_ACQUISITIONS:
        raise ValueError(
            f"Choose an acquisition count between 1 and {DEFAULT_ACQUISITIONS}"
        )
    return [
        {"id": acquisition, "tiles": tiles}
        for acquisition, tiles in list(RASTER_FILES.items())[:count]
    ]


def geometry_acquisition_ids(geometry):
    """Read the acquisition order stored with a run's grids."""
    import h5py

    with h5py.File(geometry) as handle:
        return handle["acquisition_id"].asstr()[:].tolist()


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


def dump(path, value):
    """Write a small JSON receipt atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def ds(f, key, value):
    """Write an HDF5 array with the standard lossless compression."""
    array = np.asarray(value)
    return (
        f.create_dataset(
            key, data=array, compression="gzip", compression_opts=2, shuffle=True
        )
        if array.ndim and array.size
        else f.create_dataset(key, data=array)
    )


def axis(name):
    """Construct one fixed grid axis from analysis.json."""
    definition = SETTINGS["grid"][name]
    return definition["start"] + definition["step"] * np.arange(definition["size"])


def raster_path(folder, title, acquisition):
    folder = Path(folder)
    for path in (folder / title, folder / acquisition / title):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Pinned Raster file missing: {title}")

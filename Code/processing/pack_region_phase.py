"""Pack completed native region corrections as a sparse 100 m layer.

The source native archives include the union of original and horizontally
moved 100 m cells. `after_wse` is stored at the moved cell location; NaN means
the original cell has no corrected native replacement. Only affected cells are
packed, excluding the separate unchanged northern native context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def as_text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def build(manifest: Path, roots: list[Path], destination: Path) -> dict:
    """Pack completed native-region archives into one sparse acquisition layer."""
    names = list(json.loads(manifest.read_text())["groups"])
    acquisition_count = len(names)
    if not names:
        raise ValueError("The acquisition manifest is empty")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".building")
    if temporary.exists():
        raise FileExistsError(temporary)
    string = h5py.string_dtype("utf-8")
    with h5py.File(temporary, "w") as out:
        out.attrs["schema"] = "s4is-native-phase-region-sparse-layer-v1"
        out.attrs["corrections_applied"] = (
            "native PIXC region-wide accepted integer shifts"
        )
        out.attrs["horizontal_correction"] = (
            "yes: corrected native longitude and latitude regridded to EPSG:3031 100m cell_id"
        )
        out.attrs["vertical_correction"] = (
            "yes: corrected native height and geophysical datum components; WSE above geoid in metres"
        )
        out.attrs["no_quality_filter_claim"] = (
            "The native assessment has source quality safeguards; the Raster base remains separately unfiltered."
        )
        out.attrs["phase_correction_stage"] = (
            "region-wide only; residual-pixel stage is separate"
        )
        out.create_dataset(
            "acquisition_id", data=np.asarray(names, dtype=object), dtype=string
        )
        start = out.create_dataset(
            "observation_start", shape=(acquisition_count + 1,), dtype="i8"
        )
        for name, dtype in [
            ("cell_id", "i8"),
            ("before_wse", "f4"),
            ("after_wse", "f4"),
            ("region_affected", "u1"),
        ]:
            out.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(65536,),
                compression="gzip",
                compression_opts=2,
            )
        out.create_dataset("source_archive", shape=(acquisition_count,), dtype=string)
        out.create_dataset("source_report", shape=(acquisition_count,), dtype=string)
        counts = []
        cursor = 0
        for index, name in enumerate(names):
            root = next(
                (
                    root
                    for root in roots
                    if (root / "shelf_native_passes" / f"{name}.json").exists()
                ),
                None,
            )
            if root is None:
                raise FileNotFoundError(name)
            report_path = root / "shelf_native_passes" / f"{name}.json"
            archive = root / "shelf_native_passes" / f"{name}.npz"
            report = json.loads(report_path.read_text())
            if report.get("status") != "assessed":
                raise ValueError(f"Incomplete first-stage report: {name}")
            with np.load(archive) as source:
                ids = source["cell"].astype("i8")
                affected = source["affected"].astype(bool)
                before = source["raw"].astype("f4")
                after = source["corrected"].astype("f4")
            if not (len(ids) == len(affected) == len(before) == len(after)):
                raise ValueError(f"Native archive length mismatch: {name}")
            ids, before, after = ids[affected], before[affected], after[affected]
            if len(ids) != int(report["affected_cells"]):
                raise ValueError(f"Affected-cell count mismatch: {name}")
            if len(ids) and (np.any(np.diff(ids) <= 0) or np.any(ids < 0)):
                raise ValueError(f"Cell IDs not strictly increasing: {name}")
            begin = cursor
            stop = begin + len(ids)
            for key, values in [
                ("cell_id", ids),
                ("before_wse", before),
                ("after_wse", after),
                ("region_affected", np.ones(len(ids), "u1")),
            ]:
                ds = out[key]
                ds.resize((stop,))
                ds[begin:stop] = values
            start[index] = begin
            cursor = stop
            out["source_archive"][index] = str(archive)
            out["source_report"][index] = str(report_path)
            counts.append(len(ids))
            if index % 20 == 0:
                print("PACKED_REGION", index, name, len(ids), flush=True)
        start[acquisition_count] = sum(counts)
        out.attrs["acquisitions"] = acquisition_count
        out.attrs["affected_cells_total"] = int(sum(counts))
        out.flush()
    temporary.rename(destination)
    with h5py.File(destination) as f:
        starts = f["observation_start"][:]
        if (
            len(starts) != acquisition_count + 1
            or starts[0] != 0
            or starts[-1] != len(f["cell_id"])
        ):
            raise ValueError("Packed layer offsets invalid")
    result = dict(
        output=str(destination),
        acquisitions=acquisition_count,
        affected_cells=int(sum(counts)),
        affected_acquisitions=sum(n > 0 for n in counts),
        max_acquisition_cells=max(counts),
    )
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--root", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.manifest, a.root, a.output), indent=2))

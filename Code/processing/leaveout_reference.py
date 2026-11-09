"""Precompute central order statistics for exact leave-acquisition-out SWOT medians.

Reads the acquisition histories in bounded spatial chunks. For each 100 m
cell, three central sorted values suffice to remove any one acquisition from
the temporal median without rereading all dates for every native PIXC tile.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import h5py
import numpy as np


def one(source_path: Path, destination: Path, coarse_chunk: int = 1500) -> None:
    """Save middle order statistics needed for leave-one-acquisition-out medians."""
    if destination.exists():
        with h5py.File(destination) as existing:
            if existing.attrs.get("complete"):
                return
        raise RuntimeError(f"Incomplete reference file: {destination}")
    temporary = destination.with_suffix(".building")
    if temporary.exists():
        raise RuntimeError(f"Stale partial reference: {temporary}")
    with h5py.File(source_path) as source, h5py.File(temporary, "w") as output:
        height_dataset = source["height"]
        weight_dataset = source["weight"]
        if (
            height_dataset.shape != weight_dataset.shape
            or height_dataset.ndim != 3
            or height_dataset.shape[0] == 0
        ):
            raise ValueError((height_dataset.shape, weight_dataset.shape))
        date_count, coarse_count, fine_positions = height_dataset.shape
        fine_count = coarse_count * fine_positions
        if fine_count == 0:
            raise ValueError("Reference history has no fine positions")
        output.create_dataset(
            "fine_id",
            shape=(fine_count,),
            dtype="i8",
            chunks=(min(65536, fine_count),),
            compression="gzip",
            compression_opts=2,
        )
        for key, dtype in [("count", "u2"), ("lo", "f4"), ("mid", "f4"), ("hi", "f4")]:
            output.create_dataset(
                key,
                shape=(fine_count,),
                dtype=dtype,
                chunks=(min(65536, fine_count),),
                compression="gzip",
                compression_opts=2,
            )
        for begin in range(0, coarse_count, coarse_chunk):
            stop = min(begin + coarse_chunk, coarse_count)
            ids = source["fine_id"][begin:stop].ravel()
            heights = height_dataset[:, begin:stop, :].reshape(date_count, -1)
            weights = weight_dataset[:, begin:stop, :].reshape(date_count, -1)
            heights[~(weights > 0)] = np.nan
            heights[:, ids < 0] = np.nan
            sorted_heights = np.sort(heights, axis=0)
            count = np.isfinite(heights).sum(axis=0).astype("u2")
            column = np.arange(len(ids))
            middle = count.astype("i4") // 2
            lo = sorted_heights[np.maximum(0, middle - 1), column].astype("f4")
            mid = sorted_heights[np.minimum(date_count - 1, middle), column].astype(
                "f4"
            )
            hi = sorted_heights[np.minimum(date_count - 1, middle + 1), column].astype(
                "f4"
            )
            for values in (lo, mid, hi):
                values[count < 1] = np.nan
            sl = slice(begin * fine_positions, stop * fine_positions)
            output["fine_id"][sl] = ids
            output["count"][sl] = count
            output["lo"][sl] = lo
            output["mid"][sl] = mid
            output["hi"][sl] = hi
        output.attrs["complete"] = True
        output.attrs["source"] = str(source_path)
        output.attrs["method"] = (
            "Temporal order statistics after weight>0; exact LOO median from lo,mid,hi and current date value"
        )
        output.flush()
    temporary.rename(destination)
    print("REFERENCE_BAND", source_path.name, fine_count, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--band-inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--band", type=int, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    one(a.band_inputs / f"{a.band:03d}.h5", a.output / f"{a.band:03d}.h5")

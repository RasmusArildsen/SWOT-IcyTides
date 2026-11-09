"""Read, inspect, and export acquisitions from the sparse raw SWOT cube."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import json
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd

DEFAULT_FIELDS = (
    "wse",
    "wse_uncert",
    "wse_qual",
    "wse_qual_bitwise",
    "geoid",
    "solid_earth_tide",
    "load_tide_fes",
    "pole_tide",
    "model_dry_tropo_cor",
    "model_wet_tropo_cor",
    "iono_cor_gim_ka",
    "illumination_time",
)


def _text(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


class RawCube(AbstractContextManager):
    """Small read-only interface to an S4IS sparse raw cube.

    Open the cube with a context manager so the HDF5 handle always closes::

        with RawCube("data/lcis_raw_version_d_100m.h5") as cube:
            print(cube.acquisitions().head())
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file: h5py.File | None = None

    def __enter__(self) -> "RawCube":
        try:
            self._file = h5py.File(self.path, "r")
        except OSError as exc:
            message = str(exc).lower()
            if "lock" in message or "resource temporarily unavailable" in message:
                raise RuntimeError(
                    f"{self.path} is currently being written. Wait for the raw-cube "
                    "build to finish before opening it in the notebook."
                ) from exc
            raise
        return self

    def __exit__(self, *exc_info):
        if self._file is not None:
            self._file.close()
            self._file = None
        return False

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError("Use RawCube inside a 'with' block")
        return self._file

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(sorted(self.file["observations"].keys()))

    def summary(self) -> dict:
        source = self.file
        return {
            "path": str(self.path),
            "schema": str(source.attrs.get("schema", "")),
            "resolution_m": int(source.attrs["resolution_m"]),
            "x_cells": int(len(source["grid/x"])),
            "y_cells": int(len(source["grid/y"])),
            "acquisitions": int(len(source["acquisitions/id"])),
            "tiles_loaded": int(len(source["tiles/title"])),
            "observations": int(len(source["observations/cell_id"])),
            "variables": len(self.variables),
            "corrections_applied": str(source.attrs.get("corrections_applied", "")),
            "quality_filter_applied": str(
                source.attrs.get("quality_filter_applied", "")
            ),
        }

    def tiles(self, acquisition: str | None = None) -> pd.DataFrame:
        source = self.file
        tiles = source["tiles"]
        starts = np.asarray(tiles["observation_start"][:], dtype="i8")
        rows = {
            "tile_index": np.arange(len(tiles["title"]), dtype="i4"),
            "title": _text(tiles["title"][:]),
            "acquisition": _text(tiles["acquisition"][:]),
            "product": _text(tiles["product"][:]),
            "revision_id": np.asarray(tiles["revision_id"][:], dtype="i4"),
            "source_crs": _text(tiles["source_crs"][:]),
            "observation_start": starts[:-1],
            "observation_stop": starts[1:],
            "observation_count": np.diff(starts),
        }
        result = pd.DataFrame(rows)
        if acquisition is not None:
            result = result.loc[result["acquisition"] == acquisition]
        return result.reset_index(drop=True)

    def acquisitions(self) -> pd.DataFrame:
        source = self.file
        acquisitions = source["acquisitions"]
        result = pd.DataFrame(
            {
                "acquisition": _text(acquisitions["id"][:]),
                "cycle": np.asarray(acquisitions["cycle"][:]),
                "pass": np.asarray(acquisitions["pass"][:]),
                "start_utc": pd.to_datetime(
                    _text(acquisitions["start_utc"][:]), utc=True
                ),
                "stop_utc": pd.to_datetime(
                    _text(acquisitions["stop_utc"][:]), utc=True
                ),
            }
        )
        completed = (
            self.tiles()
            .groupby("acquisition", as_index=False)
            .agg(
                tiles_loaded=("title", "size"),
                observation_count=("observation_count", "sum"),
            )
        )
        return result.merge(completed, on="acquisition", how="left").fillna(
            {"tiles_loaded": 0, "observation_count": 0}
        )

    def source_attributes(self, tile_index: int) -> dict:
        value = self.file["tiles/source_attributes_json"][tile_index]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(value)

    def read_acquisition(
        self,
        acquisition: str,
        *,
        fields: Sequence[str] = DEFAULT_FIELDS,
        quality: Iterable[int] | None = None,
        max_points: int | None = None,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Return one overpass as a table, optionally filtered and sampled."""

        unknown = sorted(set(fields) - set(self.variables))
        if unknown:
            raise KeyError(f"Variables absent from cube: {unknown}")
        requested = list(dict.fromkeys(("cell_id", *fields)))
        if quality is not None and "wse_qual" not in requested:
            requested.append("wse_qual")
        tiles = self.tiles(acquisition)
        if tiles.empty:
            raise KeyError(f"Acquisition {acquisition!r} is not loaded in the cube")

        observations = self.file["observations"]
        tile_rows = list(tiles.itertuples(index=False))

        # Decide which local rows to read before loading the large float fields.
        # A plotting sample therefore uses memory proportional to max_points,
        # rather than to the full multi-million-point overpass.
        local_candidates: list[np.ndarray | None] = []
        candidate_counts = []
        for row in tile_rows:
            count = int(row.observation_count)
            if quality is None:
                local = None
                selected_count = count
            else:
                start, stop = int(row.observation_start), int(row.observation_stop)
                flags = np.asarray(observations["wse_qual"][start:stop])
                local = np.flatnonzero(np.isin(flags, list(quality)))
                selected_count = len(local)
            local_candidates.append(local)
            candidate_counts.append(selected_count)

        total = int(sum(candidate_counts))
        if max_points is not None and total > max_points:
            global_rows = np.sort(
                np.random.default_rng(seed).choice(total, max_points, replace=False)
            )
            ends = np.cumsum(candidate_counts)
            tile_for_row = np.searchsorted(ends, global_rows, side="right")
            starts = np.r_[0, ends[:-1]]
            selections: list[np.ndarray | slice] = []
            for tile_number, local in enumerate(local_candidates):
                relative = (
                    global_rows[tile_for_row == tile_number] - starts[tile_number]
                )
                selections.append(relative if local is None else local[relative])
        else:
            selections = [
                slice(None) if local is None else local for local in local_candidates
            ]

        pieces: dict[str, list[np.ndarray]] = {name: [] for name in requested}
        tile_indices = []
        for row, selection in zip(tile_rows, selections):
            start, stop = int(row.observation_start), int(row.observation_stop)
            if isinstance(selection, slice):
                source_selection = slice(start, stop)
                count = stop - start
            else:
                source_selection = start + selection
                count = len(selection)
            for name in requested:
                pieces[name].append(np.asarray(observations[name][source_selection]))
            tile_indices.append(np.full(count, int(row.tile_index), dtype="i4"))
        arrays = {name: np.concatenate(values) for name, values in pieces.items()}
        arrays["tile_index"] = np.concatenate(tile_indices)

        x_axis = np.asarray(self.file["grid/x"][:])
        y_axis = np.asarray(self.file["grid/y"][:])
        row, column = np.divmod(arrays["cell_id"], len(x_axis))
        arrays["x"] = x_axis[column]
        arrays["y"] = y_axis[row]
        return pd.DataFrame(arrays)


def export_acquisition(
    cube_path: Path,
    acquisition: str,
    output: Path,
    *,
    fields: Sequence[str] = DEFAULT_FIELDS,
    quality: Sequence[int] | None = None,
    max_points: int | None = None,
) -> dict:
    """Export one overpass to a compact NPZ working file."""

    with RawCube(cube_path) as cube:
        table = cube.read_acquisition(
            acquisition, fields=fields, quality=quality, max_points=max_points
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: table[name].to_numpy() for name in table.columns}
    payload.update(
        acquisition=np.asarray(acquisition),
        source_cube=np.asarray(str(cube_path)),
        quality_selection=np.asarray(
            "all" if quality is None else ",".join(map(str, quality))
        ),
    )
    np.savez_compressed(output, **payload)
    return {
        "output": str(output),
        "acquisition": acquisition,
        "observations": int(len(table)),
        "fields": list(fields),
        "quality": None if quality is None else list(quality),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--acquisition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", action="append", dest="fields")
    parser.add_argument("--quality", nargs="*", type=int)
    parser.add_argument("--max-points", type=int)
    args = parser.parse_args(argv)
    report = export_acquisition(
        args.cube,
        args.acquisition,
        args.output,
        fields=tuple(args.fields or DEFAULT_FIELDS),
        quality=args.quality,
        max_points=args.max_points,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

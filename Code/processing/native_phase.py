"""Prepare native PIXC phase repairs and pack the two sparse correction layers.

setup() supplies the grids and input files. reference() builds a leave-one-
acquisition-out sampler for the residual search."""

from pathlib import Path
import argparse
import json
import sys
import h5py
import numpy as np
from scipy.ndimage import maximum_filter

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src"), str(HERE.parent)]
from run import resources
from prepare import axis
from aviso_dac import AVISODAC
from study_area import study_area_grid
from analysis import ds, dump
import region_phase
import residual_phase
from leaveout_reference import one as reference_band
from layout import band_indices


class Tide:
    def __init__(self, path, x, y):
        from cats import load_cats, prepare_cats_tide

        self.prepared = prepare_cats_tide(
            np.array([x.min(), x.max()]), np.array([y.min(), y.max()]), load_cats(path)
        )

    def tide(self, x, y, time):
        from cats import cats_tide_prepared

        if np.ndim(time) == 0:
            time = np.full(len(x), time, dtype="datetime64[ns]")
        out = np.full(len(x), np.nan)
        for start in range(0, len(x), 50000):
            out[start : start + 50000] = cats_tide_prepared(
                x[start : start + 50000],
                y[start : start + 50000],
                time[start : start + 50000],
                self.prepared,
            )
        return out


def queue(path):
    """Check that the native-file list matches the pinned acquisition inventory."""
    records = json.loads(path.read_text())
    expected = json.loads(
        (HERE.parent / "aux_files/native_inventory190.json").read_text()
    )
    assert len(records) == len(expected)
    for record, check in zip(records, expected):
        assert record["overpass"] == check["overpass"]
        assert [s["file"] for s in record["sources"]] == [
            s["file"] for s in check["sources"]
        ]
    return records


def setup(paths, output_dir):
    """Configure the original region and residual repair routines with explicit inputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    region_phase.R = output_dir
    region_phase.OUT = output_dir / "shelf_native_passes"
    region_phase.OUT.mkdir(exist_ok=True)
    region_phase.X, region_phase.Y = axis("x100"), axis("y100")
    region_phase.NX, region_phase.NY = len(region_phase.X), len(region_phase.Y)
    region_phase.N = region_phase.NX * region_phase.NY
    region_phase.GEO = study_area_grid(
        region_phase.X, region_phase.Y, paths["native_bedmachine_mask"]
    )
    region_phase.CONTEXT = maximum_filter(np.pad(region_phase.GEO, 30), size=61).ravel()
    with h5py.File(paths["native_rema"]) as handle:
        np.testing.assert_array_equal(handle["x"][:], region_phase.X)
        np.testing.assert_array_equal(handle["y"][:], region_phase.Y)
        reference_dtype = handle["ellipsoid"].dtype

    # Reference interpolation reads a bounded rectangle per native tile.
    def reference(x, y):
        from scipy.ndimage import map_coordinates

        result = np.full(np.asarray(x).shape, np.nan, dtype=reference_dtype)
        finite = np.isfinite(x + y)
        if not finite.any():
            return result
        column = (x[finite] - region_phase.X[0]) / 100
        row = (y[finite] - region_phase.Y[0]) / 100
        row_start = max(0, int(np.floor(row.min())) - 2)
        row_stop = min(region_phase.NY, int(np.ceil(row.max())) + 3)
        column_start = max(0, int(np.floor(column.min())) - 2)
        column_stop = min(region_phase.NX, int(np.ceil(column.max())) + 3)
        if row_start >= row_stop or column_start >= column_stop:
            return result
        with h5py.File(paths["native_rema"]) as handle:
            reference_values = handle["ellipsoid"][
                row_start:row_stop, column_start:column_stop
            ]
        result[finite] = map_coordinates(
            reference_values,
            [row - row_start, column - column_start],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        return result

    region_phase.reference = reference
    region_phase.PROVIDER = AVISODAC(paths["aviso_cache"], allow_download=False)
    region_phase.PROC = Tide(paths["cats_model"], region_phase.X, region_phase.Y)

    def download(item, folder):
        path = Path(paths["native_raw"]) / item["file"]
        if not path.is_file():
            raise FileNotFoundError(
                "Fetch the exact native_inventory190.json file first: " + str(path)
            )
        return path, False

    region_phase.download = download
    residual_phase.GRID = output_dir / "grid.h5"
    if not residual_phase.GRID.exists():
        with h5py.File(residual_phase.GRID, "w") as handle:
            ds(handle, "x100", region_phase.X)
            ds(handle, "y100", region_phase.Y)


def reference(index, paths, output_dir):
    """Make a leave-one-acquisition-out reference sampler and its bulk residual."""
    x, y = axis("x100"), axis("y100")
    columns = len(x)
    leaveout_height = np.full(columns * len(y), np.nan, "f4")
    current = leaveout_height.copy()
    for band in band_indices(paths["analysis_geometry"]):
        statistics_path = output_dir / "loo_reference" / f"{band:03d}.h5"
        statistics_path.parent.mkdir(exist_ok=True)
        source = Path(paths["phase2_reference_bands"]) / f"{band:03d}.h5"
        reference_band(source, statistics_path)
        with h5py.File(statistics_path) as statistics, h5py.File(source) as history:
            ids = statistics["fine_id"][:]
            inside = (ids >= 0) & (ids < len(leaveout_height))
            value = history["height"][index].ravel().astype("f4")
            value[~(history["weight"][index].ravel() > 0)] = np.nan
            leaveout_median = residual_phase.loo_median(
                statistics["count"][:],
                statistics["lo"][:],
                statistics["mid"][:],
                statistics["hi"][:],
                value,
            )
            leaveout_height[ids[inside]] = leaveout_median[inside]
            current[ids[inside]] = value[inside]
    paired = np.isfinite(leaveout_height + current)
    bulk = (
        float(np.median(current[paired] - leaveout_height[paired]))
        if paired.sum() >= 100
        else np.nan
    )

    def sample(xx, yy):
        column = np.rint((xx - x[0]) / 100).astype("i8")
        row = np.rint((yy - y[0]) / 100).astype("i8")
        inside = (
            np.isfinite(xx + yy)
            & (column >= 0)
            & (column < columns)
            & (row >= 0)
            & (row < len(y))
        )
        cell = np.where(inside, row * columns + column, 0)
        return np.where(inside, leaveout_height[cell], np.nan)

    return sample, bulk


def residual(index, r, output_dir, rows):
    """Find and regrid residual integer-phase repairs for one acquisition."""
    acquisition = rows[index]
    name = acquisition["overpass"]
    directory = output_dir / "residual" / name
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "report.json"
    if target.exists():
        assert json.loads(target.read_text())["complete"]
        return
    prior = json.loads(
        (output_dir / "shelf_native_passes" / f"{name}.json").read_text()
    )
    sources = {v["file"]: v for v in prior["sources"]}
    sample, bulk = reference(index, r, output_dir)
    paths = {
        v["file"]: Path(r["native_raw"]) / v["file"] for v in acquisition["sources"]
    }
    reports = []
    for item in acquisition["sources"]:
        ledger = directory / "corrections" / f"{item['file']}.npz"
        reports.append(
            residual_phase.assess_source(
                paths[item["file"]],
                item,
                sources[item["file"]],
                sample,
                bulk,
                region_phase.PROVIDER,
                ledger,
            )
        )
    grid = residual_phase.regrid_phase2(
        acquisition, sources, paths, region_phase.PROVIDER, directory
    )
    dump(
        target,
        dict(
            complete=True,
            acquisition=name,
            sources=reports,
            grid=grid,
            DAC_already_applied=True,
        ),
    )


def pack(native_dir, output_dir, rows):
    """Combine all completed region/residual archives into two sparse layers."""
    from pack_region_phase import build

    manifest = native_dir / "selection.json"
    dump(manifest, dict(groups={r["overpass"]: [] for r in rows}))
    build(manifest, [native_dir], output_dir / "phase_region190.h5")
    target = output_dir / "phase_residual190.h5"
    if target.exists():
        raise FileExistsError(target)
    with h5py.File(target.with_suffix(".tmp.h5"), "w") as output:
        starts = [0]
        ds(output, "acquisition_id", np.array([r["overpass"].encode() for r in rows]))
        for key, dtype in [
            ("cell_id", "i8"),
            ("before_wse", "f4"),
            ("after_wse", "f4"),
        ]:
            output.create_dataset(
                key,
                (0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(65536,),
                compression="gzip",
            )
        for acquisition in rows:
            folder = native_dir / "residual" / acquisition["overpass"]
            proof = json.loads((folder / "report.json").read_text())
            assert proof["complete"]
            path = folder / "grid_phase2.npz"
            count = 0
            if path.exists():
                with np.load(path) as repair:
                    count = len(repair["cell_id"])
                    begin = starts[-1]
                    end = begin + count
                    for key in ["cell_id", "before_wse", "after_wse"]:
                        output[key].resize((end,))
                        output[key][begin:end] = repair[key]
            starts.append(starts[-1] + count)
        ds(output, "observation_start", starts)
        output.attrs.update(complete=1, DAC_already_applied=1, acquisitions=len(rows))
    target.with_suffix(".tmp.h5").replace(target)


def main(argv=None):
    """Run the selected native preparation stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["region", "residual", "pack"])
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parts = parser.add_mutually_exclusive_group()
    parts.add_argument(
        "--index", type=int, help="One acquisition index from the native inventory"
    )
    parts.add_argument(
        "--all-parts", action="store_true", help="All native acquisitions, sequentially"
    )
    args = parser.parse_args(argv)
    input_paths = resources(args.resources)
    rows = queue(Path(input_paths["native_queue"]))
    out = args.work_dir.resolve()
    if args.stage == "pack":
        if args.index is not None or args.all_parts:
            parser.error("pack has no numbered parts")
        dest = args.output_dir or out
        dest.mkdir(parents=True, exist_ok=True)
        pack(out, dest, rows)
        return
    if not args.all_parts and (args.index is None or not 0 <= args.index < len(rows)):
        parser.error(f"Choose --index 0..{len(rows) - 1} or --all-parts")
    indices = range(len(rows)) if args.all_parts else [args.index]
    for index in indices:
        # Reset the same native inputs for each acquisition, as in separate jobs.
        setup(input_paths, out)
        if args.stage == "region":
            region_phase.process(rows[index])
        else:
            residual(index, input_paths, out, rows)


if __name__ == "__main__":
    main()

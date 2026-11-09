"""Small scientific controls, without downloads, real cubes or a shelf solve."""

from pathlib import Path
import sys
import json
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "processing"))
import huber
import gls
import prepare
from residual_phase import loo_median


def test_gls_operator_and_signal(tmp_path):
    gls.ROOT = tmp_path
    gls.main(["probe"])
    result = json.loads((tmp_path / "execution_probe.json").read_text())
    assert result["passed"]


def test_huber_full_unknown_translation():
    rng = np.random.default_rng(21)
    t = rng.normal(size=(2, 80, 9))
    mu = rng.normal(size=(2, 9))
    beta = rng.normal(size=(9, 2))
    h = (
        np.einsum("ntp,pn->tn", t, beta)[:, :, None]
        + mu[None]
        + rng.normal(scale=0.08, size=(80, 2, 9))
    )
    h[0] += 3
    base = huber.fit(h, t, np.ones_like(h))
    shift = rng.normal(size=(9, 2)) * 0.1
    static = rng.normal(size=(2, 9)) * 0.2
    other = huber.fit(
        h + np.einsum("ntp,pn->tn", t, shift)[:, :, None] + static[None],
        t,
        np.ones_like(h),
    )
    assert np.isfinite(base["beta"]).all()
    np.testing.assert_allclose(other["beta"] - base["beta"], shift, atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        other["terrain"] - base["terrain"], static, atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        other["normal_weight"], base["normal_weight"], atol=1e-8, rtol=0
    )


def test_leave_one_out_median_ties():
    v = np.array([1, 1, 2, 3, 5, 5, 7, 9, 11, 13], dtype="f4")
    s = np.sort(v)
    n = len(v)
    for value in np.unique(v):
        expected = np.median(np.delete(v, np.flatnonzero(v == value)[0]))
        got = loo_median(
            np.array([n]),
            s[n // 2 - 1 : n // 2],
            s[n // 2 : n // 2 + 1],
            s[n // 2 + 1 : n // 2 + 2],
            np.array([value]),
        )
        np.testing.assert_equal(got[0], expected)


def test_native_DAC_once_and_source_range(tmp_path, monkeypatch):
    out = tmp_path / "ind"
    out.mkdir()
    x = prepare.axis("x100")
    y = prepare.axis("y100")
    with h5py.File(out / "geometry.h5", "w") as f:
        f["x100"] = x
        f["y100"] = y
        f["cell_id"] = [0]
        f["fine_id"] = [[0, 1, 2, -1, -1, -1, -1, -1, -1]]
    raw = tmp_path / "raw.h5"
    name = prepare.SETTINGS["acquisitions"][0]["id"]
    with h5py.File(raw, "w") as f:
        f["acquisitions/id"] = np.array([name.encode()])
        f["tiles/acquisition"] = np.array([name.encode()])
        f["tiles/observation_start"] = [0, 3]
        for k, v in dict(
            cell_id=[0, 1, 2],
            wse=[20.0, 20.0, 20.0],
            cross_track=[11000.0, 12000.0, 9000.0],
            illumination_time=[100000.0, 100001.0, 100002.0],
        ).items():
            f["observations/" + k] = v
    layers = []
    for index, value in [(0, 22.0), (1, 23.5)]:
        path = tmp_path / f"layer{index}.h5"
        layers.append(str(path))
        with h5py.File(path, "w") as f:
            f["observation_start"] = np.r_[0, np.ones(190, dtype="i8")]
            f["cell_id"] = [index]
            f["after_wse"] = [value]
    monkeypatch.setattr(prepare, "AVISODAC", lambda *a, **kw: None)
    monkeypatch.setattr(
        prepare, "sample_dac", lambda p, t, lo, la: np.full(len(t), 0.5)
    )
    prepare.row(
        0,
        dict(
            raw_cube=str(raw),
            region_layer=layers[0],
            residual_layer=layers[1],
            aviso_cache="unused",
        ),
        out,
    )
    with h5py.File(out / "rows/000.h5") as f:
        np.testing.assert_equal(f["cell_id"][:], [0, 1])
        np.testing.assert_equal(f["height"][:], [21.5, 23.5])


def test_freeze_follows_virtual_height_dependencies(tmp_path):
    import analysis

    child = tmp_path / "child.h5"
    container = tmp_path / "input.h5"
    with h5py.File(child, "w") as f:
        f["height"] = [1.0, 2.0]
    layout = h5py.VirtualLayout((2,), dtype="f8")
    layout[:] = h5py.VirtualSource(str(child), "height", shape=(2,))
    with h5py.File(container, "w", libver="latest") as f:
        f.create_virtual_dataset("height", layout)
    manifest = tmp_path / "sources.json"
    analysis.freeze_inputs(manifest, [container])
    analysis.freeze_inputs(manifest, [container])
    assert {
        v["path"] for v in json.loads(manifest.read_text())["source_snapshots"]
    } == {str(container), str(child)}
    with h5py.File(child, "r+") as f:
        f["height"][0] = 3.0
    import pytest

    with pytest.raises(RuntimeError, match="Frozen source"):
        analysis.freeze_inputs(manifest, [container])


def test_native_reference_crop_preserves_full_grid_values_and_dtype(
    tmp_path, monkeypatch
):
    from scipy.ndimage import map_coordinates
    import native_phase

    axis = np.arange(4) * 100.0
    z = np.arange(16, dtype="f4").reshape(4, 4)
    path = tmp_path / "rema.nc"
    with h5py.File(path, "w") as f:
        f["x"] = axis
        f["y"] = axis
        f["ellipsoid"] = z
    monkeypatch.setattr(native_phase, "axis", lambda key: axis)
    monkeypatch.setattr(
        native_phase, "study_area_grid", lambda *a: np.ones((4, 4), bool)
    )
    monkeypatch.setattr(native_phase, "Tide", lambda *a: None)
    monkeypatch.setattr(native_phase, "AVISODAC", lambda *a, **kw: None)
    native_phase.setup(
        dict(
            native_bedmachine_mask="unused",
            native_rema=str(path),
            aviso_cache="unused",
            cats_model="unused",
        ),
        tmp_path / "native",
    )
    x = np.array([0, 50, 99.5, 300, np.nan, 450.0])
    y = np.array([0, 150, 299.5, 300, np.nan, 450.0])
    expected = map_coordinates(
        z, [y / 100, x / 100], order=1, mode="constant", cval=np.nan, prefilter=False
    )
    actual = native_phase.region_phase.reference(x, y)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def source_before_dimension_changes(filename, source, provenance):
    """Check reviewed dimension edits, then restore the prior code for its guards."""
    import ast
    import hashlib

    release = (
        provenance.get("python_release_refactor", {}).get("modules", {}).get(filename)
    )
    if release is not None:
        fingerprint = lambda text: hashlib.sha256(
            ast.dump(ast.parse(text), include_attributes=False).encode()
        ).hexdigest()
        assert fingerprint(source) == release["after_module_AST_sha256"], filename
        for edit in reversed(release["reverse_edits"]):
            assert source.count(edit["after"]) == 1, filename
            source = source.replace(edit["after"], edit["before"], 1)
        assert fingerprint(source) == release["before_module_AST_sha256"], filename

    record = provenance["dimension_refactor"]["modules"].get(filename)
    if record is None:
        return source

    def fingerprint(text):
        return hashlib.sha256(
            ast.dump(ast.parse(text), include_attributes=False).encode()
        ).hexdigest()

    assert fingerprint(source) == record["after_module_AST_sha256"], filename
    for edit in reversed(record["reverse_edits"]):
        assert source.count(edit["after"]) == 1, filename
        source = source.replace(edit["after"], edit["before"], 1)
    assert fingerprint(source) == record["before_module_AST_sha256"], filename
    return source


def test_dimension_helpers_and_partition_settings_are_recorded():
    import ast
    import hashlib

    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / "aux_files/source_provenance.json").read_text())
    record = provenance["dimension_refactor"]
    for filename, expected in record["new_helpers"].items():
        tree = ast.parse((root.parent / filename).read_text())
        assert (
            hashlib.sha256(
                ast.dump(tree, include_attributes=False).encode()
            ).hexdigest()
            == expected
        )
    assert (
        json.loads((root / "aux_files/processing_layout.json").read_text())
        == record["partition_settings"]
    )


def undo_cleanup(tree, record):
    """Restore names and removed aliases for comparison with the archived code."""
    import ast

    module_names = {new: old for old, new in record.get("module_names", {}).items()}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = module_names.get(node.id, node.id)

    def functions(body, prefix=""):
        for node in body:
            if isinstance(node, ast.ClassDef):
                functions(node.body, prefix + node.name + ".")
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = prefix + node.name
            inverse = {
                new: old
                for old, new in record.get("functions", {}).get(name, {}).items()
            }
            inverse.update(record.get("extra_inverse_aliases", {}).get(name, {}))
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    child.id = inverse.get(child.id, child.id)
                elif isinstance(child, ast.arg):
                    child.arg = inverse.get(child.arg, child.arg)
            calls = {
                new: old for old, new in record.get("calls", {}).get(name, {}).items()
            }

            class RestoreCalls(ast.NodeTransformer):
                def visit_Attribute(self, value):
                    expression = ast.unparse(value)
                    if expression in calls:
                        return ast.parse(calls[expression], mode="eval").body
                    return self.generic_visit(value)

                def visit_Name(self, value):
                    if value.id in calls:
                        return ast.parse(calls[value.id], mode="eval").body
                    return value

            RestoreCalls().visit(node)

    functions(tree.body)
    for index, node in enumerate(tree.body):
        original = record.get("import_replacements", {}).get(ast.unparse(node))
        if original:
            tree.body[index] = ast.parse(original).body[0]
    for removed in sorted(
        record.get("removed_statements", []), key=lambda item: item["index"]
    ):
        tree.body.insert(removed["index"], ast.parse(removed["source"]).body[0])
    return tree


def restore_local_names(function, record):
    """Undo readability-only names before checking the original frozen syntax."""
    import ast

    inverse = {
        new: old
        for old, new in record.get("local_renames", {}).get(function.name, {}).items()
    }
    inverse.update(record.get("extra_name_aliases", {}).get(function.name, {}))
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            node.id = inverse.get(node.id, node.id)
    return function


def test_frozen_scientific_function_bodies():
    """Check original functions after reversing reviewed naming and dimension edits."""
    import ast
    import hashlib

    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / "aux_files/source_provenance.json").read_text())
    modules = provenance["readability_refactor"]["modules"]
    for filename, symbols in provenance["ported_function_AST_sha256"].items():
        source = source_before_dimension_changes(
            "Code/processing/" + filename,
            (root / "processing" / filename).read_text(),
            provenance,
        )
        tree = ast.parse(source)
        tree = undo_cleanup(
            tree,
            provenance["cleanup_refactor"]["modules"].get(
                "Code/processing/" + filename, {}
            ),
        )
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        record = modules["Code/processing/" + filename]
        for name, expected in symbols.items():
            function = restore_local_names(functions[name], record)
            if name in record["added_docstrings"]:
                function.body.pop(0)
            actual = hashlib.sha256(
                ast.dump(function, include_attributes=False).encode()
            ).hexdigest()
            assert actual == expected, (filename, name)


def test_all_scientific_modules_preserve_operations():
    """Check every revised module and the original operations behind its count edits."""
    import ast
    import hashlib

    class Canonical(ast.NodeTransformer):
        def __init__(self, io_additions):
            self.io_additions = io_additions

        def generic_visit(self, node):
            fingerprint = hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()
            ).hexdigest()
            if fingerprint in self.io_additions:
                return None
            return self.normalize(node)

        def visit_Import(self, node):
            return [ast.Import(names=[alias]) for alias in node.names]

        def normalize(self, node):
            super().generic_visit(node)
            if isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    node.body.pop(0)
            if isinstance(node, ast.Module):
                node.body = [
                    item
                    for item in node.body
                    if not (
                        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name == "main"
                    )
                    and not (
                        isinstance(item, ast.If)
                        and "__name__" in ast.unparse(item.test)
                    )
                ]
            return node

    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / "aux_files/source_provenance.json").read_text())
    for filename, record in provenance["readability_refactor"]["modules"].items():
        if "scientific_module_ast_sha256" not in record:
            continue
        source = source_before_dimension_changes(
            filename, (root.parent / filename).read_text(), provenance
        )
        tree = ast.parse(source)
        tree = undo_cleanup(
            tree, provenance["cleanup_refactor"]["modules"].get(filename, {})
        )
        for function in tree.body:
            if isinstance(function, ast.FunctionDef):
                restore_local_names(function, record)
        additions = {
            **record.get("checked_mask_io_additions", {}),
            **record.get("callable_interface_additions", {}),
        }
        normalized = Canonical(additions).visit(tree)
        actual = hashlib.sha256(
            ast.dump(normalized, include_attributes=False).encode()
        ).hexdigest()
        assert actual == record["scientific_module_ast_sha256"], filename


def test_stage_part_counts_and_cli_selection(tmp_path):
    import pytest

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import run

    assert list(run.part_indices("row", 190, tmp_path)) == list(range(190))
    assert list(run.part_indices("row", 191, tmp_path)) == list(range(191))
    geometry = tmp_path / "geometry.h5"
    with h5py.File(geometry, "w") as handle:
        handle.attrs.update(band_ids=[2, 5, 9], fit_parts_per_band=2)
    assert list(run.part_indices("band", 191, tmp_path, geometry=geometry)) == [2, 5, 9]
    assert list(
        run.part_indices("independent", 191, tmp_path, geometry=geometry)
    ) == list(range(6))
    (tmp_path / "prepared.json").write_text(
        json.dumps({"node_parts": 7, "edge_parts": 543})
    )
    assert list(run.part_indices("gls-node", 191, tmp_path)) == list(range(7))
    assert run.part_indices("band", 191, tmp_path, 5, geometry) == [5]
    with pytest.raises(ValueError):
        run.part_indices("band", 191, tmp_path, 0, geometry)
    assert list(run.part_indices("gls-edge", 191, tmp_path)) == list(range(543))
    assert run.part_indices("row", 191, tmp_path, 190) == [190]
    with pytest.raises(ValueError):
        run.part_indices("row", 190, tmp_path, 190)
    assert run.parse_arguments(["row", "--all-parts"]).all_parts
    assert run.parse_arguments(["independent", "--index", "3"]).index == 3
    with pytest.raises(SystemExit):
        run.parse_arguments(["row"])
    with pytest.raises(SystemExit):
        run.parse_arguments(["row", "--index", "1", "--all-parts"])


def test_coarse_admission_precedes_fine_ownership():
    x = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    y = np.array([0.0, 100.0])
    cx = np.array([0.0, 250.0, 500.0])
    cy = np.array([0.0, 250.0])
    ids, fine = prepare.domain_mapping(x, y, cx, cy, lambda xx, yy: xx < 230)
    np.testing.assert_array_equal(ids, [0, 3])
    # Fine x=200 is inside, but its nearest coarse centre x=250 is outside.
    np.testing.assert_array_equal(fine[0, :4], [0, 1, 6, 7])
    assert (fine[0, 4:] == -1).all()
    assert (fine[1] == -1).all()


def test_REMA_bounded_warp_matches_original_full_tile(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import reproject
    import rema

    transform = from_origin(-500, 700, 32, 32)
    shape = (40, 50)
    n = np.ones(shape, "f4")
    n[10:15, 10:15] = 0
    raster = np.arange(np.prod(shape), dtype="f4").reshape(shape)
    raster[4, 5] = -9999
    originals = {}
    for key, suffix, method in rema.FIELDS:
        z = n if key == "count" else raster.copy()
        with rasterio.open(
            tmp_path / f"tile_{suffix}.tif",
            "w",
            driver="GTiff",
            height=shape[0],
            width=shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:3031",
            transform=transform,
            nodata=-9999,
        ) as f:
            f.write(z, 1)
        originals[key] = z.copy()
        originals[key][(z == -9999) | (n <= 0)] = np.nan
    x = np.arange(0, 301, 100.0)
    y = np.arange(0, 301, 100.0)
    actual, _ = rema.resample(tmp_path, x, y)
    for key, _, method in rema.FIELDS:
        expected = np.full((len(y), len(x)), np.nan, "f4")
        reproject(
            originals[key],
            expected,
            src_transform=transform,
            src_crs="EPSG:3031",
            src_nodata=np.nan,
            dst_transform=from_origin(x[0] - 50, y[-1] + 50, 100, 100),
            dst_crs="EPSG:3031",
            dst_nodata=np.nan,
            resampling=method,
            init_dest_nodata=False,
            num_threads=1,
        )
        np.testing.assert_array_equal(actual[key], expected[::-1])


def test_all_part_dispatch_keeps_the_original_indices(tmp_path, monkeypatch):
    """New convenience loops call the same scientific entry points in order."""
    import run
    import native_phase
    import reference_replay
    import selection_reference
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(prepare, "row", lambda index, paths, out: calls.append(index))
    args = SimpleNamespace(stage="row", acquisitions=190, index=None)
    run.run_stage(args, {}, tmp_path, tmp_path / "independent190")
    assert calls == list(range(190))

    calls.clear()
    records = [{"overpass": str(index)} for index in range(190)]
    monkeypatch.setattr(
        native_phase, "resources", lambda path: {"native_queue": "unused"}
    )
    monkeypatch.setattr(native_phase, "queue", lambda path: records)
    monkeypatch.setattr(native_phase, "setup", lambda paths, out: None)
    monkeypatch.setattr(
        native_phase.region_phase,
        "process",
        lambda record: calls.append(record["overpass"]),
    )
    native_phase.main(
        ["region", "--resources", "unused", "--work-dir", str(tmp_path), "--all-parts"]
    )
    assert calls == [str(index) for index in range(190)]

    calls.clear()
    geometry = tmp_path / "geometry.h5"
    with h5py.File(geometry, "w") as handle:
        handle.attrs["band_ids"] = [2, 5, 9]
    monkeypatch.setattr(run, "resources", lambda path: {"analysis_geometry": geometry})
    monkeypatch.setattr(
        reference_replay,
        "pack_band",
        lambda paths, out, index, settings: calls.append(index),
    )
    reference_replay.main(
        ["band", "--resources", "unused", "--work-dir", str(tmp_path), "--all-parts"]
    )
    assert calls == [2, 5, 9]

    calls.clear()
    monkeypatch.setattr(
        selection_reference,
        "build",
        lambda raw, geometry, rema, out, index: calls.append(index),
    )
    selection_reference.main(
        [
            "--raw-cube",
            "unused",
            "--geometry",
            str(geometry),
            "--rema",
            "unused",
            "--output",
            str(tmp_path),
            "--all-parts",
        ]
    )
    assert calls == [2, 5, 9]


def test_downloader_retains_the_entire_pinned_raster_inventory(tmp_path):
    """Resolve every pinned filename in catalogue order."""
    from swot_input import resolve_raster_files

    settings = prepare.SETTINGS
    acquisitions = settings["acquisitions"][:190] + [
        dict(
            id=settings["added_acquisition"]["acquisition_id"],
            tiles=settings["added_acquisition"]["tiles"],
        )
    ]
    raw = tmp_path / "Raster"
    raw.mkdir()
    expected = [title for acquisition in acquisitions for title in acquisition["tiles"]]
    for title in expected:
        (raw / title).touch()
    granules = resolve_raster_files(acquisitions, raw, tmp_path / "cache")
    assert len(granules) == 633
    assert [granule.title for granule in granules] == expected
    assert len({granule.acquisition for granule in granules}) == 191


def test_saved_filter_mask_matches_original_and_rejects_edits(tmp_path):
    import filtering
    import pytest

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    height = np.array([[[1.0, 10.0, np.nan]], [[2.0, -30.0, 3.0]]], dtype="f4")
    height = np.pad(height, ((0, 0), (0, 0), (0, 6)), constant_values=np.nan)
    reference = np.array(
        [[2.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
    )
    source = inputs / "000.h5"
    with h5py.File(source, "w") as handle:
        handle["height"] = height
        handle["reference_geoid"] = reference
        handle["cell_id"] = [5]
    geometry = dict(
        source_band=np.array([0]), source_local=np.array([0]), cell_id=np.array([5])
    )
    rule = filtering.SCREENING
    expected = np.isfinite(height) & (
        ~np.isfinite(reference)[None]
        | (
            abs(height - reference[None] - rule["centre_m"])
            <= rule["multiplier"] * rule["NMAD_m"]
        )
    )
    filtering.build_band(tmp_path, 0, 2)
    actual = filtering.checked_mask(tmp_path, np.array([0]), geometry, expected)
    np.testing.assert_array_equal(actual, expected)
    with h5py.File(source) as handle:
        np.testing.assert_array_equal(handle["height"][:], height)
    with h5py.File(tmp_path / "admission/000.h5", "r+") as handle:
        handle["admitted"][0, 0, 0] = not expected[0, 0, 0]
    with pytest.raises(AssertionError, match="Saved mask differs"):
        filtering.checked_mask(tmp_path, np.array([0]), geometry, expected)


def test_mask_files_are_frozen_with_fit_inputs(tmp_path):
    import run
    from types import SimpleNamespace

    folder = tmp_path / "independent191"
    (folder / "admission").mkdir(parents=True)
    with h5py.File(folder / "geometry.h5", "w") as handle:
        handle.attrs["band_ids"] = [2, 5, 9]
    captured = []
    analysis_stub = SimpleNamespace(
        freeze_inputs=lambda path, files: captured.extend(files)
    )
    run.freeze_analysis_inputs(analysis_stub, folder)
    assert [path.name for path in captured if path.parent.name == "admission"] == [
        f"{index:03d}.h5" for index in (2, 5, 9)
    ]


def test_plain_map_exports_preserve_existing_values(tmp_path):
    from export_maps import copy_product

    source = tmp_path / "source.h5"
    values = np.array([[1.0, np.nan], [3.0, 4.0]], dtype="f4")
    with h5py.File(source, "w") as handle:
        handle["x"] = [0.0, 100.0]
        handle["y"] = [0.0, 100.0]
        handle["mean"] = values
        handle.attrs.update(complete=1, units="m")
    target = tmp_path / "dem_100m.h5"
    copy_product(source, target, {"x": "x", "y": "y", "mean": "elevation"})
    with h5py.File(target) as handle:
        np.testing.assert_array_equal(handle["elevation"][:], values)
        assert handle["elevation"].dtype == values.dtype
        assert handle.attrs["units"] == "m"

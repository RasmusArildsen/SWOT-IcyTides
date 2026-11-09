"""Check the numbered scripts without downloads or a shelf-wide solve."""

from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import step_setup

SCRIPTS = [
    "01_import_data.py",
    "02_phase_corrections.py",
    "03_geophysical_and_dac.py",
    "04_filter_observations.py",
    "05_dem_and_tides.py",
    "06_make_figures.py",
]


def load_step(filename):
    spec = importlib.util.spec_from_file_location(
        "step_" + filename[:2], CODE / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", SCRIPTS)
def test_numbered_script_dry_run_does_not_create_outputs(tmp_path, script):
    output = tmp_path / "new_run"
    result = subprocess.run(
        [
            sys.executable,
            str(CODE / script),
            "--inputs",
            str(CODE / "input_paths.example.json"),
            "--work-dir",
            str(output),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert str(output) in result.stdout
    assert not output.exists()


def test_one_input_file_defines_all_generated_paths_and_rejects_changes(tmp_path):
    inputs = tmp_path / "inputs.json"
    inputs.write_text((CODE / "input_paths.example.json").read_text())
    args = step_setup.argument_parser("test").parse_args(
        [
            "--inputs",
            str(inputs),
            "--work-dir",
            str(tmp_path / "run"),
            "--acquisitions",
            "190",
        ]
    )
    args = step_setup.setup(args)
    values = json.loads(args.resources.read_text())
    assert values["analysis_geometry"] == str(
        tmp_path / "run/independent190/geometry.h5"
    )
    assert values["phase2_reference_bands"] == str(
        tmp_path / "run/phase_reference/bands"
    )
    changed = json.loads(inputs.read_text())
    changed["cats_model"] = "another_model.nc"
    inputs.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Input paths changed"):
        step_setup.setup(args)


def test_import_resolves_the_pinned_local_files_and_keeps_manifest_on_resume(
    tmp_path, monkeypatch
):
    import prepare
    from s4is import raw_cube

    module = load_step(SCRIPTS[0])
    raw = tmp_path / "Raster"
    raw.mkdir()
    settings = prepare.SETTINGS
    expected = [title for date in settings["acquisitions"] for title in date["tiles"]]
    expected += settings["added_acquisition"]["tiles"]
    for title in expected:
        (raw / title).touch()
    paths = {
        name: tmp_path / name
        for name in (
            "raw_cube",
            "bedmachine_source",
            "native_bedmachine_mask",
            "analysis_bedmachine_mask",
            "rema_source",
            "native_rema",
            "cats_model",
        )
    }
    paths["raster_raw"] = raw
    for name in ("native_bedmachine_mask", "analysis_bedmachine_mask", "native_rema"):
        paths[name].touch()
    calls = []
    monkeypatch.setattr(prepare, "axis", lambda name: np.arange(3) * 100)
    monkeypatch.setattr(
        raw_cube, "build_raw_cube", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(prepare, "geometry", lambda *args, **kwargs: None)
    independent = tmp_path / "independent191"
    module.import_data(paths, tmp_path, independent, 191)
    manifest_path = tmp_path / "raster_manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    assert manifest["granule_count"] == 633
    assert manifest["acquisition_count"] == 191
    assert [row["title"] for row in manifest["granules"]] == expected
    assert calls[0][1] == {"keep_raw_tiles": True}
    module.import_data(paths, tmp_path, independent, 191)
    assert manifest_path.read_bytes() == original
    assert calls[0] == calls[1]

    import swot_input

    downloads = []
    paths["native_raw"] = tmp_path / "PIXC"
    monkeypatch.setattr(
        swot_input,
        "cache_native_files",
        lambda folder, count: downloads.append((folder, count)),
    )
    module.import_data(paths, tmp_path, independent, 191, download_swot=True)
    assert downloads == [(paths["native_raw"], 191)]
    assert manifest_path.read_bytes() == original


def test_phase_correction_preserves_reference_order_and_stops_on_failure(
    tmp_path, monkeypatch
):
    import native_phase
    import region_phase
    import reference_replay

    module = load_step(SCRIPTS[1])
    calls = []
    dates = [{"overpass": str(i)} for i in range(3)]
    aux = tmp_path / "code/aux_files"
    aux.mkdir(parents=True)
    (aux / "analysis.json").write_text(
        json.dumps({"acquisitions": [{"id": date["overpass"]} for date in dates]})
    )
    (aux / "reference_settings.json").write_bytes(
        (CODE / "aux_files/reference_settings.json").read_bytes()
    )
    monkeypatch.setattr(module, "CODE", aux.parent)
    paths = {
        name: tmp_path / name
        for name in (
            "native_queue",
            "analysis_geometry",
            "legacy_reference_surface",
            "cats_model",
            "legacy_reference_geometry",
            "first_native_archives",
            "legacy_native_archives",
        )
    }
    with h5py.File(paths["analysis_geometry"], "w") as geometry_file:
        geometry_file.attrs["band_ids"] = [2, 5, 9]
    monkeypatch.setattr(native_phase, "queue", lambda path: dates)
    monkeypatch.setattr(native_phase, "setup", lambda paths, out: calls.append("setup"))
    monkeypatch.setattr(
        region_phase, "process", lambda date: calls.append(("region", date["overpass"]))
    )
    monkeypatch.setattr(reference_replay, "ARGS", None)

    def geometry(*args):
        assert reference_replay.ARGS is not None
        assert args == (
            paths["analysis_geometry"],
            paths["legacy_reference_surface"],
            paths["cats_model"],
            paths["legacy_reference_geometry"],
        )
        calls.append("geometry")

    monkeypatch.setattr(reference_replay, "prepare_geometry", geometry)
    monkeypatch.setattr(
        reference_replay, "archive_native", lambda *args: calls.append("archives")
    )
    monkeypatch.setattr(
        reference_replay, "build_rows", lambda p, out, i, s: calls.append(("row", i))
    )
    monkeypatch.setattr(
        reference_replay, "pack_band", lambda p, out, i, s: calls.append(("band", i))
    )
    monkeypatch.setattr(
        native_phase, "residual", lambda i, p, out, rows: calls.append(("residual", i))
    )
    monkeypatch.setattr(native_phase, "pack", lambda *args: calls.append("pack"))

    def no_process(*args, **kwargs):
        pytest.fail("The numbered script started another Python process")

    monkeypatch.setattr(subprocess, "run", no_process)
    monkeypatch.setattr(subprocess, "Popen", no_process)
    module.correct_phases(paths, tmp_path / "native", tmp_path / "reference", tmp_path)
    expected = []
    for i in range(len(dates)):
        expected.extend(["setup", ("region", str(i))])
    expected += ["geometry", "archives"]
    expected += [("row", i) for i in range(len(dates))]
    expected += [("band", i) for i in (2, 5, 9)]
    for i in range(len(dates)):
        expected.extend(["setup", ("residual", i)])
    assert calls == expected + ["pack"]

    calls.clear()

    def failed_residual(*args):
        raise RuntimeError("Residual assessment failed")

    monkeypatch.setattr(native_phase, "residual", failed_residual)
    with pytest.raises(RuntimeError, match="Residual assessment failed"):
        module.correct_phases(
            paths, tmp_path / "native", tmp_path / "reference", tmp_path
        )
    assert "pack" not in calls

    calls.clear()
    dates.reverse()
    with pytest.raises(ValueError, match="same order"):
        module.correct_phases(
            paths, tmp_path / "native", tmp_path / "reference", tmp_path
        )
    assert calls == []


@pytest.mark.parametrize("acquisitions", [190, 191])
def test_corrected_observation_preparation_keeps_every_date(
    tmp_path, monkeypatch, acquisitions
):
    import prepare
    import selection_reference

    module = load_step(SCRIPTS[2])
    paths = {
        name: tmp_path / name
        for name in (
            "raw_cube",
            "analysis_geometry",
            "native_rema",
            "reference_bands",
        )
    }
    calls = []
    monkeypatch.setattr(
        selection_reference,
        "build",
        lambda *args: calls.append(("reference", args[-1])),
    )
    monkeypatch.setattr(prepare, "row", lambda i, p, out: calls.append(("row", i)))
    monkeypatch.setattr(
        prepare, "band", lambda i, p, out, nt: calls.append(("band", i, nt))
    )
    with h5py.File(paths["analysis_geometry"], "w") as geometry_file:
        geometry_file.attrs["band_ids"] = [2, 5, 9]
    module.prepare_observations(paths, tmp_path, acquisitions)
    assert calls == (
        [("reference", i) for i in (2, 5, 9)]
        + [("row", i) for i in range(acquisitions)]
        + [("band", i, acquisitions) for i in (2, 5, 9)]
    )


def test_numbered_filter_writes_real_masks_without_changing_heights(tmp_path):
    module = load_step(SCRIPTS[3])
    folder = tmp_path / "inputs"
    folder.mkdir()
    with h5py.File(tmp_path / "geometry.h5", "w") as geometry_file:
        geometry_file.attrs["band_ids"] = [2, 5, 9]
    heights = np.full((11, 1, 9), 1.0, dtype="f4")
    heights[0, 0, :3] = [np.nan, 100.0, -2.0]
    for band in (2, 5, 9):
        with h5py.File(folder / f"{band:03d}.h5", "w") as output:
            output["cell_id"] = [band]
            output["height"] = heights
            output["reference_geoid"] = np.zeros((1, 9))
    module.filter_observations(tmp_path, len(heights))
    for band in (2, 5, 9):
        with h5py.File(tmp_path / "admission" / f"{band:03d}.h5") as mask:
            assert mask.attrs["complete"] == 1
            assert not mask["admitted"][0, 0, 0]
            assert not mask["admitted"][0, 0, 1]
            assert mask["admitted"][0, 0, 2]
        with h5py.File(folder / f"{band:03d}.h5") as inputs:
            np.testing.assert_array_equal(inputs["height"][:], heights)

    import filtering

    (tmp_path / "admission/005.h5").unlink()
    with pytest.raises(FileNotFoundError):
        filtering.check_complete(tmp_path, len(heights))


@pytest.mark.parametrize(
    "acquisitions,fail_solve", [(190, False), (191, False), (191, True)]
)
def test_in_process_fit_uses_current_paths_and_exports_only_after_success(
    tmp_path,
    monkeypatch,
    acquisitions,
    fail_solve,
):
    import analysis
    import filtering
    import gls
    import figure_products
    import export_maps

    module = load_step(SCRIPTS[4])
    independent = tmp_path / f"independent{acquisitions}"
    gls_dir = tmp_path / "gls"
    independent.mkdir()
    with h5py.File(independent / "geometry.h5", "w") as geometry_file:
        geometry_file.attrs.update(band_ids=[2, 5], fit_parts_per_band=3)
    calls = []
    # Simulate modules already imported for a different run/date count.
    monkeypatch.setattr(analysis, "ROOT", tmp_path / "stale")
    monkeypatch.setattr(analysis, "OLD", tmp_path / "stale")
    monkeypatch.setattr(analysis, "NT", 7)
    monkeypatch.setattr(gls, "ROOT", tmp_path / "stale_gls")
    monkeypatch.setattr(gls, "NT", 7)

    def current_paths():
        assert analysis.ROOT == analysis.OLD == independent
        assert analysis.NT == gls.NT == acquisitions
        assert gls.ROOT == gls_dir

    monkeypatch.setattr(
        filtering, "check_complete", lambda *args: calls.append("masks")
    )

    def freeze(path, files):
        current_paths()
        calls.append("freeze")

    monkeypatch.setattr(analysis, "freeze_inputs", freeze)
    monkeypatch.setattr(analysis, "fit_part", lambda i: calls.append(("fit", i)))
    monkeypatch.setattr(analysis, "collect", lambda: calls.append("collect"))
    monkeypatch.setattr(gls, "probe", lambda: calls.append("probe"))

    def prepare():
        gls_dir.mkdir()
        (gls_dir / "prepared.json").write_text(
            json.dumps({"node_parts": 5, "edge_parts": 3})
        )
        calls.append("prepare")

    monkeypatch.setattr(gls, "prepare", prepare)
    monkeypatch.setattr(gls, "node_part", lambda i: calls.append(("node", i)))
    monkeypatch.setattr(gls, "edge_part", lambda i: calls.append(("edge", i)))
    monkeypatch.setattr(gls, "assemble", lambda: calls.append("assemble"))

    def solve():
        calls.append("solve")
        if fail_solve:
            raise RuntimeError("Signal control did not pass")

    monkeypatch.setattr(gls, "solve", solve)
    monkeypatch.setattr(
        figure_products, "build", lambda *args: calls.append("products")
    )
    monkeypatch.setattr(
        export_maps, "copy_product", lambda *args: calls.append("export")
    )
    if fail_solve:
        with pytest.raises(RuntimeError, match="Signal control did not pass"):
            module.fit_dem_and_tides({}, tmp_path, acquisitions)
        assert calls[-1] == "solve"
        assert "products" not in calls and "export" not in calls
    else:
        module.fit_dem_and_tides({}, tmp_path, acquisitions)
        current_paths()
        assert calls[0] == "masks"
        assert [call for call in calls if isinstance(call, tuple)] == (
            [("fit", i) for i in range(6)]
            + [("node", i) for i in range(5)]
            + [("edge", i) for i in range(3)]
        )
        assert [call for call in calls if isinstance(call, str)] == [
            "masks",
            "freeze",
            "freeze",
            "collect",
            "probe",
            "freeze",
            "prepare",
            "freeze",
            "freeze",
            "freeze",
            "assemble",
            "freeze",
            "solve",
            "products",
            "export",
            "export",
        ]


def test_callable_atl06_and_download_routines_keep_original_operations():
    provenance = json.loads((CODE / "aux_files/source_provenance.json").read_text())
    direct = provenance["direct_python_steps"]
    tree = ast.parse((CODE / "processing/atl06.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_reference"
    )
    work_index = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "WORK" for t in node.targets)
    )
    body = ast.Module(body=function.body[work_index + 1 :], type_ignores=[])
    fingerprint = hashlib.sha256(
        ast.dump(body, include_attributes=False).encode()
    ).hexdigest()
    assert fingerprint == direct["atl06_original_processing_body_AST_sha256"]
    tree = ast.parse((CODE / "processing/swot_input.py").read_text())
    for function in tree.body:
        if isinstance(function, ast.FunctionDef):
            fingerprint = hashlib.sha256(
                ast.dump(function, include_attributes=False).encode()
            ).hexdigest()
            assert (
                fingerprint
                == direct["moved_download_function_AST_sha256"][function.name]
            )

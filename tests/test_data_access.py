"""Data selection, raw-cube precision and restart checks on small fixtures."""

import json
from pathlib import Path
import numpy as np
import pytest
import h5py
from s4is.earthdata import (
    Granule,
    group_acquisitions,
    parse_granule_title,
    read_manifest,
    select_inventory,
    select_newest_versions,
    write_manifest,
)
from s4is.raw_cube import reproject_raw_tile
from s4is.raw_access import RawCube, export_acquisition

TILES = [
    "SWOT_L2_HR_Raster_100m_UTM19D_N_x_x_x_027_449_017F_20250129T171930_20250129T171951_PGD0_01.nc",
    "SWOT_L2_HR_Raster_100m_UTM20D_N_x_x_x_027_449_018F_20250129T171950_20250129T172011_PGD0_01.nc",
    "SWOT_L2_HR_Raster_100m_UTM20D_N_x_x_x_028_337_014F_20250215T140139_20250215T140150_PGD0_01.nc",
]


def test_tiles_group_into_dated_cycle_pass_acquisitions():
    rows = [parse_granule_title(name, f"G{i}") for i, name in enumerate(TILES)]
    groups = group_acquisitions(rows)
    assert [group["id"] for group in groups] == [
        "2025-01-29_027_449",
        "2025-02-15_028_337",
    ]
    assert groups[0]["tile_count"] == 2
    assert rows[0].scene == "017F"


def test_manifest_is_reviewable_and_validated(tmp_path):
    rows = [parse_granule_title(name, f"G{i}") for i, name in enumerate(TILES)]
    path = tmp_path / "manifest.json"
    payload = write_manifest(
        path,
        rows,
        bbox=(-66, -70.2, -60.2, -65.8),
        temporal=("2024-11-23", "2026-08-20"),
    )
    assert payload["granule_count"] == 3
    assert payload["acquisition_count"] == 2
    assert read_manifest(path)["acquisitions"][0]["tile_count"] == 2
    broken = json.loads(path.read_text())
    broken["granule_count"] = 100
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="Granule count"):
        read_manifest(path)


def test_frozen_inventory_selects_exact_filenames(tmp_path):
    rows = [parse_granule_title(name, f"G{i}") for i, name in enumerate(TILES)]
    inventory = tmp_path / "tiles.txt"
    inventory.write_text(TILES[1] + "\n" + TILES[0] + "\n")
    assert [row.title for row in select_inventory(rows, inventory)] == [
        TILES[1],
        TILES[0],
    ]
    inventory.write_text("missing.nc\n")
    with pytest.raises(RuntimeError, match="not returned by CMR"):
        select_inventory(rows, inventory)


def test_newest_selection_prefers_current_product_then_cmr_revision():
    old = parse_granule_title(TILES[0], "old", "SWOT_L2_HR_Raster_100m_2.0")
    current = Granule(
        **{
            **old.__dict__,
            "concept_id": "current",
            "product": "SWOT_L2_HR_Raster_100m_D",
            "revision_id": 2,
            "revision_date": "2026-01-01T00:00:00Z",
        }
    )
    selected, report = select_newest_versions(
        [old, current],
        preferred_products=["SWOT_L2_HR_Raster_100m_2.0", "SWOT_L2_HR_Raster_100m_D"],
    )
    assert selected == [current]
    assert report["superseded_granules"] == 1


def test_raw_cube_reprojection_retains_bad_flags_and_source_precision(tmp_path):
    import xarray as xr
    from pyproj import CRS

    shape = (2, 2)
    source = xr.Dataset(
        data_vars={
            "wse": (
                ("y", "x"),
                np.array([[1, 2], [3, 4]], dtype="f4"),
                {"grid_mapping": "crs", "units": "m"},
            ),
            "wse_qual": (("y", "x"), np.array([[0, 1], [2, 3]], dtype="u1")),
            "longitude": (
                ("y", "x"),
                np.array(
                    [
                        [-63.123456789, -63.123456781],
                        [-63.123456772, -63.123456763],
                    ],
                    dtype="f8",
                ),
                {"units": "degrees_east"},
            ),
            "illumination_time": (
                ("y", "x"),
                np.full(shape, 800000000.123456, dtype="f8"),
            ),
            "crs": ((), 0, {"spatial_ref": CRS.from_epsg(3031).to_wkt()}),
        },
        coords={"x": np.array([0.0, 100.0]), "y": np.array([0.0, 100.0])},
        attrs={"product_version": "test"},
    )
    path = tmp_path / "raw.nc"
    source.to_netcdf(path, engine="netcdf4")

    result = reproject_raw_tile(path, np.array([0.0, 100.0]), np.array([0.0, 100.0]))

    np.testing.assert_array_equal(np.sort(result["wse_qual"]), [0, 1, 2, 3])
    assert result["longitude"].dtype == np.dtype("f8")
    assert result["illumination_time"].dtype == np.dtype("f8")
    assert '"units":"degrees_east"' in result["source_attributes_json"]


def test_raw_cube_reader_selects_acquisition_quality_and_grid_coordinates(tmp_path):
    path = tmp_path / "cube.h5"
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as target:
        target.attrs.update(
            schema="s4is-raw-swot-sparse-cube-v1",
            resolution_m=100,
            corrections_applied="none",
            quality_filter_applied="none",
        )
        grid = target.create_group("grid")
        grid.create_dataset("x", data=[1000.0, 1100.0])
        grid.create_dataset("y", data=[2000.0, 2100.0])
        acquisitions = target.create_group("acquisitions")
        acquisitions.create_dataset("id", data=np.asarray(["a", "b"], dtype=string))
        acquisitions.create_dataset("cycle", data=[1, 2])
        acquisitions.create_dataset("pass", data=[10, 20])
        acquisitions.create_dataset(
            "start_utc",
            data=np.asarray(
                ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], dtype=string
            ),
        )
        acquisitions.create_dataset(
            "stop_utc",
            data=np.asarray(
                ["2025-01-01T00:01:00Z", "2025-01-02T00:01:00Z"], dtype=string
            ),
        )
        tiles = target.create_group("tiles")
        tiles.create_dataset(
            "title", data=np.asarray(["tile-a", "tile-b"], dtype=string)
        )
        tiles.create_dataset("acquisition", data=np.asarray(["a", "b"], dtype=string))
        tiles.create_dataset("product", data=np.asarray(["D", "D"], dtype=string))
        tiles.create_dataset("revision_id", data=[1, 1])
        tiles.create_dataset(
            "source_crs", data=np.asarray(["EPSG:3031"] * 2, dtype=string)
        )
        tiles.create_dataset(
            "source_attributes_json", data=np.asarray(["{}"] * 2, dtype=string)
        )
        tiles.create_dataset("observation_start", data=[0, 3, 5])
        observations = target.create_group("observations")
        observations.create_dataset("cell_id", data=[0, 1, 3, 0, 2])
        observations.create_dataset("wse", data=[10, 11, 12, 20, 21])
        observations.create_dataset(
            "wse_qual", data=np.asarray([0, 1, 3, 2, 0], dtype="u1")
        )

    with RawCube(path) as cube:
        assert cube.summary()["tiles_loaded"] == 2
        table = cube.read_acquisition("a", fields=("wse",), quality=(0, 1))
        np.testing.assert_allclose(table["x"], [1000, 1100])
        np.testing.assert_allclose(table["y"], [2000, 2000])
        np.testing.assert_allclose(table["wse"], [10, 11])
        sampled = cube.read_acquisition(
            "a", fields=("wse",), quality=(0, 1), max_points=1, seed=4
        )
        assert len(sampled) == 1
        assert sampled["wse_qual"].iloc[0] in (0, 1)

    output = tmp_path / "a.npz"
    report = export_acquisition(path, "a", output, fields=("wse",), quality=(0, 1))
    assert report["observations"] == 2
    with np.load(output) as result:
        np.testing.assert_allclose(result["wse"], [10, 11])


def test_download_prefers_https_without_a_compute_platform(tmp_path, monkeypatch):
    """NASA access keeps using local files and the existing HTTPS URL order."""
    import sys
    from types import SimpleNamespace
    from s4is.earthdata import cache_acquisition

    calls = []
    title = TILES[0]

    def download(links, directory, threads):
        calls.extend(links)
        (Path(directory) / title).write_text("download fixture")

    monkeypatch.setitem(
        sys.modules,
        "earthaccess",
        SimpleNamespace(login=lambda **kwargs: None, download=download),
    )
    manifest = {
        "granules": [
            {
                "acquisition": "example",
                "title": title,
                "external_url": "https://example.test/" + title,
                "direct_url": "s3://example/" + title,
            }
        ]
    }
    paths = cache_acquisition(manifest, "example", tmp_path)
    assert calls == ["https://example.test/" + title]
    assert paths == [tmp_path / "example" / title]


def test_raw_cube_build_and_resume_from_a_local_acquisition(tmp_path, monkeypatch):
    """Exercise the actual cache/context-manager path, append and restart logic."""
    from s4is import raw_cube

    granule = parse_granule_title(TILES[0])
    manifest = tmp_path / "manifest.json"
    write_manifest(
        manifest,
        [granule],
        bbox=(-66, -70.2, -60.2, -65.8),
        temporal=("2024-11-23", "2026-10-01"),
    )
    grid = tmp_path / "grid.npz"
    np.savez(grid, x=[0.0, 100.0], y=[0.0, 100.0])
    cache = tmp_path / "cache"
    tile = cache / granule.acquisition / granule.title
    tile.parent.mkdir(parents=True)
    tile.write_text("cached local file fixture")
    values = dict(
        cell_id=np.array([0, 3]),
        wse=np.array([1.0, 4.0], dtype="f4"),
        wse_qual=np.array([0, 3], dtype="u1"),
        illumination_time=np.array([800000000.123456, 800000000.234567], dtype="f8"),
    )
    monkeypatch.setattr(raw_cube, "reproject_raw_tile", lambda path, x, y: values)
    output = tmp_path / "raw.h5"
    report = raw_cube.build_raw_cube(manifest, grid, output, cache, keep_raw_tiles=True)
    assert report["complete"] and report["raw_observations"] == 2
    assert tile.is_file()
    with h5py.File(output) as handle:
        np.testing.assert_array_equal(handle["observations/wse"][:], values["wse"])
        np.testing.assert_array_equal(
            handle["observations/wse_qual"][:], values["wse_qual"]
        )
        np.testing.assert_array_equal(
            handle["observations/illumination_time"][:], values["illumination_time"]
        )
    # A second call must keep the same observations, without reopening a tile.
    monkeypatch.setattr(
        raw_cube,
        "reproject_raw_tile",
        lambda *args: pytest.fail("Completed tile reprocessed"),
    )
    again = raw_cube.build_raw_cube(manifest, grid, output, cache, keep_raw_tiles=True)
    assert again["raw_observations"] == 2

"""Verify date/pass pairing and the acquisition count used by the downloader."""

from pathlib import Path
from types import SimpleNamespace
import json
import sys
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE), str(CODE / "processing")]
import source_inventory
from source_inventory import matched_native_inventory


@pytest.mark.parametrize("count,files", [(190, 2080), (191, 2096)])
def test_native_sources_match_every_selected_raster_acquisition(count, files):
    records = matched_native_inventory(count)
    assert len(records) == count
    assert sum(len(record["sources"]) for record in records) == files
    if count == 191:
        original = matched_native_inventory(190)
        assert records[:-1] == original
        assert records[-1]["overpass"] == "2026-10-01_056_578"
        assert len(records[-1]["sources"]) == 16


def test_missing_or_mispaired_native_acquisition_is_rejected(tmp_path, monkeypatch):
    records = matched_native_inventory(191)
    path = tmp_path / "native_inventory191.json"
    monkeypatch.setattr(source_inventory, "AUX", tmp_path)
    path.write_text(json.dumps(records[:-1]))
    with pytest.raises(ValueError, match="catalogues do not match"):
        matched_native_inventory(191)
    records[-1]["sources"][0]["file"] = records[0]["sources"][0]["file"]
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="does not match"):
        matched_native_inventory(191)


@pytest.mark.parametrize("count", [190, 191])
def test_native_downloader_receives_the_selected_acquisition_count(
    tmp_path, monkeypatch, count
):
    import swot_input

    calls = []
    name = "SWOT_L2_HR_PIXC_056_578_272L_20261001T102318_20261001T102327_PID0_01.nc"

    def inventory(selected):
        calls.append(selected)
        return [{"overpass": "example", "sources": [{"file": name}]}]

    def download(results, directory):
        (Path(directory) / name).touch()

    monkeypatch.setattr(swot_input, "matched_native_inventory", inventory)
    monkeypatch.setitem(
        sys.modules,
        "earthaccess",
        SimpleNamespace(
            search_data=lambda **kwargs: [{"umm": {"GranuleUR": name}}],
            login=lambda **kwargs: None,
            download=download,
        ),
    )
    swot_input.cache_native_files(tmp_path, count)
    assert calls == [count]
    assert (tmp_path / name).is_file()

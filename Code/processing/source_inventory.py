"""Match the native PIXC catalogue to the selected Raster acquisitions."""

from pathlib import Path
import json
import re
from layout import DEFAULT_ACQUISITIONS, selected_acquisitions

AUX = Path(__file__).resolve().parents[1] / "aux_files"
PIXC_NAME = re.compile(r"SWOT_L2_HR_PIXC_(\d{3})_(\d{3})_\d{3}[LR]_(\d{8})T")


def matched_native_inventory(acquisitions=DEFAULT_ACQUISITIONS):
    """Require the same date, cycle and pass for both input catalogues.

    Raster scenes and native PIXC tiles have different file subdivisions;
    compare acquisition identities, rather than requiring equal file counts.
    """
    expected = [row["id"] for row in selected_acquisitions(acquisitions)]
    path = AUX / f"native_inventory{acquisitions}.json"
    records = json.loads(path.read_text())
    if [row["overpass"] for row in records] != expected:
        raise ValueError("PIXC and Raster acquisition catalogues do not match")
    seen = set()
    for record in records:
        if not record["sources"]:
            raise ValueError("No PIXC sources for " + record["overpass"])
        for source in record["sources"]:
            name = source["file"]
            match = PIXC_NAME.match(name)
            if match is None:
                raise ValueError("Invalid PIXC filename: " + name)
            cycle, number, date = match.groups()
            identity = f"{date[:4]}-{date[4:6]}-{date[6:]}_{cycle}_{number}"
            if identity != record["overpass"] or name in seen:
                raise ValueError("PIXC source does not match its acquisition: " + name)
            seen.add(name)
    return records

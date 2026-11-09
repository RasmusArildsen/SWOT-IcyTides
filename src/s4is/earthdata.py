"""Find NASA Earthdata files and download one acquisition at a time.

This module deliberately stops at data access.  Scientific processing lives in
the following pipeline stages, so searching NASA Earthdata cannot silently
change a surface or tide solution.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Iterator, Sequence
from urllib.parse import unquote, urlparse

SWOT_PRODUCT = "SWOT_L2_HR_Raster_100m_D"
LCIS_BBOX = (-66.0, -70.2, -60.2, -65.8)
GRANULE_PATTERN = re.compile(
    r"_(?P<cycle>\d{3})_(?P<pass_number>\d{3})_(?P<scene>[^_]+)_"
    r"(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_"
)


@dataclass(frozen=True)
class Granule:
    """The small amount of metadata needed to reproduce an input selection."""

    title: str
    concept_id: str
    cycle: int
    pass_number: int
    scene: str
    start_utc: str
    stop_utc: str
    acquisition: str
    product: str
    revision_id: int = 0
    revision_date: str = ""
    direct_url: str | None = None
    external_url: str | None = None


def parse_granule_title(
    title: str, concept_id: str = "", product: str = SWOT_PRODUCT
) -> Granule:
    """Parse a Version-D 100 m Raster filename into acquisition metadata."""

    match = GRANULE_PATTERN.search(Path(title).name)
    if match is None:
        raise ValueError(f"Unrecognised SWOT Raster granule name: {title}")
    fields = match.groupdict()
    start = datetime.strptime(fields["start"], "%Y%m%dT%H%M%S")
    stop = datetime.strptime(fields["stop"], "%Y%m%dT%H%M%S")
    acquisition = f"{start:%Y-%m-%d}_{fields['cycle']}_{fields['pass_number']}"
    return Granule(
        title=Path(title).name,
        concept_id=concept_id,
        cycle=int(fields["cycle"]),
        pass_number=int(fields["pass_number"]),
        scene=fields["scene"],
        start_utc=start.isoformat() + "Z",
        stop_utc=stop.isoformat() + "Z",
        acquisition=acquisition,
        product=product,
    )


def group_acquisitions(granules: Sequence[Granule]) -> list[dict]:
    """Group adjacent geographic tiles from the same dated cycle/pass."""

    grouped: dict[str, list[Granule]] = {}
    for granule in sorted(granules, key=lambda item: (item.start_utc, item.title)):
        grouped.setdefault(granule.acquisition, []).append(granule)
    return [
        {
            "id": acquisition,
            "cycle": rows[0].cycle,
            "pass": rows[0].pass_number,
            "start_utc": min(row.start_utc for row in rows),
            "stop_utc": max(row.stop_utc for row in rows),
            "tile_count": len(rows),
            "tiles": [row.title for row in rows],
        }
        for acquisition, rows in grouped.items()
    ]


def _first_url(result, access: str) -> str | None:
    try:
        links = result.data_links(access=access)
    except (AttributeError, TypeError, ValueError):
        return None
    suffixes = (".nc", ".h5", ".hdf5")
    return next(
        (
            str(link)
            for link in links
            if str(link).lower().split("?", 1)[0].endswith(suffixes)
        ),
        None,
    )


def _from_result(result, product: str) -> Granule:
    umm = result.get("umm", {})
    cmr_title = str(umm.get("GranuleUR") or umm.get("ProducerGranuleId") or "")
    meta = result.get("meta", {})
    concept_id = str(meta.get("concept-id", ""))
    direct_url = _first_url(result, "direct")
    external_url = _first_url(result, "external")
    data_url = direct_url or external_url
    title = (
        Path(unquote(urlparse(data_url).path)).name
        if data_url
        else cmr_title.removesuffix("_swot")
    )
    parsed = parse_granule_title(title, concept_id, product)
    return Granule(
        **{
            **asdict(parsed),
            "revision_id": int(meta.get("revision-id", 0)),
            "revision_date": str(meta.get("revision-date", "")),
            "direct_url": direct_url,
            "external_url": external_url,
        }
    )


def search_swot(
    *,
    bbox: Sequence[float] = LCIS_BBOX,
    temporal: Sequence[str],
    product: str = SWOT_PRODUCT,
    login: bool = True,
) -> list[Granule]:
    """Search NASA CMR through earthaccess without opening file contents."""

    try:
        import earthaccess
    except ImportError as exc:
        raise RuntimeError("Install data support with: pip install -e .") from exc
    if login:
        earthaccess.login(strategy="all", persist=False)
    results = earthaccess.search_data(
        short_name=product,
        cloud_hosted=True,
        bounding_box=tuple(map(float, bbox)),
        temporal=tuple(temporal),
        count=-1,
    )
    granules = [_from_result(result, product) for result in results]
    # CMR may return the same concept more than once across pagination retries.
    unique = {(row.concept_id or row.title): row for row in granules}
    return sorted(unique.values(), key=lambda row: (row.start_utc, row.title))


def search_products(
    *,
    bbox: Sequence[float],
    temporal: Sequence[str],
    products: Sequence[str],
    login: bool = True,
) -> list[Granule]:
    """Search one or more product versions and combine unique filenames."""

    rows = []
    for index, product in enumerate(products):
        rows.extend(
            search_swot(
                bbox=bbox,
                temporal=temporal,
                product=product,
                login=login and index == 0,
            )
        )
    unique = {(row.product, row.concept_id or row.title): row for row in rows}
    return sorted(unique.values(), key=lambda row: (row.start_utc, row.title))


def logical_tile_id(granule: Granule) -> str:
    """Identity of an observation tile, independent of processing release."""

    match = GRANULE_PATTERN.search(granule.title)
    if match is None:
        raise ValueError(f"Unrecognised SWOT Raster granule name: {granule.title}")
    # Everything through the measurement stop time identifies the same scene.
    return granule.title[: match.end("stop")]


def select_newest_versions(
    granules: Sequence[Granule],
    *,
    preferred_products: Sequence[str],
) -> tuple[list[Granule], dict]:
    """Keep one current granule per measured tile with an auditable priority."""

    priority = {product: index for index, product in enumerate(preferred_products)}
    grouped: dict[str, list[Granule]] = {}
    for row in granules:
        grouped.setdefault(logical_tile_id(row), []).append(row)

    selected = []
    superseded = []
    for identity, rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda row: (
                priority.get(row.product, -1),
                row.revision_id,
                row.revision_date,
                row.title,
            ),
            reverse=True,
        )
        selected.append(ordered[0])
        superseded.extend(
            {
                "logical_tile_id": identity,
                "selected": ordered[0].title,
                "superseded": row.title,
                "superseded_product": row.product,
            }
            for row in ordered[1:]
        )
    selected.sort(key=lambda row: (row.start_utc, row.title))
    report = {
        "policy": "one granule per zone/cycle/pass/scene/start/stop; prefer later product priority, then CMR revision",
        "preferred_products_low_to_high": list(preferred_products),
        "candidate_granules": len(granules),
        "selected_granules": len(selected),
        "superseded_granules": len(superseded),
        "superseded": superseded,
    }
    return selected, report


def select_inventory(granules: Sequence[Granule], inventory: Path) -> list[Granule]:
    """Select exact filenames from a frozen inventory and reject omissions."""

    wanted = [
        Path(line.strip()).name
        for line in inventory.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(wanted) != len(set(wanted)):
        raise ValueError(f"Duplicate filenames in {inventory}")
    available = {row.title: row for row in granules}
    missing = [title for title in wanted if title not in available]
    if missing:
        raise RuntimeError(
            f"{len(missing)} inventory tiles were not returned by CMR; first: {missing[:3]}"
        )
    return [available[title] for title in wanted]


def write_manifest(
    path: Path,
    granules: Sequence[Granule],
    *,
    bbox: Sequence[float],
    temporal: Sequence[str],
    products: Sequence[str] | None = None,
    selection: dict | None = None,
) -> dict:
    """Write a deterministic, reviewable acquisition inventory."""

    payload = {
        "schema": "s4is-swot-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "products": list(products or sorted({row.product for row in granules})),
        "bbox": list(map(float, bbox)),
        "temporal": list(temporal),
        "granule_count": len(granules),
        "acquisition_count": len({row.acquisition for row in granules}),
        "granules": [asdict(row) for row in granules],
        "acquisitions": group_acquisitions(granules),
    }
    if selection is not None:
        payload["selection"] = selection
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def read_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") not in {
        "s4is-swot-manifest-v1",
        "s4is-swot-cloud-manifest-v1",
    }:
        raise ValueError(f"Unsupported manifest schema in {path}")
    if payload.get("granule_count") != len(payload.get("granules", [])):
        raise ValueError(f"Granule count does not match manifest contents in {path}")
    return payload


def acquisition_ids(manifest: dict) -> list[str]:
    return [row["id"] for row in manifest["acquisitions"]]


def cache_acquisition(
    manifest: dict,
    acquisition: str,
    cache_root: Path,
    *,
    threads: int = 4,
) -> list[Path]:
    """Download one acquisition and verify every expected tile is present."""

    selected = [
        row for row in manifest["granules"] if row["acquisition"] == acquisition
    ]
    if not selected:
        raise KeyError(f"Acquisition {acquisition!r} is not in the manifest")
    folder = cache_root / acquisition
    folder.mkdir(parents=True, exist_ok=True)
    expected = [folder / row["title"] for row in selected]
    missing = [row for row, path in zip(selected, expected) if not path.exists()]
    if missing:
        try:
            import earthaccess
        except ImportError as exc:
            raise RuntimeError("Install data support with: pip install -e .") from exc
        earthaccess.login(strategy="all", persist=False)
        links = [row.get("external_url") or row.get("direct_url") for row in missing]
        if any(link is None for link in links):
            raise RuntimeError("Manifest lacks a usable URL for one or more tiles")
        earthaccess.download(links, str(folder), threads=threads)
    absent = [path for path in expected if not path.exists()]
    if absent:
        raise RuntimeError(f"Acquisition cache is incomplete: {absent[:3]}")
    return expected


@contextmanager
def temporary_acquisition(
    manifest: dict,
    acquisition: str,
    cache_root: Path,
    *,
    keep: bool = False,
    threads: int = 4,
) -> Iterator[list[Path]]:
    """Yield local tile paths and optionally remove them after processing."""

    paths = cache_acquisition(
        manifest,
        acquisition,
        cache_root,
        threads=threads,
    )
    try:
        yield paths
    finally:
        if not keep:
            shutil.rmtree(cache_root / acquisition, ignore_errors=True)


def _bbox(text: str) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in text.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser(
        "search", help="write an Earthdata file/acquisition manifest"
    )
    search.add_argument("--bbox", type=_bbox, default=LCIS_BBOX)
    search.add_argument("--temporal", nargs=2, required=True, metavar=("START", "STOP"))
    search.add_argument(
        "--product",
        action="append",
        dest="products",
        help="repeat to combine collection versions; default is Version D",
    )
    search.add_argument(
        "--inventory", type=Path, help="optional text file of exact granule filenames"
    )
    search.add_argument(
        "--zone",
        action="append",
        dest="zones",
        help="repeat to retain filename UTM zones, e.g. UTM19D",
    )
    search.add_argument(
        "--keep-all-versions",
        action="store_true",
        help="disable the default newest-version selection",
    )
    search.add_argument("--manifest", type=Path, required=True)
    cache = commands.add_parser("cache", help="cache one acquisition from a manifest")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--acquisition", required=True)
    cache.add_argument("--cache-dir", type=Path, required=True)
    cache.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    if args.command == "search":
        products = args.products or [SWOT_PRODUCT]
        rows = search_products(
            bbox=args.bbox,
            temporal=args.temporal,
            products=products,
        )
        if args.zones:
            rows = [
                row
                for row in rows
                if any(f"_{zone}_" in row.title for zone in args.zones)
            ]
        selection = None
        if not args.keep_all_versions:
            rows, selection = select_newest_versions(rows, preferred_products=products)
        if args.inventory:
            rows = select_inventory(rows, args.inventory)
        payload = write_manifest(
            args.manifest,
            rows,
            bbox=args.bbox,
            temporal=args.temporal,
            products=products,
            selection=selection,
        )
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "tiles": payload["granule_count"],
                    "acquisitions": payload["acquisition_count"],
                },
                indent=2,
            )
        )
    else:
        paths = cache_acquisition(
            read_manifest(args.manifest),
            args.acquisition,
            args.cache_dir,
            threads=args.threads,
        )
        print("\n".join(map(str, paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Find the pinned SWOT files on disk or fetch them through Earthdata."""

from s4is import earthdata
from layout import DEFAULT_ACQUISITIONS
from source_inventory import matched_native_inventory


def cache_native_files(cache_dir, acquisitions=DEFAULT_ACQUISITIONS):
    """Fetch exact PIXC filenames; reject missing or ambiguous search results."""
    import earthaccess

    inventory = matched_native_inventory(acquisitions)
    titles = sorted(
        {source["file"] for record in inventory for source in record["sources"]}
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    for title in titles:
        if (cache_dir / title).exists():
            continue
        results = earthaccess.search_data(granule_name=title, count=10)
        exact = [
            result
            for result in results
            if result["umm"]["GranuleUR"] == title
            or result["umm"]["GranuleUR"] == title.removesuffix(".nc")
        ]
        if len(exact) != 1:
            raise RuntimeError("No unique exact native file: " + title)
        earthaccess.login(strategy="all", persist=False)
        earthaccess.download(exact, str(cache_dir))
        if not (cache_dir / title).is_file():
            raise FileNotFoundError(title)


def resolve_raster_files(acquisitions, local_dir, cache_dir):
    """Resolve the exact Raster names in the same acquisition/tile order."""
    granules = []
    for acquisition in acquisitions:
        for title in acquisition["tiles"]:
            if local_dir:
                original = (local_dir / title).resolve()
                if not original.is_file():
                    original = (local_dir / acquisition["id"] / title).resolve()
                if not original.is_file():
                    raise FileNotFoundError(f"Pinned Raster file missing: {title}")
                granule = earthdata.parse_granule_title(title)
                folder = cache_dir / acquisition["id"]
                folder.mkdir(parents=True, exist_ok=True)
                link = folder / title
                if not link.exists():
                    link.symlink_to(original)
            else:
                import earthaccess

                results = earthaccess.search_data(
                    short_name=earthdata.SWOT_PRODUCT,
                    granule_name=title,
                    count=10,
                )
                found = [
                    earthdata._from_result(result, earthdata.SWOT_PRODUCT)
                    for result in results
                ]
                found = [granule for granule in found if granule.title == title]
                if len(found) != 1:
                    raise RuntimeError(
                        f"Expected one exact granule: {title}; got {len(found)}"
                    )
                granule = found[0]
            if granule.acquisition != acquisition["id"]:
                raise ValueError("Catalogue/acquisition identity mismatch: " + title)
            granules.append(granule)
    return granules

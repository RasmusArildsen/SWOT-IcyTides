"""Download the listed SWOT and ICESat-2 source files."""

import argparse
import re
from pathlib import Path

from step_setup import CODE, read_input_paths
from layout import (
    DEFAULT_ACQUISITIONS,
    FILE_LISTS,
    grouped_files,
    raster_path,
    selected_acquisitions,
)


def download_file(title, folder, **search_options):
    """Fetch one exact filename from Earthdata."""
    import earthaccess

    results = earthaccess.search_data(granule_name=title, count=10, **search_options)
    matches = [
        result
        for result in results
        if result["umm"]["GranuleUR"] in (title, Path(title).stem)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"No unique Earthdata result for {title}")
    folder.mkdir(parents=True, exist_ok=True)
    earthaccess.login(strategy="all", persist=False)
    earthaccess.download(matches, str(folder))
    path = folder / title
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def download_if_missing(title, folder, download, **search_options):
    """Use the local source file or fetch it from Earthdata."""
    path = folder / title
    if path.is_file():
        return path
    if not download:
        raise FileNotFoundError(path)
    return download_file(title, folder, **search_options)


def resolve_raster_files(acquisitions, folder, download=False):
    """Return (acquisition ID, local path) for each tile in catalogue order."""
    files = []
    for date in acquisitions:
        date_code, cycle, pass_number = date["id"].split("_")
        for title in date["tiles"]:
            match = re.search(r"_(\d{3})_(\d{3})_[^_]+_(\d{8})T\d{6}_", title)
            if match is None or match.groups() != (
                cycle,
                pass_number,
                date_code.replace("-", ""),
            ):
                raise ValueError(f"Raster tile does not match {date['id']}: {title}")
            try:
                path = raster_path(folder, title, date["id"])
            except FileNotFoundError:
                if not download:
                    raise
                path = download_file(
                    title, folder / date["id"], short_name="SWOT_L2_HR_Raster_100m_D"
                )
            files.append((date["id"], path))
    return files


def download_satellite_files(paths, acquisitions, download=True):
    """Use the listed filenames for every selected acquisition."""
    dates = selected_acquisitions(acquisitions)
    resolve_raster_files(dates, paths["raster_raw"], download=download)
    for files in grouped_files(FILE_LISTS / "icesat2_atl06.tsv").values():
        for title in files:
            download_if_missing(
                title, paths["atl06_raw"], download, short_name="ATL06", version="007"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=CODE / "input_paths.json")
    parser.add_argument("--acquisitions", type=int, default=DEFAULT_ACQUISITIONS)
    parser.add_argument(
        "--local-only", action="store_true", help="Check local files without downloading"
    )
    args = parser.parse_args(argv)
    paths = read_input_paths(args.inputs, {"raster_raw", "atl06_raw"})
    download_satellite_files(paths, args.acquisitions, download=not args.local_only)


if __name__ == "__main__":
    main()

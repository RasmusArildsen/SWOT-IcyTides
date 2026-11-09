"""Import SWOT observations and prepare the grids used by the later scripts.

Read the pinned Raster files, write an unfiltered cube, then prepare
BedMachine masks, the REMA reference and the shared analysis geometry.
"""

from step_setup import argument_parser, setup, existing_file


def import_data(paths, work_dir, independent_dir, acquisitions, download_swot=False):
    import numpy as np
    import prepare
    import ancillary
    import rema
    from layout import selected_acquisitions
    from swot_input import resolve_raster_files, cache_native_files
    from source_inventory import matched_native_inventory
    from s4is import earthdata
    from s4is.raw_cube import build_raw_cube

    # 1. Match the source dates/passes, then find the listed Raster files.
    matched_native_inventory(acquisitions)
    manifest_path = work_dir / "raster_manifest.json"
    if not existing_file(manifest_path):
        dates = selected_acquisitions(acquisitions, prepare.SETTINGS)
        local_folder = None if download_swot else paths["raster_raw"]
        files = resolve_raster_files(dates, local_folder, paths["raster_raw"])
        manifest = earthdata.write_manifest(
            manifest_path,
            files,
            bbox=earthdata.LCIS_BBOX,
            temporal=[
                min(file.start_utc for file in files),
                max(file.stop_utc for file in files),
            ],
            selection=dict(
                policy="Exact pinned filenames from Code/aux_files/analysis.json; no newdate/coverage/version substitution",
                analysis_acquisitions=acquisitions,
            ),
        )
        for date in manifest["acquisitions"]:
            earthdata.cache_acquisition(manifest, date["id"], paths["raster_raw"])
    if download_swot:
        cache_native_files(paths["native_raw"], acquisitions)

    # 2. Write the 100 m grid and keep each source tile distinct in the cube.
    grid_path = work_dir / "grid.npz"
    np.savez_compressed(grid_path, x=prepare.axis("x100"), y=prepare.axis("y100"))
    build_raw_cube(
        manifest_path,
        grid_path,
        paths["raw_cube"],
        paths["raster_raw"],
        keep_raw_tiles=True,
    )

    # 3. Crop BedMachine and independently resample REMA onto the 100 m grid.
    if not existing_file(paths["native_bedmachine_mask"]):
        ancillary.mask(paths["bedmachine_source"], paths["native_bedmachine_mask"])
    if not existing_file(paths["analysis_bedmachine_mask"]):
        ancillary.mask(
            paths["bedmachine_source"], paths["analysis_bedmachine_mask"], expanded=True
        )
    if not existing_file(paths["native_rema"]):
        rema.build(paths["rema_source"], paths["native_rema"])

    # 4. Assign fine cells to coarse cells and prepare clocks and CATS comparisons.
    independent_dir.mkdir(parents=True, exist_ok=True)
    prepare.geometry(
        independent_dir,
        paths["analysis_bedmachine_mask"],
        paths["cats_model"],
        paths["native_bedmachine_mask"],
        nt=acquisitions,
    )


def main(argv=None):
    parser = argument_parser(__doc__)
    parser.add_argument(
        "--download-swot",
        action="store_true",
        help="Fetch the pinned Raster and PIXC files from Earthdata",
    )
    args = parser.parse_args(argv)
    setup(args)
    if args.dry_run:
        print(f"Work folder: {args.work_dir}")
        print(
            "Find pinned SWOT files; build the raw cube; prepare BedMachine, REMA and geometry."
        )
        return
    import_data(
        args.paths,
        args.work_dir,
        args.independent,
        args.acquisitions,
        download_swot=args.download_swot,
    )


if __name__ == "__main__":
    main()

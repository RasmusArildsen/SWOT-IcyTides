"""Assemble corrected acquisition rows and time-by-cell observation cubes.

Raster WSE keeps its supplied geophysical convention. Native PIXC heights
were converted during phase correction. Remove AVISO DAC once from Raster
and region-repair values; insert residual repairs with their DAC already
removed. Keep ocean tide for the fit.
"""

from step_setup import argument_parser, setup


def prepare_observations(paths, independent_dir, acquisitions):
    import selection_reference
    import prepare
    from layout import band_indices

    # 1. Prepare the REMA/geoid reference used by the filter in script 04.
    # This is a screening reference, not a height offset.
    for band in band_indices(paths["analysis_geometry"]):
        selection_reference.build(
            paths["raw_cube"],
            paths["analysis_geometry"],
            paths["native_rema"],
            paths["reference_bands"],
            band,
        )

    # 2. Combine Raster and native repairs, enforce the range guard and remove DAC.
    for date_index in range(acquisitions):
        prepare.row(date_index, paths, independent_dir)

    # 3. Arrange the corrected rows into the spatial bands defined by the geometry.
    for band in band_indices(paths["analysis_geometry"]):
        prepare.band(band, paths, independent_dir, acquisitions)


def main(argv=None):
    args = argument_parser(__doc__).parse_args(argv)
    setup(args)
    if args.dry_run:
        print(f"Work folder: {args.work_dir}")
        print(
            "Prepare screening references; combine repairs and DAC; pack observation bands."
        )
        return
    prepare_observations(args.paths, args.independent, args.acquisitions)


if __name__ == "__main__":
    main()

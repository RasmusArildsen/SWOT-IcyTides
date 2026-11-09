"""Save the observation masks used by the DEM and tidal fit.

Keep finite heights with abs(height - REMA - centre) <= 3.5 * NMAD.
A missing REMA value retains an otherwise finite height. This does not
change heights or clocks. Huber weights and fit-support checks belong to 05.
"""

from step_setup import argument_parser, setup


def filter_observations(independent_dir, acquisitions):
    import filtering
    from layout import band_indices

    for band in band_indices(independent_dir / "geometry.h5"):
        filtering.build_band(independent_dir, band, acquisitions)
    filtering.check_complete(independent_dir, acquisitions)


def main(argv=None):
    args = argument_parser(__doc__).parse_args(argv)
    setup(args)
    if args.dry_run:
        print(f"Work folder: {args.work_dir}")
        print(
            "Apply the REMA residual gate and save admission masks for the bands defined by the geometry."
        )
        return
    filter_observations(args.independent, args.acquisitions)


if __name__ == "__main__":
    main()

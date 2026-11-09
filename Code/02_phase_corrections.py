"""Correct integer phase ambiguities in the native PIXC observations.

Move both height and horizontal position before regridding. The second pass
uses a reference with the current acquisition left out. Geophysical and DAC
corrections needed during phase selection are not applied a second time in 03.
"""

import json
from step_setup import CODE, argument_parser, setup
from layout import band_indices


def correct_phases(paths, native_dir, reference_dir, output_dir):
    import native_phase
    import region_phase
    import reference_replay

    dates = native_phase.queue(paths["native_queue"])
    settings = json.loads((CODE / "aux_files/analysis.json").read_text())

    if [date["overpass"] for date in dates] != [
        date["id"] for date in settings["acquisitions"]
    ]:
        raise ValueError(
            "Native inventory and reference catalogue must have the same order"
        )

    # 1. Correct the broad region phase errors, one acquisition at a time.
    for date in dates:
        native_phase.setup(paths, native_dir)
        region_phase.process(date)

    # 2. Rebuild the earlier reference used to choose the remaining integers.
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_replay.configure(
        json.loads((CODE / "aux_files/reference_settings.json").read_text())
    )
    reference_replay.prepare_geometry(
        paths["analysis_geometry"],
        paths["legacy_reference_surface"],
        paths["cats_model"],
        paths["legacy_reference_geometry"],
    )
    reference_replay.archive_native(
        paths["first_native_archives"], paths["legacy_native_archives"], settings
    )
    for date_index in range(len(dates)):
        reference_replay.build_rows(paths, reference_dir, date_index, settings)
    for band in band_indices(paths["analysis_geometry"]):
        reference_replay.pack_band(paths, reference_dir, band, settings)

    # 3. Use leave-one-acquisition-out residuals to assess the remaining errors.
    for date_index in range(len(dates)):
        native_phase.setup(paths, native_dir)
        native_phase.residual(date_index, paths, native_dir, dates)

    # 4. Write the two sparse correction layers for script 03.
    native_phase.pack(native_dir, output_dir, dates)


def main(argv=None):
    args = argument_parser(__doc__).parse_args(argv)
    setup(args)
    if args.dry_run:
        print(f"Work folder: {args.work_dir}")
        print(
            "Region phase correction; reference histories; residual correction; pack layers."
        )
        return
    correct_phases(args.paths, args.native, args.ancestor, args.work_dir)


if __name__ == "__main__":
    main()

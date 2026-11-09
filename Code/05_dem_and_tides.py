"""Fit mean elevation and M2, S2, K1 and O1 from the corrected observations.

Fit each coarse cell with its own fine-cell levels, then combine absolute
observations and neighbour differences in the connected GLS model. Export
after the numerical residual and full-signal checks pass.
"""

import json
from step_setup import argument_parser, setup


def fit_dem_and_tides(paths, work_dir, acquisitions):
    import analysis
    import filtering
    import gls
    import figure_products
    from export_maps import copy_product
    from run import freeze_analysis_inputs
    from layout import fit_part_indices

    independent_dir = work_dir / f"independent{acquisitions}"
    gls_dir = work_dir / "gls"
    products_dir = work_dir / "figure_products"

    # Set the folders and date count used by the original fitting functions.
    analysis.ROOT = independent_dir
    analysis.OLD = independent_dir
    analysis.NT = acquisitions
    gls.ROOT = gls_dir
    gls.NT = acquisitions

    # 1. Check the saved masks and fit each cell group defined by the geometry.
    filtering.check_complete(independent_dir, acquisitions)
    freeze_analysis_inputs(analysis, independent_dir)
    for part in fit_part_indices(independent_dir / "geometry.h5"):
        analysis.fit_part(part)
    freeze_analysis_inputs(analysis, independent_dir)
    analysis.collect()

    # 2. Check the solver and prepare the connected system.
    gls.probe()
    freeze_analysis_inputs(analysis, independent_dir, gls_dir)
    gls.prepare()

    # 3. Build absolute-observation blocks and neighbour-difference blocks.
    freeze_analysis_inputs(analysis, independent_dir, gls_dir)
    prepared = json.loads((gls_dir / "prepared.json").read_text())
    for part in range(prepared["node_parts"]):
        gls.node_part(part)
    edge_parts = prepared["edge_parts"]
    freeze_analysis_inputs(analysis, independent_dir, gls_dir)
    for part in range(edge_parts):
        gls.edge_part(part)

    # 4. Assemble and solve. The solve checks its residual and injected signal.
    freeze_analysis_inputs(analysis, independent_dir, gls_dir)
    gls.assemble()
    freeze_analysis_inputs(analysis, independent_dir, gls_dir)
    gls.solve()

    # 5. Extract the fitted DEM, tidal maps and transect for plotting.
    figure_products.build(
        gls_dir / "native_tides.h5", independent_dir, paths, products_dir
    )
    copy_product(
        products_dir / "figure1.h5",
        work_dir / "maps/dem_100m.h5",
        {"x": "x", "y": "y", "mean": "elevation"},
    )
    amplitude_fields = {"x": "x", "y": "y"}
    for constituent in ("m2", "s2", "k1", "o1"):
        for suffix in ("amp", "cats_amp"):
            name = constituent + "_" + suffix
            amplitude_fields[name] = name
    copy_product(
        products_dir / "maps.h5",
        work_dir / "maps/tidal_amplitudes_250m.h5",
        amplitude_fields,
    )


def main(argv=None):
    args = argument_parser(__doc__).parse_args(argv)
    setup(args)
    if args.dry_run:
        print(f"Work folder: {args.work_dir}")
        print(
            "Check masks; fit independent cells; build and solve GLS; export DEM and tides."
        )
        return
    fit_dem_and_tides(args.paths, args.work_dir, args.acquisitions)


if __name__ == "__main__":
    main()

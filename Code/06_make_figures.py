"""Prepare the ATL06 comparison and draw Figures 1–3 from the completed fit.

The laser reference keeps its pinned source window and native ATL06 DAC.
Plotting does not refit the observations or fill unsupported cells.
"""

from step_setup import argument_parser, setup, existing_file


def make_figures(paths, work_dir):
    import h5py
    import atl06
    from Figure1.plot_figure1 import figure1
    from Figure2.plot_figure2 import figure2
    from Figure3.plot_figure3 import figure3
    from paper_statistics import constituent_table

    # 1. Build the comparison surface from the pinned laser segments, if needed.
    if not existing_file(paths["atl06_reference"]):
        atl06.build_reference(paths, paths["atl06_reference"].parent)

    # 2. Check the completed map and transect products.
    products_dir = work_dir / "figure_products"
    output_dir = work_dir / "figures"
    for filename in ("maps.h5", "figure1.h5", "figure3.h5"):
        with h5py.File(products_dir / filename) as product:
            assert product.attrs.get("complete") == 1
    output_dir.mkdir(parents=True, exist_ok=True)
    # 3. Draw each paper figure using the same explicit inputs.
    figure1(products_dir, paths, output_dir)
    figure2(products_dir, paths, output_dir)
    figure3(products_dir, paths, output_dir)
    constituent_table(
        work_dir / "gls/native_tides.h5", paths["analysis_geometry"], output_dir
    )


def main(argv=None):
    args = argument_parser(__doc__).parse_args(argv)
    setup(args)
    if args.dry_run:
        print(f"Work folder: {args.work_dir}")
        print("Build the ATL06 reference if needed; check products; draw Figures 1–3.")
        return
    make_figures(args.paths, args.work_dir)


if __name__ == "__main__":
    main()

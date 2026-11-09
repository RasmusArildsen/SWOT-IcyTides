# SWOT ice-shelf elevations and tides

Code for the Larsen C study, from the selected SWOT observations to Figures 1–3.
The default uses 191 matched SWOT HR 100 m Raster/PIXC acquisitions. The original
190-acquisition catalogue uses the same scripts with `--acquisitions 190`.

Start with [the data list](Code/DATA.md), then run these steps in order:

1. [Import the observations and build the raw cube](Code/01_import_data.py).
2. [Correct native phase ambiguities](Code/02_phase_corrections.py).
3. [Assemble geophysical and DAC corrections](Code/03_geophysical_and_dac.py).
4. [Filter the observations](Code/04_filter_observations.py).
5. [Fit the DEM and tidal maps](Code/05_dem_and_tides.py).
6. [Draw Figures 1–3 and write the statistics](Code/06_make_figures.py).

Each script uses Python function calls and ordinary loops. Calculations are
in `Code/processing/`; none of the steps launches another Python script.
Data access and raw-cube readers are in `src/s4is/`. Processing runs locally
or on an HPC machine with the required memory.

## Paper figures

The current manuscript PNGs are included in [figures/](figures/README.md):
[Figure 1](figures/figure1.png), [Figure 2](figures/figure2.png) and
[Figure 3](figures/figure3.png).

## Install and run

Use Python 3.12 in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]' -r Code/requirements.txt
cp Code/input_paths.example.json Code/input_paths.json
```

Edit the paths in `Code/input_paths.json`. Then follow the commands in
[Code/README.md](Code/README.md). Each step accepts `--dry-run`, which prints
its work without reading the data. Generated files go under `work/november191/`.

Raw observations, model grids and generated cubes stay outside Git. The earlier
phase-selection reference and optical mosaics must be supplied separately;
[DATA.md](Code/DATA.md) identifies these prerequisites. The small inventories,
map outlines and fixed Figure 3 comparison profiles are included.

```bash
python -m pytest -q
```

The tests use small fixtures. They check corrections, data selection, fitting
algebra, numerical controls and restart behaviour without a full shelf run.
[Code/METHODS.md](Code/METHODS.md) describes the scientific choices and limitations.

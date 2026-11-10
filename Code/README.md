# Larsen C: raw data to DEM and tidal maps

Start with [DATA.md](DATA.md). It lists the observations, models, reference
surface and map backgrounds used by the in the final 191-acquisition analysis.

Run the six scripts below from the repository root. Each script states what
it reads and writes, then carries out its steps with ordinary Python function
calls and loops. None starts another Python script or builds command strings.
Shared calculations are imported from `processing/`.

For example, open `02_phase_corrections.py` and read `correct_phases()` from top
to bottom: region correction, reference preparation, residual correction, then
packing the layers. The small `main()` function at the bottom only reads the
options and starts that work. Follow a function's import into `processing/`
to read the calculation behind it.

## Set up

Use Python 3.12.

```bash
python -m pip install -e '.[test]' -r Code/requirements.txt
cp Code/input_paths.example.json Code/input_paths.json
```

Edit `input_paths.json` to point to the external data. Generated cubes,
references, masks and maps go under `work/november191/`. You do not need to
enter paths for those intermediate files.

## 1. Import data

```bash
python Code/01_import_data.py
```

Reads the exact local Raster files, builds the unfiltered sparse cube and
prepares the BedMachine masks, 100 m REMA reference and common geometry.
Raster files can be in one directory or in acquisition subdirectories.

Add `--download-swot` to fetch the pinned Raster and PIXC files through
Earthdata, with the same selected dates and passes for both products. Other
external inputs must already be available. Raw Raster tiles
are kept because phase-reference preparation reads them again.

**Outputs:** `raw_cube.h5`, reference grids and `independent191/geometry.h5`.

## 2. Phase corrections

```bash
python Code/02_phase_corrections.py
```

Runs the region-wide native phase correction, prepares the original reference
histories, then runs the residual phase search. Accepted integer shifts move
both height and horizontal location before regridding.

**Outputs:** `phase_region190.h5`, `phase_residual190.h5` and native reports.

## 3. Geophysical corrections and DAC

```bash
python Code/03_geophysical_and_dac.py
```

Combines raw Raster observations with the native repairs and applies AVISO
DAC once. Raster WSE keeps its supplied geophysical convention. Native PIXC
height has already been converted with geoid, solid-Earth, pole and load-tide
fields inside the phase calculations. Residual repair values already have DAC
removed and are inserted directly. Ocean tide remains for the tidal fit.

**Outputs:** one row per acquisition and observation bands defined by the geometry.

## 4. Filtering

```bash
python Code/04_filter_observations.py
```

Saves a boolean admission mask for every date, coarse cell and fine position.
The rule is finite height plus the original REMA residual gate:

```text
abs(height - REMA_reference - centre) <= 3.5 × NMAD
centre = -2.0492310523986816 m
NMAD   =  1.6168210713386535 m
```

**Outputs:** one mask file per geometry band in `independent191/admission/`,
including counts of admitted and rejected observations. The fitter reads these masks and checks
them against the original rule.

## 5. DEM and tidal maps

```bash
python Code/05_dem_and_tides.py
```

Fits M2, S2, K1, O1, linear TIME and one static elevation per observed fine
position. It then combines the absolute observations and neighbour differences
with the connected GLS equations. Huber weighting, date support, condition
checks and solver controls keep their existing settings.

**Outputs:**

- `maps/dem_100m.h5`: `x`, `y`, `elevation`, at the original reference epoch.
- `maps/tidal_amplitudes_250m.h5`: M2/S2/K1/O1 amplitudes and CATS comparisons.
- `gls/native_tides.h5`: fitted tide/TIME coefficients and fine-cell levels.
- `figure_products/`: the matching map, DEM and transect inputs for plotting.

The named map exports copy the fitted arrays without resampling or filling.
The numerical residual and full-signal checks must pass before native export.

## 6. Figures

```bash
python Code/06_make_figures.py
```

Builds the pinned ATL06 comparison if it is missing, then renders Figures 1–3.
The figure scripts do not fit the observations again.

**Outputs:** three PNG/PDF pairs, surface and transect statistics, and
`Table1_rows.tex` under `work/november191/figures/`.

Figure 1 keeps the four-panel paper layout; Figure 2 uses the shelf maps and
western optical detail; Figure 3 has two maps and two full-width profile panels.
The Figure 3 observed curves retain the original static normalization, so a new
SWOT fit cannot change the CATS benchmark. The code verifies the dates, raw
heights, CATS predictions and footprint against the included reference before
using it. Predictions come from the selected fit. Table 1 uses the native
coefficients; phases use the paper's wrapped-median rule.

## Options and checks

To inspect a script without reading data or starting a calculation:

```bash
python Code/03_geophysical_and_dac.py --inputs Code/input_paths.example.json --dry-run
```

Each script accepts `--inputs`, `--work-dir`, and `--dry-run`. The available 
acquisition counts come from the catalogue in`aux_files/analysis.json`; 
the phase step uses the separate native inventory. Loops use the length of 
these lists rather than fixed date counts. 

Spatial band IDs come from `geometry.h5`. The chosen number of bands and fit
parts per band are set once in `aux_files/processing_layout.json` and recorded
in new geometry files. Connected solver part counts come from its prepared
metadata. Keep these settings fixed when resuming a run.

Use the same options throughout a run. Changing input paths
requires a new work folder. Each script does its work in the current Python
process. Large stages can be split into the original indexed jobs
shown in [METHODS.md](METHODS.md).

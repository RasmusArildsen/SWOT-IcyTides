# Processing methods

Run the six processing steps in the order shown in the [README](../README.md).
Download any missing satellite files first.

## Observation preparation

- Step 1 reprojects the Raster tiles to the study grid. The raw cube keeps
  tiles separate and preserves the supplied fields and precision without
  quality or geophysical filtering.
- The Raster file list gives the acquisition order and pass numbers. The
  nominal time is the mean of the tile midpoints in the filenames. The
  analysis settings list the acquisitions that use the earliest midpoint.
  [Raster file list](../file_lists/swot_raster.tsv), [analysis settings](aux_files/analysis.json)
- Step 2 applies the saved region and residual phase repairs to Raster WSE.
  The layers contain native PIXC heights regridded after phase shifts in
  height and position.
- Fits use each cell's measured illumination time. For a missing cell clock,
  Step 2 takes the acquisition median of the admitted 100 m Raster cell
  times, after the region repair and before residual repairs and REMA
  screening. The 1 October 2026 acquisition uses the original fixed interior
  sample of 6,386 EPSG:3031 cell centres. If there are no eligible times, the
  fallback is the nominal time from the Raster filenames.
  [Interior sample coordinates](aux_files/swot_reference_sample.npz)
- AVISO DAC is removed once. The region/Raster values receive it downstream;
  the residual layer already has it removed. Ocean tide is retained for fitting.
- The source cross-track guard is `abs(cross_track) >= 10000 m`. The excluded acquisition listed in the analysis settings stays empty. Extra cross-track correction coefficients are zero.
- Base/subtract/add clock arithmetic is preserved when residual repairs replace
  fine observations. TIME uses the same fixed calibration epoch and scale for
  any selection.

The analysis mask retains the inherited 1 km extension into class-2 grounded
cells beside the seeded floating LCIS component. The floating seed, shelf
boundaries and two regional geoid overwrite boxes are in
[study_area.json](aux_files/study_area.json). The screening geoid uses tiles
through the configured end date, in Raster file-list order. The ICESat-2
comparison uses the floating-component mask. Shared projections and
nearest-cell lookup are in [study_area.py](processing/study_area.py).

## Filtering and fitting

Step 3 writes the finite-height/REMA mask for each observation band. The
fitter also computes the original rule on read and requires an exact match to
any saved mask. Heights and clocks are unchanged. Missing REMA is retained.
A changed mask or source file fails the checks.

The model has eight M2/S2/K1/O1 coefficients, linear TIME and one static level
per observed fine position. Initial weights are one per admitted fine
observation. UTide 0.4.0 supplies the full Huber row multipliers; the normal
weights use their squares. The source SHA, tune 1.345, tolerance 0.001 and
50-iteration limit are retained.

Accepted fits require at least 12 unique/effective dates, full rank and a
normalized joint-design condition no larger than 20. Failed fits stay missing.
The connected model retains the original pair fits, 50 mm scatter floor,
accepted-link degree balance and working covariance calibration. All eligible
cells remain free. No tide coefficient prior, preserved mean, map smoothing
or unsupported-cell fill is introduced.

The physical normal residual must be below `1e-7`. The full tide/TIME/static
injected-signal error must be below `1e-5 m` before native export. Completed
inputs, masks and code are recorded before fitting; changes require a new run
folder.

## CATS and the ICESat-2 comparison

Current CATS comparisons multiply each native complex constituent by the
supplied flexure before interpolation, including supplied values above 100%.
The 12 km crop padding limits file reads and is not a flexure distance.
The SWOT fit retains the ocean tide in the observations.

Both ICESat-2 and SWOT elevations are referenced to the mean-tide geoid.

## Arrays and outputs

| Array | Axes |
| --- | --- |
| `height`, `normal_weight`, `admitted` | date, coarse cell, fine position |
| design matrix | coarse cell, date, 9 tide/TIME columns |
| `beta` | 9 tide/TIME coefficients, coarse cell |
| `terrain` | coarse cell, 9 fine positions |

A coarse tide cell is 250 m. Its up to nine fine positions are 100 m.
`figure_products/figure1.h5` contains the fitted fine-cell elevations at the original
reference epoch. The amplitude export uses metres; figure scripts convert to
centimetres for display. Native maps receive no coefficient smoothing or fill.

Figure 3 keeps the selected 2 km corridor, dates and admitted fine footprint.
Whole bins are withheld if any selected model point is unavailable. The
inherited five-point median applies only to displayed predictions outside the
two declared feature intervals. Observations and residual curves receive
no additional smoothing.

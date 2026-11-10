# Processing details and batch jobs

The main sequence is in [README.md](README.md). The numbered scripts use the
same numerical routines as the November analysis. This file records the
choices that matter when inspecting the code or splitting work on HPC.

## Observation preparation

- The raw Raster cube keeps source tiles separate and preserves the supplied
  fields and precision. It applies no quality or geophysical filter.
- Native phase corrections change height, longitude and latitude together.
  The region pass and residual pass use the original quality/support tests.
- The residual search uses the earlier calibration surface and the original
  leave-one-acquisition-out histories. Changing that reference changes the
  phase decisions.
- Native height is converted with geoid, solid-Earth, pole and FES load-tide
  fields. Raster WSE is used in its supplied convention.
- AVISO DAC is removed once. The region/Raster values receive it downstream;
  the residual layer already has it removed. Ocean tide is retained for fitting.
- The source cross-track guard is `abs(cross_track) >= 10000 m`. The original
  empty row 189 stays empty. Extra cross-track correction coefficients are zero.
- Base/subtract/add clock arithmetic is preserved when residual repairs replace
  fine observations. The 191-date run keeps the original 190-date TIME centre
  and span.

The analysis mask retains the inherited 1 km extension into class-2 grounded
cells beside the seeded floating LCIS component, with the existing east/south
exclusions. The native phase search uses the original floating-component mask.
The historical 2 km BedMachine annotation is diagnostic; it fixes no GLS cells.

## Filtering and fitting

Script 04 writes the finite-height/REMA mask for each observation band. The
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
accepted-link degree balance and working covariance calibration. 

## CATS and the ICESat-2 comparison

Current CATS comparisons multiply each native complex constituent by the
supplied flexure before interpolation, including supplied values above 100%.
The 12 km crop padding limits file reads and is not a flexure distance.
The earlier phase-selection reference retains its historical clipped,
nearest-flexure convention. CATS does not constrain the fitted SWOT coefficients.

The ATL06 reference retains its original pinned source window, native DAC,
mean-tide conversion, full CATS plus minor inference, within-overpass medians
and temporal screen. It is not recomputed from a new date selection.

## Arrays and outputs

| Array | Axes |
| --- | --- |
| `height`, `normal_weight`, `admitted` | date, coarse cell, fine position |
| design matrix | coarse cell, date, 9 tide/TIME columns |
| `beta` | 9 tide/TIME coefficients, coarse cell |
| `terrain` | coarse cell, 9 fine positions |

A coarse tide cell is 250 m. Its up to nine fine positions are 100 m.
`maps/dem_100m.h5` contains the fitted fine-cell elevations at the original
reference epoch. The amplitude export uses metres; figure scripts convert to
centimetres for display. Native maps receive no coefficient smoothing or fill.

Figure 3 keeps the selected 2 km corridor, dates and admitted fine footprint.
Whole bins are withheld if any selected model point is unavailable. The
inherited five-point median applies only to displayed predictions outside the
two declared feature intervals. Unsmoothed predictions are saved. Observations
and residual curves receive no additional smoothing.

## Splitting stages on HPC

The numbered scripts call the processing functions directly, running parts
sequentially in one Python process. Script 02 sets the frozen reference
settings; script 05 sets the fitting folders and date count explicitly, so
previously imported modules cannot leave those pointing at another run.
Script 06 calls the three plotting functions with the same completed products.

To distribute the large stages across separate jobs, use the
underlying commands with the runtime paths created by script 01. For example:

```bash
python Code/run.py independent --resources work/november191/resources.json --work-dir work/november191 --index 0
```

Run all required indices before collecting or assembling them:

| Stage | Indices |
| --- | --- |
| Native `region`, `residual` | 0–189 |
| Ancestor `row` | 0–189 |
| Ancestor `band`, selection-reference band | 0–15 |
| Corrected `row` | 0–190; 0–189 for the 190-date run |
| Prepared `band`, filtering | 0–15 |
| `independent`, `gls-node` | 0–63 |
| `gls-edge` | 0 to `edge_parts - 1` in `gls/prepared.json` |

The filtering command for one band is:

```bash
python Code/processing/filtering.py --input-dir work/november191/independent191 --index 0
```

After all independent parts, run `collect`, `gls-probe` and `gls-prepare`.
After all node/edge parts, run `gls-assemble`, `gls-solve` and `figure-products`.
Each uses the same `--resources` and `--work-dir` as the example above. Pass
`--acquisitions 190` consistently for the original catalogue. No script submits
scheduler jobs.

## Validation limits

Tests cover source admission, DAC-once composition, mask reuse, robust fitting,
GLS operator/signal controls and the retained scientific operations. Fresh
fits at eight shelf-spanning cells match the previous 191-date results bit
for bit when saved masks are read. Earlier full-grid checks matched all
804,643 coarse centres, 5,016,221 fine assignments and 80,056 historical
BedMachine annotations. The current paper layouts were rendered from the completed products and
visually checked. Figure 1 reference statistics and all Table 1 values match
the verified paper results. On the 214 paired Figure 3 positions, the fixed
observation comparison gives 32.9 cm for CATS and 24.6 cm for SWOT. Its original
observation normalization is checked and retained separately from the new fit.
The release passes 58 focused tests, including source pairing and preservation checks against the
recorded scientific operations.

The full shelf has not been rerun through the new numbered entry points.
The earlier calibration surface and optical mosaic remain external inputs.
New GLS marginal conditioning and calibrated uncertainties remain unvalidated.
The transect is an in-sample illustration, and REMA is used upstream for
selection. Numerical convergence, CATS resemblance and appearance alone do
not establish accuracy.

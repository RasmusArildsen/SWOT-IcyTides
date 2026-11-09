# Data used

The input catalogues match by acquisition: 191 SWOT HR 100 m Raster dates
and 191 corresponding PIXC dates, with identical cycle/pass identities.
The original catalogue has 190 acquisitions. Exact filenames are in the
manifests below; a new search window or newer product is not a substitute.

## Observations and models

| Data | Source inventory | Use |
| --- | --- | --- |
| [SWOT HR Raster, Version D](https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_Raster_D) | 191 acquisitions: 633 Raster files (629 original tiles plus four from 1 October 2026). | WSE, acquisition clocks, cross-track distance, geoid and raw product metadata. |
| [SWOT HR PIXC, Version D](https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_PIXC_D) | The same 191 acquisitions: 2,096 native files (2,080 original files plus 16 matching 1 October 2026). | Native phase-processing sources. The paper’s archived integer repairs cover the original 190 dates; see below. |
| [CATS2008_v2023](https://www.usap-dc.org/view/dataset/601772) | `CATS2008_v2023.nc`, including its supplied flexure field. | Tidal comparison and phase/reference calculations. The fitted SWOT tide is not constrained to CATS. |
| [AVISO dynamic atmospheric correction](https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/dynamic-atmospheric-correction.html) | Six-hour `dac_dif` NetCDF fields, stored by year in the AVISO cache. | Atmospheric correction at the required locations and times. |
| [REMA mosaic v2](https://www.pgc.umn.edu/data/rema/) | Original 32 m DEM tiles with `_count`, `_mad`, `_mindate` and `_maxdate` companions. | Independent 100 m reference, observation screening and native phase assessment. |
| [BedMachine Antarctica v4](https://nsidc.org/data/documentation/measures-bedmachine-antarctica-version-4-user-guide) | `NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc`. | Floating-ice mask and the inherited study-area definitions. |
| [ICESat-2 ATL06 v007](https://nsidc.org/data/atl06/versions/7) | 335 pinned files/overpass groups in the original source window (times below). | The laser-elevation comparison in Figure 1. Uses native ATL06 DAC. |

Raster and PIXC file totals are not expected to be equal. NASA distributes
[Raster scenes and smaller left/right PIXC tiles](https://podaac.github.io/tutorials/quarto_text/SWOT.html).
The required match is date,
cycle and pass. `source_inventory.py` checks this before downloading native
files; a missing acquisition, duplicate filename or incorrectly paired source
fails the check.

The native geoid, solid-Earth tide, pole tide and FES load-tide values are read
from the PIXC files. They are not downloaded as additional model grids. Raster
WSE is used as supplied; its mission corrections are not applied again.

The ATL06 source window is `2024-11-23T19:59:33 <= time <
2026-07-07T08:07:39` UTC, as recorded in `atl06_inventory.json`.

The inventories describe source files, not observations retained by the fit.
The 191-date PIXC catalogue includes all 16 native sources recorded for the four
1 October Raster files in the existing raw-cube metadata. Those records show
complete source availability and zero missing PIXC files; exact filenames were
also checked against NASA CMR.

The saved paper result reuses integer phase-repair layers for the original
190 dates. The added 1 October date uses the supplied Raster WSE without a newly
estimated integer repair. Matching the source inventories does not mean a new
191-date phase-correction analysis has been run. Existing maps and statistics
retain their original processing history.

The supplied CATS flexure scales native complex coefficients before spatial
interpolation. The code's 12 km read buffer is only padding around a model
subset. It does not set a physical flexure distance.

## Project reference and map backgrounds

| Input | Expected file | Use |
| --- | --- | --- |
| Earlier mean/reference surface | `frozen_ancestor_mean_reference_100m.nc` | Reproduces the reference screening used to select residual phase integers. It is a project calibration file, not a public data product. |
| Sentinel-2 RGB mosaic | `sentinel2_rgb_10m.tif` | Optical panel in Figure 2. Supply the existing project mosaic; these scripts do not rebuild it from individual scenes. |
| LIMA RGB mosaic | `LIMA_Larsen_C_RGB_WMS_export_100m.tif` | Shelf background in Figures 2 and 3. |
| [Natural Earth land, 1:50 million](https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-land/) | `ne_50m_land.shp` and its companion files | Source of the included locator geometry. Used during ancillary preparation. |

The earlier reference surface must contain `swot_wse_geoid_cats_corrected_mean`
and `rema_dem_ellipsoid` on the declared 100 m grid. Keep the original surface
when reproducing the phase decisions. Replacing it with the new fitted DEM
would change the analysis.

## Small files included here

| File | Contents |
| --- | --- |
| [analysis.json](aux_files/analysis.json) | Raster filenames, dates, clocks, grids and screening settings. |
| [native_inventory191.json](aux_files/native_inventory191.json) | Complete PIXC source catalogue matching the default 191 Raster acquisitions. |
| [native_inventory190.json](aux_files/native_inventory190.json) | Original PIXC catalogue used by the archived integer phase-repair history; unchanged. |
| [atl06_inventory.json](aux_files/atl06_inventory.json) | Exact ATL06 filenames and source window. |
| [legacy_coverage190.json](aux_files/legacy_coverage190.json), [reference_settings.json](aux_files/reference_settings.json) | Earlier phase-reference admission decisions and settings. |
| [figure3_sampling.h5](aux_files/figure3_sampling.h5), [figure3_line.json](aux_files/figure3_line.json) | The selected transect, dates and admitted fine-cell footprint; no measured heights or tide coefficients. |
| [figure3_reference_profiles.npz](aux_files/figure3_reference_profiles.npz) | Original observed transect profiles, dates, raw-height medians, support and CATS predictions. These keep the observation comparison fixed; they are not inputs to the tide fit. |
| [figure1_locator.json](aux_files/figure1_locator.json), [paper_map_geometry.json](aux_files/paper_map_geometry.json) | Paper locator drawing coordinates and selected ice-rise outlines from the Moholdt and Matsuoka (2015) inventory v5. |
| [source_provenance.json](aux_files/source_provenance.json) | Source fingerprints and the checks on the numerical code. |

Put local paths in `input_paths.json`, using
[input_paths.example.json](input_paths.example.json) as the template. Only
external inputs are listed there. The numbered scripts create the remaining
paths under their work folder. Earthdata and AVISO credentials stay outside Git.

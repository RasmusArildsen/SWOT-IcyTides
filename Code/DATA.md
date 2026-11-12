# Data sources

## Satellite files

| File list | Data source | Use |
| --- | --- | --- |
| [SWOT HR Raster, 100 m](../file_lists/swot_raster.tsv) | [NASA PO.DAAC, Version D](https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_Raster_D) | WSE, geoid, acquisition time and cross-track distance. |
| [SWOT HR PIXC](../file_lists/swot_pixc.tsv) | [NASA PO.DAAC, Version D](https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_PIXC_D) | Native pixel-cloud inputs for phase repair. |
| [ICESat-2 ATL06](../file_lists/icesat2_atl06.tsv) | [NSIDC, Version 7](https://nsidc.org/data/atl06/versions/7) | Elevations for the Figure 1 comparison. |

The download script uses the Raster and ATL06 lists. Use a
[NASA Earthdata Login](https://urs.earthdata.nasa.gov/) for downloads.
Place Raster files together or in acquisition subfolders and place the
ATL06 files in their own folder. Use `Code/download_data.py --local-only`
to check the lists against files on disk.

The [saved phase repair archive](https://github.com/RasmusArildsen/SWOT-IcyTides/releases/download/figure-inputs-v1/larsen-c-phase-repairs.zip)
contains the region and residual layers derived from HR PIXC and used in the
verified replay.
The layers use the Raster acquisition order. The final Raster acquisition
has no archived repair rows. Unzip into `inputs/phase_repairs/`.

## Models and corrections

| Input | Where to get it | File or folder |
| --- | --- | --- |
| REMA mosaic v2 | [Polar Geospatial Center](https://www.pgc.umn.edu/data/rema/) | Original 32 m DEM tiles and their count rasters. |
| CATS2008_v2023 | [U.S. Antarctic Program Data Center](https://www.usap-dc.org/view/dataset/601772) | `CATS2008_v2023.nc`, including the flexure field. |
| BedMachine Antarctica v4 | [NSIDC](https://nsidc.org/data/nsidc-0756/versions/4) | `NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc`. |
| Dynamic atmospheric correction | [AVISO+](https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/dynamic-atmospheric-correction.html) | Six-hourly `dac_dif` NetCDF fields in year folders. |
| LIMA mosaic | [USGS LIMA](https://lima.usgs.gov/access.php) | The Larsen C RGB GeoTIFF used in Figures 2 and 3. |
| Sentinel-2 mosaic | Project input | `sentinel2_rgb_10m.tif`, the project mosaic used in Figure 2. |

Set these paths in [the input template](input_paths.example.json).

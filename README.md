# WSED-WSBD

Data pipeline for computing wind and solar capacity factors from climate model output, used to study Wind-Solar Energy Droughts (WSED) and Wind-Solar Budget Droughts (WSBD).

## What it does

`calculate_cf.py` turns raw CMIP6 GCM variables (`tas`, `tasmax`, `rsds`, `sfcWind` or `uas`/`vas`) into downscaled wind and solar capacity factor (DS_CF) time series, bias-corrected against a reanalysis product (default: W5E5), and aggregated to country/region level.

Main steps:

- **`DS_CFConfig`** — physical constants for the wind power curve (cut-in/rated/cut-out speeds, Hellmann exponent) and the PV cell-temperature model (Huld et al.).
- **`unbias_GCM`** — trains an MBCn (multivariate bias correction) adjustment on historical GCM data against reanalysis, then applies it to future global-warming-level (GWL) time slices as parallel Dask tasks.
- **`calculate_ds_cf_reanalysis_grid_GCM`**, **`calculate_ds_cf_GCM`**, **`calculate_ds_cf_reanalysis`** — compute wind (`wcf`) and solar (`scf`) capacity factor time series from reanalysis data or bias-corrected GCM data.
- **`aggregate_ds_cf`**, **`aggregate_ds_cf_reanalysis`** — spatially aggregate `wcf`/`scf` to shapefile regions using `xagg`, weighted either by grid-cell area or by mean reference capacity factor.
- **`build_available_df`** — inventories which GCM/run/GWL combinations have already been processed, to resume batch runs.

## Dependencies

`xarray`, `xesmf`, `xclim`, `dask`, `geopandas`, `xagg`, `rasterio`, `pandas`, `numpy`

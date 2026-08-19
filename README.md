# WSED-WSBD

Code repository for the article "Global wind-solar energy droughts under climate change", Nature Communications, under review.

## What it does

`calculate_cf.py` turns raw CMIP6 GCM variables (`tas`, `rsds`, `sfcWind` or `uas`/`vas`) into downscaled wind and solar capacity factor (DS_CF) time series, bias-corrected against a reanalysis product (default: W5E5), and aggregated to country/region level. Input files (GCM and reanalysis) can be either NetCDF (`.nc`) or Zarr (`.zarr`).

Main steps:

- **`DS_CFConfig`** — physical constants for the wind power curve (cut-in/rated/cut-out speeds, reference/hub height).
- **`get_local_shear_exponent`** — fits a per-pixel Hellmann shear exponent from reanalysis 10 m/100 m wind over a reference period (default 1982-2001), via `fit_local_shear.py` (from the sibling `como24_group5/code_review` project), caching the result. Used to extrapolate 10 m wind speed to hub height (default 100 m) instead of a single global exponent.
- **`unbias_GCM`** — trains an MBCn (multivariate bias correction) adjustment on historical GCM data against reanalysis, then applies it to future global-warming-level (GWL) time slices as parallel Dask tasks.
- **`calculate_ds_cf_reanalysis_grid_GCM`**, **`calculate_ds_cf_GCM`**, **`calculate_ds_cf_reanalysis`** — compute wind (`wcf`) and solar (`scf`) capacity factor time series from reanalysis data or bias-corrected GCM data. Solar potential uses the PVGIS relative-efficiency + Faiman module-temperature model (`compute_solar_cf`, from `calculate_wind_solar_cf.py`), which only needs `tas`/`rsds`/`sfcWind` (no `tasmax`).
- **`aggregate_ds_cf`**, **`aggregate_ds_cf_reanalysis`** — spatially aggregate `wcf`/`scf` to shapefile regions using `xagg`, weighted either by grid-cell area or by mean reference capacity factor.
- **`build_available_df`** — inventories which GCM/run/GWL combinations have already been processed, to resume batch runs.

## Dependencies

`xarray`, `zarr`, `xesmf`, `xclim`, `dask`, `geopandas`, `xagg`, `rasterio`, `pandas`, `numpy`

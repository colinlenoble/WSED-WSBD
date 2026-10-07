# WSED-WSBD

Code repository for the article "Global wind-solar energy droughts under climate change", Nature Communications, under review.

## Layout

| Folder | Content |
|---|---|
| `config.py` | Paths and run parameters shared by every script (edit once per machine). Importing it also puts the four folders below on `sys.path`, so scripts import each other by module name. |
| `main_pipeline/` | Data production: capacity factors, gridded/aggregated indicators, residual-load CSVs, trend evaluation. |
| `main_figs/` | Main-text figures. |
| `supp_figs/` | Extended Data figures and tables, named after their number in `latex/dunkelflaute_review_annotated.tex` (`suppfigN_*`, `supptabN_*`). |
| `aux_code/` | ERA5 preprocessing helpers, exploratory figures, non-paper figure variants and `diagnostics/`. |

Every script is run from anywhere as `python <folder>/<script>.py [--help]`.

## Run order

1. `main_pipeline/calculate_cf.py` -- raw CMIP6 GCM variables (`tas`, `rsds`, `sfcWind` or `uas`/`vas`) to bias-adjusted wind and solar capacity factor (DS_CF) time series, aggregated to country/region level; also writes the bias-adjustment scores (Extended Data Table 1).
2. `main_pipeline/make_grid_files.py`, `main_pipeline/make_agg_files.py` -- gridded and aggregated SWED indicators.
3. `main_pipeline/trend_sev_eval.py`, `main_pipeline/trend_sev_eval_wasserstein.py` -- model-vs-ERA5 trend agreement masks and Wasserstein trend distances.
4. `main_pipeline/make_rl_files.py` -- residual-load (SWBD) CSVs used by Figs. 4-5 and Extended Data Figs. 6, 8, 9, 11, 12, 13.
5. Figure / table scripts below.

Helpers used by the pipeline: `compute_solar_cf.py`, `wind_potential.py`, `fit_local_shear.py`, `compute_shear_by_gcm.py`, `duration_decomposition.py`, `io_utils.py`.

## Figures and tables

| Paper | Script |
|---|---|
| Fig. 1 | `main_figs/fig1.py` |
| Fig. 2 | `main_figs/fig2_trend_validation_diagnosis.ipynb` |
| Fig. 3 | `main_figs/fig3.py` |
| Figs. 4-5 | `main_figs/fig45.py` |
| ED Fig. 1 | `supp_figs/suppfig1_swed_swbd_schema.py` |
| ED Fig. 2 | `supp_figs/suppfig2_mean_variables_6panel.py` |
| ED Fig. 3 | `supp_figs/suppfig3_gcm_trend_wasserstein_composite.py` |
| ED Fig. 4 | `supp_figs/suppfig4_valuebyalpha_all_gwl.py` |
| ED Fig. 5 | `supp_figs/suppfig5_projected_change_wasserstein.py` |
| ED Fig. 6 | `supp_figs/suppfig6_duration_distribution_swed_swbd.py` |
| ED Fig. 7 | `supp_figs/suppfig7_agreement_variability.py` |
| ED Fig. 8 | `supp_figs/suppfig8_driver_effects.py` |
| ED Fig. 9 | `supp_figs/suppfig9_re_share_effect.py` |
| ED Fig. 10 | `supp_figs/suppfig10_aggregation_comparison.py` |
| ED Fig. 11 | `supp_figs/suppfig11_mix_gwl_effects.py` |
| ED Fig. 12 | `supp_figs/suppfig12_combined_threshold_sensitivity.py` |
| ED Fig. 13 | `supp_figs/suppfig13_demand_sensitivity.py` |
| ED Table 1 | written by `main_pipeline/calculate_cf.py` (`config.VALIDATION_DIR`) |
| ED Table 2 | `supp_figs/supptab2_cf_sensitivity.py` |
| ED Table 3 | `supp_figs/supptab3_wind_extrapolation_sensitivity.py` |
| ED Table 4 | `supp_figs/supptab4_cf_validation.py` |

The ED scripts derived from `fig1.py`, `fig3.py` and `fig45.py` reuse those modules' data loading and plotting helpers.

## Pipeline details (`main_pipeline/calculate_cf.py`)

Input files (GCM and reanalysis) can be either NetCDF (`.nc`) or Zarr (`.zarr`).

- **`DS_CFConfig`** -- physical constants for the wind power curve (cut-in/rated/cut-out speeds, reference/hub height).
- **`get_local_shear_exponent`** -- fits a per-pixel Hellmann shear exponent from reanalysis 10 m/100 m wind over a reference period (default 1982-2001), via `fit_local_shear.py`, caching the result. Used to extrapolate 10 m wind speed to hub height (default 100 m) instead of a single global exponent.
- **`unbias_GCM`** -- trains an MBCn (multivariate bias correction) adjustment on historical GCM data against reanalysis, then applies it to future global-warming-level (GWL) time slices as parallel Dask tasks.
- **`calculate_ds_cf_reanalysis_grid_GCM`**, **`calculate_ds_cf_GCM`**, **`calculate_ds_cf_reanalysis`** -- compute wind (`wcf`) and solar (`scf`) capacity factor time series from reanalysis data or bias-corrected GCM data. Solar potential uses the PVGIS relative-efficiency + Faiman module-temperature model (`compute_solar_cf`, from `compute_solar_cf.py`), which only needs `tas`/`rsds`/`sfcWind` (no `tasmax`).
- **`aggregate_ds_cf`**, **`aggregate_ds_cf_reanalysis`** -- spatially aggregate `wcf`/`scf` to shapefile regions using `xagg`, weighted either by grid-cell area or by mean reference capacity factor.
- **`build_available_df`** -- inventories which GCM/run/GWL combinations have already been processed, to resume batch runs.

## Dependencies

`xarray`, `zarr`, `xesmf`, `xclim`, `dask`, `geopandas`, `xagg`, `rasterio`, `pandas`, `numpy`, `matplotlib`, `cartopy`, `cmocean`, `seaborn`

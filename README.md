# Solar-wind energy droughts under global warming: analysis code

Code for computing compound solar-wind energy droughts (SWED) and solar-wind
budget droughts (SWBD) from ERA5 and bias-adjusted CMIP6 projections at
several global warming levels (GWLs), and for producing every figure and
table of the paper.

- **SWED**: days on which both the wind and the solar capacity factor fall
  below their 10th percentile (non-zero values, reference period). Annual
  severity = frequency x duration x intensity.
- **SWBD**: days on which the residual load (temperature-driven demand minus
  wind and solar supply) exceeds its 99th percentile, aggregated by country
  or region.

## Data

The input data (more than 60 GB) are not included.

- Reanalysis: ERA5, regridded to the W5E5 0.5° grid
  (https://www.isimip.org/gettingstarted/input-data-bias-adjustment/details/114/).
- CMIP6 projections (ssp245, daily tas, rsds, uas/vas or sfcWind) from the
  Pangeo cloud archive. 20-year GWL windows follow the warming levels of
  Seneviratne & Hauser (2020).
- Regional aggregation shapefile (`shp_re.shp`), current and future renewable
  shares (`share_renewable.csv`) and a population raster for
  population-weighted temperature.
- Validation only: Renewables.ninja v1.1 national capacity factors and
  ENTSO-E actual generation exports.

All paths are set in `config.py`.

## Repository structure

```
config.py            paths and run parameters (imported by every script)
main_pipeline/       data production
  calculate_cf.py               bias adjustment (MBCn), wind/solar capacity factors,
                                bias-adjustment validation (ED Table 1), regional aggregation
  wind_potential.py             wind power curve and hub-height extrapolation
  compute_solar_cf.py           PVGIS solar capacity-factor model
  fit_local_shear.py            per-pixel wind shear exponent from ERA5
  compute_shear_by_gcm.py       shear exponent on each GCM's native grid
  make_grid_files.py            gridded SWED indicators per GCM run and GWL
  make_agg_files.py             regional SWED indicators
  make_rl_files.py              residual-load (SWBD) tables
  trend_sev_eval.py             GCM vs ERA5 trend evaluation and agreement masks
  trend_sev_eval_wasserstein.py Wasserstein distances used to weight the ensemble
  duration_decomposition.py     event tables and event-duration classes
  io_utils.py                   NetCDF/Zarr file helpers
main_figs/           main-text figures
  fig1.py  fig2_trend_validation_diagnosis.ipynb  fig3.py  fig45.py
  map_overlays.py               shared map layers (excluded pixels, low model agreement)
supp_figs/           Extended Data figures (suppfigN_*) and tables (supptabN_*)
  gcm_trend_agreement.py, duration_latitude.py   helpers for ED Figs. 3 and 6
```

## Environment

Two conda environments were used:

- `xenv` for most scripts;
- `xclim` for `fig3.py` and ED Figs. 4–5.

Main dependencies: `xarray`, `dask`, `zarr`, `netCDF4`, `xesmf`, `xclim`,
`xagg`, `geopandas`, `shapely`, `rasterio`, `rioxarray`, `cartopy`, `numpy`,
`pandas`, `scipy`, `matplotlib`, `seaborn`, `cmocean`, `pycountry`,
`openpyxl` (and `joypy` for the Fig. 2 notebook).

Edit the paths in `config.py` (including the `esmf.mk` and cartopy cache
locations of each environment) before running anything.

## Workflow

Run from the repository root, in this order:

1. `python main_pipeline/compute_shear_by_gcm.py`: wind shear exponent per GCM.
2. `python main_pipeline/calculate_cf.py`: for each GCM run (set at the bottom
   of the script), bias adjustment, daily capacity factors, validation scores
   and regional aggregation. The ERA5 capacity factors and their aggregation
   are produced by `calculate_ds_cf_reanalysis` and
   `aggregate_ds_cf_reanalysis`.
3. `python main_pipeline/make_rl_files.py`: residual-load tables.
4. `python main_pipeline/trend_sev_eval.py --aggregated`: trend CIs and
   agreement masks (per pixel and per region).
5. `python main_pipeline/trend_sev_eval_wasserstein.py --aggregated`:
   Wasserstein distances for ensemble weighting.
6. Figure and table scripts (below).

`make_grid_files.py` writes gridded indicators to disk but is optional:
`fig3.py` rebuilds them in memory. `make_agg_files.py` is called by ED Fig. 7.

Figures are written to `../final_figs` by default (`--output_dir`); most
scripts accept `--help`.

## Figures and tables

| Output | Script |
|---|---|
| Fig. 1 | `main_figs/fig1.py` |
| Fig. 2 | `main_figs/fig2_trend_validation_diagnosis.ipynb` |
| Fig. 3 | `main_figs/fig3.py` |
| Figs. 4–5 | `main_figs/fig45.py` |
| ED Fig. 1 (method schema) | `supp_figs/suppfig1_swed_swbd_schema.py` |
| ED Fig. 2 (reference-period means) | `supp_figs/suppfig2_mean_variables_6panel.py` |
| ED Fig. 3 (trend skill, Wasserstein) | `supp_figs/suppfig3_gcm_trend_wasserstein_composite.py` |
| ED Fig. 4 (SWED change at all GWLs) | `supp_figs/suppfig4_valuebyalpha_all_gwl.py` |
| ED Fig. 5 (weighted vs multi-model mean) | `supp_figs/suppfig5_projected_change_wasserstein.py` |
| ED Fig. 6 (event durations by latitude) | `supp_figs/suppfig6_duration_distribution_swed_swbd.py` |
| ED Fig. 7 (model agreement, variability) | `supp_figs/suppfig7_agreement_variability.py` |
| ED Fig. 8 (supply and demand drivers) | `supp_figs/suppfig8_driver_effects.py` |
| ED Fig. 9 (renewable share) | `supp_figs/suppfig9_re_share_effect.py` |
| ED Fig. 10 (aggregation weighting) | `supp_figs/suppfig10_aggregation_comparison.py` |
| ED Fig. 11 (future mix vs warming) | `supp_figs/suppfig11_mix_gwl_effects.py` |
| ED Fig. 12 (threshold sensitivity) | `supp_figs/suppfig12_combined_threshold_sensitivity.py` |
| ED Fig. 13 (demand-model sensitivity) | `supp_figs/suppfig13_demand_sensitivity.py` |
| ED Table 1 (bias-adjustment skill) | `main_pipeline/calculate_cf.py` (`validate_bias_adjustment`) |
| ED Table 2 (capacity-factor parameters) | `supp_figs/supptab2_cf_sensitivity.py` |
| ED Table 3 (wind extrapolation) | `supp_figs/supptab3_wind_extrapolation_sensitivity.py` |
| ED Table 4 (capacity-factor validation) | `supp_figs/supptab4_cf_validation.py` |

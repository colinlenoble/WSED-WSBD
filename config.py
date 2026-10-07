"""Paths and run parameters for the Compound_ER pipeline.

Code layout (every script imports this module, which lives at the repo root):
    main_pipeline/  data production (calculate_cf.py, make_grid_files.py,
                    make_agg_files.py, make_rl_files.py, trend_sev_eval*.py, ...)
    main_figs/      main-text figures (fig1.py, fig2_*.ipynb, fig3.py, fig45.py)
    supp_figs/      Extended Data figures/tables, named after their number in
                    latex/dunkelflaute_review_annotated.tex (suppfigN_*, supptabN_*)
    aux_code/       preprocessing helpers, exploratory figures and diagnostics

Importing this module puts the repo root and those four folders on sys.path,
so scripts keep importing each other by bare module name (``from fig45 import
...``). A script run directly only needs, before its first local import::

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

Every script reads its values here instead of hard-coding machine-specific
paths -- edit the values below once and every script picks them up. Values
below are the paths that used to be hard-coded separately in each script
(JUICCE HPC cluster); update them if you move to a different machine.
"""
import os as _os
import sys as _sys

CODE_ROOT = _os.path.dirname(_os.path.abspath(__file__))
CODE_DIRS = ("main_pipeline", "main_figs", "supp_figs", "aux_code")
for _d in (CODE_ROOT,) + tuple(_os.path.join(CODE_ROOT, d) for d in CODE_DIRS):
    if _d not in _sys.path:
        _sys.path.insert(1, _d)

# -------------------------
# Environment variables (HPC-specific; read before heavy imports use them)
# -------------------------
# Two conda envs were used across scripts ("xenv" for calculate_cf.py,
# fig1.py, fig45.py, make_agg_files.py, make_grid_files.py; "xclim" for
# fig3.py and its ED 4/5 scripts), each with its own esmf.mk / cartopy cache.
ESMFMKFILE_XENV = "/gpfs/workdir/shared/juicce/envs/xenv/lib/esmf.mk"
ESMFMKFILE_XCLIM = "/gpfs/workdir/shared/juicce/envs/xclim/lib/esmf.mk"
CARTOPY_DATA_DIR_XENV = "/gpfs/workdir/shared/juicce/envs/xenv/cartopy_cache"
CARTOPY_DATA_DIR_XCLIM = "/gpfs/workdir/shared/juicce/envs/xclim/cartopy_cache"

# -------------------------
# Paths
# -------------------------
PATH_FOLDER = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_raw/"        # root folder containing raw GCM / reanalysis netCDF files
PATH_PREPROCESSED = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/"
SHAPEFILE_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/shapefile_data/shp_re.shp"
SHAPEFILE_PATH_LIGHT = "/gpfs/workdir/shared/juicce/RE_Colin/shapefile_data/ne_mix_adm0_adm1_light/ne_mix_adm0_adm1.shp"
TEMP_FOLDER = "/gpfs/workdir/shared/juicce/RE_Colin/temp/"

# Model-agreement-with-ERA5 masks, built by trend_sev_eval.py's
# build_agreement_mask() from its own grid_ic_ref.nc / agg_ic_ann_sev_GCMs_all_year_*.nc
# outputs (see that module for the exact overlap test). AGREEMENT_NC_PATH is the
# per-pixel (lat, lon) mask; AGREEMENT_AGGREGATED_NC_PATH is the polygon-native
# twin built from wcf_agg_*/scf_agg_* (one value per poly_idx, same shapefile as
# AGREEMENT_SUFFIX_SHP/make_rl_files.py's RL pipeline) -- fig45.py prefers this one over
# re-aggregating the pixel mask onto polygons on the fly.
AGREEMENT_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/trend_agreement_mask_ERA5.nc"
AGREEMENT_AGGREGATED_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/trend_agreement_mask_aggregated_ERA5_v1.nc"
AGREEMENT_SUFFIX_SHP = "v1"  # shapefile-version suffix on wcf_agg_*/scf_agg_* files (see calculate_cf.py)

# Per-(GCM, run), per-pixel empirical Wasserstein trend-distance file, built by
# trend_sev_eval_wasserstein.py's wasserstein_empirical_grid(). dims (realization,
# lat, lon); variables w2_distance (raw) and w2_normalized (w2_distance / ERA5's own
# native-grid bootstrap trend std); coords GCM, run. Used by fig3.py's helpers to build
# Extended Data Fig. 5 (supp_figs/suppfig5_projected_change_wasserstein.py).
WASSERSTEIN_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/agg_wasserstein_empirical_GCMs_all_year_ERA5.nc"
# Aggregated-domain (per-polygon) twin of WASSERSTEIN_NC_PATH, built by
# trend_sev_eval_wasserstein.py's wasserstein_empirical_agg() from wcf_agg_*/
# scf_agg_* (one series per poly_idx, same AGREEMENT_SUFFIX_SHP shapefile as
# AGREEMENT_AGGREGATED_NC_PATH/make_rl_files.py's RL pipeline) instead of the full
# (lat, lon) grid. dims (realization, poly_idx); same w2_distance/w2_normalized
# variables and GCM/run coords as WASSERSTEIN_NC_PATH.
WASSERSTEIN_AGGREGATED_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/agg_wasserstein_empirical_GCMs_aggregated_ERA5_v1.nc"
SHARE_RENEWABLE_CSV = "/gpfs/workdir/shared/juicce/RE_Colin/socioeconomic_data/share_renewable.csv"
POP_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/socioeconomic_data/ppp_2020_1km_Aggregated.tif"
# Observed generation used to validate the regional reanalysis capacity factors
# (supp_figs/supptab4_cf_validation.py): Renewables.ninja v1.1 national
# hourly capacity factors, and one ENTSO-E "Actual Generation per Production
# Type" .xlsx export per country and year.
NINJA_WIND_CSV = "/gpfs/workdir/shared/juicce/RE_Colin/socioeconomic_data/ninja_europe_wind_v1.1/ninja_wind_europe_v1.1_current_national.csv"
NINJA_PV_CSV = "/gpfs/workdir/shared/juicce/RE_Colin/socioeconomic_data/ninja_europe_pv_v1.1/ninja_pv_europe_v1.1_merra2.csv"
ENTSOE_DIR = "/gpfs/workdir/shared/juicce/RE_Colin/socioeconomic_data/ENTSO-E/"
# Residual-load (SWBD) CSVs written by main_pipeline/make_rl_files.py and read
# by main_figs/fig45.py, main_figs/fig1.py and several supp_figs scripts.
RL_OUT_DIR = PATH_PREPROCESSED + "agg_datasets/rl_out/"
# Bias-adjustment skill scores appended by calculate_cf.py (Extended Data Table 1).
VALIDATION_DIR = _os.path.join(CODE_ROOT, "validation")
SUMMARY_FIGS_DIR ="/gpfs/workdir/shared/juicce/RE_Colin/figures/summary_figures/"

# Glob pattern for the reanalysis daily 10 m/100 m wind files (u10/v10/u100/v100,
# .nc or .zarr), used to fit the local wind shear exponent (see
# calculate_cf.get_local_shear_exponent). Only needed the first time -- the
# fit is cached under PATH_PREPROCESSED/ERA5/ afterwards. Used by
# calculate_ds_cf_reanalysis (native reanalysis grid, no target GCM).
ERA5_WIND_PATTERN = '/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_raw/ERA5/ERA5_daily_*.zarr'

# Regridded ERA5 archive (W5E5 0.5 deg grid, Zarr format 2; u10/v10/u100/
# v100/t2m/ssrd -- see aux_code/regrid_era5_to_w5e5.py + aux_code/convert_regrid_to_zarr2.py).
# Used by supp_figs/supptab3_wind_extrapolation_sensitivity.py to compare the three DS_CFConfig.wind_method
# options (needs u100/v100 for the 'wind100' method).
ERA5_REGRID_ZARR2_DIR = '/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_raw/ERA5/'

# Folder holding one precomputed local shear exponent file per target GCM,
# already regridded to that GCM's own native grid: shear_by_gcm/shear_exponent_{GCM}_{start}_{end}.nc
# (see main_pipeline/compute_shear_by_gcm.py). Used by calculate_ds_cf_GCM and
# calculate_ds_cf_reanalysis_grid_GCM in place of get_local_shear_exponent +
# regrid_alpha_to_grid, since alpha is already on the right grid for these 14
# GCMs -- no interpolation needed. Windows path below is where these were
# computed locally; update if you move to a different machine.
SHEAR_BY_GCM_DIR = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_raw/ERA5/shear_by_gcm"

# -------------------------
# Run parameters
# -------------------------
SSP = 'ssp245'
GWL_LIST = ['GWL0-61', 'GWL1', 'GWL1-5', 'GWL2', 'GWL3']
GWL_LEVELS = ['1.5', '2.0', '3.0']  # projection-only subset (no GWL0-61/GWL1) used by fig3.py and supp_figs/suppfig4/5
REANALYSIS = 'ERA5'
SHEAR_REF_PERIOD = ('1982-01-01', '2001-12-31')  # local wind shear exponent fit period
EXCLUDE_GCM_RUN = ['EC-Earth3-Veg-LR:r3i1p1f1', 'NorESM2-MM:r2i1p1f1']  # GCM:run pairs excluded from ensemble figures
AGREEMENT_THRESHOLD = 50.0  # % of models whose trend CI overlaps the ERA5 reference CI, below which cells/polygons are hatched on figures
SHOW_AGREEMENT_HATCHING = True  # whether fig3.py/fig45.py draw the low-agreement hatch overlay on maps at all

# Land pixels (per SHAPEFILE_PATH, "shp_re") where the ERA5 wcf reanalysis is
# exactly 0 on every day of SHEAR_REF_PERIOD -- flags a data/model artifact
# (no wind capacity factor ever computed there), not a "low wind resource"
# judgement call. Distinct from AGREEMENT_NC_PATH's GCM-trend-agreement
# hatching: a pixel is greyed out for this reason regardless of whether it
# passes or fails that separate test. Built once by fig1.py's
# build_wcf_zero_mask()/save_wcf_zero_mask(), then loaded as-is by fig1.py
# and fig3.py so every figure greys out the same pixels from the same file.
WCF_ZERO_MASK_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/wcf_zero_land_mask_ERA5.nc"

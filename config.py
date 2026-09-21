"""Paths and run parameters for the Compound_ER pipeline (calculate_cf.py,
make_grid_files.py, make_agg_files.py, fig1.py, fig2.py, fig3.py, fig45.py,
fig_persistent.py).

Every other script in this folder imports this module and reads its values
instead of hard-coding machine-specific paths -- edit the values below once
and every script picks them up. Values below are the paths that used to be
hard-coded separately in each script (JUICCE HPC cluster); update them if
you move to a different machine.
"""

# -------------------------
# Environment variables (HPC-specific; read before heavy imports use them)
# -------------------------
# Two conda envs were used across scripts ("xenv" for calculate_cf.py,
# fig1.py, fig45.py, make_agg_files.py, make_grid_files.py; "xclim" for
# fig2.py and fig3.py), each with its own esmf.mk / cartopy cache.
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
# AGREEMENT_SUFFIX_SHP/fig45.py's RL pipeline) -- fig45.py prefers this one over
# re-aggregating the pixel mask onto polygons on the fly.
AGREEMENT_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/trend_agreement_mask_ERA5.nc"
AGREEMENT_AGGREGATED_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/trend_agreement_mask_aggregated_ERA5_v1.nc"
AGREEMENT_SUFFIX_SHP = "v1"  # shapefile-version suffix on wcf_agg_*/scf_agg_* files (see calculate_cf.py)

# Per-(GCM, run), per-pixel empirical Wasserstein trend-distance file, built by
# trend_sev_eval_wasserstein.py's wasserstein_empirical_grid(). dims (realization,
# lat, lon); variables w2_distance (raw) and w2_normalized (w2_distance / ERA5's own
# native-grid bootstrap trend std); coords GCM, run. Used by fig3.py to build the
# suppfig_projected_change_valuebyalpha_GWLxx_wasserstein figures.
WASSERSTEIN_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/agg_wasserstein_empirical_GCMs_all_year_ERA5.nc"
# Aggregated-domain (per-polygon) twin of WASSERSTEIN_NC_PATH, built by
# trend_sev_eval_wasserstein.py's wasserstein_empirical_agg() from wcf_agg_*/
# scf_agg_* (one series per poly_idx, same AGREEMENT_SUFFIX_SHP shapefile as
# AGREEMENT_AGGREGATED_NC_PATH/fig45.py's RL pipeline) instead of the full
# (lat, lon) grid. dims (realization, poly_idx); same w2_distance/w2_normalized
# variables and GCM/run coords as WASSERSTEIN_NC_PATH.
WASSERSTEIN_AGGREGATED_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/agg_wasserstein_empirical_GCMs_aggregated_ERA5_v1.nc"
SHARE_RENEWABLE_CSV = "/gpfs/workdir/shared/juicce/RE_Colin/socioeconomic_data/share_renewable.csv"
POP_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/socioeconomic_data/ppp_2020_1km_Aggregated.tif"
SUMMARY_FIGS_DIR = "/gpfs/workdir/shared/juicce/RE_Colin/figures/summary_figures/"

# Glob pattern for the reanalysis daily 10 m/100 m wind files (u10/v10/u100/v100,
# .nc or .zarr), used to fit the local wind shear exponent (see
# calculate_cf.get_local_shear_exponent). Only needed the first time -- the
# fit is cached under PATH_PREPROCESSED/ERA5/ afterwards. Used by
# calculate_ds_cf_reanalysis (native reanalysis grid, no target GCM).
ERA5_WIND_PATTERN = '/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_raw/ERA5/ERA5_daily_*.zarr'

# Regridded ERA5 archive (W5E5 0.5 deg grid, Zarr format 2; u10/v10/u100/
# v100/t2m/ssrd -- see regrid_era5_to_w5e5.py + convert_regrid_to_zarr2.py).
# Used by compare_wind_methods.py to compare the three DS_CFConfig.wind_method
# options (needs u100/v100 for the 'wind100' method).
ERA5_REGRID_ZARR2_DIR = '/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_raw/ERA5/'

# Folder holding one precomputed local shear exponent file per target GCM,
# already regridded to that GCM's own native grid: shear_by_gcm/shear_exponent_{GCM}_{start}_{end}.nc
# (see shear_by_gcm/compute_shear_by_gcm.py). Used by calculate_ds_cf_GCM and
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
GWL_LEVELS = ['1.5', '2.0', '3.0']  # projection-only subset (no GWL0-61/GWL1) used by fig2.py/fig3.py
REANALYSIS = 'ERA5'
SHEAR_REF_PERIOD = ('1982-01-01', '2001-12-31')  # local wind shear exponent fit period
EXCLUDE_GCM_RUN = ['EC-Earth3-Veg-LR:r3i1p1f1']  # GCM:run pairs excluded from ensemble figures
AGREEMENT_THRESHOLD = 50.0  # % of models whose trend CI overlaps the ERA5 reference CI, below which cells/polygons are hatched on figures
SHOW_AGREEMENT_HATCHING = False  # whether fig3.py/fig45.py draw the low-agreement hatch overlay on maps at all

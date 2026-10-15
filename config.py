"""Paths and run parameters shared by every script.

Importing this module puts the repo root, main_pipeline/, main_figs/ and
supp_figs/ on sys.path, so scripts import each other by bare module name.
Paths below point at the HPC cluster used for the paper; edit them once here
to run elsewhere.
"""
import os as _os
import sys as _sys

CODE_ROOT = _os.path.dirname(_os.path.abspath(__file__))
CODE_DIRS = ("main_pipeline", "main_figs", "supp_figs")
for _d in (CODE_ROOT,) + tuple(_os.path.join(CODE_ROOT, d) for d in CODE_DIRS):
    if _d not in _sys.path:
        _sys.path.insert(1, _d)

# -------------------------
# Environment variables (one esmf.mk / cartopy cache per conda env:
# "xenv" for most scripts, "xclim" for fig3.py and ED Figs. 4-5)
# -------------------------
ESMFMKFILE_XENV = "/gpfs/workdir/shared/juicce/envs/xenv/lib/esmf.mk"
ESMFMKFILE_XCLIM = "/gpfs/workdir/shared/juicce/envs/xclim/lib/esmf.mk"
CARTOPY_DATA_DIR_XENV = "/gpfs/workdir/shared/juicce/envs/xenv/cartopy_cache"
CARTOPY_DATA_DIR_XCLIM = "/gpfs/workdir/shared/juicce/envs/xclim/cartopy_cache"

# -------------------------
# Paths
# -------------------------
PATH_FOLDER = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_raw/"        # raw GCM / reanalysis netCDF files
PATH_PREPROCESSED = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/"
SHAPEFILE_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/shapefile_data/shp_re.shp"
SHAPEFILE_PATH_LIGHT = "/gpfs/workdir/shared/juicce/RE_Colin/shapefile_data/ne_mix_adm0_adm1_light/ne_mix_adm0_adm1.shp"
TEMP_FOLDER = "/gpfs/workdir/shared/juicce/RE_Colin/temp/"

# GCM-ERA5 trend-agreement masks built by trend_sev_eval.py: per pixel, and
# per polygon (the latter is used by fig45.py).
AGREEMENT_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/trend_agreement_mask_ERA5.nc"
AGREEMENT_AGGREGATED_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/trend_agreement_mask_aggregated_ERA5_v1.nc"
AGREEMENT_SUFFIX_SHP = "v1"  # shapefile-version suffix on wcf_agg_*/scf_agg_* files

# Empirical Wasserstein trend distance to ERA5 per (GCM, run), built by
# trend_sev_eval_wasserstein.py: per pixel (realization, lat, lon) and per
# polygon (realization, poly_idx). Variables w2_distance and w2_normalized
# (divided by ERA5's bootstrap trend std).
WASSERSTEIN_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/agg_wasserstein_empirical_GCMs_all_year_ERA5.nc"
WASSERSTEIN_AGGREGATED_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/agg_wasserstein_empirical_GCMs_aggregated_ERA5_v1.nc"
# Same, per polygon, for SWBD severity (main scenario), built by trend_sev_eval_wasserstein_swbd.py.
WASSERSTEIN_SWBD_AGGREGATED_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/agg_wasserstein_empirical_swbd_GCMs_aggregated_ERA5_v1.nc"
SHARE_RENEWABLE_CSV = "/gpfs/workdir/shared/juicce/RE_Colin/socioeconomic_data/share_renewable.csv"
POP_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/socioeconomic_data/ppp_2020_1km_Aggregated.tif"
# Observed generation for CF validation (supptab4): Renewables.ninja v1.1
# national capacity factors and ENTSO-E actual generation exports.
NINJA_WIND_CSV = "/gpfs/workdir/shared/juicce/RE_Colin/socioeconomic_data/ninja_europe_wind_v1.1/ninja_wind_europe_v1.1_current_national.csv"
NINJA_PV_CSV = "/gpfs/workdir/shared/juicce/RE_Colin/socioeconomic_data/ninja_europe_pv_v1.1/ninja_pv_europe_v1.1_merra2.csv"
ENTSOE_DIR = "/gpfs/workdir/shared/juicce/RE_Colin/socioeconomic_data/ENTSO-E/"
RL_OUT_DIR = PATH_PREPROCESSED + "agg_datasets/rl_out/"  # residual-load (SWBD) CSVs from make_rl_files.py
VALIDATION_DIR = _os.path.join(CODE_ROOT, "validation")  # bias-adjustment skill scores (ED Table 1)
SUMMARY_FIGS_DIR ="/gpfs/workdir/shared/juicce/RE_Colin/figures/summary_figures/"

# Daily ERA5 10 m/100 m winds, used once to fit the local wind shear exponent
# (cached under PATH_PREPROCESSED/ERA5/ afterwards).
ERA5_WIND_PATTERN = '/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_raw/ERA5/ERA5_daily_*.zarr'

# ERA5 regridded to the W5E5 0.5 deg grid (Zarr v2; u10/v10/u100/v100/t2m/ssrd),
# used by supptab3 to compare wind extrapolation methods.
ERA5_REGRID_ZARR2_DIR = '/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_raw/ERA5/'

# Local shear exponent per GCM, already on that GCM's native grid
# (compute_shear_by_gcm.py): shear_exponent_{GCM}_{start}_{end}.nc
SHEAR_BY_GCM_DIR = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_raw/ERA5/shear_by_gcm"

# -------------------------
# Run parameters
# -------------------------
SSP = 'ssp245'
GWL_LIST = ['GWL0-61', 'GWL1', 'GWL1-5', 'GWL2', 'GWL3']
GWL_LEVELS = ['1.5', '2.0', '3.0']  # projection-only GWLs (fig3.py, ED Figs. 4-5)
REANALYSIS = 'ERA5'
SHEAR_REF_PERIOD = ('1982-01-01', '2001-12-31')  # local wind shear exponent fit period
EXCLUDE_GCM_RUN = ['EC-Earth3-Veg-LR:r3i1p1f1', 'NorESM2-MM:r2i1p1f1']  # GCM:run pairs excluded from ensemble figures
AGREEMENT_THRESHOLD = 50.0  # % of models whose trend CI overlaps ERA5's, below which maps are hatched
SHOW_AGREEMENT_HATCHING = True

# Land pixels where ERA5 wcf is 0 on every day of SHEAR_REF_PERIOD (a data
# artifact), greyed out on all maps. Built by fig1.py.
WCF_ZERO_MASK_NC_PATH = "/gpfs/workdir/shared/juicce/RE_Colin/climate_data/climate_proc/trend_evaluation/wcf_zero_land_mask_ERA5.nc"

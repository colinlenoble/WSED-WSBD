# -*- coding: utf-8 -*-
"""
Fit the local (per-pixel) Hellmann wind shear exponent (10 m -> 100 m) on
each target GCM's own native grid, for the reference period
1982-01-01..2001-12-31.

Unlike compute_era5_regrid_shear.py (which fits alpha on the ERA5_regrid
0.5 deg grid) or calculate_cf.regrid_alpha_to_grid (which interpolates an
*already-fit* alpha field onto the target grid), this script regrids the
wind speeds first: ERA5 u10/v10/u100/v100 (from ERA5_regrid, the 0.5 deg
W5E5-grid archive; see regrid_era5_to_w5e5.py) are bilinearly interpolated
onto each GCM's native lat/lon grid, and alpha is computed from those
regridded wind speeds. This is the approach requested for the 14-GCM
ensemble used in this project, so that the wind shear used to extrapolate
each GCM's own 10 m wind to hub height reflects that GCM's own resolution.

GCM grids are read directly off one real preprocessed file per GCM under
E:/preprocessed_gwl1/{GCM}/ rather than reconstructed from resolution specs
-- several of them (CanESM5, CNRM-CM6-1, MPI-ESM1-2-LR, ...) are Gaussian
(spectral) grids that are not evenly spaced in latitude.

xesmf/ESMF is not installed on this machine (config.py's ESMFMKFILE paths
are HPC-only), so this uses xarray's bilinear .interp() rather than
conservative regridding. All 14 target GCM grids are coarser than the
0.5 deg ERA5_regrid source grid, so this is downsampling, not upsampling.

Sea pixels are masked out (per-GCM, after regridding) using
shp_re.shp (same shapefile as fit_local_shear.py's DEFAULT_SHAPEFILE).

Output: one file per GCM,
    {OUT_DIR}/shear_exponent_{GCM}_1982-01-01_2001-12-31.nc
with data variables `alpha` (unitless shear exponent) and `n_days`.

Usage:
    python compute_shear_by_gcm.py                 # all 14 GCMs
    python compute_shear_by_gcm.py --gcm CanESM5    # one GCM
    python compute_shear_by_gcm.py --overwrite
"""
import argparse
import glob
import os
import sys

# Must be imported before xarray/zarr (pulled in below): on this machine,
# zarr's codec libs shadow netCDF4's HDF5 DLL of the same name if netCDF4 is
# imported second, causing "DLL load failed" at write time.
import netCDF4  # noqa: F401

import numpy as np
import xarray as xr

_FIT_LOCAL_SHEAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', 'como24_group5', 'code_review')
if _FIT_LOCAL_SHEAR_DIR not in sys.path:
    sys.path.insert(0, _FIT_LOCAL_SHEAR_DIR)
from fit_local_shear import DEFAULT_SHAPEFILE, rasterize_region_mask, safe_to_netcdf

GCMS = [
    "ACCESS-CM2", "BCC-CSM2-MR", "CanESM5", "CMCC-ESM2", "CNRM-CM6-1",
    "EC-Earth3-Veg-LR", "GFDL-ESM4", "INM-CM5-0", "IPSL-CM6A-LR",
    "KACE-1-0-G", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-MM", "TaiESM1",
]

ERA5_REGRID_DIR = r"E:/climate_data/ERA5/daily_regrid"
GCM_SRC_DIR = r"E:/preprocessed_gwl1"
OUT_DIR = r"E:/climate_data/ERA5/shear_by_gcm"
REF_PERIOD = ("1982-01-01", "2001-12-31")


def get_gcm_grid(gcm, src_dir=GCM_SRC_DIR):
    """Native 1-D lat/lon coordinates for `gcm`, read off any one of its
    preprocessed files (grid is the same across variables/runs)."""
    files = sorted(glob.glob(os.path.join(src_dir, gcm, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No preprocessed files found for {gcm!r} in {src_dir}")
    with xr.open_dataset(files[0]) as ds:
        lat = ds["lat"].values.copy()
        lon = ds["lon"].values.copy()
    return lat, lon


def load_era5_wind(era5_dir=ERA5_REGRID_DIR, ref_period=REF_PERIOD):
    """u10/v10/u100/v100 from the regridded (W5E5 0.5 deg) ERA5 archive,
    restricted to `ref_period`, sorted so latitude is ascending (required
    for .interp() below)."""
    files = sorted(glob.glob(os.path.join(era5_dir, "ERA5_daily_*.zarr")))
    if not files:
        raise FileNotFoundError(f"No ERA5_regrid zarr stores found in {era5_dir}")
    ds = xr.open_mfdataset(files, engine="zarr", chunks={}, combine="by_coords")
    ds = ds.sel(valid_time=slice(ref_period[0], ref_period[1]))
    if ds.sizes["valid_time"] == 0:
        raise ValueError(f"No timesteps in {ref_period[0]}..{ref_period[1]} among {era5_dir}")
    return ds[["u10", "v10", "u100", "v100"]].sortby("latitude")


def _extend_lon_periodic(ds, lon_dim="longitude"):
    """Pad the (global, periodic) longitude axis by one cell on each side,
    so .interp() near the -180/180 seam (e.g. a GCM grid point exactly at
    lon=-180) isn't extrapolating past ERA5's cell-center bounds."""
    lon = ds[lon_dim].values
    left = ds.isel({lon_dim: [-1]}).assign_coords({lon_dim: [lon[-1] - 360.0]})
    right = ds.isel({lon_dim: [0]}).assign_coords({lon_dim: [lon[0] + 360.0]})
    return xr.concat([left, ds, right], dim=lon_dim)


def regrid_wind_to_gcm(era5_wind, gcm_lat, gcm_lon):
    """Bilinearly interpolate u10/v10/u100/v100 onto the (gcm_lat, gcm_lon)
    grid. Target latitudes are clipped to ERA5's cell-center range (+-89.75)
    before interpolating -- GCM grids that include the exact pole (+-90,
    e.g. IPSL-CM6A-LR, NorESM2-MM) would otherwise extrapolate; the returned
    field still carries the true (unclipped) pole latitude as its
    coordinate, it just reuses the nearest ERA5 row's value there."""
    era5_ext = _extend_lon_periodic(era5_wind)
    lat_min = float(era5_wind["latitude"].min())
    lat_max = float(era5_wind["latitude"].max())
    lat_clip = np.clip(gcm_lat, lat_min, lat_max)

    out = era5_ext.interp(latitude=lat_clip, longitude=gcm_lon, method="linear")
    out = out.rename({"latitude": "lat", "longitude": "lon"})
    out = out.assign_coords(lat=("lat", gcm_lat), lon=("lon", gcm_lon))
    return out


def compute_alpha(wind_gcm, time_dim="valid_time"):
    """alpha(lat, lon) = mean_t[log(ws100/ws10)/log(10)], plus n_days."""
    ws10 = np.hypot(wind_gcm["u10"], wind_gcm["v10"])
    ws100 = np.hypot(wind_gcm["u100"], wind_gcm["v100"])
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_daily = np.log(ws100 / ws10) / np.log(10.0)
    alpha = alpha_daily.mean(time_dim).compute()
    n_days = alpha_daily.notnull().sum(time_dim).compute()
    return alpha, n_days


def compute_shear_for_gcm(gcm, out_dir=OUT_DIR, gcm_src_dir=GCM_SRC_DIR,
                          era5_dir=ERA5_REGRID_DIR, ref_period=REF_PERIOD,
                          shapefile_path=DEFAULT_SHAPEFILE, era5_wind=None,
                          overwrite=False):
    """Compute and save the GCM-grid local shear exponent for one GCM.

    era5_wind : optional, pre-loaded output of load_era5_wind(), to avoid
                reopening the ERA5_regrid archive for every GCM in a batch.
    Returns the output path.
    """
    out_path = os.path.join(out_dir, f"shear_exponent_{gcm}_{ref_period[0]}_{ref_period[1]}.nc")
    if os.path.exists(out_path) and not overwrite:
        print(f"[{gcm}] Already exists, skipping: {out_path}")
        return out_path

    print(f"[{gcm}] Reading native grid...")
    gcm_lat, gcm_lon = get_gcm_grid(gcm, gcm_src_dir)

    if era5_wind is None:
        print(f"[{gcm}] Loading ERA5_regrid wind ({ref_period[0]}..{ref_period[1]})...")
        era5_wind = load_era5_wind(era5_dir, ref_period)

    print(f"[{gcm}] Regridding wind to {gcm} grid ({len(gcm_lat)}x{len(gcm_lon)})...")
    wind_gcm = regrid_wind_to_gcm(era5_wind, gcm_lat, gcm_lon)

    print(f"[{gcm}] Computing shear exponent (triggers the dask computation)...")
    alpha, n_days = compute_alpha(wind_gcm)

    print(f"[{gcm}] Masking sea pixels with {shapefile_path}...")
    mask = rasterize_region_mask(shapefile_path, alpha["lat"], alpha["lon"])
    alpha = alpha.where(mask)
    n_days = n_days.where(mask)

    out = xr.Dataset({"alpha": alpha, "n_days": n_days})
    # alpha/n_days inherit u10's attrs (units "m s**-1", GRIB_* fields, ...)
    # through np.log/np.hypot/.mean() -- clear before setting alpha's own attrs.
    out["alpha"].attrs = {}
    out["n_days"].attrs = {}
    out["alpha"].attrs.update(
        long_name=f"Local Hellmann shear exponent, 10 m -> 100 m, on the {gcm} native grid",
        description=(
            "ERA5_regrid (W5E5 0.5 deg) u10/v10/u100/v100 bilinearly interpolated "
            f"to the {gcm} native grid, then alpha = mean_t[log(ws100/ws10)/log(10)] "
            "computed from the regridded wind speeds. Sea pixels masked out with shp_re.shp. "
            "In-sample fit -- see fit_local_shear.py's module docstring."
        ),
        gcm=gcm,
        time_slice=f"{ref_period[0]}..{ref_period[1]}",
        source="ERA5_regrid (regrid_era5_to_w5e5.py)",
        shapefile=shapefile_path,
    )
    out["n_days"].attrs["long_name"] = "Number of days averaged into alpha at each pixel"

    os.makedirs(out_dir, exist_ok=True)
    safe_to_netcdf(out, out_path)
    n_land = int(mask.sum())
    print(f"[{gcm}] Wrote {out_path} "
          f"({n_land}/{mask.size} land pixels, grid-mean alpha={float(alpha.mean()):.4f})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gcm", default=None, choices=GCMS,
                        help="Only process this GCM (default: all 14).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    gcms = [args.gcm] if args.gcm else GCMS

    print("Loading ERA5_regrid wind once, reused across all GCMs...")
    era5_wind = load_era5_wind()

    for gcm in gcms:
        compute_shear_for_gcm(gcm, era5_wind=era5_wind, overwrite=args.overwrite)

    print("Done.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Fit the local (per-pixel) Hellmann wind shear exponent (10 m -> 100 m) from
the regridded ERA5 archive (ERA5_regrid, W5E5 0.5 deg grid; produced by
regrid_era5_to_w5e5.py) over the reference period 1982-01-01..2001-12-31.

alpha(lat, lon) = mean_t[ log(ws100(t) / ws10(t)) / log(10) ]

Reuses fit_local_shear (como24_group5/code_review/fit_local_shear.py, already
generic over file_pattern/time_slice) pointed at the regridded zarr stores
instead of the native 0.25 deg ERA5 archive, so the exponent is fit on the
same grid calculate_cf.py's wcf calculation consumes it on -- no interp step
needed downstream.

Output path/filename match what calculate_cf.get_local_shear_exponent
expects for its cache ({path_preprocessed}/ERA5/shear_exponent_local_{start}_{end}.nc),
so setting config.PATH_PREPROCESSED to OUT_DIR's parent lets that function
pick this file up directly.

Usage:
    python compute_era5_regrid_shear.py
"""
import os
import sys

# Must be imported before xarray/zarr/dask (pulled in below via fit_local_shear):
# on this machine, zarr's codec libs shadow netCDF4's HDF5 DLL of the same name
# if netCDF4 is imported second, causing "DLL load failed" at write time.
import netCDF4  # noqa: F401

SRC_DIR = r"E:/climate_data/ERA5/daily_regrid"
OUT_DIR = r"E:/climate_data/ERA5"
REF_PERIOD = ("1982-01-01", "2001-12-31")

_FIT_LOCAL_SHEAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'como24_group5', 'code_review')
if _FIT_LOCAL_SHEAR_DIR not in sys.path:
    sys.path.insert(0, _FIT_LOCAL_SHEAR_DIR)
from fit_local_shear import fit_local_shear, safe_to_netcdf


def compute_era5_regrid_shear(src_dir=SRC_DIR, ref_period=REF_PERIOD, out_dir=OUT_DIR):
    """
    Fit the local Hellmann shear exponent on the regridded (W5E5-grid) ERA5
    archive over `ref_period`, from u10/v10/u100/v100, and save it to
    {out_dir}/shear_exponent_local_{start}_{end}.nc.

    Returns the output path.
    """
    file_pattern = os.path.join(src_dir, "ERA5_daily_*.zarr")
    out_path = os.path.join(out_dir, f"shear_exponent_local_{ref_period[0]}_{ref_period[1]}.nc")

    alpha_ds = fit_local_shear(file_pattern, time_slice=ref_period)
    safe_to_netcdf(alpha_ds, out_path)
    print(f"Wrote local shear exponent to {out_path}")
    print(f"Grid-mean alpha: {float(alpha_ds['alpha'].mean()):.4f}")
    return out_path


if __name__ == "__main__":
    compute_era5_regrid_shear()

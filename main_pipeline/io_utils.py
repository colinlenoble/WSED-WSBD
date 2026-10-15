"""
File lookup, opening and atomic writes for NetCDF or Zarr data (Zarr preferred
when both exist). Depends only on xarray.
"""
import os
import glob
import shutil

import xarray as xr


def match_files(pattern_base):
    """Files matching pattern_base + '.zarr', else + '.nc' (pattern_base has no
    extension). Returns (files, 'zarr' | 'netcdf' | None)."""
    zarr_files = sorted(glob.glob(pattern_base + '.zarr'))
    if zarr_files:
        return zarr_files, 'zarr'
    nc_files = sorted(glob.glob(pattern_base + '.nc'))
    if nc_files:
        return nc_files, 'netcdf'
    return [], None


def glob_any(pattern_base):
    """match_files without the format: all files for a wildcard pattern."""
    zarr_files = sorted(glob.glob(pattern_base + '.zarr'))
    if zarr_files:
        return zarr_files
    return sorted(glob.glob(pattern_base + '.nc'))


def open_dataset_any(path, chunks=None, **kwargs):
    """Open a single dataset, whether it is a NetCDF file or a Zarr store."""
    if str(path).rstrip('/\\').endswith('.zarr'):
        return xr.open_dataset(path, engine='zarr', chunks=chunks, **kwargs)
    return xr.open_dataset(path, chunks=chunks, **kwargs)


def open_mfdataset_any(paths, chunks=None, **kwargs):
    """Open one or several datasets (all NetCDF or all Zarr) as one; mfdataset
    kwargs are ignored for a single file."""
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError("No files provided to open_mfdataset_any")
    is_zarr = str(paths[0]).rstrip('/\\').endswith('.zarr')
    if len(paths) == 1:
        engine = 'zarr' if is_zarr else None
        return xr.open_dataset(paths[0], chunks=chunks,
                               **({'engine': engine} if engine else {}))
    mf_kwargs = dict(kwargs)
    if is_zarr:
        mf_kwargs['engine'] = 'zarr'
    return xr.open_mfdataset(paths, chunks=chunks, **mf_kwargs)


def safe_to_netcdf(ds, path, mode="w", **kwargs):
    """Write ds to a temporary file, then rename to path."""
    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_netcdf(tmp_path, mode=mode, **kwargs)
    os.replace(tmp_path, path)


def safe_to_zarr(ds, path, mode="w", **kwargs):
    """Write ds to a temporary zarr store, then move it to path."""
    path = path.rstrip("/\\")
    tmp_path = path + ".tmp.zarr"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    ds.to_zarr(tmp_path, mode=mode, **kwargs)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(tmp_path, path)

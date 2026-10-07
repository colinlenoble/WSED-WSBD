"""
Zarr/NetCDF-agnostic file lookup, opening, and atomic-write helpers shared
across calculate_cf.py and every fig*.py / make_*_files.py script.

Kept dependency-free (only glob/os/shutil/xarray) so importing it doesn't
drag in xesmf/xclim/xagg the way importing calculate_cf.py directly would.
"""
import os
import glob
import shutil

import xarray as xr


def match_files(pattern_base):
    """
    Search for files/stores matching pattern_base with either a '.zarr' or
    '.nc' suffix (zarr is preferred when both are present).

    pattern_base is a glob pattern *without* its trailing extension, e.g.
    "/data/GCM/tas_day_GCM_ssp245_r1i1p1f1*GWL2" -- the ".zarr"/".nc" suffix
    is appended before globbing.

    Returns
    -------
    (files, fmt) : (list of str, 'zarr' | 'netcdf' | None)
    """
    zarr_files = sorted(glob.glob(pattern_base + '.zarr'))
    if zarr_files:
        return zarr_files, 'zarr'
    nc_files = sorted(glob.glob(pattern_base + '.nc'))
    if nc_files:
        return nc_files, 'netcdf'
    return [], None


def glob_any(pattern_base):
    """
    List all files/stores matching pattern_base + '.zarr' (or '.nc' if no
    zarr matches exist), for glob patterns with wildcards that are expected
    to match several files (e.g. one per GCM). zarr is preferred when both
    are present, mirroring match_files.
    """
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
    """
    Open one or several datasets (NetCDF or Zarr, not mixed) as a single
    combined dataset. mfdataset-only kwargs (e.g. combine, parallel) are
    ignored when a single store is opened.
    """
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
    """
    Write ds to a temporary zarr store, then atomically move it to path.

    Zarr stores are directories, not single files, so the write-temp-then-
    rename trick needs rmtree/os.replace on directories instead of the
    plain file rename safe_to_netcdf uses.
    """
    path = path.rstrip("/\\")
    tmp_path = path + ".tmp.zarr"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    ds.to_zarr(tmp_path, mode=mode, **kwargs)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(tmp_path, path)

# -*- coding: utf-8 -*-
"""
Re-encode the regridded ERA5 archive (E:/climate_data/ERA5/daily_regrid,
W5E5 0.5 deg grid, produced by regrid_era5_to_w5e5.py, written in Zarr
format 3) into Zarr format 2, keeping every variable -- including u100/v100
-- so the wind-100m capacity-factor method in calculate_cf.py has a
zarr2 source to read directly.

Mirrors the (already-run) daily_regrid -> daily_regrid_wo_w100_zarr2
conversion, but keeps u100/v100 instead of dropping them.

Requires a zarr-python >=3 install to *read* the source (Zarr format 3)
store; on this machine that's the "era5_dl" conda env. Run with:

    conda run -n era5_dl python convert_regrid_to_zarr2.py
    conda run -n era5_dl python convert_regrid_to_zarr2.py --limit 2   # test

Output stores are Zarr format 2 (consolidated metadata: .zgroup/.zattrs/
.zmetadata + one .zarray/.zattrs per variable), readable by any zarr>=2
install (e.g. the "xesmf_env" / "xr_env" envs used to run calculate_cf.py).
"""
import argparse
import glob
import os
import re
import shutil

import xarray as xr

SRC_DIR = r"E:/climate_data/ERA5/daily_regrid"
DST_DIR = r"E:/climate_data/ERA5/daily_regrid_zarr2"


def write_zarr2(ds, path):
    """Write ds to path as a consolidated Zarr-format-2 store, via a
    temp-then-rename so a killed/interrupted run can't leave a partial
    store at the final path."""
    path = path.rstrip("/\\")
    tmp_path = path + ".tmp.zarr"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    # Each variable still carries the v3 source store's encoding (Zstd
    # `compressors` + a `serializer`, from opening the Zarr-format-3
    # daily_regrid store) -- xarray only overrides matching keys in the
    # `encoding=` dict passed to to_zarr, so the leftover `serializer` key
    # survives and Zarr format 2 arrays reject it. Clear it so only the
    # chunking below is applied and the default v2 compressor (blosc/lz4)
    # is picked, matching the existing daily_regrid_wo_w100_zarr2 conversion.
    ds = ds.copy()
    for name in list(ds.data_vars) + list(ds.coords):
        ds[name].encoding = {}

    ntime = ds.sizes["valid_time"]
    encoding = {
        v: {"chunks": (ntime, ds.sizes["latitude"], ds.sizes["longitude"])}
        for v in ds.data_vars
    }
    ds.to_zarr(tmp_path, mode="w", zarr_format=2, consolidated=True, encoding=encoding)

    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(tmp_path, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src_dir", default=SRC_DIR)
    ap.add_argument("--dst_dir", default=DST_DIR)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N months (for testing)")
    ap.add_argument("--overwrite", action="store_true", help="reconvert months whose output already exists")
    args = ap.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    src_paths = sorted(glob.glob(os.path.join(args.src_dir, "ERA5_daily_*.zarr")))
    if args.limit:
        src_paths = src_paths[: args.limit]

    for i, src_path in enumerate(src_paths):
        month = re.search(r"ERA5_daily_(\d{4}-\d{2})\.zarr", src_path).group(1)
        dst_path = os.path.join(args.dst_dir, f"ERA5_daily_{month}.zarr")

        if os.path.exists(dst_path) and not args.overwrite:
            print(f"[{i+1}/{len(src_paths)}] {month}: already exists, skipping", flush=True)
            continue

        print(f"[{i+1}/{len(src_paths)}] {month}", flush=True)
        ds = xr.open_zarr(src_path)
        write_zarr2(ds, dst_path)
        ds.close()

    print("Done.")


if __name__ == "__main__":
    main()

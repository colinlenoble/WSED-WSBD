"""
Land-mask + conservative regrid of the ERA5 daily zarr archive
(E:/climate_data/ERA5/daily, 0.25 deg, pole-to-pole, 0-360 lon) onto a
0.5 deg grid (W5E5 convention: cell-centered, -179.75..179.75 lon,
89.75..-89.75 lat), for every ERA5_daily_YYYY-MM.zarr store.

Pipeline (two stages, because they need two different conda envs -- see
"Environments" below):

  1. mask   -- open each source store, set ocean cells to NaN using the
               shapefile at SHAPEFILE_PATH (land = inside a polygon),
               write the result as an intermediate Zarr-format-2 store.
  2. regrid -- open each masked intermediate store, regrid onto the 0.5
               deg grid with xesmf (method="conservative", skipna=True --
               a target cell is the area-weighted average of only the
               non-NaN source cells that overlap it, renormalized, so a
               coastal target cell with a single non-NaN land pixel just
               takes that pixel's value; only an all-ocean target cell
               stays NaN), and write the final Zarr-format-2 store to
               DST_DIR. The intermediate file is deleted afterwards
               unless --keep_intermediate is passed.

Every output store (both stages) is written as a single chunk covering
its whole array, for every variable -- e.g. every regridded store is
chunked (n_days_in_month, 360, 720). The two spatial dims are always the
same fixed size, so this chunking scheme is identical across every
year-month batch; only the time-chunk length tracks that month's own
day count, same as the rest of this pipeline (see convert_regrid_to_zarr2.py).

Environments
------------
zarr-python 2.x (default `to_zarr`, Zarr format 2 only) cannot read the
source archive, which is Zarr format 3. zarr-python >=3 can read both,
but the env with it (era5_dl) has no xesmf. So:

  - stage "mask"   must run in an env with zarr-python >=3 (era5_dl).
  - stage "regrid" must run in an env with xesmf (xesmf_env); its
    zarr-python 2.x writes Zarr format 2 natively, which is also the
    only format it can read back -- hence the intermediate conversion.

Usage:
    conda run -n era5_dl    python adapt_era5.py --stage mask   --limit 2
    conda run -n xesmf_env  python adapt_era5.py --stage regrid --limit 2

    # or drive both stages (one month at a time, deleting each
    # intermediate as soon as it's been regridded, to cap disk usage):
    python adapt_era5.py --stage all
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

import numpy as np

SRC_DIR = r"E:/climate_data/ERA5/daily"
INTERMEDIATE_DIR = r"E:/climate_data/ERA5/_masked_zarr2"
DST_DIR = r"E:/climate_data/ERA5"
WEIGHTS_DIR = r"E:/climate_data/ERA5/_regrid_weights"
SHAPEFILE_PATH = r"C:\Users\colin\Documents\These\Recherche\Compound_ER\shapefiles\final_shp\shp_re.shp"

MASK_ENV = "era5_dl"
REGRID_ENV = "xesmf_env"
ESMFMKFILE = "C:/Users/colin/anaconda3/envs/xesmf_env/Library/lib/esmf.mk"

N_LAT_SRC, N_LON_SRC = 721, 1440
N_LAT_TGT, N_LON_TGT = 360, 720

MONTH_RE = re.compile(r"ERA5_daily_(\d{4}-\d{2})\.zarr")


# -------------------------
# Shared helpers
# -------------------------
def discover_months(src_dir):
    months = []
    for path in glob.glob(os.path.join(src_dir, "ERA5_daily_*.zarr")):
        m = MONTH_RE.search(os.path.basename(path.rstrip("/\\")))
        if m:
            months.append(m.group(1))
    return sorted(months)


def filter_months(months, args):
    if args.months:
        wanted = set(args.months.split(","))
        months = [m for m in months if m in wanted]
    else:
        if args.start:
            months = [m for m in months if m >= args.start]
        if args.end:
            months = [m for m in months if m <= args.end]
        if args.limit:
            months = months[: args.limit]
    return months


def write_zarr2(ds, path):
    """Write ds to path as a consolidated Zarr-format-2 store (single
    chunk per variable, covering its whole array), via a temp-then-rename
    so a killed/interrupted run can't leave a partial store at the final
    path. Works whether the running env's zarr-python is 2.x (which only
    ever writes format 2) or >=3 (which needs zarr_format=2 to be told to)."""
    import zarr

    ds = ds.copy()
    for name in list(ds.data_vars) + list(ds.coords):
        ds[name].encoding = {}
    encoding = {v: {"chunks": ds[v].shape} for v in ds.data_vars}

    path = path.rstrip("/\\")
    tmp_path = path + ".tmp.zarr"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    kwargs = dict(mode="w", consolidated=True, encoding=encoding)
    if int(zarr.__version__.split(".")[0]) >= 3:
        kwargs["zarr_format"] = 2
    ds.to_zarr(tmp_path, **kwargs)

    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(tmp_path, path)


# -------------------------
# Stage 1: land mask (run in an env with zarr-python >=3, e.g. era5_dl)
# -------------------------
def build_land_mask(shapefile_path, lat, lon):
    """Boolean DataArray, True on land (inside a shapefile polygon), same
    shape as (lat, lon). Static -- computed once and reused for every
    month, since the ERA5 grid and the shapefile don't change."""
    import geopandas as gpd
    import shapely
    from shapely.ops import unary_union

    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    land = unary_union(gdf.geometry.buffer(0).values)  # buffer(0) fixes invalid rings

    lon_180 = ((lon + 180) % 360) - 180  # ERA5 lon is 0-360; shapefile is -180..180
    lon2d, lat2d = np.meshgrid(lon_180, lat)
    mask = shapely.contains_xy(land, lon2d, lat2d)

    import xarray as xr

    return xr.DataArray(mask, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})


def stage_mask(args):
    import xarray as xr

    os.makedirs(args.intermediate_dir, exist_ok=True)
    months = filter_months(discover_months(args.src_dir), args)

    land_mask = None
    for i, month in enumerate(months):
        dst_path = os.path.join(args.intermediate_dir, f"ERA5_daily_{month}.zarr")
        if os.path.exists(dst_path) and not args.overwrite:
            print(f"[{i + 1}/{len(months)}] {month}: masked store already exists, skipping", flush=True)
            continue

        print(f"[{i + 1}/{len(months)}] mask {month}", flush=True)
        src_path = os.path.join(args.src_dir, f"ERA5_daily_{month}.zarr")
        ds = xr.open_zarr(src_path).load()

        if land_mask is None:
            land_mask = build_land_mask(args.shapefile, ds.latitude.values, ds.longitude.values)

        ds_masked = ds.where(land_mask)
        for v in ds_masked.data_vars:
            ds_masked[v] = ds_masked[v].astype(ds[v].dtype)

        write_zarr2(ds_masked, dst_path)
        ds.close()

    print("mask stage done.")


# -------------------------
# Stage 2: conservative regrid (run in an env with xesmf, e.g. xesmf_env)
# -------------------------
def edges_from_centers(centers, first_edge, last_edge):
    mid = (centers[:-1] + centers[1:]) / 2
    return np.concatenate([[first_edge], mid, [last_edge]])


def build_target_grid():
    lat = np.round(89.75 - 0.5 * np.arange(N_LAT_TGT), 3)
    lon = np.round(-179.75 + 0.5 * np.arange(N_LON_TGT), 3)
    return lat, lon


def build_regridder(lat_in, lon_in, weights_dir):
    import xesmf as xe
    import xarray as xr

    os.makedirs(weights_dir, exist_ok=True)
    lat_b_in = edges_from_centers(lat_in, 90.0, -90.0)
    lon_b_in = np.concatenate([lon_in - 0.125, [lon_in[-1] + 0.125]])
    lat_out, lon_out = build_target_grid()
    lat_b_out = edges_from_centers(lat_out, 90.0, -90.0)
    lon_b_out = np.concatenate([lon_out - 0.25, [lon_out[-1] + 0.25]])

    ds_in = xr.Dataset(coords=dict(lat=("lat", lat_in), lon=("lon", lon_in),
                                    lat_b=("lat_b", lat_b_in), lon_b=("lon_b", lon_b_in)))
    ds_out = xr.Dataset(coords=dict(lat=("lat", lat_out), lon=("lon", lon_out),
                                     lat_b=("lat_b", lat_b_out), lon_b=("lon_b", lon_b_out)))

    weight_file = os.path.join(
        weights_dir, f"era5_conservative_{N_LAT_SRC}x{N_LON_SRC}_to_{N_LAT_TGT}x{N_LON_TGT}.nc"
    )
    reuse = os.path.exists(weight_file)
    regridder = xe.Regridder(ds_in, ds_out, method="conservative", periodic=True,
                              filename=weight_file, reuse_weights=reuse)
    return regridder, lat_out, lon_out


def regrid_dataset(ds, regridder, lat_out, lon_out):
    import xarray as xr

    out_vars = {}
    for name, da in ds.data_vars.items():
        da_r = da.rename({"latitude": "lat", "longitude": "lon"})
        out = regridder(da_r, skipna=True, keep_attrs=True)
        out_vars[name] = out.rename({"lat": "latitude", "lon": "longitude"}).astype(np.float32)

    result = xr.Dataset(
        out_vars,
        coords={
            "valid_time": ds["valid_time"],
            "latitude": ("latitude", lat_out),
            "longitude": ("longitude", lon_out),
        },
        attrs=ds.attrs,
    )
    result["latitude"].attrs = ds["latitude"].attrs
    result["longitude"].attrs = ds["longitude"].attrs
    return result


def stage_regrid(args):
    os.environ.setdefault("ESMFMKFILE", ESMFMKFILE)
    import xarray as xr
    import xesmf  # noqa: F401  (import before any dask/zarr I/O below -- importing it later,
    # after ds.load() has pulled in zarr/blosc, hits a Windows DLL search-order
    # conflict that breaks shapely.lib inside xesmf.util)

    os.makedirs(args.dst_dir, exist_ok=True)
    months = filter_months(discover_months(args.intermediate_dir), args)

    regridder = lat_out = lon_out = None
    for i, month in enumerate(months):
        dst_path = os.path.join(args.dst_dir, f"ERA5_daily_{month}.zarr")
        src_path = os.path.join(args.intermediate_dir, f"ERA5_daily_{month}.zarr")

        if os.path.exists(dst_path) and not args.overwrite:
            print(f"[{i + 1}/{len(months)}] {month}: regridded store already exists, skipping", flush=True)
            if not args.keep_intermediate and os.path.exists(src_path):
                shutil.rmtree(src_path)
            continue

        print(f"[{i + 1}/{len(months)}] regrid {month}", flush=True)
        ds = xr.open_zarr(src_path).load()

        if regridder is None:
            regridder, lat_out, lon_out = build_regridder(ds.latitude.values, ds.longitude.values, args.weights_dir)

        out = regrid_dataset(ds, regridder, lat_out, lon_out)
        write_zarr2(out, dst_path)
        ds.close()

        if not args.keep_intermediate:
            shutil.rmtree(src_path)

    print("regrid stage done.")


# -------------------------
# Orchestration: run both stages, one month at a time, in their
# respective envs, so peak intermediate disk usage stays ~1 month.
# -------------------------
def stage_all(args):
    months = filter_months(discover_months(args.src_dir), args)
    this_file = os.path.abspath(__file__)

    common = [
        "--src_dir", args.src_dir,
        "--intermediate_dir", args.intermediate_dir,
        "--dst_dir", args.dst_dir,
        "--shapefile", args.shapefile,
        "--weights_dir", args.weights_dir,
    ]
    if args.overwrite:
        common.append("--overwrite")

    for i, month in enumerate(months):
        print(f"=== [{i + 1}/{len(months)}] {month} ===", flush=True)
        subprocess.run(
            ["conda", "run", "-n", MASK_ENV, "python", this_file, "--stage", "mask", "--months", month] + common,
            check=True,
        )
        regrid_cmd = ["conda", "run", "-n", REGRID_ENV, "python", this_file,
                      "--stage", "regrid", "--months", month] + common
        if args.keep_intermediate:
            regrid_cmd.append("--keep_intermediate")
        subprocess.run(regrid_cmd, check=True)

    print("all done.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["mask", "regrid", "all"])
    ap.add_argument("--src_dir", default=SRC_DIR)
    ap.add_argument("--intermediate_dir", default=INTERMEDIATE_DIR)
    ap.add_argument("--dst_dir", default=DST_DIR)
    ap.add_argument("--shapefile", default=SHAPEFILE_PATH)
    ap.add_argument("--weights_dir", default=WEIGHTS_DIR)
    ap.add_argument("--months", default=None, help="comma-separated YYYY-MM list; overrides --start/--end/--limit")
    ap.add_argument("--start", default=None, help="YYYY-MM, inclusive")
    ap.add_argument("--end", default=None, help="YYYY-MM, inclusive")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N months (for testing)")
    ap.add_argument("--overwrite", action="store_true", help="reprocess months whose output already exists")
    ap.add_argument("--keep_intermediate", action="store_true",
                     help="don't delete the masked intermediate store after regridding it")
    args = ap.parse_args()

    if args.stage == "mask":
        stage_mask(args)
    elif args.stage == "regrid":
        stage_regrid(args)
    elif args.stage == "all":
        stage_all(args)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Per-pixel Hellmann shear exponent for extrapolating wind from 10 m to 100 m,
alpha = mean_t[log(ws100/ws10) / log(10)], from ERA5 daily u10/v10/u100/v100.
The fit is in-sample, so its skill on other periods is optimistic.
"""
import argparse
import glob
import os

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.features import geometry_mask

DEFAULT_SHAPEFILE = (
    r"C:\Users\colin\Documents\These\Recherche\Compound_ER\shapefiles\final_shp\shp_re.shp"
)


def rasterize_region_mask(shapefile_path, lat, lon):
    """Boolean DataArray, True where the grid-cell center falls inside the shapefile."""
    shapefile = gpd.read_file(shapefile_path)
    transform = rasterio.transform.from_bounds(
        float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()),
        len(lon), len(lat),
    )
    mask = geometry_mask(
        geometries=shapefile.geometry,
        out_shape=(len(lat), len(lon)),
        transform=transform,
        invert=True,
        all_touched=True,
    )
    # geometry_mask row 0 = north (lat.max()); flip to match an ascending lat coordinate.
    if lat.values[0] < lat.values[-1]:
        mask = mask[::-1, :]
    return xr.DataArray(mask, dims=(lat.dims[0], lon.dims[0]),
                         coords={lat.dims[0]: lat, lon.dims[0]: lon})


def safe_to_netcdf(ds, path, mode="w", **kwargs):
    """Write ds to a temporary file, then atomically rename to path."""
    tmp_path = path + ".tmp"
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ds.to_netcdf(tmp_path, mode=mode, **kwargs)
    os.replace(tmp_path, path)


def fit_local_shear(file_pattern, shapefile_path=None, file_format=None, time_slice=None):
    """alpha(lat, lon) = mean_t[log(ws100 / ws10) / log(10)] over the days in
    `file_pattern` (optionally within `time_slice`, a (start, end) pair).
    file_format: "netcdf", "zarr" or None (from the file extension).
    Returns a Dataset with `alpha` and `n_days`."""
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {file_pattern}")
    print(f"Found {len(files)} files")
    for f in files:
        print(" ", f)

    if file_format is None:
        file_format = "zarr" if files[0].rstrip("/\\").endswith(".zarr") else "netcdf"
    if file_format not in ("netcdf", "zarr"):
        raise ValueError(f"file_format must be 'netcdf' or 'zarr', got {file_format!r}")
    open_kwargs = {"engine": "zarr", "chunks": {}} if file_format == "zarr" else {}

    ds = xr.open_mfdataset(files, combine="by_coords", **open_kwargs)
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"

    if time_slice is not None:
        ds = ds.sel({time_dim: slice(time_slice[0], time_slice[1])})
        if ds.sizes[time_dim] == 0:
            raise ValueError(
                f"No timesteps found in {time_slice[0]}..{time_slice[1]} "
                f"among the files matched by {file_pattern!r}"
            )

    # 0-360 -> -180-180, then re-sort so longitude is monotonically increasing.
    ds = ds.assign_coords(longitude=(((ds["longitude"] + 180) % 360) - 180)).sortby("longitude")

    ws10 = np.hypot(ds["u10"], ds["v10"])
    ws100 = np.hypot(ds["u100"], ds["v100"])

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_daily = np.log(ws100 / ws10) / np.log(10.0)

    if shapefile_path:
        region_mask = rasterize_region_mask(shapefile_path, ds["latitude"], ds["longitude"])
        print(f"Fraction of grid kept by region mask: {float(region_mask.mean()):.2%}")
        alpha_daily = alpha_daily.where(region_mask)

    print("Computing per-pixel mean shear exponent (this triggers the dask computation)...")
    alpha_local = alpha_daily.mean(time_dim).compute()
    n_days = alpha_daily.notnull().sum(time_dim).compute()
    print("Done.")

    out = xr.Dataset({"alpha": alpha_local, "n_days": n_days})
    # alpha_local/n_days inherit u10's attrs (units "m s**-1", GRIB_* fields, ...)
    # through np.log/np.hypot/.mean() -- clear before setting alpha's own attrs.
    out["alpha"].attrs = {}
    out["n_days"].attrs = {}
    out["alpha"].attrs.update(
        long_name="Local (per-pixel) Hellmann shear exponent, 10 m -> 100 m",
        description=(
            "Climatological mean of log(ws100/ws10)/log(10) over all available days. "
            "In-sample fit -- see wind_100m_extrapolation_2018.ipynb Section 5 / Notes."
        ),
        source_pattern=file_pattern,
        time_slice=f"{time_slice[0]}..{time_slice[1]}" if time_slice else "all",
    )
    out["n_days"].attrs["long_name"] = "Number of days averaged into alpha at each pixel"
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Fit a per-pixel shear exponent from ERA5 daily 10 m/100 m wind.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", default="E:/climate_data/ERA5/daily",
                         help="Used to build the default file pattern (ignored if --file_pattern is set).")
    parser.add_argument("--file_pattern", default=None,
                         help="Glob pattern of the ERA5 daily wind files "
                              "(default: '{data_dir}/ERA5_daily_*.zarr').")
    parser.add_argument("--shapefile", default=DEFAULT_SHAPEFILE,
                         help="Mask shapefile (empty string: full grid).")
    parser.add_argument("--file_format", default=None, choices=[None, "netcdf", "zarr"],
                         help="Force the file format (default: from the extension).")
    parser.add_argument("--start", default=None,
                         help="Reference-period start date (with --end).")
    parser.add_argument("--end", default=None,
                         help="Reference-period end date (e.g. '2001-12-31').")
    parser.add_argument("--output", default="local_shear_exponent.nc")
    args = parser.parse_args()

    file_pattern = args.file_pattern or f"{args.data_dir}/ERA5_daily_*.zarr"
    if bool(args.start) != bool(args.end):
        parser.error("--start and --end must be given together")
    time_slice = (args.start, args.end) if args.start else None

    out = fit_local_shear(file_pattern, shapefile_path=args.shapefile or None,
                          file_format=args.file_format, time_slice=time_slice)
    safe_to_netcdf(out, args.output)
    print(f"Wrote local shear exponent to {args.output}")
    print(f"Grid-mean alpha: {float(out['alpha'].mean()):.4f}")


if __name__ == "__main__":
    main()

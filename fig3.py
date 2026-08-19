# -*- coding: cp1252 -*-
import os
import config
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XCLIM
os.environ['ESMFMKFILE'] = config.ESMFMKFILE_XCLIM

import argparse
import glob
import gc

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xesmf as xe
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from string import ascii_lowercase
from itertools import groupby

import cmocean as cmo

# =============================================================================
# Figure size constants (LaTeX-compatible)
# =============================================================================
FIG_WIDTH_IN = 5.15   # column width � fontsizes in pt will match LaTeX

# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build aggregated gridded datasets (if needed) and produce "
            "value-by-alpha maps for projected compound energy drought changes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Preprocessed / raw data path (used for dataset building) ---
    parser.add_argument(
        "--preprocessed_path",
        default=config.PATH_PREPROCESSED,
        help=(
            "Root folder of the preprocessed daily wpp/spp .nc files "
            "(e.g. <GCM>/wpp_day_*.nc). Used when building the aggregated datasets."
        ),
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        default=False,
        help=(
            "Force re-building the aggregated gridded datasets even if they "
            "already exist on disk."
        ),
    )

    # --- GWL data (used for plotting) ---
    parser.add_argument(
        "--path_gwl",
        default=config.PATH_PREPROCESSED + "agg_datasets/",
        help=(
            "Root folder containing GWL sub-directories: "
            "gridded_GWL0-61/, gridded_GWL1-5/, gridded_GWL2/, gridded_GWL3/."
        ),
    )
    parser.add_argument(
        "--gwl_levels",
        nargs="+",
        default=config.GWL_LEVELS,
        help="GWL levels to process. Choose from 1.5, 2.0, 3.0 (default: all three).",
    )
    parser.add_argument(
        "--exclude_gcm",
        nargs="+",
        default=[],
        help="GCM(s) to exclude from the ensemble (default: none).",
    )
    parser.add_argument(
        "--exclude_gcm_run",
        nargs="+",
        default=config.EXCLUDE_GCM_RUN,
        help="GCM:run pairs to exclude (default: EC-Earth3-Veg-LR:r3i1p1f1).",
    )

    # --- Pre-computed data (optional) ---
    parser.add_argument(
        "--regional_csv",
        default=None,
        help=(
            "Path to a pre-computed regional CSV file "
            "(columns: GWL, region, realization, GCM, frequency, severity, duration). "
            "If not provided, the regional DataFrame is computed on-the-fly."
        ),
    )
    parser.add_argument(
        "--agreement_path",
        default=config.AGREEMENT_NC_PATH,
        help=(
            "Path to a pre-computed model-agreement DataArray (.nc). "
            "Values represent the percentage of models agreeing on the sign of change. "
            "Cells with value <= agreement_threshold are hatched on the map. "
            "If not provided, no hatching is applied."
        ),
    )
    parser.add_argument(
        "--agreement_threshold",
        type=float,
        default=config.AGREEMENT_THRESHOLD,
        help="Threshold below which cells are hatched (default: 15 %% of models agreeing).",
    )

    # --- Shapefile / output ---
    parser.add_argument(
        "--shapefile",
        default=config.SHAPEFILE_PATH,
        help="Path to the admin shapefile (.shp).",
    )
    parser.add_argument(
        "--output_dir",
        default="../final_figs",
        help="Directory where output figures are saved.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures (default: 300).")
    parser.add_argument(
        "--path_hist",
        default=config.PATH_PREPROCESSED + 'agg_datasets/ds_final_non_zero_0.1_W5E5.nc',
        help=(
            "Path to the historical ds_final .nc file (W5E5 reanalysis). "
            "Used to draw the dark grey layer for pixels with no wind potential "
            "(where historical duration is null). If not provided, falls back to "
            "the GCM baseline null pattern."
        ),
    )

    return parser.parse_args()


# =============================================================================
# -- DATASET BUILDING (from load_gridded_data_compound) ----------------------
# =============================================================================

def compute_severity(comp_da, spp_ds, wpp_ds, spp_thr, wpp_thr):
    """
    Expected shortfall: mean positive deficit on compound-event days,
    aggregated yearly.
    """
    deficit_spp = xr.where(spp_ds["spp"] < spp_thr, spp_thr - spp_ds["spp"], 0)
    deficit_wpp = xr.where(wpp_ds["wpp"] < wpp_thr, wpp_thr - wpp_ds["wpp"], 0)
    daily_deficit = deficit_spp + deficit_wpp

    masked = xr.where(comp_da == 1, daily_deficit, np.nan)
    severity = masked.resample(time="YE").mean().fillna(0)
    return severity


def duration_xr(da):
    """
    Compute mean event duration and event frequency per (year, lat, lon).

    Returns
    -------
    ds      : Dataset with 'duration' (mean days per event)
    ds_freq : Dataset with 'frequency' (number of events)
    """
    da = da.convert_calendar("standard")
    da = da.sortby("lat").sortby("lon")
    da["lat"] = da["lat"].astype(float)
    da["lon"] = da["lon"].astype(float)

    first_time = pd.Timestamp(da.time[0].values)
    da_dur = xr.concat(
        [
            xr.zeros_like(da.isel(time=0)).expand_dims(
                time=[first_time - pd.Timedelta(days=1)]
            ),
            da,
        ],
        dim="time",
    )

    start_event = da_dur.diff(dim="time", label="lower") > 0
    start_event["time"] = da.time
    start_event["year"] = start_event.time.dt.year
    id_event = start_event.cumsum(dim="time") * da
    id_event = id_event.where(id_event > 0)

    stacked = id_event.stack(z=("lat", "lon", "time"))
    valid = stacked.notnull()
    stacked = stacked.where(valid, drop=True)

    event_ids = stacked.values.astype(int)
    lat_idxs  = stacked["lat"].values
    lon_idxs  = stacked["lon"].values
    year_idxs = stacked["year"].values.astype(int)

    df = pd.DataFrame(
        {"event_id": event_ids, "lat": lat_idxs, "lon": lon_idxs, "year": year_idxs}
    )
    df["year"] = df.groupby(["event_id", "lat", "lon"])["year"].transform("min")

    combined_keys = np.core.defchararray.add(
        np.core.defchararray.add(
            np.core.defchararray.add(df["event_id"].values.astype(str), ";"),
            np.core.defchararray.add(df["lat"].values.astype(str), ";"),
        ),
        np.core.defchararray.add(
            df["lon"].values.astype(str),
            np.core.defchararray.add(";", df["year"].values.astype(str)),
        ),
    )
    unique_keys, counts = np.unique(combined_keys, return_counts=True)
    event_ids_s, lat_s, lon_s, year_s = zip(*(k.split(";") for k in unique_keys))

    dur_da = xr.DataArray(
        counts,
        dims="event_instance",
        coords={
            "event_instance": np.arange(len(counts)),
            "event_id": ("event_instance", np.array(event_ids_s, dtype=int)),
            "lat":      ("event_instance", np.array(lat_s,      dtype=float)),
            "lon":      ("event_instance", np.array(lon_s,      dtype=float)),
            "year":     ("event_instance", np.array(year_s,     dtype=int)),
        },
    ).to_dataset(name="duration")

    df2    = dur_da.to_dataframe()
    ds     = df2.groupby(["year", "lat", "lon"]).mean().to_xarray()[["duration"]]
    ds_freq = (
        df2.groupby(["year", "lat", "lon"])
        .count()["duration"]
        .to_xarray()
        .to_dataset(name="frequency")
    )
    ds_freq["frequency"] = ds_freq["frequency"].fillna(0)
    ds["duration"]       = ds["duration"].fillna(0)
    return ds, ds_freq


def _agg_output_path(preprocessed_path, gwl, gcm, run, ssp):
    """Return the standard output .nc path for one GCM / GWL combination."""
    out_dir = os.path.join(preprocessed_path, "agg_datasets", f"gridded_{gwl}")
    return os.path.join(out_dir, f"agg_{gcm}_{run}_{ssp}_{gwl}_W5E5.nc")


def _build_single_gcm(preprocessed_path, gwl, gcm, run, ssp, wpp_rea, force_rebuild=False):
    """
    Build the aggregated gridded dataset for one (GCM, GWL) pair and save it.
    Skips if the output file already exists and force_rebuild is False.
    """
    out_path = _agg_output_path(preprocessed_path, gwl, gcm, run, ssp)
    if os.path.exists(out_path) and not force_rebuild:
        print(f"    [skip] {out_path} already exists.")
        return

    print(f"    Building {gcm} / {gwl} �")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load projection data
    wpp = xr.open_dataset(
        os.path.join(preprocessed_path, gcm, f"wpp_day_{gcm}_{ssp}_{run}_{gwl}_W5E5.nc")
    )
    spp = xr.open_dataset(
        os.path.join(preprocessed_path, gcm, f"spp_day_{gcm}_{ssp}_{run}_{gwl}_W5E5.nc")
    )
    wpp["time"] = pd.to_datetime(wpp.time.dt.strftime("%Y-%m-%d").values)
    spp["time"] = pd.to_datetime(spp.time.dt.strftime("%Y-%m-%d").values)

    # Load reference (GWL0-61) data for threshold computation
    wpp_ref_paths = glob.glob(
        os.path.join(preprocessed_path, gcm, f"wpp_day_{gcm}*ssp*{run}_GWL0-61_W5E5.nc")
    )
    spp_ref_paths = glob.glob(
        os.path.join(preprocessed_path, gcm, f"spp_day_{gcm}*ssp*{run}_GWL0-61_W5E5.nc")
    )
    wpp_ref = xr.open_dataset(wpp_ref_paths[0])
    spp_ref = xr.open_dataset(spp_ref_paths[0])
    wpp_ref["time"] = pd.to_datetime(wpp_ref.time.dt.strftime("%Y-%m-%d").values)
    spp_ref["time"] = pd.to_datetime(spp_ref.time.dt.strftime("%Y-%m-%d").values)

    # 10th-percentile thresholds (over positive values only)
    wpp_thr = wpp_ref.wpp.where(wpp_ref.wpp > 0).quantile(0.1, dim="time")
    spp_thr = spp_ref.spp.where(spp_ref.spp > 0).quantile(0.1, dim="time")

    # Compound event flag
    wpp["low_wind"]   = xr.where(wpp.wpp <= wpp_thr, 1, 0)
    spp["low_solar"]  = xr.where(spp.spp <= spp_thr, 1, 0)
    compound = (wpp.low_wind * spp.low_solar).to_dataset(name="start_cooc")

    compound = compound.convert_calendar("standard")
    wpp      = wpp.convert_calendar("standard")
    spp      = spp.convert_calendar("standard")

    # Severity, duration, frequency
    severity_ds     = compute_severity(compound.start_cooc, spp, wpp, spp_thr, wpp_thr)
    ds_dur, ds_freq = duration_xr(compound.start_cooc)
    ds_dur  = ds_dur.reindex( {"lat": spp.lat, "lon": spp.lon})
    ds_freq = ds_freq.reindex({"lat": spp.lat, "lon": spp.lon})

    severity_ds["time"] = severity_ds.time.dt.year
    severity_ds         = severity_ds.rename({"time": "year"})

    pdd = severity_ds * ds_dur.duration * ds_freq.frequency

    ds_final              = ds_dur.copy()
    ds_final["frequency"] = ds_freq.frequency
    ds_final["severity"]  = severity_ds
    ds_final["pdd"]       = pdd

    comp_annual           = compound.resample(time="YE").sum()
    comp_annual["time"]   = comp_annual.time.dt.year
    comp_annual           = comp_annual.rename({"time": "year"})
    ds_final["nb_days"]   = comp_annual.start_cooc

    # Regrid to W5E5 grid and attach metadata
    ds_final            = ds_final.expand_dims({"realization": [0]})
    regrid              = xe.Regridder(ds_final, wpp_rea, method="nearest_s2d")
    ds_final            = regrid(ds_final)
    ds_final["GCM"]     = xr.DataArray([gcm], dims="realization")
    ds_final["run"]     = xr.DataArray([run],  dims="realization")
    ds_final["ssp"]     = xr.DataArray([ssp],  dims="realization")
    ds_final["gwl"]     = xr.DataArray([gwl],  dims="realization")

    ds_final.load().to_netcdf(out_path)
    print(f"    Saved ? {out_path}")

    # Release memory
    del wpp, spp, wpp_ref, spp_ref, compound, severity_ds, ds_dur, ds_freq, ds_final
    gc.collect()


def build_gridded_datasets(preprocessed_path, gwl_list, exclude_gcm=None, exclude_gcm_run=None, force_rebuild=False):
    """
    Build all aggregated gridded datasets that are missing (or all, if force_rebuild).

    Parameters
    ----------
    preprocessed_path : str
        Root directory containing per-GCM subdirectories with daily .nc files.
    gwl_list : list of str
        GWL keys to process, e.g. ['GWL0-61', 'GWL1-5', 'GWL2', 'GWL3'].
    exclude_gcm : list of str or None
        GCMs to skip entirely.
    force_rebuild : bool
        If True, rebuild even if the output file already exists.
    """
    exclude_gcm     = exclude_gcm or []
    exclude_gcm_run = set(tuple(x.split(":")) for x in (exclude_gcm_run or []))

    # Load the W5E5 reference grid (needed for regridding)
    rea_paths = glob.glob(os.path.join(preprocessed_path, "W5E5", "wpp_day*.nc"))
    if not rea_paths:
        raise FileNotFoundError(
            f"No W5E5 reference file found under {preprocessed_path}/W5E5/. "
            "Cannot determine target regrid grid."
        )
    wpp_rea = xr.open_dataset(rea_paths[0]).isel(time=slice(0, 2))

    for gwl in gwl_list:
        print(f"\n  -- Building datasets for {gwl} --")

        # Discover available projection files for this GWL
        wpp_paths = sorted(
            glob.glob(os.path.join(preprocessed_path, "*/wpp_day_*ssp*" + gwl + "_W5E5.nc"))
        )
        if not wpp_paths:
            print(f"  No files found for {gwl}, skipping.")
            continue

        gcm_list = [p.split("_")[-5] for p in wpp_paths]
        run_list = [p.split("_")[-3] for p in wpp_paths]
        ssp_list = [p.split("_")[-4] for p in wpp_paths]

        for i, (gcm, run, ssp) in enumerate(zip(gcm_list, run_list, ssp_list)):
            if gcm in exclude_gcm or (gcm, run) in exclude_gcm_run:
                print(f"    [excluded] {gcm} {run}")
                continue
            try:
                _build_single_gcm(
                    preprocessed_path, gwl, gcm, run, ssp, wpp_rea,
                    force_rebuild=force_rebuild,
                )
            except Exception as exc:
                print(f"    [ERROR] {gcm} / {gwl}: {exc}")
            gc.collect()


# =============================================================================
# Helper functions (plotting side)
# =============================================================================

def rasterize_shapefile(shapefile, shape, transform):
    geometries = shapefile["geometry"]
    mask = geometry_mask(
        geometries=geometries,
        all_touched=True,
        out_shape=shape,
        transform=transform,
        invert=True,
    )
    return mask


def build_land_mask(ref_2d, shapefile_path):
    """
    Build a boolean land mask on the lat/lon grid of ref_2d (a 2-D DataArray).
    ref_2d must have dims (lat, lon) � no time or realization.
    """
    shapefile = gpd.read_file(shapefile_path)
    transform = rasterio.transform.from_bounds(
        ref_2d.lon.min().item(), ref_2d.lat.min().item(),
        ref_2d.lon.max().item(), ref_2d.lat.max().item(),
        len(ref_2d.lon), len(ref_2d.lat),
    )
    mask = rasterize_shapefile(shapefile, ref_2d.shape, transform)
    mask = mask[::-1, :]
    mask_update = ref_2d.isnull()
    mask = mask & (~mask_update)
    return mask


def _reduce_to_2d(da):
    """Average out 'year' and 'realization' dims to obtain a (lat, lon) DataArray."""
    if "year" in da.dims:
        da = da.mean(dim="year")
    if "realization" in da.dims:
        da = da.mean(dim="realization")
    return da


def load_gwl_dataset(path_gwl, gwl_key, exclude_gcm, exclude_gcm_run=None):
    """Load a GWL dataset and filter out unwanted GCMs."""
    files = sorted(glob.glob(os.path.join(path_gwl, f"gridded_{gwl_key}", f"agg*{gwl_key}*.nc")))
    if not files:
        raise FileNotFoundError(f"No files found under {path_gwl}/gridded_{gwl_key}/agg*{gwl_key}*.nc")
    print(f"  Found {len(files)} file(s) for {gwl_key}")
    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="realization",
        join="override",
    )
    exclude_gcm_run = set(tuple(x.split(":")) for x in (exclude_gcm_run or []))
    gcms = ds.GCM.values
    runs = ds.run.values
    keep = [i for i, (g, r) in enumerate(zip(gcms, runs))
            if g not in exclude_gcm and (g, r) not in exclude_gcm_run]
    ds = ds.isel(realization=keep)
    return ds


def load_baseline_dataset(path_gwl, exclude_gcm, exclude_gcm_run=None):
    """Load the GWL0-61 baseline dataset."""
    files = sorted(glob.glob(os.path.join(path_gwl, "gridded_GWL0-61", "agg*.nc")))
    if not files:
        raise FileNotFoundError(f"No files found under {path_gwl}/gridded_GWL0-61/agg*.nc")
    print(f"  Found {len(files)} file(s) for GWL0-61")
    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="realization",
        join="override"  # handles any coord conflicts on lat/lon/year
    )
    exclude_gcm_run = set(tuple(x.split(":")) for x in (exclude_gcm_run or []))
    gcms = ds.GCM.values
    runs = ds.run.values
    keep = [i for i, (g, r) in enumerate(zip(gcms, runs))
            if g not in exclude_gcm and (g, r) not in exclude_gcm_run]
    ds = ds.isel(realization=keep)
    return ds


# =============================================================================
# Align baseline and projection datasets by common GCMs
# =============================================================================

def from_ds_to_plot_decomp(ds_gwl, ds_ref):
    """
    Align baseline (ds_ref) and projection (ds_gwl) datasets by common GCMs.
    Returns (ref_freq, ref_sev, ref_dur, proj_freq, proj_sev, proj_dur, weight).
    """
    gcm_gwl = ds_gwl.GCM.values
    gcm_ref = ds_ref.GCM.values

    # Find indices *separately* in each dataset, then sort both by GCM name
    # so that paired realizations correspond to the same GCM.
    common_gcms = set(gcm_ref) & set(gcm_gwl)
    ref_indices = [i for i, g in enumerate(gcm_ref) if g in common_gcms]
    gwl_indices = [i for i, g in enumerate(gcm_gwl) if g in common_gcms]
    ds_ref = ds_ref.isel(realization=ref_indices)
    ds_gwl = ds_gwl.isel(realization=gwl_indices)

    # Sort both by GCM name so realizations are paired consistently
    ref_order = np.argsort(ds_ref.GCM.values)
    gwl_order = np.argsort(ds_gwl.GCM.values)
    ds_ref = ds_ref.isel(realization=ref_order)
    ds_gwl = ds_gwl.isel(realization=gwl_order)

    if "year" in ds_gwl.dims:
        ds_gwl = ds_gwl.mean(dim="year")
    if "year" in ds_ref.dims:
        ds_ref = ds_ref.mean(dim="year")

    ds_gwl["realization"] = ds_ref.realization.astype(int)

    # Promote GCM (data variable) to a coordinate so it is carried by every
    # DataArray extracted from this dataset (needed by the GCM bootstrap).
    ds_gwl = ds_gwl.assign_coords(GCM=("realization", ds_gwl.GCM.values))

    weight_count = pd.Series(ds_gwl.GCM.values).value_counts()
    weights = [1.0 / weight_count[g] / weight_count.size for g in ds_gwl.GCM.values]
    weight = xr.DataArray(weights, dims="realization")

    return (
        ds_ref.frequency, ds_ref.severity, ds_ref.duration,
        ds_gwl.frequency, ds_gwl.severity, ds_gwl.duration,
        weight,
    )


# =============================================================================
# Regional DataFrame
# =============================================================================

def _robust_slice(da, lat_lo, lat_hi, lon_lo, lon_hi):
    lat_slice = (slice(lat_lo, lat_hi) if da.lat[0] < da.lat[-1]
                 else slice(lat_hi, lat_lo))
    lon_slice = (slice(lon_lo, lon_hi) if da.lon[0] < da.lon[-1]
                 else slice(lon_hi, lon_lo))
    return da.sel(lat=lat_slice, lon=lon_slice)


def create_dataframe_regional(path_gwl, mask, exclude_gcm, exclude_gcm_run=None, regions=None):
    """
    For each GWL level and each region, extract the spatial mean of frequency,
    severity and duration per realization.
    Datasets are loaded one at a time and freed immediately to limit memory use.
    """
    if regions is None:
        regions = [
            {"name": "Western U.S.",    "lat": [35,  50],  "lon": [-125, -105]},
            {"name": "Southern Europe", "lat": [35,  50],  "lon": [5,   25]},
            {"name": "South Africa",    "lat": [-35, -22], "lon": [16,     33]},
            {"name": "Kenya",           "lat": [-5,   5],  "lon": [33,     42]},
            {"name": "India",           "lat": [10,  30],  "lon": [70,     90]},
        ]

    # Loaders are callables so datasets are opened one at a time, not all at once
    gwl_loaders = [
        ("GWL0-61", lambda: load_baseline_dataset(path_gwl, exclude_gcm, exclude_gcm_run)),
        ("GWL1-5",  lambda: load_gwl_dataset(path_gwl, "GWL1-5", exclude_gcm, exclude_gcm_run)),
        ("GWL2",    lambda: load_gwl_dataset(path_gwl, "GWL2",   exclude_gcm, exclude_gcm_run)),
        ("GWL3",    lambda: load_gwl_dataset(path_gwl, "GWL3",   exclude_gcm, exclude_gcm_run)),
    ]

    rows = []
    for gwl_label, loader in gwl_loaders:
        print(f"  Loading {gwl_label} �")
        ds = loader()

        ds["frequency"] = ds["frequency"].where(mask == 1)
        ds["severity"]  = ds["severity"].where(mask == 1)
        ds["duration"]  = ds["duration"].where(mask == 1)
        if "year" in ds.dims:
            ds = ds.mean(dim="year")

        # Compute eagerly so all dask arrays are resolved before we free the dataset
        ds = ds.compute()
        gcms = ds.GCM.values

        for reg in regions:
            lat_lo, lat_hi = reg["lat"]
            lon_lo, lon_hi = reg["lon"]

            freq_sub  = _robust_slice(ds.frequency, lat_lo, lat_hi, lon_lo, lon_hi)
            sev_sub   = _robust_slice(ds.severity,  lat_lo, lat_hi, lon_lo, lon_hi)
            dur_sub   = _robust_slice(ds.duration,  lat_lo, lat_hi, lon_lo, lon_hi)

            freq_mean = freq_sub.mean(dim=("lat", "lon"), skipna=True).values
            sev_mean  = sev_sub.mean( dim=("lat", "lon"), skipna=True).values
            dur_mean  = dur_sub.mean( dim=("lat", "lon"), skipna=True).values

            for ridx in range(len(gcms)):
                rows.append({
                    "GWL":         gwl_label,
                    "region":      reg["name"],
                    "realization": ridx,
                    "GCM":         gcms[ridx],
                    "frequency":   float(freq_mean[ridx]),
                    "severity":    float(sev_mean[ridx]),
                    "duration":    float(dur_mean[ridx]),
                })

        del ds
        gc.collect()

    return pd.DataFrame(rows)


def add_pdd_and_weights(df):
    """Add 'pdd' column and inverse-frequency GCM 'weight' column."""
    df = df.copy()
    df["pdd"] = df["frequency"] * df["severity"] * df["duration"]

    anchor_region = df["region"].iloc[0]
    anchor  = df[(df["region"] == anchor_region) & (df["GWL"] == "GWL0-61")]
    wcount  = anchor["GCM"].value_counts()
    weight_dict = {gcm: 1.0 / wcount[gcm] / wcount.size for gcm in wcount.index}
    df["weight"] = df["GCM"].map(weight_dict)

    return df


# =============================================================================
# Main figure � value-by-alpha map + regional boxplots
# =============================================================================

def plot_gwl_valuebyalpha_discrete(
    da_ref_freq, da_ref_sev, da_ref_dur,
    da_proj_freq, da_proj_sev, da_proj_dur,
    weight,
    mask,
    shapefile_path,
    df_regions,
    gwl_label,
    hatchings=None,
    agreement_threshold=15.0,
    map_title=None,
    relchange_label="Relative change (%)",
    sev_label="Average annual\nseverity (0.61 �C)",
    lat_min=-60,
    lat_max=68,
    regions=None,
    n_bins_change=5,
    n_bins_sev=5,
    hist_null_da=None,
):
    if regions is None:
        regions = [
            {"name": "Western U.S.",    "lat": [35,  50],  "lon": [-125, -105]},
            {"name": "South Africa",    "lat": [-35, -22], "lon": [16,     33]},
            {"name": "Kenya",           "lat": [-5,   5],  "lon": [33,     42]},
            {"name": "India",           "lat": [10,  30],  "lon": [70,     90]},
        ]

    # --- 1. Compound index ---
    base_comp = da_ref_freq  * da_ref_sev  * da_ref_dur
    proj_comp = da_proj_freq * da_proj_sev * da_proj_dur

    def _crop_lat(da):
        return da.where(da.lat > lat_min, drop=True).where(da.lat < lat_max, drop=True)

    base_comp = _crop_lat(base_comp).where(mask == 1)
    proj_comp = _crop_lat(proj_comp).where(mask == 1)

    # --- 2. Weighted ensemble mean ---
    base_mean  = base_comp.weighted(weight).mean(dim="realization")
    proj_mean  = proj_comp.weighted(weight).mean(dim="realization")
    rel_change = 100.0 * (proj_mean - base_mean) / base_mean
    rel_change = rel_change.where(np.isfinite(rel_change))
    sev        = base_mean
    dChange    = rel_change

    # --- 3. Discrete colour bins ---
    change_edges = [-100, -25, -10, 10, 25, 100]
    change_bin   = np.digitize(dChange.values, change_edges[1:-1])

    base_cmap    = cm.get_cmap("coolwarm")
    color_levels = base_cmap(np.linspace(0, 1, n_bins_change))

    # --- 4. Discrete alpha bins ---
    max_sev   = 1.0
    sev_edges = np.linspace(0, max_sev, n_bins_sev + 1) ** 2 / max_sev
    sev_bin   = np.digitize(sev.values, sev_edges[1:-1])

    alpha_min, alpha_max = 0.4, 1.0
    alpha_levels = np.linspace(alpha_min, alpha_max, n_bins_sev)

    # --- 5. RGBA assembly ---
    nlat, nlon = dChange.shape
    valid_mask = np.isfinite(dChange.values) & np.isfinite(sev.values)
    rgba_map   = np.zeros((nlat, nlon, 4), dtype=float)
    cb = np.clip(change_bin, 0, n_bins_change - 1)
    sb = np.clip(sev_bin,    0, n_bins_sev    - 1)
    rgba_map[valid_mask, :3] = color_levels[cb[valid_mask], :3]
    rgba_map[valid_mask,  3] = alpha_levels[sb[valid_mask]]

    # --- 6. Figure layout (LaTeX-compatible width) ---
    fig_width_in  = FIG_WIDTH_IN
    fig_height_in = fig_width_in * (12 / 20)   # keep original aspect ratio

    ncols  = max(1, len(regions))
    fig    = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=300)
    gs     = GridSpec(2, ncols, height_ratios=[2.9, 1], hspace=0.4, wspace=0.5, figure=fig)
    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())

    # --- 7. Draw map ---
    shp = gpd.read_file(shapefile_path)
    ax_map.imshow(
        rgba_map,
        extent=[sev.lon.min().item(), sev.lon.max().item(),
                sev.lat.min().item(), sev.lat.max().item()],
        origin="lower",
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        rasterized=True,
    )

    da_mask = da_ref_freq.isel(realization=0)
    t_mask  = rasterio.transform.from_bounds(
        da_mask.lon.min().item(), da_mask.lat.min().item(),
        da_mask.lon.max().item(), da_mask.lat.max().item(),
        len(da_mask.lon), len(da_mask.lat),
    )
    land_shp   = rasterize_shapefile(shp, da_mask.shape, t_mask)
    land_shp   = land_shp[::-1, :]
    # Dark grey layer: land pixels with no wind potential
    # Prefer the W5E5 historical null mask (hist_null_da); fall back to GCM baseline mask.
    if hist_null_da is not None:
        _null_float = hist_null_da.astype(float).interp(
            lat=da_mask.lat, lon=da_mask.lon, method="nearest"
        )
        no_wind_mask = land_shp & (_null_float.values > 0.5)
    else:
        _mask_float = mask.astype(float).interp(
            lat=da_mask.lat, lon=da_mask.lon, method="nearest"
        )
        no_wind_mask = land_shp & (_mask_float.values < 0.5)
    ax_map.contourf(
        da_mask.lon, da_mask.lat, no_wind_mask.astype(float),
        levels=[0.5, 1], colors=["#404040"],
        transform=ccrs.PlateCarree(), zorder=5,
    )
    shp.boundary.plot(ax=ax_map, color="black", linewidth=0.15,
                      transform=ccrs.PlateCarree(), zorder=10)

    if hatchings is not None:
        ax_map.contourf(
            hatchings.lon, hatchings.lat,
            (hatchings <= agreement_threshold).values.astype(float),
            transform=ccrs.PlateCarree(),
            colors="none", levels=[0.5, 1.5],
            hatches=[21 * "/", 21 * "/"], zorder=8,
        )

    if map_title is None:
        map_title = f"Projected change in annual severity under {gwl_label} warming"
    elif "{gwl_label}" in map_title:
        map_title = map_title.format(gwl_label=gwl_label)
    ax_map.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.15)
    ax_map.annotate(
        "$\\mathbf{a}$",
        xy=(0.02, 1.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=7,
        path_effects=[withStroke(linewidth=1.5, foreground="white")],
    )
    ax_map.set_title(map_title, fontsize=7, pad=6)
    ax_map.set_extent([-180, 180, -58, 68], crs=ccrs.PlateCarree())
    ax_map.spines["geo"].set_visible(False)

    # --- 8. Legend block ---
    legend_rgba = np.zeros((n_bins_change, n_bins_sev, 4))
    for ic in range(n_bins_change):
        legend_rgba[ic, :, :3] = color_levels[ic, :3]
        legend_rgba[ic, :,  3] = alpha_levels

    legend_ax = fig.add_axes([0.2, 0.45, 0.16, 0.16])
    legend_ax.imshow(legend_rgba, origin="lower", aspect="equal")
    legend_ax.set_xticks([0, n_bins_sev // 2, n_bins_sev - 1])
    legend_ax.set_xticklabels(["low", "mid", "high"], fontsize=5, ha="center")
    legend_ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    legend_ax.set_yticklabels(["-25%", "-10%", "10%", "25%"], fontsize=5, va="center")
    legend_ax.set_xlabel(sev_label, fontsize=5, labelpad=4)
    legend_ax.set_ylabel(relchange_label, fontsize=5, labelpad=4)
    legend_ax.tick_params(axis="both", which="both", length=0)

    # --- 9. Region boxes on map ---
    panellabels = [chr(98 + i) for i in range(len(regions))]
    for ridx, reg in enumerate(regions):
        lat_lo, lat_hi = reg["lat"]
        lon_lo, lon_hi = reg["lon"]
        ax_map.plot(
            [lon_lo, lon_hi, lon_hi, lon_lo, lon_lo],
            [lat_lo, lat_lo, lat_hi, lat_hi, lat_lo],
            color="black", linewidth=0.5,
            transform=ccrs.PlateCarree(),
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )
        ax_map.annotate(
            panellabels[ridx],
            xy=(lon_lo + 0.5, lat_hi - 0.5),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax_map),
            fontsize=6, fontweight="bold",
            path_effects=[withStroke(linewidth=2, foreground="white")],
            zorder=1000,
        )

    # --- 10. Regional violin plots (all 4 GWL levels) ---
    gwl_order   = ["GWL0-61", "GWL1-5", "GWL2", "GWL3"]
    gwl_display = ["0.61�C",  "1.5�C",  "2.0�C", "3.0�C"]

    # Shared y-axis limits across all regions and GWL levels
    y_min, y_max = float("inf"), float("-inf")
    for reg in regions:
        df_reg = df_regions[df_regions["region"] == reg["name"]]
        for gwl_grp in gwl_order:
            vals = df_reg[df_reg["GWL"] == gwl_grp]["pdd"].dropna().values
            if vals.size > 0:
                y_min = min(y_min, np.nanmin(vals))
                y_max = max(y_max, np.nanmax(vals))
    y_min *= 0.95
    y_max *= 1.05

    for ridx, reg in enumerate(regions):
        ax_ts  = fig.add_subplot(gs[1, ridx])
        df_reg = df_regions[df_regions["region"] == reg["name"]]

        data_box = [
            df_reg[df_reg["GWL"] == gwl_grp]["pdd"].dropna().values
            for gwl_grp in gwl_order
        ]

        # Scatter dots: light blue, very small, behind the violin
        for i, gwl_grp in enumerate(gwl_order):
            sub = df_reg[df_reg["GWL"] == gwl_grp]["pdd"].dropna().values
            x_jitter = np.random.normal(i + 1, 0.07, size=len(sub))
            ax_ts.scatter(x_jitter, sub, s=0.8, color="#0a3a60", alpha=0.5,
                          linewidths=0, zorder=4)

        # Violin: dark blue distribution on top of dots
        violin_data = [d for d in data_box if d.size > 1]
        violin_pos  = [i + 1 for i, d in enumerate(data_box) if d.size > 1]
        if violin_data:
            vp = ax_ts.violinplot(
                violin_data,
                positions=violin_pos,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for pc in vp["bodies"]:
                pc.set_facecolor("#5987aa")
                pc.set_edgecolor("black")
                pc.set_linewidth(0.4)
                pc.set_alpha(0.55)
                pc.set_zorder(3)

        # Weighted mean: solid red horizontal line
        for i, gwl_grp in enumerate(gwl_order):
            sub = df_reg[df_reg["GWL"] == gwl_grp]["pdd"].dropna().values
            w   = df_reg[df_reg["GWL"] == gwl_grp]["weight"].dropna().values
            if sub.size > 0:
                try:
                    wm = np.average(sub, weights=w)
                except Exception:
                    wm = np.nanmean(sub)
                ax_ts.plot(
                    [i + 1 - 0.18, i + 1 + 0.18], [wm, wm],
                    color="#c0392b", linewidth=1.2, solid_capstyle="round",
                    zorder=5,
                )

        ax_ts.set_xticks(range(1, len(gwl_order) + 1))
        ax_ts.set_xticklabels(gwl_display, fontsize=4, rotation=30, ha="right")
        ax_ts.tick_params(axis="y", labelsize=4)
        ax_ts.set_ylim(y_min, y_max)
        for spine in ax_ts.spines.values():
            spine.set_linewidth(0.4)
        ax_ts.grid(True, linestyle="--", alpha=0.4)
        ax_ts.annotate(
            f"$\\mathbf{{{panellabels[ridx]}}}$",
            xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=6,
        )
        ax_ts.set_title(reg['name'], fontsize=6)
        if ridx == 0:
            ax_ts.set_ylabel("Annual severity", fontsize=5)

    plt.tight_layout()
    return fig


def plot_supp_valuebyalpha_stacked(
    gwl_items,
    shapefile_path,
    da_mask_ref,
    no_wind_mask,
    hatchings=None,
    agreement_threshold=15.0,
    change_edges=None,
    sev_edges=None,
    n_bins_change=5,
    n_bins_sev=5,
    color_levels=None,
    alpha_levels=None,
    relchange_label="Relative change (%)",
    sev_label="Average annual\nseverity (0.61 �C)",
):
    """
    Supplementary figure: value-by-alpha maps for multiple GWL levels stacked
    vertically (one row per GWL), without violin plots.

    Parameters
    ----------
    gwl_items : list of dict with keys 'rgba_map', 'extent', 'gwl_label'
    da_mask_ref : 2-D DataArray (one realization) used for the no-wind contour
    no_wind_mask : 2-D boolean array (True = no wind potential)
    """
    if change_edges is None:
        change_edges = [-100, -25, -10, 10, 25, 100]
    if color_levels is None:
        color_levels = cm.get_cmap("coolwarm")(np.linspace(0, 1, n_bins_change))
    if alpha_levels is None:
        alpha_levels = np.linspace(0.4, 1.0, n_bins_sev)
    if sev_edges is None:
        sev_edges = np.linspace(0, 1.0, n_bins_sev + 1) ** 2

    shp = gpd.read_file(shapefile_path)
    n   = len(gwl_items)

    # LaTeX-compatible width; height scales with number of rows
    fig_width_in  = FIG_WIDTH_IN
    fig_height_in = fig_width_in * (6 / 14) * n   # keep per-row aspect ratio

    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=300)
    gs  = GridSpec(n, 1, hspace=0.06, figure=fig)

    legend_rgba = np.zeros((n_bins_change, n_bins_sev, 4))
    for ic in range(n_bins_change):
        legend_rgba[ic, :, :3] = color_levels[ic, :3]
        legend_rgba[ic, :,  3] = alpha_levels

    for i, item in enumerate(gwl_items):
        ax        = fig.add_subplot(gs[i, 0], projection=ccrs.Robinson())
        rgba_map  = item["rgba_map"]
        extent    = item["extent"]
        gwl_label = item["gwl_label"]

        ax.imshow(
            rgba_map,
            extent=extent,
            origin="lower",
            transform=ccrs.PlateCarree(),
            interpolation="nearest",
            rasterized=True,
        )
        ax.contourf(
            da_mask_ref.lon, da_mask_ref.lat, no_wind_mask.astype(float),
            levels=[0.5, 1], colors=["#404040"],
            transform=ccrs.PlateCarree(), zorder=5,
        )
        shp.boundary.plot(ax=ax, color="black", linewidth=0.15,
                          transform=ccrs.PlateCarree(), zorder=10)

        if hatchings is not None:
            ax.contourf(
                hatchings.lon, hatchings.lat,
                (hatchings <= agreement_threshold).values.astype(float),
                transform=ccrs.PlateCarree(),
                colors="none", levels=[0.5, 1.5],
                hatches=[21 * "/", 21 * "/"], zorder=8,
            )

        panel_letter = ascii_lowercase[i]
        panel_gwl    = gwl_label.replace(".0�C", "�C")
        ax.annotate(
            f"$\\mathbf{{{panel_letter}}}$",
            xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=8,
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
            zorder=1000,
        )
        ax.set_title(panel_gwl, fontsize=8)
        ax.set_extent([-180, 180, -58, 68], crs=ccrs.PlateCarree())
        ax.spines["geo"].set_visible(False)

        # Inset legend
        legend_ax = inset_axes(ax, width="14%", height="50%", loc="center left", borderpad=0.5)
        legend_ax.imshow(legend_rgba, origin="lower", aspect="equal")
        legend_ax.set_xticks([0, n_bins_sev // 2, n_bins_sev - 1])
        legend_ax.set_xticklabels(["low", "mid", "high"], fontsize=5, ha="center")
        legend_ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        legend_ax.set_yticklabels(["-25%", "-10%", "10%", "25%"], fontsize=5, va="center")
        legend_ax.set_xlabel(sev_label, fontsize=5, labelpad=4)
        legend_ax.set_ylabel(relchange_label, fontsize=5, labelpad=4)
        legend_ax.tick_params(axis="both", which="both", length=0)

    return fig


# =============================================================================
# Global change statistics with spatial block-bootstrap CI
# =============================================================================

def _compute_ensemble_rel_change(
    da_ref_freq, da_ref_sev, da_ref_dur,
    da_proj_freq, da_proj_sev, da_proj_dur,
    weight, mask, lat_min=-60, lat_max=68,
):
    """
    Return the ensemble-weighted mean relative change (%) as a 2-D numpy array
    on the cropped W5E5 grid.  Used to freeze colour-bin membership at GWL1.5
    and apply the same spatial masks to higher GWLs.
    """
    def _crop(da):
        return da.where(da.lat > lat_min, drop=True).where(da.lat < lat_max, drop=True)
    base = (da_ref_freq  * da_ref_sev  * da_ref_dur ).weighted(weight).mean(dim="realization")
    proj = (da_proj_freq * da_proj_sev * da_proj_dur).weighted(weight).mean(dim="realization")
    base = _crop(base).where(mask == 1)
    proj = _crop(proj).where(mask == 1)
    rc = 100.0 * (proj - base) / base
    return rc.where(np.isfinite(rc)).values   # numpy (nlat, nlon)

def _compute_rgba_map(
    da_ref_freq, da_ref_sev, da_ref_dur,
    da_proj_freq, da_proj_sev, da_proj_dur,
    weight, mask, lat_min=-60, lat_max=68,
    n_bins_change=5, n_bins_sev=5,
):
    """Compute RGBA map array for value-by-alpha visualization (lightweight)."""
    def _crop(da):
        return da.where(da.lat > lat_min, drop=True).where(da.lat < lat_max, drop=True)

    base_comp = _crop(da_ref_freq  * da_ref_sev  * da_ref_dur ).where(mask == 1)
    proj_comp = _crop(da_proj_freq * da_proj_sev * da_proj_dur).where(mask == 1)
    base_mean  = base_comp.weighted(weight).mean(dim="realization")
    proj_mean  = proj_comp.weighted(weight).mean(dim="realization")
    rel_change = 100.0 * (proj_mean - base_mean) / base_mean
    rel_change = rel_change.where(np.isfinite(rel_change))
    sev     = base_mean
    dChange = rel_change

    change_edges = [-100, -25, -10, 10, 25, 100]
    change_bin   = np.digitize(dChange.values, change_edges[1:-1])
    base_cmap    = cm.get_cmap("coolwarm")
    color_levels = base_cmap(np.linspace(0, 1, n_bins_change))

    max_sev   = 1.0
    sev_edges = np.linspace(0, max_sev, n_bins_sev + 1) ** 2 / max_sev
    sev_bin   = np.digitize(sev.values, sev_edges[1:-1])
    alpha_min, alpha_max = 0.4, 1.0
    alpha_levels = np.linspace(alpha_min, alpha_max, n_bins_sev)

    nlat, nlon    = dChange.shape
    valid_px      = np.isfinite(dChange.values) & np.isfinite(sev.values)
    rgba_map      = np.zeros((nlat, nlon, 4), dtype=float)
    cb = np.clip(change_bin, 0, n_bins_change - 1)
    sb = np.clip(sev_bin,    0, n_bins_sev    - 1)
    rgba_map[valid_px, :3] = color_levels[cb[valid_px], :3]
    rgba_map[valid_px,  3] = alpha_levels[sb[valid_px]]

    extent = [
        float(sev.lon.min()), float(sev.lon.max()),
        float(sev.lat.min()), float(sev.lat.max()),
    ]
    return rgba_map, extent, change_edges, sev_edges, color_levels, alpha_levels


def compute_global_change_stats_gwl(
    da_ref_freq, da_ref_sev, da_ref_dur,
    da_proj_freq, da_proj_sev, da_proj_dur,
    weight,
    mask,
    lat_min=-60,
    lat_max=68,
    n_bootstrap=1000,
    block_size=10,
):
    """
    Compute the global area-weighted mean relative change in the compound index
    and its 95 % spatial block-bootstrap confidence interval.

    Parameters
    ----------
    da_ref_*  / da_proj_* : xr.DataArray
        Frequency, severity, duration for baseline and projection (realization, lat, lon).
    weight : xr.DataArray
        Per-realization GCM weights (dim realization).
    mask : array-like
        Land mask (1 = valid).
    lat_min / lat_max : float
        Latitude crop before computing global mean.
    n_bootstrap : int
        Number of block-bootstrap iterations.
    block_size : float
        Block size in degrees for the spatial bootstrap (respects spatial autocorrelation).

    Returns
    -------
    global_rel_change : float   global weighted-mean relative change (%)
    ci_lower_rel      : float   2.5th percentile of bootstrap distribution (%)
    ci_upper_rel      : float   97.5th percentile of bootstrap distribution (%)
    """
    def _crop(da):
        return da.where(da.lat > lat_min, drop=True).where(da.lat < lat_max, drop=True)

    base_comp = (da_ref_freq  * da_ref_sev  * da_ref_dur ).weighted(weight).mean(dim="realization")
    proj_comp = (da_proj_freq * da_proj_sev * da_proj_dur).weighted(weight).mean(dim="realization")

    base_comp = _crop(base_comp).where(mask == 1)
    proj_comp = _crop(proj_comp).where(mask == 1)
    abs_change = proj_comp - base_comp

    lat_weights = np.cos(np.deg2rad(base_comp.lat))
    lat_weights.name = "weights"

    global_early = float(base_comp.weighted(lat_weights).mean(dim=["lat", "lon"]).values)
    global_late  = float(proj_comp.weighted(lat_weights).mean(dim=["lat", "lon"]).values)
    global_rel_change = 100.0 * (global_late - global_early) / global_early

    # Spatial block-bootstrap on the 2-D absolute-change field
    data         = abs_change.values
    lats         = abs_change.lat.values
    lons         = abs_change.lon.values
    weight_1d    = np.cos(np.deg2rad(lats))   # (nlat,) area weights

    lat_blocks = np.arange(lats.min(), lats.max(), block_size)
    lon_blocks = np.arange(lons.min(), lons.max(), block_size)

    rng = np.random.default_rng()
    bootstrap_means = []
    for _ in range(n_bootstrap):
        s_lat = rng.choice(lat_blocks, size=len(lat_blocks), replace=True)
        s_lon = rng.choice(lon_blocks, size=len(lon_blocks), replace=True)

        b_vals, b_wts = [], []
        for lb in s_lat:
            for lob in s_lon:
                li_arr = np.where((lats >= lb) & (lats < lb + block_size))[0]
                lo_arr = np.where((lons >= lob) & (lons < lob + block_size))[0]
                if li_arr.size and lo_arr.size:
                    for li in li_arr:
                        for loi in lo_arr:
                            v = data[li, loi]
                            if not np.isnan(v):
                                b_vals.append(v)
                                b_wts.append(weight_1d[li])

        if b_vals:
            b_vals = np.array(b_vals)
            b_wts  = np.array(b_wts)
            bootstrap_means.append(np.sum(b_vals * b_wts) / np.sum(b_wts))

    bootstrap_means = np.array(bootstrap_means)
    ci_lower_rel = 100.0 * np.percentile(bootstrap_means, 2.5)  / global_early
    ci_upper_rel = 100.0 * np.percentile(bootstrap_means, 97.5) / global_early

    # GCM bootstrap: sample GCMs with replacement; for multi-run GCMs draw one run
    gcm_vals    = da_proj_freq.GCM.values
    unique_gcms = np.unique(gcm_vals)
    gcm_to_idx  = {g: np.where(gcm_vals == g)[0] for g in unique_gcms}
    n_gcms      = len(unique_gcms)

    base_per_real = _crop(da_ref_freq  * da_ref_sev  * da_ref_dur ).where(mask == 1)
    proj_per_real = _crop(da_proj_freq * da_proj_sev * da_proj_dur).where(mask == 1)
    base_np  = base_per_real.values                # (n_real, nlat, nlon)
    proj_np  = proj_per_real.values
    lats_gcm = base_per_real.lat.values
    w2d_gcm  = np.outer(np.cos(np.deg2rad(lats_gcm)), np.ones(base_np.shape[2]))

    rng_gcm = np.random.default_rng()
    gcm_boot_means = []
    for _ in range(n_bootstrap):
        sel_gcms = rng_gcm.choice(unique_gcms, size=n_gcms, replace=True)
        idx      = np.array([rng_gcm.choice(gcm_to_idx[g]) for g in sel_gcms])
        b_base   = np.nanmean(base_np[idx], axis=0)
        b_proj   = np.nanmean(proj_np[idx], axis=0)
        early_b  = np.nansum(b_base * w2d_gcm) / np.nansum(np.where(np.isfinite(b_base), w2d_gcm, 0.0))
        late_b   = np.nansum(b_proj * w2d_gcm) / np.nansum(np.where(np.isfinite(b_proj), w2d_gcm, 0.0))
        gcm_boot_means.append(100.0 * (late_b - early_b) / early_b)

    gcm_boot_means   = np.array(gcm_boot_means)
    gcm_ci_lower_rel = np.percentile(gcm_boot_means, 2.5)
    gcm_ci_upper_rel = np.percentile(gcm_boot_means, 97.5)

    return global_rel_change, ci_lower_rel, ci_upper_rel, gcm_ci_lower_rel, gcm_ci_upper_rel


def compute_bin_change_stats_gwl(
    da_ref_freq, da_ref_sev, da_ref_dur,
    da_proj_freq, da_proj_sev, da_proj_dur,
    weight,
    mask,
    lat_min=-60,
    lat_max=68,
    change_edges=None,
    n_bootstrap=1000,
    block_size=10,
    reference_data=None,
):
    """
    For each discrete colour bin of the value-by-alpha map, compute the
    area-weighted mean relative change in compound index and its 95 %
    spatial block-bootstrap CI.

    Bins follow change_edges = [-100, -25, -10, 10, 25, 100] by default,
    matching the 5 colour categories (dark blue ? light blue ? gray ? orange ? red).

    Parameters
    ----------
    reference_data : 2-D numpy array or None
        If provided (e.g., GWL1.5 rel_change from _compute_ensemble_rel_change),
        bin *membership* is determined from this reference field while the
        bootstrap *values* are taken from the current GWL's rel_change.
        If None, both membership and values come from the current GWL.

    Returns
    -------
    list of dict with keys: 'bin', 'mean', 'ci_lower', 'ci_upper', 'n_pixels'
    """
    if change_edges is None:
        change_edges = [-100, -25, -10, 10, 25, 100]

    bin_labels = [
        f"< {change_edges[1]:.0f}%  (dark blue)",
        f"[{change_edges[1]:.0f}%, {change_edges[2]:.0f}%]  (light blue)",
        f"[{change_edges[2]:.0f}%, {change_edges[3]:.0f}%]  (gray)",
        f"[{change_edges[3]:.0f}%, {change_edges[4]:.0f}%]  (orange)",
        f"> {change_edges[4]:.0f}%  (red)",
    ]

    def _crop(da):
        return da.where(da.lat > lat_min, drop=True).where(da.lat < lat_max, drop=True)

    base_comp = (da_ref_freq  * da_ref_sev  * da_ref_dur ).weighted(weight).mean(dim="realization")
    proj_comp = (da_proj_freq * da_proj_sev * da_proj_dur).weighted(weight).mean(dim="realization")
    base_comp = _crop(base_comp).where(mask == 1)
    proj_comp = _crop(proj_comp).where(mask == 1)

    rel_change = 100.0 * (proj_comp - base_comp) / base_comp
    rel_change = rel_change.where(np.isfinite(rel_change))

    lats      = rel_change.lat.values
    lons      = rel_change.lon.values
    data      = rel_change.values          # values to average (current GWL)
    # Bin membership source: GWL1.5 field if provided, otherwise current GWL
    bin_src   = reference_data if reference_data is not None else data
    weight_1d = np.cos(np.deg2rad(lats))
    w2d       = np.outer(weight_1d, np.ones(len(lons)))

    lat_blocks = np.arange(lats.min(), lats.max(), block_size)
    lon_blocks = np.arange(lons.min(), lons.max(), block_size)
    rng = np.random.default_rng()

    # Precompute per-realization compound arrays for the GCM bootstrap
    gcm_vals    = da_proj_freq.GCM.values
    unique_gcms = np.unique(gcm_vals)
    gcm_to_idx  = {g: np.where(gcm_vals == g)[0] for g in unique_gcms}
    n_gcms      = len(unique_gcms)
    base_per_real = _crop(da_ref_freq  * da_ref_sev  * da_ref_dur ).where(mask == 1)
    proj_per_real = _crop(da_proj_freq * da_proj_sev * da_proj_dur).where(mask == 1)
    base_np   = base_per_real.values   # (n_real, nlat, nlon)
    proj_np   = proj_per_real.values
    rng_gcm   = np.random.default_rng()

    results = []
    for k in range(len(change_edges) - 1):
        lo, hi = change_edges[k], change_edges[k + 1]
        # Pixels valid in BOTH the reference bin definition AND the current GWL
        ref_valid = np.isfinite(bin_src)
        if k == 0:
            bin_pix = np.isfinite(data) & ref_valid & (bin_src < hi)
        elif k == len(change_edges) - 2:
            bin_pix = np.isfinite(data) & ref_valid & (bin_src >= lo)
        else:
            bin_pix = np.isfinite(data) & ref_valid & (bin_src >= lo) & (bin_src < hi)

        n_pixels = int(np.sum(bin_pix))
        if n_pixels == 0:
            results.append({
                "bin": bin_labels[k], "mean": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan,
                "gcm_ci_lower": np.nan, "gcm_ci_upper": np.nan,
                "n_pixels": 0,
            })
            continue

        # Area-weighted mean within bin
        mean_val = (
            np.nansum(np.where(bin_pix, data * w2d, np.nan))
            / np.sum(np.where(bin_pix, w2d, 0.0))
        )

        # Spatial block-bootstrap restricted to bin pixels
        bootstrap_means = []
        for _ in range(n_bootstrap):
            s_lat = rng.choice(lat_blocks, size=len(lat_blocks), replace=True)
            s_lon = rng.choice(lon_blocks, size=len(lon_blocks), replace=True)
            b_vals, b_wts = [], []
            for lb in s_lat:
                for lob in s_lon:
                    li_arr = np.where((lats >= lb) & (lats < lb + block_size))[0]
                    lo_arr = np.where((lons >= lob) & (lons < lob + block_size))[0]
                    if li_arr.size and lo_arr.size:
                        for li in li_arr:
                            for loi in lo_arr:
                                if bin_pix[li, loi]:
                                    b_vals.append(data[li, loi])
                                    b_wts.append(weight_1d[li])
            if b_vals:
                b_arr = np.array(b_vals)
                w_arr = np.array(b_wts)
                bootstrap_means.append(np.sum(b_arr * w_arr) / np.sum(w_arr))

        if bootstrap_means:
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
        else:
            ci_lower = ci_upper = np.nan

        # GCM bootstrap for this bin: resample GCMs w/ replacement, pick one run each
        gcm_bin_boot = []
        for _ in range(n_bootstrap):
            sel_gcms = rng_gcm.choice(unique_gcms, size=n_gcms, replace=True)
            idx      = np.array([rng_gcm.choice(gcm_to_idx[g]) for g in sel_gcms])
            b_base   = np.nanmean(base_np[idx], axis=0)   # (nlat, nlon)
            b_proj   = np.nanmean(proj_np[idx], axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rc_b = 100.0 * (b_proj - b_base) / b_base
            pix_vals = rc_b[bin_pix]
            pix_wts  = w2d[bin_pix]
            finite   = np.isfinite(pix_vals)
            if finite.any():
                gcm_bin_boot.append(
                    np.sum(pix_vals[finite] * pix_wts[finite]) / np.sum(pix_wts[finite])
                )
        gcm_ci_lower = np.percentile(gcm_bin_boot, 2.5)  if gcm_bin_boot else np.nan
        gcm_ci_upper = np.percentile(gcm_bin_boot, 97.5) if gcm_bin_boot else np.nan

        results.append({
            "bin": bin_labels[k],
            "mean": mean_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "gcm_ci_lower": gcm_ci_lower,
            "gcm_ci_upper": gcm_ci_upper,
            "n_pixels": n_pixels,
        })

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    level_to_key = {"1.5": "GWL1-5", "2.0": "GWL2", "3.0": "GWL3"}
    level_to_label = {"1.5": "1.5�C", "2.0": "2.0�C", "3.0": "3.0�C"}
    level_to_fname = {
        "1.5": "fig2_projected_change_valuebyalpha_GWL1-5.png",
        "2.0": "fig2_projected_change_valuebyalpha_GWL2.png",
        "3.0": "fig2_projected_change_valuebyalpha_GWL3.png",
    }

    for lv in args.gwl_levels:
        if lv not in level_to_key:
            raise ValueError(f"Unknown GWL level '{lv}'. Valid: {list(level_to_key.keys())}.")

    # ------------------------------------------------------------------
    # STEP 0 � Build missing aggregated gridded datasets
    # ------------------------------------------------------------------
    # We always build GWL0-61 (baseline) plus every requested projection level.
    gwl_keys_to_build = ["GWL0-61"] + [level_to_key[lv] for lv in args.gwl_levels]
    print("=" * 60)
    print("STEP 0 � Building aggregated gridded datasets (if needed)")
    print("=" * 60)
    build_gridded_datasets(
        preprocessed_path=args.preprocessed_path,
        gwl_list=gwl_keys_to_build,
        exclude_gcm=args.exclude_gcm,
        exclude_gcm_run=args.exclude_gcm_run,
        force_rebuild=args.force_rebuild,
    )

    # ------------------------------------------------------------------
    # STEP 1 � Load baseline dataset and build land mask
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1 � Loading baseline dataset and building land mask")
    print("=" * 60)
    ds_baseline = load_baseline_dataset(args.path_gwl, args.exclude_gcm, args.exclude_gcm_run)
    ref_2d = _reduce_to_2d(ds_baseline.duration)
    mask   = build_land_mask(ref_2d, args.shapefile)

    # Historical W5E5 null mask (True = no wind potential) for the dark grey layer
    hist_null_da = None
    if args.path_hist is not None:
        if os.path.exists(args.path_hist):
            print(f"  Loading historical null mask from {args.path_hist} �")
            _ds_hist = xr.open_dataset(args.path_hist)
            _dur_hist = _ds_hist.duration
            if "year" in _dur_hist.dims:
                _dur_hist = _dur_hist.isel(year=0)
            hist_null_da = _dur_hist.isnull().load()
            _ds_hist.close()
        else:
            print(f"  [warn] --path_hist not found: {args.path_hist}. Dark grey layer uses GCM baseline.")

    # ------------------------------------------------------------------
    # STEP 2 � Regional DataFrame
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 � Regional DataFrame")
    print("=" * 60)
    # Default save/load path � can be overridden with --regional_csv
    default_csv = os.path.join(args.output_dir, "regional_data_projections.csv")
    regional_csv_path = args.regional_csv if args.regional_csv is not None else default_csv

    if os.path.exists(regional_csv_path):
        print(f"Found existing regional DataFrame at {regional_csv_path} � loading it.")
        df_regions = pd.read_csv(regional_csv_path)
    else:
        print(f"No regional DataFrame found at {regional_csv_path} � computing �")
        df_regions = create_dataframe_regional(
            path_gwl=args.path_gwl,
            mask=mask,
            exclude_gcm=args.exclude_gcm,
            exclude_gcm_run=args.exclude_gcm_run,
        )
        os.makedirs(os.path.dirname(regional_csv_path), exist_ok=True)
        df_regions.to_csv(regional_csv_path, index=False)
        print(f"  Saved ? {regional_csv_path}")

    df_regions = add_pdd_and_weights(df_regions)

    # ------------------------------------------------------------------
    # STEP 3 � Optional agreement hatching
    # ------------------------------------------------------------------
    hatchings = None
    if args.agreement_path is not None and os.path.exists(args.agreement_path):
        print(f"\nLoading agreement mask from {args.agreement_path} �")
        hatchings = xr.open_dataarray(args.agreement_path)
    elif args.agreement_path is not None:
        print(f"  [warn] Agreement file not found: {args.agreement_path}. Hatching disabled.")

    # ------------------------------------------------------------------
    # STEP 4 � Loop over GWL levels and produce figures
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 � Producing figures")
    print("=" * 60)
    # GWL1.5 rel_change field used as fixed colour-bin mask for all GWL levels
    gwl15_rel_change_data = None
    # Data collected for the supplementary stacked figure
    supp_items          = []
    supp_meta           = {}   # change_edges, sev_edges, color_levels, alpha_levels
    da_mask_ref_supp    = None
    no_wind_mask_supp   = None

    for level in args.gwl_levels:
        gwl_key   = level_to_key[level]
        gwl_label = level_to_label[level]
        fname     = level_to_fname[level]

        print(f"\nProcessing {gwl_label} �")
        ds_proj = load_gwl_dataset(args.path_gwl, gwl_key, args.exclude_gcm, args.exclude_gcm_run)

        print(f"  Aligning GCMs �")
        (da_ref_freq, da_ref_sev, da_ref_dur,
         da_proj_freq, da_proj_sev, da_proj_dur,
         weight) = from_ds_to_plot_decomp(ds_proj, ds_baseline)

        # Freeze colour-bin membership at GWL1.5 so higher GWLs report stats
        # for the same spatial zones that were red/orange/gray/etc. at 1.5�C.
        if level == "1.5":
            gwl15_rel_change_data = _compute_ensemble_rel_change(
                da_ref_freq, da_ref_sev, da_ref_dur,
                da_proj_freq, da_proj_sev, da_proj_dur,
                weight=weight, mask=mask, lat_min=-60, lat_max=68,
            )

        print(f"  Computing global change statistics �")
        global_chg, ci_lo, ci_hi, gcm_ci_lo, gcm_ci_hi = compute_global_change_stats_gwl(
            da_ref_freq, da_ref_sev, da_ref_dur,
            da_proj_freq, da_proj_sev, da_proj_dur,
            weight=weight, mask=mask,
            lat_min=-60, lat_max=68,
        )
        print(
            f"  Global mean change under {gwl_label}: {global_chg:+.2f}% "
            f"[spatial CI: {ci_lo:+.2f}%, {ci_hi:+.2f}%] "
            f"[GCM CI: {gcm_ci_lo:+.2f}%, {gcm_ci_hi:+.2f}%]"
        )
        print(f"  Computing per-bin change statistics �")
        bin_stats = compute_bin_change_stats_gwl(
            da_ref_freq, da_ref_sev, da_ref_dur,
            da_proj_freq, da_proj_sev, da_proj_dur,
            weight=weight, mask=mask,
            lat_min=-60, lat_max=68,
            reference_data=gwl15_rel_change_data,
        )
        print(f"  Per-bin statistics under {gwl_label}:")
        for s in bin_stats:
            print(
                f"    {s['bin']:45s}  mean={s['mean']:+7.2f}%  "
                f"[spatial CI: {s['ci_lower']:+7.2f}%, {s['ci_upper']:+7.2f}%]  "
                f"[GCM CI: {s['gcm_ci_lower']:+7.2f}%, {s['gcm_ci_upper']:+7.2f}%]  "
                f"({s['n_pixels']} pixels)"
            )
        print(f"  Plotting �")
        fig = plot_gwl_valuebyalpha_discrete(
            da_ref_freq=da_ref_freq, da_ref_sev=da_ref_sev, da_ref_dur=da_ref_dur,
            da_proj_freq=da_proj_freq, da_proj_sev=da_proj_sev, da_proj_dur=da_proj_dur,
            weight=weight, mask=mask,
            shapefile_path=args.shapefile,
            df_regions=df_regions,
            gwl_label=gwl_label,
            hatchings=hatchings,
            agreement_threshold=args.agreement_threshold,
            map_title=f"Annual severity change under {gwl_label} warming",
            relchange_label="Relative change (%)",
            sev_label="Average annual\nseverity (0.61 �C)",
            lat_min=-60, lat_max=68,
            hist_null_da=hist_null_da,
        )
        out_path = os.path.join(args.output_dir, "main", fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved ? {out_path}")

        # ------ Collect data for the supplementary stacked figure ------
        print(f"  Collecting rgba map for supplementary figure �")
        _rgba, _extent, _cedges, _sedges, _clvl, _alvl = _compute_rgba_map(
            da_ref_freq, da_ref_sev, da_ref_dur,
            da_proj_freq, da_proj_sev, da_proj_dur,
            weight=weight, mask=mask, lat_min=-60, lat_max=68,
        )
        supp_items.append({"rgba_map": _rgba, "extent": _extent, "gwl_label": gwl_label})
        if not supp_meta:
            supp_meta = {
                "change_edges": _cedges, "sev_edges": _sedges,
                "color_levels": _clvl,   "alpha_levels": _alvl,
            }
        # Build the no-wind mask once (same grid for all GWLs)
        if da_mask_ref_supp is None:
            _shp_tmp   = gpd.read_file(args.shapefile)
            da_mask_ref_supp = da_ref_freq.isel(realization=0).load()
            _t = rasterio.transform.from_bounds(
                da_mask_ref_supp.lon.min().item(), da_mask_ref_supp.lat.min().item(),
                da_mask_ref_supp.lon.max().item(), da_mask_ref_supp.lat.max().item(),
                len(da_mask_ref_supp.lon), len(da_mask_ref_supp.lat),
            )
            _land = rasterize_shapefile(_shp_tmp, da_mask_ref_supp.shape, _t)[::-1, :]
            if hist_null_da is not None:
                _null = hist_null_da.astype(float).interp(
                    lat=da_mask_ref_supp.lat, lon=da_mask_ref_supp.lon, method="nearest"
                )
                no_wind_mask_supp = _land & (_null.values > 0.5)
            else:
                _mf = mask.astype(float).interp(
                    lat=da_mask_ref_supp.lat, lon=da_mask_ref_supp.lon, method="nearest"
                )
                no_wind_mask_supp = _land & (_mf.values < 0.5)
        # ----------------------------------------------------------------

        del ds_proj, da_ref_freq, da_ref_sev, da_ref_dur
        del da_proj_freq, da_proj_sev, da_proj_dur, weight, fig
        gc.collect()

    # ------------------------------------------------------------------
    # STEP 5 � Supplementary stacked figure (all GWL, no violin plots)
    # ------------------------------------------------------------------
    if supp_items:
        print("\n" + "=" * 60)
        print("STEP 5 � Supplementary stacked value-by-alpha figure")
        print("=" * 60)
        fig_supp = plot_supp_valuebyalpha_stacked(
            gwl_items=supp_items,
            shapefile_path=args.shapefile,
            da_mask_ref=da_mask_ref_supp,
            no_wind_mask=no_wind_mask_supp,
            hatchings=hatchings,
            agreement_threshold=args.agreement_threshold,
            **supp_meta,
        )
        out_supp = os.path.join(args.output_dir, "supp", "fig_supp_valuebyalpha_all_gwl.png")
        os.makedirs(os.path.dirname(out_supp), exist_ok=True)
        fig_supp.savefig(out_supp, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_supp)
        print(f"  Saved ? {out_supp}")

    print("\nDone.")


if __name__ == "__main__":
    main()
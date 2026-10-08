# -*- coding: utf-8 -*-
"""
Figure 3: projected changes in annual SWED severity at 1.5, 2 and 3 degC of
global warming -- value-by-alpha maps plus regional violin panels -- and
the global/per-bin change statistics quoted in the text.

The ensemble average weights each realization by 1/n_gcm times the inverse
of its normalized Wasserstein trend distance to ERA5 (per pixel for maps and
statistics, per region box for the violins' mean line); --weighting mmm
restores the flat multi-model mean.

prepare_inputs()/iter_gwl_decomp() and the CLI are reused by
supp_figs/suppfig4_valuebyalpha_all_gwl.py and
supp_figs/suppfig5_projected_change_wasserstein.py.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; also puts main_pipeline/, main_figs/, supp_figs/, aux_code/ on sys.path
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XCLIM
os.environ['ESMFMKFILE'] = config.ESMFMKFILE_XCLIM

import argparse
import gc
from types import SimpleNamespace

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xesmf as xe
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.features import geometry_mask
from scipy import stats

# Zarr/NetCDF-agnostic file lookup + opener, shared with calculate_cf.py
# (prefers a .zarr store when present, falls back to .nc).
from io_utils import match_files, glob_any, open_dataset_any
# Grey (no wind capacity) / dots (obs.-projection trend discrepancy)
# exclusion layers, shared with fig1.py and fig45.py.
from map_overlays import (draw_wcf_zero_overlay, draw_discrepancy_dots,
                          add_exclusion_legend)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patheffects import withStroke
from matplotlib import cm
from itertools import groupby


# =============================================================================
# Figure size constants (LaTeX-compatible)
# =============================================================================
FIG_WIDTH_IN = 5.15   # column width - fontsizes in pt will match LaTeX

# Latitude band shown on EqualEarth maps in this module (matches the
# analysis's own poleward exclusion; see Methods: "Regions poleward of 68N
# and 58S were excluded due to artifacts in the duration metric"). Applied
# by masking data/shapefiles to this band and calling ax.set_global() --
# NOT ax.set_extent(), which miscalibrates on EqualEarth's curved meridians:
# it clips to the bounding rectangle of the extent box's own corners, whose
# right edge only touches the true 180 deg meridian at MAP_LAT_SOUTH/NORTH
# themselves, sitting well short of it at other latitudes -- slicing
# through real land (e.g. eastern Australia) even though it's nominally
# within +/-180 deg longitude.
MAP_LAT_SOUTH = -58
MAP_LAT_NORTH = 68
EQUAL_EARTH_ASPECT = 2.05   # width / height of a set_global() EqualEarth axes


def mask_poles(ax, lat_south=MAP_LAT_SOUTH, lat_north=MAP_LAT_NORTH, zorder=12):
    """
    White out everything poleward of [lat_south, lat_north] on a set_global()
    EqualEarth map. Needed because cfeature.COASTLINE/ax.coastlines() draw
    the *entire* globe's coastlines (Antarctica, remote Arctic islands)
    regardless of how the data/shapefile were masked to this band, and
    shp.cx[:, lat_south:lat_north] keeps whole country geometries (e.g.
    Russia, Canada, Greenland) rather than clipping them at the band's edge
    -- both leak real content poleward of the intended crop (see
    MAP_LAT_SOUTH/NORTH above). Draws a white cap over each pole, in
    PlateCarree and densely sampled in longitude so it follows the
    projection's own curved boundary, on top of coastlines/boundaries but
    below panel labels/region boxes (zorder 20+ elsewhere in this module).
    """
    lons = np.linspace(-180.0, 180.0, 361)
    for lat_edge, lat_pole in ((lat_south, -90.0), (lat_north, 90.0)):
        cap = Polygon(
            list(zip(lons, np.full_like(lons, lat_edge))) +
            list(zip(lons[::-1], np.full_like(lons, lat_pole)))
        )
        ax.add_geometries([cap], crs=ccrs.PlateCarree(),
                          facecolor="white", edgecolor="none", zorder=zorder)


# =============================================================================
# CLI arguments
# =============================================================================

def build_parser(description=(
        "Figure 3: projected changes in annual SWED severity (value-by-alpha maps "
        "and regional violins) at each global warming level.")):
    """CLI shared by fig3.py and supp_figs/suppfig4_*/suppfig5_*."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Preprocessed / raw data path (used for dataset building) ---
    parser.add_argument(
        "--preprocessed_path",
        default=config.PATH_PREPROCESSED,
        help=(
            "Root folder of the preprocessed daily wcf/scf files (zarr, falls "
            "back to .nc) (e.g. <GCM>/wcf_day_*.zarr). Used to build the "
            "aggregated datasets in memory on every run."
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
            "(columns: GWL, region, realization, GCM, frequency, intensity, duration). "
            "If not provided, the regional DataFrame is computed on-the-fly."
        ),
    )
    parser.add_argument(
        "--agreement_path",
        default=config.AGREEMENT_NC_PATH,
        help=(
            "Path to a pre-computed model-agreement DataArray (.nc), built by "
            "trend_sev_eval.py's build_agreement_mask(). Values are agreement_pct: the %% "
            "of GCM realizations whose bootstrap trend CI overlaps the ERA5 reference trend "
            "CI at each cell. Cells with value <= agreement_threshold are hatched on the map. "
            "If not provided, no hatching is applied."
        ),
    )
    parser.add_argument(
        "--agreement_threshold",
        type=float,
        default=config.AGREEMENT_THRESHOLD,
        help=(
            "Threshold below which cells are hatched (default: config.AGREEMENT_THRESHOLD, "
            "%.0f%%%% of models with an overlapping trend CI)." % config.AGREEMENT_THRESHOLD
        ),
    )
    parser.add_argument(
        "--wcf_zero_mask_path",
        default=config.WCF_ZERO_MASK_NC_PATH,
        help=(
            "Path to the cached wcf-zero land mask (.nc, built by fig1.py's "
            "build_wcf_zero_mask()/save_wcf_zero_mask()). Land pixels where "
            "ERA5 wcf is exactly 0 over the whole reference period; greyed "
            "out on every gridded map here, independent of the no-wind "
            "quantile mask and of any agreement hatching. If not found, "
            "this overlay is skipped."
        ),
    )

    # --- Ensemble weighting ---
    parser.add_argument(
        "--weighting",
        choices=["w2", "mmm"],
        default="w2",
        help=(
            "Ensemble averaging: 'w2' (default) weights each realization by 1/n_gcm "
            "times the inverse of its normalized Wasserstein trend distance to ERA5 "
            "at each pixel (see _build_wasserstein_pixel_weight); 'mmm' is the flat "
            "multi-model mean (1/n_gcm only)."
        ),
    )
    parser.add_argument(
        "--wasserstein_path",
        default=config.WASSERSTEIN_NC_PATH,
        help=(
            "Path to a pre-computed per-(GCM, run), per-pixel empirical Wasserstein "
            "trend-distance DataArray (.nc), built by trend_sev_eval_wasserstein.py's "
            "wasserstein_empirical_grid() (variables w2_distance/w2_normalized, dims "
            "realization/lat/lon, coords GCM/run). Required with --weighting w2."
        ),
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

    return parser


def parse_args():
    return build_parser().parse_args()


# =============================================================================
# -- DATASET BUILDING (from load_gridded_data_compound) ----------------------
# =============================================================================

def compute_intensity(comp_da, scf_ds, wcf_ds, scf_thr, wcf_thr):
    """
    Expected shortfall: mean positive deficit on compound-event days,
    aggregated yearly.
    """
    deficit_scf = xr.where(scf_ds["scf"] < scf_thr, scf_thr - scf_ds["scf"], 0)
    deficit_wcf = xr.where(wcf_ds["wcf"] < wcf_thr, wcf_thr - wcf_ds["wcf"], 0)
    daily_deficit = deficit_scf + deficit_wcf

    masked = xr.where(comp_da == 1, daily_deficit, np.nan)
    intensity = masked.resample(time="YE").mean().fillna(0)
    return intensity


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

    counts_df = (
        df.groupby(["event_id", "lat", "lon", "year"], sort=False)
        .size()
        .reset_index(name="duration")
    )

    dur_da = xr.DataArray(
        counts_df["duration"].to_numpy(),
        dims="event_instance",
        coords={
            "event_instance": np.arange(len(counts_df)),
            "event_id": ("event_instance", counts_df["event_id"].to_numpy(dtype=int)),
            "lat":      ("event_instance", counts_df["lat"].to_numpy(dtype=float)),
            "lon":      ("event_instance", counts_df["lon"].to_numpy(dtype=float)),
            "year":     ("event_instance", counts_df["year"].to_numpy(dtype=int)),
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


def _build_single_gcm(preprocessed_path, gwl, gcm, run, ssp, wcf_rea):
    """
    Build the aggregated gridded dataset for one (GCM, GWL) pair and return it
    in memory. Nothing is written to disk here -- the caller concatenates the
    per-GCM datasets directly, avoiding a save/reload round trip.
    """
    print(f"    Building {gcm} / {gwl}")

    # Load projection data (zarr preferred, falls back to .nc)
    wcf_files, _ = match_files(
        os.path.join(preprocessed_path, gcm, f"wcf_day_{gcm}_{ssp}_{run}_{gwl}_ERA5"))
    scf_files, _ = match_files(
        os.path.join(preprocessed_path, gcm, f"scf_day_{gcm}_{ssp}_{run}_{gwl}_ERA5"))
    wcf = open_dataset_any(wcf_files[0])
    scf = open_dataset_any(scf_files[0])
    wcf["time"] = pd.to_datetime(wcf.time.dt.strftime("%Y-%m-%d").values)
    scf["time"] = pd.to_datetime(scf.time.dt.strftime("%Y-%m-%d").values)

    # Load reference (GWL0-61) data for threshold computation
    wcf_ref_paths = glob_any(
        os.path.join(preprocessed_path, gcm, f"wcf_day_{gcm}*ssp*{run}_GWL0-61_ERA5"))
    scf_ref_paths = glob_any(
        os.path.join(preprocessed_path, gcm, f"scf_day_{gcm}*ssp*{run}_GWL0-61_ERA5"))
    wcf_ref = open_dataset_any(wcf_ref_paths[0])
    scf_ref = open_dataset_any(scf_ref_paths[0])
    wcf_ref["time"] = pd.to_datetime(wcf_ref.time.dt.strftime("%Y-%m-%d").values)
    scf_ref["time"] = pd.to_datetime(scf_ref.time.dt.strftime("%Y-%m-%d").values)

    # 10th-percentile thresholds (over positive values only)
    wcf_thr = wcf_ref.wcf.where(wcf_ref.wcf > 0).quantile(0.1, dim="time")
    scf_thr = scf_ref.scf.where(scf_ref.scf > 0).quantile(0.1, dim="time")

    # Compound event flag
    wcf["low_wind"]   = xr.where(wcf.wcf <= wcf_thr, 1, 0)
    scf["low_solar"]  = xr.where(scf.scf <= scf_thr, 1, 0)
    compound = (wcf.low_wind * scf.low_solar).to_dataset(name="start_cooc")

    compound = compound.convert_calendar("standard")
    wcf      = wcf.convert_calendar("standard")
    scf      = scf.convert_calendar("standard")

    # Intensity, duration, frequency
    intensity_ds    = compute_intensity(compound.start_cooc, scf, wcf, scf_thr, wcf_thr)
    ds_dur, ds_freq = duration_xr(compound.start_cooc)
    ds_dur  = ds_dur.reindex( {"lat": scf.lat, "lon": scf.lon})
    ds_freq = ds_freq.reindex({"lat": scf.lat, "lon": scf.lon})

    intensity_ds["time"] = intensity_ds.time.dt.year
    intensity_ds         = intensity_ds.rename({"time": "year"})

    severity = intensity_ds * ds_dur.duration * ds_freq.frequency

    ds_final              = ds_dur.copy()
    ds_final["frequency"] = ds_freq.frequency
    ds_final["intensity"] = intensity_ds
    ds_final["severity"]  = severity

    comp_annual           = compound.resample(time="YE").sum()
    comp_annual["time"]   = comp_annual.time.dt.year
    comp_annual           = comp_annual.rename({"time": "year"})
    ds_final["nb_days"]   = comp_annual.start_cooc

    # Regrid to ERA5 grid and attach metadata
    ds_final            = ds_final.expand_dims({"realization": [0]})
    regrid              = xe.Regridder(ds_final, wcf_rea, method="nearest_s2d")
    ds_final            = regrid(ds_final)
    ds_final["GCM"]     = xr.DataArray([gcm], dims="realization")
    ds_final["run"]     = xr.DataArray([run],  dims="realization")
    ds_final["ssp"]     = xr.DataArray([ssp],  dims="realization")
    ds_final["gwl"]     = xr.DataArray([gwl],  dims="realization")

    ds_final = ds_final.load()

    # Release memory
    del wcf, scf, wcf_ref, scf_ref, compound, intensity_ds, ds_dur, ds_freq
    gc.collect()

    return ds_final


def _print_gcm_run_summary(label, ds):
    """
    Diagnostic: print every (GCM, run) pair in ds and flag any that appear
    more than once. Duplicate (GCM, run) pairs within a single GWL's built
    dataset break from_ds_to_plot_decomp's alignment (a "common" pair would
    then be counted multiple times on one side but not the other).
    """
    gcms = ds.GCM.values
    runs = ds.run.values
    pairs = list(zip(gcms, runs))
    print(f"    [{label}] {len(pairs)} realizations:")
    for g, r in pairs:
        print(f"      {g} : {r}")

    counts = {}
    for p in pairs:
        counts[p] = counts.get(p, 0) + 1
    dupes = {p: c for p, c in counts.items() if c > 1}
    if dupes:
        print(f"    [{label}] DUPLICATE (GCM, run) pairs found:")
        for (g, r), c in dupes.items():
            print(f"      {g} : {r}  (x{c})")
    else:
        print(f"    [{label}] no duplicate (GCM, run) pairs ({len(counts)} unique).")


def build_gridded_datasets(preprocessed_path, gwl_list, exclude_gcm=None, exclude_gcm_run=None):
    """
    Build the aggregated gridded dataset for every requested GWL, entirely in
    memory (no per-GCM/GWL cache file is written to or read from disk).

    Parameters
    ----------
    preprocessed_path : str
        Root directory containing per-GCM subdirectories with daily .nc files.
    gwl_list : list of str
        GWL keys to process, e.g. ['GWL0-61', 'GWL1-5', 'GWL2', 'GWL3'].
    exclude_gcm : list of str or None
        GCMs to skip entirely.
    exclude_gcm_run : list of str or None
        "GCM:run" pairs to skip.

    Returns
    -------
    dict of {gwl_key: xr.Dataset}
        One dataset per GWL that had at least one usable GCM, concatenated
        over a fresh 'realization' dimension.
    """
    exclude_gcm     = exclude_gcm or []
    exclude_gcm_run = set(tuple(x.split(":")) for x in (exclude_gcm_run or []))

    # Load the ERA5 reference grid (needed for regridding)
    rea_files, _ = match_files(os.path.join(preprocessed_path, "ERA5", "wcf_day*"))
    if not rea_files:
        raise FileNotFoundError(
            f"No ERA5 reference file found under {preprocessed_path}/ERA5/. "
            "Cannot determine target regrid grid."
        )
    wcf_rea = open_dataset_any(rea_files[0]).isel(time=slice(0, 2))

    built = {}
    for gwl in gwl_list:
        print(f"\n  -- Building datasets for {gwl} --")

        # Discover available projection files for this GWL
        wcf_paths = glob_any(
            os.path.join(preprocessed_path, "*/wcf_day_*ssp*" + gwl + "_ERA5"))
        if not wcf_paths:
            print(f"  No files found for {gwl}, skipping.")
            continue

        gcm_list = [p.split("_")[-5] for p in wcf_paths]
        run_list = [p.split("_")[-3] for p in wcf_paths]
        ssp_list = [p.split("_")[-4] for p in wcf_paths]

        gcm_datasets = []
        for gcm, run, ssp in zip(gcm_list, run_list, ssp_list):
            if gcm in exclude_gcm or (gcm, run) in exclude_gcm_run:
                print(f"    [excluded] {gcm} {run}")
                continue
            try:
                gcm_datasets.append(_build_single_gcm(preprocessed_path, gwl, gcm, run, ssp, wcf_rea))
            except Exception as exc:
                print(f"    [ERROR] {gcm} / {gwl}: {exc}")
            gc.collect()

        if not gcm_datasets:
            print(f"  No datasets built for {gwl}.")
            continue

        gwl_ds = xr.concat(gcm_datasets, dim="realization")
        gwl_ds["realization"] = np.arange(len(gcm_datasets))
        built[gwl] = gwl_ds
        _print_gcm_run_summary(gwl, gwl_ds)
        del gcm_datasets
        gc.collect()

    return built


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
    ref_2d must have dims (lat, lon) - no time or realization.
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


def load_wcf_zero_mask(path):
    """
    Load the wcf-zero land mask built by fig1.py's build_wcf_zero_mask()/
    save_wcf_zero_mask() (land pixels, per shp_re, where ERA5 wcf is exactly
    0 across the whole reference period). This is the sole "no wind
    resource" grey layer on the gridded maps in this file -- replaces the
    quantile-based (10th-percentile wcf > 0) mask this module used to build
    itself, which crashed on newer xarray (interp() rejects bool dtype) and
    duplicated what fig1.py already computes. Independent of any
    GCM-trend-agreement hatching -- see draw_wcf_zero_overlay.
    """
    return xr.open_dataarray(path).astype(bool)


def fit_to_width(fig, width_in=FIG_WIDTH_IN, n_iter=4, tol=0.002):
    """
    Rescale the figure canvas (both dimensions, fonts untouched) so that
    savefig(..., bbox_inches="tight") yields an image exactly width_in wide.
    Without this, the tight crop leaves each figure at a different width, so
    once LaTeX scales them all to the column width their fonts (panel
    letters, titles, legends) end up at different effective point sizes.
    Accounts for savefig's own pad_inches on each side.
    """
    target = width_in - 2 * plt.rcParams["savefig.pad_inches"]
    for _ in range(n_iter):
        fig.canvas.draw()
        scale = target / fig.get_tightbbox(fig.canvas.get_renderer()).width
        if abs(scale - 1) < tol:
            break
        fig.set_size_inches(fig.get_figwidth() * scale, fig.get_figheight() * scale)
    return fig


def _reduce_to_2d(da):
    """Average out 'year' and 'realization' dims to obtain a (lat, lon) DataArray."""
    if "year" in da.dims:
        da = da.mean(dim="year")
    if "realization" in da.dims:
        da = da.mean(dim="realization")
    return da


# =============================================================================
# Align baseline and projection datasets by common GCMs
# =============================================================================

def from_ds_to_plot_decomp(ds_gwl, ds_ref):
    """
    Align baseline (ds_ref) and projection (ds_gwl) datasets by common
    (GCM, run) pairs -- not just GCM name, since a given GCM can have a
    different set of available runs at different GWLs (e.g. one run may
    not reach a higher warming level at all). Matching on name alone can
    pick a different number of realizations from each dataset whenever
    that happens, which then fails when they're recombined.
    Returns (ref_freq, ref_int, ref_dur, proj_freq, proj_int, proj_dur, weight).
    """
    pairs_gwl = list(zip(ds_gwl.GCM.values, ds_gwl.run.values))
    pairs_ref = list(zip(ds_ref.GCM.values, ds_ref.run.values))

    # Find indices *separately* in each dataset, then sort both by the same
    # (GCM, run) key so that paired realizations correspond to the same
    # GCM/run rather than just the same GCM.
    common_pairs = set(pairs_ref) & set(pairs_gwl)
    ref_indices = [i for i, p in enumerate(pairs_ref) if p in common_pairs]
    gwl_indices = [i for i, p in enumerate(pairs_gwl) if p in common_pairs]

    if len(ref_indices) != len(gwl_indices):
        def _dupes(pairs, indices):
            counts = {}
            for i in indices:
                counts[pairs[i]] = counts.get(pairs[i], 0) + 1
            return {p: c for p, c in counts.items() if c > 1}
        raise ValueError(
            f"GCM/run alignment mismatch: {len(ref_indices)} baseline realizations "
            f"vs {len(gwl_indices)} projection realizations share a common (GCM, run) pair. "
            f"This means one side lists the same (GCM, run) more than once. "
            f"Duplicate pairs on baseline side: {_dupes(pairs_ref, ref_indices)}. "
            f"Duplicate pairs on projection side: {_dupes(pairs_gwl, gwl_indices)}."
        )

    ds_ref = ds_ref.isel(realization=ref_indices)
    ds_gwl = ds_gwl.isel(realization=gwl_indices)

    # Sort both by (GCM, run) so realizations are paired consistently
    ref_order = sorted(range(ds_ref.sizes["realization"]),
                        key=lambda i: (ds_ref.GCM.values[i], ds_ref.run.values[i]))
    gwl_order = sorted(range(ds_gwl.sizes["realization"]),
                        key=lambda i: (ds_gwl.GCM.values[i], ds_gwl.run.values[i]))
    ds_ref = ds_ref.isel(realization=ref_order)
    ds_gwl = ds_gwl.isel(realization=gwl_order)

    if "year" in ds_gwl.dims:
        ds_gwl = ds_gwl.mean(dim="year")
    if "year" in ds_ref.dims:
        ds_ref = ds_ref.mean(dim="year")

    ds_gwl["realization"] = ds_ref.realization.astype(int)

    # Promote GCM/run (data variables) to coordinates so they are carried by
    # every DataArray extracted from this dataset (GCM is needed by the GCM
    # bootstrap; run is needed to match realizations against the Wasserstein
    # distance file's own (GCM, run) coords in _build_wasserstein_pixel_weight).
    ds_gwl = ds_gwl.assign_coords(
        GCM=("realization", ds_gwl.GCM.values),
        run=("realization", ds_gwl.run.values),
    )

    weight_count = pd.Series(ds_gwl.GCM.values).value_counts()
    weights = [1.0 / weight_count[g] / weight_count.size for g in ds_gwl.GCM.values]
    weight = xr.DataArray(weights, dims="realization")

    return (
        ds_ref.frequency, ds_ref.intensity, ds_ref.duration,
        ds_gwl.frequency, ds_gwl.intensity, ds_gwl.duration,
        weight,
    )


# =============================================================================
# Inverse-Wasserstein-distance ensemble weighting
# =============================================================================

W2_VAR = "w2_normalized"
W2_EPS = 1e-3
W2_SIGMA_REF_MIN = 1e-6   # same as trend_sev_eval_wasserstein.SIGMA_REF_MIN


def _w2_pair_index(da_proj_freq, ds_wasserstein):
    """Positional (proj_idx, w2_idx) pairs matching da_proj_freq's realizations
    to ds_wasserstein's by (GCM, run), plus the unmatched (GCM, run) pairs."""
    w2_index = {(str(g), str(r)): i for i, (g, r) in
                enumerate(zip(ds_wasserstein.GCM.values, ds_wasserstein.run.values))}
    proj_idx, w2_idx, missing = [], [], []
    for i, (g, r) in enumerate(zip(da_proj_freq.GCM.values, da_proj_freq.run.values)):
        p = (str(g), str(r))
        if p in w2_index:
            proj_idx.append(i)
            w2_idx.append(w2_index[p])
        else:
            missing.append(p)
    return proj_idx, w2_idx, missing


def _build_wasserstein_pixel_weight(da_proj_freq, ds_wasserstein, base_weight,
                                     var=W2_VAR, eps=W2_EPS):
    """
    Combine the usual per-realization 1/n_gcm weight with a per-pixel weight
    equal to the inverse of that realization's normalized empirical
    Wasserstein trend distance to ERA5 at that pixel (ds_wasserstein's
    w2_normalized, from trend_sev_eval_wasserstein.wasserstein_empirical_grid()):
    a realization whose bootstrap trend distribution is closer to ERA5's at a
    given location counts for more there, on top of (not instead of) the
    existing 1/n_gcm de-duplication across multi-run GCMs.

    Matching is by (GCM, run) pair, since ds_wasserstein's realization axis
    (built from GWL1 files only) does not generally line up positionally with
    da_proj_freq's own realization axis (built per-GWL in from_ds_to_plot_decomp).
    Realizations in da_proj_freq with no Wasserstein match get weight 0
    (excluded) everywhere; eps floors w2_normalized so a near-zero distance
    cannot make a single realization dominate the pixel mean.

    Returns
    -------
    (weight_pix, inv_w2): two xr.DataArrays with dims (realization, lat, lon),
    positionally aligned with da_proj_freq's realization axis (no
    'realization' coordinate, matching base_weight's own convention).
    weight_pix = base_weight * inv_w2 is the ensemble-mean weight; inv_w2
    alone is what the GCM bootstraps use, since resampling GCMs already
    provides the 1/n_gcm part. (None, None) if no (GCM, run) pair matches.
    """
    proj_idx, w2_idx, missing = _w2_pair_index(da_proj_freq, ds_wasserstein)
    if missing:
        print(f"    [warn] no Wasserstein match for {len(missing)} realization(s), "
              f"excluded from the weighted mean: {missing}")
    if not proj_idx:
        return None, None

    w2_sel = ds_wasserstein[var].isel(realization=w2_idx)
    w2_sel = w2_sel.interp(lat=da_proj_freq.lat, lon=da_proj_freq.lon, method="nearest")
    inv_matched = np.nan_to_num(1.0 / w2_sel.clip(min=eps).values, nan=0.0)

    shape = (da_proj_freq.sizes["realization"],) + inv_matched.shape[1:]
    inv_full = np.zeros(shape, dtype=float)
    inv_full[proj_idx] = inv_matched
    weight_full = inv_full * np.asarray(base_weight.values)[:, None, None]

    def _da(arr):
        return xr.DataArray(arr, dims=("realization", "lat", "lon"),
                            coords={"lat": da_proj_freq.lat, "lon": da_proj_freq.lon})
    return _da(weight_full), _da(inv_full)


def load_wasserstein(args):
    """Open args.wasserstein_path when args.weighting == 'w2' (None for 'mmm').
    Missing file is a hard error: the figures never silently fall back to
    the multi-model mean."""
    if args.weighting != "w2":
        return None
    if not os.path.exists(args.wasserstein_path):
        raise SystemExit(f"Wasserstein file not found: {args.wasserstein_path} -- run "
                         "main_pipeline/trend_sev_eval_wasserstein.py first, or pass "
                         "--weighting mmm.")
    print(f"Loading Wasserstein distance dataset from {args.wasserstein_path} ...")
    return drop_near_zero_reference_std(xr.open_dataset(args.wasserstein_path))


def drop_near_zero_reference_std(ds_w2, sigma_ref_min=W2_SIGMA_REF_MIN):
    """NaN out w2_normalized wherever ERA5's own bootstrap-trend std (recovered
    as w2_distance / w2_normalized) is below sigma_ref_min. Files written before
    trend_sev_eval_wasserstein.py's SIGMA_REF_MIN fix divided by a ~1e-12 floor
    there instead, giving w2_normalized ~1e8-1e9 that swamp every region mean:
    the 1-2 realizations without such pixels then take ~all of the inverse-W2
    weight. A no-op on files built after the fix (those pixels are already NaN)."""
    wd, wn = ds_w2.w2_distance, ds_w2.w2_normalized
    keep = (wd == 0) | ((wd / wn) > sigma_ref_min)
    n_total = int(np.isfinite(wn.values).sum())
    n_dropped = int((np.isfinite(wn.values) & ~keep.values).sum())
    if n_dropped:
        print(f"  Dropping {n_dropped}/{n_total} pixel-realizations ({n_dropped / n_total:.1%}) "
              f"with a near-zero ERA5 bootstrap-trend std (< {sigma_ref_min:g}).")
    return ds_w2.assign(w2_normalized=wn.where(keep))


def ensemble_weight(da_proj_freq, base_weight, inputs):
    """
    Ensemble weight for one GWL's aligned fields: (weight_pix, inv_w2) from
    _build_wasserstein_pixel_weight under inverse-W2 weighting, or
    (base_weight, None) for the flat multi-model mean (inputs.ds_wasserstein
    is None).
    """
    if inputs.ds_wasserstein is None:
        return base_weight, None
    weight_pix, inv_w2 = _build_wasserstein_pixel_weight(
        da_proj_freq, inputs.ds_wasserstein, base_weight)
    if weight_pix is None:
        raise ValueError("No (GCM, run) overlap between the projection ensemble "
                         "and the Wasserstein dataset.")
    return weight_pix, inv_w2


def region_mean_w2(ds_wasserstein, region_mask, var=W2_VAR):
    """
    cos(latitude)-weighted mean of w2_normalized over the True pixels of
    region_mask (a boolean (lat, lon) DataArray), per realization of
    ds_wasserstein. Returns a pd.Series indexed by (GCM, run) -- the
    region-level distance whose inverse (times 1/n_gcm) weights that
    realization's region-mean value.
    """
    region_mask = region_mask.reset_coords(drop=True)   # keep only lat/lon
    w2 = ds_wasserstein[var].interp(lat=region_mask.lat, lon=region_mask.lon,
                                    method="nearest")
    w2 = w2.where(region_mask)
    lat_w = np.cos(np.deg2rad(w2.lat))
    mean = w2.weighted(lat_w.fillna(0)).mean(dim=("lat", "lon"), skipna=True).values
    idx = pd.MultiIndex.from_arrays(
        [np.asarray(ds_wasserstein.GCM.values).astype(str),
         np.asarray(ds_wasserstein.run.values).astype(str)], names=["GCM", "run"])
    return pd.Series(mean, index=idx, name="w2_mean")


def _weighted_nanmean(x, w):
    """Mean of x over axis 0, weighted by w (same shape), ignoring NaNs in x."""
    w = np.where(np.isfinite(x), w, 0.0)
    wsum = w.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(wsum > 0, np.nansum(x * w, axis=0) / wsum, np.nan)


# =============================================================================
# Regional DataFrame
# =============================================================================

def _robust_slice(da, lat_lo, lat_hi, lon_lo, lon_hi):
    lat_slice = (slice(lat_lo, lat_hi) if da.lat[0] < da.lat[-1]
                 else slice(lat_hi, lat_lo))
    lon_slice = (slice(lon_lo, lon_hi) if da.lon[0] < da.lon[-1]
                 else slice(lon_hi, lon_lo))
    return da.sel(lat=lat_slice, lon=lon_slice)


REGIONS_DF = [
    {"name": "Western U.S.",    "lat": [35,  50],  "lon": [-125, -105]},
    {"name": "Southern Europe", "lat": [35,  50],  "lon": [5,   25]},
    {"name": "South Africa",    "lat": [-35, -22], "lon": [16,     33]},
    {"name": "Kenya",           "lat": [-5,   5],  "lon": [33,     42]},
    {"name": "India",           "lat": [10,  30],  "lon": [70,     90]},
]


def create_dataframe_regional(built_datasets, mask, regions=None):
    """
    For each already-built (in-memory) GWL dataset and each region, extract
    the spatial mean of frequency, intensity and duration per realization.
    """
    if regions is None:
        regions = REGIONS_DF

    rows = []
    for gwl_label, base_ds in built_datasets.items():
        print(f"  Aggregating regional stats for {gwl_label}")

        # Subsetting first gives us an independent Dataset, so masking below
        # does not mutate the shared dataset the main plotting loop reuses.
        ds = base_ds[["frequency", "intensity", "duration", "GCM", "run"]]
        ds["frequency"] = ds["frequency"].where(mask == 1)
        ds["intensity"] = ds["intensity"].where(mask == 1)
        ds["duration"]  = ds["duration"].where(mask == 1)
        if "year" in ds.dims:
            ds = ds.mean(dim="year")

        gcms = ds.GCM.values
        runs = ds.run.values

        for reg in regions:
            lat_lo, lat_hi = reg["lat"]
            lon_lo, lon_hi = reg["lon"]

            freq_sub  = _robust_slice(ds.frequency, lat_lo, lat_hi, lon_lo, lon_hi)
            int_sub   = _robust_slice(ds.intensity, lat_lo, lat_hi, lon_lo, lon_hi)
            dur_sub   = _robust_slice(ds.duration,  lat_lo, lat_hi, lon_lo, lon_hi)

            freq_mean = freq_sub.mean(dim=("lat", "lon"), skipna=True).values
            int_mean  = int_sub.mean( dim=("lat", "lon"), skipna=True).values
            dur_mean  = dur_sub.mean( dim=("lat", "lon"), skipna=True).values

            for ridx in range(len(gcms)):
                rows.append({
                    "GWL":         gwl_label,
                    "region":      reg["name"],
                    "realization": ridx,
                    "GCM":         gcms[ridx],
                    "run":         runs[ridx],
                    "frequency":   float(freq_mean[ridx]),
                    "intensity":   float(int_mean[ridx]),
                    "duration":    float(dur_mean[ridx]),
                })

    return pd.DataFrame(rows)


def region_w2_table(ds_wasserstein, mask, regions=None):
    """
    Long (region, GCM, run, w2_mean) table: each region box's
    cos(latitude)-weighted mean normalized Wasserstein distance over its
    land pixels (region_mean_w2), per realization of ds_wasserstein.
    """
    if regions is None:
        regions = REGIONS_DF
    land = mask.astype(bool)
    rows = []
    for reg in regions:
        lat_lo, lat_hi = reg["lat"]
        lon_lo, lon_hi = reg["lon"]
        in_box = ((land.lat >= lat_lo) & (land.lat <= lat_hi) &
                  (land.lon >= lon_lo) & (land.lon <= lon_hi))
        s = region_mean_w2(ds_wasserstein, land & in_box).reset_index()
        s["region"] = reg["name"]
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def add_severity_and_weights(df, region_w2=None, eps=W2_EPS):
    """
    Add 'severity' column (frequency x intensity x duration) and the
    per-realization 'weight' column: the inverse-frequency GCM weight
    (1/n_gcm, split across a GCM's runs), times -- when region_w2 (from
    region_w2_table) is given -- the inverse of that realization's mean
    normalized Wasserstein distance over the region. Realizations with no
    Wasserstein match get a NaN weight (excluded from the weighted mean).
    """
    df = df.copy()
    df["severity"] = df["frequency"] * df["intensity"] * df["duration"]

    anchor_region = df["region"].iloc[0]
    anchor  = df[(df["region"] == anchor_region) & (df["GWL"] == "GWL0-61")]
    wcount  = anchor["GCM"].value_counts()
    weight_dict = {gcm: 1.0 / wcount[gcm] / wcount.size for gcm in wcount.index}
    df["weight"] = df["GCM"].map(weight_dict)

    if region_w2 is not None:
        df["GCM"] = df["GCM"].astype(str)
        df["run"] = df["run"].astype(str)
        df = df.merge(region_w2, on=["region", "GCM", "run"], how="left")
        df["weight"] = df["weight"] / df["w2_mean"].clip(lower=eps)

    return df


# =============================================================================
# Main figure - value-by-alpha map + regional boxplots
# =============================================================================

def plot_gwl_valuebyalpha_discrete(
    da_ref_freq, da_ref_int, da_ref_dur,
    da_proj_freq, da_proj_int, da_proj_dur,
    weight,
    mask,
    shapefile_path,
    df_regions,
    gwl_label,
    hatchings=None,
    agreement_threshold=config.AGREEMENT_THRESHOLD,
    map_title=None,
    relchange_label="Relative change (%)",
    sev_label="Average annual\nseverity (0.61 °C)",
    lat_min=-60,
    lat_max=68,
    regions=None,
    n_bins_change=5,
    n_bins_sev=5,
    wcf_zero_mask=None,
):
    if regions is None:
        regions = [
            {"name": "Western U.S.",    "lat": [35,  50],  "lon": [-125, -105]},
            {"name": "South Africa",    "lat": [-35, -22], "lon": [16,     33]},
            {"name": "Kenya",           "lat": [-5,   5],  "lon": [33,     42]},
            {"name": "India",           "lat": [10,  30],  "lon": [70,     90]},
        ]

    # --- 1. Compound index ---
    base_comp = da_ref_freq  * da_ref_int  * da_ref_dur
    proj_comp = da_proj_freq * da_proj_int * da_proj_dur

    def _crop_lat(da):
        return da.where(da.lat > lat_min, drop=True).where(da.lat < lat_max, drop=True)

    base_comp = _crop_lat(base_comp).where(mask == 1)
    proj_comp = _crop_lat(proj_comp).where(mask == 1)

    # --- 2. Weighted ensemble mean ---
    base_mean  = base_comp.weighted(weight).mean(dim="realization")
    proj_mean  = proj_comp.weighted(weight).mean(dim="realization")
    rel_change = 100.0 * (proj_mean - base_mean) / base_mean
    rel_change = rel_change.where(np.isfinite(rel_change))
    severity   = base_mean
    dChange    = rel_change

    # --- 3. Discrete colour bins ---
    change_edges = [-100, -25, -10, 10, 25, 100]
    change_bin   = np.digitize(dChange.values, change_edges[1:-1])

    base_cmap    = cm.get_cmap("coolwarm")
    color_levels = base_cmap(np.linspace(0, 1, n_bins_change))

    # --- 4. Discrete alpha bins ---
    max_sev   = 1.0
    sev_edges = np.linspace(0, max_sev, n_bins_sev + 1) ** 2 / max_sev
    sev_bin   = np.digitize(severity.values, sev_edges[1:-1])

    alpha_min, alpha_max = 0.4, 1.0
    alpha_levels = np.linspace(alpha_min, alpha_max, n_bins_sev)

    # --- 5. RGBA assembly ---
    nlat, nlon = dChange.shape
    valid_mask = np.isfinite(dChange.values) & np.isfinite(severity.values)
    # RGB defaults to white (not black) for invalid/masked (e.g. ocean) pixels --
    # alpha stays 0 so they're still fully transparent, but this avoids a solid
    # black sea in viewers that don't alpha-composite the PNG correctly.
    rgba_map   = np.ones((nlat, nlon, 4), dtype=float)
    rgba_map[..., 3] = 0.0
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
    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.EqualEarth())

    # --- 7. Draw map ---
    shp = gpd.read_file(shapefile_path)
    shp_band = shp.cx[:, MAP_LAT_SOUTH:MAP_LAT_NORTH]
    ax_map.imshow(
        rgba_map,
        extent=[severity.lon.min().item(), severity.lon.max().item(),
                severity.lat.min().item(), severity.lat.max().item()],
        origin="lower",
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        rasterized=True,
    )

    da_mask = da_ref_freq.isel(realization=0).sel(
        lat=slice(MAP_LAT_SOUTH, MAP_LAT_NORTH))
    t_mask  = rasterio.transform.from_bounds(
        da_mask.lon.min().item(), da_mask.lat.min().item(),
        da_mask.lon.max().item(), da_mask.lat.max().item(),
        len(da_mask.lon), len(da_mask.lat),
    )
    land_shp   = rasterize_shapefile(shp_band, da_mask.shape, t_mask)
    land_shp   = land_shp[::-1, :]
    shp_band.boundary.plot(ax=ax_map, color="black", linewidth=0.15,
                           transform=ccrs.PlateCarree(), zorder=10)

    if hatchings is not None:
        # Dot hatching: land pixels that failed the trend-agreement evaluation
        # (agreement_pct <= agreement_threshold).
        _agree_float = hatchings.interp(
            lat=da_mask.lat, lon=da_mask.lon, method="nearest"
        )
        failed_eval_mask = land_shp & (_agree_float.values <= agreement_threshold)
        draw_discrepancy_dots(ax_map, failed_eval_mask, da_mask.lat, da_mask.lon)

    grey_drawn = draw_wcf_zero_overlay(ax_map, wcf_zero_mask, land_shp, da_mask.lat, da_mask.lon,
                                       nan_data=da_mask.isnull().values)
    add_exclusion_legend(ax_map, show_discrepancy=hatchings is not None,
                         show_wcf_zero=grey_drawn,
                         wcf_zero_label="Excluded: no wind capacity")

    if map_title is None:
        map_title = f"Projected change in annual severity under {gwl_label} warming"
    elif "{gwl_label}" in map_title:
        map_title = map_title.format(gwl_label=gwl_label)
    ax_map.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.15)
    ax_map.annotate(
        "$\\mathbf{a}$",
        xy=(0.02, 1.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=8,
        path_effects=[withStroke(linewidth=1.5, foreground="white")],
    )
    ax_map.set_title(map_title, fontsize=7, pad=6)
    ax_map.set_global()
    mask_poles(ax_map)
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
    gwl_display = ["0.61°C",  "1.5°C",  "2.0°C", "3.0°C"]

    # Shared y-axis limits across all regions and GWL levels
    y_min, y_max = float("inf"), float("-inf")
    for reg in regions:
        df_reg = df_regions[df_regions["region"] == reg["name"]]
        for gwl_grp in gwl_order:
            vals = df_reg[df_reg["GWL"] == gwl_grp]["severity"].dropna().values
            if vals.size > 0:
                y_min = min(y_min, np.nanmin(vals))
                y_max = max(y_max, np.nanmax(vals))
    y_min *= 0.95
    y_max *= 1.05

    for ridx, reg in enumerate(regions):
        ax_ts  = fig.add_subplot(gs[1, ridx])
        df_reg = df_regions[df_regions["region"] == reg["name"]]

        data_box = [
            df_reg[df_reg["GWL"] == gwl_grp]["severity"].dropna().values
            for gwl_grp in gwl_order
        ]

        # Scatter dots: light blue, very small, behind the violin
        for i, gwl_grp in enumerate(gwl_order):
            sub = df_reg[df_reg["GWL"] == gwl_grp]["severity"].dropna().values
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
            grp = df_reg[df_reg["GWL"] == gwl_grp][["severity", "weight"]].dropna()
            grp = grp[grp["weight"] > 0]
            if len(grp) > 0:
                wm = np.average(grp["severity"].values, weights=grp["weight"].values)
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
            ha="left", va="bottom", fontsize=8,
        )
        ax_ts.set_title(reg['name'], fontsize=6)
        if ridx == 0:
            ax_ts.set_ylabel("Annual severity", fontsize=5)

    plt.tight_layout()
    return fig


# =============================================================================
# Global change statistics with spatial block-bootstrap CI
# =============================================================================

def _compute_ensemble_rel_change(
    da_ref_freq, da_ref_int, da_ref_dur,
    da_proj_freq, da_proj_int, da_proj_dur,
    weight, mask, lat_min=-60, lat_max=68,
):
    """
    Return the ensemble-weighted mean relative change (%) as a 2-D numpy array
    on the cropped ERA5 grid.  Used to freeze colour-bin membership at GWL1.5
    and apply the same spatial masks to higher GWLs.
    """
    def _crop(da):
        return da.where(da.lat > lat_min, drop=True).where(da.lat < lat_max, drop=True)
    base = (da_ref_freq  * da_ref_int  * da_ref_dur ).weighted(weight).mean(dim="realization")
    proj = (da_proj_freq * da_proj_int * da_proj_dur).weighted(weight).mean(dim="realization")
    base = _crop(base).where(mask == 1)
    proj = _crop(proj).where(mask == 1)
    rc = 100.0 * (proj - base) / base
    return rc.where(np.isfinite(rc)).values   # numpy (nlat, nlon)

def _compute_rgba_map(
    da_ref_freq, da_ref_int, da_ref_dur,
    da_proj_freq, da_proj_int, da_proj_dur,
    weight, mask, lat_min=-60, lat_max=68,
    n_bins_change=5, n_bins_sev=5,
):
    """Compute RGBA map array for value-by-alpha visualization (lightweight)."""
    def _crop(da):
        return da.where(da.lat > lat_min, drop=True).where(da.lat < lat_max, drop=True)

    base_comp = _crop(da_ref_freq  * da_ref_int  * da_ref_dur ).where(mask == 1)
    proj_comp = _crop(da_proj_freq * da_proj_int * da_proj_dur).where(mask == 1)
    base_mean  = base_comp.weighted(weight).mean(dim="realization")
    proj_mean  = proj_comp.weighted(weight).mean(dim="realization")
    rel_change = 100.0 * (proj_mean - base_mean) / base_mean
    rel_change = rel_change.where(np.isfinite(rel_change))
    severity = base_mean
    dChange  = rel_change

    change_edges = [-100, -25, -10, 10, 25, 100]
    change_bin   = np.digitize(dChange.values, change_edges[1:-1])
    base_cmap    = cm.get_cmap("coolwarm")
    color_levels = base_cmap(np.linspace(0, 1, n_bins_change))

    max_sev   = 1.0
    sev_edges = np.linspace(0, max_sev, n_bins_sev + 1) ** 2 / max_sev
    sev_bin   = np.digitize(severity.values, sev_edges[1:-1])
    alpha_min, alpha_max = 0.4, 1.0
    alpha_levels = np.linspace(alpha_min, alpha_max, n_bins_sev)

    nlat, nlon    = dChange.shape
    valid_px      = np.isfinite(dChange.values) & np.isfinite(severity.values)
    # RGB defaults to white (not black) for invalid/masked (e.g. ocean) pixels --
    # see plot_gwl_valuebyalpha_discrete's rgba_map for the same fix/rationale.
    rgba_map      = np.ones((nlat, nlon, 4), dtype=float)
    rgba_map[..., 3] = 0.0
    cb = np.clip(change_bin, 0, n_bins_change - 1)
    sb = np.clip(sev_bin,    0, n_bins_sev    - 1)
    rgba_map[valid_px, :3] = color_levels[cb[valid_px], :3]
    rgba_map[valid_px,  3] = alpha_levels[sb[valid_px]]

    extent = [
        float(severity.lon.min()), float(severity.lon.max()),
        float(severity.lat.min()), float(severity.lat.max()),
    ]
    return rgba_map, extent, change_edges, sev_edges, color_levels, alpha_levels


def _block_index_lists(lats, lons, block_size):
    """
    Precompute, once, the pixel row/col indices belonging to each spatial
    block along lat and lon. Reused across every bootstrap draw instead of
    being recomputed inside the resampling loop.
    """
    lat_edges = np.arange(lats.min(), lats.max(), block_size)
    lon_edges = np.arange(lons.min(), lons.max(), block_size)
    row_idx_by_block = [
        np.where((lats >= lb) & (lats < lb + block_size))[0] for lb in lat_edges
    ]
    col_idx_by_block = [
        np.where((lons >= lob) & (lons < lob + block_size))[0] for lob in lon_edges
    ]
    return row_idx_by_block, col_idx_by_block


def _draw_block_resample(rng, row_idx_by_block, col_idx_by_block):
    """
    One block-bootstrap draw: resample lat-blocks and lon-blocks with
    replacement and return the resulting pixel row/col indices (repeated
    when a block is drawn more than once), matching the cartesian product
    that the equivalent nested-loop formulation would visit.
    """
    n_lat_b = len(row_idx_by_block)
    n_lon_b = len(col_idx_by_block)
    sel_lat = rng.integers(0, n_lat_b, size=n_lat_b) if n_lat_b else np.array([], dtype=int)
    sel_lon = rng.integers(0, n_lon_b, size=n_lon_b) if n_lon_b else np.array([], dtype=int)
    rows = np.concatenate([row_idx_by_block[b] for b in sel_lat]) if n_lat_b else np.array([], dtype=int)
    cols = np.concatenate([col_idx_by_block[b] for b in sel_lon]) if n_lon_b else np.array([], dtype=int)
    return rows, cols


def compute_global_change_stats_gwl(
    da_ref_freq, da_ref_int, da_ref_dur,
    da_proj_freq, da_proj_int, da_proj_dur,
    weight,
    mask,
    lat_min=-60,
    lat_max=68,
    n_bootstrap=1000,
    block_size=10,
    inv_w2=None,
):
    """
    Compute the global area-weighted mean relative change in the compound index
    and its 95 % spatial block-bootstrap confidence interval.

    Parameters
    ----------
    da_ref_*  / da_proj_* : xr.DataArray
        Frequency, intensity, duration for baseline and projection (realization, lat, lon).
    weight : xr.DataArray
        Ensemble weights: per-realization GCM weights (dim realization), or
        per-pixel inverse-W2 weights (realization, lat, lon) from
        ensemble_weight().
    inv_w2 : xr.DataArray or None
        Per-pixel inverse-W2 weight alone (realization, lat, lon), used to
        weight the realizations drawn in each GCM-bootstrap sample (GCM
        resampling already supplies the 1/n_gcm part). None: plain mean.
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

    base_comp = (da_ref_freq  * da_ref_int  * da_ref_dur ).weighted(weight).mean(dim="realization")
    proj_comp = (da_proj_freq * da_proj_int * da_proj_dur).weighted(weight).mean(dim="realization")

    base_comp = _crop(base_comp).where(mask == 1)
    proj_comp = _crop(proj_comp).where(mask == 1)
    abs_change = proj_comp - base_comp

    lat_weights = np.cos(np.deg2rad(base_comp.lat))
    lat_weights.name = "weights"

    global_early = float(base_comp.weighted(lat_weights).mean(dim=["lat", "lon"]).values)
    global_late  = float(proj_comp.weighted(lat_weights).mean(dim=["lat", "lon"]).values)
    global_rel_change = 100.0 * (global_late - global_early) / global_early

    # Spatial block-bootstrap on the 2-D absolute-change field (vectorized:
    # block membership is precomputed once, then each draw is a single
    # numpy fancy-indexing extraction instead of a per-pixel Python loop).
    data         = abs_change.values
    lats         = abs_change.lat.values
    lons         = abs_change.lon.values
    weight_1d    = np.cos(np.deg2rad(lats))   # (nlat,) area weights

    row_idx_by_block, col_idx_by_block = _block_index_lists(lats, lons, block_size)

    rng = np.random.default_rng()
    bootstrap_means = np.full(n_bootstrap, np.nan)
    for it in range(n_bootstrap):
        rows, cols = _draw_block_resample(rng, row_idx_by_block, col_idx_by_block)
        if rows.size == 0 or cols.size == 0:
            continue
        sub_data = data[np.ix_(rows, cols)]
        sub_w    = weight_1d[rows][:, None]
        valid    = np.isfinite(sub_data)
        wsum = np.sum(np.where(valid, sub_w, 0.0))
        if wsum > 0:
            bootstrap_means[it] = np.sum(np.where(valid, sub_data * sub_w, 0.0)) / wsum

    bootstrap_means = bootstrap_means[np.isfinite(bootstrap_means)]
    ci_lower_rel = 100.0 * np.percentile(bootstrap_means, 2.5)  / global_early
    ci_upper_rel = 100.0 * np.percentile(bootstrap_means, 97.5) / global_early

    # GCM bootstrap: sample GCMs with replacement; for multi-run GCMs draw one run
    gcm_vals    = da_proj_freq.GCM.values
    unique_gcms = np.unique(gcm_vals)
    gcm_to_idx  = {g: np.where(gcm_vals == g)[0] for g in unique_gcms}
    n_gcms      = len(unique_gcms)

    base_per_real = _crop(da_ref_freq  * da_ref_int  * da_ref_dur ).where(mask == 1)
    proj_per_real = _crop(da_proj_freq * da_proj_int * da_proj_dur).where(mask == 1)
    base_np  = base_per_real.values                # (n_real, nlat, nlon)
    proj_np  = proj_per_real.values
    inv_np   = (_crop(inv_w2).values if inv_w2 is not None
                else np.ones_like(base_np))
    lats_gcm = base_per_real.lat.values
    w2d_gcm  = np.outer(np.cos(np.deg2rad(lats_gcm)), np.ones(base_np.shape[2]))

    rng_gcm = np.random.default_rng()
    gcm_boot_means = []
    for _ in range(n_bootstrap):
        sel_gcms = rng_gcm.choice(unique_gcms, size=n_gcms, replace=True)
        idx      = np.array([rng_gcm.choice(gcm_to_idx[g]) for g in sel_gcms])
        b_base   = _weighted_nanmean(base_np[idx], inv_np[idx])
        b_proj   = _weighted_nanmean(proj_np[idx], inv_np[idx])
        early_b  = np.nansum(b_base * w2d_gcm) / np.nansum(np.where(np.isfinite(b_base), w2d_gcm, 0.0))
        late_b   = np.nansum(b_proj * w2d_gcm) / np.nansum(np.where(np.isfinite(b_proj), w2d_gcm, 0.0))
        gcm_boot_means.append(100.0 * (late_b - early_b) / early_b)

    gcm_boot_means   = np.array(gcm_boot_means)
    gcm_ci_lower_rel = np.percentile(gcm_boot_means, 2.5)
    gcm_ci_upper_rel = np.percentile(gcm_boot_means, 97.5)

    return global_rel_change, ci_lower_rel, ci_upper_rel, gcm_ci_lower_rel, gcm_ci_upper_rel


def compute_bin_change_stats_gwl(
    da_ref_freq, da_ref_int, da_ref_dur,
    da_proj_freq, da_proj_int, da_proj_dur,
    weight,
    mask,
    lat_min=-60,
    lat_max=68,
    change_edges=None,
    n_bootstrap=1000,
    block_size=10,
    reference_data=None,
    inv_w2=None,
):
    """
    For each discrete colour bin of the value-by-alpha map, compute the
    area-weighted mean relative change in compound index and its 95 %
    spatial block-bootstrap CI.

    Bins follow change_edges = [-100, -25, -10, 10, 25, 100] by default,
    matching the 5 colour categories (dark blue ? light blue ? gray ? orange ? red).

    Parameters
    ----------
    inv_w2 : xr.DataArray or None
        See compute_global_change_stats_gwl.
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

    base_comp = (da_ref_freq  * da_ref_int  * da_ref_dur ).weighted(weight).mean(dim="realization")
    proj_comp = (da_proj_freq * da_proj_int * da_proj_dur).weighted(weight).mean(dim="realization")
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

    rng = np.random.default_rng()

    # Precompute per-realization compound arrays for the GCM bootstrap
    gcm_vals    = da_proj_freq.GCM.values
    unique_gcms = np.unique(gcm_vals)
    gcm_to_idx  = {g: np.where(gcm_vals == g)[0] for g in unique_gcms}
    n_gcms      = len(unique_gcms)
    base_per_real = _crop(da_ref_freq  * da_ref_int  * da_ref_dur ).where(mask == 1)
    proj_per_real = _crop(da_proj_freq * da_proj_int * da_proj_dur).where(mask == 1)
    base_np   = base_per_real.values   # (n_real, nlat, nlon)
    proj_np   = proj_per_real.values
    inv_np    = (_crop(inv_w2).values if inv_w2 is not None
                 else np.ones_like(base_np))
    rng_gcm   = np.random.default_rng()

    n_bins = len(change_edges) - 1
    ref_valid = np.isfinite(bin_src)
    bin_masks = []
    for k in range(n_bins):
        lo, hi = change_edges[k], change_edges[k + 1]
        if k == 0:
            bin_pix = np.isfinite(data) & ref_valid & (bin_src < hi)
        elif k == n_bins - 1:
            bin_pix = np.isfinite(data) & ref_valid & (bin_src >= lo)
        else:
            bin_pix = np.isfinite(data) & ref_valid & (bin_src >= lo) & (bin_src < hi)
        bin_masks.append(bin_pix)
    n_pixels_list = [int(np.sum(bp)) for bp in bin_masks]

    # Spatial block-bootstrap (vectorized): draw once per iteration and
    # compute every bin's weighted mean from that same resampled field,
    # instead of resampling independently -- and via nested pixel loops --
    # for each bin.
    row_idx_by_block, col_idx_by_block = _block_index_lists(lats, lons, block_size)
    boot_means = np.full((n_bootstrap, n_bins), np.nan)
    for it in range(n_bootstrap):
        rows, cols = _draw_block_resample(rng, row_idx_by_block, col_idx_by_block)
        if rows.size == 0 or cols.size == 0:
            continue
        sub_data = data[np.ix_(rows, cols)]
        sub_w    = weight_1d[rows][:, None]
        for k, bin_pix in enumerate(bin_masks):
            if n_pixels_list[k] == 0:
                continue
            sub_bin = bin_pix[np.ix_(rows, cols)]
            valid   = sub_bin & np.isfinite(sub_data)
            wsum = np.sum(np.where(valid, sub_w, 0.0))
            if wsum > 0:
                boot_means[it, k] = np.sum(np.where(valid, sub_data * sub_w, 0.0)) / wsum

    results = []
    for k in range(n_bins):
        bin_pix  = bin_masks[k]
        n_pixels = n_pixels_list[k]
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

        bin_boot_means = boot_means[:, k]
        bin_boot_means = bin_boot_means[np.isfinite(bin_boot_means)]
        if bin_boot_means.size:
            ci_lower = np.percentile(bin_boot_means, 2.5)
            ci_upper = np.percentile(bin_boot_means, 97.5)
        else:
            ci_lower = ci_upper = np.nan

        # GCM bootstrap for this bin: resample GCMs w/ replacement, pick one run each
        gcm_bin_boot = []
        for _ in range(n_bootstrap):
            sel_gcms = rng_gcm.choice(unique_gcms, size=n_gcms, replace=True)
            idx      = np.array([rng_gcm.choice(gcm_to_idx[g]) for g in sel_gcms])
            b_base   = _weighted_nanmean(base_np[idx], inv_np[idx])   # (nlat, nlon)
            b_proj   = _weighted_nanmean(proj_np[idx], inv_np[idx])
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

LEVEL_TO_KEY   = {"1.5": "GWL1-5", "2.0": "GWL2", "3.0": "GWL3"}
LEVEL_TO_LABEL = {"1.5": "1.5°C", "2.0": "2.0°C", "3.0": "3.0°C"}


def prepare_inputs(args, with_regions=True):
    """Inputs shared by Fig. 3 and Extended Data Figs. 4-5: the in-memory
    ensemble datasets (GWL0-61 + args.gwl_levels), the land mask, the
    wcf-zero mask, the agreement hatching and (for Fig. 3 only) the regional
    DataFrame behind the violin panels."""
    os.makedirs(args.output_dir, exist_ok=True)
    for lv in args.gwl_levels:
        if lv not in LEVEL_TO_KEY:
            raise ValueError(f"Unknown GWL level '{lv}'. Valid: {list(LEVEL_TO_KEY.keys())}.")
    # Checked first, so a missing Wasserstein file fails before the
    # (long) dataset build rather than after it.
    ds_wasserstein = load_wasserstein(args)

    # ------------------------------------------------------------------
    # STEP 0 - Build all aggregated gridded datasets, in memory
    # ------------------------------------------------------------------
    # We always build GWL0-61 (baseline) plus every requested projection level.
    # Nothing is written to/read from disk here; the resulting datasets are
    # reused directly below for the regional dataframe and the figures.
    gwl_keys_to_build = ["GWL0-61"] + [LEVEL_TO_KEY[lv] for lv in args.gwl_levels]
    print("=" * 60)
    print("STEP 0 - Building aggregated gridded datasets (in memory)")
    print("=" * 60)
    built = build_gridded_datasets(
        preprocessed_path=args.preprocessed_path,
        gwl_list=gwl_keys_to_build,
        exclude_gcm=args.exclude_gcm,
        exclude_gcm_run=args.exclude_gcm_run,
    )
    if "GWL0-61" not in built:
        raise FileNotFoundError("No baseline (GWL0-61) data could be built.")

    # ------------------------------------------------------------------
    # STEP 1 - Build land mask from the baseline dataset
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1 - Building land mask")
    print("=" * 60)
    ds_baseline = built["GWL0-61"]
    ref_2d = _reduce_to_2d(ds_baseline.duration)
    mask   = build_land_mask(ref_2d, args.shapefile)

    # wcf-zero land mask (land pixels where ERA5 wcf is exactly 0 over the
    # whole reference period), built and cached by fig1.py. This is the
    # sole "no wind resource" grey layer on the maps below -- see
    # draw_wcf_zero_overlay. Independent of any agreement hatching.
    wcf_zero_mask = None
    if args.wcf_zero_mask_path is not None and os.path.exists(args.wcf_zero_mask_path):
        print(f"  Loading wcf-zero land mask from {args.wcf_zero_mask_path} ...")
        wcf_zero_mask = load_wcf_zero_mask(args.wcf_zero_mask_path)
    elif args.wcf_zero_mask_path is not None:
        print(f"  [warn] wcf-zero mask file not found: {args.wcf_zero_mask_path}. "
              f"Run fig1.py first to build it. Skipping this overlay.")

    # ------------------------------------------------------------------
    # STEP 2 - Regional DataFrame
    # ------------------------------------------------------------------
    df_regions = None
    if with_regions:
        print("\n" + "=" * 60)
        print("STEP 2 - Regional DataFrame")
        print("=" * 60)
        # Default save/load path - can be overridden with --regional_csv
        default_csv = os.path.join(args.output_dir, "regional_data_projections.csv")
        regional_csv_path = args.regional_csv if args.regional_csv is not None else default_csv

        if os.path.exists(regional_csv_path):
            print(f"Found existing regional DataFrame at {regional_csv_path}, loading it.")
            df_regions = pd.read_csv(regional_csv_path)
            if "run" not in df_regions.columns:
                # Older cache without the run column needed to match
                # realizations to the Wasserstein file -- rebuild it.
                print("  Cached regional DataFrame has no 'run' column, recomputing.")
                df_regions = None
        if df_regions is None:
            print(f"No regional DataFrame found at {regional_csv_path}, computing...")
            df_regions = create_dataframe_regional(built, mask)
            os.makedirs(os.path.dirname(regional_csv_path), exist_ok=True)
            df_regions.to_csv(regional_csv_path, index=False)
            print(f"  Saved -> {regional_csv_path}")

        region_w2 = (region_w2_table(ds_wasserstein, mask)
                     if ds_wasserstein is not None else None)
        df_regions = add_severity_and_weights(df_regions, region_w2=region_w2)

    # ------------------------------------------------------------------
    # STEP 3 - Optional agreement hatching
    # ------------------------------------------------------------------
    hatchings = None
    if not config.SHOW_AGREEMENT_HATCHING:
        print("\nAgreement hatching disabled (config.SHOW_AGREEMENT_HATCHING=False).")
    elif args.agreement_path is not None and os.path.exists(args.agreement_path):
        print(f"\nLoading agreement mask from {args.agreement_path} ...")
        hatchings = xr.open_dataarray(args.agreement_path)
    elif args.agreement_path is not None:
        print(f"  [warn] Agreement file not found: {args.agreement_path}. Hatching disabled.")

    return SimpleNamespace(built=built, ds_baseline=ds_baseline, mask=mask,
                           wcf_zero_mask=wcf_zero_mask, df_regions=df_regions,
                           hatchings=hatchings, ds_wasserstein=ds_wasserstein)


def iter_gwl_decomp(args, inputs):
    """Yield (level, gwl_key, gwl_label, fields) for each requested GWL, where
    `fields` is from_ds_to_plot_decomp's (ref freq/int/dur, proj freq/int/dur,
    weight) tuple aligned against the GWL0-61 baseline."""
    print("\n" + "=" * 60)
    print("STEP 4 - Producing figures")
    print("=" * 60)
    for level in args.gwl_levels:
        gwl_key   = LEVEL_TO_KEY[level]
        gwl_label = LEVEL_TO_LABEL[level]
        if gwl_key not in inputs.built:
            print(f"  [warn] No data built for {gwl_label}, skipping.")
            continue
        print(f"\nProcessing {gwl_label}")
        print(f"  Aligning GCMs")
        yield level, gwl_key, gwl_label, from_ds_to_plot_decomp(
            inputs.built[gwl_key], inputs.ds_baseline)


def main():
    args = parse_args()
    inputs = prepare_inputs(args, with_regions=True)
    mask = inputs.mask

    # GWL1.5 rel_change field used as fixed colour-bin mask for all GWL levels
    gwl15_rel_change_data = None
    for level, gwl_key, gwl_label, fields in iter_gwl_decomp(args, inputs):
        (da_ref_freq, da_ref_int, da_ref_dur,
         da_proj_freq, da_proj_int, da_proj_dur, base_weight) = fields
        # Inverse-W2 pixel weights (or the flat 1/n_gcm with --weighting mmm)
        weight, inv_w2 = ensemble_weight(da_proj_freq, base_weight, inputs)

        # Freeze colour-bin membership at GWL1.5 so higher GWLs report stats
        # for the same spatial zones that were red/orange/gray/etc. at 1.5°C.
        if level == "1.5":
            gwl15_rel_change_data = _compute_ensemble_rel_change(
                da_ref_freq, da_ref_int, da_ref_dur,
                da_proj_freq, da_proj_int, da_proj_dur,
                weight=weight, mask=mask, lat_min=-60, lat_max=68,
            )

        print(f"  Computing global change statistics ...")
        global_chg, ci_lo, ci_hi, gcm_ci_lo, gcm_ci_hi = compute_global_change_stats_gwl(
            da_ref_freq, da_ref_int, da_ref_dur,
            da_proj_freq, da_proj_int, da_proj_dur,
            weight=weight, mask=mask,
            lat_min=-60, lat_max=68, inv_w2=inv_w2,
        )
        print(
            f"  Global mean change under {gwl_label}: {global_chg:+.2f}% "
            f"[spatial CI: {ci_lo:+.2f}%, {ci_hi:+.2f}%] "
            f"[GCM CI: {gcm_ci_lo:+.2f}%, {gcm_ci_hi:+.2f}%]"
        )
        print(f"  Computing per-bin change statistics ...")
        bin_stats = compute_bin_change_stats_gwl(
            da_ref_freq, da_ref_int, da_ref_dur,
            da_proj_freq, da_proj_int, da_proj_dur,
            weight=weight, mask=mask,
            lat_min=-60, lat_max=68,
            reference_data=gwl15_rel_change_data, inv_w2=inv_w2,
        )
        print(f"  Per-bin statistics under {gwl_label}:")
        for s in bin_stats:
            print(
                f"    {s['bin']:45s}  mean={s['mean']:+7.2f}%  "
                f"[spatial CI: {s['ci_lower']:+7.2f}%, {s['ci_upper']:+7.2f}%]  "
                f"[GCM CI: {s['gcm_ci_lower']:+7.2f}%, {s['gcm_ci_upper']:+7.2f}%]  "
                f"({s['n_pixels']} pixels)"
            )
        print(f"  Plotting ...")
        fig = plot_gwl_valuebyalpha_discrete(
            da_ref_freq=da_ref_freq, da_ref_int=da_ref_int, da_ref_dur=da_ref_dur,
            da_proj_freq=da_proj_freq, da_proj_int=da_proj_int, da_proj_dur=da_proj_dur,
            weight=weight, mask=mask,
            shapefile_path=args.shapefile,
            df_regions=inputs.df_regions,
            gwl_label=gwl_label,
            hatchings=inputs.hatchings,
            agreement_threshold=args.agreement_threshold,
            map_title=f"Annual severity change under {gwl_label} warming",
            relchange_label="Relative change (%)",
            sev_label="Average annual\nseverity (0.61 °C)",
            lat_min=-60, lat_max=68,
            wcf_zero_mask=inputs.wcf_zero_mask,
        )
        out_path = os.path.join(args.output_dir, "main",
                                f"fig3_projected_change_valuebyalpha_{gwl_key}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fit_to_width(fig)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out_path}")

        del fields, da_ref_freq, da_ref_int, da_ref_dur
        del da_proj_freq, da_proj_int, da_proj_dur, weight, base_weight, inv_w2, fig
        gc.collect()

    if inputs.ds_wasserstein is not None:
        inputs.ds_wasserstein.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

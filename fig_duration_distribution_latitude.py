# -*- coding: utf-8 -*-
"""
Distribution of WSED (wind-solar energy drought) event duration by latitude
zone, decomposed by global warming level (GWL) and pooled across every
available GCM/run realization.

Latitude zones (5), using this project's own poleward exclusion band
(MAP_LAT_SOUTH/MAP_LAT_NORTH -- see fig1.py/fig3.py: "Regions poleward of
68N and 58S were excluded due to artifacts in the duration metric"). Zone
edges live in LAT_ZONE_EDGES below -- this docstring just names them:
    Tropical          -20 .. 20
    Subtropical (N)     20 .. 40
    Subtropical (S)    -40 .. -20
    Midlatitude (N)     40 .. MAP_LAT_NORTH
    Midlatitude (S)     MAP_LAT_SOUTH .. -40

Event durations come from duration_decomposition.compute_event_table, the
same gap-free run-length encoding of the classic daily wind+solar
coincidence field used by fig1.py/fig3.py -- one row per contiguous compound
low-production spell (event) at one pixel, not a per-pixel-year mean, so the
resulting distribution reflects actual individual event lengths pooled over
every land pixel in the zone and every available GCM/run realization.

Duration is a small integer count (1, 2, 3, ... days), so each distribution
is drawn at each integer duration (share or raw count -- see below) on a log
y-axis, rather than as a continuous KDE -- a Gaussian KDE would fabricate
density between integers that cannot occur and, with a bandwidth narrow
enough to resolve the dominant 1-day spike, oscillates between them.

Every (zone, GWL) group is pooled with *equal GCM weighting*, not a plain
sum over every contributing (GCM, run) realization: each GCM's own runs are
averaged together first, then every GCM counts equally regardless of how
many runs or how many GCMs happen to be available (see _group_counts). This
project's ensemble is uneven enough that this matters a lot -- e.g. CanESM5
contributes 10 runs and MPI-ESM1-2-LR 9, out of ~33 realizations at
GWL0-61/1.5/2, while GWL3 drops to 18 realizations from only 8 GCMs (several,
including MPI-ESM1-2-LR, absent entirely). A plain sum would let those two
GCMs dominate every GWL and would make GWL3 look artificially low purely
from having fewer contributing GCMs, not from any real change in event
frequency.

Three figures are produced, all on the normalized *share* scale (each
duration's share of that (zone, GWL) group's GCM-weighted total events,
summing to 1) rather than raw/GCM-weighted counts -- shape-of-distribution
comparisons across GWLs are what these figures are for:
  1. fig_duration_distribution_by_latitude_share.png
     Only the pooled line -- shows how the duration mix changes with
     warming. X-axis cropped to the 99th percentile of pooled duration
     (max_duration_days).
  2. fig_duration_distribution_by_latitude_share_bootstrap.png
     Same crop, with a shaded confidence band from bootstrap-resampling
     which GCMs contribute (each GCM's runs pre-averaged, same weighting),
     instead of drawing each realization's own line -- see
     _bootstrap_band_from_counts.
  3. fig_duration_distribution_by_latitude_share_full.png
     Same as (2) (pooled line + bootstrap confidence band), but the x-axis
     extends well past the 99th-percentile crop, out to
     FULL_FIGURE_MAX_DURATION_DAYS, so much more of the tail is visible --
     capped there rather than at the true longest duration on record (which
     can run into the hundreds of days for some (zone, GWL) tails) to keep
     the zoomed panel legible.
Every figure carries a vertical dashed line at each GWL's mean duration, and
a vertical dotted line at each GWL's 99.5th-percentile ("high duration
tail") duration -- the latter computed from the *full* duration record for
that (zone, GWL), not cropped to whatever window a given figure plots (see
_group_percentile_full), so it still marks the true tail location even on
the two cropped figures.

Each of the 5 latitude-zone rows (panels b-f; panel a is the locator map) is
itself split into two side-by-side log-scale panels: a wide main panel (days
1-DURATION_SPLIT_DAY) and a narrower zoomed panel (DURATION_SPLIT_DAY
onward, to max_duration_days). Each panel plots only its own window's data,
so the split doesn't crush the long thinning tail flat under the dominant
short-duration spike. Day DURATION_SPLIT_DAY is marked with a thin vertical
line in the main panel where the split occurs. The y-axis is shared *within*
each panel type (main panels share one range across all 5 zones; zoomed
panels likewise share their own common range) so panel-to-panel amplitude is
directly comparable across latitude zones, rather than each row autoscaling
to its own data. Each panel's title also states its zone's explicit latitude
range (e.g. "40N-68N"), not just the zone's plain-language name.
"""
import os
import config
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV

import argparse
import gc
import glob
import json

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import cartopy.crs as ccrs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionPatch

# Zarr/NetCDF-agnostic file lookup + opener, shared with calculate_cf.py /
# every other fig*.py script.
from io_utils import match_files, glob_any, open_dataset_any

# Generic event-table builder (gap-free run-length encoding of a boolean/0-1
# compound-day field into one row per event), shared with fig1.py/fig3.py's
# duration-class decomposition.
from duration_decomposition import compute_event_table

# =============================================================================
# Figure size / fontsize / latitude-band constants (match fig1.py/fig3.py's
# and classify_gcm_trend_agreement.py's build_wasserstein_composite_figure
# convention: one FIG_WIDTH_IN constant drives every figsize, and text gets
# named, purpose-specific fontsize constants instead of ad hoc numbers at
# each call site)
# =============================================================================
FIG_WIDTH_IN = 5.15   # single column width -- matches fig1.py's LaTeX width

XLABEL_FONTSIZE = 8     # shared x-axis label
LEGEND_FONTSIZE = 7     # bottom GWL legend
LETTER_FONTSIZE = 8     # bold panel letters (a, b, c, ...)
ZONE_TITLE_FONTSIZE = 8  # per-panel zone name, next to its letter
AXIS_LABEL_FONTSIZE = 7.5  # per-panel y-axis label
TICK_FONTSIZE = 7       # per-panel tick labels

# Each latitude-zone row is split into a wide main panel (duration <=
# DURATION_SPLIT_DAY) and a narrower zoomed panel (duration >
# DURATION_SPLIT_DAY) sharing one log-scale y-axis, so the long thinning tail
# isn't crushed flat by the dominant short-duration spike. The split point
# itself (DURATION_SPLIT_DAY) is included in both panels' visible range, so
# the two halves visually connect.
DURATION_SPLIT_DAY = 5
DURATION_ZOOM_WIDTH_RATIOS = [45, 55]
DURATION_SPLIT_LINE_COLOR = "#777777"

# Percentile marked by the dotted "high duration tail" vertical line drawn
# per GWL alongside the existing dashed mean-duration line -- see
# _group_percentile_full.
HIGH_DURATION_PCTILE = 99.5

# X-axis cap for the uncropped "_full" figure -- the true longest recorded
# duration can run into the hundreds of days for some (zone, GWL) tails,
# which packs the zoomed panel with more points than are legible; 20 days
# still shows well past DURATION_SPLIT_DAY without that crowding.
FULL_FIGURE_MAX_DURATION_DAYS = 20.0

# Fixed length of every GWL's time window (see calculate_cf.py's ~20-year
# GWL slicing) -- the exposure denominator for the return-period figure
# (plot_return_periods / _group_return_period).
GWL_WINDOW_YEARS = 20.0

MAP_LAT_SOUTH = -58.0
MAP_LAT_NORTH = 68.0

# Longitude bounds shown on the locator map -- cropped in from the full
# -180/180 globe (which left the map essentially edge-to-edge with the
# distribution-panel column) and left-anchored (see _add_locator_map), so
# the freed-up width becomes visible whitespace between the map and the
# panels rather than being split as padding on both sides of the map.
MAP_LON_WEST = -155.0
MAP_LON_EAST = 155.0

LAT_ZONE_EDGES = [MAP_LAT_SOUTH, -40, -20, 20, 40, MAP_LAT_NORTH]
LAT_ZONE_LABELS = [
    "Midlatitude (S)", "Subtropical (S)", "Tropical", "Subtropical (N)", "Midlatitude (N)",
]
# (lat_lo, lat_hi) per zone, derived from LAT_ZONE_EDGES -- used both for the
# locator-map bands and for sizing the distribution panels proportionally to
# their true latitudinal extent.
ZONE_BOUNDS = {
    label: (LAT_ZONE_EDGES[i], LAT_ZONE_EDGES[i + 1])
    for i, label in enumerate(LAT_ZONE_LABELS)
}
# One distinct colour per zone (5 of ColorBrewer's "Dark2" qualitative
# palette), used only to tie each map band to its panel (spine tab,
# connector line, zone-name text) -- deliberately avoids GWL_COLORS' whole
# blue/light-blue/orange/red hue range below, since both color sets appear
# together in every panel and a zone color that reads as "blue" or "red"
# could be mistaken for a particular GWL line.
ZONE_MAP_COLORS = {
    "Midlatitude (S)": "#1B9E77",  # teal-green
    "Subtropical (S)": "#66A61E",  # green
    "Tropical":        "#A6761D",  # brown
    "Subtropical (N)": "#E7298A",  # magenta
    "Midlatitude (N)": "#7570B3",  # purple
}

GWL_KEYS = ["GWL0-61", "GWL1-5", "GWL2", "GWL3"]
GWL_LABELS = {
    "GWL0-61": "Baseline (~0.61°C)",
    "GWL1-5":  "+1.5°C",
    "GWL2":    "+2.0°C",
    "GWL3":    "+3.0°C",
}
# Cool (baseline) -> warm (highest GWL) progression, ColorBrewer RdYlBu-4.
GWL_COLORS = {
    "GWL0-61": "#2c7bb6",
    "GWL1-5":  "#abd9e9",
    "GWL2":    "#fdae61",
    "GWL3":    "#d7191c",
}


# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Distribution of WSED event duration by latitude zone, "
            "decomposed by GWL and pooled over every available GCM/run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preprocessed_path", default=config.PATH_PREPROCESSED)
    parser.add_argument("--ssp", default=config.SSP)
    parser.add_argument(
        "--gwl_list", nargs="+", default=GWL_KEYS,
        help="GWL keys to include (default: GWL0-61 GWL1-5 GWL2 GWL3).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Quantile defining a low-production day, from the GWL0-61 reference "
             "period (default: 0.1, matching fig1.py/fig3.py).",
    )
    parser.add_argument("--exclude_gcm", nargs="+", default=[])
    parser.add_argument("--exclude_gcm_run", nargs="+", default=config.EXCLUDE_GCM_RUN)
    parser.add_argument("--shapefile", default=config.SHAPEFILE_PATH)
    parser.add_argument(
        "--era5_grid_path", default=None,
        help=(
            "Override the default preprocessed_path/ERA5/wcf_day* lookup with any "
            "single NetCDF/Zarr file carrying the ERA5 lat/lon grid (only its "
            "coordinates are read). Useful for local testing off a cached counts "
            "table when the raw HPC-only preprocessed archive isn't available."
        ),
    )
    parser.add_argument("--output_dir", default="../final_figs")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--max_duration_days", type=float, default=None,
        help="X-axis cap shared by every panel (default: 99th percentile of "
             "every pooled event duration across all requested GWLs).",
    )
    parser.add_argument(
        "--min_events", type=int, default=5,
        help="Minimum pooled events a (zone, GWL) group needs to be drawn at all "
             "(default: 5).",
    )
    parser.add_argument(
        "--n_boot", type=int, default=500,
        help="Bootstrap resamples (over GCM/run realizations) for the bootstrap "
             "figure's uncertainty band (default: 500).",
    )
    parser.add_argument(
        "--ci", type=float, default=90.0,
        help="Bootstrap confidence interval width in %% (default: 90).",
    )
    parser.add_argument(
        "--cache_csv", default=None,
        help=(
            "Path to the event-duration-count cache CSV (one row per "
            "gwl/GCM/run/zone/duration, with its event count). Default: "
            "<output_dir>/event_duration_counts_cache.csv. If it exists (and "
            "was built with the same latitude bands, threshold and ssp -- "
            "checked automatically), it is loaded instead of rebuilding from "
            "the raw wcf/scf files, which is by far the most expensive step."
        ),
    )
    parser.add_argument(
        "--recompute", action="store_true", default=False,
        help="Ignore any existing cache and rebuild the event-duration counts from scratch.",
    )
    parser.add_argument(
        "--min_exceed", type=int, default=10,
        help="Minimum pooled (unweighted, across every GCM/run) exceedance count a duration "
             "needs to be drawn on the return-period figure (default: 10).",
    )
    parser.add_argument(
        "--raw_grid_path", default=config.PATH_FOLDER,
        help="Root folder of the raw (pre-bias-adjustment) GCM archive, one tas_day_*.nc per "
             "GCM/run/gwl -- used only to read each GCM's native lat/lon grid for the "
             "return-period figure's per-GCM pixel-count denominator (default: config.PATH_FOLDER).",
    )
    parser.add_argument(
        "--uncertainty_gwl", default="GWL3",
        help="GWL key to draw the single-GWL all-realizations uncertainty check for (default: "
             "GWL3, this project's smallest/most uncertain ensemble). Pass '' to skip that figure.",
    )
    return parser.parse_args()


# =============================================================================
# Land mask (same rasterize-from-grid approach as fig1.py/fig3.py)
# =============================================================================

def rasterize_shapefile(shapefile, shape, transform):
    return geometry_mask(
        geometries=shapefile["geometry"], all_touched=True,
        out_shape=shape, transform=transform, invert=True,
    )


def build_land_mask_from_grid(lat, lon, shapefile_path):
    shapefile = gpd.read_file(shapefile_path)
    shape = (len(lat), len(lon))
    transform = rasterio.transform.from_bounds(
        float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat)),
        len(lon), len(lat),
    )
    mask = rasterize_shapefile(shapefile, shape, transform)
    return mask[::-1, :]


# =============================================================================
# Latitude-zone assignment
# =============================================================================

def assign_lat_zone(lat_values):
    """Vectorized zone label for an array of latitudes, per LAT_ZONE_EDGES."""
    idx = np.digitize(np.asarray(lat_values, dtype=float), LAT_ZONE_EDGES[1:-1])
    return np.array(LAT_ZONE_LABELS, dtype=object)[idx]


def load_era5_reference_grid(preprocessed_path, era5_grid_path=None):
    """
    (lat, lon) of the common ERA5 reference grid, cropped to
    MAP_LAT_SOUTH..MAP_LAT_NORTH -- shared by compute_land_area_share_per_zone
    (the zone land-area-share computation) and main()'s pixel scatter for the
    return-period figure's locator map (see plot_return_periods), so both use
    the exact same reference coordinates.

    `era5_grid_path` overrides the default preprocessed_path/ERA5/wcf_day*
    lookup with any single NetCDF/Zarr file that carries the ERA5 lat/lon
    grid (only its coordinates are read, not any data variable) -- useful
    when no raw ERA5 wcf/scf file is available locally but a regridded
    product on the same grid is (e.g. for local testing off a cached
    counts table instead of the raw HPC-only preprocessed archive).
    """
    if era5_grid_path:
        rea_files = [era5_grid_path]
    else:
        rea_files, _ = match_files(os.path.join(preprocessed_path, "ERA5", "wcf_day*"))
    if not rea_files:
        raise FileNotFoundError(
            f"No ERA5 reference grid found under {os.path.join(preprocessed_path, 'ERA5')} "
            "(pass --era5_grid_path to point at one explicitly). Cannot determine the "
            "reference grid."
        )
    ds = open_dataset_any(rea_files[0]).sortby("lat").sortby("lon")
    ds = ds.sel(lat=slice(MAP_LAT_SOUTH, MAP_LAT_NORTH))
    return ds.lat.values, ds.lon.values


def compute_land_area_share_per_zone(lat, lon, shapefile_path):
    """
    Percent of the analysis domain's total land area, on the common ERA5
    reference grid (see load_era5_reference_grid), that falls in each
    latitude zone (sums to 100 across the 5 zones). ERA5 is the right grid
    to measure this on: every GCM/run is regridded onto it before being
    compared against reanalysis anywhere else in this project (e.g.
    make_grid_files.py / fig3.py's
    `xe.Regridder(ds_final, wcf_rea, method="nearest_s2d")`), so it is the
    one common spatial definition shared across the whole ensemble --
    independent of which GCM's own native grid actually produced a given
    zone's event durations (build_daily_compound keeps each realization at
    its own native resolution, precisely so this reference count doesn't
    have to be a per-GCM quantity).

    Area, not a raw pixel count: on a regular lat/lon grid, a cell's true
    surface area shrinks toward the poles (fixed dlat/dlon spans less real
    distance in longitude at high latitude), scaling with cos(latitude) --
    so a plain pixel count over-represents high-latitude zones relative to
    their actual share of land. Each land pixel's contribution is weighted
    by cos(its latitude) before summing.

    Land/ocean comes only from the shapefile mask (build_land_mask_from_grid
    -- the same one build_events_for_realization uses), not additionally
    intersected with any one day of reanalysis data being non-NaN. An
    earlier version of this function did intersect with a single day's
    (isel(time=0)) non-NaN mask, meant to also exclude perennially
    below-cut-in ("grey") land pixels -- but keyed off one arbitrary daily
    snapshot, so a region that happened to be entirely calm that one day
    (plausible for, e.g., a Southern-Hemisphere-summer day) was silently
    zeroed out; that is what produced the "0 px" bug for the Southern zones
    this function replaces. Using only the shapefile mask also keeps this
    figure consistent with what actually generates the plotted events, since
    build_events_for_realization does not apply that cut-in exclusion
    either.
    """
    land_mask = build_land_mask_from_grid(lat, lon, shapefile_path)
    row_area_weight = np.cos(np.deg2rad(lat))
    zone_of_row = assign_lat_zone(lat)

    land_area = {}
    for zlabel in LAT_ZONE_LABELS:
        row_sel = zone_of_row == zlabel
        land_area[zlabel] = float(
            (land_mask[row_sel, :].sum(axis=1) * row_area_weight[row_sel]).sum())

    total_area = sum(land_area.values())
    return {z: (100.0 * a / total_area if total_area > 0 else 0.0)
            for z, a in land_area.items()}


# =============================================================================
# Discover available GCM/run realizations for one GWL
# =============================================================================

def discover_realizations(preprocessed_path, gwl, ssp, exclude_gcm, exclude_gcm_run):
    """
    Glob every wcf_day_*<ssp>*<gwl>_ERA5(.zarr|.nc) file under
    preprocessed_path/*/ and parse out (GCM, run) from the filename -- same
    convention as fig3.py's build_gridded_datasets (filenames are always
    wcf_day_{GCM}_{ssp}_{run}_{gwl}_ERA5.ext).
    """
    exclude_gcm = set(exclude_gcm or [])
    exclude_pairs = set(tuple(x.split(":")) for x in (exclude_gcm_run or []))

    wcf_paths = glob_any(os.path.join(preprocessed_path, "*", f"wcf_day_*{ssp}*{gwl}_ERA5"))
    realizations = []
    for p in wcf_paths:
        gcm = p.split("_")[-5]
        run = p.split("_")[-3]
        if gcm in exclude_gcm or (gcm, run) in exclude_pairs:
            print(f"    [excluded] {gcm} {run}")
            continue
        realizations.append((gcm, run))
    return realizations


def discover_gcm_grids(raw_grid_path, exclude_gcm):
    """
    One (lat, lon) native grid per GCM, read from any single raw CMIP6
    tas_day_{GCM}_*.nc file under raw_grid_path/{GCM}/ -- every run/GWL/date
    range of the same GCM is assumed to share its native output grid (same
    model, just a different ensemble member or time window), so one file is
    enough, and "tas" is used because it's the one variable every GCM in
    this archive reliably has (unlike e.g. uas/vas, sometimes derived from
    sfcWind instead -- see calculate_cf.py's load_ds).

    This is `config.PATH_FOLDER`, the *raw* pre-bias-adjustment archive
    (filenames like tas_day_ACCESS-CM2_ssp245_r1i1p1f1_19950101-20141231_
    GWL0-61.nc, no "_ERA5" suffix) -- not preprocessed_path's derived
    wcf_day_*_ERA5 files discover_realizations reads, which don't exist for
    every ssp/gwl/run combination locally. Grid discovery doesn't care:
    build_daily_compound never regrids a GCM off its own native grid, so any
    raw file for that GCM carries the same coordinates the wcf_day_*_ERA5
    field would. Only coordinates are read (no data loaded), sorted the same
    way build_daily_compound sorts them (sortby lat/lon) before use. Feeds
    compute_gcm_zone_pixel_counts, the per-GCM pixel-count denominator for
    the return-period figure (see _group_return_period) -- needed because
    build_daily_compound keeps every GCM at its own native resolution rather
    than a common regridded grid, so this count genuinely differs from GCM
    to GCM.
    """
    exclude_gcm = set(exclude_gcm or [])
    grids = {}
    for gcm_dir in sorted(glob.glob(os.path.join(raw_grid_path, "*"))):
        if not os.path.isdir(gcm_dir):
            continue
        gcm = os.path.basename(gcm_dir.rstrip("/\\"))
        if gcm in exclude_gcm:
            continue
        tas_files = sorted(glob.glob(os.path.join(gcm_dir, "tas_day_*.nc")))
        if not tas_files:
            continue
        # decode_times=False: only lat/lon are read here, and some raw
        # archive files carry a time_bnds encoding pandas/cftime can't
        # decode (e.g. an out-of-range reference date) -- irrelevant to the
        # grid itself, so decoding time is skipped entirely rather than
        # worked around.
        ds = open_dataset_any(tas_files[0], decode_times=False).sortby("lat").sortby("lon")
        grids[gcm] = (ds.lat.values.astype(float), ds.lon.values.astype(float))
    return grids


def compute_gcm_zone_pixel_counts(grids, shapefile_path):
    """
    cos(latitude)-area-weighted land-pixel count per (GCM, zone), on each
    GCM's own native grid (see discover_gcm_grids), restricted to the same
    MAP_LAT_SOUTH..MAP_LAT_NORTH band build_daily_compound crops every
    compound field to -- so these counts match the actual footprint that
    generated the plotted events. Same area-weighting convention as
    compute_land_area_share_per_zone (a fixed dlat/dlon grid cell's true
    surface area shrinks toward the poles, so each land pixel's row is
    scaled by cos(its latitude) before summing) -- applied here on the
    return-period figure's per-GCM pixel-count *denominator* only.

    The exceedance-count *numerator* in _group_return_period comes from
    counts_df, the cached (zone, duration) event histogram, which pools raw
    per-event counts with no per-pixel latitude retained -- so it can't
    itself be area-weighted without rebuilding that (expensive) cache from
    the raw per-event data. This weighted pixel count is the one piece of
    the return-period calculation area weighting can reach without that
    rebuild, and is cheap enough (coordinates + a shapefile mask, no daily
    fields) to (re)compute fresh every run regardless.
    """
    out = {}
    for gcm, (lat, lon) in grids.items():
        lat_sorted = np.sort(lat)
        lon_sorted = np.sort(lon)
        in_band = (lat_sorted >= MAP_LAT_SOUTH) & (lat_sorted <= MAP_LAT_NORTH)
        lat_band = lat_sorted[in_band]
        land_mask = build_land_mask_from_grid(lat_band, lon_sorted, shapefile_path)
        row_area_weight = np.cos(np.deg2rad(lat_band))
        zone_of_row = assign_lat_zone(lat_band)
        out[gcm] = {
            zlabel: float(
                (land_mask[zone_of_row == zlabel, :].sum(axis=1)
                 * row_area_weight[zone_of_row == zlabel]).sum())
            for zlabel in LAT_ZONE_LABELS
        }
    return out


# =============================================================================
# Per-realization daily compound field + event table
# =============================================================================

def build_daily_compound(preprocessed_path, gwl, gcm, run, ssp, threshold):
    """
    Classic daily wind+solar coincidence field (no rolling mean), same
    definition as fig1.py's build_daily_pipeline / fig3.py's
    _build_single_gcm: a day is "low wind" / "low solar" if wcf/scf falls at
    or below the `threshold` quantile of positive values over the GWL0-61
    reference period; "compound" is both at once. Kept at native GCM
    resolution (no regridding -- we only need per-pixel event lengths, not a
    cross-GCM common grid).
    """
    wcf_files, _ = match_files(
        os.path.join(preprocessed_path, gcm, f"wcf_day_{gcm}_{ssp}_{run}_{gwl}_ERA5"))
    scf_files, _ = match_files(
        os.path.join(preprocessed_path, gcm, f"scf_day_{gcm}_{ssp}_{run}_{gwl}_ERA5"))
    if not wcf_files or not scf_files:
        raise FileNotFoundError(f"Missing wcf/scf files for {gcm}/{run}/{gwl}")
    wcf = open_dataset_any(wcf_files[0]).convert_calendar("standard")
    scf = open_dataset_any(scf_files[0]).convert_calendar("standard")

    wcf_ref_paths = glob_any(
        os.path.join(preprocessed_path, gcm, f"wcf_day_{gcm}*{ssp}*{run}_GWL0-61_ERA5"))
    scf_ref_paths = glob_any(
        os.path.join(preprocessed_path, gcm, f"scf_day_{gcm}*{ssp}*{run}_GWL0-61_ERA5"))
    if not wcf_ref_paths or not scf_ref_paths:
        raise FileNotFoundError(f"Missing GWL0-61 reference files for {gcm}/{run}")
    wcf_ref = open_dataset_any(wcf_ref_paths[0]).convert_calendar("standard")
    scf_ref = open_dataset_any(scf_ref_paths[0]).convert_calendar("standard")

    wcf_thr = wcf_ref.wcf.where(wcf_ref.wcf > 0).quantile(threshold, dim="time")
    scf_thr = scf_ref.scf.where(scf_ref.scf > 0).quantile(threshold, dim="time")

    low_wind  = xr.where(wcf.wcf <= wcf_thr, 1, 0)
    low_solar = xr.where(scf.scf <= scf_thr, 1, 0)
    compound  = (low_wind * low_solar).astype(int)
    compound  = compound.sortby("lat").sortby("lon")
    compound["lat"] = compound["lat"].astype(float)
    compound["lon"] = compound["lon"].astype(float)
    compound  = compound.sel(lat=slice(MAP_LAT_SOUTH, MAP_LAT_NORTH))
    compound  = compound.load()

    del wcf, scf, wcf_ref, scf_ref
    gc.collect()
    return compound


def build_events_for_realization(preprocessed_path, gwl, gcm, run, ssp, threshold, shapefile_path):
    """
    One row per WSED event (contiguous compound low-production spell) at one
    land pixel, with its total duration (days), the year it started, and its
    latitude zone. Ocean pixels are zeroed out (not NaN) before event
    detection so they simply generate zero events, rather than needing
    NaN-aware handling downstream.
    """
    compound = build_daily_compound(preprocessed_path, gwl, gcm, run, ssp, threshold)

    land_mask = build_land_mask_from_grid(compound.lat.values, compound.lon.values, shapefile_path)
    land_mask_da = xr.DataArray(land_mask, dims=("lat", "lon"),
                                 coords={"lat": compound.lat, "lon": compound.lon})
    compound = compound.where(land_mask_da, 0).astype(int)

    df = compute_event_table(compound)
    df = df.drop_duplicates(["event_id", "lat", "lon"])[["lat", "year", "duration"]].copy()
    df["zone"] = assign_lat_zone(df["lat"].to_numpy())
    df["GCM"] = gcm
    df["run"] = run
    df["gwl"] = gwl

    del compound, land_mask_da
    gc.collect()
    return df


COUNTS_CACHE_COLUMNS = ["gwl", "GCM", "run", "zone", "duration", "count"]


def build_counts_table(preprocessed_path, gwl_list, ssp, threshold, shapefile_path,
                        exclude_gcm, exclude_gcm_run):
    """
    One row per (gwl, GCM, run, zone, duration) with the number of WSED
    events of that exact duration -- built by aggregating each realization's
    event table (build_events_for_realization) immediately, one at a time,
    rather than concatenating every raw per-event row across every
    realization first. This is the expensive step (opens and processes every
    GCM/run's daily wcf/scf files); its result is what main() caches to CSV
    (see save_counts_cache/load_counts_cache) so repeat plotting runs don't
    have to redo it.
    """
    rows = []
    for gwl in gwl_list:
        print(f"\n  -- {gwl} --")
        realizations = discover_realizations(preprocessed_path, gwl, ssp, exclude_gcm, exclude_gcm_run)
        if not realizations:
            print(f"    No files found for {gwl}, skipping.")
            continue

        for gcm, run in realizations:
            print(f"    {gcm} / {run}")
            try:
                df_events = build_events_for_realization(
                    preprocessed_path, gwl, gcm, run, ssp, threshold, shapefile_path)
            except Exception as exc:
                print(f"      [ERROR] {gcm}/{run}/{gwl}: {exc}")
                continue
            frag = (df_events.groupby(["zone", "duration"]).size()
                    .reset_index(name="count"))
            frag["gwl"] = gwl
            frag["GCM"] = gcm
            frag["run"] = run
            rows.append(frag[COUNTS_CACHE_COLUMNS])
            del df_events, frag
            gc.collect()

    if not rows:
        return pd.DataFrame(columns=COUNTS_CACHE_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def _cache_meta(threshold, ssp):
    """Parameters that change the counts, guarded against on cache load."""
    return {"lat_zone_edges": LAT_ZONE_EDGES, "threshold": threshold, "ssp": ssp}


def save_counts_cache(counts_df, path, threshold, ssp):
    """
    Write the counts cache as a plain CSV with one leading '#'-commented
    metadata line (latitude bands, threshold, ssp) that load_counts_cache
    checks before trusting the cache -- so a later change to LAT_ZONE_EDGES
    (or --threshold/--ssp) doesn't silently reuse counts binned under the
    old definition.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(f"# {json.dumps(_cache_meta(threshold, ssp))}\n")
        counts_df.to_csv(f, index=False)


def load_counts_cache(path, threshold, ssp):
    """
    Returns the cached counts DataFrame if `path` exists and its leading
    metadata line matches the current latitude bands/threshold/ssp;
    otherwise None (caller falls back to rebuilding from scratch).
    """
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        first_line = f.readline()
    if not first_line.startswith("#"):
        print(f"  [cache] {path} has no metadata header -- ignoring stale/foreign cache.")
        return None
    try:
        meta = json.loads(first_line[1:].strip())
    except (json.JSONDecodeError, ValueError):
        print(f"  [cache] {path} has an unreadable metadata header -- ignoring.")
        return None
    expected = _cache_meta(threshold, ssp)
    if meta != expected:
        print(f"  [cache] {path} was built with different settings {meta} "
              f"than requested {expected} -- ignoring and rebuilding.")
        return None
    return pd.read_csv(path, comment="#")


# =============================================================================
# Plotting
# =============================================================================

def _for_line(arr):
    """NaN out zero bins (not drawn as 0) so a log-scale line shows real gaps as gaps."""
    return np.where(arr > 0, arr, np.nan)


def _weighted_percentile(values, weights, pct):
    """pct-th percentile of `values` weighted by `weights` (e.g. event counts per duration)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights)
    if cum.size == 0 or cum[-1] <= 0:
        return float(values[-1]) if values.size else 0.0
    idx = min(int(np.searchsorted(cum, pct / 100.0 * cum[-1])), len(values) - 1)
    return float(values[idx])


def _fmt_lat_range(zone_label):
    """'40N-68N' / '58S-40S' / '20S-20N' style explicit range for a zone's panel title."""
    lo, hi = ZONE_BOUNDS[zone_label]
    fmt = lambda d: "0" + "°" if d == 0 else f"{abs(d):.0f}°{'N' if d > 0 else 'S'}"
    return f"{fmt(lo)}–{fmt(hi)}"


def _group_percentile_full(counts_df, gwl, zone, pct):
    """
    GCM-weighted `pct`-th percentile duration for one (gwl, zone), computed
    over the *full* duration range on record for that group -- not capped to
    whatever max_duration_days window a given figure happens to plot. Same
    equal-GCM weighting as _group_counts's arr_share (each GCM's own runs
    averaged first, then each GCM's own normalized duration-share vector
    averaged with weight 1/n_GCM), so a heavily-resampled or grid-dense GCM
    doesn't skew the tail estimate. Used for the high-duration (q99.5)
    marker line, which needs the true tail location even on figures whose
    x-axis is cropped well below it. Returns None if there is no data for
    this (gwl, zone).
    """
    sub = counts_df[(counts_df["gwl"] == gwl) & (counts_df["zone"] == zone)]
    if sub.empty:
        return None

    durations = np.sort(sub["duration"].unique())
    dur_idx = {d: j for j, d in enumerate(durations)}
    weight_sum = np.zeros(len(durations))
    n_gcm = 0
    for gcm, gsub in sub.groupby("GCM"):
        n_runs = gsub["run"].nunique()
        s = gsub.groupby("duration")["count"].sum() / n_runs
        total_g = float(s.sum())
        if total_g <= 0:
            continue
        n_gcm += 1
        for d, c in s.items():
            weight_sum[dur_idx[d]] += c / total_g
    if n_gcm == 0:
        return None
    weight_sum /= n_gcm
    return _weighted_percentile(durations, weight_sum, pct)


def _group_counts(counts_df, gwl, zone, x_int):
    """
    Pooled (arr_share, arr_count, mean_duration, weighted_total, raw_total)
    for one (gwl, zone), with equal GCM weighting (1 / n_GCM each,
    regardless of how many runs or how many GCMs happen to be available --
    same inverse-run-count convention as fig3.py's
    align_realizations_and_weight / add_severity_and_weights). Without this,
    a GCM sampled with many runs (e.g. CanESM5's 10 runs vs. most GCMs'
    single run in this project's ensemble) dominates a plain pooled sum, and
    a GWL missing some GCMs entirely (e.g. GWL3 dropping from 14 to 8 GCMs
    once MPI-ESM1-2-LR and others drop out) ends up on a different,
    non-comparable scale purely from having fewer contributors -- not from
    any real change in event frequency. Returns None if there is no data for
    this (gwl, zone).

    `arr_share` is a genuine equal-GCM-weighted average of each GCM's own
    *normalized* duration share (that GCM's own run-averaged count at each
    duration, divided by that GCM's own run-averaged total across every
    duration on record, *then* averaged across GCMs with weight 1/n_GCM) --
    this is what "equal GCM weighting" has to mean for a shape/share
    comparison. Naively normalizing a GCM-weighted *count* instead (i.e.
    dividing a GCM-weighted sum by another GCM-weighted sum) does not
    achieve this: the same per-GCM weight appears in both the numerator and
    denominator and cancels out, so the result collapses to a plain pooled
    ratio in which a GCM contributing more raw events overall (e.g. from a
    finer native grid with more land pixels, independent of its actual
    duration mix) still dominates the shape -- exactly the distortion this
    weighting is meant to prevent. `arr_count` is the older GCM-weighted
    *count* (equal-weighted mean of each GCM's own run-averaged count),
    still correct on its own terms and used only for the normalize=False
    "mean events per GCM" display. `mean_dur` is likewise now the
    equal-GCM-weighted average of each GCM's own mean duration, for the same
    reason -- not a count-weighted mean duration over the pooled sum.
    `weighted_total` (mean of each GCM's own total, on the same "mean events
    per GCM" scale) covers every duration on record, not just those within
    x_int. `raw_total` is the true pooled event count, kept separately only
    to gate min_events on actual sample size rather than a reweighted scale.
    """
    sub = counts_df[(counts_df["gwl"] == gwl) & (counts_df["zone"] == zone)]
    if sub.empty:
        return None
    raw_total = float(sub["count"].sum())

    gcm_series, gcm_totals, gcm_mean_durs = [], [], []
    for gcm, gsub in sub.groupby("GCM"):
        n_runs = gsub["run"].nunique()
        s = gsub.groupby("duration")["count"].sum() / n_runs
        total_g = float(s.sum())
        if total_g <= 0:
            continue
        gcm_series.append(s)
        gcm_totals.append(total_g)
        gcm_mean_durs.append(float((s.index.to_numpy() * s.to_numpy()).sum() / total_g))

    n_gcm = len(gcm_series)
    if n_gcm == 0:
        return None

    arr_share = np.zeros(len(x_int))
    arr_count = np.zeros(len(x_int))
    for s, total_g in zip(gcm_series, gcm_totals):
        vals = np.array([s.get(d, 0.0) for d in x_int], dtype=float)
        arr_share += (vals / total_g) / n_gcm
        arr_count += vals / n_gcm

    weighted_total = float(np.mean(gcm_totals))
    mean_dur = float(np.mean(gcm_mean_durs))
    return arr_share, arr_count, mean_dur, weighted_total, raw_total


def _return_period_rates_by_gcm(counts_df, gwl, zone, gcm_pixel_counts, x_int):
    """
    Per-GCM (not equal-GCM-averaged) per-(weighted-pixel) annual exceedance
    rate lambda_g(d), for every GCM in this (gwl, zone) group with a usable
    pixel count -- the shared building block behind both
    _group_return_period (which equal-GCM-weight-averages these into the
    one pooled curve every GWL line uses) and plot_gwl_uncertainty_check
    (which instead plots each individual per-GCM curve on its own, for a
    single-model-spread check). See _group_return_period's docstring for
    what each step of the rate computation means and why.

    Returns {gcm: (rate_array, exceed_raw_array)}, both arrays the same
    length as x_int -- exceed_raw is kept alongside rate so callers can
    gate on a GCM's own raw exceedance count (or the pooled sum of them)
    without recomputing it.
    """
    sub = counts_df[(counts_df["gwl"] == gwl) & (counts_df["zone"] == zone)]
    out = {}
    for gcm, gsub in sub.groupby("GCM"):
        n_pixels = gcm_pixel_counts.get(gcm, {}).get(zone, 0)
        if n_pixels <= 0:
            continue
        n_runs = gsub["run"].nunique()
        hist = gsub.groupby("duration")["count"].sum()
        counts_at_x = np.array([hist.get(d, 0.0) for d in x_int], dtype=float)
        exceed_raw = counts_at_x[::-1].cumsum()[::-1]
        rate = (exceed_raw / n_runs) / (GWL_WINDOW_YEARS * n_pixels)
        out[gcm] = (rate, exceed_raw)
    return out


def _group_return_period(counts_df, gwl, zone, gcm_pixel_counts, x_int, min_exceed=10):
    """
    Per-pixel return period T(d): expected number of years between
    duration->=d events at a single land pixel picked at random in this
    (gwl, zone), for every duration d in x_int. `gcm_pixel_counts` (see
    compute_gcm_zone_pixel_counts) carries cos(latitude) area weighting on
    the pixel-count *denominator*; the exceedance *numerator* comes from
    counts_df, the cached (zone, duration) histogram, which has no
    per-pixel latitude and so stays unweighted -- area weighting only
    reaches the half of this calculation the cache can support without a
    rebuild.

    Built from _return_period_rates_by_gcm's per-GCM rates lambda_g(d) (see
    its docstring for how each is computed: reverse-cumsum the duration
    histogram for the exceedance count, divide by that GCM's own n_runs,
    then by GWL_WINDOW_YEARS times that GCM's own area-weighted zone pixel
    count -- what makes the rate resolution-independent *within* that GCM).

    Equal-GCM-weighted mean of those already-normalized per-GCM rates (same
    "normalize per GCM first, then average with weight 1/n_GCM" convention
    as _group_counts's arr_share -- not a plain pooled sum, and not an
    average of T itself, which would over-weight GCMs with a low, noisy
    rate) is then inverted to get the plotted T. Returns None if no GCM in
    this group has a usable pixel count.

    Durations whose *pooled, unweighted* exceedance count (summed raw over
    every GCM/run, not equal-GCM-weighted) is below min_exceed are set to
    NaN -- past that point too few actual long events were observed for the
    empirical estimate to be shown.
    """
    by_gcm = _return_period_rates_by_gcm(counts_df, gwl, zone, gcm_pixel_counts, x_int)
    if not by_gcm:
        return None

    rates = [rate for rate, _ in by_gcm.values()]
    total_exceed = np.sum([exceed for _, exceed in by_gcm.values()], axis=0)
    mean_rate = np.mean(rates, axis=0)
    T = np.where(mean_rate > 0, 1.0 / mean_rate, np.nan)
    T = np.where(total_exceed >= min_exceed, T, np.nan)
    return T


def _bootstrap_band_from_counts(counts_df, gwl, zone, x_int, normalize, n_boot=500, ci=90, rng=None):
    """
    (lo, hi) envelope at each integer duration in x_int from resampling
    *GCMs* (not raw realizations) with replacement, n_boot times, each GCM's
    own runs averaged together first -- the same equal-GCM-weighting as the
    pooled line in _group_counts. Resampling raw (GCM, run) pairs instead
    would let a heavily-resampled GCM dominate the bootstrap draws too, not
    just the pooled sum, and would understate uncertainty. Works directly
    off the aggregated counts table (no raw per-event data needed). Returns
    (None, None) if fewer than 2 GCMs are available.

    When normalize=True, each bootstrap draw resamples GCMs' own normalized
    share vectors (S, one row per GCM: that GCM's run-averaged counts
    divided by that GCM's own run-averaged total) and averages those, not
    the raw counts divided by the resampled mean total -- the same
    ratio-of-means-vs-mean-of-ratios fix as _group_counts's arr_share (a
    high-volume GCM's raw counts would otherwise dominate both the
    numerator and denominator of every draw and distort the shape).
    """
    sub = counts_df[(counts_df["gwl"] == gwl) & (counts_df["zone"] == zone)]
    if sub.empty:
        return None, None
    gcms = sorted(sub["GCM"].unique())
    n_gcm = len(gcms)
    if n_gcm < 2:
        return None, None

    dur_idx = {d: j for j, d in enumerate(x_int)}
    M = np.zeros((n_gcm, len(x_int)))
    totals = np.zeros(n_gcm)
    for gi, gcm in enumerate(gcms):
        gsub = sub[sub["GCM"] == gcm]
        n_runs = gsub["run"].nunique()
        for dur, cnt in zip(gsub["duration"], gsub["count"]):
            j = dur_idx.get(dur)
            if j is not None:
                M[gi, j] += cnt / n_runs
        totals[gi] = gsub["count"].sum() / n_runs

    S = np.divide(M, totals[:, None], out=np.zeros_like(M), where=totals[:, None] > 0)

    rng = rng if rng is not None else np.random.default_rng(12345)
    boot = np.empty((n_boot, len(x_int)))
    for b in range(n_boot):
        idx = rng.integers(0, n_gcm, size=n_gcm)
        boot[b] = S[idx].mean(axis=0) if normalize else M[idx].mean(axis=0)
    alpha = (100.0 - ci) / 2.0
    lo = np.percentile(boot, alpha, axis=0)
    hi = np.percentile(boot, 100.0 - alpha, axis=0)
    return lo, hi


def _realization_curves(counts_df, gwl, zone, x_int, normalize=True):
    """
    Every individual (GCM, run) realization's own curve for one (gwl,
    zone) -- the raw per-realization counterpart to _group_counts's
    equal-GCM-weighted pooled line, used only by plot_distributions'
    uncertainty='realizations' mode, which draws each one as a thin
    spaghetti-plot line instead of a bootstrap CI band -- so individual-
    model (and individual-run) spread is visible directly rather than
    summarized into a resampled envelope. Unlike _bootstrap_band_from_counts
    and _group_counts, runs of the same GCM are *not* averaged together
    first: each run is its own line, since the point here is to see every
    realization, not a per-GCM summary.

    normalize=True: each run's own count at duration d divided by that
    run's own total count across its *full* duration record on file (not
    cropped to x_int) -- same per-realization share definition arr_share
    averages across GCMs, just left un-averaged here. normalize=False: raw
    per-run event count, unscaled.

    Returns [(gcm, run, curve_array), ...] -- curve_array the same length
    as x_int -- one entry per realization with at least one event on
    record in this (gwl, zone).
    """
    sub = counts_df[(counts_df["gwl"] == gwl) & (counts_df["zone"] == zone)]
    curves = []
    for (gcm, run), gsub in sub.groupby(["GCM", "run"]):
        hist = gsub.set_index("duration")["count"]
        vals = np.array([hist.get(d, 0.0) for d in x_int], dtype=float)
        if normalize:
            total = float(hist.sum())
            if total <= 0:
                continue
            curve = vals / total
        else:
            curve = vals
        curves.append((gcm, run, curve))
    return curves


def _add_locator_map(fig, gs_column, zone_order):
    """
    Small PlateCarree map spanning the analysis band (MAP_LAT_SOUTH ..
    MAP_LAT_NORTH), shaded and outlined at each zone boundary, occupying the
    whole left-hand gridspec column (gs_column = gs[:, 0]) so it lines up
    vertically with the stacked distribution panels in the right-hand
    column. Returns (ax_map, {zone_label: lat_mid}) -- the latter consumed by
    the caller to draw the connector lines to each panel.
    """
    ax_map = fig.add_subplot(gs_column, projection=ccrs.PlateCarree())
    ax_map.set_extent([MAP_LON_WEST, MAP_LON_EAST, MAP_LAT_SOUTH - 2, MAP_LAT_NORTH + 2],
                       crs=ccrs.PlateCarree())
    # Left-anchored within its gridspec column: PlateCarree keeps a fixed
    # aspect ratio, so cropping the longitude range (above) shrinks the
    # rendered map -- anchoring it to the column's left edge (rather than
    # the default centering) pushes all of that freed-up width to the map's
    # right, directly adjoining the distribution-panel column.
    ax_map.set_anchor("W")
    ax_map.coastlines(resolution="110m", linewidth=0.3, color="#444444", zorder=3)

    lat_mid = {}
    for zlabel in zone_order:
        lo, hi = ZONE_BOUNDS[zlabel]
        lat_mid[zlabel] = (lo + hi) / 2.0
        ax_map.axhspan(lo, hi, facecolor=ZONE_MAP_COLORS[zlabel], alpha=0.35, zorder=1)
    for edge in LAT_ZONE_EDGES:
        ax_map.axhline(edge, color="black", linewidth=0.5, zorder=2)

    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    return ax_map, lat_mid


def _add_locator_map_pixels(fig, gs_column, zone_order, lat, lon, land_mask):
    """
    Same footprint, extent and coastlines as _add_locator_map (see its
    docstring for the left-anchoring rationale), used by plot_return_periods
    instead: rather than a flat per-zone axhspan color fill, scatters the
    actual ERA5 reference grid's land pixels (lat, lon, land_mask -- see
    load_era5_reference_grid/build_land_mask_from_grid), each colored by its
    zone. This is the concrete picture behind why the return-period figure
    needs a *grid* at all (see _group_return_period /
    compute_gcm_zone_pixel_counts): the same per-pixel exposure idea shown
    here for ERA5 is what's computed per-GCM, at each GCM's own native
    resolution, to normalize its event counts before averaging across GCMs.
    Horizontal lines still mark the zone edges for orientation. Returns
    (ax_map, {zone_label: lat_mid}) like _add_locator_map.
    """
    ax_map = fig.add_subplot(gs_column, projection=ccrs.PlateCarree())
    ax_map.set_extent([MAP_LON_WEST, MAP_LON_EAST, MAP_LAT_SOUTH - 2, MAP_LAT_NORTH + 2],
                       crs=ccrs.PlateCarree())
    ax_map.set_anchor("W")
    ax_map.coastlines(resolution="110m", linewidth=0.3, color="#444444", zorder=3)

    lon2d, lat2d = np.meshgrid(lon, lat)
    zone_of_row = assign_lat_zone(lat)

    lat_mid = {}
    for zlabel in zone_order:
        lo, hi = ZONE_BOUNDS[zlabel]
        lat_mid[zlabel] = (lo + hi) / 2.0
        pts_mask = (zone_of_row == zlabel)[:, None] & land_mask
        ax_map.scatter(lon2d[pts_mask], lat2d[pts_mask], s=1.5, marker="s",
                        color=ZONE_MAP_COLORS[zlabel], alpha=0.85, linewidths=0,
                        transform=ccrs.PlateCarree(), zorder=2)
    for edge in LAT_ZONE_EDGES:
        ax_map.axhline(edge, color="black", linewidth=0.5, zorder=3)

    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    return ax_map, lat_mid


def plot_distributions(counts_df, land_area_pct, gwl_list, output_path, dpi=300,
                        max_duration_days=None, normalize=True, uncertainty=None,
                        n_boot=500, ci=90, min_events=5):
    """
    normalize=True plots each duration's share of that (zone, GWL) group's
    GCM-weighted total events (sums to 1); normalize=False plots the
    GCM-weighted event count itself, so a GWL with more events overall
    visibly sits above one with fewer, which the normalized share alone
    cannot show. Both are pooled with equal GCM weighting (_group_counts),
    not a plain sum over every (GCM, run) realization, so a heavily-resampled
    GCM (or a GWL missing some GCMs entirely) doesn't distort the result --
    see _group_counts. uncertainty=None draws only the pooled line;
    uncertainty='bootstrap' additionally shades a `ci`% envelope from
    n_boot resamples of the contributing GCMs (see
    _bootstrap_band_from_counts), each on the same equal-weighting, instead
    of drawing each realization's own line; uncertainty='realizations'
    instead draws every individual (GCM, run)'s own curve as a thin
    semi-transparent spaghetti line underneath the pooled line (see
    _realization_curves) -- meant for a single-GWL call (a full ensemble's
    worth of individually-drawn lines across every GWL would be unreadable)
    to show actual per-model/per-run spread directly rather than a
    resampled envelope. `counts_df` is the aggregated event-duration counts
    table (see build_counts_table / load_counts_cache): one row per (gwl,
    GCM, run, zone, duration) with that combination's event count.
    """
    counts_in_scope = counts_df[counts_df["gwl"].isin(gwl_list)]
    if max_duration_days is None:
        if counts_in_scope.empty:
            max_duration_days = 20.0
        else:
            by_dur = counts_in_scope.groupby("duration")["count"].sum()
            max_duration_days = float(max(
                5.0, _weighted_percentile(by_dur.index.to_numpy(), by_dur.to_numpy(), 99)))
    x_int = np.arange(1, int(np.ceil(max_duration_days)) + 1)

    # North -> south so the panel stack (top to bottom) reads the same way as
    # the locator map (north at the top) -- LAT_ZONE_LABELS itself stays
    # south -> north since that's the order np.digitize needs. Panels are
    # equal height for readability; the map's *own* y-axis (true latitude,
    # via set_extent) already shows each zone's true width, and the
    # connector lines bridge the two scales.
    zone_order = list(reversed(LAT_ZONE_LABELS))

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN))
    
    # wspace is the blank gap between the locator-map column and the
    # distribution-panel column. top/bottom are taller than that gap alone
    # would need: no fig-level title/subtitle anymore (top), and the shared
    # GWL legend now lives below the panels (bottom) instead of above them.
    # The map column's width_ratio (1.2, vs. the map's own unchanged render
    # size, fixed by MAP_LON_WEST/EAST and row height -- see
    # _add_locator_map) is bigger than the map itself needs, purely to push
    # column b's left edge further right: at 1.0 its own y-axis tick labels
    # sat close enough to overlap the map.
    gs = GridSpec(len(zone_order), 2, width_ratios=[1.2, 2.6],
                  left=0.14, right=0.97, top=0.96, bottom=0.13,
                  hspace=0.85, wspace=0.32, figure=fig)

    ax_map, lat_mid = _add_locator_map(fig, gs[:, 0], zone_order)
    ax_map.text(-0.02, 1.03, "a", transform=ax_map.transAxes,
                fontsize=LETTER_FONTSIZE, fontweight="bold")

    # Split each row into a wide main panel (duration <= DURATION_SPLIT_DAY)
    # and a narrower zoomed panel (duration > DURATION_SPLIT_DAY). Only worth
    # doing if the requested range actually extends past the split day.
    do_split = max_duration_days > DURATION_SPLIT_DAY
    split_idx = DURATION_SPLIT_DAY  # x_int[:split_idx] == days 1..DURATION_SPLIT_DAY

    dist_axes = []       # main-panel axis per row -- sharex anchor + legend handles
    dist_axes_zoom = []  # zoomed-panel axis per row (None if do_split is False)
    for i, zlabel in enumerate(zone_order):
        if do_split:
            inner_gs = GridSpecFromSubplotSpec(
                1, 2, subplot_spec=gs[i, 1], width_ratios=DURATION_ZOOM_WIDTH_RATIOS, wspace=0.08)
            ax = fig.add_subplot(inner_gs[0], sharex=dist_axes[0] if dist_axes else None)
            # No sharey: the zoomed panel autoscales to only its own (much
            # smaller) value range instead of inheriting the main panel's
            # multi-decade span, so GWL differences in the thinning tail are
            # actually visible instead of flattened against a shared axis.
            ax_zoom = fig.add_subplot(
                inner_gs[1], sharex=dist_axes_zoom[0] if dist_axes_zoom else None)
        else:
            ax = fig.add_subplot(gs[i, 1], sharex=dist_axes[0] if dist_axes else None)
            ax_zoom = None
        dist_axes.append(ax)
        dist_axes_zoom.append(ax_zoom)
        row_axes = [ax] if ax_zoom is None else [ax, ax_zoom]

        # Each panel plots only its own window's slice of the data (day
        # DURATION_SPLIT_DAY repeated at the start of the zoom slice, so the
        # line still reads as continuous across the break) -- this is what
        # lets each panel's y-autoscale reflect only its own window instead
        # of the pooled full-range data.
        if do_split:
            windows = [(ax, slice(0, split_idx)), (ax_zoom, slice(split_idx - 1, None))]
        else:
            windows = [(ax, slice(None))]

        for gwl in gwl_list:
            color = GWL_COLORS.get(gwl, "gray")
            group = _group_counts(counts_df, gwl, zlabel, x_int)
            if group is None:
                continue
            arr_share, arr_count, mean_dur, weighted_total, raw_total = group
            if raw_total < min_events:
                continue

            y_pooled = _for_line(arr_share if normalize else arr_count)
            lo = hi = None
            if uncertainty == "bootstrap":
                lo, hi = _bootstrap_band_from_counts(
                    counts_df, gwl, zlabel, x_int, normalize, n_boot=n_boot, ci=ci)

            realization_curves = None
            if uncertainty == "realizations":
                realization_curves = _realization_curves(counts_df, gwl, zlabel, x_int, normalize)

            # High-duration tail marker (q-HIGH_DURATION_PCTILE), computed
            # over this (gwl, zone)'s full duration record -- not cropped to
            # x_int/max_duration_days -- so it still marks the true tail
            # even on the two cropped figures (see _group_percentile_full).
            p_high = _group_percentile_full(counts_df, gwl, zlabel, HIGH_DURATION_PCTILE)

            for a, sl in windows:
                if lo is not None:
                    a.fill_between(x_int[sl], lo[sl], hi[sl], color=color, alpha=0.22,
                                    linewidth=0, zorder=2)
                if realization_curves:
                    for gcm, run, curve in realization_curves:
                        a.plot(x_int[sl], _for_line(curve)[sl], color=color,
                               linewidth=0.5, alpha=0.25, zorder=2)
                a.plot(x_int[sl], y_pooled[sl], color=color, marker="o", markersize=2.5,
                       linewidth=1.4, zorder=3, label=GWL_LABELS.get(gwl, gwl))
                # Drawn at each GWL's true value, even where several GWLs'
                # values coincide closely -- semi-transparent (rather than
                # nudged sideways into a falsely-more-separated position) so
                # a genuine near-overlap still reads as a visibly darker,
                # blended line: an honest signal that those GWLs are close,
                # not an invented gap between them.
                a.axvline(mean_dur, color=color, linestyle="--", linewidth=1.2,
                          alpha=0.75, zorder=4)
                if p_high is not None:
                    a.axvline(p_high, color=color, linestyle=":", linewidth=1.1,
                              alpha=0.75, zorder=4)

        # "Share of events" is the same quantity in every row -- only the
        # top panel spells it out; the rest keep just their tick numbers, so
        # the narrower (square-figure) panels aren't spending width on five
        # repeats of the same label.
        base_label = "Share of events" if normalize else "Mean events per GCM"
        if i == 0:
            ax.set_ylabel(base_label, fontsize=AXIS_LABEL_FONTSIZE)
        # Panel letter directly beside the zone's land-area share (not a
        # separate corner label, and not the zone name -- the zone-coloured
        # spine/connector line back to the locator map already identifies
        # which band this row is) -- built from two ax.text calls rather than
        # set_title so the letter (black) and share (zone-coloured) can carry
        # different colors on the same line. The land-area share folds into
        # this same line (rather than the ylabel) so it survives even on rows
        # with no ylabel text.
        pct = land_area_pct.get(zlabel)
        lat_range_text = _fmt_lat_range(zlabel)
        zone_label_text = (f"{lat_range_text} ({pct:.1f}% of land area)" if pct is not None
                            else lat_range_text)
        ax.text(0.0, 1.03, chr(ord("b") + i), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=LETTER_FONTSIZE, fontweight="bold")
        # Pushed further right than the letter's own glyph width would need
        # (0.05 sat right on top of the bold letter once the main panel
        # narrowed to fit the wider zoomed panel -- DURATION_ZOOM_WIDTH_RATIOS)
        # so the two never visually merge.
        ax.text(0.14, 1.03, zone_label_text, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=ZONE_TITLE_FONTSIZE, fontweight="bold",
                color=ZONE_MAP_COLORS[zlabel])
        for a in row_axes:
            a.set_yscale("log")  # shared range applied across rows below, once every row is plotted
            a.tick_params(labelsize=TICK_FONTSIZE)
            # Unbounded integer ticks on the main panel (always the narrow
            # 1..DURATION_SPLIT_DAY window, so one tick per day reads fine),
            # but capped to a handful on the zoomed panel -- its window can
            # be as wide as the full uncropped duration range (the "_full"
            # figure), where an unbounded integer locator packs in far more
            # labels than the narrow panel has room for and they overlap.
            locator = (matplotlib.ticker.MaxNLocator(integer=True) if a is ax
                       else matplotlib.ticker.MaxNLocator(integer=True, nbins=6))
            a.xaxis.set_major_locator(locator)
            a.grid(True, linestyle="--", alpha=0.3)
            for spine in a.spines.values():
                spine.set_linewidth(0.4)

        if do_split:
            ax.axvline(DURATION_SPLIT_DAY, color=DURATION_SPLIT_LINE_COLOR,
                       linewidth=0.8, linestyle="-", zorder=1)
            ax.set_xlim(0.5, DURATION_SPLIT_DAY + 0.5)
            ax_zoom.set_xlim(DURATION_SPLIT_DAY - 0.5, max_duration_days + 0.5)
            # The zoomed panel has its own (tighter) y-range now, not the
            # main panel's -- moved to the right edge (rather than hidden)
            # so its scale reads as clearly distinct from the main panel's
            # left-side labels instead of looking like a missing duplicate.
            # Only label the power-of-ten major ticks: with a narrow
            # autoscaled range matplotlib often falls back to also labeling
            # in-between minor ticks (2e-2, 6e-3, ...), which reads as
            # cluttered next to a plain "10^n" scale.
            ax_zoom.yaxis.tick_right()
            ax_zoom.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.set_xlim(0.5, max_duration_days + 0.5)

        # Colour-coded tab on the panel's own left edge, plus a dashed
        # connector back to this zone's band on the locator map, so the
        # correspondence is explicit rather than relying on stacking order.
        ax.spines["left"].set_color(ZONE_MAP_COLORS[zlabel])
        ax.spines["left"].set_linewidth(2.5)
        con = ConnectionPatch(
            xyA=(MAP_LON_EAST, lat_mid[zlabel]), coordsA=ax_map.transData,
            xyB=(0, 0.5), coordsB=ax.transAxes,
            color=ZONE_MAP_COLORS[zlabel], linewidth=0.9, linestyle="--",
            alpha=0.85, zorder=1,
        )
        fig.add_artist(con)

    # Same y-axis scale across every latitude-band row, so panel-to-panel
    # amplitude differences are directly comparable instead of each row
    # autoscaling to its own data -- shared separately per panel type (main
    # vs. zoomed), since those two windows' value ranges differ by design
    # (see module docstring). Applied only now, once every row has been
    # plotted and each axis's own autoscaled range is known.
    # Rows with no plotted line at all (every GWL skipped for min_events)
    # keep whatever arbitrary default range matplotlib assigns an empty log
    # axis, which would otherwise badly skew the shared min/max -- excluded
    # from the range calculation, though they still get the final shared
    # range applied like every other row for a consistent look.
    main_ylims = [a.get_ylim() for a in dist_axes if a.get_lines()]
    if main_ylims:
        y_lo_shared = min(y[0] for y in main_ylims)
        y_hi_shared = max(y[1] for y in main_ylims)
        for a in dist_axes:
            a.set_ylim(y_lo_shared, y_hi_shared)

    if do_split:
        zoom_ylims = [a.get_ylim() for a in dist_axes_zoom if a.get_lines()]
        if zoom_ylims:
            y_lo_zoom_shared = min(y[0] for y in zoom_ylims)
            y_hi_zoom_shared = max(y[1] for y in zoom_ylims)
            for a in dist_axes_zoom:
                a.set_ylim(y_lo_zoom_shared, y_hi_zoom_shared)

        # Visual bridge between the two windows, drawn only now that every
        # row shares its final y-range: dotted lines from the
        # day-DURATION_SPLIT_DAY level in the main panel (matched to the
        # zoomed panel's own y-max/y-min, i.e. where its curves start/end)
        # to the top-left and bottom-left corners of the zoomed panel, so
        # the break reads as a continuation rather than two unrelated plots
        # -- plus the classic diagonal hash marks on both spines at the
        # break itself, the standard "broken axis" convention, as an extra
        # unambiguous sign of a discontinuity.
        for ax, ax_zoom in zip(dist_axes, dist_axes_zoom):
            y_lo_zoom, y_hi_zoom = ax_zoom.get_ylim()
            y_lo_main, y_hi_main = sorted(ax.get_ylim())
            for y_zoom, y_frac_zoom in ((y_hi_zoom, 1), (y_lo_zoom, 0)):
                y_anchor = min(max(y_zoom, y_lo_main), y_hi_main)
                zoom_link = ConnectionPatch(
                    xyA=(DURATION_SPLIT_DAY, y_anchor), coordsA=ax.transData,
                    xyB=(0, y_frac_zoom), coordsB=ax_zoom.transAxes,
                    color=DURATION_SPLIT_LINE_COLOR, linewidth=0.7, linestyle=":", zorder=1,
                )
                fig.add_artist(zoom_link)

            d = 0.02  # half-length (axes fraction) of each diagonal break mark
            break_kwargs = dict(color="black", linewidth=0.8, clip_on=False, zorder=5)
            for y0 in (0, 1):
                ax.plot((1 - d, 1 + d), (y0 - d, y0 + d), transform=ax.transAxes, **break_kwargs)
                ax_zoom.plot((-d, d), (y0 - d, y0 + d), transform=ax_zoom.transAxes, **break_kwargs)

    dist_axes[-1].set_xlabel("WSED event duration (days)", fontsize=XLABEL_FONTSIZE)
    handles, labels = dist_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(gwl_list), fontsize=LEGEND_FONTSIZE,
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_return_periods(counts_df, gcm_pixel_counts, era5_lat, era5_lon, era5_land_mask,
                         land_area_pct, gwl_list, output_path, dpi=300,
                         max_duration_days=None, min_exceed=10):
    """
    "Return period" companion to plot_distributions: per latitude zone,
    T(d) = expected years between duration->=d events at a single land
    pixel (see _group_return_period, area-weighted on its pixel-count
    denominator via compute_gcm_zone_pixel_counts, unweighted on its
    exceedance-count numerator since that comes straight from the existing
    counts_df cache), instead of plot_distributions' normalized
    share-of-events. Same locator-map + connector-line + colored-spine
    layout as plot_distributions, except the map (see
    _add_locator_map_pixels) scatters the ERA5 reference grid's actual land
    pixels (era5_lat/era5_lon/era5_land_mask -- see
    load_era5_reference_grid/build_land_mask_from_grid) instead of flat
    per-zone color bands -- a concrete picture of the per-pixel exposure
    idea the return-period denominator is built on. Panel titles carry each
    zone's latitude range plus its ERA5 land-area share (land_area_pct --
    see compute_land_area_share_per_zone), not its plain-language name,
    since the map and the connector/spine color already identify it -- same
    convention as plot_distributions' zone_label_text. No main/zoom panel
    split (unlike plot_distributions) -- one log-scale panel per zone.
    """
    counts_in_scope = counts_df[counts_df["gwl"].isin(gwl_list)]
    if max_duration_days is None:
        if counts_in_scope.empty:
            max_duration_days = 20.0
        else:
            by_dur = counts_in_scope.groupby("duration")["count"].sum()
            max_duration_days = float(max(
                5.0, _weighted_percentile(by_dur.index.to_numpy(), by_dur.to_numpy(), 99)))
    x_int = np.arange(1, int(np.ceil(max_duration_days)) + 1)

    # North -> south, same stacking convention as plot_distributions.
    zone_order = list(reversed(LAT_ZONE_LABELS))

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN))
    gs = GridSpec(len(zone_order), 2, width_ratios=[1.2, 2.6],
                  left=0.14, right=0.97, top=0.96, bottom=0.13,
                  hspace=0.85, wspace=0.32, figure=fig)

    ax_map, lat_mid = _add_locator_map_pixels(
        fig, gs[:, 0], zone_order, era5_lat, era5_lon, era5_land_mask)
    ax_map.text(-0.02, 1.03, "a", transform=ax_map.transAxes,
                fontsize=LETTER_FONTSIZE, fontweight="bold")

    dist_axes = []
    for i, zlabel in enumerate(zone_order):
        ax = fig.add_subplot(gs[i, 1], sharex=dist_axes[0] if dist_axes else None)
        dist_axes.append(ax)

        for gwl in gwl_list:
            color = GWL_COLORS.get(gwl, "gray")
            T = _group_return_period(
                counts_df, gwl, zlabel, gcm_pixel_counts, x_int, min_exceed=min_exceed)
            if T is None:
                continue
            ax.plot(x_int, T, color=color, marker="o", markersize=2.5, linewidth=1.4,
                    zorder=3, label=GWL_LABELS.get(gwl, gwl))

        ax.set_yscale("log")
        # Plain "1"/"10"/"100" tick labels instead of matplotlib's default
        # log-scale "10^0"/"10^1"/"10^2" scientific notation.
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        if i == 0:
            ax.set_ylabel("Return period\n(years)", fontsize=AXIS_LABEL_FONTSIZE)
        pct = land_area_pct.get(zlabel)
        lat_range_text = _fmt_lat_range(zlabel)
        zone_label_text = (f"{lat_range_text} ({pct:.1f}% of land area)" if pct is not None
                            else lat_range_text)
        ax.text(0.0, 1.03, chr(ord("b") + i), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=LETTER_FONTSIZE, fontweight="bold")
        ax.text(0.14, 1.03, zone_label_text, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=ZONE_TITLE_FONTSIZE, fontweight="bold",
                color=ZONE_MAP_COLORS[zlabel])
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        ax.set_xlim(0.5, max_duration_days + 0.5)

        ax.spines["left"].set_color(ZONE_MAP_COLORS[zlabel])
        ax.spines["left"].set_linewidth(2.5)
        con = ConnectionPatch(
            xyA=(MAP_LON_EAST, lat_mid[zlabel]), coordsA=ax_map.transData,
            xyB=(0, 0.5), coordsB=ax.transAxes,
            color=ZONE_MAP_COLORS[zlabel], linewidth=0.9, linestyle="--",
            alpha=0.85, zorder=1,
        )
        fig.add_artist(con)

    # Same y-axis scale across every latitude-band row, so panel-to-panel
    # return-period differences are directly comparable instead of each row
    # autoscaling to its own data (same convention as plot_distributions'
    # main_ylims). Rows with no plotted line at all (every GWL skipped, e.g.
    # for lack of a usable pixel count) are excluded from the range
    # calculation but still get the shared range applied.
    ylims = [a.get_ylim() for a in dist_axes if a.get_lines()]
    if ylims:
        y_lo_shared = min(y[0] for y in ylims)
        y_hi_shared = max(y[1] for y in ylims)
        for a in dist_axes:
            a.set_ylim(y_lo_shared, y_hi_shared)

    dist_axes[-1].set_xlabel("WSED event duration (days)", fontsize=XLABEL_FONTSIZE)
    handles, labels = dist_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(gwl_list), fontsize=LEGEND_FONTSIZE,
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_gwl_uncertainty_check(counts_df, gcm_pixel_counts, era5_lat, era5_lon, era5_land_mask,
                                land_area_pct, gwl, output_path, dpi=300,
                                max_duration_days=None, min_exceed=10):
    """
    Single-GWL companion to plot_return_periods: per zone, the equal-GCM-
    weighted pooled return-period curve (_group_return_period, drawn on top
    in the GWL's own color) against every individual GCM's own curve
    (_return_period_rates_by_gcm -- one coherent GCM's full curve each, not
    a per-duration min/max envelope across different GCMs), drawn thin and
    unlabeled in light red behind it so the full spread is visible at a
    glance without a legend entry per GCM. Meant as a quick single-model
    spread check for one GWL at a time -- typically GWL3, this project's
    smallest/most uncertain ensemble (as few as 8 GCMs / 18 runs vs ~14
    GCMs / 33 runs at the other GWLs) -- not a substitute for the bootstrap
    CI already drawn on fig_duration_distribution_by_latitude_share_bootstrap.png.
    Same map/panel layout as plot_return_periods, restricted to one GWL.
    """
    counts_in_scope = counts_df[counts_df["gwl"] == gwl]
    if max_duration_days is None:
        if counts_in_scope.empty:
            max_duration_days = 20.0
        else:
            by_dur = counts_in_scope.groupby("duration")["count"].sum()
            max_duration_days = float(max(
                5.0, _weighted_percentile(by_dur.index.to_numpy(), by_dur.to_numpy(), 99)))
    x_int = np.arange(1, int(np.ceil(max_duration_days)) + 1)

    zone_order = list(reversed(LAT_ZONE_LABELS))
    color = GWL_COLORS.get(gwl, "gray")

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN))
    gs = GridSpec(len(zone_order), 2, width_ratios=[1.2, 2.6],
                  left=0.14, right=0.97, top=0.96, bottom=0.13,
                  hspace=0.85, wspace=0.32, figure=fig)

    ax_map, lat_mid = _add_locator_map_pixels(
        fig, gs[:, 0], zone_order, era5_lat, era5_lon, era5_land_mask)
    ax_map.text(-0.02, 1.03, "a", transform=ax_map.transAxes,
                fontsize=LETTER_FONTSIZE, fontweight="bold")

    dist_axes = []
    for i, zlabel in enumerate(zone_order):
        ax = fig.add_subplot(gs[i, 1], sharex=dist_axes[0] if dist_axes else None)
        dist_axes.append(ax)

        by_gcm = _return_period_rates_by_gcm(counts_df, gwl, zlabel, gcm_pixel_counts, x_int)
        for rate, exceed_raw in by_gcm.values():
            T_g = np.where(rate > 0, 1.0 / rate, np.nan)
            T_g = np.where(exceed_raw >= min_exceed, T_g, np.nan)
            if np.all(np.isnan(T_g)):
                continue
            ax.plot(x_int, T_g, color="lightcoral", linewidth=0.6, alpha=0.5, zorder=2)

        T_pooled = _group_return_period(
            counts_df, gwl, zlabel, gcm_pixel_counts, x_int, min_exceed=min_exceed)
        if T_pooled is not None:
            ax.plot(x_int, T_pooled, color=color, marker="o", markersize=2.5,
                    linewidth=1.6, zorder=4)

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        if i == 0:
            ax.set_ylabel("Return period\n(years)", fontsize=AXIS_LABEL_FONTSIZE)

        pct = land_area_pct.get(zlabel)
        lat_range_text = _fmt_lat_range(zlabel)
        zone_label_text = (f"{lat_range_text} ({pct:.1f}% of land area)" if pct is not None
                            else lat_range_text)
        ax.text(0.0, 1.03, chr(ord("b") + i), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=LETTER_FONTSIZE, fontweight="bold")
        ax.text(0.14, 1.03, zone_label_text, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=ZONE_TITLE_FONTSIZE, fontweight="bold",
                color=ZONE_MAP_COLORS[zlabel])
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        ax.set_xlim(0.5, max_duration_days + 0.5)

        ax.spines["left"].set_color(ZONE_MAP_COLORS[zlabel])
        ax.spines["left"].set_linewidth(2.5)
        con = ConnectionPatch(
            xyA=(MAP_LON_EAST, lat_mid[zlabel]), coordsA=ax_map.transData,
            xyB=(0, 0.5), coordsB=ax.transAxes,
            color=ZONE_MAP_COLORS[zlabel], linewidth=0.9, linestyle="--",
            alpha=0.85, zorder=1,
        )
        fig.add_artist(con)

    ylims = [a.get_ylim() for a in dist_axes if a.get_lines()]
    if ylims:
        y_lo_shared = min(y[0] for y in ylims)
        y_hi_shared = max(y[1] for y in ylims)
        for a in dist_axes:
            a.set_ylim(y_lo_shared, y_hi_shared)

    dist_axes[-1].set_xlabel("WSED event duration (days)", fontsize=XLABEL_FONTSIZE)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1 - Land area share per latitude zone (ERA5 reference grid)")
    print("=" * 60)
    era5_lat, era5_lon = load_era5_reference_grid(
        args.preprocessed_path, era5_grid_path=args.era5_grid_path)
    era5_land_mask = build_land_mask_from_grid(era5_lat, era5_lon, args.shapefile)
    land_area_pct = compute_land_area_share_per_zone(era5_lat, era5_lon, args.shapefile)
    for z, pct in land_area_pct.items():
        print(f"  {z}: {pct:.1f}% of land area")

    print("\n" + "=" * 60)
    print("STEP 2 - Event-duration counts per zone/GCM/run (cached)")
    print("=" * 60)
    cache_path = args.cache_csv or os.path.join(
        args.output_dir, "event_duration_counts_cache.csv")
    counts_df = None if args.recompute else load_counts_cache(
        cache_path, args.threshold, args.ssp)
    if counts_df is not None:
        print(f"  Loaded cached counts from {cache_path} "
              f"({len(counts_df)} rows) -- skipping the expensive rebuild.")
        missing_gwl = set(args.gwl_list) - set(counts_df["gwl"].unique())
        if missing_gwl:
            print(f"  [warn] cache has no rows for {sorted(missing_gwl)} -- "
                  "pass --recompute if these GWLs should have data.")
    else:
        counts_df = build_counts_table(
            args.preprocessed_path, args.gwl_list, args.ssp, args.threshold,
            args.shapefile, args.exclude_gcm, args.exclude_gcm_run,
        )
        save_counts_cache(counts_df, cache_path, args.threshold, args.ssp)
        print(f"  Saved counts cache -> {cache_path}")

    for gwl in args.gwl_list:
        sub = counts_df[counts_df["gwl"] == gwl]
        n_sims = sub[["GCM", "run"]].drop_duplicates().shape[0]
        print(f"  {gwl}: {int(sub['count'].sum())} events from {n_sims} GCM-run realizations")

    print("\n" + "=" * 60)
    print("STEP 3 - Plotting")
    print("=" * 60)

    out_share = os.path.join(args.output_dir, "fig_duration_distribution_by_latitude_share.png")
    plot_distributions(
        counts_df, land_area_pct, args.gwl_list, out_share, dpi=args.dpi,
        max_duration_days=args.max_duration_days, normalize=True, uncertainty=None,
        min_events=args.min_events,
    )
    print(f"  Saved -> {out_share}")

    out_share_boot = os.path.join(
        args.output_dir, "fig_duration_distribution_by_latitude_share_bootstrap.png")
    plot_distributions(
        counts_df, land_area_pct, args.gwl_list, out_share_boot, dpi=args.dpi,
        max_duration_days=args.max_duration_days, normalize=True, uncertainty="bootstrap",
        n_boot=args.n_boot, ci=args.ci, min_events=args.min_events,
    )
    print(f"  Saved -> {out_share_boot}")

    # Same pooled-line figure as (1) plus the same bootstrap band as (2), but
    # with the x-axis extended well past the 99th-percentile default --
    # capped at FULL_FIGURE_MAX_DURATION_DAYS rather than the true longest
    # duration on record, which can run into the hundreds of days and crowd
    # the zoomed panel past legibility.
    out_share_full = os.path.join(
        args.output_dir, "fig_duration_distribution_by_latitude_share_full.png")
    plot_distributions(
        counts_df, land_area_pct, args.gwl_list, out_share_full, dpi=args.dpi,
        max_duration_days=FULL_FIGURE_MAX_DURATION_DAYS, normalize=True, uncertainty="bootstrap",
        n_boot=args.n_boot, ci=args.ci, min_events=args.min_events,
    )
    print(f"  Saved -> {out_share_full}")

    print("\n" + "=" * 60)
    print("STEP 4 - Return-period figure (area-weighted pixel-count denominator, "
          "unweighted cached exceedance counts)")
    print("=" * 60)
    grids = discover_gcm_grids(args.raw_grid_path, args.exclude_gcm)
    gcm_pixel_counts = compute_gcm_zone_pixel_counts(grids, args.shapefile)
    for gcm, zones in gcm_pixel_counts.items():
        print(f"  {gcm}: " + ", ".join(f"{z}={n:.1f}wpx" for z, n in zones.items()))

    out_return_period = os.path.join(
        args.output_dir, "fig_duration_return_period_by_latitude.png")
    plot_return_periods(
        counts_df, gcm_pixel_counts, era5_lat, era5_lon, era5_land_mask,
        land_area_pct, args.gwl_list, out_return_period, dpi=args.dpi,
        max_duration_days=args.max_duration_days or FULL_FIGURE_MAX_DURATION_DAYS,
        min_exceed=args.min_exceed,
    )
    print(f"  Saved -> {out_return_period}")

    if args.uncertainty_gwl:
        print("\n" + "=" * 60)
        print(f"STEP 5 - {args.uncertainty_gwl} worst/best-GCM uncertainty check")
        print("=" * 60)
        out_uncertainty = os.path.join(
            args.output_dir,
            f"fig_duration_return_period_{args.uncertainty_gwl}_uncertainty.png")
        plot_gwl_uncertainty_check(
            counts_df, gcm_pixel_counts, era5_lat, era5_lon, era5_land_mask,
            land_area_pct, args.uncertainty_gwl, out_uncertainty, dpi=args.dpi,
            max_duration_days=args.max_duration_days or FULL_FIGURE_MAX_DURATION_DAYS,
            min_exceed=args.min_exceed,
        )
        print(f"  Saved -> {out_uncertainty}")

        print("\n" + "=" * 60)
        print(f"STEP 6 - {args.uncertainty_gwl} all-realizations spaghetti plot")
        print("=" * 60)
        out_realizations = os.path.join(
            args.output_dir,
            f"fig_duration_distribution_{args.uncertainty_gwl}_realizations.png")
        plot_distributions(
            counts_df, land_area_pct, [args.uncertainty_gwl], out_realizations, dpi=args.dpi,
            max_duration_days=FULL_FIGURE_MAX_DURATION_DAYS, normalize=True,
            uncertainty="realizations", min_events=args.min_events,
        )
        print(f"  Saved -> {out_realizations}")


if __name__ == "__main__":
    main()

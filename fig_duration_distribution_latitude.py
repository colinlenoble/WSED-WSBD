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
many runs or how many GCMs happen to be available (see _gcm_weights). This
project's ensemble is uneven enough that this matters a lot -- e.g. CanESM5
contributes 10 runs and MPI-ESM1-2-LR 9, out of ~33 realizations at
GWL0-61/1.5/2, while GWL3 drops to 18 realizations from only 8 GCMs (several,
including MPI-ESM1-2-LR, absent entirely). A plain sum would let those two
GCMs dominate every GWL and would make GWL3 look artificially low purely
from having fewer contributing GCMs, not from any real change in event
frequency.

Two figures are produced, both on the normalized *share* scale (each
duration's share of that (zone, GWL) group's GCM-weighted total events,
summing to 1) rather than raw/GCM-weighted counts -- shape-of-distribution
comparisons across GWLs are what these figures are for, and the share scale
keeps that comparison legible across zones whose absolute event counts differ
by orders of magnitude (see per-row independent y-axis ranges below):
  1. fig_duration_distribution_by_latitude_share.png
     Only the pooled line -- shows how the duration mix changes with
     warming.
  2. fig_duration_distribution_by_latitude_share_bootstrap.png
     Same, with a shaded confidence band from bootstrap-resampling which
     GCMs contribute (each GCM's runs pre-averaged, same weighting), instead
     of drawing each realization's own line -- see
     _bootstrap_band_from_counts.
Every figure carries a vertical dashed line at each GWL's mean duration.

Each of the 5 latitude-zone rows (panels b-f; panel a is the locator map) is
itself split into two side-by-side, independently-autoscaled log-scale
panels: a wide main panel (days 1-DURATION_SPLIT_DAY) and a narrower zoomed
panel (DURATION_SPLIT_DAY onward, to max_duration_days). Both panels plot
only their own window's data, so each y-range reflects only what's visible
there instead of the full pooled range -- this matters a lot for the zoomed
panel, whose own value range is much narrower than the main panel's, so
sharing one axis would flatten GWL differences in the tail almost to
invisibility. Row-to-row (and now panel-to-panel within a row) the y-range is
always independent, since amplitude varies strongly between e.g. the tropics
and Midlatitude (S). Day DURATION_SPLIT_DAY is marked with a thin vertical
line in the main panel where the split occurs.
"""
import os
import config
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV

import argparse
import gc
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
DURATION_ZOOM_WIDTH_RATIOS = [65, 35]
DURATION_SPLIT_LINE_COLOR = "#777777"

MAP_LAT_SOUTH = -58.0
MAP_LAT_NORTH = 68.0

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


def compute_pixel_counts_per_zone(preprocessed_path, shapefile_path):
    """
    Number of land pixels per latitude zone on the common ERA5 reference grid
    (the grid every GCM/run is ultimately compared against elsewhere in this
    project), independent of which GCM's own native grid produced the event
    durations -- so the "n pixels" shown per zone is a single, well-defined
    spatial property of the analysis domain, not a per-GCM count.
    """
    rea_files, _ = match_files(os.path.join(preprocessed_path, "ERA5", "wcf_day*"))
    if not rea_files:
        raise FileNotFoundError(
            f"No ERA5 reference file found under {os.path.join(preprocessed_path, 'ERA5')}. "
            "Cannot count reference-grid pixels per zone."
        )
    da = open_dataset_any(rea_files[0]).isel(time=0).wcf
    da = da.sortby("lat").sortby("lon")
    da = da.sel(lat=slice(MAP_LAT_SOUTH, MAP_LAT_NORTH))

    land_mask = build_land_mask_from_grid(da.lat.values, da.lon.values, shapefile_path)
    land_mask = land_mask & da.notnull().values

    zone_of_row = assign_lat_zone(da.lat.values)
    counts = {}
    for zlabel in LAT_ZONE_LABELS:
        row_sel = zone_of_row == zlabel
        counts[zlabel] = int(land_mask[row_sel, :].sum())
    return counts


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


def _gcm_weights(sub):
    """
    Per-row weight giving each GCM equal total weight (1 / n_GCM), split
    evenly across however many runs that GCM happens to contribute -- same
    inverse-run-count convention as fig3.py's align_realizations_and_weight /
    add_severity_and_weights. Without this, a GCM sampled with many runs
    (e.g. CanESM5's 10 runs vs. most GCMs' single run in this project's
    ensemble) dominates a plain pooled sum, and a GWL missing some GCMs
    entirely (e.g. GWL3 dropping from 14 to 8 GCMs once MPI-ESM1-2-LR and
    others drop out) ends up on a different, non-comparable scale purely
    from having fewer contributors -- not from any real change in event
    frequency. Weights always sum to 1 over `sub`.
    """
    runs_per_gcm = sub.groupby("GCM")["run"].transform("nunique")
    n_gcm = sub["GCM"].nunique()
    return 1.0 / (runs_per_gcm * n_gcm)


def _group_counts(counts_df, gwl, zone, x_int):
    """
    (count_array over x_int, mean_duration, weighted_total, raw_total) for
    one (gwl, zone), pooled with equal GCM weighting (_gcm_weights) rather
    than a plain sum over every contributing (GCM, run) realization -- see
    _gcm_weights for why. Returns None if there is no data for this
    (gwl, zone).

    `weighted_total` is on a "mean events per GCM" scale (weights sum to 1)
    and is what the plotted arr/mean_dur are normalized against; it covers
    every duration on record, not just those within x_int, so a normalized
    share computed from it can legitimately sum to less than 1 over x_int
    alone (see module docstring). `raw_total` is the true pooled event
    count, kept separately only to gate min_events on actual sample size
    rather than the reweighted scale.
    """
    sub = counts_df[(counts_df["gwl"] == gwl) & (counts_df["zone"] == zone)]
    if sub.empty:
        return None
    w = _gcm_weights(sub)
    agg = (sub["count"] * w).groupby(sub["duration"]).sum()
    weighted_total = float(agg.sum())
    raw_total = float(sub["count"].sum())
    arr = np.array([agg.get(d, 0) for d in x_int], dtype=float)
    mean_dur = (float((agg.index.to_numpy() * agg.to_numpy()).sum() / weighted_total)
                if weighted_total > 0 else np.nan)
    return arr, mean_dur, weighted_total, raw_total


def _bootstrap_band_from_counts(counts_df, gwl, zone, x_int, normalize, n_boot=500, ci=90, rng=None):
    """
    (lo, hi) envelope at each integer duration in x_int from resampling
    *GCMs* (not raw realizations) with replacement, n_boot times, each GCM's
    own runs averaged together first -- the same equal-GCM-weighting as the
    pooled line in _group_counts (see _gcm_weights). Resampling raw
    (GCM, run) pairs instead would let a heavily-resampled GCM dominate the
    bootstrap draws too, not just the pooled sum, and would understate
    uncertainty. Works directly off the aggregated counts table (no raw
    per-event data needed). Returns (None, None) if fewer than 2 GCMs are
    available.
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

    rng = rng if rng is not None else np.random.default_rng(12345)
    boot = np.empty((n_boot, len(x_int)))
    for b in range(n_boot):
        idx = rng.integers(0, n_gcm, size=n_gcm)
        arr = M[idx].mean(axis=0)
        if normalize:
            tot = totals[idx].mean()
            arr = arr / tot if tot > 0 else arr
        boot[b] = arr
    alpha = (100.0 - ci) / 2.0
    lo = np.percentile(boot, alpha, axis=0)
    hi = np.percentile(boot, 100.0 - alpha, axis=0)
    return lo, hi


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
    ax_map.set_extent([-180, 180, MAP_LAT_SOUTH - 2, MAP_LAT_NORTH + 2], crs=ccrs.PlateCarree())
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


def plot_distributions(counts_df, pixel_counts, gwl_list, output_path, dpi=300,
                        max_duration_days=None, normalize=True, uncertainty=None,
                        n_boot=500, ci=90, min_events=5):
    """
    normalize=True plots each duration's share of that (zone, GWL) group's
    GCM-weighted total events (sums to 1); normalize=False plots the
    GCM-weighted event count itself, so a GWL with more events overall
    visibly sits above one with fewer, which the normalized share alone
    cannot show. Both are pooled with equal GCM weighting (_group_counts /
    _gcm_weights), not a plain sum over every (GCM, run) realization, so a
    heavily-resampled GCM (or a GWL missing some GCMs entirely) doesn't
    distort the result -- see _gcm_weights. uncertainty=None draws only the
    pooled line; uncertainty='bootstrap' additionally shades a `ci`%
    envelope from n_boot resamples of the contributing GCMs (see
    _bootstrap_band_from_counts), each on the same equal-weighting, instead
    of drawing each realization's own line. `counts_df` is the aggregated
    event-duration counts table (see build_counts_table / load_counts_cache):
    one row per (gwl, GCM, run, zone, duration) with that combination's
    event count.
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
    # distribution-panel column -- kept small so the panels (and their
    # xticks) get as much of the figure width as possible. top/bottom are
    # taller than that gap alone would need: no fig-level title/subtitle
    # anymore (top), and the shared GWL legend now lives below the panels
    # (bottom) instead of above them.
    gs = GridSpec(len(zone_order), 2, width_ratios=[1.0, 2.6],
                  left=0.14, right=0.97, top=0.96, bottom=0.13,
                  hspace=0.5, wspace=0.15, figure=fig)

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

        for gwl in gwl_list:
            color = GWL_COLORS.get(gwl, "gray")
            group = _group_counts(counts_df, gwl, zlabel, x_int)
            if group is None:
                continue
            arr, mean_dur, weighted_total, raw_total = group
            if raw_total < min_events:
                continue

            y_pooled = _for_line(arr / weighted_total if normalize else arr)
            lo = hi = None
            if uncertainty == "bootstrap":
                lo, hi = _bootstrap_band_from_counts(
                    counts_df, gwl, zlabel, x_int, normalize, n_boot=n_boot, ci=ci)

            if do_split:
                # Each panel plots only its own window's slice (day
                # DURATION_SPLIT_DAY repeated at the start of the zoom slice,
                # so the line still reads as continuous across the break) --
                # this is what lets each panel's y-autoscale reflect only its
                # own window instead of the pooled full-range data.
                windows = [(ax, slice(0, split_idx)), (ax_zoom, slice(split_idx - 1, None))]
            else:
                windows = [(ax, slice(None))]

            for a, sl in windows:
                if lo is not None:
                    a.fill_between(x_int[sl], lo[sl], hi[sl], color=color, alpha=0.22,
                                    linewidth=0, zorder=2)
                a.plot(x_int[sl], y_pooled[sl], color=color, marker="o", markersize=2.5,
                       linewidth=1.4, zorder=3, label=GWL_LABELS.get(gwl, gwl))
                a.axvline(mean_dur, color=color, linestyle="--", linewidth=1.2, zorder=4)

        # "Share of events" is the same quantity in every row -- only the
        # top panel spells it out; the rest keep just their tick numbers, so
        # the narrower (square-figure) panels aren't spending width on five
        # repeats of the same label.
        base_label = "Share of events" if normalize else "Mean events per GCM"
        if i == 0:
            ax.set_ylabel(base_label, fontsize=AXIS_LABEL_FONTSIZE)
        # Panel letter directly beside the zone name (not a separate corner
        # label) -- built from two ax.text calls rather than set_title so the
        # letter (black) and zone name (zone-coloured) can carry different
        # colors on the same line. Pixel count folds into this same line
        # (rather than the ylabel) so it survives even on rows with no
        # ylabel text.
        n_px = pixel_counts.get(zlabel)
        zone_label_text = f"{zlabel} (n={n_px:,} px)" if n_px is not None else zlabel
        ax.text(0.0, 1.03, chr(ord("b") + i), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=LETTER_FONTSIZE, fontweight="bold")
        ax.text(0.05, 1.03, zone_label_text, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=ZONE_TITLE_FONTSIZE, fontweight="bold",
                color=ZONE_MAP_COLORS[zlabel])
        for a in row_axes:
            a.set_yscale("log")  # independent per panel now (no sharey) -- see ax_zoom comment above
            a.tick_params(labelsize=TICK_FONTSIZE)
            a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
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

            # Visual bridge between the two windows: dotted lines from the
            # day-DURATION_SPLIT_DAY level in the main panel (matched to the
            # zoomed panel's own y-max/y-min, i.e. where its curves start/
            # end) to the top-left and bottom-left corners of the zoomed
            # panel, so the break reads as a continuation rather than two
            # unrelated plots -- plus the classic diagonal hash marks on
            # both spines at the break itself, the standard "broken axis"
            # convention, as an extra unambiguous sign of a discontinuity.
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
        else:
            ax.set_xlim(0.5, max_duration_days + 0.5)

        # Colour-coded tab on the panel's own left edge, plus a dashed
        # connector back to this zone's band on the locator map, so the
        # correspondence is explicit rather than relying on stacking order.
        ax.spines["left"].set_color(ZONE_MAP_COLORS[zlabel])
        ax.spines["left"].set_linewidth(2.5)
        con = ConnectionPatch(
            xyA=(180, lat_mid[zlabel]), coordsA=ax_map.transData,
            xyB=(0, 0.5), coordsB=ax.transAxes,
            color=ZONE_MAP_COLORS[zlabel], linewidth=0.9, linestyle="--",
            alpha=0.85, zorder=1,
        )
        fig.add_artist(con)

    dist_axes[-1].set_xlabel("WSED event duration (days)", fontsize=XLABEL_FONTSIZE)
    handles, labels = dist_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(gwl_list), fontsize=LEGEND_FONTSIZE,
               bbox_to_anchor=(0.5, 0.0), frameon=False)
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
    print("STEP 1 - Counting land pixels per latitude zone (ERA5 reference grid)")
    print("=" * 60)
    pixel_counts = compute_pixel_counts_per_zone(args.preprocessed_path, args.shapefile)
    for z, n in pixel_counts.items():
        print(f"  {z}: {n} pixels")

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
        counts_df, pixel_counts, args.gwl_list, out_share, dpi=args.dpi,
        max_duration_days=args.max_duration_days, normalize=True, uncertainty=None,
        min_events=args.min_events,
    )
    print(f"  Saved -> {out_share}")

    out_share_boot = os.path.join(
        args.output_dir, "fig_duration_distribution_by_latitude_share_bootstrap.png")
    plot_distributions(
        counts_df, pixel_counts, args.gwl_list, out_share_boot, dpi=args.dpi,
        max_duration_days=args.max_duration_days, normalize=True, uncertainty="bootstrap",
        n_boot=args.n_boot, ci=args.ci, min_events=args.min_events,
    )
    print(f"  Saved -> {out_share_boot}")


if __name__ == "__main__":
    main()

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

Three figures are produced:
  1. fig_duration_distribution_by_latitude_share.png
     Each duration's *share* of that (zone, GWL) group's total events (sums
     to 1) -- shows how the duration mix changes with warming, but two GWLs
     with the same mix and different overall event counts look identical.
  2. fig_duration_distribution_by_latitude_counts.png
     Same, but the raw event count at each duration instead of its share --
     so a GWL with more events overall (a frequency change, not just a
     duration-mix change) visibly sits above one with fewer.
  3. fig_duration_distribution_by_latitude_counts_bootstrap.png
     Same as (2), with a shaded confidence band from bootstrap-resampling
     which (GCM, run) realizations contribute, instead of drawing each
     realization's own line -- see _bootstrap_band_from_counts.
Every figure carries a vertical dashed line at each GWL's mean duration.
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch

# Zarr/NetCDF-agnostic file lookup + opener, shared with calculate_cf.py /
# every other fig*.py script.
from io_utils import match_files, glob_any, open_dataset_any

# Generic event-table builder (gap-free run-length encoding of a boolean/0-1
# compound-day field into one row per event), shared with fig1.py/fig3.py's
# duration-class decomposition.
from duration_decomposition import compute_event_table

# =============================================================================
# Figure size / latitude-band constants (match fig1.py/fig3.py conventions)
# =============================================================================
FIG_WIDTH_IN = 5.15   # single column width -- fontsizes match LaTeX

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
# One distinct colour per zone (Paul Tol "bright" qualitative palette) --
# unrelated to GWL_COLORS, used only to tie each map band to its panel.
ZONE_MAP_COLORS = {
    "Midlatitude (S)": "#4477AA",
    "Subtropical (S)": "#66CCEE",
    "Tropical":        "#CCBB44",
    "Subtropical (N)": "#EE6677",
    "Midlatitude (N)": "#AA3377",
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


def _group_counts(counts_df, gwl, zone, x_int):
    """
    (count_array over x_int, mean_duration, total_count) for one (gwl, zone),
    pooled (summed) over every contributing (GCM, run) realization. Returns
    None if there is no data for this (gwl, zone). `total_count` covers every
    duration on record, not just those within x_int, so a normalized share
    computed from it can legitimately sum to less than 1 over x_int alone
    (see module docstring).
    """
    sub = counts_df[(counts_df["gwl"] == gwl) & (counts_df["zone"] == zone)]
    if sub.empty:
        return None
    agg = sub.groupby("duration")["count"].sum()
    total = float(agg.sum())
    arr = np.array([agg.get(d, 0) for d in x_int], dtype=float)
    mean_dur = float((agg.index.to_numpy() * agg.to_numpy()).sum() / total) if total > 0 else np.nan
    return arr, mean_dur, total


def _bootstrap_band_from_counts(counts_df, gwl, zone, x_int, normalize, n_boot=500, ci=90, rng=None):
    """
    (lo, hi) envelope at each integer duration in x_int from resampling
    *realizations* (GCM, run) with replacement, n_boot times -- the
    appropriate bootstrap unit here, since events within one realization are
    not independent draws but different GCM/runs plausibly are. Works
    directly off the aggregated counts table (no raw per-event data needed).
    Returns (None, None) if fewer than 2 realizations are available.
    """
    sub = counts_df[(counts_df["gwl"] == gwl) & (counts_df["zone"] == zone)]
    if sub.empty:
        return None, None
    keys = list(sub[["GCM", "run"]].drop_duplicates().itertuples(index=False, name=None))
    n_keys = len(keys)
    if n_keys < 2:
        return None, None

    key_idx = {k: i for i, k in enumerate(keys)}
    dur_idx = {d: j for j, d in enumerate(x_int)}
    M = np.zeros((n_keys, len(x_int)))
    totals = np.zeros(n_keys)
    for gcm, run, dur, cnt in zip(sub["GCM"], sub["run"], sub["duration"], sub["count"]):
        i = key_idx[(gcm, run)]
        totals[i] += cnt
        j = dur_idx.get(dur)
        if j is not None:
            M[i, j] += cnt

    rng = rng if rng is not None else np.random.default_rng(12345)
    boot = np.empty((n_boot, len(x_int)))
    for b in range(n_boot):
        idx = rng.integers(0, n_keys, size=n_keys)
        arr = M[idx].sum(axis=0)
        if normalize:
            tot = totals[idx].sum()
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
                        n_boot=500, ci=90, min_events=5, title=None, subtitle=None):
    """
    normalize=True plots each duration's share of that (zone, GWL) group's
    total events (sums to 1); normalize=False plots the raw event count, so
    a GWL with more events overall visibly sits above one with fewer, which
    the normalized share alone cannot show. uncertainty=None draws only the
    pooled line; uncertainty='bootstrap' additionally shades a `ci`%
    envelope from n_boot resamples of the contributing (GCM, run)
    realizations (see _bootstrap_band_from_counts) instead of drawing each
    realization's own line. `counts_df` is the aggregated event-duration
    counts table (see build_counts_table / load_counts_cache): one row per
    (gwl, GCM, run, zone, duration) with that combination's event count.
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

    fig = plt.figure(figsize=(FIG_WIDTH_IN * 1.9, FIG_WIDTH_IN * 1.7))
    gs = GridSpec(len(zone_order), 2, width_ratios=[1.0, 2.4],
                  left=0.14, right=0.97, top=0.85, bottom=0.08,
                  hspace=0.25, wspace=0.55, figure=fig)

    ax_map, lat_mid = _add_locator_map(fig, gs[:, 0], zone_order)

    dist_axes = []
    for i, zlabel in enumerate(zone_order):
        ax = fig.add_subplot(gs[i, 1], sharex=dist_axes[0] if dist_axes else None)
        dist_axes.append(ax)
        for gwl in gwl_list:
            color = GWL_COLORS.get(gwl, "gray")
            group = _group_counts(counts_df, gwl, zlabel, x_int)
            if group is None:
                continue
            arr, mean_dur, total = group
            if total < min_events:
                continue

            if uncertainty == "bootstrap":
                lo, hi = _bootstrap_band_from_counts(
                    counts_df, gwl, zlabel, x_int, normalize, n_boot=n_boot, ci=ci)
                if lo is not None:
                    ax.fill_between(x_int, lo, hi, color=color, alpha=0.22,
                                     linewidth=0, zorder=2)

            y_pooled = _for_line(arr / total if normalize else arr)
            ax.plot(x_int, y_pooled, color=color, marker="o", markersize=2.5,
                    linewidth=1.4, zorder=3, label=GWL_LABELS.get(gwl, gwl))
            ax.axvline(mean_dur, color=color, linestyle="--", linewidth=1.2, zorder=4)

        n_px = pixel_counts.get(zlabel)
        base_label = "Share of events" if normalize else "Number of events"
        ylabel = f"{base_label}\n(n={n_px:,} px)" if n_px is not None else base_label
        ax.set_ylabel(ylabel, fontsize=6.5)
        ax.set_title(zlabel, fontsize=7, loc="left", color=ZONE_MAP_COLORS[zlabel],
                     fontweight="bold", pad=2)
        ax.set_yscale("log")
        ax.tick_params(labelsize=6)
        if i < len(zone_order) - 1:
            ax.tick_params(labelbottom=False)
        ax.set_xlim(0.5, max_duration_days + 0.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(True, linestyle="--", alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
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

    dist_axes[-1].set_xlabel("WSED event duration (days)", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=8, y=0.995)
    if subtitle:
        fig.text(0.5, 0.955, subtitle, fontsize=6.5, ha="center", style="italic")
    handles, labels = dist_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(gwl_list), fontsize=7,
               bbox_to_anchor=(0.5, 0.92), frameon=False)
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
    base_title = "WSED event-duration distribution by latitude zone and GWL"

    out_share = os.path.join(args.output_dir, "fig_duration_distribution_by_latitude_share.png")
    plot_distributions(
        counts_df, pixel_counts, args.gwl_list, out_share, dpi=args.dpi,
        max_duration_days=args.max_duration_days, normalize=True, uncertainty=None,
        min_events=args.min_events, title=base_title,
        subtitle="(share of each group's events -- shape only, not overall frequency)",
    )
    print(f"  Saved -> {out_share}")

    out_counts = os.path.join(args.output_dir, "fig_duration_distribution_by_latitude_counts.png")
    plot_distributions(
        counts_df, pixel_counts, args.gwl_list, out_counts, dpi=args.dpi,
        max_duration_days=args.max_duration_days, normalize=False, uncertainty=None,
        min_events=args.min_events, title=base_title,
        subtitle="(raw event counts -- also reflects overall frequency differences)",
    )
    print(f"  Saved -> {out_counts}")

    out_boot = os.path.join(
        args.output_dir, "fig_duration_distribution_by_latitude_counts_bootstrap.png")
    plot_distributions(
        counts_df, pixel_counts, args.gwl_list, out_boot, dpi=args.dpi,
        max_duration_days=args.max_duration_days, normalize=False, uncertainty="bootstrap",
        n_boot=args.n_boot, ci=args.ci, min_events=args.min_events, title=base_title,
        subtitle=f"(raw event counts; shaded band = {args.ci:.0f}% bootstrap CI "
                 "over GCM-run realizations)",
    )
    print(f"  Saved -> {out_boot}")


if __name__ == "__main__":
    main()

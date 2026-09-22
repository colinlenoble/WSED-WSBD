# -*- coding: utf-8 -*-
"""
Latitude-band event-duration-share figure combining both of this project's
"drought" definitions, one column each:
  - SWED (Solar-Wind Energy Drought): this project's per-pixel compound
    wind+solar capacity-factor coincidence-below-threshold event, exactly
    what fig_duration_distribution_latitude.py's own figures already show
    -- reused here from its cached counts table, not recomputed.
  - SWBD (Solar-Wind Budget Drought): fig45.py's residual-load metric
    (temperature-driven demand minus a 50%-renewable-penetration supply,
    exceeding its own 99th-percentile reference threshold), which lives on
    admin regions (poly_idx, config.SHAPEFILE_PATH_LIGHT), not pixels, and
    whose day-level exceedance field fig45.py's own pipeline never
    retained (compute_rl_one_gcm collapses straight to one annual
    cumulative scalar, rl_cum). Built here from scratch: fig45._calculate_rl
    is reused with return_daily=True to get the boolean day-level field,
    then run-length encoded into an event table exactly like
    duration_decomposition.compute_event_table, just stacked over poly_idx
    instead of (lat, lon) since SWBD has no pixel grid.

Both are normalized the same way -- each duration's share of that (zone,
GWL) group's total events, summing to 1 -- so the two duration mixes are
directly comparable despite coming from structurally different data.

Latitude-zone assignment for SWBD, since a region is an irregular polygon
rather than a single pixel: each region is assigned to whichever of the 5
latitude zones (LAT_ZONE_EDGES, imported from
fig_duration_distribution_latitude.py) holds the largest *overlap area*
with it (see assign_regions_to_zones) -- not its centroid or bounding box,
so a region straddling a zone edge lands on whichever side actually holds
more of its own area. Pooling within a zone then equal-GCM-weights each
region's own curve first (same "normalize per GCM, then average with
weight 1/n_GCM" convention as _group_counts's arr_share), then averages
those per-region curves across the zone's regions weighted by each
region's own true (equal-area-projection) area -- so a large region (e.g.
Western China) doesn't count the same as a small one (e.g. a Caribbean
island), the region-level analogue of the pixel-grid figures' cos(latitude)
area weighting.

Figure layout (2 columns x 6 rows; see plot_swed_swbd_distributions): column
0 is SWED, column 1 is SWBD. Row 0 holds each side's own locator map --
SWED's is the flat per-zone latitude-band map (a pixel's zone is just its
own latitude); SWBD's instead choropleths every admin region by the zone
assign_regions_to_zones actually assigned it (the same max-overlap-area
rule above), since a region can straddle a zone edge. Rows 1-5, one per
latitude zone, plot each non-baseline GWL's duration-share curve as a
ratio to the GWL0-61 baseline's own curve at that duration (day 1 ..
max_duration_days on x), log-scale and harmonized across all 10 panels so
every factor-of-4 tick (1/16, 1/4, 1, 4, 16, ...) sits the same distance
apart everywhere and a dashed line marks ratio = 1 -- rare outlier ratios
are clipped off the shared range rather than stretching it for every other
panel (see _shared_ratio_ylim). A duration with too few events on either
side of the ratio is left as a gap rather than dropping its whole (zone,
GWL) line -- every connected run of points still draws its own segment --
but a point left isolated by that gap (no valid neighbor on either side)
is removed too, rather than drawn as a disconnected floating marker (see
_drop_isolated_points).
"""
import os
import config
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
os.environ["ESMFMKFILE"]       = config.ESMFMKFILE_XENV

import argparse
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import xarray as xr
import cartopy.crs as ccrs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# Reused rather than duplicated: SWED cache + shared constants/helpers (zone
# edges, GWL colors, the locator map, the equal-GCM-weighted pooling
# convention, ...). fig45 (this project's residual-load formula + GCM/run
# discovery) is imported lazily, inside the two functions that actually
# compute SWBD from raw data (_daily_swbd_exceedance/build_swbd_counts_table)
# -- it drags in xagg -> xesmf -> esmpy, which needs a real ESMFMKFILE and so
# only works on the HPC. Everything else here (region/zone assignment,
# pooling, and plotting off an already-built SWBD cache) doesn't need it and
# should keep working in a plain xarray/geopandas/matplotlib environment.
import fig_duration_distribution_latitude as swed_mod

# fig45.py's headline SWBD configuration (MAIN_THR/MAIN_TOT_RE/MAIN_MIX),
# duplicated here as plain literals rather than reached into at import time,
# so building the CLI parser doesn't itself require fig45 (and therefore
# xesmf) to import successfully.
SWBD_MAIN_THR    = 0.99
SWBD_MAIN_TOT_RE = 0.5
SWBD_MAIN_MIX    = "current"

# GWL every ratio panel divides by -- this project's reference/baseline
# period (see fig_duration_distribution_latitude.py's own GWL0-61 handling).
BASELINE_GWL  = "GWL0-61"
RATIO_YLABEL  = "Ratio nb of events\nat each GWL divided\nby reference 0.61°C"


# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Latitude-band event-duration-share figure combining SWED (filled) "
            "and SWBD (dotted) lines."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preprocessed_path", default=config.PATH_PREPROCESSED)
    parser.add_argument("--ssp", default=config.SSP)
    parser.add_argument("--reanalysis", default=config.REANALYSIS)
    parser.add_argument("--suffix_shp", default=config.AGREEMENT_SUFFIX_SHP)
    parser.add_argument("--regions_shapefile", default=config.SHAPEFILE_PATH_LIGHT,
                         help="Admin-region shapefile SWBD's poly_idx is built from "
                              "(default: config.SHAPEFILE_PATH_LIGHT, same one xagg used "
                              "to build wcf_agg_*/scf_agg_*/tas_pop_agg_*).")
    parser.add_argument(
        "--gwl_list", nargs="+", default=swed_mod.GWL_KEYS,
        help="GWL keys to include (default: GWL0-61 GWL1-5 GWL2 GWL3).",
    )
    parser.add_argument("--threshold", type=float, default=0.1,
                         help="SWED quantile threshold (default: 0.1, matching "
                              "fig_duration_distribution_latitude.py).")
    parser.add_argument("--swbd_thr", type=float, default=SWBD_MAIN_THR,
                         help=f"SWBD residual-load quantile threshold (default: "
                              f"{SWBD_MAIN_THR}, matching fig45.py's MAIN_THR).")
    parser.add_argument("--swbd_tot_re", type=float, default=SWBD_MAIN_TOT_RE,
                         help=f"SWBD renewable-penetration fraction (default: "
                              f"{SWBD_MAIN_TOT_RE}, matching fig45.py's MAIN_TOT_RE).")
    parser.add_argument("--swbd_mix", default=SWBD_MAIN_MIX,
                         help=f"SWBD solar/wind mix key (default: {SWBD_MAIN_MIX!r}, "
                              f"matching fig45.py's MAIN_MIX).")
    parser.add_argument("--exclude_gcm_run", nargs="+", default=config.EXCLUDE_GCM_RUN)
    parser.add_argument("--shapefile", default=config.SHAPEFILE_PATH,
                         help="Land shapefile for the SWED side (pixel land mask / "
                              "locator map), same as fig_duration_distribution_latitude.py.")
    parser.add_argument("--era5_grid_path", default=None)
    parser.add_argument("--swed_cache_csv", default=None,
                         help="Path to fig_duration_distribution_latitude.py's own "
                              "event_duration_counts_cache.csv (default: "
                              "<output_dir>/event_duration_counts_cache.csv).")
    parser.add_argument("--swbd_cache_csv", default=None,
                         help="Path to this script's own SWBD counts cache CSV (default: "
                              "<output_dir>/swbd_duration_counts_cache.csv).")
    parser.add_argument("--recompute_swbd", action="store_true", default=False)
    parser.add_argument("--output_dir", default="../final_figs")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max_duration_days", type=float, default=12.0,
                         help="X-axis cap in days, shared by every panel (default: 12 -- "
                              "shorter than fig_duration_distribution_latitude.py's own "
                              "20-day 'full' figure, since past ~12 days this figure's "
                              "per-duration GWL/baseline ratio is built from too few "
                              "events per zone to be meaningful).")
    parser.add_argument("--min_events", type=int, default=5)
    return parser.parse_args()


# =============================================================================
# Region -> latitude-zone assignment (max overlap area) + region area weight
# =============================================================================

def assign_regions_to_zones(shapefile_path):
    """
    For every region (poly_idx = row index, matching how xagg/fig45.py
    assign poly_idx off this same shapefile) in the SWBD admin-region
    shapefile, determine which of the 5 latitude zones (LAT_ZONE_EDGES) it
    overlaps *most* with, by actual polygon intersection area -- not
    centroid or bounding box, so a region straddling a zone edge is
    assigned to whichever side holds more of its own area.

    Both the overlap comparison and the returned region areas are computed
    in an equal-area projection (EPSG:6933), not raw lat/lon degrees, for
    the same reason compute_land_area_share_per_zone cos(latitude)-weights
    pixel rows: a degree of longitude covers less true distance toward the
    poles, so comparing raw-degree polygon areas would over-weight
    high-latitude regions.

    Returns (zone_of_poly: {poly_idx: zone_label}, area_of_poly:
    {poly_idx: area_m2}) -- area_of_poly covers every region with valid
    geometry, zone_of_poly only those successfully assigned to a zone.
    """
    gdf = gpd.read_file(shapefile_path)
    gdf["poly_idx"] = gdf.index
    gdf_eq = gdf.to_crs("EPSG:6933")

    band_geoms_eq = {
        zlabel: gpd.GeoSeries([box(-180, lo, 180, hi)], crs="EPSG:4326")
                   .to_crs("EPSG:6933").iloc[0]
        for zlabel, (lo, hi) in swed_mod.ZONE_BOUNDS.items()
    }

    zone_of_poly, area_of_poly = {}, {}
    for poly_idx, geom in zip(gdf_eq["poly_idx"], gdf_eq.geometry):
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        area_of_poly[poly_idx] = geom.area
        overlaps = {zlabel: geom.intersection(band).area
                    for zlabel, band in band_geoms_eq.items()}
        best_zone = max(overlaps, key=overlaps.get)
        if overlaps[best_zone] > 0:
            zone_of_poly[poly_idx] = best_zone
    return zone_of_poly, area_of_poly


def _add_region_zone_map(fig, gs_cell, shapefile_path, zone_of_poly):
    """
    SWBD's own locator map, drawn beside SWED's flat per-zone latitude-band
    map (swed_mod._add_locator_map) in the figure's top row. SWED pixels are
    single points, so assigning one to a zone is just its own latitude --
    nothing to show beyond the band itself. SWBD's admin regions are
    polygons that can straddle a zone edge, so instead this choropleths
    every region by whichever zone assign_regions_to_zones actually
    assigned it (max overlap-area rule): a literal picture of that
    attribution, one colored patch per region rather than one band per
    zone. Same map extent/coastlines/zone-edge-line styling as
    swed_mod._add_locator_map, so the two maps read as a matched pair.
    Regions with no assigned zone (zero overlap with every band, e.g. a
    sliver entirely poleward of MAP_LAT_SOUTH/NORTH) are drawn in flat gray.
    """
    gdf = gpd.read_file(shapefile_path)
    gdf["poly_idx"] = gdf.index
    colors = gdf["poly_idx"].map(zone_of_poly).map(swed_mod.ZONE_MAP_COLORS).fillna("#cccccc")

    ax_map = fig.add_subplot(gs_cell, projection=ccrs.PlateCarree())
    ax_map.set_extent(
        [swed_mod.MAP_LON_WEST, swed_mod.MAP_LON_EAST,
         swed_mod.MAP_LAT_SOUTH - 2, swed_mod.MAP_LAT_NORTH + 2],
        crs=ccrs.PlateCarree())
    ax_map.set_anchor("W")
    gdf.plot(ax=ax_map, color=colors, edgecolor="white", linewidth=0.15,
             transform=ccrs.PlateCarree(), zorder=2)
    ax_map.coastlines(resolution="110m", linewidth=0.3, color="#444444", zorder=3)
    for edge in swed_mod.LAT_ZONE_EDGES:
        ax_map.axhline(edge, color="black", linewidth=0.5, zorder=4)

    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    return ax_map


# =============================================================================
# SWBD: daily per-region exceedance field + poly_idx-native event table
# =============================================================================

def _daily_swbd_exceedance(GCM, run, ssp, gwl, thr, tot_re, mix, path_preprocessed,
                            df_share, reanalysis, suffix_shp, demand_cfg=None):
    """
    Daily (time, poly_idx) boolean 'SWBD day' field for one (GCM, run,
    GWL): True where the residual load exceeds its own GWL0-61-reference-
    period `thr`-quantile threshold -- fig45._calculate_rl's own formula,
    reused (via return_daily=True) rather than duplicated, just kept at
    daily resolution instead of immediately collapsed into fig45.py's
    rl_cum (which sums away the day-level structure a duration figure
    needs). Mirrors fig45.compute_rl_one_gcm's reference/current-GWL
    branch structure: GWL0-61 supplies both its own field and the
    threshold/demand_bas every other GWL reuses.
    """
    import fig45  # lazy: pulls in xagg -> xesmf -> esmpy, HPC-only (see module docstring)
    gwl_ref = "GWL0-61"
    dtas_ref, dds_cf_ref, dds_cf_ref_mean = fig45._load_data(
        GCM, run, ssp, gwl_ref, reanalysis, suffix_shp, path_preprocessed, df_share, mix)
    _, threshold, demand_bas, exceeds_ref = fig45._calculate_rl(
        dtas_ref, dds_cf_ref, dds_cf_ref_mean, thr, "Annual", tot_re,
        demand_cfg=demand_cfg, return_daily=True)

    if gwl == gwl_ref:
        exceeds = exceeds_ref
    else:
        dtas, dds_cf, _ = fig45._load_data(
            GCM, run, ssp, gwl, reanalysis, suffix_shp, path_preprocessed, df_share, mix)
        _, _, _, exceeds = fig45._calculate_rl(
            dtas, dds_cf, dds_cf_ref_mean, thr, "Annual", tot_re,
            threshold=threshold, demand_bas=demand_bas,
            demand_cfg=demand_cfg, return_daily=True)

    exceeds["time"] = pd.to_datetime(exceeds["time"].dt.strftime("%Y-%m-%d").values)
    return exceeds


def compute_event_table_by_poly(da, time_dim="time"):
    """
    poly_idx-native twin of duration_decomposition.compute_event_table:
    one row per (event day, poly_idx) for every contiguous run of True in
    boolean/0-1 (time, poly_idx) DataArray `da`, each row carrying the
    event's total duration -- identical run-length-encoding logic, just
    stacked over 'poly_idx' instead of ('lat', 'lon') since SWBD lives on
    the admin-region shapefile's regions, not a pixel grid.
    """
    da = da.astype(int)
    first_time = pd.Timestamp(da[time_dim][0].values)
    da_pad = xr.concat(
        [xr.zeros_like(da.isel({time_dim: 0})).expand_dims(
             {time_dim: [first_time - pd.Timedelta(days=1)]}),
         da],
        dim=time_dim,
    )
    start_event = da_pad.diff(dim=time_dim, label="lower") > 0
    start_event[time_dim] = da[time_dim]
    id_event = start_event.cumsum(dim=time_dim) * da
    id_event = id_event.where(id_event > 0)

    stacked = id_event.stack(z=("poly_idx", time_dim)).dropna("z")
    df = pd.DataFrame({
        "event_id": stacked.values.astype(int),
        "poly_idx": stacked["poly_idx"].values,
        "time":     stacked[time_dim].values,
    })
    df["duration"] = df.groupby(["event_id", "poly_idx"])["event_id"].transform("count")
    return df


SWBD_COUNTS_CACHE_COLUMNS = ["gwl", "GCM", "run", "poly_idx", "duration", "count"]


def build_swbd_counts_table(path_preprocessed, gwl_list, ssp, reanalysis, suffix_shp,
                             thr, tot_re, mix, df_share, exclude_gcm_run, demand_cfg=None):
    """
    One row per (gwl, GCM, run, poly_idx, duration) with the number of SWBD
    events of that exact duration -- the poly_idx-native counterpart to
    fig_duration_distribution_latitude.py's build_counts_table, aggregating
    each realization immediately rather than concatenating every raw
    per-event row first (this is the expensive step: it opens every GCM/
    run's tas_pop_agg_*/wcf_agg_*/scf_agg_* files).
    """
    import fig45  # lazy: see _daily_swbd_exceedance
    exclude_pairs = set(tuple(x.split(":")) for x in (exclude_gcm_run or []))
    rows = []
    for gwl in gwl_list:
        print(f"\n  -- {gwl} --")
        for GCM, run in fig45._iter_gcm_runs(path_preprocessed, ssp, reanalysis, suffix_shp):
            if (GCM, run) in exclude_pairs:
                print(f"    [excluded] {GCM} {run}")
                continue
            print(f"    {GCM} / {run}")
            try:
                exceeds = _daily_swbd_exceedance(
                    GCM, run, ssp, gwl, thr, tot_re, mix, path_preprocessed,
                    df_share, reanalysis, suffix_shp, demand_cfg=demand_cfg)
                df_events = compute_event_table_by_poly(exceeds)
                df_events = df_events.drop_duplicates(["event_id", "poly_idx"])
            except Exception as exc:
                print(f"      [ERROR] {GCM}/{run}/{gwl}: {exc}")
                continue
            frag = (df_events.groupby(["poly_idx", "duration"]).size()
                    .reset_index(name="count"))
            frag["gwl"] = gwl
            frag["GCM"] = GCM
            frag["run"] = run
            rows.append(frag[SWBD_COUNTS_CACHE_COLUMNS])

    if not rows:
        return pd.DataFrame(columns=SWBD_COUNTS_CACHE_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def _swbd_cache_meta(thr, tot_re, mix, ssp, suffix_shp):
    return {"threshold": thr, "tot_re": tot_re, "mix": mix, "ssp": ssp, "suffix_shp": suffix_shp}


def save_swbd_counts_cache(counts_df, path, thr, tot_re, mix, ssp, suffix_shp):
    """Same '#'-commented-metadata-line convention as
    fig_duration_distribution_latitude.py's save_counts_cache."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(f"# {json.dumps(_swbd_cache_meta(thr, tot_re, mix, ssp, suffix_shp))}\n")
        counts_df.to_csv(f, index=False)


def load_swbd_counts_cache(path, thr, tot_re, mix, ssp, suffix_shp):
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
    expected = _swbd_cache_meta(thr, tot_re, mix, ssp, suffix_shp)
    if meta != expected:
        print(f"  [cache] {path} was built with different settings {meta} "
              f"than requested {expected} -- ignoring and rebuilding.")
        return None
    return pd.read_csv(path, comment="#")


# =============================================================================
# Two-level pooling: equal-GCM-weighted per region, then area-weighted across
# a zone's regions
# =============================================================================

def _region_group_counts(counts_df, gwl, poly_idx, x_int):
    """
    Equal-GCM-weighted share-of-events curve for a single region (poly_idx)
    -- same "normalize per GCM, then average with weight 1/n_GCM"
    convention as fig_duration_distribution_latitude.py's _group_counts,
    scoped to one region instead of one latitude zone. The building block
    zone_group_counts_swbd area-weights together across a zone's regions.
    Returns (arr_share, raw_total) or None if no GCM has data for this
    (gwl, poly_idx).
    """
    sub = counts_df[(counts_df["gwl"] == gwl) & (counts_df["poly_idx"] == poly_idx)]
    if sub.empty:
        return None
    raw_total = float(sub["count"].sum())

    gcm_series, gcm_totals = [], []
    for gcm, gsub in sub.groupby("GCM"):
        n_runs = gsub["run"].nunique()
        s = gsub.groupby("duration")["count"].sum() / n_runs
        total_g = float(s.sum())
        if total_g <= 0:
            continue
        gcm_series.append(s)
        gcm_totals.append(total_g)

    n_gcm = len(gcm_series)
    if n_gcm == 0:
        return None
    arr_share = np.zeros(len(x_int))
    for s, total_g in zip(gcm_series, gcm_totals):
        vals = np.array([s.get(d, 0.0) for d in x_int], dtype=float)
        arr_share += (vals / total_g) / n_gcm
    return arr_share, raw_total


def zone_group_counts_swbd(counts_df, gwl, zone, zone_of_poly, area_of_poly, x_int, min_events=5):
    """
    Area-weighted mean, across every region assigned to `zone` (see
    assign_regions_to_zones), of each region's own equal-GCM-weighted
    share-of-events curve (_region_group_counts) -- so a large region
    doesn't get the same say in the zone's pooled shape as a small one (a
    plain per-region average would let e.g. a small island count exactly
    as much as all of Western China). Regions whose own pooled raw SWBD
    event count falls under min_events are excluded, same gating
    convention as plot_distributions'/build_swbd_counts_table callers'
    min_events. Returns arr_share, or None if no region in this zone
    qualifies.
    """
    polys = [p for p, z in zone_of_poly.items() if z == zone]
    curves, weights = [], []
    for poly_idx in polys:
        group = _region_group_counts(counts_df, gwl, poly_idx, x_int)
        if group is None:
            continue
        arr_share, raw_total = group
        if raw_total < min_events:
            continue
        area = area_of_poly.get(poly_idx, 0.0)
        if area <= 0:
            continue
        curves.append(arr_share)
        weights.append(area)

    if not curves:
        return None
    weights = np.array(weights)
    curves = np.array(curves)
    return (curves * weights[:, None]).sum(axis=0) / weights.sum()


# =============================================================================
# Plotting
# =============================================================================

def _shared_ratio_ylim(ratio_arrays, outlier_pct=2.0, pad=1.08):
    """
    Symmetric-in-log ylim for every ratio panel -- shared across both
    columns and all 5 zones, so a given vertical distance means the same
    fold-change everywhere and e.g. 1/4x and 4x sit equidistant from the
    dashed ratio=1 line (log(1/4) == -log(4)) -- see
    plot_swed_swbd_distributions.

    Bounded by the `outlier_pct`-th-from-the-edge percentile of every
    finite, positive ratio actually drawn (folded around 1 via abs(log10)),
    not the true min/max: a handful of rare/noisy long-duration ratios
    (thin tails, a handful of events on one side of the baseline division)
    would otherwise blow the one shared axis out for every other panel.
    Those points are left off the visible range -- clipped, not rescaled
    for -- rather than dropped from the underlying data. Never narrower
    than a 4x/0.25x span, so the axis always resolves at least the first
    factor-of-4 tick beyond 1 in both directions (see _ratio_yticks).
    """
    min_half_decades = np.log10(4.0)
    finite_chunks = [a[np.isfinite(a) & (a > 0)] for a in ratio_arrays if a is not None]
    finite_chunks = [c for c in finite_chunks if c.size > 0]
    if finite_chunks:
        logs = np.abs(np.log10(np.concatenate(finite_chunks)))
        bound = max(np.percentile(logs, 100 - outlier_pct), min_half_decades)
    else:
        bound = min_half_decades
    bound *= pad
    return 10.0 ** -bound, 10.0 ** bound


def _ratio_yticks(ylim):
    """Powers of 4 within `ylim` (..., 1/16, 1/4, 1, 4, 16, ...), one tick per factor of 4."""
    lo, hi = ylim
    return [t for t in (4.0 ** k for k in range(-6, 7)) if lo * 0.98 <= t <= hi * 1.02]


def _ratio_tick_label(v, _pos=None):
    """'1/4'/'1/16' below 1, plain integers at/above 1 -- reads as a fold-change, not a decimal."""
    if v >= 1:
        return f"{v:g}"
    return f"1/{round(1.0 / v):g}"


def _drop_isolated_points(arr):
    """
    NaN out any point in `arr` with no valid neighbor on either side.
    A duration with too few events at either the GWL or the baseline (see
    plot_swed_swbd_distributions) is left as NaN in the ratio array; plotted
    as-is, a NaN with valid points on both sides just breaks the line
    there, which is fine -- but a single valid point surrounded by NaN on
    both sides would still draw as a lone, disconnected marker (no line
    into or out of it), which reads as a real data point rather than the
    single-duration gap it actually is. Runs of 2+ consecutive valid points
    are left untouched and still draw as connected line segments, even if
    the overall curve is broken elsewhere (e.g. a southern-band SWBD line
    with a few short-duration gaps still shows every segment it has data
    for, rather than being dropped outright).
    """
    valid = np.isfinite(arr)
    left_valid = np.concatenate(([False], valid[:-1]))
    right_valid = np.concatenate((valid[1:], [False]))
    isolated = valid & ~left_valid & ~right_valid
    out = arr.copy()
    out[isolated] = np.nan
    return out


def plot_swed_swbd_distributions(swed_counts_df, swbd_counts_df, land_area_pct,
                                  zone_of_poly, area_of_poly, regions_shapefile,
                                  gwl_list, output_path, dpi=300,
                                  max_duration_days=12.0, min_events=5):
    """
    2-column x 6-row layout: column 0 is SWED, column 1 is SWBD.

    Row 0 holds each side's own locator map. SWED's (swed_mod's own
    _add_locator_map) is the flat per-zone latitude-band map -- a pixel's
    zone is just its own latitude, nothing else to show. SWBD's
    (_add_region_zone_map) instead choropleths every admin region by the
    zone assign_regions_to_zones actually assigned it (max overlap-area
    rule), since a region is a polygon that can straddle a zone edge.

    Rows 1-5 are the five latitude zones (day 1 .. max_duration_days, no
    main/zoom split -- max_duration_days is kept short precisely so one
    panel is enough), each with a single log-scale y-axis (harmonized
    across all 10 panels via _shared_ratio_ylim, factor-of-4 ticks, dashed
    line at 1): each non-baseline GWL's duration-share curve
    (swed_mod._group_counts / zone_group_counts_swbd, same equal-GCM/
    area-weighted pooling as every other figure here) divided elementwise
    by the GWL0-61 baseline's own curve at that duration -- so > 1 means
    that duration is over-represented at that GWL relative to baseline, < 1
    under-represented, independent of the two metrics' very different
    absolute share scales. Durations with too little data (either side
    short of min_events) are left as gaps; an isolated single point
    surrounded by gaps on both sides is dropped too rather than drawn as a
    disconnected floating marker -- see _drop_isolated_points -- but a
    (zone, GWL) curve with several such gaps still draws every connected
    segment it has, instead of being dropped outright.
    """
    x_int = np.arange(1, int(np.ceil(max_duration_days)) + 1)
    zone_order = list(reversed(swed_mod.LAT_ZONE_LABELS))
    ratio_gwls = [g for g in gwl_list if g != BASELINE_GWL]

    def _swed_share(gwl, zlabel):
        group = swed_mod._group_counts(swed_counts_df, gwl, zlabel, x_int)
        if group is None:
            return None
        arr_share, _, _, _, raw_total = group
        return arr_share if raw_total >= min_events else None

    def _swbd_share(gwl, zlabel):
        return zone_group_counts_swbd(
            swbd_counts_df, gwl, zlabel, zone_of_poly, area_of_poly, x_int,
            min_events=min_events)

    share_fns = {"SWED": _swed_share, "SWBD": _swbd_share}

    # Pass 1: gather every panel's ratio curves up front, so the one shared
    # ratio y-axis can be fixed before anything is drawn. A duration with
    # too little data on either side of the ratio (baseline or GWL itself
    # short of min_events/zero events at that duration) is left as a NaN
    # gap, not dropped for the whole curve -- only points left isolated by
    # that gap (no valid neighbor on either side, so they'd draw as a
    # disconnected floating marker) are removed; see _drop_isolated_points.
    panel_data = {col: [] for col in share_fns}
    for zlabel in zone_order:
        for col, share_fn in share_fns.items():
            base = share_fn(BASELINE_GWL, zlabel)
            ratios = {}
            for gwl in ratio_gwls:
                arr = share_fn(gwl, zlabel)
                if arr is None or base is None:
                    ratios[gwl] = None
                    continue
                base_safe = np.where(base > 0, base, np.nan)
                ratio = np.where(arr > 0, arr / base_safe, np.nan)
                ratio = _drop_isolated_points(ratio)
                ratios[gwl] = ratio if np.any(np.isfinite(ratio)) else None
            panel_data[col].append({"ratios": ratios})

    all_ratio_arrays = [arr for col in panel_data.values() for row in col
                         for arr in row["ratios"].values()]
    ratio_ylim = _shared_ratio_ylim(all_ratio_arrays)
    ratio_yticks = _ratio_yticks(ratio_ylim)

    n_rows = len(zone_order)
    fig = plt.figure(figsize=(swed_mod.FIG_WIDTH_IN * 1.55, swed_mod.FIG_WIDTH_IN * 1.75))
    gs = GridSpec(n_rows + 1, 2, height_ratios=[0.8] + [1.0] * n_rows,
                  left=0.11, right=0.97, top=0.93, bottom=0.09,
                  hspace=0.65, wspace=0.30, figure=fig)

    # One letter per panel, reading order (row-major, left-to-right, top to
    # bottom): a/b for the two row-0 maps, then c/d, e/f, ... for each
    # latitude zone's SWED/SWBD pair below -- not one letter per row, since
    # SWED and SWBD are now full, independently-readable panels.
    all_letters = [chr(ord("a") + k) for k in range(2 * (n_rows + 1))]

    ax_map_swed, _ = swed_mod._add_locator_map(fig, gs[0, 0], zone_order)
    ax_map_swbd = _add_region_zone_map(fig, gs[0, 1], regions_shapefile, zone_of_poly)
    for ax_map, label in ((ax_map_swed, "SWED"), (ax_map_swbd, "SWBD (region attribution)")):
        bbox = ax_map.get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.012, label, ha="center", va="bottom",
                  fontsize=swed_mod.ZONE_TITLE_FONTSIZE, fontweight="bold")
    for ax_map, letter in ((ax_map_swed, all_letters[0]), (ax_map_swbd, all_letters[1])):
        bbox = ax_map.get_position()
        fig.text(bbox.x0 - 0.02, bbox.y1 + 0.012, letter, ha="left", va="bottom",
                  fontsize=swed_mod.LETTER_FONTSIZE, fontweight="bold")

    for i, zlabel in enumerate(zone_order):
        pct = land_area_pct.get(zlabel)
        lat_range_text = swed_mod._fmt_lat_range(zlabel)
        zone_label_text = (f"{lat_range_text} ({pct:.1f}% of land area)" if pct is not None
                            else lat_range_text)

        for j, col in enumerate(share_fns):
            row = panel_data[col][i]
            ax_left = fig.add_subplot(gs[i + 1, j])

            ax_left.axhline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=1)
            for gwl in ratio_gwls:
                arr = row["ratios"].get(gwl)
                if arr is None:
                    continue
                color = swed_mod.GWL_COLORS.get(gwl, "gray")
                ax_left.plot(x_int, arr, color=color, marker="o", markersize=2.2,
                             linewidth=1.3, zorder=3)
            ax_left.set_yscale("log")
            ax_left.set_ylim(*ratio_ylim)
            ax_left.set_yticks(ratio_yticks)
            ax_left.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_ratio_tick_label))
            ax_left.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax_left.set_xlim(0.5, max_duration_days + 0.5)
            ax_left.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax_left.tick_params(labelsize=swed_mod.TICK_FONTSIZE)
            ax_left.grid(True, linestyle="--", alpha=0.3)
            for spine in ax_left.spines.values():
                spine.set_linewidth(0.4)
            ax_left.spines["left"].set_color(swed_mod.ZONE_MAP_COLORS[zlabel])
            ax_left.spines["left"].set_linewidth(2.2)

            panel_letter = all_letters[2 + i * 2 + j]
            ax_left.text(0.0, 1.05, panel_letter, transform=ax_left.transAxes,
                         ha="left", va="bottom", fontsize=swed_mod.LETTER_FONTSIZE,
                         fontweight="bold")
            if j == 0:
                ax_left.set_ylabel(RATIO_YLABEL, fontsize=swed_mod.AXIS_LABEL_FONTSIZE)
                ax_left.text(0.13, 1.05, zone_label_text, transform=ax_left.transAxes,
                             ha="left", va="bottom", fontsize=swed_mod.ZONE_TITLE_FONTSIZE,
                             fontweight="bold", color=swed_mod.ZONE_MAP_COLORS[zlabel])
            if i == n_rows - 1:
                ax_left.set_xlabel("Event duration (days)", fontsize=swed_mod.XLABEL_FONTSIZE)

    gwl_handles = [
        Line2D([0], [0], color=swed_mod.GWL_COLORS.get(gwl, "gray"), marker="o",
               markersize=3, linewidth=1.6, label=swed_mod.GWL_LABELS.get(gwl, gwl))
        for gwl in ratio_gwls
    ]
    baseline_label = swed_mod.GWL_LABELS.get(BASELINE_GWL, BASELINE_GWL)
    extra_handles = [
        Line2D([0], [0], color="black", linewidth=0.8, linestyle="--",
               label=f"Ratio = 1 ({baseline_label})"),
    ]
    fig.legend(handles=gwl_handles + extra_handles, loc="lower center",
               ncol=len(gwl_handles) + len(extra_handles), fontsize=swed_mod.LEGEND_FONTSIZE,
               bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1.3, handlelength=1.8)
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
    print("STEP 1 - Region -> latitude-zone assignment (max overlap) + area weights")
    print("=" * 60)
    zone_of_poly, area_of_poly = assign_regions_to_zones(args.regions_shapefile)
    print(f"  {len(zone_of_poly)} regions assigned across "
          f"{len(set(zone_of_poly.values()))} zones")

    print("\n" + "=" * 60)
    print("STEP 2 - SWED land area share per zone (ERA5 reference grid)")
    print("=" * 60)
    era5_lat, era5_lon = swed_mod.load_era5_reference_grid(
        args.preprocessed_path, era5_grid_path=args.era5_grid_path)
    land_area_pct = swed_mod.compute_land_area_share_per_zone(era5_lat, era5_lon, args.shapefile)
    for z, pct in land_area_pct.items():
        print(f"  {z}: {pct:.1f}% of land area")

    print("\n" + "=" * 60)
    print("STEP 3 - SWED event-duration counts (cached)")
    print("=" * 60)
    swed_cache_path = args.swed_cache_csv or os.path.join(
        args.output_dir, "event_duration_counts_cache.csv")
    swed_counts_df = swed_mod.load_counts_cache(swed_cache_path, args.threshold, args.ssp)
    if swed_counts_df is None:
        swed_counts_df = swed_mod.build_counts_table(
            args.preprocessed_path, args.gwl_list, args.ssp, args.threshold,
            args.shapefile, [], args.exclude_gcm_run,
        )
        swed_mod.save_counts_cache(swed_counts_df, swed_cache_path, args.threshold, args.ssp)
    print(f"  {len(swed_counts_df)} SWED cache rows")

    print("\n" + "=" * 60)
    print("STEP 4 - SWBD event-duration counts (cached)")
    print("=" * 60)
    swbd_cache_path = args.swbd_cache_csv or os.path.join(
        args.output_dir, "swbd_duration_counts_cache.csv")
    swbd_counts_df = None if args.recompute_swbd else load_swbd_counts_cache(
        swbd_cache_path, args.swbd_thr, args.swbd_tot_re, args.swbd_mix, args.ssp, args.suffix_shp)
    if swbd_counts_df is not None:
        print(f"  Loaded cached SWBD counts from {swbd_cache_path} "
              f"({len(swbd_counts_df)} rows) -- skipping the expensive rebuild.")
    else:
        df_share = pd.read_csv(config.SHARE_RENEWABLE_CSV)
        swbd_counts_df = build_swbd_counts_table(
            args.preprocessed_path, args.gwl_list, args.ssp, args.reanalysis, args.suffix_shp,
            args.swbd_thr, args.swbd_tot_re, args.swbd_mix, df_share, args.exclude_gcm_run,
        )
        save_swbd_counts_cache(
            swbd_counts_df, swbd_cache_path, args.swbd_thr, args.swbd_tot_re,
            args.swbd_mix, args.ssp, args.suffix_shp)
        print(f"  Saved SWBD counts cache -> {swbd_cache_path}")

    print("\n" + "=" * 60)
    print("STEP 5 - Plotting")
    print("=" * 60)
    out_path = os.path.join(args.output_dir, "fig_duration_distribution_swed_swbd_ratio.png")
    plot_swed_swbd_distributions(
        swed_counts_df, swbd_counts_df, land_area_pct, zone_of_poly, area_of_poly,
        args.regions_shapefile, args.gwl_list, out_path, dpi=args.dpi,
        max_duration_days=args.max_duration_days, min_events=args.min_events,
    )
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()

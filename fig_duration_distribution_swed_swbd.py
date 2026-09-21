# -*- coding: utf-8 -*-
"""
Latitude-band event-duration-share figure combining both of this project's
"drought" definitions on one axis per zone:
  - SWED (Solar-Wind Energy Drought): this project's per-pixel compound
    wind+solar capacity-factor coincidence-below-threshold event, exactly
    what fig_duration_distribution_latitude.py's own figures already show
    -- reused here from its cached counts table, not recomputed. Drawn as
    filled (solid) lines.
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
    instead of (lat, lon) since SWBD has no pixel grid. Drawn as dotted
    lines.

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionPatch
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
    parser.add_argument("--max_duration_days", type=float,
                         default=swed_mod.FULL_FIGURE_MAX_DURATION_DAYS)
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

def plot_swed_swbd_distributions(swed_counts_df, swbd_counts_df, land_area_pct,
                                  zone_of_poly, area_of_poly, gwl_list, output_path,
                                  dpi=300, max_duration_days=20.0, min_events=5):
    """
    Same locator-map + connector-line + colored-spine + main/zoom-split
    layout as fig_duration_distribution_latitude.py's plot_distributions,
    but drawing both event definitions on the same axis per zone: SWED
    (this script's swed_counts_df, via swed_mod._group_counts, same as the
    existing figures) as filled solid lines, and SWBD (swbd_counts_df, via
    zone_group_counts_swbd) as dotted lines -- both on the normalized share
    scale, so directly comparable despite the structurally different
    per-pixel-vs-per-region data behind them.
    """
    x_int = np.arange(1, int(np.ceil(max_duration_days)) + 1)
    zone_order = list(reversed(swed_mod.LAT_ZONE_LABELS))

    # Slightly taller than plot_distributions' own figure: bottom is bigger
    # (0.155 vs 0.13) to leave a clean gap between the shared x-axis label
    # and the legend below it (two short groups -- GWL color key, then the
    # SWED/SWBD linestyle key -- see the fig.legend call below).
    fig = plt.figure(figsize=(swed_mod.FIG_WIDTH_IN, swed_mod.FIG_WIDTH_IN * 1.03))
    gs = GridSpec(len(zone_order), 2, width_ratios=[1.2, 2.6],
                  left=0.14, right=0.97, top=0.96, bottom=0.155,
                  hspace=0.85, wspace=0.32, figure=fig)

    ax_map, lat_mid = swed_mod._add_locator_map(fig, gs[:, 0], zone_order)
    ax_map.text(-0.02, 1.03, "a", transform=ax_map.transAxes,
                fontsize=swed_mod.LETTER_FONTSIZE, fontweight="bold")

    do_split = max_duration_days > swed_mod.DURATION_SPLIT_DAY
    split_idx = swed_mod.DURATION_SPLIT_DAY

    dist_axes, dist_axes_zoom = [], []
    for i, zlabel in enumerate(zone_order):
        if do_split:
            inner_gs = GridSpecFromSubplotSpec(
                1, 2, subplot_spec=gs[i, 1],
                width_ratios=swed_mod.DURATION_ZOOM_WIDTH_RATIOS, wspace=0.08)
            ax = fig.add_subplot(inner_gs[0], sharex=dist_axes[0] if dist_axes else None)
            ax_zoom = fig.add_subplot(
                inner_gs[1], sharex=dist_axes_zoom[0] if dist_axes_zoom else None)
        else:
            ax = fig.add_subplot(gs[i, 1], sharex=dist_axes[0] if dist_axes else None)
            ax_zoom = None
        dist_axes.append(ax)
        dist_axes_zoom.append(ax_zoom)
        row_axes = [ax] if ax_zoom is None else [ax, ax_zoom]

        if do_split:
            windows = [(ax, slice(0, split_idx)), (ax_zoom, slice(split_idx - 1, None))]
        else:
            windows = [(ax, slice(None))]

        for gwl in gwl_list:
            color = swed_mod.GWL_COLORS.get(gwl, "gray")

            y_swed = None
            group = swed_mod._group_counts(swed_counts_df, gwl, zlabel, x_int)
            if group is not None:
                arr_share, _, _, _, raw_total = group
                if raw_total >= min_events:
                    y_swed = swed_mod._for_line(arr_share)

            y_swbd = zone_group_counts_swbd(
                swbd_counts_df, gwl, zlabel, zone_of_poly, area_of_poly, x_int,
                min_events=min_events)
            if y_swbd is not None:
                y_swbd = swed_mod._for_line(y_swbd)

            for a, sl in windows:
                if y_swed is not None:
                    a.plot(x_int[sl], y_swed[sl], color=color, marker="o", markersize=2.5,
                           linewidth=1.4, linestyle="-", zorder=3)
                if y_swbd is not None:
                    a.plot(x_int[sl], y_swbd[sl], color=color, marker="s", markersize=2.0,
                           linewidth=1.2, linestyle=":", zorder=3)

        base_label = "Share of events"
        if i == 0:
            ax.set_ylabel(base_label, fontsize=swed_mod.AXIS_LABEL_FONTSIZE)
        pct = land_area_pct.get(zlabel)
        lat_range_text = swed_mod._fmt_lat_range(zlabel)
        zone_label_text = (f"{lat_range_text} ({pct:.1f}% of land area)" if pct is not None
                            else lat_range_text)
        ax.text(0.0, 1.03, chr(ord("b") + i), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=swed_mod.LETTER_FONTSIZE, fontweight="bold")
        ax.text(0.14, 1.03, zone_label_text, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=swed_mod.ZONE_TITLE_FONTSIZE, fontweight="bold",
                color=swed_mod.ZONE_MAP_COLORS[zlabel])
        for a in row_axes:
            a.set_yscale("log")
            a.tick_params(labelsize=swed_mod.TICK_FONTSIZE)
            locator = (matplotlib.ticker.MaxNLocator(integer=True) if a is ax
                       else matplotlib.ticker.MaxNLocator(integer=True, nbins=6))
            a.xaxis.set_major_locator(locator)
            a.grid(True, linestyle="--", alpha=0.3)
            for spine in a.spines.values():
                spine.set_linewidth(0.4)

        if do_split:
            ax.axvline(swed_mod.DURATION_SPLIT_DAY, color=swed_mod.DURATION_SPLIT_LINE_COLOR,
                       linewidth=0.8, linestyle="-", zorder=1)
            ax.set_xlim(0.5, swed_mod.DURATION_SPLIT_DAY + 0.5)
            ax_zoom.set_xlim(swed_mod.DURATION_SPLIT_DAY - 0.5, max_duration_days + 0.5)
            ax_zoom.yaxis.tick_right()
            ax_zoom.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.set_xlim(0.5, max_duration_days + 0.5)

        ax.spines["left"].set_color(swed_mod.ZONE_MAP_COLORS[zlabel])
        ax.spines["left"].set_linewidth(2.5)
        con = ConnectionPatch(
            xyA=(swed_mod.MAP_LON_EAST, lat_mid[zlabel]), coordsA=ax_map.transData,
            xyB=(0, 0.5), coordsB=ax.transAxes,
            color=swed_mod.ZONE_MAP_COLORS[zlabel], linewidth=0.9, linestyle="--",
            alpha=0.85, zorder=1,
        )
        fig.add_artist(con)

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

        for ax, ax_zoom in zip(dist_axes, dist_axes_zoom):
            y_lo_zoom, y_hi_zoom = ax_zoom.get_ylim()
            y_lo_main, y_hi_main = sorted(ax.get_ylim())
            for y_zoom, y_frac_zoom in ((y_hi_zoom, 1), (y_lo_zoom, 0)):
                y_anchor = min(max(y_zoom, y_lo_main), y_hi_main)
                zoom_link = ConnectionPatch(
                    xyA=(swed_mod.DURATION_SPLIT_DAY, y_anchor), coordsA=ax.transData,
                    xyB=(0, y_frac_zoom), coordsB=ax_zoom.transAxes,
                    color=swed_mod.DURATION_SPLIT_LINE_COLOR, linewidth=0.7,
                    linestyle=":", zorder=1,
                )
                fig.add_artist(zoom_link)

            d = 0.02
            break_kwargs = dict(color="black", linewidth=0.8, clip_on=False, zorder=5)
            for y0 in (0, 1):
                ax.plot((1 - d, 1 + d), (y0 - d, y0 + d), transform=ax.transAxes, **break_kwargs)
                ax_zoom.plot((-d, d), (y0 - d, y0 + d), transform=ax_zoom.transAxes, **break_kwargs)

    dist_axes[-1].set_xlabel("Event duration (days)", fontsize=swed_mod.XLABEL_FONTSIZE)

    # Two short legend groups instead of one 2*len(gwl_list)-entry legend
    # (a GWL x {SWED, SWBD} label for every line would repeat "SWED"/"SWBD"
    # once per GWL) -- color already encodes GWL and linestyle already
    # encodes SWED-vs-SWBD on every line drawn above, so the legend only
    # needs to explain each encoding once: GWL_LABELS is followed by the
    # two-linestyle key together in one single-row legend.
    gwl_handles = [
        Line2D([0], [0], color=swed_mod.GWL_COLORS.get(gwl, "gray"), marker="o",
               markersize=3, linewidth=1.6, label=swed_mod.GWL_LABELS.get(gwl, gwl))
        for gwl in gwl_list
    ]
    style_handles = [
        Line2D([0], [0], color="black", marker="o", markersize=3, linewidth=1.4,
               linestyle="-", label="SWED"),
        Line2D([0], [0], color="black", marker="s", markersize=3, linewidth=1.2,
               linestyle=":", label="SWBD"),
    ]
    fig.legend(handles=gwl_handles + style_handles, loc="lower center",
               ncol=len(gwl_handles) + len(style_handles), fontsize=swed_mod.LEGEND_FONTSIZE,
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
    out_path = os.path.join(args.output_dir, "fig_duration_distribution_swed_swbd.png")
    plot_swed_swbd_distributions(
        swed_counts_df, swbd_counts_df, land_area_pct, zone_of_poly, area_of_poly,
        args.gwl_list, out_path, dpi=args.dpi,
        max_duration_days=args.max_duration_days, min_events=args.min_events,
    )
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()

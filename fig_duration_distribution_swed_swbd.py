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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

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

# Day at which plot_swed_swbd_distributions_split breaks each panel into a
# wide "short term" sub-panel (duration <= DURATION_SPLIT_DAY, own y-range
# tight around ratio=1) and a narrower "persistent event" sub-panel
# (duration > DURATION_SPLIT_DAY, own wider y-range) -- same broken-axis
# convention/day as fig_duration_distribution_latitude.py's own
# DURATION_SPLIT_DAY, reused here via swed_mod rather than duplicated.
DURATION_SPLIT_DAY_DEFAULT = swed_mod.DURATION_SPLIT_DAY


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
    parser.add_argument("--geometry_cache_json", default=None,
                         help="Path to this script's own region/zone-geometry cache JSON "
                              "(land_area_pct + zone_of_poly + area_of_poly -- see "
                              "save_geometry_cache/load_geometry_cache). Default: "
                              "<output_dir>/swed_swbd_geometry_cache.json. Together with "
                              "--swed_cache_csv/--swbd_cache_csv, lets the whole figure be "
                              "rebuilt purely from cache -- no --regions_shapefile/"
                              "--shapefile/--era5_grid_path access needed -- once all three "
                              "caches exist, e.g. for iterating on the plot locally.")
    parser.add_argument("--recompute_geometry", action="store_true", default=False,
                         help="Ignore any existing geometry cache and rebuild region/zone "
                              "assignment + land-area shares from the raw shapefiles/ERA5 grid.")
    parser.add_argument("--plot_data_json", default=None,
                         help="Path to a single-file bundle of every plotting input -- SWED "
                              "counts, SWBD counts, land_area_pct, zone_of_poly, area_of_poly "
                              "(see save_plot_data_cache/load_plot_data_cache). Default: "
                              "<output_dir>/swed_swbd_plot_data_cache.json. If present (and "
                              "built with matching settings), loading it alone skips every "
                              "other raw-data step -- no CSV caches, shapefiles, ERA5 grid or "
                              "preprocessed archive needed -- meant to be copied elsewhere "
                              "(e.g. a laptop with none of this project's raw data mounted) "
                              "to iterate on plot_swed_swbd_distributions[_split] or build "
                              "alternative plots off the same data. Always (re)written at the "
                              "end of the raw-data steps, whichever path produced them.")
    parser.add_argument("--recompute_plot_data", action="store_true", default=False,
                         help="Ignore any existing plot-data bundle even if present; fall back "
                              "to the three separate caches/raw-data steps (still rewriting the "
                              "bundle at the end).")
    parser.add_argument("--output_dir", default="../final_figs")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max_duration_days", type=float, default=12.0,
                         help="X-axis cap in days, shared by every panel (default: 12 -- "
                              "shorter than fig_duration_distribution_latitude.py's own "
                              "20-day 'full' figure, since past ~12 days this figure's "
                              "per-duration GWL/baseline ratio is built from too few "
                              "events per zone to be meaningful).")
    parser.add_argument("--min_events", type=int, default=5)
    parser.add_argument("--duration_split_day", type=float, default=DURATION_SPLIT_DAY_DEFAULT,
                         help="Day at which plot_swed_swbd_distributions_split's broken-axis "
                              f"panels split short-term from persistent events (default: "
                              f"{DURATION_SPLIT_DAY_DEFAULT:g}, matching "
                              "fig_duration_distribution_latitude.py's own DURATION_SPLIT_DAY).")
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
# Region/zone-geometry cache: land_area_pct + zone_of_poly + area_of_poly --
# the other "raw data" this figure needs besides the two counts caches above.
# Unlike those (one row per gwl/GCM/run/..., naturally a CSV), these three
# are small dicts/one dict, so a single JSON file holds all three plus a
# metadata block, same "check metadata before trusting the cache" convention
# as save_counts_cache/save_swbd_counts_cache. Once this cache and both
# counts caches exist, the whole figure can be rebuilt with no
# --regions_shapefile/--shapefile/--era5_grid_path access at all -- e.g. to
# tweak plot_swed_swbd_distributions[_split] locally without the HPC-only
# preprocessed archive or even the (much smaller, but not always at hand)
# shapefiles.
# =============================================================================

def _geometry_cache_meta(regions_shapefile, shapefile, era5_grid_path):
    return {
        "regions_shapefile": os.path.abspath(regions_shapefile),
        "shapefile": os.path.abspath(shapefile),
        "era5_grid_path": os.path.abspath(era5_grid_path) if era5_grid_path else None,
        "lat_zone_edges": swed_mod.LAT_ZONE_EDGES,
    }


def save_geometry_cache(path, land_area_pct, zone_of_poly, area_of_poly, meta):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "meta": meta,
        "land_area_pct": land_area_pct,
        # JSON object keys must be strings -- poly_idx (int) is restored on load.
        "zone_of_poly": {str(k): v for k, v in zone_of_poly.items()},
        "area_of_poly": {str(k): v for k, v in area_of_poly.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def load_geometry_cache(path, meta):
    """
    Returns (land_area_pct, zone_of_poly, area_of_poly) if `path` exists and
    was built with the same regions_shapefile/shapefile/era5_grid_path/
    lat_zone_edges as `meta`; otherwise None (caller falls back to rebuilding
    from the raw shapefiles/ERA5 grid). Same stale/foreign-cache guard
    convention as load_counts_cache/load_swbd_counts_cache.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, ValueError):
        print(f"  [cache] {path} is not readable JSON -- ignoring.")
        return None
    if payload.get("meta") != meta:
        print(f"  [cache] {path} was built with different settings {payload.get('meta')} "
              f"than requested {meta} -- ignoring and rebuilding.")
        return None
    land_area_pct = payload["land_area_pct"]
    zone_of_poly = {int(k): v for k, v in payload["zone_of_poly"].items()}
    area_of_poly = {int(k): v for k, v in payload["area_of_poly"].items()}
    return land_area_pct, zone_of_poly, area_of_poly


# =============================================================================
# Full plot-data bundle: every input plot_swed_swbd_distributions[_split]
# needs (swed_counts_df, swbd_counts_df, land_area_pct, zone_of_poly,
# area_of_poly), in one JSON file -- unlike the geometry cache above (which
# still needs the two separate counts CSVs alongside it), loading this one
# file is enough on its own to redraw either figure, tweak their styling, or
# build an entirely different plot off the same underlying data -- no CSV
# caches, shapefiles, ERA5 grid or HPC-only preprocessed archive needed at
# all. Meant to be copied wherever the figure is actually being iterated on
# (a laptop, without any of this project's raw data mounted), not as the
# primary cache main() itself relies on run to run (that's still the three
# separate, independently-invalidated caches above/below -- e.g. changing
# only --swbd_thr shouldn't force SWED or the geometry to be rebuilt too).
# =============================================================================

def _plot_data_cache_meta(regions_shapefile, shapefile, era5_grid_path,
                           threshold, ssp, swbd_thr, swbd_tot_re, swbd_mix, suffix_shp):
    """Union of every setting that changes any of the 5 bundled pieces --
    the SWED counts (swed_mod._cache_meta), the SWBD counts
    (_swbd_cache_meta) and the geometry (_geometry_cache_meta)."""
    return {
        "swed": swed_mod._cache_meta(threshold, ssp),
        "swbd": _swbd_cache_meta(swbd_thr, swbd_tot_re, swbd_mix, ssp, suffix_shp),
        "geometry": _geometry_cache_meta(regions_shapefile, shapefile, era5_grid_path),
    }


def _df_to_json(df):
    """Compact DataFrame JSON encoding: columns named once, plain row
    lists after -- cheaper than pandas' default per-row-dict 'records'
    orient, which repeats every column name on every row."""
    return {"columns": list(df.columns), "data": df.to_numpy().tolist()}


def _df_from_json(payload):
    return pd.DataFrame(payload["data"], columns=payload["columns"])


def save_plot_data_cache(path, swed_counts_df, swbd_counts_df, land_area_pct,
                          zone_of_poly, area_of_poly, meta):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "meta": meta,
        "swed_counts": _df_to_json(swed_counts_df),
        "swbd_counts": _df_to_json(swbd_counts_df),
        "land_area_pct": land_area_pct,
        "zone_of_poly": {str(k): v for k, v in zone_of_poly.items()},
        "area_of_poly": {str(k): v for k, v in area_of_poly.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def load_plot_data_cache(path, meta):
    """
    Returns (swed_counts_df, swbd_counts_df, land_area_pct, zone_of_poly,
    area_of_poly) if `path` exists and was built with the same settings as
    `meta` (SWED threshold/ssp, SWBD thr/tot_re/mix/ssp/suffix_shp, and
    regions_shapefile/shapefile/era5_grid_path/lat_zone_edges); otherwise
    None. Same stale/foreign-cache guard convention as every other cache in
    this file.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, ValueError):
        print(f"  [cache] {path} is not readable JSON -- ignoring.")
        return None
    if payload.get("meta") != meta:
        print(f"  [cache] {path} was built with different settings {payload.get('meta')} "
              f"than requested {meta} -- ignoring and rebuilding.")
        return None
    swed_counts_df = _df_from_json(payload["swed_counts"])
    swbd_counts_df = _df_from_json(payload["swbd_counts"])
    land_area_pct = payload["land_area_pct"]
    zone_of_poly = {int(k): v for k, v in payload["zone_of_poly"].items()}
    area_of_poly = {int(k): v for k, v in payload["area_of_poly"].items()}
    return swed_counts_df, swbd_counts_df, land_area_pct, zone_of_poly, area_of_poly


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


def _share_fns(swed_counts_df, swbd_counts_df, zone_of_poly, area_of_poly, min_events=5):
    """
    {"SWED": fn, "SWBD": fn}, each fn(gwl, zone, x) -> that (zone, GWL)'s
    pooled duration-share curve evaluated at durations `x`, or None if it
    has too few events (min_events). SWED is swed_mod._group_counts's
    equal-GCM-weighted arr_share; SWBD is zone_group_counts_swbd's
    area-weighted mean of each region's equal-GCM-weighted curve -- both
    normalized by the group's total over *every* duration on record, so a
    curve evaluated on a cropped `x` is simply the full curve's head.
    Shared by the ratio and the share figures so both plot the same curves.
    """
    def _swed_share(gwl, zlabel, x):
        group = swed_mod._group_counts(swed_counts_df, gwl, zlabel, x)
        if group is None:
            return None
        arr_share, _, _, _, raw_total = group
        return arr_share if raw_total >= min_events else None

    def _swbd_share(gwl, zlabel, x):
        return zone_group_counts_swbd(
            swbd_counts_df, gwl, zlabel, zone_of_poly, area_of_poly, x,
            min_events=min_events)

    return {"SWED": _swed_share, "SWBD": _swbd_share}


def _add_map_row(fig, gs, zone_order, regions_shapefile, zone_of_poly, letters):
    """Row-0 locator maps (SWED latitude bands | SWBD region attribution),
    each with its title and panel letter -- shared by every figure here."""
    ax_map_swed, _ = swed_mod._add_locator_map(fig, gs[0, 0], zone_order)
    ax_map_swbd = _add_region_zone_map(fig, gs[0, 1], regions_shapefile, zone_of_poly)
    for ax_map, label, letter in ((ax_map_swed, "SWED", letters[0]),
                                   (ax_map_swbd, "SWBD (region attribution)", letters[1])):
        bbox = ax_map.get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.012, label, ha="center", va="bottom",
                  fontsize=swed_mod.ZONE_TITLE_FONTSIZE, fontweight="bold")
        fig.text(bbox.x0 - 0.02, bbox.y1 + 0.012, letter, ha="left", va="bottom",
                  fontsize=swed_mod.LETTER_FONTSIZE, fontweight="bold")


def _add_split_break(fig, ax_main, ax_zoom, main_ylim, zoom_ylim, split_day):
    """
    Broken-axis decorations between a short-term sub-panel and its
    persistent-event sub-panel: split-day line, x-limits are left to the
    caller, right-side y-ticks on the zoom panel, diagonal break marks on
    both spines, and dotted bridges from the split day in the main
    sub-panel (at the zoom sub-panel's own y-min/y-max) to the zoom
    sub-panel's left corners -- same convention as
    fig_duration_distribution_latitude.py's plot_distributions.
    """
    ax_main.axvline(split_day, color=swed_mod.DURATION_SPLIT_LINE_COLOR,
                     linewidth=0.8, linestyle="-", zorder=1)
    ax_zoom.yaxis.tick_right()

    d = 0.02  # half-length (axes fraction) of each diagonal break mark
    break_kwargs = dict(color="black", linewidth=0.8, clip_on=False, zorder=5)
    for y0 in (0, 1):
        ax_main.plot((1 - d, 1 + d), (y0 - d, y0 + d), transform=ax_main.transAxes, **break_kwargs)
        ax_zoom.plot((-d, d), (y0 - d, y0 + d), transform=ax_zoom.transAxes, **break_kwargs)
    for y_zoom, y_frac_zoom in ((zoom_ylim[1], 1), (zoom_ylim[0], 0)):
        y_anchor = min(max(y_zoom, main_ylim[0]), main_ylim[1])
        fig.add_artist(ConnectionPatch(
            xyA=(split_day, y_anchor), coordsA=ax_main.transData,
            xyB=(0, y_frac_zoom), coordsB=ax_zoom.transAxes,
            color=swed_mod.DURATION_SPLIT_LINE_COLOR, linewidth=0.7,
            linestyle=":", zorder=1,
        ))


def _gather_ratio_panel_data(swed_counts_df, swbd_counts_df, zone_of_poly, area_of_poly,
                              gwl_list, x_int, zone_order, min_events=5):
    """
    Pass-1 gather shared by plot_swed_swbd_distributions and
    plot_swed_swbd_distributions_split: for every (zone, {SWED, SWBD}) panel,
    each non-baseline GWL's duration-share ratio to the GWL0-61 baseline
    (see the two functions' own docstrings for what the ratio means and how
    gaps/isolated points are handled). Returns (panel_data, ratio_gwls) --
    panel_data[col] is a list, one dict per zone in zone_order, each holding
    {"ratios": {gwl: arr_or_None}}.
    """
    ratio_gwls = [g for g in gwl_list if g != BASELINE_GWL]
    share_fns = _share_fns(swed_counts_df, swbd_counts_df, zone_of_poly, area_of_poly,
                            min_events=min_events)

    panel_data = {col: [] for col in share_fns}
    for zlabel in zone_order:
        for col, share_fn in share_fns.items():
            base = share_fn(BASELINE_GWL, zlabel, x_int)
            ratios = {}
            for gwl in ratio_gwls:
                arr = share_fn(gwl, zlabel, x_int)
                if arr is None or base is None:
                    ratios[gwl] = None
                    continue
                base_safe = np.where(base > 0, base, np.nan)
                ratio = np.where(arr > 0, arr / base_safe, np.nan)
                ratio = _drop_isolated_points(ratio)
                ratios[gwl] = ratio if np.any(np.isfinite(ratio)) else None
            panel_data[col].append({"ratios": ratios})
    return panel_data, ratio_gwls


def _shared_ratio_ylim_windows(panel_data, split_idx, outlier_pct=2.0, pad=1.08):
    """
    Twin of _shared_ratio_ylim for plot_swed_swbd_distributions_split's
    broken-axis panels: two shared ranges instead of one -- (main_ylim,
    zoom_ylim) -- computed separately over every panel's short-term window
    (x_int[:split_idx], i.e. days 1..DURATION_SPLIT_DAY) and persistent
    window (x_int[split_idx - 1:], i.e. DURATION_SPLIT_DAY..max_duration_days,
    day DURATION_SPLIT_DAY itself repeated in both so the two sub-panels
    visually connect -- same overlap convention as
    fig_duration_distribution_latitude.py's plot_distributions). Each range
    is otherwise identical in spirit to _shared_ratio_ylim (outlier-percentile
    bound, folded around ratio=1, never narrower than a 4x/0.25x span) --
    just scoped to its own window rather than the full duration range, so the
    short-term sub-panel isn't stretched to fit the persistent tail's much
    larger ratio swings.
    """
    main_arrays, zoom_arrays = [], []
    for col in panel_data.values():
        for row in col:
            for arr in row["ratios"].values():
                if arr is None:
                    continue
                main_arrays.append(arr[:split_idx])
                zoom_arrays.append(arr[split_idx - 1:])
    main_ylim = _shared_ratio_ylim(main_arrays, outlier_pct=outlier_pct, pad=pad)
    zoom_ylim = _shared_ratio_ylim(zoom_arrays, outlier_pct=outlier_pct, pad=pad)
    return main_ylim, zoom_ylim


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

    # Pass 1: gather every panel's ratio curves up front, so the one shared
    # ratio y-axis can be fixed before anything is drawn. A duration with
    # too little data on either side of the ratio (baseline or GWL itself
    # short of min_events/zero events at that duration) is left as a NaN
    # gap, not dropped for the whole curve -- only points left isolated by
    # that gap (no valid neighbor on either side, so they'd draw as a
    # disconnected floating marker) are removed; see _drop_isolated_points.
    panel_data, ratio_gwls = _gather_ratio_panel_data(
        swed_counts_df, swbd_counts_df, zone_of_poly, area_of_poly,
        gwl_list, x_int, zone_order, min_events=min_events)

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

    _add_map_row(fig, gs, zone_order, regions_shapefile, zone_of_poly, all_letters)

    for i, zlabel in enumerate(zone_order):
        pct = land_area_pct.get(zlabel)
        lat_range_text = swed_mod._fmt_lat_range(zlabel)
        zone_label_text = (f"{lat_range_text} ({pct:.1f}% of land area)" if pct is not None
                            else lat_range_text)

        for j, col in enumerate(panel_data):
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


def plot_swed_swbd_distributions_split(swed_counts_df, swbd_counts_df, land_area_pct,
                                        zone_of_poly, area_of_poly, regions_shapefile,
                                        gwl_list, output_path, dpi=300,
                                        max_duration_days=12.0, min_events=5,
                                        duration_split_day=DURATION_SPLIT_DAY_DEFAULT):
    """
    Broken-axis twin of plot_swed_swbd_distributions -- same data, same
    2-column x 6-row layout, locator maps, ratio definition, gap/isolated-
    point handling and GWL styling, but each of the 10 ratio panels is now
    itself split into two side-by-side sub-panels (same convention as
    fig_duration_distribution_latitude.py's plot_distributions main/zoom
    split): a wide "short term" sub-panel (day 1..duration_split_day, ratios
    that stay close to 1) and a narrower "persistent event" sub-panel
    (duration_split_day..max_duration_days, where ratios can swing much
    farther from 1). Each sub-panel type gets its own shared log-scale
    y-range (_shared_ratio_ylim_windows) instead of the single figure-wide
    range plot_swed_swbd_distributions uses -- so the short-term sub-panel
    isn't stretched flat by whatever wide range the persistent tail needs,
    and small near-1 differences between GWLs actually become visible. Day
    duration_split_day is repeated at the start of the persistent sub-panel
    (same overlap convention as the other file) so the break reads as a
    continuation, reinforced by dotted connector lines and diagonal break
    marks on both spines at the split.

    Produces a second, independent output file -- plot_swed_swbd_distributions
    itself is untouched, so the single-axis version stays available for
    direct comparison against this split one.
    """
    x_int = np.arange(1, int(np.ceil(max_duration_days)) + 1)
    zone_order = list(reversed(swed_mod.LAT_ZONE_LABELS))
    split_idx = int(duration_split_day)  # x_int[:split_idx] == days 1..duration_split_day
    do_split = max_duration_days > duration_split_day

    panel_data, ratio_gwls = _gather_ratio_panel_data(
        swed_counts_df, swbd_counts_df, zone_of_poly, area_of_poly,
        gwl_list, x_int, zone_order, min_events=min_events)

    if do_split:
        main_ylim, zoom_ylim = _shared_ratio_ylim_windows(panel_data, split_idx)
    else:
        main_ylim = zoom_ylim = _shared_ratio_ylim(
            [arr for col in panel_data.values() for row in col for arr in row["ratios"].values()])
    main_yticks, zoom_yticks = _ratio_yticks(main_ylim), _ratio_yticks(zoom_ylim)

    n_rows = len(zone_order)
    fig = plt.figure(figsize=(swed_mod.FIG_WIDTH_IN * 1.55, swed_mod.FIG_WIDTH_IN * 1.75))
    gs = GridSpec(n_rows + 1, 2, height_ratios=[0.8] + [1.0] * n_rows,
                  left=0.11, right=0.97, top=0.93, bottom=0.09,
                  hspace=0.65, wspace=0.30, figure=fig)

    all_letters = [chr(ord("a") + k) for k in range(2 * (n_rows + 1))]

    _add_map_row(fig, gs, zone_order, regions_shapefile, zone_of_poly, all_letters)

    for i, zlabel in enumerate(zone_order):
        pct = land_area_pct.get(zlabel)
        lat_range_text = swed_mod._fmt_lat_range(zlabel)
        zone_label_text = (f"{lat_range_text} ({pct:.1f}% of land area)" if pct is not None
                            else lat_range_text)

        for j, col in enumerate(panel_data):
            row = panel_data[col][i]

            if do_split:
                inner_gs = GridSpecFromSubplotSpec(
                    1, 2, subplot_spec=gs[i + 1, j],
                    width_ratios=swed_mod.DURATION_ZOOM_WIDTH_RATIOS, wspace=0.08)
                ax_main = fig.add_subplot(inner_gs[0])
                # No sharey with ax_main: the persistent-event sub-panel gets
                # its own (usually wider) shared range, computed separately
                # by _shared_ratio_ylim_windows.
                ax_zoom = fig.add_subplot(inner_gs[1])
                windows = [(ax_main, slice(0, split_idx)), (ax_zoom, slice(split_idx - 1, None))]
            else:
                ax_main = fig.add_subplot(gs[i + 1, j])
                ax_zoom = None
                windows = [(ax_main, slice(None))]

            for a, _sl in windows:
                a.axhline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=1)
            for gwl in ratio_gwls:
                arr = row["ratios"].get(gwl)
                if arr is None:
                    continue
                color = swed_mod.GWL_COLORS.get(gwl, "gray")
                for a, sl in windows:
                    a.plot(x_int[sl], arr[sl], color=color, marker="o", markersize=2.2,
                           linewidth=1.3, zorder=3)

            for a, ylim, yticks in ((ax_main, main_ylim, main_yticks),
                                     (ax_zoom, zoom_ylim, zoom_yticks)):
                if a is None:
                    continue
                a.set_yscale("log")
                a.set_ylim(*ylim)
                a.set_yticks(yticks)
                a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_ratio_tick_label))
                a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                a.tick_params(labelsize=swed_mod.TICK_FONTSIZE)
                a.grid(True, linestyle="--", alpha=0.3)
                for spine in a.spines.values():
                    spine.set_linewidth(0.4)

            if do_split:
                ax_main.set_xlim(0.5, duration_split_day + 0.5)
                ax_zoom.set_xlim(duration_split_day - 0.5, max_duration_days + 0.5)
                ax_main.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                ax_zoom.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=6))
                _add_split_break(fig, ax_main, ax_zoom, main_ylim, zoom_ylim, duration_split_day)
            else:
                ax_main.set_xlim(0.5, max_duration_days + 0.5)
                ax_main.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

            ax_main.spines["left"].set_color(swed_mod.ZONE_MAP_COLORS[zlabel])
            ax_main.spines["left"].set_linewidth(2.2)

            panel_letter = all_letters[2 + i * 2 + j]
            ax_main.text(0.0, 1.05, panel_letter, transform=ax_main.transAxes,
                         ha="left", va="bottom", fontsize=swed_mod.LETTER_FONTSIZE,
                         fontweight="bold")
            if j == 0:
                ax_main.set_ylabel(RATIO_YLABEL, fontsize=swed_mod.AXIS_LABEL_FONTSIZE)
                ax_main.text(0.13, 1.05, zone_label_text, transform=ax_main.transAxes,
                             ha="left", va="bottom", fontsize=swed_mod.ZONE_TITLE_FONTSIZE,
                             fontweight="bold", color=swed_mod.ZONE_MAP_COLORS[zlabel])
            if i == n_rows - 1:
                # Placed on the persistent-event sub-panel when split (its
                # own axis, not shared with the short-term one) so the label
                # doesn't have to describe both windows at once.
                (ax_zoom if ax_zoom is not None else ax_main).set_xlabel(
                    "Event duration (days)", fontsize=swed_mod.XLABEL_FONTSIZE)

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

    plot_data_path = args.plot_data_json or os.path.join(
        args.output_dir, "swed_swbd_plot_data_cache.json")
    plot_data_meta = _plot_data_cache_meta(
        args.regions_shapefile, args.shapefile, args.era5_grid_path,
        args.threshold, args.ssp, args.swbd_thr, args.swbd_tot_re, args.swbd_mix, args.suffix_shp)
    cached_plot_data = None if args.recompute_plot_data else load_plot_data_cache(
        plot_data_path, plot_data_meta)

    if cached_plot_data is not None:
        print("=" * 60)
        swed_counts_df, swbd_counts_df, land_area_pct, zone_of_poly, area_of_poly = cached_plot_data
        print(f"Loaded full plot-data bundle from {plot_data_path} -- skipping every raw-data "
              f"step (no shapefiles/ERA5 grid/preprocessed archive access needed).")
        print("=" * 60)
    else:
        print("=" * 60)
        print("STEP 1/2 - Region/zone geometry: assignment + area weights + land area share")
        print("=" * 60)
        geometry_cache_path = args.geometry_cache_json or os.path.join(
            args.output_dir, "swed_swbd_geometry_cache.json")
        cached_geometry = None if args.recompute_geometry else load_geometry_cache(
            geometry_cache_path, plot_data_meta["geometry"])
        if cached_geometry is not None:
            land_area_pct, zone_of_poly, area_of_poly = cached_geometry
            print(f"  Loaded cached geometry from {geometry_cache_path} -- skipping "
                  f"--regions_shapefile/--shapefile/--era5_grid_path access.")
        else:
            zone_of_poly, area_of_poly = assign_regions_to_zones(args.regions_shapefile)
            print(f"  {len(zone_of_poly)} regions assigned across "
                  f"{len(set(zone_of_poly.values()))} zones")
            era5_lat, era5_lon = swed_mod.load_era5_reference_grid(
                args.preprocessed_path, era5_grid_path=args.era5_grid_path)
            land_area_pct = swed_mod.compute_land_area_share_per_zone(
                era5_lat, era5_lon, args.shapefile)
            for z, pct in land_area_pct.items():
                print(f"  {z}: {pct:.1f}% of land area")
            save_geometry_cache(geometry_cache_path, land_area_pct, zone_of_poly, area_of_poly,
                                 plot_data_meta["geometry"])
            print(f"  Saved geometry cache -> {geometry_cache_path}")

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
            swbd_cache_path, args.swbd_thr, args.swbd_tot_re, args.swbd_mix, args.ssp,
            args.suffix_shp)
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

        save_plot_data_cache(plot_data_path, swed_counts_df, swbd_counts_df, land_area_pct,
                              zone_of_poly, area_of_poly, plot_data_meta)
        print(f"\n  Saved full plot-data bundle -> {plot_data_path}")

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

    out_path_split = os.path.join(
        args.output_dir, "fig_duration_distribution_swed_swbd_ratio_split.png")
    plot_swed_swbd_distributions_split(
        swed_counts_df, swbd_counts_df, land_area_pct, zone_of_poly, area_of_poly,
        args.regions_shapefile, args.gwl_list, out_path_split, dpi=args.dpi,
        max_duration_days=args.max_duration_days, min_events=args.min_events,
        duration_split_day=args.duration_split_day,
    )
    print(f"  Saved -> {out_path_split}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Extended Data Fig. 6: SWED and SWBD event-duration distributions by latitude
band and warming level.

- SWED: per-pixel compound wind+solar drought events, read from a cached
  counts table (see duration_latitude.build_counts_table).
- SWBD: residual-load exceedance days per admin region (make_rl_files.py's
  formula at daily resolution), run-length encoded into events.

Each SWBD region is assigned to the latitude band it overlaps most (by area).
Duration shares are pooled per band, weighting regions by area. SWED
realizations are weighted by inverse Wasserstein distance (--weighting w2) or
equally per GCM (mmm); SWBD realizations always equally per GCM. Two figures are written: the ratio of each GWL's duration
shares to the GWL0-61 baseline, and the shares themselves.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; puts the code folders on sys.path
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
import cartopy.feature as cfeature

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

# SWED constants and helpers. make_rl_files is imported lazily where SWBD is
# computed from raw data, since it needs xclim (HPC only); plotting from the
# caches doesn't.
import duration_latitude as swed_mod

# make_rl_files.py's MAIN_THR/MAIN_TOT_RE/MAIN_MIX, copied to avoid importing it.
SWBD_MAIN_THR    = 0.99
SWBD_MAIN_TOT_RE = 0.5
SWBD_MAIN_MIX    = "current"

BASELINE_GWL  = "GWL0-61"
RATIO_YLABEL  = "Relative change in\nnumber of drought\nevents"

# Broken x-axis: short-term events left of this day, persistent ones right.
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
                         help="Admin-region shapefile defining SWBD's poly_idx.")
    parser.add_argument(
        "--gwl_list", nargs="+", default=swed_mod.GWL_KEYS,
        help="GWL keys to include (default: GWL0-61 GWL1-5 GWL2 GWL3).",
    )
    parser.add_argument("--threshold", type=float, default=0.1,
                         help="SWED quantile threshold.")
    parser.add_argument("--swbd_thr", type=float, default=SWBD_MAIN_THR,
                         help="SWBD residual-load quantile threshold.")
    parser.add_argument("--swbd_tot_re", type=float, default=SWBD_MAIN_TOT_RE,
                         help="SWBD renewable-penetration fraction.")
    parser.add_argument("--swbd_mix", default=SWBD_MAIN_MIX,
                         help="SWBD solar/wind mix key.")
    parser.add_argument("--exclude_gcm_run", nargs="+", default=config.EXCLUDE_GCM_RUN)
    parser.add_argument("--shapefile", default=config.SHAPEFILE_PATH,
                         help="Land shapefile for the SWED side.")
    parser.add_argument("--era5_grid_path", default=None)
    parser.add_argument("--swed_cache_csv", default=None,
                         help="SWED counts cache (default: "
                              "<output_dir>/event_duration_counts_cache.csv).")
    parser.add_argument("--swbd_cache_csv", default=None,
                         help="SWBD counts cache (default: "
                              "<output_dir>/swbd_duration_counts_cache.csv).")
    parser.add_argument("--recompute_swbd", action="store_true", default=False)
    parser.add_argument("--geometry_cache_json", default=None,
                         help="Region/zone geometry cache (default: "
                              "<output_dir>/swed_swbd_geometry_cache.json).")
    parser.add_argument("--recompute_geometry", action="store_true", default=False,
                         help="Rebuild the geometry cache from the shapefiles.")
    parser.add_argument("--plot_data_json", default=None,
                         help="Single-file bundle of all plotting inputs (default: "
                              "<output_dir>/swed_swbd_plot_data_cache.json); enough on its "
                              "own to redraw the figures without any raw data.")
    parser.add_argument("--recompute_plot_data", action="store_true", default=False,
                         help="Ignore the plot-data bundle and rebuild it.")
    parser.add_argument("--output_dir", default="../final_figs")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max_duration_days", type=float, default=12.0,
                         help="X-axis cap (days) of the ratio figure; beyond ~12 days "
                              "there are too few events for a meaningful ratio.")
    parser.add_argument("--share_max_duration_days", type=float,
                         default=swed_mod.FULL_FIGURE_MAX_DURATION_DAYS,
                         help="X-axis cap (days) of the share figure.")
    parser.add_argument("--min_events", type=int, default=5)
    parser.add_argument("--weighting", choices=["w2", "mmm"], default="w2",
                        help="SWED only -- 'w2': inverse-Wasserstein weights; 'mmm': equal "
                             "weight per GCM. SWBD always uses equal weight per GCM.")
    parser.add_argument("--wasserstein_path", default=config.WASSERSTEIN_NC_PATH,
                        help="Per-pixel Wasserstein file (SWED zone weights).")
    parser.add_argument("--duration_split_day", type=float, default=DURATION_SPLIT_DAY_DEFAULT,
                         help="Day splitting short-term from persistent events.")
    return parser.parse_args()


# =============================================================================
# Region -> latitude-zone assignment (max overlap area) + region area weight
# =============================================================================

def assign_regions_to_zones(shapefile_path):
    """Assign each region (poly_idx = row index) to the latitude zone it overlaps
    most by area, in an equal-area projection (EPSG:6933).
    Returns ({poly_idx: zone_label}, {poly_idx: area_m2})."""
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
    """SWBD locator map: each region colored by its assigned zone (gray if none)."""
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
    """Daily (time, poly_idx) boolean SWBD field for one (GCM, run, GWL): residual
    load above the GWL0-61 `thr`-quantile (make_rl_files._calculate_rl)."""
    import make_rl_files  # lazy: needs xclim
    gwl_ref = "GWL0-61"
    dtas_ref, dds_cf_ref, dds_cf_ref_mean = make_rl_files._load_data(
        GCM, run, ssp, gwl_ref, reanalysis, suffix_shp, path_preprocessed, df_share, mix)
    _, threshold, demand_bas, exceeds_ref = make_rl_files._calculate_rl(
        dtas_ref, dds_cf_ref, dds_cf_ref_mean, thr, "Annual", tot_re,
        demand_cfg=demand_cfg, return_daily=True)

    if gwl == gwl_ref:
        exceeds = exceeds_ref
    else:
        dtas, dds_cf, _ = make_rl_files._load_data(
            GCM, run, ssp, gwl, reanalysis, suffix_shp, path_preprocessed, df_share, mix)
        _, _, _, exceeds = make_rl_files._calculate_rl(
            dtas, dds_cf, dds_cf_ref_mean, thr, "Annual", tot_re,
            threshold=threshold, demand_bas=demand_bas,
            demand_cfg=demand_cfg, return_daily=True)

    exceeds["time"] = pd.to_datetime(exceeds["time"].dt.strftime("%Y-%m-%d").values)
    return exceeds


def compute_event_table_by_poly(da, time_dim="time"):
    """duration_decomposition.compute_event_table over poly_idx instead of
    (lat, lon): one row per (event day, poly_idx) with the event's duration."""
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
    """Number of SWBD events per (gwl, GCM, run, poly_idx, duration). Expensive:
    reads every realization's aggregated files."""
    import make_rl_files  # lazy: needs xclim
    exclude_pairs = set(tuple(x.split(":")) for x in (exclude_gcm_run or []))
    rows = []
    for gwl in gwl_list:
        print(f"\n  -- {gwl} --")
        for GCM, run in make_rl_files._iter_gcm_runs(path_preprocessed, ssp, reanalysis, suffix_shp):
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
    """Write the CSV with a '#' JSON metadata first line (checked on load)."""
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
# Geometry cache (land_area_pct, zone_of_poly, area_of_poly) as one JSON file
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
    """(land_area_pct, zone_of_poly, area_of_poly) if the cache matches `meta`,
    else None."""
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
# Plot-data bundle: every plotting input in one JSON file, enough to redraw
# the figures without any raw data (the three caches above stay the primary,
# independently invalidated ones)
# =============================================================================

def _plot_data_cache_meta(regions_shapefile, shapefile, era5_grid_path,
                           threshold, ssp, swbd_thr, swbd_tot_re, swbd_mix, suffix_shp):
    """Settings of the SWED counts, SWBD counts and geometry caches combined."""
    return {
        "swed": swed_mod._cache_meta(threshold, ssp),
        "swbd": _swbd_cache_meta(swbd_thr, swbd_tot_re, swbd_mix, ssp, suffix_shp),
        "geometry": _geometry_cache_meta(regions_shapefile, shapefile, era5_grid_path),
    }


def _df_to_json(df):
    """Compact JSON encoding: column names once, then row lists."""
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
    """(swed_counts_df, swbd_counts_df, land_area_pct, zone_of_poly,
    area_of_poly) if the bundle matches `meta`, else None."""
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
    """Duration-share curve of one region: normalized per GCM, then averaged
    with weight 1/n_GCM. Returns (arr_share, raw_total) or None."""
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
    """Area-weighted mean of the zone's regional curves (_region_group_counts),
    skipping regions with fewer than min_events events. None if none qualify."""
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


def load_ensemble_weights(args):
    """{"SWED": zone weights} for --weighting w2 (None for mmm). SWBD is
    always an equal-weight multi-model mean. A missing file is an error, not
    a silent fallback."""
    if args.weighting != "w2":
        return None
    if not os.path.exists(args.wasserstein_path):
        raise SystemExit(f"Wasserstein file not found: {args.wasserstein_path} -- run "
                         "main_pipeline/trend_sev_eval_wasserstein.py first, or "
                         "pass --weighting mmm.")
    ds = xr.open_dataset(args.wasserstein_path)
    swed = swed_mod.zone_inv_w2(ds, args.shapefile)
    ds.close()
    return {"SWED": swed}


# =============================================================================
# Plotting
# =============================================================================

def _shared_ratio_ylim(ratio_arrays, outlier_pct=2.0, pad=1.08):
    """Log-symmetric ylim shared by all ratio panels, bounded by the
    (100 - outlier_pct) percentile of |log10 ratio| so rare outliers are
    clipped; never narrower than 1/4..4."""
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
    """NaN out points with no valid neighbor on either side (they would draw as
    lone markers); runs of 2+ valid points are kept."""
    valid = np.isfinite(arr)
    left_valid = np.concatenate(([False], valid[:-1]))
    right_valid = np.concatenate((valid[1:], [False]))
    isolated = valid & ~left_valid & ~right_valid
    out = arr.copy()
    out[isolated] = np.nan
    return out


def _share_fns(swed_counts_df, swbd_counts_df, zone_of_poly, area_of_poly, min_events=5,
               weights=None):
    """{"SWED": fn, "SWBD": fn} with fn(gwl, zone, x) -> pooled duration-share
    curve at durations x (normalized over all durations), or None if it has
    fewer than min_events events. Used by both figures."""
    def _swed_share(gwl, zlabel, x):
        inv = None if weights is None else weights["SWED"].get(zlabel, {})
        group = swed_mod._group_counts(swed_counts_df, gwl, zlabel, x, inv_w2=inv)
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
    """Top-row locator maps (SWED latitude bands | SWBD regions) with titles."""
    ax_map_swed, _ = swed_mod._add_locator_map(fig, gs[0, 0], zone_order)
    # Mask the ocean part of the full-width zone bands (zorder between the
    # bands at 1 and the zone-edge lines at 2), so only land is shaded.
    ax_map_swed.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="white",
                            edgecolor="none", zorder=1.5)
    ax_map_swbd = _add_region_zone_map(fig, gs[0, 1], regions_shapefile, zone_of_poly)
    for ax_map, label, letter in ((ax_map_swed, "SWED", letters[0]),
                                   (ax_map_swbd, "SWBD (region attribution)", letters[1])):
        bbox = ax_map.get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.012, label, ha="center", va="bottom",
                  fontsize=swed_mod.ZONE_TITLE_FONTSIZE, fontweight="bold")
        fig.text(bbox.x0 - 0.02, bbox.y1 + 0.012, letter, ha="left", va="bottom",
                  fontsize=swed_mod.LETTER_FONTSIZE, fontweight="bold")


def _add_split_break(fig, ax_main, ax_zoom, main_ylim, zoom_ylim, split_day):
    """Broken-axis decorations: split-day line, break marks and dotted
    connectors to the zoom sub-panel."""
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
                              gwl_list, x_int, zone_order, min_events=5, weights=None):
    """Ratio of each GWL's duration shares to the baseline, per (zone, SWED/SWBD).
    Returns (panel_data, ratio_gwls), panel_data[col][zone_i]["ratios"][gwl]."""
    ratio_gwls = [g for g in gwl_list if g != BASELINE_GWL]
    share_fns = _share_fns(swed_counts_df, swbd_counts_df, zone_of_poly, area_of_poly,
                            min_events=min_events, weights=weights)

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
    """_shared_ratio_ylim computed separately for the short-term and persistent
    windows (the split day belongs to both). Returns (main_ylim, zoom_ylim)."""
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
                                  max_duration_days=12.0, min_events=5, weights=None):
    """Ratio figure on a single axis: columns SWED | SWBD, a locator-map row,
    then one row per latitude zone showing each GWL's duration shares divided
    by the GWL0-61 baseline (> 1: more frequent than at baseline)."""
    x_int = np.arange(1, int(np.ceil(max_duration_days)) + 1)
    zone_order = list(reversed(swed_mod.LAT_ZONE_LABELS))

    # Gather all ratios first so the shared y-axis is fixed before drawing.
    panel_data, ratio_gwls = _gather_ratio_panel_data(
        swed_counts_df, swbd_counts_df, zone_of_poly, area_of_poly,
        gwl_list, x_int, zone_order, min_events=min_events, weights=weights)

    all_ratio_arrays = [arr for col in panel_data.values() for row in col
                         for arr in row["ratios"].values()]
    ratio_ylim = _shared_ratio_ylim(all_ratio_arrays)
    ratio_yticks = _ratio_yticks(ratio_ylim)

    n_rows = len(zone_order)
    fig = plt.figure(figsize=(swed_mod.FIG_WIDTH_IN * 1.55, swed_mod.FIG_WIDTH_IN * 1.75))
    gs = GridSpec(n_rows + 1, 2, height_ratios=[0.8] + [1.0] * n_rows,
                  left=0.11, right=0.97, top=0.93, bottom=0.09,
                  hspace=0.65, wspace=0.30, figure=fig)

    # One panel letter per panel, row-major.
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
                                        duration_split_day=DURATION_SPLIT_DAY_DEFAULT,
                                        weights=None):
    """plot_swed_swbd_distributions with a broken x-axis at duration_split_day:
    short-term and persistent sub-panels each get their own shared y-range."""
    x_int = np.arange(1, int(np.ceil(max_duration_days)) + 1)
    zone_order = list(reversed(swed_mod.LAT_ZONE_LABELS))
    split_idx = int(duration_split_day)  # x_int[:split_idx] == days 1..duration_split_day
    do_split = max_duration_days > duration_split_day

    panel_data, ratio_gwls = _gather_ratio_panel_data(
        swed_counts_df, swbd_counts_df, zone_of_poly, area_of_poly,
        gwl_list, x_int, zone_order, min_events=min_events, weights=weights)

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
                ax_zoom = fig.add_subplot(inner_gs[1])  # own y-range, no sharey
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


def _shared_log_ylim(arrays, pad_decades=0.12):
    """(lo, hi) spanning every finite, positive value in `arrays`, padded by
    `pad_decades` on a log scale; None if there is nothing to span."""
    vals = [a[np.isfinite(a) & (a > 0)] for a in arrays]
    vals = np.concatenate([v for v in vals if v.size] or [np.array([])])
    if vals.size == 0:
        return None
    return (10.0 ** (np.log10(vals.min()) - pad_decades),
            10.0 ** (np.log10(vals.max()) + pad_decades))


def plot_swed_swbd_share_split(swed_counts_df, swbd_counts_df, land_area_pct,
                                zone_of_poly, area_of_poly, regions_shapefile,
                                gwl_list, output_path, dpi=300,
                                max_duration_days=swed_mod.FULL_FIGURE_MAX_DURATION_DAYS,
                                min_events=5, duration_split_day=DURATION_SPLIT_DAY_DEFAULT,
                                weights=None):
    """Same layout as plot_swed_swbd_distributions_split, but plotting the
    duration shares themselves (all GWLs, log y-axis). Dashed / dotted lines
    mark each GWL's mean and q-HIGH_DURATION_PCTILE duration, computed over
    the full duration record."""
    x_int = np.arange(1, int(np.ceil(max_duration_days)) + 1)
    zone_order = list(reversed(swed_mod.LAT_ZONE_LABELS))
    split_idx = int(duration_split_day)  # x_int[:split_idx] == days 1..duration_split_day
    do_split = max_duration_days > duration_split_day
    share_fns = _share_fns(swed_counts_df, swbd_counts_df, zone_of_poly, area_of_poly,
                            min_events=min_events, weights=weights)
    counts_by_col = {"SWED": swed_counts_df, "SWBD": swbd_counts_df}

    # Gather all curves first (over the full record) to fix the shared y-ranges.
    panel_data = {col: [] for col in share_fns}
    for col, share_fn in share_fns.items():
        df = counts_by_col[col]
        max_rec = int(df["duration"].max()) if not df.empty else len(x_int)
        x_full = np.arange(1, max(max_rec, len(x_int)) + 1)
        for zlabel in zone_order:
            curves = {}
            for gwl in gwl_list:
                full = share_fn(gwl, zlabel, x_full)
                if full is None or full.sum() <= 0:
                    continue
                curves[gwl] = {
                    "share": swed_mod._for_line(full[:len(x_int)]),
                    "mean": float((x_full * full).sum() / full.sum()),
                    "p_high": swed_mod._weighted_percentile(
                        x_full, full, swed_mod.HIGH_DURATION_PCTILE),
                }
            panel_data[col].append(curves)

    all_shares = [c["share"] for col in panel_data.values() for row in col for c in row.values()]
    if do_split:
        main_ylim = _shared_log_ylim([a[:split_idx] for a in all_shares])
        zoom_ylim = _shared_log_ylim([a[split_idx - 1:] for a in all_shares])
    else:
        main_ylim = zoom_ylim = _shared_log_ylim(all_shares)

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
            curves = panel_data[col][i]

            if do_split:
                inner_gs = GridSpecFromSubplotSpec(
                    1, 2, subplot_spec=gs[i + 1, j],
                    width_ratios=swed_mod.DURATION_ZOOM_WIDTH_RATIOS, wspace=0.08)
                ax_main = fig.add_subplot(inner_gs[0])
                ax_zoom = fig.add_subplot(inner_gs[1])
                windows = [(ax_main, slice(0, split_idx)), (ax_zoom, slice(split_idx - 1, None))]
            else:
                ax_main = fig.add_subplot(gs[i + 1, j])
                ax_zoom = None
                windows = [(ax_main, slice(None))]

            for gwl in gwl_list:
                c = curves.get(gwl)
                if c is None:
                    continue
                color = swed_mod.GWL_COLORS.get(gwl, "gray")
                for a, sl in windows:
                    a.plot(x_int[sl], c["share"][sl], color=color, marker="o", markersize=2.2,
                           linewidth=1.3, zorder=3)
                    a.axvline(c["mean"], color=color, linestyle="--", linewidth=1.0,
                              alpha=0.75, zorder=4)
                    a.axvline(c["p_high"], color=color, linestyle=":", linewidth=1.0,
                              alpha=0.75, zorder=4)

            for a, ylim in ((ax_main, main_ylim), (ax_zoom, zoom_ylim)):
                if a is None:
                    continue
                a.set_yscale("log")
                if ylim is not None:
                    a.set_ylim(*ylim)
                a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                a.tick_params(labelsize=swed_mod.TICK_FONTSIZE)
                a.grid(True, linestyle="--", alpha=0.3)
                for spine in a.spines.values():
                    spine.set_linewidth(0.4)

            if do_split:
                ax_main.set_xlim(0.5, duration_split_day + 0.5)
                ax_zoom.set_xlim(duration_split_day - 0.5, max_duration_days + 0.5)
                ax_main.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                ax_zoom.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=6))
                if main_ylim is not None and zoom_ylim is not None:
                    _add_split_break(fig, ax_main, ax_zoom, main_ylim, zoom_ylim,
                                     duration_split_day)
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
                ax_main.set_ylabel("Share of events", fontsize=swed_mod.AXIS_LABEL_FONTSIZE)
                ax_main.text(0.13, 1.05, zone_label_text, transform=ax_main.transAxes,
                             ha="left", va="bottom", fontsize=swed_mod.ZONE_TITLE_FONTSIZE,
                             fontweight="bold", color=swed_mod.ZONE_MAP_COLORS[zlabel])
            if i == n_rows - 1:
                (ax_zoom if ax_zoom is not None else ax_main).set_xlabel(
                    "Event duration (days)", fontsize=swed_mod.XLABEL_FONTSIZE)

    gwl_handles = [
        Line2D([0], [0], color=swed_mod.GWL_COLORS.get(gwl, "gray"), marker="o",
               markersize=3, linewidth=1.6, label=swed_mod.GWL_LABELS.get(gwl, gwl))
        for gwl in gwl_list
    ]
    extra_handles = [
        Line2D([0], [0], color="black", linewidth=1.0, linestyle="--", label="Mean duration"),
        Line2D([0], [0], color="black", linewidth=1.0, linestyle=":",
               label=f"q{swed_mod.HIGH_DURATION_PCTILE:g} duration"),
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

    # Weights are applied at pooling time, so the counts caches above stay valid.
    weights = load_ensemble_weights(args)

    print("\n" + "=" * 60)
    print("STEP 5 - Plotting")
    print("=" * 60)
    out_path_split = os.path.join(
        args.output_dir, "suppfig6_duration_distribution_swed_swbd_ratio.png")
    plot_swed_swbd_distributions_split(
        swed_counts_df, swbd_counts_df, land_area_pct, zone_of_poly, area_of_poly,
        args.regions_shapefile, args.gwl_list, out_path_split, dpi=args.dpi,
        max_duration_days=args.max_duration_days, min_events=args.min_events,
        duration_split_day=args.duration_split_day, weights=weights,
    )
    print(f"  Saved -> {out_path_split}")

    out_path_share = os.path.join(
        args.output_dir, "fig_duration_distribution_swed_swbd_share_split.png")
    plot_swed_swbd_share_split(
        swed_counts_df, swbd_counts_df, land_area_pct, zone_of_poly, area_of_poly,
        args.regions_shapefile, args.gwl_list, out_path_share, dpi=args.dpi,
        max_duration_days=args.share_max_duration_days, min_events=args.min_events,
        duration_split_day=args.duration_split_day, weights=weights,
    )
    print(f"  Saved -> {out_path_share}")


if __name__ == "__main__":
    main()

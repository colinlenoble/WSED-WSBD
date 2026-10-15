"""
Helpers for suppfig6_duration_distribution_swed_swbd.py (ED Fig. 6): latitude
zones, SWED event-duration counts and their cache, ensemble pooling, and
shared plot styling.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; puts the code folders on sys.path
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
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

from io_utils import match_files, glob_any, open_dataset_any
from duration_decomposition import compute_event_table

# =============================================================================
# Figure size, font sizes and latitude bands
# =============================================================================
FIG_WIDTH_IN = 5.15   # single column width, as in fig1.py

XLABEL_FONTSIZE = 8     # shared x-axis label
LEGEND_FONTSIZE = 7     # bottom GWL legend
LETTER_FONTSIZE = 8     # bold panel letters (a, b, c, ...)
ZONE_TITLE_FONTSIZE = 8  # per-panel zone name, next to its letter
AXIS_LABEL_FONTSIZE = 7.5  # per-panel y-axis label
TICK_FONTSIZE = 7       # per-panel tick labels

# Broken x-axis: short-term events up to this day (included in both halves),
# persistent ones in a zoomed panel, so the tail isn't crushed.
DURATION_SPLIT_DAY = 5
DURATION_ZOOM_WIDTH_RATIOS = [45, 55]
DURATION_SPLIT_LINE_COLOR = "#777777"

HIGH_DURATION_PCTILE = 99.5  # percentile marked as the "high duration" line

FULL_FIGURE_MAX_DURATION_DAYS = 20.0  # x-axis cap of the share figure

MAP_LAT_SOUTH = -58.0
MAP_LAT_NORTH = 68.0

# Locator-map longitude range (cropped for spacing next to the panels).
MAP_LON_WEST = -155.0
MAP_LON_EAST = 155.0

LAT_ZONE_EDGES = [MAP_LAT_SOUTH, -40, -20, 20, 40, MAP_LAT_NORTH]
LAT_ZONE_LABELS = [
    "Midlatitude (S)", "Subtropical (S)", "Tropical", "Subtropical (N)", "Midlatitude (N)",
]
ZONE_BOUNDS = {
    label: (LAT_ZONE_EDGES[i], LAT_ZONE_EDGES[i + 1])
    for i, label in enumerate(LAT_ZONE_LABELS)
}
# Zone colours (ColorBrewer Dark2), chosen to avoid the GWL blues/reds.
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
# Land mask
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
    """(lat, lon) of the ERA5 grid within MAP_LAT_SOUTH..MAP_LAT_NORTH, read from
    preprocessed_path/ERA5/wcf_day* or from any file on that grid
    (era5_grid_path; only coordinates are read)."""
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
    """Share (%) of total land area in each latitude zone, on the ERA5 grid,
    with land pixels from the shapefile weighted by cos(latitude)."""
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
    """(GCM, run) pairs with a wcf_day_{GCM}_{ssp}_{run}_{gwl}_ERA5 file."""
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
    """Daily compound field (wcf and scf both at or below their GWL0-61
    `threshold` quantile of positive values), as in fig1.py/fig3.py, on the
    GCM's native grid."""
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
    """One row per SWED event at a land pixel: duration (days), start year and
    latitude zone (ocean set to 0, so no events)."""
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
    """Number of SWED events per (gwl, GCM, run, zone, duration). Expensive
    (reads every daily wcf/scf file), hence cached by the caller."""
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
    """Write the counts CSV with a '#' JSON metadata first line (zones,
    threshold, ssp), checked on load."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(f"# {json.dumps(_cache_meta(threshold, ssp))}\n")
        counts_df.to_csv(f, index=False)


def load_counts_cache(path, threshold, ssp):
    """Cached counts if the metadata line matches, else None."""
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


def zone_inv_w2(ds_wasserstein, shapefile_path, var="w2_normalized", eps=1e-3):
    """{zone: {(GCM, run): 1 / max(mean_w2, eps)}}, mean_w2 being the
    cos(lat)-weighted mean w2_normalized over the zone's land pixels."""
    ds_wasserstein = ds_wasserstein.sortby("lat")   # build_land_mask_from_grid assumes ascending lat
    lat = ds_wasserstein.lat.values
    lon = ds_wasserstein.lon.values
    land = build_land_mask_from_grid(lat, lon, shapefile_path)
    in_band = (lat >= LAT_ZONE_EDGES[0]) & (lat <= LAT_ZONE_EDGES[-1])
    zone_of_row = assign_lat_zone(lat)
    w2 = ds_wasserstein[var].values                       # (realization, lat, lon)
    gcms = np.asarray(ds_wasserstein.GCM.values).astype(str)
    runs = np.asarray(ds_wasserstein.run.values).astype(str)

    out = {}
    for zlabel in LAT_ZONE_LABELS:
        sel = land & (in_band & (zone_of_row == zlabel))[:, None]
        area_w = np.where(sel, np.cos(np.deg2rad(lat))[:, None], 0.0)
        out[zlabel] = {}
        for i, key in enumerate(zip(gcms, runs)):
            ok = np.isfinite(w2[i]) & (area_w > 0)
            if not ok.any():
                continue
            mean_w2 = float((w2[i][ok] * area_w[ok]).sum() / area_w[ok].sum())
            out[zlabel][key] = 1.0 / max(mean_w2, eps)
    return out


def _group_counts_weighted(sub, x_int, inv_w2):
    """_group_counts pooling with weights 1/n_runs(GCM)/n_GCM x inv_w2[(GCM, run)]
    (realizations missing from inv_w2 excluded).
    Returns (arr_share, arr_count, mean_dur, weighted_total) or None."""
    reals = []
    for (gcm, run), rsub in sub.groupby(["GCM", "run"]):
        w = inv_w2.get((str(gcm), str(run)), 0.0)
        s = rsub.groupby("duration")["count"].sum()
        total_r = float(s.sum())
        if w > 0 and total_r > 0:
            reals.append((str(gcm), w, s, total_r))
    if not reals:
        return None
    runs_per_gcm = pd.Series([g for g, *_ in reals]).value_counts()
    n_gcm = runs_per_gcm.size

    arr_share = np.zeros(len(x_int))
    arr_count = np.zeros(len(x_int))
    wsum = mean_dur = weighted_total = 0.0
    for gcm, w_inv, s, total_r in reals:
        w = w_inv / runs_per_gcm[gcm] / n_gcm
        vals = np.array([s.get(d, 0.0) for d in x_int], dtype=float)
        arr_share += w * vals / total_r
        arr_count += w * vals
        mean_dur += w * float((s.index.to_numpy() * s.to_numpy()).sum() / total_r)
        weighted_total += w * total_r
        wsum += w
    return arr_share / wsum, arr_count / wsum, mean_dur / wsum, weighted_total / wsum


def _group_counts(counts_df, gwl, zone, x_int, inv_w2=None):
    """Pooled duration distribution of one (gwl, zone), each GCM weighted 1/n_GCM
    whatever its number of runs, or by inverse Wasserstein distance (inv_w2).
    Returns (arr_share, arr_count, mean_dur, weighted_total, raw_total) or
    None.

    arr_share averages each GCM's own normalized share (normalizing a pooled
    count would let GCMs with more events dominate). arr_count and mean_dur
    are per-GCM means; raw_total is the plain event count, used for the
    min_events check."""
    sub = counts_df[(counts_df["gwl"] == gwl) & (counts_df["zone"] == zone)]
    if sub.empty:
        return None
    raw_total = float(sub["count"].sum())

    if inv_w2 is not None:
        pooled = _group_counts_weighted(sub, x_int, inv_w2)
        return None if pooled is None else (*pooled, raw_total)

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


def _add_locator_map(fig, gs_column, zone_order):
    """Locator map of the latitude zones (shaded bands).
    Returns (ax_map, {zone_label: lat_mid})."""
    ax_map = fig.add_subplot(gs_column, projection=ccrs.PlateCarree())
    ax_map.set_extent([MAP_LON_WEST, MAP_LON_EAST, MAP_LAT_SOUTH - 2, MAP_LAT_NORTH + 2],
                       crs=ccrs.PlateCarree())
    ax_map.set_anchor("W")  # free width goes between map and panels
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

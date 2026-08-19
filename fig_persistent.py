# -*- coding: cp1252 -*-
"""
Long-lasting (persistent) compound wind-solar energy drought analysis.

Unlike fig1.py -- which flags a compound event on any single day where the
*daily* wcf and scf are simultaneously below their 10th-percentile reference
threshold -- this script targets multi-day, long-lasting droughts:

  1. wcf/scf are smoothed with a rolling weekly mean (`--roll_window`, default
     7 days) before thresholding, so a single anomalous day cannot flip the
     "low production" flag.
  2. A "low week" is a day whose rolling-mean wcf (resp. scf) falls below the
     `--threshold` quantile (default 0.01, i.e. the driest 1 % of the
     reference period, positive values only).
  3. The compound (wind AND solar) low-production mask is then built from the
     two gap-bridged serie

Two figures are produced, mirroring fig1.py's main outputs but for this
persistent-event definition:

  - a value-by-alpha map + regional time series of the change in persistent
    compound WSE drought severity (recent vs historical period), and
  - a reference-period multi-panel figure summarising persistent-drought
    frequency/duration/severity and mean rolling wcf/scf.
"""
import os
import config
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
os.environ["ESMFMKFILE"] = config.ESMFMKFILE_XENV

import argparse
import glob
import gc

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patheffects import withStroke
from matplotlib import cm
from string import ascii_lowercase

import cmocean as cmo

# =============================================================================
# Figure size constants (LaTeX-compatible)
# =============================================================================
FIG_WIDTH_IN = 5.15   # single column width : pt fontsizes match LaTeX

# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute persistent (long-lasting) compound WSE drought index and produce figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_path", default=None,
                         help="Pre-computed ds_final .nc (skips rebuilding).")
    parser.add_argument("--path_preprocessed", default=config.PATH_PREPROCESSED)
    parser.add_argument("--reanalysis", default=config.REANALYSIS)
    parser.add_argument("--threshold", type=float, default=0.01,
                         help="Quantile of the rolling-mean reference distribution defining a "
                              "'low week' (default: 0.01, i.e. below the driest 1%%).")
    parser.add_argument("--roll_window", type=int, default=7,
                         help="Rolling-mean window (days) applied to daily wcf/scf before "
                              "thresholding (default: 7, i.e. weekly).")
    parser.add_argument("--ref_start", default=config.SHEAR_REF_PERIOD[0])
    parser.add_argument("--ref_end", default=config.SHEAR_REF_PERIOD[1])
    parser.add_argument("--shapefile", default=config.SHAPEFILE_PATH)
    parser.add_argument("--output_dir", default="../final_figs")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--save_nc", action="store_true", default=False)
    return parser.parse_args()


# =============================================================================
# Gap-tolerant persistent-event detection
# =============================================================================



def compute_event_table(da, time_dim="time"):
    """
    Long-format table with one row per (event day, lat, lon) for every
    contiguous run of True in boolean/0-1 DataArray `da`. Each row carries the
    event's total length in the 'duration' column (via a groupby transform),
    so the same table can be used both to compute annual event statistics and
    to rebuild a day-level mask of "events lasting >= N days" without
    re-walking the runs.
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

    stacked = id_event.stack(z=("lat", "lon", time_dim)).dropna("z")
    df = pd.DataFrame({
        "event_id": stacked.values.astype(int),
        "lat":      stacked["lat"].values,
        "lon":      stacked["lon"].values,
        "time":     stacked[time_dim].values,
    })
    df["year"] = pd.DatetimeIndex(df["time"]).year
    df["year"] = df.groupby(["event_id", "lat", "lon"])["year"].transform("min")
    df["duration"] = df.groupby(["event_id", "lat", "lon"])["event_id"].transform("count")
    return df


def events_stats_from_table(df, template_da, time_dim="time"):
    """
    From the long event table (see compute_event_table) return:

      ds_dur, ds_freq : (year, lat, lon) Datasets with mean event 'duration'
                         and event 'frequency', reindexed onto template_da's
                         full lat/lon grid and full year range, 0-filled.
    """
    df_events = df.drop_duplicates(["event_id", "lat", "lon"])
    full_years = np.unique(pd.DatetimeIndex(template_da[time_dim].values).year)

    # groupby(...).to_xarray() only densifies over the *observed* year/lat/lon
    # values, so (year, lat, lon) combos with no event there still come out
    # NaN, not just the combos missing entirely from df_events. reindex's
    # fill_value only backfills newly-added labels, so it won't catch those --
    # fillna(0) after reindex is needed to turn "no event" into 0 everywhere.
    ds_dur = (df_events.groupby(["year", "lat", "lon"])[["duration"]]
              .mean().to_xarray())
    ds_freq = (df_events.groupby(["year", "lat", "lon"])[["duration"]]
               .count().rename(columns={"duration": "frequency"}).to_xarray())

    ds_dur = ds_dur.reindex(year=full_years, lat=template_da.lat, lon=template_da.lon,
                             fill_value=0).fillna(0)
    ds_freq = ds_freq.reindex(year=full_years, lat=template_da.lat, lon=template_da.lon,
                               fill_value=0).fillna(0)
    return ds_dur, ds_freq


def compute_severity_persistent(scf_roll, wcf_roll, scf_threshold, wcf_threshold):
    """
    Expected shortfall on persistent compound-drought days: mean positive
    rolling-mean deficit below threshold, aggregated yearly.
    """
    # roll first so the subtraction keeps (time, lat, lon) dim order --
    # `threshold - roll` would put the (lat, lon)-only threshold first and
    # leave the result transposed to (lat, lon, time).
    deficit_scf = -(scf_roll - scf_threshold)
    deficit_wcf = -(wcf_roll - wcf_threshold)
    compound_mask = (deficit_scf > 0) & (deficit_wcf > 0)
    daily_deficit = deficit_scf + deficit_wcf
    masked = xr.where(compound_mask, daily_deficit, np.nan)
    severity = masked.resample(time="YE").mean()
    return severity


# =============================================================================
# Dataset computation
# =============================================================================

def build_ds_final_persistent(
    path_preprocessed, reanalysis, threshold, ref_start, ref_end,
    roll_window=7,
):
    print(f"  Loading wcf/scf for reanalysis={reanalysis}")
    wcf_path = glob.glob(os.path.join(path_preprocessed, reanalysis, "wcf_day_*.nc"))[0]
    scf_path = glob.glob(os.path.join(path_preprocessed, reanalysis, "scf_day_*.nc"))[0]
    chunks = {"time": 1000, "lat": -1, "lon": -1}
    wcf = xr.open_dataset(wcf_path, chunks=chunks).sel(lat=slice(-58, 68))
    scf = xr.open_dataset(scf_path, chunks=chunks).sel(lat=slice(-58, 68))
    wcf = wcf.convert_calendar("standard")
    scf = scf.convert_calendar("standard")

    print(f"  {roll_window}-day rolling mean of wcf/scf")
    wcf_roll = wcf.wcf.rolling(time=roll_window, center=True, min_periods=roll_window).mean()
    scf_roll = scf.scf.rolling(time=roll_window, center=True, min_periods=roll_window).mean()

    print(f"  Computing low-week thresholds (quantile={threshold}, ref={ref_start}:{ref_end})")
    wcf_roll_ref = wcf_roll.sel(time=slice(ref_start, ref_end))
    scf_roll_ref = scf_roll.sel(time=slice(ref_start, ref_end))
    wcf_thr = wcf_roll_ref.where(wcf_roll_ref > 0).quantile(threshold, dim="time")
    scf_thr = scf_roll_ref.where(scf_roll_ref > 0).quantile(threshold, dim="time")

    print("  Flagging low-production weeks (rolling mean below threshold)")
    low_wind = (wcf_roll <= wcf_thr)
    low_solar = (scf_roll <= scf_thr)

    print("  Combining into compound (wind AND solar) low-production days")
    compound = (low_wind & low_solar).astype(int)

    print("  Building event table of compound low-production spells")
    df_events = compute_event_table(compound)
    ds_dur, ds_freq = events_stats_from_table(
        df_events, compound,
    )

    print("  Computing severity of persistent compound events")
    severity_ds = compute_severity_persistent(scf_roll, wcf_roll, scf_thr, wcf_thr)
    severity_ds["time"] = severity_ds.time.dt.year
    severity_ds = severity_ds.rename({"time": "year"}).fillna(0.0)

    print("  Building resource/land validity mask (reference-period non-NaN wcf & scf)")
    wcf_ref_mean = wcf.wcf.sel(time=slice(ref_start, ref_end)).mean("time")
    scf_ref_mean = scf.scf.sel(time=slice(ref_start, ref_end)).mean("time")
    resource_valid = wcf_ref_mean.notnull() & scf_ref_mean.notnull()
    wcf_roll_ref_mean = wcf_roll.sel(time=slice(ref_start, ref_end)).mean("time")
    scf_roll_ref_mean = scf_roll.sel(time=slice(ref_start, ref_end)).mean("time")

    ds_final = ds_dur.copy()
    ds_final["frequency"] = ds_freq.frequency
    ds_final["severity"] = severity_ds
    ds_final["resource_valid"] = resource_valid.astype("int8")
    ds_final["wcf_ref_mean"] = wcf_roll_ref_mean
    ds_final["scf_ref_mean"] = scf_roll_ref_mean

    del wcf, scf, wcf_roll, scf_roll, low_wind, low_solar, compound
    gc.collect()
    return ds_final.load()


# =============================================================================
# Helper functions
# =============================================================================

def rasterize_shapefile(shapefile, shape, transform):
    geometries = shapefile["geometry"]
    mask = geometry_mask(geometries=geometries, all_touched=True,
                          out_shape=shape, transform=transform, invert=True)
    return mask


def build_land_mask(ds_final, shapefile_path):
    """Land pixels with valid reference-period wcf & scf data (see resource_valid)."""
    shapefile = gpd.read_file(shapefile_path)
    da = ds_final["resource_valid"]
    transform = rasterio.transform.from_bounds(
        da.lon.min().item(), da.lat.min().item(),
        da.lon.max().item(), da.lat.max().item(),
        len(da.lon), len(da.lat),
    )
    land = rasterize_shapefile(shapefile, da.shape, transform)
    land = land[::-1, :]
    return land & da.values.astype(bool)


def stationary_bootstrap_ci_1d(y, years, n_boot=1000, block_size=5, ci=95):
    y = np.asarray(y, dtype=np.float64)
    if y.size < 2 or np.all(np.isnan(y)):
        return np.nan, np.nan, np.nan, np.nan
    n = y.size
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan
    p = 1.0 / float(block_size)
    slopes = np.empty(n_boot)
    intercepts = np.empty(n_boot)
    for b in range(n_boot):
        idx_parts, total = [], 0
        while total < n:
            L = np.random.geometric(p)
            s = np.random.randint(0, n)
            take = min(L, n - total)
            idx_parts.append((s + np.arange(take)) % n)
            total += take
        idx = np.concatenate(idx_parts)
        xb, yb = np.arange(n)[idx], y[idx]
        dx = xb - xb.mean()
        denom = np.dot(dx, dx)
        slope = np.nan if denom == 0 else np.dot(dx, yb - yb.mean()) / denom
        intercepts[b] = yb.mean() - slope * xb.mean() if np.isfinite(slope) else np.nan
        slopes[b] = slope
    years_arr = np.asarray(years, dtype=float)
    fitted = slopes[:, None] * years_arr[None, :] + intercepts[:, None]
    alpha = (100.0 - ci) / 2.0
    low = np.nanpercentile(fitted, alpha, axis=0)
    up = np.nanpercentile(fitted, 100.0 - alpha, axis=0)
    return float(np.nanmean(slopes)), float(np.nanmean(intercepts)), low, up


# =============================================================================
# Figure: Value-by-alpha map + regional time series (persistent events)
# =============================================================================

def plot_valuebyalpha_persistent(
    ds_final, mask, shapefile_path,
    map_title="Change (color) weighted by annual persistent-drought severity (opacity)",
    relchange_label="Relative change (2000-2019 vs 1980-1999) (%)",
    sev_label="Annual persistent severity (1980-1999 mean)",
    lat_min=-60, lat_max=72,
    period_hist=(1980, 1999), period_comp=(2000, 2019),
    regions=None, n_boot=2000, n_bins_change=5, n_bins_sev=5,
):
    # --- 1. Compound persistent-drought index ---
    da = (ds_final.frequency.where(mask == 1)
          * ds_final.severity.where(mask == 1)
          * ds_final.duration.where(mask == 1))
    da = da.sel(lat=slice(-60, 68))
    da = da.where(da.lat > lat_min, drop=True).where(da.lat < lat_max, drop=True)

    y0, y1 = period_hist
    y2, y3 = period_comp
    years_all = da.sel(year=slice(y0, y3)).year.values.astype(float)
    da_hist = da.sel(year=slice(y0, y1)).mean("year", skipna=True)
    da_comp = da.sel(year=slice(y2, y3)).mean("year", skipna=True)
    rel_change = 100.0 * (da_comp - da_hist) / da_hist
    rel_change = rel_change.where(np.isfinite(rel_change))
    sev = da.sel(year=slice(y0, y1)).mean("year", skipna=True)
    dChange = rel_change

    # --- 2. Discrete colour bins ---
    change_edges = [-100, -25, -10, 10, 25, 100]
    change_bin = np.digitize(dChange.values, change_edges[1:-1])
    base_cmap = cm.get_cmap("coolwarm")
    color_levels = base_cmap(np.linspace(0, 1, n_bins_change))

    # --- 3. Discrete alpha bins ---
    max_sev = float(np.nanmax(sev.values)) if np.isfinite(sev.values).any() else 1.0
    max_sev = max_sev if max_sev > 0 else 1.0
    sev_edges = np.linspace(0, max_sev, n_bins_sev + 1) ** 2 / max_sev
    sev_bin = np.digitize(sev.values, sev_edges[1:-1])
    alpha_min, alpha_max = 0.4, 1.0
    alpha_levels = np.linspace(alpha_min, alpha_max, n_bins_sev)

    # --- 4. RGBA assembly ---
    nlat, nlon = dChange.shape
    valid_mask = np.isfinite(dChange.values) & np.isfinite(sev.values)
    rgba_map = np.zeros((nlat, nlon, 4), dtype=float)
    cb = np.clip(change_bin, 0, n_bins_change - 1)
    sb = np.clip(sev_bin, 0, n_bins_sev - 1)
    rgba_map[valid_mask, :3] = color_levels[cb[valid_mask], :3]
    rgba_map[valid_mask, 3] = alpha_levels[sb[valid_mask]]

    # --- 5. Region definitions ---
    if regions is None:
        regions = [
            {"name": "Western U.S.",    "lat": [35, 50],   "lon": [-125, -105]},
            {"name": "Northern Amazon", "lat": [-10, 10],  "lon": [-70,  -50]},
            {"name": "Egypt",           "lat": [15, 30],   "lon": [25,    40]},
            {"name": "Kenya",           "lat": [-5,   5],  "lon": [33,    42]},
            {"name": "India",           "lat": [10,  30],  "lon": [70,    90]},
        ]
    panellabels = [chr(98 + i) for i in range(len(regions))]

    # --- 6. Figure layout ---
    fig_width_in = FIG_WIDTH_IN
    fig_height_in = fig_width_in * (12 / 20)
    ncols_total = max(1, len(regions))
    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=300)
    gs = GridSpec(2, ncols_total, height_ratios=[2.9, 1],
                  hspace=0.30, wspace=0.45, figure=fig)
    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())

    # --- 7. Draw map ---
    shp = gpd.read_file(shapefile_path)
    ax_map.imshow(
        rgba_map,
        extent=[sev.lon.min().item(), sev.lon.max().item(),
                sev.lat.min().item(), sev.lat.max().item()],
        origin="lower", transform=ccrs.PlateCarree(),
        interpolation="nearest", rasterized=True,
    )

    da_mask = ds_final.frequency.isel(year=0)
    t_mask = rasterio.transform.from_bounds(
        da_mask.lon.min().item(), da_mask.lat.min().item(),
        da_mask.lon.max().item(), da_mask.lat.max().item(),
        len(da_mask.lon), len(da_mask.lat),
    )
    land_mask = rasterize_shapefile(shp, da_mask.shape, t_mask)
    land_mask = land_mask[::-1, :]
    ocean_mask = land_mask & (mask == 0)
    ax_map.contourf(
        da_mask.lon, da_mask.lat, ocean_mask.astype(float),
        levels=[0.5, 1], colors=["gray"],
        transform=ccrs.PlateCarree(), zorder=5,
    )
    shp.boundary.plot(ax=ax_map, color="black", linewidth=0.15,
                       transform=ccrs.PlateCarree(), zorder=10)
    ax_map.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.15)
    ax_map.annotate(
        "$\\mathbf{a}$",
        xy=(0.02, 1.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=7,
        path_effects=[withStroke(linewidth=1.5, foreground="white")],
    )
    ax_map.set_title(map_title, fontsize=7, pad=6)

    # --- 8. Legend block ---
    legend_rgba = np.zeros((n_bins_change, n_bins_sev, 4))
    for ic in range(n_bins_change):
        legend_rgba[ic, :, :3] = color_levels[ic, :3]
        legend_rgba[ic, :, 3] = alpha_levels

    legend_ax = fig.add_axes([0.2, 0.45, 0.16, 0.16])
    legend_ax.imshow(legend_rgba, origin="lower", aspect="equal")
    legend_ax.set_xticks([0, n_bins_sev // 2, n_bins_sev - 1])
    legend_ax.set_xticklabels(["low", "mid", "high"], fontsize=5, ha="center")
    ytick_pos = [0.5, 1.5, 2.5, 3.5]
    ytick_labs = ["-25%", "-10%", "10%", "25%"]
    legend_ax.set_yticks(ytick_pos)
    legend_ax.set_yticklabels(ytick_labs, fontsize=5, va="center")
    legend_ax.set_xlabel(sev_label, fontsize=5, labelpad=4)
    legend_ax.set_ylabel(relchange_label, fontsize=5, labelpad=4)
    legend_ax.tick_params(axis="both", which="both", length=0)

    # --- 9. Regional time series + map boxes ---
    region_panels = []
    for ridx, reg in enumerate(regions):
        lat_lo, lat_hi = reg["lat"]
        lon_lo, lon_hi = reg["lon"]
        lat0 = lat_lo if da.lat[0] < da.lat[-1] else lat_hi
        lat1 = lat_hi if da.lat[0] < da.lat[-1] else lat_lo
        lon0 = lon_lo if da.lon[0] < da.lon[-1] else lon_hi
        lon1 = lon_hi if da.lon[0] < da.lon[-1] else lon_lo

        da_reg = da.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1),
                         year=slice(y0, y3))
        ts = da_reg.mean(("lat", "lon"), skipna=True).values
        mean_slope, mean_intercept, low_vals, up_vals = stationary_bootstrap_ci_1d(
            ts, np.arange(y3 - y0 + 1), n_boot=n_boot, block_size=3, ci=95,
        )

        ax_map.plot(
            [lon_lo, lon_hi, lon_hi, lon_lo, lon_lo],
            [lat_lo, lat_lo, lat_hi, lat_hi, lat_lo],
            color="black", linewidth=1,
            transform=ccrs.PlateCarree(),
            path_effects=[withStroke(linewidth=2.5, foreground="white")],
        )
        ax_map.annotate(
            panellabels[ridx],
            xy=(lon_lo + 0.5, lat_hi - 0.5),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax_map),
            fontsize=6, fontweight="bold",
            path_effects=[withStroke(linewidth=2, foreground="white")],
            zorder=20,
        )
        region_panels.append(dict(
            label=panellabels[ridx], name=reg["name"],
            years=years_all, ts=ts,
            fit_line=mean_intercept + mean_slope * np.arange(y3 - y0 + 1),
            low_line=low_vals, up_line=up_vals,
        ))

    # --- 10. Bottom row time series ---
    y_min = min(np.nanmin(r["ts"]) for r in region_panels) * 0.95
    y_max = max(np.nanmax(r["ts"]) for r in region_panels) * 1.05

    for ridx, rinfo in enumerate(region_panels):
        ax_ts = fig.add_subplot(gs[1, ridx])
        ax_ts.scatter(rinfo["years"], rinfo["ts"],
                      marker="x", s=6, linewidths=0.6, label="annual mean")
        ax_ts.plot(rinfo["years"], rinfo["fit_line"],
                   linestyle="-", color="red", linewidth=0.5)
        ax_ts.fill_between(rinfo["years"], rinfo["low_line"], rinfo["up_line"],
                           color="red", alpha=0.3, label="95% CI", linewidth=0.1)
        ax_ts.grid(True, linestyle="--", alpha=0.4)
        ax_ts.set_ylim(y_min, y_max)
        for spine in ax_ts.spines.values():
            spine.set_linewidth(0.4)
        ax_ts.tick_params(axis="both", labelsize=5)
        if ridx == 0:
            ax_ts.set_ylabel("Annual persistent severity", fontsize=5)
        ax_ts.set_xlabel("Year", fontsize=5)
        ax_ts.annotate(
            f"$\\mathbf{{{rinfo['label']}}}$",
            xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=6,
        )

    ax_map.spines["geo"].set_visible(False)
    ax_map.set_extent([-180, 180, -58, 68], crs=ccrs.PlateCarree())
    plt.tight_layout()
    return fig


# =============================================================================
# Figure: Reference-period persistent-drought summary
# =============================================================================

def plot_reference_persistent_drought(
    ds_final, mask, shapefile_path,
    ref_start_year, ref_end_year,
    roll_window=7,
):
    """
    6-panel reference figure: frequency / duration / intensity / annual
    severity of compound (wind AND solar rolling-mean below threshold)
    WSE drought events during the reference period, plus the mean
    reference-period rolling wind and solar capacity factor.
    """
    shapefile = gpd.read_file(shapefile_path)

    freq_mean = ds_final.frequency.where(mask == 1).sel(
        year=slice(ref_start_year, ref_end_year)).mean("year")
    dur_mean = ds_final.duration.where(mask == 1).sel(
        year=slice(ref_start_year, ref_end_year)).mean("year")
    int_mean = ds_final.severity.where(mask == 1).sel(
        year=slice(ref_start_year, ref_end_year)).mean("year")
    ann_sev_mean = (ds_final.frequency * ds_final.severity * ds_final.duration).where(
        mask == 1).sel(year=slice(ref_start_year, ref_end_year)).mean("year")
    wcf_mean = ds_final["wcf_ref_mean"].where(mask == 1)
    scf_mean = ds_final["scf_ref_mean"].where(mask == 1)

    da_comp = ds_final.frequency.isel(year=0)
    t_comp = rasterio.transform.from_bounds(
        da_comp.lon.min().item(), da_comp.lat.min().item(),
        da_comp.lon.max().item(), da_comp.lat.max().item(),
        len(da_comp.lon), len(da_comp.lat),
    )
    land_mask_comp = rasterize_shapefile(shapefile, da_comp.shape, t_comp)[::-1, :]
    ocean_mask_comp = land_mask_comp & (mask == 0)

    datasets = [freq_mean, dur_mean, int_mean, ann_sev_mean, wcf_mean, scf_mean]
    title_list = [
        "Frequency", "Duration", "Intensity", "Annual persistent\nWSE drought severity",
        "Wind capacity factor\n(7-day rolling mean)", "Solar capacity factor\n(7-day rolling mean)",
    ]
    legend_list = [
        "Events/yr", "Days/event",
        "Intensity/day of event", "Annual persistent severity",
        "Wind CF", "Solar CF",
    ]
    cmap_list = [
        cmo.cm.solar.reversed(), cmo.cm.matter, cmo.cm.dense,
        cmo.cm.balance, cmo.cm.speed, cmo.cm.thermal,
    ]
    vmin_list = [0, 0, 0, 0, 0, 0]
    vmax_list = [3, roll_window + 8, 0.05, 0.3, 0.5, 0.5]
    panellabels = list(ascii_lowercase[:6])

    fig_width_in = FIG_WIDTH_IN
    fig_height_in = fig_width_in * 0.55
    fig, axes = plt.subplots(2, 3, figsize=(fig_width_in, fig_height_in), dpi=300,
                              subplot_kw={"projection": ccrs.Robinson()})
    axes_flat = axes.flatten()

    for idx, ax in enumerate(axes_flat):
        ds = datasets[idx]
        if hasattr(ds, "load"):
            ds = ds.load()
        ax.set_global()
        ax.coastlines(resolution="50m", linewidth=0.15, color="black")
        ax.contourf(
            da_comp.lon, da_comp.lat,
            ocean_mask_comp.astype(float),
            levels=[0.5, 1], colors=["gray"],
            transform=ccrs.PlateCarree(), zorder=5,
        )
        ds.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap=cmap_list[idx], vmin=vmin_list[idx], vmax=vmax_list[idx],
            add_colorbar=True, add_labels=False,
            cbar_kwargs={"orientation": "horizontal", "shrink": 0.7,
                         "pad": 0.05, "label": legend_list[idx]},
            rasterized=True, linewidth=0,
        )
        shapefile.boundary.plot(ax=ax, color="black", linewidth=0.1,
                                transform=ccrs.PlateCarree())
        ax.annotate(
            f"$\\mathbf{{{panellabels[idx]}}}$",
            xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=5,
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )
        cbar_ax = fig.axes[-1]
        cbar_ax.set_xlabel(legend_list[idx], fontsize=5)
        cbar_ax.tick_params(labelsize=5)
        ax.set_extent([-180, 180, -60, 68], crs=ccrs.PlateCarree())
        ax.spines["geo"].set_visible(False)

    fig.suptitle(
        f"Persistent compound WSE drought, reference period {ref_start_year}-{ref_end_year}\n"
        f"(low week <= {ds_final.attrs.get('low_week_quantile', '?')} quantile of "
        f"{roll_window}-day rolling mean; wind AND solar simultaneously below threshold)",
        fontsize=6,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


# =============================================================================
# Global mean change + bootstrap CI
# =============================================================================

def compute_global_change_stats(ds_final, mask, period_hist=(1980, 1999),
                                 period_comp=(2000, 2019), n_bootstrap=1000, block_size=10):
    ds = ds_final.where(mask == 1)
    ds["annual_severity"] = ds["frequency"] * ds["severity"] * ds["duration"]
    early = ds["annual_severity"].sel(year=slice(*period_hist)).mean(dim="year")
    late = ds["annual_severity"].sel(year=slice(*period_comp)).mean(dim="year")
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    global_early = early.weighted(weights).mean(dim=["lat", "lon"]).values
    global_late = late.weighted(weights).mean(dim=["lat", "lon"]).values
    global_rel_change = (global_late - global_early) / global_early * 100

    data = (late - early).values
    lats = early.lat.values
    lons = early.lon.values
    weight_array = weights.values
    lat_blocks = np.arange(lats.min(), lats.max(), block_size)
    lon_blocks = np.arange(lons.min(), lons.max(), block_size)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        s_lat = np.random.choice(lat_blocks, size=len(lat_blocks), replace=True)
        s_lon = np.random.choice(lon_blocks, size=len(lon_blocks), replace=True)
        sample, sw = [], []
        for lb in s_lat:
            for lo in s_lon:
                li_mask = (lats >= lb) & (lats < lb + block_size)
                lo_mask = (lons >= lo) & (lons < lo + block_size)
                if np.any(li_mask) and np.any(lo_mask):
                    for li in np.where(li_mask)[0]:
                        for loi in np.where(lo_mask)[0]:
                            if not np.isnan(data[li, loi]):
                                sample.append(data[li, loi])
                                sw.append(weight_array[li])
        if sample:
            sample, sw = np.array(sample), np.array(sw)
            bootstrap_means.append(np.sum(sample * sw) / np.sum(sw))

    ci_lower_rel = np.percentile(bootstrap_means, 2.5) / global_early * 100
    ci_upper_rel = np.percentile(bootstrap_means, 97.5) / global_early * 100
    return global_rel_change, ci_lower_rel, ci_upper_rel


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    thr_str = str(args.threshold).replace(".", "")

    if args.data_path is not None:
        print(f"Loading pre-computed dataset from {args.data_path}")
        ds_final = xr.open_dataset(args.data_path)
    else:
        print("No --data_path provided -- computing ds_final on-the-fly")
        ds_final = build_ds_final_persistent(
            path_preprocessed=args.path_preprocessed,
            reanalysis=args.reanalysis,
            threshold=args.threshold,
            ref_start=args.ref_start,
            ref_end=args.ref_end,
            roll_window=args.roll_window,
        )
        ds_final.attrs.update(dict(
            roll_window_days=args.roll_window,
            low_week_quantile=args.threshold,
            reference_period=f"{args.ref_start}:{args.ref_end}",
        ))
        if args.save_nc:
            out_nc = os.path.join(
                args.path_preprocessed, "agg_datasets",
                f"ds_final_persistent_{thr_str}_roll{args.roll_window}"
                f"_{args.reanalysis}.nc",
            )
            os.makedirs(os.path.dirname(out_nc), exist_ok=True)
            print(f"  Saving dataset to {out_nc}")
            ds_final.to_netcdf(out_nc)
            print(f"  Saved: {out_nc}")

    print("Building land/resource mask")
    mask = build_land_mask(ds_final, args.shapefile)

    print("Plotting persistent-drought value-by-alpha figure")
    fig_vba = plot_valuebyalpha_persistent(
        ds_final=ds_final, mask=mask, shapefile_path=args.shapefile,
        map_title=" ",
        relchange_label="Relative change in annual\npersistent WSE drought\nseverity (%)",
        sev_label="Historical average annual\npersistent WSE drought severity",
        lat_min=-60, lat_max=75,
        period_hist=(1980, 1999), period_comp=(2000, 2019),
        n_boot=args.n_boot,
    )
    out_vba = os.path.join(
        args.output_dir, "main",
        f"fig_persistent_valuebyalpha_{thr_str}_roll{args.roll_window}.png",
    )
    os.makedirs(os.path.dirname(out_vba), exist_ok=True)
    fig_vba.savefig(out_vba, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_vba)
    print(f"Saved {out_vba}")

    print("Plotting reference persistent-drought figure")
    ref_start_year = pd.Timestamp(args.ref_start).year
    ref_end_year = pd.Timestamp(args.ref_end).year
    fig_ref = plot_reference_persistent_drought(
        ds_final=ds_final, mask=mask, shapefile_path=args.shapefile,
        ref_start_year=ref_start_year, ref_end_year=ref_end_year,
        roll_window=args.roll_window,
    )
    out_ref = os.path.join(
        args.output_dir, "supp",
        f"suppfig_persistent_reference_{thr_str}_roll{args.roll_window}.png",
    )
    os.makedirs(os.path.dirname(out_ref), exist_ok=True)
    fig_ref.savefig(out_ref, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_ref)
    print(f"Saved {out_ref}")

    print("Computing global change statistics")
    global_rel_change, ci_lower_rel, ci_upper_rel = compute_global_change_stats(
        ds_final, mask, period_hist=(1980, 1999), period_comp=(2000, 2019), n_bootstrap=1000)
    print("\nResults:")
    print(f"  Global change: {global_rel_change:.2f}%")
    print(f"  95% CI: [{ci_lower_rel:.2f}%, {ci_upper_rel:.2f}%]")
    print(f"  CI width: {ci_upper_rel - ci_lower_rel:.2f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()

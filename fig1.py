# -*- coding: cp1252 -*-
import os
import config
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
os.environ['ESMFMKFILE'] = config.ESMFMKFILE_XENV

import argparse
import gc

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from scipy import stats

# Zarr/NetCDF-agnostic file lookup + opener, shared with calculate_cf.py
# (prefers a .zarr store when present, falls back to .nc).
from io_utils import match_files, open_dataset_any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from string import ascii_lowercase
from itertools import groupby

import cmocean as cmo

# =============================================================================
# Figure size constants (LaTeX-compatible)
# =============================================================================
FIG_WIDTH_IN = 5.15   # single column width pt fontsizes match LaTeX

# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute compound energy drought index and produce figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--data_path_005", default=None)
    parser.add_argument("--data_path_02", default=None)
    parser.add_argument("--rl_path", default=None)
    parser.add_argument("--path_preprocessed", default=config.PATH_PREPROCESSED)
    parser.add_argument("--reanalysis", default=config.REANALYSIS)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--ref_start", default=config.SHEAR_REF_PERIOD[0])
    parser.add_argument("--ref_end",   default=config.SHEAR_REF_PERIOD[1])
    parser.add_argument("--shapefile", default=config.SHAPEFILE_PATH)
    parser.add_argument("--output_dir", default="../final_figs")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--save_nc", action="store_true", default=False)
    return parser.parse_args()


# =============================================================================
# Dataset computation
# =============================================================================

def compute_severity(comp_da, scf_ds, wcf_ds, scf_threshold, wcf_threshold):
    deficit_scf   = scf_threshold - scf_ds["scf"]
    deficit_wcf   = wcf_threshold - wcf_ds["wcf"]
    daily_deficit = deficit_scf + deficit_wcf
    masked        = xr.where(comp_da == 1, daily_deficit, np.nan)
    severity      = masked.resample(time="1Y").mean()
    return severity


def _pixel_duration_frequency(x, times_year, out_years):
    """
    Run-length-encode one pixel's 0/1 daily series into per-year event
    duration (mean length of events starting that year) and frequency
    (count of events starting that year). NaN where no event started in
    that year at that pixel, matching the semantics of the original
    event_id/groupby("year","lat","lon") approach (a pixel-year with zero
    events simply had no row in that groupby, i.e. NaN after to_xarray).
    Called once per pixel by duration_xr's apply_ufunc, so it only ever
    sees a single (time,) series -- no global stack/dropna over the whole
    cube.
    """
    n_years = out_years.shape[0]
    dur_sum = np.zeros(n_years)
    count   = np.zeros(n_years)
    x = np.asarray(x)
    if x.size and np.any(x > 0):
        xb     = (x > 0).astype(np.int8)
        padded = np.concatenate(([0], xb, [0]))
        d      = np.diff(padded)
        starts = np.flatnonzero(d == 1)
        stops  = np.flatnonzero(d == -1) - 1
        if starts.size:
            lengths     = (stops - starts + 1).astype(np.float64)
            start_years = times_year[starts]
            year_idx    = np.searchsorted(out_years, start_years)
            year_idx_c  = np.clip(year_idx, 0, n_years - 1)
            valid       = ((year_idx >= 0) & (year_idx < n_years)
                            & (out_years[year_idx_c] == start_years))
            np.add.at(dur_sum, year_idx_c[valid], lengths[valid])
            np.add.at(count,   year_idx_c[valid], 1)
    duration  = np.full(n_years, np.nan)
    frequency = np.full(n_years, np.nan)
    has_event = count > 0
    duration[has_event]  = dur_sum[has_event] / count[has_event]
    frequency[has_event] = count[has_event]
    return duration, frequency


def duration_xr(da, tile_lat=60, tile_lon=60):
    """
    Per-pixel event duration/frequency via a dask-chunked apply_ufunc:
    lat/lon are tiled (tile_lat x tile_lon x full time per task) so each
    dask task only ever holds one tile in memory and tiles run in
    parallel, instead of the previous approach which cumsum'd and
    stack()+dropna()'d the *entire* (time x lat x lon) global cube into
    one dense array/pandas dataframe -- that materialization (tens of GB
    for a multi-decade daily global grid) is what was causing the OOM.
    """
    da = da.chunk({"time": -1, "lat": tile_lat, "lon": tile_lon})
    years_arr = da["time"].dt.year.values
    out_years = np.unique(years_arr)

    duration, frequency = xr.apply_ufunc(
        _pixel_duration_frequency,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["year"], ["year"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64],
        dask_gufunc_kwargs={"output_sizes": {"year": out_years.size}},
        kwargs={"times_year": years_arr, "out_years": out_years},
    )
    duration  = duration.assign_coords(year=("year", out_years))
    frequency = frequency.assign_coords(year=("year", out_years))
    ds      = duration.rename("duration").to_dataset()
    ds_freq = frequency.rename("frequency").to_dataset()
    return ds, ds_freq


def build_ds_final(path_preprocessed, reanalysis, thr, ref_start, ref_end, shapefile_path):
    print(f"  Loading wcf/scf for reanalysis={reanalysis}  ")
    wcf_files, _ = match_files(os.path.join(path_preprocessed, reanalysis, "wcf_day_*"))
    scf_files, _ = match_files(os.path.join(path_preprocessed, reanalysis, "scf_day_*"))
    if not wcf_files:
        raise FileNotFoundError(
            f"No wcf_day_* file found under {os.path.join(path_preprocessed, reanalysis)}")
    if not scf_files:
        raise FileNotFoundError(
            f"No scf_day_* file found under {os.path.join(path_preprocessed, reanalysis)}")
    chunks   = {"time": 1000, "lat": -1, "lon": -1}
    wcf = open_dataset_any(wcf_files[0], chunks=chunks).sel(lat=slice(-58, 68))
    scf = open_dataset_any(scf_files[0], chunks=chunks).sel(lat=slice(-58, 68))
    wcf = wcf.convert_calendar("standard")
    scf = scf.convert_calendar("standard")

    print(f"  Computing thresholds (quantile={thr}, ref={ref_start}-{ref_end})  ")
    wcf_ref = wcf.sel(time=slice(ref_start, ref_end))
    scf_ref = scf.sel(time=slice(ref_start, ref_end))
    wcf_thr = wcf_ref.wcf.where(wcf_ref.wcf > 0).quantile(thr, dim="time")
    scf_thr = scf_ref.scf.where(scf_ref.scf > 0).quantile(thr, dim="time")

    print("  Detecting compound events  ")
    wcf["low_wind"]  = xr.where(wcf.wcf >= wcf_thr, 1, 0)
    scf["low_solar"] = xr.where(scf.scf >= scf_thr, 1, 0)
    compound = (wcf.low_wind * scf.low_solar).to_dataset(name="start_cooc")

    land_mask = build_land_mask_from_grid(compound.lat.values, compound.lon.values,
                                          shapefile_path)
    land_mask_da = xr.DataArray(land_mask, dims=("lat", "lon"),
                                coords={"lat": compound.lat, "lon": compound.lon})
    compound["start_cooc"] = compound["start_cooc"].where(land_mask_da)

    print("  Computing severity  ")
    severity_ds = compute_severity(compound.start_cooc, scf, wcf, scf_thr, wcf_thr)
    severity_ds["time"] = severity_ds.time.dt.year
    severity_ds = severity_ds.rename({"time": "year"})

    print("  Computing duration and frequency  ")
    ds_dur, ds_freq = duration_xr(compound.start_cooc)
    ds_final = ds_dur.copy()
    ds_final["frequency"] = ds_freq.frequency
    ds_final["severity"]  = severity_ds
    ds_final["nb_days"]   = (
        compound.start_cooc.resample(time="1Y").sum("time").rename({"time": "year"})
    )
    gc.collect()
    return ds_final


# =============================================================================
# Helper functions
# =============================================================================

def rasterize_shapefile(shapefile, shape, transform):
    geometries = shapefile["geometry"]
    mask = geometry_mask(geometries=geometries, all_touched=True,
                          out_shape=shape, transform=transform, invert=True)
    return mask


def build_land_mask_from_grid(lat, lon, shapefile_path):
    """
    Land/ocean mask straight from a lat/lon grid, without depending on
    ds_final (which doesn't exist yet inside build_ds_final). Used to drop
    ocean cells before the expensive duration/severity step instead of
    only masking afterwards for plotting -- a coarse superset of the
    final build_land_mask() (that one additionally excludes cells where
    duration ends up all-null, which can't be known before it's computed).
    """
    shapefile = gpd.read_file(shapefile_path)
    shape = (len(lat), len(lon))
    transform = rasterio.transform.from_bounds(
        float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat)),
        len(lon), len(lat),
    )
    mask = rasterize_shapefile(shapefile, shape, transform)
    return mask[::-1, :]


def build_land_mask(ds_final, shapefile_path):
    shapefile = gpd.read_file(shapefile_path)
    da = ds_final.frequency.isel(year=0) if "year" in ds_final.dims else ds_final.frequency
    transform = rasterio.transform.from_bounds(
        da.lon.min().item(), da.lat.min().item(),
        da.lon.max().item(), da.lat.max().item(),
        len(da.lon), len(da.lat),
    )
    mask = rasterize_shapefile(shapefile, da.shape, transform)
    mask = mask[::-1, :]
    mask_update = ds_final.duration.isnull()
    mask = mask & (~mask_update)
    return mask


def stationary_bootstrap_ci_1d(y, years, n_boot=1000, block_size=5, ci=95):
    y = np.asarray(y, dtype=np.float64)
    if y.size < 2 or np.all(np.isnan(y)):
        return np.nan, np.nan, np.nan, np.nan
    n = y.size
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan
    p          = 1.0 / float(block_size)
    slopes     = np.empty(n_boot)
    intercepts = np.empty(n_boot)
    for b in range(n_boot):
        idx_parts, total = [], 0
        while total < n:
            L    = np.random.geometric(p)
            s    = np.random.randint(0, n)
            take = min(L, n - total)
            idx_parts.append((s + np.arange(take)) % n)
            total += take
        idx  = np.concatenate(idx_parts)
        xb, yb = np.arange(n)[idx], y[idx]
        dx     = xb - xb.mean()
        denom  = np.dot(dx, dx)
        slope  = np.nan if denom == 0 else np.dot(dx, yb - yb.mean()) / denom
        intercepts[b] = yb.mean() - slope * xb.mean() if np.isfinite(slope) else np.nan
        slopes[b]     = slope
    years_arr = np.asarray(years, dtype=float)
    fitted    = slopes[:, None] * years_arr[None, :] + intercepts[:, None]
    alpha     = (100.0 - ci) / 2.0
    low = np.nanpercentile(fitted, alpha,         axis=0)
    up  = np.nanpercentile(fitted, 100.0 - alpha, axis=0)
    return float(np.nanmean(slopes)), float(np.nanmean(intercepts)), low, up


# =============================================================================
# Figure 1 Value-by-alpha map + regional time series
# =============================================================================

def plot_reanalysis_disagg_timeseries_valuebyalpha_discrete(
    ds_final, mask, shapefile_path,
    map_title="Projected change (color) weighted by annual severity (opacity)",
    relchange_label="Relative change (2000-2019 vs 1980-1999) (%)",
    sev_label="Annual severity (1980-1999 mean)",
    lat_min=-60, lat_max=72,
    period_hist=(1980, 1999), period_comp=(2000, 2019),
    regions=None, n_boot=2000, n_bins_change=5, n_bins_sev=5,
):
    # --- 1. Compound index ---
    da = (ds_final.frequency.where(mask == 1)
          * ds_final.severity.where(mask == 1)
          * ds_final.duration.where(mask == 1))
    da = da.sel(lat=slice(-60, 68))
    da = da.where(da.lat > lat_min, drop=True).where(da.lat < lat_max, drop=True)

    y0, y1 = period_hist
    y2, y3 = period_comp
    years_all  = da.sel(year=slice(y0, y3)).year.values.astype(float)
    da_hist    = da.sel(year=slice(y0, y1)).mean("year", skipna=True)
    da_comp    = da.sel(year=slice(y2, y3)).mean("year", skipna=True)
    rel_change = 100.0 * (da_comp - da_hist) / da_hist
    rel_change = rel_change.where(np.isfinite(rel_change))
    sev        = da.sel(year=slice(1982, 2001)).mean("year", skipna=True)
    dChange    = rel_change

    # --- 2. Discrete colour bins ---
    change_edges = [-100, -25, -10, 10, 25, 100]
    change_bin   = np.digitize(dChange.values, change_edges[1:-1])
    base_cmap    = cm.get_cmap("coolwarm")
    color_levels = base_cmap(np.linspace(0, 1, n_bins_change))

    # --- 3. Discrete alpha bins ---
    max_sev      = 1.0
    sev_edges    = np.linspace(0, max_sev, n_bins_sev + 1) ** 2 / max_sev
    sev_bin      = np.digitize(sev.values, sev_edges[1:-1])
    alpha_min, alpha_max = 0.4, 1.0
    alpha_levels = np.linspace(alpha_min, alpha_max, n_bins_sev)

    # --- 4. RGBA assembly ---
    nlat, nlon = dChange.shape
    valid_mask = np.isfinite(dChange.values) & np.isfinite(sev.values)
    rgba_map   = np.zeros((nlat, nlon, 4), dtype=float)
    cb = np.clip(change_bin, 0, n_bins_change - 1)
    sb = np.clip(sev_bin,    0, n_bins_sev    - 1)
    rgba_map[valid_mask, :3] = color_levels[cb[valid_mask], :3]
    rgba_map[valid_mask,  3] = alpha_levels[sb[valid_mask]]

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

    # --- 6. Figure layout (LaTeX-compatible width) ---
    fig_width_in  = FIG_WIDTH_IN
    fig_height_in = fig_width_in * (12 / 20)   # keep original aspect ratio
    ncols_total   = max(1, len(regions))
    fig  = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=300)
    gs   = GridSpec(2, ncols_total, height_ratios=[2.9, 1],
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
    t_mask  = rasterio.transform.from_bounds(
        da_mask.lon.min().item(), da_mask.lat.min().item(),
        da_mask.lon.max().item(), da_mask.lat.max().item(),
        len(da_mask.lon), len(da_mask.lat),
    )
    land_mask  = rasterize_shapefile(shp, da_mask.shape, t_mask)
    land_mask  = land_mask[::-1, :]
    ocean_mask = land_mask & (da_mask.isnull())
    ax_map.contourf(
        ocean_mask.lon, ocean_mask.lat, ocean_mask.values.astype(float),
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
        legend_rgba[ic, :,  3] = alpha_levels

    legend_ax = fig.add_axes([0.2, 0.45, 0.16, 0.16])
    legend_ax.imshow(legend_rgba, origin="lower", aspect="equal")
    legend_ax.set_xticks([0, n_bins_sev // 2, n_bins_sev - 1])
    legend_ax.set_xticklabels(["low", "mid", "high"], fontsize=5, ha="center")
    ytick_pos  = [0.5, 1.5, 2.5, 3.5]
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
        ts     = da_reg.mean(("lat", "lon"), skipna=True).values
        mean_slope, mean_intercept, low_vals, up_vals = stationary_bootstrap_ci_1d(
            ts, np.arange(y3 - y0 + 1), n_boot=n_boot, block_size=3, ci=95,
        )

        # Map box (unchanged)
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
            ax_ts.set_ylabel("Annual severity", fontsize=5)
        ax_ts.set_xlabel("Year", fontsize=5)
        ax_ts.annotate(
            f"$\\mathbf{{{rinfo['label']}}}$",
            xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=6,
        )
        #ax_ts.set_title(rinfo['name'], fontsize=6)

    ax_map.spines["geo"].set_visible(False)
    ax_map.set_extent([-180, 180, -58, 68], crs=ccrs.PlateCarree())
    plt.tight_layout()
    return fig


# =============================================================================
# Figure 2 Interannual variability map
# =============================================================================

def plot_variability_map(ds_final, mask, shapefile_path, dpi=300):
    shapefile = gpd.read_file(shapefile_path)
    pdd     = (ds_final.frequency * ds_final.severity * ds_final.duration).where(mask)
    std_pdd = pdd.std(dim="year")

    fig_width_in  = FIG_WIDTH_IN
    fig_height_in = fig_width_in * (6 / 12)
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in),
                           subplot_kw={"projection": ccrs.Robinson()})
    im = ax.pcolormesh(
        std_pdd.lon, std_pdd.lat, std_pdd.values,
        transform=ccrs.PlateCarree(),
        cmap=cmo.cm.amp, vmin=0, vmax=0.25, rasterized=True,
    )
    da_mask = ds_final.frequency.isel(year=0)
    t_mask  = rasterio.transform.from_bounds(
        da_mask.lon.min().item(), da_mask.lat.min().item(),
        da_mask.lon.max().item(), da_mask.lat.max().item(),
        len(da_mask.lon), len(da_mask.lat),
    )
    mask_plot = rasterize_shapefile(shapefile, da_mask.shape, t_mask)
    mask_plot = mask_plot[::-1, :] & (da_mask.isnull())
    ax.contourf(mask_plot.lon, mask_plot.lat, mask_plot.values.astype(float),
                levels=[0.5, 1], colors=["gray"],
                transform=ccrs.PlateCarree(), zorder=5)
    shapefile.boundary.plot(ax=ax, color="black", linewidth=0.15,
                            transform=ccrs.PlateCarree(), zorder=10)
    ax.set_global()
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6, aspect=40)
    cbar.set_label("Interannual variability", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    ax.set_title("Interannual variability of annual severity (1982-2021)",
                 fontsize=8, fontweight="bold")
    ax.set_extent([-180, 180, -58, 68], crs=ccrs.PlateCarree())
    plt.tight_layout()
    return fig


# =============================================================================
# Figure 3 Mean solar & wind capacity factor maps (reference period)
# =============================================================================

def plot_mean_variables_6panel(
    ds_final, mask, shapefile_path, path_preprocessed, reanalysis,
    ref_start="1982-01-01", ref_end="2001-12-31",
):
    shapefile      = gpd.read_file(shapefile_path)
    ref_label      = f"{pd.Timestamp(ref_start).year}-{pd.Timestamp(ref_end).year}"
    ref_start_year = pd.Timestamp(ref_start).year
    ref_end_year   = pd.Timestamp(ref_end).year

    freq_mean    = ds_final.frequency.where(mask == 1).sel(
        year=slice(ref_start_year, ref_end_year)).mean("year")
    dur_mean     = ds_final.duration.where(mask == 1).sel(
        year=slice(ref_start_year, ref_end_year)).mean("year")
    int_mean     = ds_final.severity.where(mask == 1).sel(
        year=slice(ref_start_year, ref_end_year)).mean("year")
    ann_sev_mean = (ds_final.frequency * ds_final.severity * ds_final.duration).where(
        mask == 1).sel(year=slice(ref_start_year, ref_end_year)).mean("year")
    std_pdd = (ds_final.frequency * ds_final.severity * ds_final.duration).where(
        mask == 1).std(dim="year")

    da_comp = ds_final.frequency.isel(year=0)
    t_comp  = rasterio.transform.from_bounds(
        da_comp.lon.min().item(), da_comp.lat.min().item(),
        da_comp.lon.max().item(), da_comp.lat.max().item(),
        len(da_comp.lon), len(da_comp.lat),
    )
    land_mask_comp  = rasterize_shapefile(shapefile, da_comp.shape, t_comp)[::-1, :]
    ocean_mask_comp = land_mask_comp & (da_comp.isnull())

    print(f"  Loading wcf/scf for 6-panel map (ref: {ref_start}-{ref_end})  ")
    wcf_files, _ = match_files(os.path.join(path_preprocessed, reanalysis, "wcf_day_*"))
    scf_files, _ = match_files(os.path.join(path_preprocessed, reanalysis, "scf_day_*"))
    chunks   = {"time": 1000, "lat": -1, "lon": -1}
    wcf_ref  = open_dataset_any(wcf_files[0], chunks=chunks).sel(
        lat=slice(-58, 68), time=slice(ref_start, ref_end))
    scf_ref  = open_dataset_any(scf_files[0], chunks=chunks).sel(
        lat=slice(-58, 68), time=slice(ref_start, ref_end))
    da_wcf   = wcf_ref.wcf.isel(time=0)
    t_wcf    = rasterio.transform.from_bounds(
        da_wcf.lon.min().item(), da_wcf.lat.min().item(),
        da_wcf.lon.max().item(), da_wcf.lat.max().item(),
        len(da_wcf.lon), len(da_wcf.lat),
    )
    land_mask_wcf = rasterize_shapefile(shapefile, da_wcf.shape, t_wcf)[::-1, :]
    wcf_mean = wcf_ref.wcf.mean(dim="time").where(land_mask_wcf)
    scf_mean = scf_ref.scf.mean(dim="time").where(land_mask_wcf)

    datasets   = [freq_mean, dur_mean, int_mean, ann_sev_mean, wcf_mean, scf_mean, std_pdd]
    title_list = [
        "Frequency", "Duration",
        "Intensity", "Annual WSED severity",
        "Wind capacity\nfactor",     "Solar capacity\nfactor",
        "Interannual variability of\nannual severity",
    ]
    legend_list = [
        "Events/yr", "Days/event", "Intensity/day of event",
        "Annual WSED severity", "Wind CF", "Solar CF", "Std of annual severity",
    ]
    cmap_list = [
        cmo.cm.solar.reversed(), cmo.cm.matter, cmo.cm.dense,
        cmo.cm.balance, cmo.cm.speed, cmo.cm.thermal, cmo.cm.amp,
    ]
    vmin_list = [0,  1,  0,    0, 0,   0,   0]
    vmax_list = [36, 3,  0.05, 1, 0.5, 0.5, 0.25]
    panellabels = list(ascii_lowercase[:7])

    # 3x3 layout, same width; height scales proportionally
    fig_width_in  = FIG_WIDTH_IN
    fig_height_in = fig_width_in * 0.6
    fig, axes = plt.subplots(3, 3, figsize=(fig_width_in, fig_height_in), dpi=300,
                             subplot_kw={"projection": ccrs.Robinson()})
    axes_flat = axes.flatten()


    for idx, ax in enumerate(axes_flat):
        if idx >= 7:
            ax.set_visible(False)
            continue
        ds = datasets[idx]
        if hasattr(ds, "load"):
            ds = ds.load()
        ax.set_global()
        ax.coastlines(resolution="50m", linewidth=0.15, color="black")
        if idx in (0, 1, 2, 3, 6):
            ax.contourf(
                ocean_mask_comp.lon, ocean_mask_comp.lat,
                ocean_mask_comp.values.astype(float),
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
        # ax.set_title(title_list[idx], fontsize=5)
        cbar_ax = fig.axes[-1]
        cbar_ax.set_xlabel(legend_list[idx], fontsize=5)
        cbar_ax.tick_params(labelsize=5)
        ax.set_extent([-180, 180, -60, 68], crs=ccrs.PlateCarree())
        ax.spines["geo"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# Supplementary Value-by-alpha sensitivity (thresholds 0.05 & 0.20)
# =============================================================================

def _compute_valuebyalpha_rgba(
    ds_final, mask,
    period_hist=(1980, 1999), period_comp=(2000, 2019),
    n_bins_change=5, n_bins_sev=5,
):
    da = (ds_final.frequency.where(mask == 1)
          * ds_final.severity.where(mask == 1)
          * ds_final.duration.where(mask == 1))
    da = da.sel(lat=slice(-60, 68))
    y0, y1 = period_hist
    y2, y3 = period_comp
    da_hist    = da.sel(year=slice(y0, y1)).mean("year", skipna=True)
    da_comp    = da.sel(year=slice(y2, y3)).mean("year", skipna=True)
    rel_change = 100.0 * (da_comp - da_hist) / da_hist
    rel_change = rel_change.where(np.isfinite(rel_change))
    sev        = da.sel(year=slice(1982, 2001)).mean("year", skipna=True)

    change_edges = [-100, -25, -10, 10, 25, 100]
    change_bin   = np.digitize(rel_change.values, change_edges[1:-1])
    base_cmap    = cm.get_cmap("coolwarm")
    color_levels = base_cmap(np.linspace(0, 1, n_bins_change))
    max_sev      = 1.0
    sev_edges    = np.linspace(0, max_sev, n_bins_sev + 1) ** 2 / max_sev
    sev_bin      = np.digitize(sev.values, sev_edges[1:-1])
    alpha_levels = np.linspace(0.4, 1.0, n_bins_sev)

    nlat, nlon = rel_change.shape
    valid      = np.isfinite(rel_change.values) & np.isfinite(sev.values)
    rgba_map   = np.zeros((nlat, nlon, 4), dtype=float)
    cb = np.clip(change_bin, 0, n_bins_change - 1)
    sb = np.clip(sev_bin,    0, n_bins_sev    - 1)
    rgba_map[valid, :3] = color_levels[cb[valid], :3]
    rgba_map[valid,  3] = alpha_levels[sb[valid]]
    return rgba_map, rel_change, sev, color_levels, alpha_levels, sev_edges, change_edges


def plot_valuebyalpha_sensitivity(
    ds_005, mask_005, ds_02, mask_02, shapefile_path,
    period_hist=(1980, 1999), period_comp=(2000, 2019),
    n_bins_change=5, n_bins_sev=5,
):
    shapefile = gpd.read_file(shapefile_path)
    rgba_005, _, sev_005, color_levels, alpha_levels, sev_edges, change_edges = \
        _compute_valuebyalpha_rgba(ds_005, mask_005, period_hist, period_comp,
                                   n_bins_change, n_bins_sev)
    rgba_02,  _, sev_02,  _, _, _, _ = \
        _compute_valuebyalpha_rgba(ds_02,  mask_02,  period_hist, period_comp,
                                   n_bins_change, n_bins_sev)

    fig_width_in  = FIG_WIDTH_IN  # 2 side-by-side panels double width
    fig_height_in = FIG_WIDTH_IN * 1.5
    fig, axes = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in), dpi=300,
                             subplot_kw={"projection": ccrs.Robinson()})

    panel_configs = [
        (axes[0], rgba_005, sev_005, ds_005, "a",
         "Historical annual WSED severity change\nThreshold = 0.05"),
        (axes[1], rgba_02,  sev_02,  ds_02,  "b",
         "Historical annual WSED severity change\nThreshold = 0.20"),
    ]
    for ax, rgba_map, sev_da, ds_src, letter, title in panel_configs:
        ax.imshow(
            rgba_map,
            extent=[sev_da.lon.min().item(), sev_da.lon.max().item(),
                    sev_da.lat.min().item(), sev_da.lat.max().item()],
            origin="lower", transform=ccrs.PlateCarree(),
            interpolation="nearest", rasterized=True,
        )
        da_m  = ds_src.frequency.isel(year=0)
        t_m   = rasterio.transform.from_bounds(
            da_m.lon.min().item(), da_m.lat.min().item(),
            da_m.lon.max().item(), da_m.lat.max().item(),
            len(da_m.lon), len(da_m.lat),
        )
        land_m  = rasterize_shapefile(shapefile, da_m.shape, t_m)[::-1, :]
        ocean_m = land_m & (da_m.isnull())
        ax.contourf(ocean_m.lon, ocean_m.lat, ocean_m.values.astype(float),
                    levels=[0.5, 1], colors=["gray"],
                    transform=ccrs.PlateCarree(), zorder=5)
        shapefile.boundary.plot(ax=ax, color="black", linewidth=0.15,
                                transform=ccrs.PlateCarree(), zorder=10)
        ax.annotate(
            f"$\\mathbf{{{letter}}}$",
            xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=8,
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )
        ax.set_title(title, fontsize=8)
        ax.set_extent([-180, 180, -58, 68], crs=ccrs.PlateCarree())
        ax.spines["geo"].set_visible(False)

    # Shared bivariate legend
    legend_rgba = np.zeros((n_bins_change, n_bins_sev, 4))
    for ic in range(n_bins_change):
        legend_rgba[ic, :, :3] = color_levels[ic, :3]
        legend_rgba[ic, :,  3] = alpha_levels
    legend_ax = fig.add_axes([0.07, 0.12, 0.08, 0.14])
    legend_ax.imshow(legend_rgba, origin="lower", aspect="equal")
    legend_ax.set_xticks([0, n_bins_sev // 2, n_bins_sev - 1])
    legend_ax.set_xticklabels(["low", "mid", "high"], fontsize=5, ha="center")
    legend_ax.set_yticks([0, n_bins_change // 2, n_bins_change - 1])
    legend_ax.set_yticklabels(
        [f"<{change_edges[1]:.0f}%",
         f"{change_edges[n_bins_change // 2]:.0f}%;{change_edges[n_bins_change // 2 + 1]:.0f}%",
         f">+{change_edges[-2]:.0f}%"],
        fontsize=5, va="center",
    )
    legend_ax.set_xlabel("Reference\nseverity", fontsize=6, labelpad=4)
    legend_ax.set_ylabel("Rel. change (%)",     fontsize=6, labelpad=4)
    legend_ax.tick_params(axis="both", which="both", length=0)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# =============================================================================
# Supplementary combined threshold sensitivity
# =============================================================================

def _load_wsbd_gwl2(csv_path, share_re="current", vmax=800):
    df = pd.read_csv(csv_path, index_col=0)
    keys      = ["poly_idx", "GCM", "run", "share_re"]
    gwl_label = "GWL2"
    base      = (df["gwl_tas"] == "GWL0-61") & (df["gwl_ds_cf"] == "GWL0-61")
    df_ref    = df[base][keys + ["rl_cum"]].reset_index(drop=True).rename(
        columns={"rl_cum": "cum_rl_ref"})
    mask_b    = (df["gwl_tas"] == gwl_label) & (df["gwl_ds_cf"] == gwl_label)
    df_gwl    = df[mask_b][keys + ["rl_cum"]].reset_index(drop=True).rename(
        columns={"rl_cum": "cum_rl_gwl"})
    df_m      = df_ref.merge(df_gwl, on=keys, how="left")
    df_m      = df_m[df_m["share_re"] == share_re].copy()
    df_m["Combined_Effect"] = (
        (df_m["cum_rl_gwl"] - df_m["cum_rl_ref"]) / df_m["cum_rl_ref"] * 100)
    mmm = (df_m[["poly_idx", "GCM", "Combined_Effect"]]
           .groupby(["GCM", "poly_idx"])["Combined_Effect"].mean()
           .reset_index()
           .groupby("poly_idx")["Combined_Effect"].mean()
           .reset_index())
    mmm.loc[mmm["Combined_Effect"] > vmax, "Combined_Effect"] = vmax
    return mmm


def plot_combined_threshold_sensitivity(
    ds_005, mask_005, ds_01, mask_01, ds_02, mask_02,
    shapefile_path, csv_thr95, csv_thr99, csv_thr995,
    period_hist=(1980, 1999), period_comp=(2000, 2019),
    n_bins_change=5, n_bins_sev=5, dpi=300,
):
    shapefile = gpd.read_file(shapefile_path)

    rgba_005, _, sev_005, color_levels, alpha_levels, sev_edges, change_edges = \
        _compute_valuebyalpha_rgba(ds_005, mask_005, period_hist, period_comp,
                                   n_bins_change, n_bins_sev)
    rgba_01, _, sev_01, _, _, _, _ = \
        _compute_valuebyalpha_rgba(ds_01, mask_01, period_hist, period_comp,
                                   n_bins_change, n_bins_sev)
    rgba_02, _, sev_02, _, _, _, _ = \
        _compute_valuebyalpha_rgba(ds_02, mask_02, period_hist, period_comp,
                                   n_bins_change, n_bins_sev)

    mmm_95  = _load_wsbd_gwl2(csv_thr95)
    mmm_99  = _load_wsbd_gwl2(csv_thr99)
    mmm_995 = _load_wsbd_gwl2(csv_thr995)

    n_neg    = 50
    n_pos    = int(n_neg * 800 / 100)
    base_wsbd = plt.get_cmap("RdYlGn_r")
    cols_wsbd = ([base_wsbd(v) for v in np.linspace(0.0, 0.45, n_neg)] +
                [base_wsbd(v) for v in np.linspace(0.55, 1.0, n_pos)])
    cmap_wsbd = LinearSegmentedColormap.from_list("wsbd_cmap", cols_wsbd, N=300)
    norm_wsbd = mcolors.Normalize(vmin=-100, vmax=800)
    gdf_re   = gpd.read_file(shapefile_path)
    gdf_re["poly_idx"] = gdf_re.index

    # 3-row 2-col layout left: value-by-alpha, right: WSBD
    fig_width_in  = FIG_WIDTH_IN        # same column width as all other figures
    fig_height_in = fig_width_in   # 3 rows: approx 3� single-map height
    proj = ccrs.Robinson()
    fig  = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    gs   = fig.add_gridspec(3, 2, hspace=0.25, wspace=0.05,
                            left=0.01, right=0.99, top=0.97, bottom=0.07)

    axes_left  = [fig.add_subplot(gs[i, 0], projection=proj) for i in range(3)]
    axes_right = [fig.add_subplot(gs[i, 1], projection=proj) for i in range(3)]

    REF_COLOR = "#c0392b"

    # --- Left column: value-by-alpha maps ---
    left_cfgs = [
        (axes_left[0],  rgba_005, sev_005, ds_005, "a", "thr = 0.05",              False),
        (axes_left[1],  rgba_01,  sev_01,  ds_01,  "b", "thr = 0.10  [reference]", True),
        (axes_left[2],  rgba_02,  sev_02,  ds_02,  "c", "thr = 0.20",              False),
    ]
    for ax, rgba_map, sev_da, ds_src, letter, thr_label, is_ref in left_cfgs:
        ax.imshow(
            rgba_map,
            extent=[sev_da.lon.min().item(), sev_da.lon.max().item(),
                    sev_da.lat.min().item(), sev_da.lat.max().item()],
            origin="lower", transform=ccrs.PlateCarree(),
            interpolation="nearest", rasterized=True,
        )
        da_m = ds_src.frequency.isel(year=0)
        t_m  = rasterio.transform.from_bounds(
            da_m.lon.min().item(), da_m.lat.min().item(),
            da_m.lon.max().item(), da_m.lat.max().item(),
            len(da_m.lon), len(da_m.lat),
        )
        land_m  = rasterize_shapefile(shapefile, da_m.shape, t_m)[::-1, :]
        ocean_m = land_m & (da_m.isnull())
        ax.contourf(ocean_m.lon, ocean_m.lat, ocean_m.values.astype(float),
                    levels=[0.5, 1], colors=["gray"],
                    transform=ccrs.PlateCarree(), zorder=5)
        shapefile.boundary.plot(ax=ax, color="black", linewidth=0.15,
                                transform=ccrs.PlateCarree(), zorder=10)
        ax.annotate(
            f"$\\mathbf{{{letter}}}$",
            xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=6,
            color=REF_COLOR if is_ref else "black",
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )
        ax.set_title(f"{thr_label}", fontsize=6,
                     color=REF_COLOR if is_ref else "black")
        ax.set_extent([-180, 180, -58, 68], crs=ccrs.PlateCarree())
        ax.spines["geo"].set_visible(False)
        if is_ref:
            try:
                ax.spines["geo"].set_edgecolor(REF_COLOR)
                ax.spines["geo"].set_linewidth(2.0)
                ax.spines["geo"].set_visible(True)
            except Exception:
                pass

    # Bivariate legend (lower-left of left column)
    legend_rgba = np.zeros((n_bins_change, n_bins_sev, 4))
    for ic in range(n_bins_change):
        legend_rgba[ic, :, :3] = color_levels[ic, :3]
        legend_rgba[ic, :,  3] = alpha_levels
    leg_ax = fig.add_axes([0.03, 0.06, 0.08, 0.10])
    leg_ax.imshow(legend_rgba, origin="lower", aspect="equal")
    leg_ax.set_xticks([0, n_bins_sev // 2, n_bins_sev - 1])
    leg_ax.set_xticklabels(["low", "mid", "high"], fontsize=5, ha="center")
    leg_ax.set_yticks([0, n_bins_change // 2, n_bins_change - 1])
    leg_ax.set_yticklabels(
        [f"<{change_edges[1]:.0f}%",
         f"{change_edges[n_bins_change // 2]:.0f}%;{change_edges[n_bins_change // 2 + 1]:.0f}%",
         f">+{change_edges[-2]:.0f}%"],
        fontsize=5, va="center",
    )
    leg_ax.set_xlabel("Reference\nseverity", fontsize=5, labelpad=4)
    leg_ax.set_ylabel("Rel. change (%)",     fontsize=5, labelpad=4)
    leg_ax.tick_params(axis="both", which="both", length=0)

    # --- Right column: WSBD maps ---
    right_cfgs = [
        (axes_right[0], mmm_95,  "d", "RL thr = 0.95",              False),
        (axes_right[1], mmm_99,  "e", "RL thr = 0.99  [reference]", True),
        (axes_right[2], mmm_995, "f", "RL thr = 0.995",             False),
    ]
    for ax, mmm, letter, thr_label, is_ref in right_cfgs:
        gdf  = gdf_re.copy().merge(mmm, on="poly_idx", how="left")
        vals = gdf["Combined_Effect"].to_numpy()
        fcs  = [cmap_wsbd(norm_wsbd(v)) if np.isfinite(v) else (0.8, 0.8, 0.8, 1.0)
                for v in vals]
        for geom, fc in zip(gdf.geometry, fcs):
            if geom is None:
                continue
            ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor=fc, edgecolor="none", zorder=2)
        shapefile.boundary.plot(ax=ax, color="black", linewidth=0.15,
                                transform=ccrs.PlateCarree(), zorder=10)
        ax.set_extent([-180, 180, -58, 68], crs=ccrs.PlateCarree())
        ax.annotate(
            f"$\\mathbf{{{letter}}}$",
            xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=6,
            color=REF_COLOR if is_ref else "black",
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )
        ax.set_title(f"{thr_label}",
                     fontsize=6, color=REF_COLOR if is_ref else "black")
        ax.spines["geo"].set_visible(False)
        if is_ref:
            try:
                ax.spines["geo"].set_edgecolor(REF_COLOR)
                ax.spines["geo"].set_linewidth(2.0)
                ax.spines["geo"].set_visible(True)
            except Exception:
                pass

    # WSBD shared colorbar (bottom of right column)
    cbar_ax = fig.add_axes([0.54, 0.025, 0.44, 0.012])
    sm = plt.cm.ScalarMappable(cmap=cmap_wsbd, norm=norm_wsbd)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="max")
    cb.set_label(
        "WSBDs change under 2°C warming compared to 0.61°C (%)",
        fontsize=5,
    )
    cb.ax.tick_params(labelsize=5)
    return fig


# =============================================================================
# Global mean change + bootstrap CI
# =============================================================================

def compute_global_change_stats(ds_final, mask, n_bootstrap=1000, block_size=10):
    ds = ds_final.where(mask == 1)
    ds["annual_severity"] = ds["frequency"] * ds["severity"] * ds["duration"]
    early = ds["annual_severity"].sel(year=slice(1980, 1999)).mean(dim="year")
    late  = ds["annual_severity"].sel(year=slice(2000, 2019)).mean(dim="year")
    weights      = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    global_early      = early.weighted(weights).mean(dim=["lat", "lon"]).values
    global_late       = late.weighted(weights).mean(dim=["lat", "lon"]).values
    global_rel_change = (global_late - global_early) / global_early * 100

    data         = (late - early).values
    lats         = early.lat.values
    lons         = early.lon.values
    weight_array = weights.values
    lat_blocks   = np.arange(lats.min(), lats.max(), block_size)
    lon_blocks   = np.arange(lons.min(), lons.max(), block_size)

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

    ci_lower_rel = np.percentile(bootstrap_means, 2.5)  / global_early * 100
    ci_upper_rel = np.percentile(bootstrap_means, 97.5) / global_early * 100
    return global_rel_change, ci_lower_rel, ci_upper_rel


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.data_path is not None:
        print(f"Loading pre-computed dataset from {args.data_path}")
        ds_final = xr.open_dataset(args.data_path)
    else:
        print("No --data_path provided computing ds_final on-the-fly ")
        ds_final = build_ds_final(
            path_preprocessed=args.path_preprocessed,
            reanalysis=args.reanalysis,
            thr=args.threshold,
            ref_start=args.ref_start,
            ref_end=args.ref_end,
            shapefile_path=args.shapefile,
        )
        if args.save_nc:
            thr_str = str(args.threshold).replace(".", "")
            out_nc  = os.path.join(args.path_preprocessed, "agg_datasets",
                                   f"ds_final_high_non_zero_{thr_str}_{args.reanalysis}.nc")
            os.makedirs(os.path.dirname(out_nc), exist_ok=True)
            print(f"  Saving dataset to {out_nc}  ")
            ds_final.to_netcdf(out_nc)
            print(f"  Saved : {out_nc}")

    print("Building land mask")
    mask = build_land_mask(ds_final, args.shapefile)

    print("Plotting value-by-alpha figure")
    fig1 = plot_reanalysis_disagg_timeseries_valuebyalpha_discrete(
        ds_final=ds_final, mask=mask, shapefile_path=args.shapefile,
        map_title=" ",
        relchange_label="Relative change in\nannual WSED severity(%)",
        sev_label="Historical average\nannual WSED severity",
        lat_min=-60, lat_max=75,
        period_hist=(1980, 1999), period_comp=(2000, 2019),
        n_boot=args.n_boot,
    )
    out1 = os.path.join(args.output_dir, "main",
                        f"fig1_valuebyalpha_slides_{str(args.threshold).replace('.', '')}.png")
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    fig1.savefig(out1, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved {out1}")

    print("Plotting variability map  ")
    fig2 = plot_variability_map(ds_final, mask, args.shapefile, dpi=args.dpi)
    out2 = os.path.join(args.output_dir, "supp", "suppfig2_pdd_std_map.png")
    os.makedirs(os.path.dirname(out2), exist_ok=True)
    fig2.savefig(out2, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out2}")

    print("Plotting mean variable maps (6 panels)")
    fig3 = plot_mean_variables_6panel(
        ds_final=ds_final, mask=mask, shapefile_path=args.shapefile,
        path_preprocessed=args.path_preprocessed,
        reanalysis=args.reanalysis,
        ref_start=args.ref_start, ref_end=args.ref_end,
    )
    out3 = os.path.join(args.output_dir, "supp", "suppfig3_mean_variables_6panel.png")
    os.makedirs(os.path.dirname(out3), exist_ok=True)
    fig3.savefig(out3, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved {out3}")

    _have_vba = (args.data_path_005 is not None and args.data_path_02 is not None)
    _have_rl  = args.rl_path is not None

    if _have_vba and _have_rl:
        csv_thr95  = os.path.join(args.rl_path,
                                  "rl_agg_adaptation_Annual_0.95_ren_pen_0.5_current_v2.csv")
        csv_thr99  = os.path.join(args.rl_path,
                                  "rl_agg_adaptation_Annual_0.99_ren_pen_0.5_current_v2.csv")
        csv_thr995 = os.path.join(args.rl_path,
                                  "rl_agg_adaptation_Annual_0.995_ren_pen_0.5_current_v2.csv")
        print("Plotting combined threshold-sensitivity figure  ")
        ds_005   = xr.open_dataset(args.data_path_005)
        ds_02    = xr.open_dataset(args.data_path_02)
        mask_005 = build_land_mask(ds_005, args.shapefile)
        mask_02  = build_land_mask(ds_02,  args.shapefile)
        fig_comb = plot_combined_threshold_sensitivity(
            ds_005, mask_005, ds_final, mask, ds_02, mask_02,
            shapefile_path=args.shapefile,
            csv_thr95=csv_thr95, csv_thr99=csv_thr99, csv_thr995=csv_thr995,
            dpi=args.dpi,
        )
        out_comb = os.path.join(args.output_dir, "supp",
                                "suppfig_combined_threshold_sensitivity.png")
        os.makedirs(os.path.dirname(out_comb), exist_ok=True)
        fig_comb.savefig(out_comb, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_comb)
        print(f"Saved : {out_comb}")
    elif _have_vba:
        print("Plotting value-by-alpha sensitivity figure  ")
        ds_005   = xr.open_dataset(args.data_path_005)
        ds_02    = xr.open_dataset(args.data_path_02)
        mask_005 = build_land_mask(ds_005, args.shapefile)
        mask_02  = build_land_mask(ds_02,  args.shapefile)
        fig_sens = plot_valuebyalpha_sensitivity(
            ds_005, mask_005, ds_02, mask_02, shapefile_path=args.shapefile)
        out_sens = os.path.join(args.output_dir, "supp",
                                "suppfig_valuebyalpha_sensitivity.png")
        os.makedirs(os.path.dirname(out_sens), exist_ok=True)
        fig_sens.savefig(out_sens, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_sens)
        print(f"Saved : {out_sens}")
    else:
        print("Skipping sensitivity suppfig.")

    print("Computing global change statistics  ")
    global_rel_change, ci_lower_rel, ci_upper_rel = compute_global_change_stats(
        ds_final, mask, n_bootstrap=1000)
    print(f"\nRésultats:")
    print(f"  Changement global : {global_rel_change:.2f}%")
    print(f"  95% CI            : [{ci_lower_rel:.2f}%, {ci_upper_rel:.2f}%]")
    print(f"  Largeur du CI     : {ci_upper_rel - ci_lower_rel:.2f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
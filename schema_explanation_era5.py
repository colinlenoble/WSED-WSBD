# -*- coding: utf-8 -*-
"""
ERA5 version of the combined RED+EBD schema figure
(`suppfig1_combined_red_ebd_schema`) from `3.7 schema_explanation.ipynb`
(`code_final`).

The notebook builds that figure from GCM (W5E5-bias-corrected) wpp/spp/tas
already aggregated to one poly_idx (country). This script reproduces the
same series -- and feeds them into the *same* `plot_combined_red_ebd_schema`
-- but computes wpp/spp/tas directly from raw ERA5 fields for one country,
using `calculate_wind_solar_cf.py` (wind: local/global-shear power curve;
solar: PVGIS) instead of the GCM pipeline.

NOT RUNNABLE YET: no ERA5 daily file/zarr store covering the target region
and period has been downloaded. Get one with `0.1 download_era5_hourly.py`
(-> resampled daily netCDF) or `0.2 download_era5_yearly.py` (-> daily
zarr), point DATA_DIR / FILE_FORMAT at it below, then run this script.

Everything downstream of the ERA5 load (thresholds, the current-mix `epp`
proxy, the temperature-driven demand proxy, the residual-load threshold, the
figure itself) is copied unchanged from the notebook cells -- it only
depends on having wpp/spp/tas time series for one region, not on where they
came from.
"""
import os

# netCDF4 must be imported before geopandas/rasterio below -- on this
# machine, importing rasterio/GDAL first loads a conflicting HDF5 DLL that
# breaks netCDF4 (ImportError: DLL load failed while importing _netCDF4),
# which then also breaks xr.open_mfdataset's default netcdf4 engine used by
# load_era5 for FILE_FORMAT="netcdf" (see edh_common.py / the "era5_dl"
# kernel note in wcf_scf_era5_review.ipynb).
import netCDF4  # noqa: F401

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from rasterio.features import geometry_mask
import rasterio

from calculate_wind_solar_cf import (
    load_era5, compute_wind_cf, compute_solar_cf,
    WindConfig, DEFAULT_WIND_CONFIG, DEFAULT_PVGIS_COEFFICIENTS, load_local_shear,
)

# -------------------------------------------------------------------------
# Config -- adjust to taste once ERA5 data is available
# -------------------------------------------------------------------------

DATA_DIR = "E:/climate_data/ERA5"
FILE_FORMAT = "zarr"  # "zarr" (0.2 download_era5_yearly.py) or "netcdf" (0.1 download_era5_hourly.py)
FREQ = "daily"

SHAPEFILE_PATH = "../../shapefiles/final_shp/ne_mix_adm0_adm1.shp"
SHARE_RENEWABLE_CSV = "../../socio_data/share_renewable.csv"
OUT_DIR = "../../final_figs/schemas/"

REGION_NAME = "Morocco"
PERIODS = [("1990-01-01", "2020-12-31")]  # ERA5 window to load; also the RED-threshold reference window

# Alpha (Hellmann shear exponent) for the 10 m -> 100 m wind extrapolation.
# None -> ALPHA_GLOBAL (EPPConfig.wind_height_exponent, 2.1 calculate_epp_GCM_clean.py),
# same fallback used in wcf_scf_era5_review.ipynb. Pass a fit_local_shear.py
# output path here to use the per-pixel exponent instead.
ALPHA_PATH = None
ALPHA_GLOBAL = 0.143

TOT_RE = 0.5  # renewable penetration used for the EBD residual load (rl = demand - TOT_RE * epp)

FIG_WIDTH_IN    = 5.15
FIG_HEIGHT_IN   = 2.5
FS_PANEL_TITLE  = 7
FS_AXIS_LABEL   = 5
FS_TICK         = 5
FS_REGION_LETTER = 7


# -------------------------------------------------------------------------
# ERA5 -> region-mean wpp/spp/tas
# -------------------------------------------------------------------------

def rasterize_single_region(shapefile_path, region_name, lat, lon, name_field="name"):
    """
    Boolean mask (True inside `region_name`'s polygon), on the ERA5 lat/lon
    grid, plus that region's row index in the shapefile (`poly_idx`, used to
    look up its current wind/solar mix in SHARE_RENEWABLE_CSV -- same index
    convention as the notebook's `idx = shapefile[shapefile['name']==region].index[0]`).

    Same geometry_mask approach as fit_local_shear.rasterize_region_mask,
    but scoped to the one matching row instead of every geometry in the
    shapefile (that helper rasterizes the whole file, which is fine for a
    single-region file like shp_re.shp but not for a multi-country one like
    ne_mix_adm0_adm1.shp).
    """
    shapefile = gpd.read_file(shapefile_path)
    matches = shapefile[shapefile[name_field] == region_name]
    if matches.empty:
        raise ValueError(f"Region {region_name!r} not found in {shapefile_path}")
    poly_idx = matches.index[0]

    transform = rasterio.transform.from_bounds(
        float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()),
        len(lon), len(lat),
    )
    mask = geometry_mask(
        geometries=matches.geometry,
        out_shape=(len(lat), len(lon)),
        transform=transform,
        invert=True,
        all_touched=True,
    )
    if lat.values[0] < lat.values[-1]:
        mask = mask[::-1, :]
    mask_da = xr.DataArray(mask, dims=(lat.dims[0], lon.dims[0]),
                           coords={lat.dims[0]: lat, lon.dims[0]: lon})
    return mask_da, poly_idx


def load_region_cf_series(
    data_dir, shapefile_path, region_name,
    freq=FREQ, periods=PERIODS, file_format=FILE_FORMAT,
    alpha_path=ALPHA_PATH, wind_cfg: WindConfig = DEFAULT_WIND_CONFIG,
    pv_cfg=DEFAULT_PVGIS_COEFFICIENTS,
):
    """
    Load raw ERA5 fields for `periods`, compute gridded wind/solar capacity
    factor (calculate_wind_solar_cf.py), then average over `region_name`'s
    polygon to get one wpp/spp/tas daily time series for that country --
    the ERA5 analog of wpp_ref/spp_ref/tas_ref in the notebook.

    Returns (wpp_ref, spp_ref, tas_ref, poly_idx): wpp_ref/spp_ref are
    single-variable Datasets (.wpp / .spp, matching what
    plot_combined_red_ebd_schema expects), tas_ref is a DataArray.
    """
    ds = load_era5(data_dir, freq=freq, periods=periods, file_format=file_format)

    mask, poly_idx = rasterize_single_region(shapefile_path, region_name, ds["lat"], ds["lon"])
    kept_pct = 100 * float(mask.sum()) / mask.size
    print(f"{region_name}: {kept_pct:.2f}% of the ERA5 grid kept by the region mask")

    alpha = ALPHA_GLOBAL if alpha_path is None else load_local_shear(alpha_path, target_grid=ds)

    wpp = compute_wind_cf(ds["sfcWind"], alpha, cfg=wind_cfg)
    spp = compute_solar_cf(ds["tas"], ds["rsds"], ds["sfcWind"], cfg=pv_cfg)

    wpp_ref = wpp.where(mask).mean(["lat", "lon"]).compute().to_dataset(name="wpp")
    spp_ref = spp.where(mask).mean(["lat", "lon"]).compute().to_dataset(name="spp")
    tas_ref = ds["tas"].where(mask).mean(["lat", "lon"]).compute()

    return wpp_ref, spp_ref, tas_ref, poly_idx


# -------------------------------------------------------------------------
# Thresholds, current-mix epp proxy, demand proxy, RL threshold
# (unchanged from the "3.7 schema_explanation.ipynb" cells)
# -------------------------------------------------------------------------

def compute_red_thresholds(wpp_ref, spp_ref):
    """10th percentile of non-zero wpp/spp over the loaded period -- the RED threshold."""
    wpp_thr = wpp_ref.where(wpp_ref.wpp > 0).wpp.quantile(0.1, dim="time")
    spp_thr = spp_ref.where(spp_ref.spp > 0).spp.quantile(0.1, dim="time")
    return wpp_thr, spp_thr


def compute_epp_and_demand(wpp_ref, spp_ref, tas_ref, poly_idx, share_csv_path=SHARE_RENEWABLE_CSV, tot_re=TOT_RE):
    """
    epp: current-mix renewable production proxy, blending this region's ERA5
    wpp/spp by its current wind/solar generation split (SHARE_RENEWABLE_CSV).
    demand: temperature-driven demand proxy, scaled so its mean matches
    mean(epp) (i.e. renewable supply == demand on average).
    rl_thr: 99th percentile of the resulting residual load (demand - tot_re*epp),
    the EBD threshold.
    """
    df = pd.read_csv(share_csv_path)
    share_by_poly = xr.DataArray(df["current_ratio"].values, coords=[df["poly_idx"].values], dims=["poly_idx"])
    current_share = float(share_by_poly.sel(poly_idx=poly_idx).values)

    epp = current_share * spp_ref.spp + (1 - current_share) * wpp_ref.wpp
    epp_mean = epp.mean(dim="time")

    demand_temp = xr.where(
        tas_ref < 12.5, (12.5 - tas_ref) * 0.026,
        xr.where(tas_ref > 19.6, (tas_ref - 19.6) * 0.035, 0.0)
    )
    demand_bas = epp_mean / (1 + demand_temp.mean(dim="time"))
    demand = demand_bas * (1 + demand_temp)

    demand["time"] = pd.to_datetime(demand["time"].dt.strftime("%Y-%m-%d").values)
    epp["time"] = pd.to_datetime(epp["time"].dt.strftime("%Y-%m-%d").values)

    rl = demand - tot_re * epp
    rl_thr = rl.quantile(0.99, dim="time")

    return demand, epp, rl_thr


# -------------------------------------------------------------------------
# Figure (copied as-is from 3.7 schema_explanation.ipynb -- generic over
# any wpp_ref/spp_ref/demand/epp time series, GCM- or ERA5-derived)
# -------------------------------------------------------------------------

def plot_combined_red_ebd_schema(
    wpp_ref, spp_ref, wpp_thr, spp_thr,
    demand, epp, rl_thr,
    year=1990, month=1, region_name="Morocco",
):
    """
    Combined two-panel figure:
      a) RED - Renewable Energy Drought schema
      b) EBD - Energy Balance Deficit schema
    """

    def _contiguous_segments(mask):
        mask = np.asarray(mask).astype(bool)
        if mask.sum() == 0:
            return []
        idx = np.where(mask)[0]
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends   = np.r_[idx[breaks] + 1, idx[-1] + 1]
        return list(zip(starts, ends))

    start = f"{year}-{month:02d}-01"
    end   = f"{year}-12-31"

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, wspace=0.06, width_ratios=[1, 1])

    # -- panel a -----------------------------------------------------------

    gs_a = subfigs[0].add_gridspec(2, 1, height_ratios=[3.5, 1.2], hspace=0.05)
    ax_a1 = subfigs[0].add_subplot(gs_a[0])
    ax_a2 = ax_a1.twinx()
    ax_ab = subfigs[0].add_subplot(gs_a[1], sharex=ax_a1)

    w = wpp_ref.sel(time=slice(start, end)).wpp
    s = spp_ref.sel(time=slice(start, end)).spp
    t = pd.to_datetime(w["time"].values)

    wthr = wpp_thr.sel(time=slice(start, end)).values if "time" in wpp_thr.dims else np.full(w.size, float(wpp_thr))
    sthr = spp_thr.sel(time=slice(start, end)).values if "time" in spp_thr.dims else np.full(s.size, float(spp_thr))

    dw = np.maximum(0.0, wthr - w.values)
    ds = np.maximum(0.0, sthr - s.values)
    compound_mask = ((w.values <= wthr) & (s.values <= sthr)).astype(int)
    intensity_w   = compound_mask * dw
    intensity_s   = compound_mask * ds

    ax_a1.plot(t, w.values, lw=0.55, label="WCF", color="tab:blue")
    ax_a2.plot(t, s.values, lw=0.55, label="SCF", color="tab:orange")
    ax_a1.plot(t, wthr, "--", lw=0.45, alpha=0.8, label="Threshold", color="tab:blue")
    ax_a2.plot(t, sthr, "--", lw=0.45, alpha=0.8, color="tab:orange")

    for i0, i1 in _contiguous_segments(compound_mask):
        span_start = t[i0]  - pd.Timedelta(hours=12)
        span_end   = t[i1-1]+ pd.Timedelta(hours=12)
        ax_a1.axvspan(span_start, span_end, alpha=0.12, zorder=0)
        ax_ab.axvspan(span_start, span_end, alpha=0.12, zorder=0)

    ax_ab.bar(t, intensity_w, width=1.0, alpha=1, label="Wind", color="tab:blue")
    ax_ab.bar(t, intensity_s, width=1.0, alpha=0.80, bottom=intensity_w, label="Solar", color="tab:orange")

    ax_a1.set_title("")
    ax_a1.text(
        0.01, 1.12, "a",
        transform=ax_a1.transAxes,
        ha="left", va="top",
        fontsize=FS_REGION_LETTER, fontweight="bold"
    )

    ax_a1.set_ylabel("Wind capacity factor", fontsize=FS_AXIS_LABEL)
    ax_a2.set_ylabel("Solar capacity factor", fontsize=FS_AXIS_LABEL)
    ax_ab.set_ylabel("Intensity\n(EFLD)", fontsize=FS_AXIS_LABEL)

    ax_a1.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax_ab.grid(True, axis="y", alpha=0.25, linestyle="--")
    plt.setp(ax_a1.get_xticklabels(), visible=False)

    ax_ab.xaxis.set_major_locator(mdates.WeekdayLocator(interval=10))
    ax_ab.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b-%d"))

    for tick in ax_ab.get_xticklabels():
        tick.set_rotation(20)
        tick.set_fontsize(FS_TICK)

    ax_a1.tick_params(labelsize=FS_TICK)
    ax_a2.tick_params(labelsize=FS_TICK)
    ax_ab.tick_params(labelsize=FS_TICK)

    red_patch = Patch(alpha=0.12, label="RED window")
    leg_a1 = ax_a1.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", lw=0.55, label="WCF"),
            Line2D([0], [0], color="tab:orange", lw=0.55, label="SCF"),
            Line2D([0], [0], color="black", lw=0.45, ls="--", label="Threshold"),
            red_patch,
        ],
        loc="upper left", bbox_to_anchor=(0.56, 1),
        bbox_transform=ax_a1.transAxes,
        frameon=False, fontsize=FS_TICK,
        handlelength=1.8, borderpad=0.2, labelspacing=0.25
    )
    leg_a1.set_zorder(1000)

    leg_ab = ax_ab.legend(
        loc="upper left", bbox_to_anchor=(0.1, 1.0),
        bbox_transform=ax_ab.transAxes,
        frameon=False, fontsize=FS_TICK,
        handlelength=1.4, borderpad=0.2, labelspacing=0.25
    )
    leg_ab.set_zorder(1000)

    # -- panel b -----------------------------------------------------------

    gs_b = subfigs[1].add_gridspec(3, 1, height_ratios=[2, 1.1, 1.1], hspace=0.05)
    ax_b1 = subfigs[1].add_subplot(gs_b[0])
    ax_b2 = subfigs[1].add_subplot(gs_b[1], sharex=ax_b1)
    ax_b3 = subfigs[1].add_subplot(gs_b[2], sharex=ax_b1)

    d    = demand.sel(time=slice(start, end))
    s_re = epp.sel(time=slice(start, end))
    tt   = pd.to_datetime(d["time"].values)
    ren_tot = TOT_RE
    rl_vals = d.values - ren_tot*s_re.values
    thr     = float(rl_thr)
    extreme = rl_vals > thr
    anomaly_pos = np.where(rl_vals - thr > 0, rl_vals - thr, 0)

    cmap = plt.get_cmap('PuOr')
    ax_b1.plot(tt, d.values, lw=0.6, label="Demand", color=cmap(0.82))
    ax_b1.plot(tt, ren_tot*s_re.values, lw=0.6, label="Renewable supply", color=cmap(0.18))

    ax_b2.plot(tt, rl_vals, lw=0.4, color="black", label="Residual load")
    ax_b2.axhline(thr, ls="--", lw=0.7, color="black", label="Threshold")
    ax_b3.bar(tt, anomaly_pos, width=1.0, color="#CC2626", alpha=0.85, label="Exceedance")
    ax_b3.set_ylabel("Exceedance\n(EBD)", fontsize=FS_AXIS_LABEL)
    shade_color, shade_alpha = "tab:blue", 0.12
    for i0, i1 in _contiguous_segments(extreme):
        ts = tt[i0]  - pd.Timedelta(hours=12)
        te = tt[i1-1]+ pd.Timedelta(hours=12)
        ax_b1.axvspan(ts, te, color=shade_color, alpha=shade_alpha, zorder=0)
        ax_b2.axvspan(ts, te, color=shade_color, alpha=shade_alpha, zorder=0)
        ax_b3.axvspan(ts, te, color=shade_color, alpha=shade_alpha, zorder=0)

    ax_b1.set_title("")
    ax_b1.text(
        0.01, 1.16, "b",
        transform=ax_b1.transAxes,
        ha="left", va="top",
        fontsize=FS_REGION_LETTER, fontweight="bold"
    )

    ax_b1.set_ylabel("Energy (normalised)", fontsize=FS_AXIS_LABEL)
    ax_b2.set_ylabel("Residual load", fontsize=FS_AXIS_LABEL)

    for ax in (ax_b1, ax_b2, ax_b3):
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=FS_TICK)

    plt.setp(ax_b1.get_xticklabels(), visible=False)
    plt.setp(ax_b2.get_xticklabels(), visible=False)

    ax_b3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=10))
    ax_b3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b-%d"))

    for tick in ax_b3.get_xticklabels():
        tick.set_rotation(20)
        tick.set_fontsize(FS_TICK)

    ebd_patch = Patch(facecolor=shade_color, alpha=shade_alpha, label="EBD window")
    leg_b1 = ax_b1.legend(
        handles=[
            Line2D([0], [0], color=cmap(0.82), lw=0.6, label="Demand"),
            Line2D([0], [0], color=cmap(0.18), lw=0.6, label="Renewable supply"),
            ebd_patch,
        ],
        loc="upper left", bbox_to_anchor=(0.55, 1.0),
        bbox_transform=ax_b1.transAxes,
        frameon=False, fontsize=FS_TICK,
        handlelength=1.8, borderpad=0.2, labelspacing=0.25
    )
    leg_b1.set_zorder(10)

    leg_b2 = ax_b2.legend(
        loc="upper left", bbox_to_anchor=(0.6, 0.5),
        bbox_transform=ax_b2.transAxes,
        frameon=False, fontsize=FS_TICK,
        handlelength=1.6, borderpad=0.2, labelspacing=0.25
    )
    leg_b2.set_zorder(10)

    return fig


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

if __name__ == "__main__":
    wpp_ref, spp_ref, tas_ref, poly_idx = load_region_cf_series(
        DATA_DIR, SHAPEFILE_PATH, REGION_NAME,
        freq=FREQ, periods=PERIODS, file_format=FILE_FORMAT,
    )
    wpp_thr, spp_thr = compute_red_thresholds(wpp_ref, spp_ref)
    demand, epp, rl_thr = compute_epp_and_demand(wpp_ref, spp_ref, tas_ref, poly_idx)

    plot_year = int(PERIODS[0][0][:4])
    fig = plot_combined_red_ebd_schema(
        wpp_ref=wpp_ref,
        spp_ref=spp_ref,
        wpp_thr=wpp_thr,
        spp_thr=spp_thr,
        demand=demand,
        epp=epp,
        rl_thr=rl_thr,
        year=plot_year,
        month=1,
        region_name=REGION_NAME,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "suppfig1_combined_red_ebd_schema_era5.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.show()

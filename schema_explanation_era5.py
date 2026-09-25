# -*- coding: utf-8 -*-
"""
Methodology schema figure (SWED + SWBD) for one region, built from the
country-aggregated GCM files (GFDL-ESM4, ssp245, GWL0.61 reference window,
ERA5-bias-adjusted) -- the same wcf/scf/tas_pop `*_agg_*.nc` inputs fig45.py
and fig_duration_distribution_swed_swbd.py read.

  a) SWED (Solar-Wind Energy Drought): days where both the wind and the
     solar capacity factor fall below their own 10th-percentile threshold
     (percentile of non-zero values over the reference window).
  b) SWBD: days where the residual load (demand - TOT_RE * renewable supply)
     exceeds its 99th-percentile threshold. Demand is the temperature-driven
     proxy of fig45._calculate_rl (Tc=12.5 C, Th=19.6 C, a=0.026/0.035),
     scaled so its mean equals mean renewable supply; supply blends wcf/scf
     by the region's current solar share (share_renewable.csv).
"""
import os

# netCDF4 must be imported before anything that pulls in rasterio/GDAL --
# otherwise a conflicting HDF5 DLL breaks netCDF4 on this machine.
import netCDF4  # noqa: F401

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = "C:/Users/colin/Downloads/final_figs"
FILE_SUFFIX = "_agg_GFDL-ESM4_ssp245_r1i1p1f1_GWL0-61_ERA5_v1.nc"
SHARE_RENEWABLE_CSV = os.path.join(HERE, "..", "socio_data", "share_renewable.csv")
OUT_DIR = os.path.join(HERE, "..", "final_figs", "schemas")

REGION_NAME = "Pakistan"
PLOT_YEAR = None  # None -> the year with the most SWED + SWBD days

SWED_THR = 0.1   # quantile of non-zero wcf/scf
SWBD_THR = 0.99  # quantile of residual load
SWED_METHOD = "global"  # "global": one threshold over the whole period;
                         # "seasonal": per calendar day, rolling window of SEASONAL_WINDOW days
SEASONAL_WINDOW = 31     # days, centred (+-15 d) -- a rolling month
TOT_RE = 0.5     # renewable penetration: rl = demand - TOT_RE * supply

FIG_WIDTH_IN    = 5.15
FIG_HEIGHT_IN   = 2.7
FS_PANEL_TITLE  = 7
FS_AXIS_LABEL   = 5.5
FS_TICK         = 5

# One colour per variable across the whole figure
C_WIND     = "#1f77b4"
C_SOLAR    = "#f0a202"
C_DEMAND   = "#404040"
C_SUPPLY   = "#2ca02c"
C_RL       = "#7b3294"
C_THR      = "0.45"
C_EVENT    = "#d62728"
EVENT_ALPHA = 0.30
THR_STYLE  = dict(color=C_THR, ls="--", lw=0.6)
MONTH_INITIALS = "JFMAMJJASOND"


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------

def load_region_series(region_name, data_dir=DATA_DIR, suffix=FILE_SUFFIX):
    """Daily wcf, scf, tas (deg C) for `region_name`, plus its poly_idx."""
    ds_w = xr.open_dataset(os.path.join(data_dir, f"wcf{suffix}"))
    ds_s = xr.open_dataset(os.path.join(data_dir, f"scf{suffix}"))
    ds_t = xr.open_dataset(os.path.join(data_dir, f"tas_pop{suffix}"))

    names = ds_w["name"].values
    matches = np.where(names == region_name)[0]
    if matches.size == 0:
        raise ValueError(f"Region {region_name!r} not found in wcf{suffix}")
    poly_idx = int(ds_w["poly_idx"].values[matches[0]])

    wcf = ds_w["wcf"].sel(poly_idx=poly_idx).load()
    scf = ds_s["scf"].sel(poly_idx=poly_idx).load()
    tas = (ds_t["tas"].sel(poly_idx=poly_idx) - 273.15).load()
    return wcf, scf, tas, poly_idx


def _seasonal_quantile(da, q, window=SEASONAL_WINDOW):
    """
    Day-of-year threshold: for each calendar day d, the q-quantile of the
    non-zero values of every year whose day-of-year lies within +-window//2
    of d (circular, so late December pools with early January). Returned
    mapped back onto `da`'s time axis.
    """
    vals = da.where(da > 0).values
    doy = da["time"].dt.dayofyear.values
    half = window // 2
    thr_doy = np.full(367, np.nan)
    for d in range(1, 367):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 365 - dist)
        thr_doy[d] = np.nanquantile(vals[dist <= half], q)
    return xr.DataArray(thr_doy[doy], coords={"time": da["time"]}, dims="time")


def compute_swed(wcf, scf, q=SWED_THR, method=SWED_METHOD, window=SEASONAL_WINDOW):
    """
    method="global":   one q-quantile of non-zero values over the whole period
                       (wthr/sthr are floats).
    method="seasonal": a q-quantile per calendar day over a rolling `window`
                       (wthr/sthr are DataArrays on the time axis).
    """
    if method == "global":
        wthr = float(wcf.where(wcf > 0).quantile(q))
        sthr = float(scf.where(scf > 0).quantile(q))
    elif method == "seasonal":
        wthr = _seasonal_quantile(wcf, q, window)
        sthr = _seasonal_quantile(scf, q, window)
    else:
        raise ValueError(f"method must be 'global' or 'seasonal', got {method!r}")
    event = (wcf <= wthr) & (scf <= sthr)
    return wthr, sthr, event


def compute_swbd(wcf, scf, tas, poly_idx, share_csv=SHARE_RENEWABLE_CSV,
                 tot_re=TOT_RE, q=SWBD_THR):
    df = pd.read_csv(share_csv).set_index("poly_idx")
    solar_share = float(df.loc[poly_idx, "current_ratio"])

    supply = solar_share * scf + (1 - solar_share) * wcf
    demand_temp = xr.where(
        tas < 12.5, (12.5 - tas) * 0.026,
        xr.where(tas > 19.6, (tas - 19.6) * 0.035, 0.0)
    )
    demand_bas = supply.mean() / (1 + demand_temp.mean())
    demand = demand_bas * (1 + demand_temp)

    rl = demand - tot_re * supply
    rl_thr = float(rl.quantile(q))
    event = rl > rl_thr
    return demand, supply, rl, rl_thr, event, solar_share


# -------------------------------------------------------------------------
# Figure
# -------------------------------------------------------------------------

def _contiguous_segments(mask):
    mask = np.asarray(mask).astype(bool)
    if mask.sum() == 0:
        return []
    idx = np.where(mask)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends   = np.r_[idx[breaks] + 1, idx[-1] + 1]
    return list(zip(starts, ends))


def _shade_events(axes, t, mask):
    for i0, i1 in _contiguous_segments(mask):
        ts = t[i0] - pd.Timedelta(hours=12)
        te = t[i1 - 1] + pd.Timedelta(hours=12)
        for ax in axes:
            ax.axvspan(ts, te, color=C_EVENT, alpha=EVENT_ALPHA, lw=0, zorder=0)


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FS_TICK, width=0.5, length=2)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.5)


def _month_axis(ax, year):
    ax.set_xlim(pd.Timestamp(f"{year}-01-01") - pd.Timedelta(hours=12),
                pd.Timestamp(f"{year}-12-31") + pd.Timedelta(hours=12))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: MONTH_INITIALS[mdates.num2date(x).month - 1]))
    ax.set_xlabel(str(year), fontsize=FS_AXIS_LABEL)


def _legend(sf, handles):
    """Legend below the panel (subfigure), outside the axes so it never covers data."""
    return sf.legend(handles=handles, loc="outside lower center", ncol=3,
                     frameon=False, fontsize=FS_TICK, handlelength=1.6,
                     borderpad=0.2, labelspacing=0.3, handletextpad=0.5,
                     columnspacing=1.2)


def plot_swed_swbd_schema(wcf, scf, wthr, sthr, swed,
                          demand, supply, rl, rl_thr, swbd, year,
                          tot_re=TOT_RE, swed_q=SWED_THR, swbd_q=SWBD_THR,
                          swed_method=SWED_METHOD):
    sel = dict(time=slice(f"{year}-01-01", f"{year}-12-31"))
    t = pd.to_datetime(wcf.sel(**sel)["time"].values)

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), constrained_layout=True)
    sf_a, sf_b = fig.subfigures(1, 2, wspace=0.04)

    # -- panel a: SWED ------------------------------------------------------
    ax_a = sf_a.subplots()
    ax_a.plot(t, wcf.sel(**sel).values, lw=0.6, color=C_WIND)
    ax_a.plot(t, scf.sel(**sel).values, lw=0.6, color=C_SOLAR)
    for thr in (wthr, sthr):
        if isinstance(thr, xr.DataArray):
            ax_a.plot(t, thr.sel(**sel).values, **THR_STYLE)
        else:
            ax_a.axhline(thr, **THR_STYLE)
    _shade_events([ax_a], t, swed.sel(**sel).values)

    ax_a.set_ylabel("Capacity factor", fontsize=FS_AXIS_LABEL)
    ax_a.set_ylim(bottom=0)
    _month_axis(ax_a, year)
    _style_axis(ax_a)
    _legend(sf_a, [
        Line2D([0], [0], color=C_WIND, lw=0.8, label="Wind"),
        Line2D([0], [0], color=C_SOLAR, lw=0.8, label="Solar"),
        Line2D([0], [0], label=(f"Seasonal threshold (P{swed_q * 100:g})" if swed_method == "seasonal"
                                  else f"Threshold (P{swed_q * 100:g})"), **THR_STYLE),
        Patch(color=C_EVENT, alpha=EVENT_ALPHA, lw=0, label="SWED day"),
    ])

    # -- panel b: SWBD ------------------------------------------------------
    ax_b1, ax_b2 = sf_b.subplots(2, 1, sharex=True, height_ratios=[1.2, 1])
    ax_b1.plot(t, demand.sel(**sel).values, lw=0.6, color=C_DEMAND)
    ax_b1.plot(t, tot_re * supply.sel(**sel).values, lw=0.6, color=C_SUPPLY)
    ax_b2.plot(t, rl.sel(**sel).values, lw=0.6, color=C_RL)
    ax_b2.axhline(rl_thr, **THR_STYLE)
    _shade_events([ax_b1, ax_b2], t, swbd.sel(**sel).values)

    ax_b1.set_ylabel("Energy\n(normalised)", fontsize=FS_AXIS_LABEL)
    ax_b2.set_ylabel("Residual load", fontsize=FS_AXIS_LABEL)
    ax_b1.tick_params(labelbottom=False)
    _month_axis(ax_b2, year)
    ax_b1.set_xlabel("")
    for ax in (ax_b1, ax_b2):
        _style_axis(ax)

    _legend(sf_b, [
        Line2D([0], [0], color=C_DEMAND, lw=0.8, label="Demand"),
        Line2D([0], [0], color=C_SUPPLY, lw=0.8, label="Renewable supply"),
        Line2D([0], [0], color=C_RL, lw=0.8, label="Residual load"),
        Line2D([0], [0], label=f"Threshold (P{swbd_q * 100:g})", **THR_STYLE),
        Patch(color=C_EVENT, alpha=EVENT_ALPHA, lw=0, label="SWBD day"),
    ])

    # -- panel titles (same letter size and placement in both) ---------------
    for sf, letter, title in ((sf_a, "a", "Wind and solar droughts"),
                              (sf_b, "b", "Energy balance droughts")):
        sf.suptitle(rf"$\bf{{{letter}}}$   {title}", x=0.02, ha="left",
                    fontsize=FS_PANEL_TITLE)

    return fig


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

if __name__ == "__main__":
    wcf, scf, tas, poly_idx = load_region_series(REGION_NAME)
    wthr, sthr, swed = compute_swed(wcf, scf)
    demand, supply, rl, rl_thr, swbd, solar_share = compute_swbd(wcf, scf, tas, poly_idx)

    counts = (swed.astype(int) + swbd.astype(int)).groupby("time.year").sum()
    year = int(counts.idxmax()) if PLOT_YEAR is None else PLOT_YEAR
    print(f"{REGION_NAME} (poly_idx={poly_idx}, solar share={solar_share:.2f}): "
          f"plotting {year} -- SWED days={int(swed.sel(time=str(year)).sum())}, "
          f"SWBD days={int(swbd.sel(time=str(year)).sum())}")

    fig = plot_swed_swbd_schema(wcf, scf, wthr, sthr, swed,
                                demand, supply, rl, rl_thr, swbd, year)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"suppfig1_swed_swbd_schema_{REGION_NAME.lower()}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")

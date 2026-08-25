# -*- coding: cp1252 -*-
"""
Persistent compound wind-solar Renewable Energy Drought (RED) events:
comparison of two 20-year periods (default: 1982-2001 vs 2002-2021).

Produces a 3x3 map figure:
  row 1 - longest single RED event (days) observed at each pixel
  row 2 - annual number of RED events lasting more than 4 days
  row 3 - annual number of RED events lasting more than 6 days

Each row shows the historical period, the recent period, and their absolute
difference (recent minus historical). Reuses the event-detection pipeline
from fig_persistent.py (compound low-wind-AND-low-solar rolling-mean days,
grouped into contiguous events).
"""
import os
import config
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
os.environ["ESMFMKFILE"] = config.ESMFMKFILE_XENV

import argparse

import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize, BoundaryNorm, ListedColormap
from matplotlib.patheffects import withStroke
from string import ascii_lowercase

import cmocean as cmo

from fig_persistent import (
    FIG_WIDTH_IN, build_ds_final_persistent, build_land_mask, rasterize_shapefile,
)

DURATION_THRESHOLDS = (4, 6)


# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare longest-event duration and event frequency (by duration "
                     "threshold) of persistent compound WSE droughts between two periods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_path", default=None,
                         help="Pre-computed ds_final .nc (skips rebuilding). Must be used "
                              "together with --events_path.")
    parser.add_argument("--events_path", default=None,
                         help="Pre-computed event-table .parquet (skips rebuilding). Must "
                              "be used together with --data_path.")
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
    parser.add_argument("--output_dir",
                         default=os.path.join(config.SUMMARY_FIGS_DIR, "persistance"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--save_nc", action="store_true", default=False)
    parser.add_argument("--save_events", action="store_true", default=False)
    return parser.parse_args()


# =============================================================================
# Longest single event per pixel, within a given period
# =============================================================================

def compute_longest_event_duration(df_events_dedup, template_da, period):
    """
    Per-pixel duration (days) of the single longest RED event whose start
    year falls within `period` (inclusive). Pixels with no qualifying event
    get 0. `template_da` supplies the full lat/lon grid to reindex onto.
    """
    y0, y1 = period
    sub = df_events_dedup[(df_events_dedup["year"] >= y0) & (df_events_dedup["year"] <= y1)]
    da_max = sub.groupby(["lat", "lon"])["duration"].max().to_xarray()
    da_max = da_max.reindex(lat=template_da.lat, lon=template_da.lon, fill_value=0).fillna(0)
    return da_max


# =============================================================================
# Discrete-classification helpers
# =============================================================================

def _sequential_discrete_edges(vmax, n_bins=5):
    """
    Bin edges for a >=0 quantity, front-loaded near zero (edges follow a
    squared curve) so low-count regions are still told apart from each
    other, not just lumped below the handful of high-count outliers.
    """
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    edges = np.linspace(0.0, vmax, n_bins + 1) ** 2 / vmax
    edges[0] = 0.0
    return edges


def _diverging_discrete_edges(vmax):
    """Symmetric +/- bin edges scaled to the observed extreme (same relative
    breakpoints as the value-by-alpha change classification elsewhere)."""
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    frac_edges = np.array([-1.0, -0.25, -0.1, 0.1, 0.25, 1.0])
    return frac_edges * vmax


def _discrete_cmap_norm(edges, base_cmap):
    n_bins = len(edges) - 1
    colors = base_cmap(np.linspace(0, 1, n_bins))
    return ListedColormap(colors), BoundaryNorm(edges, n_bins)


# =============================================================================
# Figure
# =============================================================================

def plot_red_period_change(
    ds_final, df_events_dedup, mask, shapefile_path,
    period_hist=(1982, 2001), period_comp=(2002, 2021),
    duration_thresholds=DURATION_THRESHOLDS,
    lat_min=-60, lat_max=72,
):
    """
    3x3 panel figure comparing `period_hist` and `period_comp` for:
      row 0 - longest single RED event (days) per pixel
      row 1 - annual number of RED events lasting > duration_thresholds[0] days
      row 2 - annual number of RED events lasting > duration_thresholds[1] days
    Column 3 of every row is the absolute difference (comp minus hist).
    """
    shp = gpd.read_file(shapefile_path)
    y0, y1 = period_hist
    y2, y3 = period_comp

    template_da = ds_final["frequency"].isel(year=0)
    t_ref = rasterio.transform.from_bounds(
        template_da.lon.min().item(), template_da.lat.min().item(),
        template_da.lon.max().item(), template_da.lat.max().item(),
        len(template_da.lon), len(template_da.lat),
    )
    land_mask_full = rasterize_shapefile(shp, template_da.shape, t_ref)[::-1, :]
    ocean_mask = land_mask_full & (mask == 0)

    # --- Row 0: longest single event ---
    longest_hist = compute_longest_event_duration(df_events_dedup, template_da, period_hist)
    longest_comp = compute_longest_event_duration(df_events_dedup, template_da, period_comp)
    longest_hist = longest_hist.where(mask == 1)
    longest_comp = longest_comp.where(mask == 1)
    longest_diff = longest_comp - longest_hist

    # --- Rows 1-2: event count per decade above each duration threshold ---
    freq_rows = []
    for thr in duration_thresholds:
        da_thr = ds_final["n_events_gt_duration"].sel(duration_threshold=thr).where(mask == 1)
        f_hist = da_thr.sel(year=slice(y0, y1)).mean("year") * 10.0
        f_comp = da_thr.sel(year=slice(y2, y3)).mean("year") * 10.0
        freq_rows.append((f_hist, f_comp, f_comp - f_hist))

    row_data = [(longest_hist, longest_comp, longest_diff)] + freq_rows
    row_labels = (
        ["Longest RED event (days)"]
        + [f"RED events/decade lasting > {thr} d" for thr in duration_thresholds]
    )
    row_cmaps = [cmo.cm.matter, cmo.cm.solar.reversed(), cmo.cm.solar.reversed()]
    diverging_cmap = cm.get_cmap("RdBu_r")
    col_titles = [f"{y0}-{y1}", f"{y2}-{y3}", "Difference"]
    panellabels = list(ascii_lowercase[:9])

    fig_width_in = FIG_WIDTH_IN * 1.7
    fig_height_in = fig_width_in * 0.95
    fig, axes = plt.subplots(3, 3, figsize=(fig_width_in, fig_height_in), dpi=300,
                              subplot_kw={"projection": ccrs.Robinson()})

    for r, (hist_da, comp_da, diff_da) in enumerate(row_data):
        if r == 0:
            # Longest event: log scale (duration spans orders of magnitude,
            # from single-week events to multi-month ones) and a diff scale
            # fixed to +/-5 d so typical changes aren't swamped by a few
            # extreme pixels.
            vmax_period = np.nanmax([np.nanmax(hist_da.values), np.nanmax(comp_da.values)])
            vmax_period = vmax_period if np.isfinite(vmax_period) and vmax_period > 1 else 2.0
            log_norm = LogNorm(vmin=1.0, vmax=vmax_period)
            panel_specs = [
                (hist_da.where(hist_da > 0), row_cmaps[r], log_norm, None),
                (comp_da.where(comp_da > 0), row_cmaps[r], log_norm, None),
                (diff_da, diverging_cmap, Normalize(vmin=-5.0, vmax=5.0), [-5, -2.5, 0, 2.5, 5]),
            ]
        else:
            # Event counts: discrete classes so low- vs high-frequency
            # regions are told apart at a glance, rather than blended along
            # a continuous gradient.
            vmax_period = np.nanmax([np.nanmax(hist_da.values), np.nanmax(comp_da.values)])
            seq_edges = _sequential_discrete_edges(vmax_period)
            seq_cmap, seq_norm = _discrete_cmap_norm(seq_edges, row_cmaps[r])
            vabs_diff = np.nanmax(np.abs(diff_da.values))
            div_edges = _diverging_discrete_edges(vabs_diff)
            div_cmap, div_norm = _discrete_cmap_norm(div_edges, diverging_cmap)
            panel_specs = [
                (hist_da, seq_cmap, seq_norm, seq_edges),
                (comp_da, seq_cmap, seq_norm, seq_edges),
                (diff_da, div_cmap, div_norm, div_edges),
            ]

        for c, (da, cmap, norm, ticks) in enumerate(panel_specs):
            ax = axes[r, c]
            ax.set_global()
            ax.coastlines(resolution="50m", linewidth=0.15, color="black")
            ax.contourf(
                template_da.lon, template_da.lat, ocean_mask.astype(float),
                levels=[0.5, 1], colors=["gray"],
                transform=ccrs.PlateCarree(), zorder=5,
            )
            mesh = ax.pcolormesh(
                da.lon, da.lat, da.values,
                transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
                rasterized=True, zorder=3,
            )
            shp.boundary.plot(ax=ax, color="black", linewidth=0.1,
                               transform=ccrs.PlateCarree(), zorder=6)
            if ticks is not None:
                cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.7,
                                     pad=0.05, ticks=ticks)
                cbar.ax.set_xticklabels([f"{t:.1f}" for t in ticks], fontsize=4, rotation=45)
            else:
                cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.7, pad=0.05)
            cbar.ax.tick_params(labelsize=5)
            if c == 0:
                cbar.set_label(row_labels[r], fontsize=5)
            ax.annotate(
                f"$\\mathbf{{{panellabels[r * 3 + c]}}}$",
                xy=(0.02, 1.02), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=6,
                path_effects=[withStroke(linewidth=1.5, foreground="white")],
            )
            if r == 0:
                ax.set_title(col_titles[c], fontsize=6)
            ax.set_extent([-180, 180, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.spines["geo"].set_visible(False)

    fig.suptitle(
        "Persistent compound wind-solar Renewable Energy Drought (RED): "
        f"{y0}-{y1} vs {y2}-{y3}",
        fontsize=7,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    thr_str = str(args.threshold).replace(".", "")
    period_hist = (1982, 2001)
    period_comp = (2002, 2021)

    if args.data_path is not None and args.events_path is not None:
        print(f"Loading pre-computed dataset from {args.data_path}")
        ds_final = xr.open_dataset(args.data_path)
        print(f"Loading pre-computed event table from {args.events_path}")
        df_events_dedup = pd.read_parquet(args.events_path)
    else:
        print("No cached --data_path/--events_path provided -- computing on-the-fly")
        ds_final, df_events_dedup = build_ds_final_persistent(
            path_preprocessed=args.path_preprocessed,
            reanalysis=args.reanalysis,
            threshold=args.threshold,
            ref_start=args.ref_start,
            ref_end=args.ref_end,
            roll_window=args.roll_window,
            duration_thresholds=DURATION_THRESHOLDS,
            return_events=True,
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
            ds_final.to_netcdf(out_nc)
            print(f"  Saved dataset: {out_nc}")
        if args.save_events:
            out_events = os.path.join(
                args.path_preprocessed, "agg_datasets",
                f"events_persistent_{thr_str}_roll{args.roll_window}"
                f"_{args.reanalysis}.parquet",
            )
            os.makedirs(os.path.dirname(out_events), exist_ok=True)
            df_events_dedup.to_parquet(out_events)
            print(f"  Saved event table: {out_events}")

    print("Building land/resource mask")
    mask = build_land_mask(ds_final, args.shapefile)

    print("Plotting RED period-comparison figure")
    fig = plot_red_period_change(
        ds_final=ds_final, df_events_dedup=df_events_dedup, mask=mask,
        shapefile_path=args.shapefile,
        period_hist=period_hist, period_comp=period_comp,
        duration_thresholds=DURATION_THRESHOLDS,
        lat_min=-60, lat_max=75,
    )
    out_path = os.path.join(
        args.output_dir, "main",
        f"fig_red_periodchange_{thr_str}_roll{args.roll_window}.png",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()

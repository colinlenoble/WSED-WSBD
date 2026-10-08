# -*- coding: utf-8 -*-
"""
Extended Data Fig. 10: sensitivity of the regional ERA5 capacity factors
to the spatial aggregation weighting.

  a - map of the mean SCF difference between the two weightings (v1 - v2)
  b - same for WCF
  c - SCF v1 vs. v2, one dot per region, coloured by |v1 - v2|
  d - same for WCF

v1 / v2 are the two weighting schemes of
calculate_cf.aggregate_ds_cf_reanalysis():
  v1 : weighted by the mean capacity factor over the reanalysis period
  v2 : weighted by grid-cell area only

Inputs are {preprocessed_path}/{reanalysis}/{scf,wcf}_agg_{reanalysis}_{v1,v2}.nc
(one value per poly_idx and day) and the shapefile they were aggregated on
(config.SHAPEFILE_PATH, the one calculate_cf.py passes), matched by position.

Port of como24_group5/code_final/1.9 aggregation_figure.ipynb (which used
the W5E5 aggregates instead of ERA5).
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; also puts main_pipeline/, main_figs/, supp_figs/, aux_code/ on sys.path
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
os.environ["ESMFMKFILE"]       = config.ESMFMKFILE_XENV

import argparse

import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from io_utils import open_dataset_any
from fig1 import mask_poles, MAP_LAT_SOUTH, MAP_LAT_NORTH

FIG_WIDTH_IN = 5.15
FIG_HEIGHT_IN = 4.3
DIFF_PERCENTILE = 98   # colour range of the difference maps (outliers saturate)
N_DIFF_LEVELS = 8

FS_PANEL_LETTER = 8
FS_AXIS_LABEL = 6
FS_TICK = 5

# var: (short label, long label, scatter colour)
VARIABLES = {
    "scf": ("SCF", "Solar CF", "#d98c00"),
    "wcf": ("WCF", "Wind CF", "#2f6f9f"),
}
VERSION_LABELS = {"v1": "potential-weighted", "v2": "area-weighted"}


# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="ERA5 capacity factors aggregated with capacity-factor vs. "
                    "area weighting (supplementary figure 10).")
    p.add_argument("--preprocessed_path", default=config.PATH_PREPROCESSED,
                   help="Root holding <reanalysis>/{scf,wcf}_agg_<reanalysis>_{v1,v2}.nc.")
    p.add_argument("--reanalysis", default=config.REANALYSIS)
    p.add_argument("--shapefile", default=config.SHAPEFILE_PATH,
                   help="Shapefile the aggregates were built on (same poly_idx order).")
    p.add_argument("--output_dir", default="../final_figs")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


# =============================================================================
# Data
# =============================================================================

def load_time_mean(preprocessed_path, reanalysis, var, version):
    """Time mean of {var}_agg_{reanalysis}_{version}.nc, as a 1-D array over poly_idx."""
    path = os.path.join(preprocessed_path, reanalysis, f"{var}_agg_{reanalysis}_{version}.nc")
    with open_dataset_any(path) as ds:
        da = ds[var]
        time_dim = "valid_time" if "valid_time" in da.dims else "time"
        mean = da.mean(dim=time_dim).squeeze()
        if mean.ndim != 1:
            raise ValueError(f"{path}: expected one region dim after the time mean, got {mean.dims}")
        return mean.values


def load_aggregates(shapefile, preprocessed_path, reanalysis):
    """Shapefile with {var}_v1, {var}_v2 and {var}_diff (v1 - v2) columns added."""
    gdf = gpd.read_file(shapefile)
    for var in VARIABLES:
        for version in VERSION_LABELS:
            values = load_time_mean(preprocessed_path, reanalysis, var, version)
            if len(values) != len(gdf):
                raise ValueError(
                    f"{var}_agg_{reanalysis}_{version}.nc has {len(values)} regions, "
                    f"{shapefile} has {len(gdf)} -- not the shapefile used for aggregation?")
            gdf[f"{var}_{version}"] = values
        gdf[f"{var}_diff"] = gdf[f"{var}_v1"] - gdf[f"{var}_v2"]
    return gdf


def print_summary(gdf):
    for var, (short, _, _) in VARIABLES.items():
        v1, v2 = gdf[f"{var}_v1"], gdf[f"{var}_v2"]
        diff = (v1 - v2).abs()
        print(f"  {short}: mean |v1 - v2| = {diff.mean():.4f}, max = {diff.max():.4f}, "
              f"corr = {v1.corr(v2):.4f}")


# =============================================================================
# Figure
# =============================================================================

def _panel_letter(ax, letter):
    ax.text(0.0, 1.02, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=FS_PANEL_LETTER, fontweight="bold")


def _draw_diff_map(fig, ax, gdf, col, label, letter):
    """Choropleth of the v1 - v2 difference on a discrete, symmetric diverging
    scale capped at the DIFF_PERCENTILE of |diff| (larger values saturate)."""
    vabs = float(np.nanpercentile(np.abs(gdf[col]), DIFF_PERCENTILE))
    levels = MaxNLocator(nbins=N_DIFF_LEVELS, symmetric=True).tick_values(-vabs, vabs)
    cmap = plt.get_cmap("PuOr_r")
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend="both")

    band = gdf.cx[:, MAP_LAT_SOUTH:MAP_LAT_NORTH]
    band.plot(column=col, ax=ax, cmap=cmap, norm=norm,
              edgecolor="0.35", linewidth=0.1, transform=ccrs.PlateCarree())
    ax.coastlines(resolution="110m", linewidth=0.2)
    ax.set_global()
    mask_poles(ax)
    ax.spines["geo"].set_visible(False)

    cax = ax.inset_axes([0.15, -0.09, 0.7, 0.05])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax,
                      orientation="horizontal", extend="both")
    cb.set_ticks(levels[(len(levels) // 2) % 2::2])   # every other edge, incl. 0
    cb.set_label(f"Δ mean {label} (potential − area weighted)", fontsize=FS_AXIS_LABEL,
                 labelpad=2)
    cb.ax.tick_params(labelsize=FS_TICK, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    _panel_letter(ax, letter)


def _draw_scatter(ax, x, y, label, colour, letter):
    """Potential- (x) vs. area-weighted (y) CF per region, with the 1:1 line
    and summary statistics."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    ax.scatter(x, y, s=4, color=colour, alpha=0.6, edgecolors="none", zorder=3)

    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    margin = 0.03 * (hi - lo)
    lims = [lo - margin, hi + margin]
    ax.plot(lims, lims, color="0.2", linestyle="--", linewidth=0.6, zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    r = np.corrcoef(x, y)[0, 1]
    stats = (f"N = {len(x)}\n"
             f"r = {r:.3f}\n"
             f"bias (area − pot.) = {np.mean(y - x):+.3f}\n"
             f"mean |Δ| = {np.mean(np.abs(y - x)):.3f}")
    ax.text(0.04, 0.96, stats, transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_TICK, linespacing=1.3,
            bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.4,
                      boxstyle="round,pad=0.3", alpha=0.9), zorder=4)

    ax.set_xlabel(f"{label}, potential-weighted", fontsize=FS_AXIS_LABEL)
    ax.set_ylabel(f"{label}, area-weighted", fontsize=FS_AXIS_LABEL)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=FS_TICK, length=2, width=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.grid(True, linestyle="-", linewidth=0.3, alpha=0.25)
    _panel_letter(ax, letter)


def plot_suppfig10(gdf):
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)
    gs = GridSpec(2, 2, height_ratios=[0.62, 1.0], hspace=0.38, wspace=0.28,
                  left=0.08, right=0.98, top=0.96, bottom=0.08, figure=fig)

    for col, (var, (short, long_, colour)) in enumerate(VARIABLES.items()):
        ax_map = fig.add_subplot(gs[0, col], projection=ccrs.EqualEarth())
        _draw_diff_map(fig, ax_map, gdf, f"{var}_diff", short, "ab"[col])

        ax_sc = fig.add_subplot(gs[1, col])
        _draw_scatter(ax_sc, gdf[f"{var}_v1"].to_numpy(), gdf[f"{var}_v2"].to_numpy(),
                      long_, colour, "cd"[col])
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print(f"Loading {args.reanalysis} aggregates from {args.preprocessed_path}")
    gdf = load_aggregates(args.shapefile, args.preprocessed_path, args.reanalysis)
    print_summary(gdf)

    print("Plotting supplementary figure 10")
    fig = plot_suppfig10(gdf)
    os.makedirs(os.path.join(args.output_dir, "supp"), exist_ok=True)
    out_path = os.path.join(args.output_dir, "supp", "suppfig10_aggregation_comparison.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

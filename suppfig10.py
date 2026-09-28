# -*- coding: utf-8 -*-
"""
Supplementary figure 10: sensitivity of the regional ERA5 capacity factors
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
import config
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from io_utils import open_dataset_any

FIG_WIDTH_IN = 5.15
FIG_HEIGHT_IN = 5.5
MAP_EXTENT = [-180, 180, -58, 68]

FS_PANEL_LETTER = 8
FS_AXIS_LABEL = 6
FS_TICK = 5

# var: (short label, long label)
VARIABLES = {
    "scf": ("SCF", "Solar CF"),
    "wcf": ("WCF", "Wind CF"),
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
    for var, (short, _) in VARIABLES.items():
        v1, v2 = gdf[f"{var}_v1"], gdf[f"{var}_v2"]
        diff = (v1 - v2).abs()
        print(f"  {short}: mean |v1 - v2| = {diff.mean():.4f}, max = {diff.max():.4f}, "
              f"corr = {v1.corr(v2):.4f}")


# =============================================================================
# Figure
# =============================================================================

def _panel_letter(ax, letter):
    # Above the axes rather than inside: the cropped PlateCarree maps have
    # land right up to their top-left corner.
    ax.text(0.0, 1.02, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=FS_PANEL_LETTER, fontweight="bold")


def _draw_diff_map(fig, ax, gdf, col, label, letter):
    """Choropleth of the v1 - v2 difference, symmetric diverging colormap."""
    vabs = float(np.nanmax(np.abs(gdf[col])))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    gdf.plot(column=col, ax=ax, cmap="RdBu_r", norm=norm,
             edgecolor="black", linewidth=0.15, transform=ccrs.PlateCarree())
    ax.coastlines(resolution="110m", linewidth=0.15)
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.spines["geo"].set_visible(False)

    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm), ax=ax,
                      orientation="horizontal", pad=0.02, shrink=0.65)
    cb.locator = MaxNLocator(nbins=4, symmetric=True)
    cb.update_ticks()
    cb.set_label(f"Mean {label} difference (v1 − v2)", fontsize=FS_AXIS_LABEL)
    cb.ax.tick_params(labelsize=FS_TICK)
    _panel_letter(ax, letter)


def _draw_scatter(fig, ax, x, y, xlabel, ylabel, letter):
    """v1 vs. v2 per region, coloured by |v1 - v2|, with the 1:1 line."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    sc = ax.scatter(x, y, c=np.abs(x - y), cmap="YlOrRd", s=10, alpha=0.75, zorder=3)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.18, fraction=0.06)
    cb.set_label("|v1 − v2|", fontsize=FS_AXIS_LABEL)
    cb.ax.tick_params(labelsize=FS_TICK)

    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    margin = 0.02 * (hi - lo)
    lims = [lo - margin, hi + margin]
    ax.plot(lims, lims, "k--", linewidth=1.2, zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(xlabel, fontsize=FS_AXIS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, linestyle="--", alpha=0.4)
    _panel_letter(ax, letter)


def plot_suppfig10(gdf):
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)
    gs = GridSpec(2, 2, hspace=0.15, wspace=0.25, figure=fig)

    for col, (var, (short, long_)) in enumerate(VARIABLES.items()):
        ax_map = fig.add_subplot(gs[0, col], projection=ccrs.PlateCarree())
        _draw_diff_map(fig, ax_map, gdf, f"{var}_diff", short, "ab"[col])

        ax_sc = fig.add_subplot(gs[1, col])
        _draw_scatter(fig, ax_sc,
                      gdf[f"{var}_v1"].to_numpy(), gdf[f"{var}_v2"].to_numpy(),
                      f"{long_} (v1, {VERSION_LABELS['v1']})",
                      f"{long_} (v2, {VERSION_LABELS['v2']})",
                      "cd"[col])
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

# -*- coding: utf-8 -*-
"""
Extended Data Fig. 9: effect of the renewable share on SWBD changes.

SWBD changes at each warming level for a 25/50/75 % renewable share, in days
of baseline demand (previously fig45.py STEP 9). Reads the residual-load CSVs
written by main_pipeline/make_rl_files.py; map helpers come from
main_figs/fig45.py.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; also puts main_pipeline/, main_figs/, supp_figs/, aux_code/ on sys.path
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
os.environ["ESMFMKFILE"]       = config.ESMFMKFILE_XENV
os.environ["MPLBACKEND"]       = "Agg"

import argparse
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
import cartopy.crs as ccrs
from map_overlays import discrepancy_mask_legend_handle

from fig45 import (FIG_WIDTH_IN, PATHS, _draw_map, _mmm_absolute_days,
                   _save_fig, load_gwl_dfs, load_hatch_agg)


# =============================================================================
# Extended Data Fig. 9 -- RE share effect, absolute (days of baseline demand)
# =============================================================================

def plot_re_share_effect_absolute(gwl_dfs_by_share, shapefile_path, hatch_df,
                                  output_dir, tag, dpi=300):
    """Same 3x3 grid layout as plot_re_share_effect, but each panel plots the
    absolute change in cumulative residual load (GWL - GWL0.61) normalized by
    each region's non-thermosensitive baseline demand (_mmm_absolute_days),
    so the anomaly reads in "days of baseline demand" instead of percent --
    see plot_main_gwl_maps_absolute for the analogous main-figure twin."""
    TOT_RE_VALS   = [0.25, 0.5, 0.75]
    GWL_TITLES    = ["1.5°C", "2.0°C", "3.0°C"]
    PANEL_LETTERS = list("abcdefghi")

    abs_vals = []
    for tot_re in TOT_RE_VALS:
        for gwl_idx in [0, 1, 2]:
            eff = (_mmm_absolute_days(gwl_dfs_by_share[tot_re][gwl_idx], share_re="current")
                   ["Absolute_Days"].replace([np.inf, -np.inf], np.nan).dropna())
            if len(eff):
                abs_vals.append(np.abs(eff.values))
    vmax_days = (max(1.0, np.ceil(np.nanpercentile(np.concatenate(abs_vals), 95)))
                 if abs_vals else 1.0)
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mcolors.TwoSlopeNorm(vmin=-vmax_days, vcenter=0, vmax=vmax_days)

    fig_w = FIG_WIDTH_IN
    fig_h = fig_w * ((5.2 * 3 + 1.4) / (8.5 * 3)) * 1.35
    fig   = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs    = fig.add_gridspec(3, 3, hspace=0.30, wspace=0.04,
                             left=0.02, right=0.99, top=0.92, bottom=0.09)

    gdf_base = gpd.read_file(shapefile_path)
    gdf_base["poly_idx"] = gdf_base.index

    panel = 0
    for row, gwl_idx in enumerate([0, 1, 2]):
        for col, tot_re in enumerate(TOT_RE_VALS):
            ax     = fig.add_subplot(gs[row, col], projection=ccrs.EqualEarth())
            df_gwl = gwl_dfs_by_share[tot_re][gwl_idx]
            mmm    = _mmm_absolute_days(df_gwl, share_re="current")
            gdf    = gdf_base.copy().merge(mmm, on="poly_idx", how="left")
            gdf    = gdf.merge(hatch_df[["poly_idx", "var"]], on="poly_idx", how="left")
            gdf["do_hatch"] = gdf["var"].le(config.AGREEMENT_THRESHOLD).fillna(False) & config.SHOW_AGREEMENT_HATCHING
            _draw_map(ax, gdf, "Absolute_Days", cmap, norm, hatch_df,
                      title="", panel_letter="")
            ax.annotate(
                f"$\\mathbf{{{PANEL_LETTERS[panel]}}}$",
                xy=(0.02, 1.02), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=8,
                path_effects=[withStroke(linewidth=1.5, foreground="white")],
            )
            ax.set_title(
                f"GWL {GWL_TITLES[row]}\nSolar-wind penetration: {int(tot_re * 100)}%",
                fontsize=6, pad=4,
            )
            panel += 1

    cbar_ax = fig.add_axes([0.25, 0.045, 0.50, 0.018])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="both")
    cbar.set_label("Combined effect on SWBDs (days of baseline demand)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    fig.text(0.5, 0.995, "Effect of solar-wind penetration level on SWBDs",
             ha="center", va="top", fontsize=8, fontweight="bold")
    fig.text(0.5, 0.970, "Inverse-Wasserstein-weighted mean, current mix, threshold = 0.99",
             ha="center", va="top", fontsize=6, style="italic", color="#444444")

    legend_handles = [
        Patch(facecolor="white", edgecolor="black", hatch="\\" * 10, label="No solar-wind capacities"),
        discrepancy_mask_legend_handle("Low model agreement"),
    ]
    last_panel = fig.axes[8]
    last_panel.legend(handles=legend_handles, loc="upper center",
                      bbox_to_anchor=(0.5, -0.08), bbox_transform=last_panel.transAxes,
                      fontsize=5, framealpha=0.85, handlelength=1.0, handletextpad=0.4,
                      borderpad=0.4)
    _save_fig(fig, os.path.join(output_dir, "supp",
                                "suppfig9_re_share_effect_absolute_days.png"), dpi)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--output_dir", default="../final_figs")
    p.add_argument("--dpi",        type=int, default=300)
    return p.parse_args()


def _require(csv):
    if not os.path.exists(csv):
        raise SystemExit(f"{csv} missing -- run main_pipeline/make_rl_files.py first.")
    return csv


def _hatch_df():
    hatch_df, _ = load_hatch_agg(
        PATHS["agreement_nc"], PATHS["shapefile"],
        agreement_aggregated_nc=PATHS["agreement_aggregated_nc"],
    )
    return hatch_df


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "supp"), exist_ok=True)
    hatch_df = _hatch_df()
    gwl_dfs_by_share = {
        tot_re: load_gwl_dfs(_require(os.path.join(PATHS["out_dir"],
            f"rl_agg_adaptation_Annual_0.99_ren_pen_{tot_re}_current_v2.csv")))
        for tot_re in (0.25, 0.5, 0.75)
    }
    plot_re_share_effect_absolute(gwl_dfs_by_share, PATHS["shapefile"], hatch_df,
                                  args.output_dir, "re_share_effect", dpi=args.dpi)
    print("\nDone.")


if __name__ == "__main__":
    main()

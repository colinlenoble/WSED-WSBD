# -*- coding: utf-8 -*-
"""
Extended Data Fig. 8: SWBD changes at 2 °C split into supply and demand drivers.

Decomposition of SWBD changes at 2 °C warming into solar/wind supply and
temperature-driven demand contributions, in days of baseline demand
(previously fig45.py STEP 8). Reads the residual-load CSVs written by
main_pipeline/make_rl_files.py; map helpers come from main_figs/fig45.py.
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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import cartopy.crs as ccrs
from map_overlays import discrepancy_mask_legend_handle

from fig45 import (FIG_WIDTH_IN, MAIN_THR, MAIN_TOT_RE, PATHS, _add_colorbar,
                   _build_gdf, _draw_map, _mmm_demand_days, _mmm_supply_days,
                   _save_fig, load_gwl_dfs, load_hatch_agg, w2_weighted_mean)


# =============================================================================
# Extended Data Fig. 8 -- combined 2x2 driver effects at GWL 2.0°C
# =============================================================================

def _driver_ratio_dfs(df_gwl2, share_re):
    """Per-region supply share of (a) the absolute inverse-W2-weighted mean
    driver effect (w2_weighted_mean) and (b) the inter-model spread (std
    across GCMs' run-averaged effects, unweighted). 0 = demand dominates,
    1 = supply."""
    df = df_gwl2[df_gwl2["share_re"] == share_re].copy()
    df["RE_Effect"]   = df["cum_rl_ds_cf"] - df["cum_rl_ref"]
    df["Temp_Effect"] = df["cum_rl_gwl"] - df["cum_rl_ref"]
    per_gcm = (df[["poly_idx", "GCM", "RE_Effect", "Temp_Effect"]]
               .groupby(["GCM", "poly_idx"]).mean().reset_index())

    mean = (w2_weighted_mean(df, ["RE_Effect", "Temp_Effect"])
            .set_index("poly_idx").abs())
    df_change = (mean["RE_Effect"] / (mean["RE_Effect"] + mean["Temp_Effect"])
                 ).rename("ratio").reset_index()

    std = per_gcm.groupby("poly_idx")[["RE_Effect", "Temp_Effect"]].std()
    df_spread = (std["RE_Effect"] / (std["RE_Effect"] + std["Temp_Effect"])
                 ).rename("ratio").reset_index()
    return df_change, df_spread


def _driver_effects_figure(df_supply, supply_col, df_demand, demand_col,
                           effect_label, df_gwl2, shapefile_path, hatch_df,
                           share_re, out_path, dpi):
    """2x2 SWBD driver decomposition at GWL 2.0°C:
    a supply effect, b demand effect (one shared symmetric colorbar),
    c dominant driver of the change, d dominant driver of the inter-model
    spread (one shared 0-1 colorbar). Axes are placed at fixed inch positions
    on a FIG_WIDTH_IN-wide canvas so panel letters print at the same size as
    in the other supplementary figures."""
    fig_w, fig_h = FIG_WIDTH_IN, 4.05
    fig  = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    proj = ccrs.EqualEarth()

    def _rect(x_in, y_in, w_in, h_in):
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    map_w, map_h = 2.50, 1.22     # EqualEarth global aspect ~2.05:1
    x_left, x_right = 0.04, fig_w - 0.04 - map_w
    y_top_row, y_bot_row = 2.42, 0.46
    ax_a = fig.add_axes(_rect(x_left,  y_top_row, map_w, map_h), projection=proj)
    ax_b = fig.add_axes(_rect(x_right, y_top_row, map_w, map_h), projection=proj)
    ax_c = fig.add_axes(_rect(x_left,  y_bot_row, map_w, map_h), projection=proj)
    ax_d = fig.add_axes(_rect(x_right, y_bot_row, map_w, map_h), projection=proj)

    # a/b: shared symmetric scale from the 95th percentile of |effect| over both panels
    abs_vals = np.abs(np.concatenate([
        df_supply[supply_col].replace([np.inf, -np.inf], np.nan).dropna().values,
        df_demand[demand_col].replace([np.inf, -np.inf], np.nan).dropna().values,
    ]))
    vmax_eff = max(1.0, np.ceil(np.nanpercentile(abs_vals, 95))) if len(abs_vals) else 1.0
    cmap_eff = plt.get_cmap("RdYlGn_r")
    norm_eff = mcolors.TwoSlopeNorm(vmin=-vmax_eff, vcenter=0, vmax=vmax_eff)
    _draw_map(ax_a, _build_gdf(shapefile_path, df_supply, hatch_df), supply_col,
              cmap_eff, norm_eff, hatch_df, "Supply effect", "a", title_fontsize=6)
    _draw_map(ax_b, _build_gdf(shapefile_path, df_demand, hatch_df), demand_col,
              cmap_eff, norm_eff, hatch_df, "Demand effect", "b", title_fontsize=6)
    cb_eff = _add_colorbar(fig, cmap_eff, norm_eff, effect_label,
                           pos=_rect(fig_w * 0.25, 2.24, fig_w * 0.5, 0.06), extend="both")
    cb_eff.outline.set_linewidth(0.4)

    # c/d: shared 0-1 ratio scale
    df_change, df_spread = _driver_ratio_dfs(df_gwl2, share_re)
    cmap_ratio = plt.get_cmap("PiYG_r")
    norm_ratio = mcolors.Normalize(vmin=0, vmax=1)
    _draw_map(ax_c, _build_gdf(shapefile_path, df_change, hatch_df), "ratio",
              cmap_ratio, norm_ratio, hatch_df,
              "Dominant driver of the change", "c", title_fontsize=6)
    _draw_map(ax_d, _build_gdf(shapefile_path, df_spread, hatch_df), "ratio",
              cmap_ratio, norm_ratio, hatch_df,
              "Dominant driver of the uncertainty", "d", title_fontsize=6)
    cb_ratio = _add_colorbar(fig, cmap_ratio, norm_ratio, "",
                             pos=_rect(fig_w * 0.25, 0.32, fig_w * 0.5, 0.06))
    cb_ratio.set_ticks([0, 0.5, 1.0])
    cb_ratio.set_ticklabels(["Demand dominates", "Equal", "Supply dominates"])
    cb_ratio.outline.set_linewidth(0.4)

    fig.text(0.5, 4.00 / fig_h, "SWBD driver decomposition under 2°C",
             ha="center", va="top", fontsize=8, fontweight="bold")
    fig.legend(handles=[
        Patch(facecolor="white", edgecolor="black", hatch="\\" * 10, label="No RE capacity"),
        discrepancy_mask_legend_handle("Low model agreement"),
    ], ncol=2, loc="center", bbox_to_anchor=(0.5, 0.09 / fig_h), bbox_transform=fig.transFigure,
       fontsize=5, framealpha=0.85, handlelength=1.0, handletextpad=0.4, borderpad=0.4)

    _save_fig(fig, out_path, dpi)


def plot_supp_combined_driver_effects_absolute(df_gwl2, shapefile_path, hatch_df,
                                               output_dir, share_re="current", dpi=300):
    """Same layout as plot_supp_combined_driver_effects, but panels a/b plot
    the absolute change in cumulative residual load normalized by each
    region's baseline demand (_mmm_supply_days/_mmm_demand_days), so they read
    in "days of baseline demand". Panels c/d are unitless ratios either way."""
    _driver_effects_figure(
        _mmm_supply_days(df_gwl2, share_re), "Supply_Days",
        _mmm_demand_days(df_gwl2, share_re), "Demand_Days",
        "Effect on SWBDs (days of baseline demand)",
        df_gwl2, shapefile_path, hatch_df, share_re,
        os.path.join(output_dir, "supp",
                     "suppfig8_combined_driver_effects_absolute_days.png"), dpi)


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
    csv = _require(os.path.join(PATHS["out_dir"],
        f"rl_agg_adaptation_Annual_{MAIN_THR}_ren_pen_{MAIN_TOT_RE}_current_v2.csv"))
    _, df_gwl2, _ = load_gwl_dfs(csv)
    plot_supp_combined_driver_effects_absolute(df_gwl2, PATHS["shapefile"], hatch_df,
                                               args.output_dir, share_re="current", dpi=args.dpi)
    print("\nDone.")


if __name__ == "__main__":
    main()

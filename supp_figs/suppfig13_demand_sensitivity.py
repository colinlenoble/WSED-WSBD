# -*- coding: utf-8 -*-
"""
Extended Data Fig. 13: sensitivity of SWBD changes to demand-model parameters.

One panel per DemandConfig (make_rl_files.DEMAND_CONFIGS), in days of baseline
demand (previously fig45.py STEP 5). Reads the demand-sensitivity CSVs written
by main_pipeline/make_rl_files.py; map helpers come from main_figs/fig45.py.
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

from make_rl_files import DEMAND_CONFIGS
from fig45 import (FIG_WIDTH_IN, MAIN_MIX, MAIN_THR,
                   MAIN_TOT_RE, PATHS, _build_gdf, _draw_map,
                   _mmm_absolute_days, _save_fig, load_gwl_dfs,
                   load_hatch_agg)


# =============================================================================
# Extended Data Fig. 13 -- demand-sensitivity grid, absolute (days of baseline demand)
# =============================================================================

def plot_supp_demand_sensitivity_absolute(shapefile_path, hatch_df, output_dir,
                                          agg_datasets_dir, dpi=300, period="Annual"):
    """Same layout as plot_supp_demand_sensitivity, but each panel plots the
    absolute change in cumulative residual load (GWL2 - GWL0.61) normalized
    by the region's non-thermosensitive baseline demand, so the anomaly reads
    in units of "days of baseline demand" instead of percent."""
    REF_COLOR = "#c0392b"
    ALT_COLOR = "#2c3e50"
    _meta = {
        "default"  : ("Reference",            "Tc=12.5°C  ·  Th=19.6°C  ·  a=0.026/0.035"),
        "cold_low" : ("Cold threshold -2°C",  "Tc=10.5°C"),
        "cold_high": ("Cold threshold +2°C",  "Tc=14.5°C"),
        "hot_low"  : ("Hot threshold -2°C",   "Th=17.6°C"),
        "hot_high" : ("Hot threshold +2°C",   "Th=21.6°C"),
        "coef_low" : ("Coefficients -20%",    "ac=0.021  ·  ah=0.028"),
        "coef_high": ("Coefficients +20%",    "ac=0.031  ·  ah=0.042"),
        "strict"   : ("Wide comfort zone",    "Tc=10.5°C  ·  Th=21.6°C  ·  a=0.021/0.028"),
        "sensitive": ("Narrow comfort zone",  "Tc=14.5°C  ·  Th=17.6°C  ·  a=0.031/0.042"),
    }
    names = list(DEMAND_CONFIGS.keys())
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))

    cmap_ref = plt.get_cmap("RdYlGn_r")
    cmap_diff = plt.get_cmap("PuOr_r")

    effect_by_name = {}
    for demand_name in names:
        csv = os.path.join(
            agg_datasets_dir,
            (f"rl_agg_adaptation_{period}_{MAIN_THR}"
             f"_ren_pen_{MAIN_TOT_RE}_{MAIN_MIX}"
             f"_demand-{demand_name}_v2.csv"),
        )
        if not os.path.exists(csv):
            effect_by_name[demand_name] = None
            continue
        _, df_gwl2, _ = load_gwl_dfs(csv)
        effect_by_name[demand_name] = (
            _mmm_absolute_days(df_gwl2, MAIN_MIX).set_index("poly_idx")["Absolute_Days"])

    default_eff = effect_by_name.get("default")
    diff_by_name = {}
    max_abs_ref  = 0.0
    max_abs_diff = 0.0
    if default_eff is not None:
        finite_ref = default_eff.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite_ref):
            max_abs_ref = np.nanpercentile(np.abs(finite_ref.values), 95)
    for demand_name in names:
        if demand_name == "default" or default_eff is None or effect_by_name[demand_name] is None:
            continue
        diff = (effect_by_name[demand_name] - default_eff).replace([np.inf, -np.inf], np.nan).dropna()
        diff_by_name[demand_name] = diff
        if len(diff):
            max_abs_diff = max(max_abs_diff, np.nanpercentile(np.abs(diff.values), 95))

    vmax_ref  = max(1.0, np.ceil(max_abs_ref))
    vmax_diff = max(1.0, np.ceil(max_abs_diff))
    norm_ref  = mcolors.TwoSlopeNorm(vmin=-vmax_ref, vcenter=0, vmax=vmax_ref)
    norm_diff = mcolors.TwoSlopeNorm(vmin=-vmax_diff, vcenter=0, vmax=vmax_diff)

    fig_w = FIG_WIDTH_IN
    fig_h = fig_w * ((5.5 * nrows + 1.2) / (8 * 3)) * 1.05
    fig   = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs    = fig.add_gridspec(nrows, ncols, hspace=0.20, wspace=0.04,
                             left=0.01, right=0.99, top=0.90, bottom=0.07)

    ref_ax  = None
    last_ax = None
    for idx, demand_name in enumerate(names):
        row, col   = divmod(idx, ncols)
        ax         = fig.add_subplot(gs[row, col], projection=ccrs.EqualEarth())
        last_ax    = ax
        is_ref     = (demand_name == "default")
        label, params = _meta.get(demand_name, (demand_name, ""))
        letter     = f"{chr(97 + idx)}."
        title_color = REF_COLOR if is_ref else ALT_COLOR

        if is_ref:
            ref_ax = ax

        eff = effect_by_name.get(demand_name)
        missing = (eff is None) or (not is_ref and demand_name not in diff_by_name)
        if missing:
            ax.text(0.5, 0.5, "[CSV missing]", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6, color="gray")
            ax.annotate(
                f"$\\mathbf{{{letter}}}$",
                xy=(0.02, 1.10), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=8, color=title_color,
                clip_on=False,
            )
            ax.text(0.5, 1.10, label, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=5.5, color=title_color,
                    clip_on=False)
            ax.set_axis_off()
            continue

        if is_ref:
            value_col, panel_cmap, panel_norm = "Absolute_Days", cmap_ref, norm_ref
            df_plot = eff.reset_index()
        else:
            value_col, panel_cmap, panel_norm = "Diff_Days", cmap_diff, norm_diff
            df_plot = diff_by_name[demand_name].reset_index(name="Diff_Days")
        gdf = _build_gdf(shapefile_path, df_plot, hatch_df)
        _draw_map(ax, gdf, value_col, panel_cmap, panel_norm, hatch_df,
                  title="", panel_letter="")
        ax.annotate(
            f"$\\mathbf{{{letter}}}$",
            xy=(0.02, 1.10), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=8, color=title_color,
            clip_on=False,
        )
        ax.text(0.5, 1.10, label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=5.5, color=title_color,
                clip_on=False)
        ax.text(0.5, 1.02, params, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=5,
                color="#444444", style="italic")

        if is_ref:
            try:
                ax.spines["geo"].set_edgecolor(REF_COLOR)
                ax.spines["geo"].set_linewidth(2.0)
            except (KeyError, AttributeError):
                try:
                    ax.outline_patch.set_edgecolor(REF_COLOR)
                    ax.outline_patch.set_linewidth(2.0)
                except AttributeError:
                    pass

    legend_ax = ref_ax if ref_ax is not None else last_ax
    if legend_ax is not None:
        legend_ax.legend(handles=[
            Patch(facecolor="white", edgecolor="black", hatch="\\" * 10,
                  label="No RE capacities"),
            discrepancy_mask_legend_handle("Low model agreement"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.08),
           bbox_transform=legend_ax.transAxes,
           fontsize=5, framealpha=0.85, handlelength=1.5,
           handletextpad=0.4, borderpad=0.4, ncol=2)

    for extra in range(len(names), nrows * ncols):
        row, col = divmod(extra, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    cbar_ref_ax = fig.add_axes([0.09, 0.025, 0.36, 0.020])
    sm_ref = plt.cm.ScalarMappable(cmap=cmap_ref, norm=norm_ref)
    sm_ref.set_array([])
    cb_ref = fig.colorbar(sm_ref, cax=cbar_ref_ax, orientation="horizontal", extend="both")
    cb_ref.set_label("Change in SWBDs (days of baseline demand), default", fontsize=6)
    cb_ref.ax.tick_params(labelsize=5)

    cbar_diff_ax = fig.add_axes([0.55, 0.025, 0.36, 0.020])
    sm_diff = plt.cm.ScalarMappable(cmap=cmap_diff, norm=norm_diff)
    sm_diff.set_array([])
    cb_diff = fig.colorbar(sm_diff, cax=cbar_diff_ax, orientation="horizontal", extend="both")
    cb_diff.set_label("Change vs default (days of baseline demand)", fontsize=6)
    cb_diff.ax.tick_params(labelsize=5)

    fig.text(0.5, 0.962, "Sensitivity to demand-model parameters (absolute)",
             ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.text(0.5, 0.933,
             f"thr = {MAIN_THR}  |  tot_re = {MAIN_TOT_RE}  |  mix = {MAIN_MIX}",
             ha="center", va="bottom", fontsize=6, color="#555555")

    _save_fig(fig, os.path.join(output_dir, "supp",
                                "suppfig13_demand_sensitivity_absolute_days.png"), dpi)


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
    plot_supp_demand_sensitivity_absolute(
        shapefile_path=PATHS["shapefile"], hatch_df=hatch_df,
        output_dir=args.output_dir, agg_datasets_dir=PATHS["out_dir"], dpi=args.dpi)
    print("\nDone.")


if __name__ == "__main__":
    main()

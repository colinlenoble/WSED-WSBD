# -*- coding: utf-8 -*-
"""
Non-paper variants of the fig45.py figures.

Relative-change (%) twins of Figs. 4-5 and Extended Data Figs. 8/9/13, the
inverse-Wasserstein-weighted companions of Figs. 4-5, and older supplementary
panels (GWL maps, isolated supply/demand effects, decomposition, inter-model
spread) that fig45.py had already stopped calling. Reads the residual-load
CSVs written by main_pipeline/make_rl_files.py.
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
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cmo
from map_overlays import draw_discrepancy_mask_polygons, discrepancy_mask_legend_handle

from make_rl_files import DEMAND_CONFIGS
from fig45 import (DICT_LABELS, FIG_WIDTH_IN, MAIN_MIX,
                   MAIN_THR, MAIN_TOT_RE, MAP_LAT_NORTH, MAP_LAT_SOUTH,
                   PATHS, REGION_NAMES, _add_colorbar, _build_gdf, _draw_map,
                   _make_cmap, _mmm_combined, _mmm_re, _mmm_tas, _save_fig,
                   _three_panel_map, load_gwl_dfs, load_hatch_agg,
                   mask_poles, _load_wasserstein_agg,
                   _wasserstein_weight_table, w2_weighted_mean)
from suppfig8_driver_effects import _driver_effects_figure


def _print_combined_effect_direction_stats(df_gwl, gwl_label, share_re="current"):
    """Diagnostic across all polygons at a given GWL: how many regions have a
    positive (increasing) vs. negative (decreasing) flat multi-model-mean
    Combined_Effect -- SWBDs change relative to 0.61 C, same metric/weighting
    as plot_main_gwl_maps -- out of how many regions have a finite value."""
    combined = (_mmm_combined(df_gwl, share_re, vmax=None,
                              weighting="mmm")["Combined_Effect"]
                .replace([np.inf, -np.inf], np.nan).dropna())
    n_total = len(combined)
    n_up    = int((combined > 0).sum())
    n_down  = int((combined < 0).sum())
    print(f"  [INFO] {gwl_label}: multi-model mean SWBDs increasing in "
          f"{n_up}/{n_total} regions, decreasing in {n_down}/{n_total}")


# =============================================================================
# INVERSE-WASSERSTEIN-DISTANCE POLYGON WEIGHTING
#
# Loader/weight table/weighted mean live in fig45.py (_load_wasserstein_agg,
# _wasserstein_weight_table, w2_weighted_mean), where they are also the
# default ensemble averaging of _mmm(). The wrappers below take an explicit
# w2_table, for the companion figures that compare against the flat
# multi-model mean (_mmm(..., weighting="mmm")).
# =============================================================================

def _mmm_wasserstein(df_gwl, effect_col, share_re, vmax, compute_fn, w2_table):
    """Inverse-W2-weighted per-polygon mean of effect_col using w2_table
    (see fig45.w2_weighted_mean)."""
    df = df_gwl[df_gwl["share_re"] == share_re].copy()
    compute_fn(df)
    out = w2_weighted_mean(df, [effect_col], w2_table=w2_table)
    if vmax is not None:
        out.loc[out[effect_col] > vmax, effect_col] = vmax
    return out


def _mmm_absolute_days_wasserstein(df_gwl, w2_table, share_re="current"):
    """Inverse-W2-weighted twin of _mmm_absolute_days()."""
    def fn(df):
        df["Absolute_Days"] = (df["cum_rl_gwl"] - df["cum_rl_ref"]) / df["demand_bas"]
    return _mmm_wasserstein(df_gwl, "Absolute_Days", share_re, vmax=None,
                            compute_fn=fn, w2_table=w2_table)


def _mmm_combined_wasserstein(df_gwl, w2_table, share_re="current", vmax=None):
    """Inverse-W2-weighted twin of _mmm_combined() -- same relative-change (%)
    metric plotted by plot_main_gwl_maps, but averaged per polygon with
    _wasserstein_weight_table's weights instead of a flat multi-model mean."""
    def fn(df):
        df["Combined_Effect"] = (df["cum_rl_gwl"] - df["cum_rl_ref"]) / df["cum_rl_ref"] * 100
    return _mmm_wasserstein(df_gwl, "Combined_Effect", share_re, vmax,
                            compute_fn=fn, w2_table=w2_table)


# =============================================================================
# Variants of Figs. 4-5: relative change (%) and Wasserstein-weighted
# =============================================================================

def plot_main_gwl_maps(df_gwl15, df_gwl2, df_gwl3,
                       shapefile_path, hatch_df, output_dir,
                       dpi=300, share_re="current"):
    cmap, norm = _make_cmap(vmin=-100, vmax=800)
    for df_gwl, gwl_label in ((df_gwl15, "1.5°C"), (df_gwl2, "2°C"), (df_gwl3, "3°C")):
        _print_combined_effect_direction_stats(df_gwl, gwl_label, share_re)
    fig = _three_panel_map(
        df_gwl15, df_gwl2, df_gwl3, shapefile_path, hatch_df, cmap, norm,
        value_fn=_mmm_combined, value_col="Combined_Effect",
        title_gwl2="2°C",
        title_gwl15="1.5°C",
        title_gwl3="3°C",
        cbar_label="SWBDs change compared to 0.61°C (%)", dpi=dpi,
    )
    _save_fig(fig, os.path.join(output_dir, "main", "fig_main_gwl_maps.png"), dpi)


def plot_main_gwl_maps_absolute_wasserstein(df_gwl15, df_gwl2, df_gwl3,
                                            shapefile_path, hatch_df, w2_table,
                                            output_dir, dpi=300, share_re="current"):
    """Companion to plot_main_gwl_maps_absolute: each polygon's value is the
    inverse-Wasserstein-weighted average of the absolute change in cumulative
    residual load (GWL - GWL0.61) instead of the flat multi-model mean --
    see _mmm_absolute_days_wasserstein/_wasserstein_weight_table. A
    realization whose bootstrap trend distribution is closer to ERA5's own at
    a given polygon counts for more there."""
    def value_fn(df_gwl, share_re_):
        return _mmm_absolute_days_wasserstein(df_gwl, w2_table, share_re_)

    abs_vals = []
    for df_gwl in (df_gwl15, df_gwl2, df_gwl3):
        eff = (value_fn(df_gwl, share_re)["Absolute_Days"]
               .replace([np.inf, -np.inf], np.nan).dropna())
        if len(eff):
            abs_vals.append(np.abs(eff.values))
    vmax_days = (max(1.0, np.ceil(np.nanpercentile(np.concatenate(abs_vals), 95)))
                 if abs_vals else 1.0)
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mcolors.TwoSlopeNorm(vmin=-vmax_days, vcenter=0, vmax=vmax_days)
    fig = _three_panel_map(
        df_gwl15, df_gwl2, df_gwl3, shapefile_path, hatch_df, cmap, norm,
        value_fn=value_fn, value_col="Absolute_Days",
        title_gwl2="2°C",
        title_gwl15="1.5°C",
        title_gwl3="3°C",
        cbar_label="SWBDs change vs 0.61°C, inverse-W2 weighted\n(days of baseline demand)",
        dpi=dpi,
    )
    _save_fig(fig, os.path.join(output_dir, "main",
                                "fig_main_gwl_maps_absolute_days_wasserstein.png"), dpi)


def plot_gwl2_wasserstein_vs_mmm(df_gwl2, shapefile_path, hatch_df, w2_table,
                                 output_dir, dpi=300, share_re="current"):
    """
    Three-panel supplementary companion to the GWL maps, at GWL2 (2 deg C)
    only -- aggregated (poly_idx) twin of
    suppfig5_projected_change_wasserstein.py, using the SWBDs relative-change
    (%) metric (Combined_Effect):
      a) average SWBDs change vs 0.61 deg C, weighted per polygon by each
         realization's 1/n_gcm times its inverse normalized Wasserstein trend
         distance to ERA5 at that polygon (_mmm_combined_wasserstein),
      b) the flat multi-model mean (_mmm_combined(..., weighting="mmm")),
         side by side with a on one shared continuous colour scale,
      c) a - b in percentage points, full width, diverging (cmo.cm.balance)
         on its own scale. Agreement masking is shown in a/b but omitted in
         c, since it describes trend agreement, not this reweighting.
    """
    df_flat = _mmm_combined(df_gwl2, share_re=share_re, vmax=None, weighting="mmm")
    df_w2   = _mmm_combined_wasserstein(df_gwl2, w2_table, share_re=share_re, vmax=None)

    # a/b: shared continuous scale from the data (2nd-98th percentile of
    # both fields, centred on 0) rather than the fixed -100..800 % of the
    # GWL maps, so the a-b differences stay visible.
    ab = (pd.concat([df_flat["Combined_Effect"], df_w2["Combined_Effect"]])
          .replace([np.inf, -np.inf], np.nan).dropna())
    lo = min(-1.0, float(np.nanpercentile(ab, 2))) if len(ab) else -1.0
    hi = max(1.0, float(np.nanpercentile(ab, 98))) if len(ab) else 1.0
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mcolors.TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)

    diff = df_flat.merge(df_w2, on="poly_idx", suffixes=("_flat", "_w2"))
    diff["Diff_Effect"] = diff["Combined_Effect_w2"] - diff["Combined_Effect_flat"]
    diff = diff[["poly_idx", "Diff_Effect"]]

    finite = diff["Diff_Effect"].replace([np.inf, -np.inf], np.nan).dropna()
    diff_vmax = max(1.0, float(np.nanpercentile(np.abs(finite), 98))) if len(finite) else 1.0
    diff_cmap = cmo.cm.balance
    diff_norm = mcolors.TwoSlopeNorm(vmin=-diff_vmax, vcenter=0, vmax=diff_vmax)

    hatch_df_none = hatch_df.copy()
    hatch_df_none["var"] = np.nan

    # Fixed inch layout: row 1 two half-width maps (a | b) + shared
    # colorbar, row 2 one full-width map (c) + its colorbar.
    aspect = 2.05   # width / height of a set_global() EqualEarth axes
    fig_w  = FIG_WIDTH_IN
    margin, hgap = 0.05, 0.10
    half_w = (fig_w - 2 * margin - hgap) / 2
    half_h = half_w / aspect
    full_w = fig_w - 2 * margin
    full_h = full_w / aspect
    title_h, cbar_zone, row_gap, suptitle_h = 0.24, 0.50, 0.10, 0.20
    fig_h  = (suptitle_h + title_h + half_h + cbar_zone + row_gap + title_h
              + full_h + cbar_zone)
    fig    = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    def _rect(x_in, y_in, w_in, h_in):
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    y_c   = cbar_zone
    y_top = y_c + full_h + title_h + row_gap + cbar_zone
    proj  = ccrs.EqualEarth()
    ax_a = fig.add_axes(_rect(margin, y_top, half_w, half_h), projection=proj)
    ax_b = fig.add_axes(_rect(margin + half_w + hgap, y_top, half_w, half_h), projection=proj)
    ax_c = fig.add_axes(_rect(margin, y_c, full_w, full_h), projection=proj)

    _draw_map(ax_a, _build_gdf(shapefile_path, df_w2, hatch_df), "Combined_Effect",
              cmap, norm, hatch_df, "Inverse-Wasserstein weighted", "a", title_fontsize=6)
    _draw_map(ax_b, _build_gdf(shapefile_path, df_flat, hatch_df), "Combined_Effect",
              cmap, norm, hatch_df, "Multi-model mean", "b", title_fontsize=6)
    _draw_map(ax_c, _build_gdf(shapefile_path, diff, hatch_df_none), "Diff_Effect",
              diff_cmap, diff_norm, hatch_df_none, "Difference (a $-$ b)", "c",
              title_fontsize=7)
    fig.suptitle("SWBDs change under 2.0°C warming", fontsize=7,
                 y=1 - 0.04 / fig_h, va="top")

    cbar_w = 0.5 * full_w
    _add_colorbar(fig, cmap, norm, "SWBDs change compared to 0.61°C (%)",
                  pos=_rect((fig_w - cbar_w) / 2, y_top - 0.20, cbar_w, 0.07),
                  extend="both")
    _add_colorbar(fig, diff_cmap, diff_norm,
                  "Difference, inverse-W2 weighted\nminus multi-model mean (pp)",
                  pos=_rect((fig_w - cbar_w) / 2, y_c - 0.20, cbar_w, 0.07),
                  extend="both")

    _save_fig(fig, os.path.join(output_dir, "supp",
                                "suppfig_main_gwl_maps_GWL2_wasserstein_vs_mmm.png"), dpi)


def _print_effect_outliers(df_db, region_name, effect_col="RE_Effect", n_top=5):
    """Diagnostic: print the GCM/run pairs with the most extreme effect_col
    values for a single named region -- e.g. to identify which realizations
    are driving an apparently multi-modal (two-group) spread of per-(GCM,run)
    points far from 0 in the dumbbell's violin/strip plot."""
    sub = df_db.loc[df_db["name"] == region_name, ["GCM", "run", effect_col]].dropna()
    if sub.empty:
        print(f"  [INFO] No rows for region '{region_name}' to check {effect_col} outliers.")
        return
    sub = sub.sort_values(effect_col)
    print(f"\n  {effect_col} outliers for {region_name} ({len(sub)} GCM-run rows):")
    print("    Most negative:")
    for _, r in sub.head(n_top).iterrows():
        print(f"      {r['GCM']:<25} {r['run']:<10} {r[effect_col]:8.1f}%")
    print("    Most positive:")
    for _, r in sub.tail(n_top).iterrows():
        print(f"      {r['GCM']:<25} {r['run']:<10} {r[effect_col]:8.1f}%")


def plot_main_dumbbell(df_gwl2, shapefile_path, dpi=300, share_re="current",
                       output_dir=None):
    shp      = gpd.read_file(shapefile_path)
    name_col = "name" if "name" in shp.columns else shp.columns[1]
    df_db    = df_gwl2[df_gwl2["share_re"] == share_re].copy()
    df_db["name"] = df_db["poly_idx"].map(shp[name_col].to_dict())
    for eff, num in [("Combined_Effect", "cum_rl_gwl"),
                     ("Temp_Effect",     "cum_rl_tas"),
                     ("RE_Effect",       "cum_rl_ds_cf")]:
        df_db[eff] = (df_db[num] - df_db["cum_rl_ref"]) / df_db["cum_rl_ref"] * 100
    _print_effect_outliers(df_db, "Sichuan", "RE_Effect")
    df_db["label"] = df_db["name"].map(DICT_LABELS)
    df_db = df_db[df_db["name"].isin(REGION_NAMES)].dropna(subset=["label"])
    stats = (df_db[["label", "GCM", "Combined_Effect", "Temp_Effect", "RE_Effect"]]
             .groupby(["label", "GCM"]).mean().groupby("label").mean())
    order = stats["Combined_Effect"].sort_values(ascending=False).index.tolist()
    stats = stats.loc[order] if order else stats
    df_long = (df_db[["label", "Temp_Effect", "RE_Effect"]]
               .melt(id_vars="label", var_name="Effect", value_name="Value")
               .dropna())
    df_long["Effect"] = df_long["Effect"].map(
        {"Temp_Effect": "Demand driver", "RE_Effect": "Supply driver"})
    pal = plt.get_cmap("PuOr")
    demand_color, re_color = pal(0.8), pal(0.2)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Broken-axis layout -- width = 2x FIG_WIDTH_IN
    fig_w = FIG_WIDTH_IN
    fig_h = fig_w * (16 / 14)
    fig   = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs    = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.04, figure=fig)
    ax_db  = fig.add_subplot(gs[0])
    ax_log = fig.add_subplot(gs[1], sharey=ax_db)

    def _draw_on(a, xlim, xscale="linear"):
        if len(df_long) > 0 and len(order) > 0:
            try:
                sns.violinplot(
                    data=df_long, x="Value", y="label", hue="Effect",
                    order=order, split=True, inner=None, width=1.8,
                    palette={"Demand driver": demand_color, "Supply driver": re_color},
                    ax=a,
                )
                sns.stripplot(
                    data=df_long, x="Value", y="label", hue="Effect",
                    order=order, dodge=True, size=2.5, alpha=0.55,
                    palette={"Demand driver": demand_color, "Supply driver": re_color},
                    ax=a,
                )
                for coll in a.collections:
                    if hasattr(coll, "get_alpha") and (coll.get_alpha() is None
                                                        or coll.get_alpha() > 0.35):
                        coll.set_alpha(0.35)
                if a.get_legend():
                    a.get_legend().remove()
                for i, sub in enumerate(order):
                    if sub not in stats.index:
                        continue
                    r = stats.loc[sub]
                    a.scatter(r["Temp_Effect"],     i, color=demand_color, s=30, zorder=5, alpha=0.9)
                    a.scatter(r["RE_Effect"],       i, color=re_color,     s=30, zorder=5, alpha=0.9)
                    a.scatter(r["Combined_Effect"], i, color="black",      s=30, zorder=6, alpha=0.9)
            except Exception as exc:
                print(f"  [WARN] Dumbbell failed: {exc}")
        a.axvline(0, color="black", lw=1.2, alpha=0.6, linestyle="--")
        a.set_xlim(xlim)
        if xscale == "log":
            a.set_xscale("symlog", linthresh=900)
        if not (xlim[0] <= 0 <= xlim[1]):
            a.tick_params(axis="y", labelleft=False)
            a.set_ylabel("")

    _draw_on(ax_db,  (-150, 800))
    _draw_on(ax_log, (800, 1500), xscale="log")

    # Break marks
    d   = 0.015
    kw  = dict(transform=ax_db.transAxes,  color="#aaaaaa", clip_on=False, lw=0.8)
    kw2 = dict(transform=ax_log.transAxes, color="#aaaaaa", clip_on=False, lw=0.8)
    ax_db.plot( (1 - d, 1 + d), (-d, +d),       **kw)
    ax_db.plot( (1 - d, 1 + d), (1 - d, 1 + d), **kw)
    ax_log.plot((-d, +d),       (-d, +d),       **kw2)
    ax_log.plot((-d, +d),       (1 - d, 1 + d), **kw2)

    ax_db.spines["right"].set_visible(False)
    ax_log.spines["left"].set_visible(False)
    ax_log.tick_params(axis="y", left=False)

    ax_db.set_xticks([-100, 0, 100, 200, 400, 600, 800])
    ax_db.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax_db.tick_params(axis="x", labelsize=5)
    ax_log.set_xticks([900, 1200, 1500])
    ax_log.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax_log.tick_params(axis="x", labelsize=5)

    ax_db.set_yticks(range(len(order)))
    ax_db.set_yticklabels(order, fontsize=6)
    ax_log.tick_params(axis="y", labelleft=False)
    ax_db.set_ylabel("")
    ax_db.set_xlabel("Relative change (%)", fontsize=6, labelpad=2)
    ax_log.set_xlabel("")

    # Direction arrows
    ax_db.annotate("", xy=(-130, 1.04), xycoords=("data", "axes fraction"),
                   xytext=(0, 1.04), textcoords=("data", "axes fraction"),
                   arrowprops=dict(arrowstyle="->", lw=0.8, color="#777777"))
    ax_db.text(-75, 1.055, "Lower SWBDs",
               transform=ax_db.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=6, color="black")
    ax_db.annotate("", xy=(700, 1.04), xycoords=("data", "axes fraction"),
                   xytext=(30, 1.04), textcoords=("data", "axes fraction"),
                   arrowprops=dict(arrowstyle="->", lw=0.8, color="#777777"))
    ax_db.text(370, 1.055, "Higher SWBDs",
               transform=ax_db.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=6, color="black")

    # Legend
    ax_log.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="None",
               color=demand_color, label="Demand driver", markersize=5),
        Line2D([0], [0], marker="o", linestyle="None",
               color=re_color, label="Supply driver", markersize=5),
        Line2D([0], [0], marker="o", linestyle="None",
               color="black", label="Combined effect", markersize=5),
    ], title="Multi-model mean effect",
       title_fontproperties={"weight": "bold", "size": 6},
       loc="lower right", fontsize=5)

    _save_fig(fig, os.path.join(output_dir, "main", "fig_main_dumbbell.png"), dpi)


def plot_main_dumbbell_absolute_wasserstein(df_gwl2, shapefile_path, w2_table,
                                            dpi=300, share_re="current",
                                            output_dir=None):
    """Companion to plot_main_dumbbell_absolute: each region's Combined_Effect
    is shown as two dots instead of one -- the original flat multi-model-mean
    value (black circle, as in plot_main_dumbbell_absolute), plus a second
    value (teal diamond) obtained by averaging the same per-(GCM, run)
    Combined_Effect with the inverse-Wasserstein-distance polygon weight from
    _wasserstein_weight_table/_mmm_wasserstein instead of a flat per-GCM mean
    (see trend_sev_eval_wasserstein.wasserstein_empirical_agg()) -- a
    realization whose bootstrap trend distribution is closer to ERA5's own at
    that region's polygon counts for more there. Demand/RE driver violins are
    unchanged (still flat multi-model mean)."""
    shp      = gpd.read_file(shapefile_path)
    name_col = "name" if "name" in shp.columns else shp.columns[1]
    df_db    = df_gwl2[df_gwl2["share_re"] == share_re].copy()
    df_db["name"] = df_db["poly_idx"].map(shp[name_col].to_dict())
    for eff, num in [("Combined_Effect", "cum_rl_gwl"),
                     ("Temp_Effect",     "cum_rl_tas"),
                     ("RE_Effect",       "cum_rl_ds_cf")]:
        df_db[eff] = (df_db[num] - df_db["cum_rl_ref"]) / df_db["demand_bas"]
    df_db["label"] = df_db["name"].map(DICT_LABELS)
    df_db = df_db[df_db["name"].isin(REGION_NAMES)].dropna(subset=["label"])
    stats = (df_db[["label", "GCM", "Combined_Effect", "Temp_Effect", "RE_Effect"]]
             .groupby(["label", "GCM"]).mean().groupby("label").mean())
    order = stats["Combined_Effect"].sort_values(ascending=False).index.tolist()
    stats = stats.loc[order] if order else stats

    def fn(df):
        df["Combined_Effect"] = (df["cum_rl_gwl"] - df["cum_rl_ref"]) / df["demand_bas"]
    poly_w2 = _mmm_wasserstein(df_gwl2, "Combined_Effect", share_re, vmax=None,
                               compute_fn=fn, w2_table=w2_table)
    poly_w2["name"]  = poly_w2["poly_idx"].map(shp[name_col].to_dict())
    poly_w2["label"] = poly_w2["name"].map(DICT_LABELS)
    w2_by_label = (poly_w2.dropna(subset=["label"])
                          .set_index("label")["Combined_Effect"])

    df_long = (df_db[["label", "Temp_Effect", "RE_Effect"]]
               .melt(id_vars="label", var_name="Effect", value_name="Value")
               .dropna())
    df_long["Effect"] = df_long["Effect"].map(
        {"Temp_Effect": "Demand driver", "RE_Effect": "Supply driver"})
    pal = plt.get_cmap("PuOr")
    demand_color, re_color = pal(0.8), pal(0.2)
    w2_color = "#1b9e77"
    plt.style.use("seaborn-v0_8-whitegrid")

    fig_w = FIG_WIDTH_IN
    fig_h = fig_w * (16 / 14)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    if len(df_long) > 0 and len(order) > 0:
        try:
            sns.violinplot(
                data=df_long, x="Value", y="label", hue="Effect",
                order=order, split=True, inner=None, width=1.8,
                palette={"Demand driver": demand_color, "Supply driver": re_color},
                ax=ax,
            )
            sns.stripplot(
                data=df_long, x="Value", y="label", hue="Effect",
                order=order, dodge=True, size=2.5, alpha=0.55,
                palette={"Demand driver": demand_color, "Supply driver": re_color},
                ax=ax,
            )
            for coll in ax.collections:
                if hasattr(coll, "get_alpha") and (coll.get_alpha() is None
                                                    or coll.get_alpha() > 0.35):
                    coll.set_alpha(0.35)
            if ax.get_legend():
                ax.get_legend().remove()
            for i, sub in enumerate(order):
                if sub not in stats.index:
                    continue
                r = stats.loc[sub]
                ax.scatter(r["Temp_Effect"], i, color=demand_color, s=30, zorder=5, alpha=0.9)
                ax.scatter(r["RE_Effect"],   i, color=re_color,     s=30, zorder=5, alpha=0.9)
                ax.scatter(r["Combined_Effect"], i, color="black", s=34, zorder=6,
                           alpha=0.9, marker="o")
                if sub in w2_by_label.index and np.isfinite(w2_by_label.loc[sub]):
                    ax.scatter(w2_by_label.loc[sub], i, color=w2_color, s=34, zorder=7,
                              alpha=0.95, marker="D")
        except Exception as exc:
            print(f"  [WARN] Absolute wasserstein dumbbell failed: {exc}")

    ax.axvline(0, color="black", lw=1.2, alpha=0.6, linestyle="--")
    finite_vals = (pd.concat([df_long["Value"], stats["Combined_Effect"], w2_by_label])
                   .replace([np.inf, -np.inf], np.nan).dropna())
    vmax_abs = max(1.0, np.nanpercentile(np.abs(finite_vals.values), 98)) if len(finite_vals) else 1.0
    ax.set_xlim(-vmax_abs * 1.15, vmax_abs * 1.15)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6)
    ax.set_ylabel("")
    ax.set_xlabel("Change (days of baseline demand)", fontsize=6, labelpad=2)
    ax.tick_params(axis="x", labelsize=5)

    xlim = ax.get_xlim()
    ax.annotate("", xy=(xlim[0] * 0.85, 1.04), xycoords=("data", "axes fraction"),
               xytext=(0, 1.04), textcoords=("data", "axes fraction"),
               arrowprops=dict(arrowstyle="->", lw=0.8, color="#777777"))
    ax.text(xlim[0] * 0.45, 1.055, "Lower SWBDs",
           transform=ax.get_xaxis_transform(),
           ha="center", va="bottom", fontsize=6, color="black")
    ax.annotate("", xy=(xlim[1] * 0.85, 1.04), xycoords=("data", "axes fraction"),
               xytext=(0, 1.04), textcoords=("data", "axes fraction"),
               arrowprops=dict(arrowstyle="->", lw=0.8, color="#777777"))
    ax.text(xlim[1] * 0.45, 1.055, "Higher SWBDs",
           transform=ax.get_xaxis_transform(),
           ha="center", va="bottom", fontsize=6, color="black")

    ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="None",
               color=demand_color, label="Demand driver", markersize=5),
        Line2D([0], [0], marker="o", linestyle="None",
               color=re_color, label="Supply driver", markersize=5),
        Line2D([0], [0], marker="o", linestyle="None",
               color="black", label="Combined effect (multi-model mean)", markersize=5),
        Line2D([0], [0], marker="D", linestyle="None",
               color=w2_color, label="Combined effect (inverse-W2 weighted)", markersize=5),
    ], title="Combined effect, two weighting schemes",
       title_fontproperties={"weight": "bold", "size": 6},
       loc="lower right", fontsize=5)

    _save_fig(fig, os.path.join(output_dir, "main",
                                "fig_main_dumbbell_absolute_days_wasserstein.png"), dpi)


# =============================================================================
# Supp (not in the paper): three-panel GWL combined-effect maps
# =============================================================================

def plot_supp_gwl_maps(df_gwl15, df_gwl2, df_gwl3,
                       shapefile_path, hatch_df,
                       output_dir, tag, share_re, dpi=300):
    cmap, norm = _make_cmap(vmin=-100, vmax=800)
    fig = _three_panel_map(
        df_gwl15, df_gwl2, df_gwl3, shapefile_path, hatch_df, cmap, norm,
        value_fn=_mmm_combined, value_col="Combined_Effect",
        title_gwl2="SWBDs change - 2.0°C warming",
        title_gwl15="SWBDs change - 1.5°C warming",
        title_gwl3="SWBDs change - 3.0°C warming",
        cbar_label="Combined effect on SWBDs (%)", dpi=dpi,
    )
    _save_fig(fig, os.path.join(output_dir, "supp",
                                f"suppfig_gwl_maps_{tag}.png"), dpi)


# =============================================================================
# Supp (not in the paper): isolated RE-supply effect
# =============================================================================

def plot_supp_re_effect(df_gwl15, df_gwl2, df_gwl3,
                        shapefile_path, hatch_df,
                        output_dir, tag, share_re, dpi=300):
    n    = 50
    base = plt.get_cmap("RdYlGn_r")
    cols = ([base(v) for v in np.linspace(0.0, 0.45, n)] +
            [base(v) for v in np.linspace(0.55, 1.0, n)])
    cmap = LinearSegmentedColormap.from_list("re_cmap", cols, N=300)
    norm = mcolors.Normalize(vmin=-100, vmax=100)
    fig = _three_panel_map(
        df_gwl15, df_gwl2, df_gwl3, shapefile_path, hatch_df, cmap, norm,
        value_fn=_mmm_re, value_col="RE_Effect",
        title_gwl2="RE supply effect - 2.0°C warming",
        title_gwl15="RE supply effect - 1.5°C warming",
        title_gwl3="RE supply effect - 3.0°C warming",
        cbar_label="RE supply effect on SWBDs (%)", dpi=dpi,
    )
    _save_fig(fig, os.path.join(output_dir, "supp",
                                f"suppfig_re_effect_{tag}.png"), dpi)


# =============================================================================
# Supp (not in the paper): isolated TAS-demand effect
# =============================================================================

def plot_supp_tas_effect(df_gwl15, df_gwl2, df_gwl3,
                         shapefile_path, hatch_df,
                         output_dir, tag, share_re, dpi=300):
    n    = 50
    base = plt.get_cmap("RdYlBu_r")
    cols = ([base(v) for v in np.linspace(0.0, 0.45, n)] +
            [base(v) for v in np.linspace(0.55, 1.0, n)])
    cmap = LinearSegmentedColormap.from_list("tas_cmap", cols, N=300)
    norm = mcolors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=200)
    fig = _three_panel_map(
        df_gwl15, df_gwl2, df_gwl3, shapefile_path, hatch_df, cmap, norm,
        value_fn=_mmm_tas, value_col="TAS_Effect",
        title_gwl2="Demand effect - 2.0°C warming",
        title_gwl15="Demand effect - 1.5°C warming",
        title_gwl3="Demand effect - 3.0°C warming",
        cbar_label="Demand effect on SWBDs (%)", dpi=dpi,
    )
    _save_fig(fig, os.path.join(output_dir, "supp",
                                f"suppfig_tas_effect_{tag}.png"), dpi)


# =============================================================================
# Supp (not in the paper): driver-decomposition ratio
# =============================================================================

def plot_supp_decomp(df_gwl2, shapefile_path, hatch_df,
                     output_dir, tag, share_re, dpi=300):
    df = df_gwl2[df_gwl2["share_re"] == share_re].copy()
    df = (df[["poly_idx", "GCM", "cum_rl_ref", "cum_rl_ds_cf", "cum_rl_gwl"]]
          .groupby(["GCM", "poly_idx"]).mean().reset_index()
          .groupby("poly_idx").agg({"cum_rl_ref": "mean", "cum_rl_ds_cf": "mean",
                                    "cum_rl_gwl": "mean"}).reset_index())
    df["RE_eff"]  = np.abs(df["cum_rl_ds_cf"] - df["cum_rl_ref"])
    df["TAS_eff"] = np.abs(df["cum_rl_gwl"] - df["cum_rl_ref"])
    df["ratio"]   = df["RE_eff"] / (df["RE_eff"] + df["TAS_eff"])

    gdf = gpd.read_file(shapefile_path)
    gdf["poly_idx"] = gdf.index
    gdf = (gdf.merge(df[["poly_idx", "ratio"]], on="poly_idx", how="left")
               .merge(hatch_df[["poly_idx", "var"]], on="poly_idx", how="left"))
    gdf["do_hatch"] = gdf["var"].le(config.AGREEMENT_THRESHOLD).fillna(False) & config.SHOW_AGREEMENT_HATCHING
    gdf = gdf.cx[:, MAP_LAT_SOUTH:MAP_LAT_NORTH]

    cmap_c = plt.get_cmap("PiYG_r")
    norm_c = mcolors.Normalize(vmin=0, vmax=1)

    fig_w = FIG_WIDTH_IN
    fig_h = fig_w * (7 / 14)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi,
                           subplot_kw={"projection": ccrs.EqualEarth()})

    vals     = gdf["ratio"].to_numpy()
    nan_mask = ~np.isfinite(vals)
    fcs      = [(1.0, 1.0, 1.0, 1.0) if n else cmap_c(norm_c(v))
                for v, n in zip(vals, nan_mask)]
    hpats = np.where(gdf["do_hatch"].to_numpy(), "/" * 21, "")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)
    dot_geoms = []   # low model agreement -> black mask
    for geom, fc, hp, is_nan in zip(gdf.geometry, fcs, hpats, nan_mask):
        if geom is None:
            continue
        ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                          facecolor=fc, edgecolor="black", linewidth=0.15, zorder=2)
        if is_nan:
            ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor="none", edgecolor="black",
                              linewidth=0.0, hatch="\\" * 10, zorder=3)
        if hp:
            dot_geoms.append(geom)
    draw_discrepancy_mask_polygons(ax, dot_geoms, zorder=4)
    ax.set_global()
    mask_poles(ax)
    try:
        ax.spines["geo"].set_visible(False)
    except KeyError:
        ax.outline_patch.set_visible(False)
    ax.set_title("Driver decomposition: RE vs. demand share of SWBD change (GWL 2.0°C)",
                 fontsize=8, fontweight="bold", pad=6)

    sm = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_c)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.03, pad=0.04)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("RE supply contribution to total driver effect", fontsize=6)
    cb.ax.tick_params(labelsize=5)

    ax.legend(handles=[
        Patch(facecolor="white", edgecolor="black", hatch="\\" * 10, label="No renewable capacities"),
        discrepancy_mask_legend_handle("Low model-agreement"),
    ], loc="lower right", bbox_to_anchor=(1.0, 0.0), bbox_transform=ax.transAxes,
       fontsize=5, framealpha=0.85, handlelength=1.0, handletextpad=0.4, borderpad=0.4)

    _save_fig(fig, os.path.join(output_dir, "supp",
                                f"suppfig_decomp_{tag}.png"), dpi)


# =============================================================================
# Supp (not in the paper): demand-sensitivity grid
# =============================================================================

def plot_supp_demand_sensitivity(shapefile_path, hatch_df, output_dir,
                                 agg_datasets_dir, dpi=300, period="Annual"):
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
    cmap_ref, norm_ref = _make_cmap(vmin=-100, vmax=800)
    diff_colors = ["#08519c", "#f7f7f7", "#d94801"]
    cmap_diff = LinearSegmentedColormap.from_list("diff_cmap", diff_colors, N=300)

    # First pass: load each demand config's Combined_Effect, then take the
    # difference against "default" so non-reference panels show how much the
    # change in demand parameters shifts the GWL2-vs-GWL0.61 effect.
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
            _mmm_combined(df_gwl2, MAIN_MIX).set_index("poly_idx")["Combined_Effect"])

    default_eff = effect_by_name.get("default")
    diff_by_name = {}
    max_abs_diff = 0.0
    for demand_name in names:
        if demand_name == "default" or default_eff is None or effect_by_name[demand_name] is None:
            continue
        diff = (effect_by_name[demand_name] - default_eff).dropna()
        diff_by_name[demand_name] = diff
        if len(diff):
            max_abs_diff = max(max_abs_diff, diff.abs().max())
    vmax_diff = max(10.0, np.ceil(max_abs_diff / 10.0) * 10.0)
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
            value_col, panel_cmap, panel_norm = "Combined_Effect", cmap_ref, norm_ref
            df_plot = eff.reset_index()
        else:
            value_col, panel_cmap, panel_norm = "Diff_Effect", cmap_diff, norm_diff
            df_plot = diff_by_name[demand_name].reset_index(name="Diff_Effect")
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
    cb_ref = fig.colorbar(sm_ref, cax=cbar_ref_ax, orientation="horizontal", extend="max")
    cb_ref.set_label("Combined effect on SWBDs (%), default - GWL 2.0C", fontsize=6)
    cb_ref.ax.tick_params(labelsize=5)

    cbar_diff_ax = fig.add_axes([0.55, 0.025, 0.36, 0.020])
    sm_diff = plt.cm.ScalarMappable(cmap=cmap_diff, norm=norm_diff)
    sm_diff.set_array([])
    cb_diff = fig.colorbar(sm_diff, cax=cbar_diff_ax, orientation="horizontal", extend="both")
    cb_diff.set_label("Change vs default (percentage points)", fontsize=6)
    cb_diff.ax.tick_params(labelsize=5)

    fig.text(0.5, 0.962, "Sensitivity to demand-model parameters",
             ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.text(0.5, 0.933,
             f"thr = {MAIN_THR}  |  tot_re = {MAIN_TOT_RE}  |  mix = {MAIN_MIX}",
             ha="center", va="bottom", fontsize=6, color="#555555")

    _save_fig(fig, os.path.join(output_dir, "supp",
                                "suppfig_demand_sensitivity.png"), dpi)


# =============================================================================
# Supp (not in the paper): inter-model uncertainty decomposition
# =============================================================================

def plot_supp_uncertainty_decomp(df_gwl2, shapefile_path, hatch_df,
                                  output_dir, tag, share_re, dpi=300):
    df = df_gwl2[df_gwl2["share_re"] == share_re].copy()
    df["RE_Effect"]   = df["cum_rl_ds_cf"] - df["cum_rl_ref"]
    df["Temp_Effect"] = df["cum_rl_gwl"] - df["cum_rl_ref"]
    df_unc = (df[["poly_idx", "GCM", "RE_Effect", "Temp_Effect"]]
              .groupby(["GCM", "poly_idx"]).mean().reset_index()
              .groupby("poly_idx")
              .agg({"RE_Effect": "std", "Temp_Effect": "std"}).reset_index()
              .rename(columns={"RE_Effect": "RE_Std", "Temp_Effect": "Temp_Std"}))
    df_unc["ratio"] = df_unc["RE_Std"] / (df_unc["RE_Std"] + df_unc["Temp_Std"])

    gdf = gpd.read_file(shapefile_path)
    gdf["poly_idx"] = gdf.index
    gdf = (gdf.merge(df_unc[["poly_idx", "ratio"]], on="poly_idx", how="left")
               .merge(hatch_df[["poly_idx", "var"]], on="poly_idx", how="left"))
    gdf["do_hatch"] = gdf["var"].le(config.AGREEMENT_THRESHOLD).fillna(False) & config.SHOW_AGREEMENT_HATCHING
    gdf = gdf.cx[:, MAP_LAT_SOUTH:MAP_LAT_NORTH]
    cmap_c = plt.get_cmap("PiYG_r")
    norm_c = mcolors.Normalize(vmin=0, vmax=1)

    fig_w = FIG_WIDTH_IN
    fig_h = fig_w * (7 / 14)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi,
                           subplot_kw={"projection": ccrs.EqualEarth()})

    vals     = gdf["ratio"].to_numpy()
    nan_mask = ~np.isfinite(vals)
    fcs      = [(1.0, 1.0, 1.0, 1.0) if n else cmap_c(norm_c(v))
                for v, n in zip(vals, nan_mask)]
    hpats = np.where(gdf["do_hatch"].to_numpy(), "/" * 21, "")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)
    dot_geoms = []   # low model agreement -> black mask
    for geom, fc, hp, is_nan in zip(gdf.geometry, fcs, hpats, nan_mask):
        if geom is None:
            continue
        ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                          facecolor=fc, edgecolor="black", linewidth=0.15, zorder=2)
        if is_nan:
            ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor="none", edgecolor="black",
                              linewidth=0.0, hatch="\\" * 10, zorder=3)
        if hp:
            dot_geoms.append(geom)
    draw_discrepancy_mask_polygons(ax, dot_geoms, zorder=4)
    ax.set_global()
    mask_poles(ax)
    try:
        ax.spines["geo"].set_visible(False)
    except KeyError:
        ax.outline_patch.set_visible(False)
    ax.set_title(
        "Inter-model uncertainty decomposition: RE supply share of total spread (GWL 2.0°C)",
        fontsize=8, fontweight="bold", pad=6)

    sm = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_c)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.03, pad=0.04)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("RE supply std / (RE supply std + demand std)", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    ax.legend(handles=[
        Patch(facecolor="white", edgecolor="black", hatch="\\" * 10, label="No renewable capacities"),
        discrepancy_mask_legend_handle("Low model-agreement"),
    ], loc="lower right", bbox_to_anchor=(1.0, 0.0), bbox_transform=ax.transAxes,
       fontsize=5, framealpha=0.85, handlelength=1.0, handletextpad=0.4, borderpad=0.4)

    _save_fig(fig, os.path.join(output_dir, "supp",
                                f"suppfig_uncertainty_decomp_{tag}.png"), dpi)


# =============================================================================
# Supp (not in the paper): absolute inter-model spread in RE supply effect
# =============================================================================

def plot_supp_re_variability(df_gwl2, shapefile_path, hatch_df,
                              output_dir, tag, share_re, dpi=300):
    df = df_gwl2[df_gwl2["share_re"] == share_re].copy()
    df["RE_Effect"] = df["cum_rl_ds_cf"] - df["cum_rl_ref"]
    df_var = (df[["poly_idx", "GCM", "RE_Effect"]]
              .groupby(["GCM", "poly_idx"]).agg({"RE_Effect": "mean"}).reset_index()
              .groupby("poly_idx").agg({"RE_Effect": "std"}).reset_index()
              .rename(columns={"RE_Effect": "RE_Std"}))

    gdf = gpd.read_file(shapefile_path)
    gdf["poly_idx"] = gdf.index
    gdf = (gdf.merge(df_var[["poly_idx", "RE_Std"]], on="poly_idx", how="left")
               .merge(hatch_df[["poly_idx", "var"]], on="poly_idx", how="left"))
    gdf["do_hatch"] = gdf["var"].le(config.AGREEMENT_THRESHOLD).fillna(False) & config.SHOW_AGREEMENT_HATCHING
    gdf = gdf.cx[:, MAP_LAT_SOUTH:MAP_LAT_NORTH]
    cmap_c = plt.get_cmap("Reds")
    vmax   = np.nanpercentile(df_var["RE_Std"].dropna().values, 95)
    norm_c = mcolors.Normalize(vmin=0, vmax=vmax)

    fig_w = FIG_WIDTH_IN
    fig_h = fig_w * (7 / 14)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi,
                           subplot_kw={"projection": ccrs.EqualEarth()})

    vals     = gdf["RE_Std"].to_numpy()
    nan_mask = ~np.isfinite(vals)
    fcs      = [(1.0, 1.0, 1.0, 1.0) if n else cmap_c(norm_c(v))
                for v, n in zip(vals, nan_mask)]
    hpats = np.where(gdf["do_hatch"].to_numpy(), "/" * 21, "")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)
    dot_geoms = []   # low model agreement -> black mask
    for geom, fc, hp, is_nan in zip(gdf.geometry, fcs, hpats, nan_mask):
        if geom is None:
            continue
        ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                          facecolor=fc, edgecolor="black", linewidth=0.15, zorder=2)
        if is_nan:
            ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor="none", edgecolor="black",
                              linewidth=0.0, hatch="\\" * 10, zorder=3)
        if hp:
            dot_geoms.append(geom)
    draw_discrepancy_mask_polygons(ax, dot_geoms, zorder=4)
    ax.set_global()
    mask_poles(ax)
    try:
        ax.spines["geo"].set_visible(False)
    except KeyError:
        ax.outline_patch.set_visible(False)
    ax.set_title(
        "Inter-model spread in RE supply effect on SWBDs (std across GCMs, GWL 2.0°C)",
        fontsize=8, fontweight="bold", pad=6)
    sm = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_c)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.03, pad=0.04)
    cb.set_label("Inter-model std of RE supply effect on SWBDs", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    ax.legend(handles=[
        Patch(facecolor="white", edgecolor="black", hatch="\\" * 10, label="No renewable capacities"),
        discrepancy_mask_legend_handle("Low model-agreement"),
    ], loc="lower right", bbox_to_anchor=(1.0, 0.0), bbox_transform=ax.transAxes,
       fontsize=5, framealpha=0.85, handlelength=1.0, handletextpad=0.4, borderpad=0.4)

    _save_fig(fig, os.path.join(output_dir, "supp",
                                f"suppfig_re_variability_{tag}.png"), dpi)


def plot_re_share_effect(gwl_dfs_by_share, shapefile_path, hatch_df,
                         output_dir, tag, dpi=300):
    TOT_RE_VALS   = [0.25, 0.5, 0.75]
    GWL_TITLES    = ["1.5°C", "2.0°C", "3.0°C"]
    PANEL_LETTERS = list("abcdefghi")
    cmap, norm    = _make_cmap(vmin=-100, vmax=800)

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
            mmm    = _mmm_combined(df_gwl, share_re="current", vmax=800)
            gdf    = gdf_base.copy().merge(mmm, on="poly_idx", how="left")
            gdf    = gdf.merge(hatch_df[["poly_idx", "var"]], on="poly_idx", how="left")
            gdf["do_hatch"] = gdf["var"].le(config.AGREEMENT_THRESHOLD).fillna(False) & config.SHOW_AGREEMENT_HATCHING
            _draw_map(ax, gdf, "Combined_Effect", cmap, norm, hatch_df,
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
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="max")
    cbar.set_label("Combined effect on SWBDs (%)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    fig.text(0.5, 0.995, "Effect of solar-wind penetration level on SWBDs",
             ha="center", va="top", fontsize=8, fontweight="bold")
    fig.text(0.5, 0.970, "Multi-model mean, current mix, threshold = 0.99",
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
                                f"suppfig_re_share_effect.png"), dpi)


def plot_supp_combined_driver_effects(df_gwl2, shapefile_path, hatch_df,
                                      output_dir, share_re="current", dpi=300):
    _driver_effects_figure(
        _mmm_re(df_gwl2, share_re, vmax=None), "RE_Effect",
        _mmm_tas(df_gwl2, share_re, vmax=None), "TAS_Effect",
        "Effect on SWBDs (%)",
        df_gwl2, shapefile_path, hatch_df, share_re,
        os.path.join(output_dir, "supp", "suppfig_combined_driver_effects.png"), dpi)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Non-paper variants of the fig45.py figures.")
    p.add_argument("--output_dir", default="../final_figs")
    p.add_argument("--dpi",        type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    for sub in ("main", "supp"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    hatch_df, _ = load_hatch_agg(
        PATHS["agreement_nc"], PATHS["shapefile"],
        agreement_aggregated_nc=PATHS["agreement_aggregated_nc"],
    )
    ds_wasserstein_agg = _load_wasserstein_agg(PATHS["wasserstein_aggregated_nc"])
    w2_table = (_wasserstein_weight_table(ds_wasserstein_agg)
                if ds_wasserstein_agg is not None else None)

    csv_current = os.path.join(PATHS["out_dir"],
        f"rl_agg_adaptation_Annual_{MAIN_THR}_ren_pen_{MAIN_TOT_RE}_{MAIN_MIX}_v2.csv")
    if os.path.exists(csv_current):
        df_gwl15, df_gwl2, df_gwl3 = load_gwl_dfs(csv_current)
        plot_main_gwl_maps(df_gwl15, df_gwl2, df_gwl3,
                           PATHS["shapefile"], hatch_df,
                           args.output_dir, dpi=args.dpi, share_re=MAIN_MIX)
        plot_main_dumbbell(df_gwl2, PATHS["shapefile"], dpi=args.dpi,
                           share_re=MAIN_MIX, output_dir=args.output_dir)
        if w2_table is not None:
            plot_main_gwl_maps_absolute_wasserstein(
                df_gwl15, df_gwl2, df_gwl3, PATHS["shapefile"], hatch_df, w2_table,
                args.output_dir, dpi=args.dpi, share_re=MAIN_MIX)
            plot_main_dumbbell_absolute_wasserstein(
                df_gwl2, PATHS["shapefile"], w2_table, dpi=args.dpi,
                share_re=MAIN_MIX, output_dir=args.output_dir)
            plot_gwl2_wasserstein_vs_mmm(
                df_gwl2, PATHS["shapefile"], hatch_df, w2_table,
                args.output_dir, dpi=args.dpi, share_re=MAIN_MIX)
        plot_supp_combined_driver_effects(df_gwl2, PATHS["shapefile"], hatch_df,
                                          args.output_dir, share_re="current", dpi=args.dpi)

    if ds_wasserstein_agg is not None:
        ds_wasserstein_agg.close()

    plot_supp_demand_sensitivity(
        shapefile_path=PATHS["shapefile"], hatch_df=hatch_df,
        output_dir=args.output_dir, agg_datasets_dir=PATHS["out_dir"], dpi=args.dpi)

    csvs = {tot_re: os.path.join(PATHS["out_dir"],
                f"rl_agg_adaptation_Annual_0.99_ren_pen_{tot_re}_current_v2.csv")
            for tot_re in (0.25, 0.5, 0.75)}
    if all(os.path.exists(c) for c in csvs.values()):
        gwl_dfs_by_share = {k: load_gwl_dfs(c) for k, c in csvs.items()}
        plot_re_share_effect(gwl_dfs_by_share, PATHS["shapefile"], hatch_df,
                             args.output_dir, "re_share_effect", dpi=args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()

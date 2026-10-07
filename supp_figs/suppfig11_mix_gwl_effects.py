# -*- coding: utf-8 -*-
"""
Extended Data Fig. 11: effect of the future renewable mix vs. effect of
2°C global warming on SWBDs (cumulative residual-load exceedances).

  a - scatter, per region: mix effect (future vs. current wind/solar share,
      both at GWL0-61) against warming effect (GWL2 vs. GWL0-61, current mix),
      coloured by the direction of the mix change (more wind / more solar /
      no change)
  b - map classifying each region by the signs of those two effects

Two versions are written to <output_dir>/supp/:
  suppfig_mix_gwl_effects.png                 relative change (%)
  suppfig_mix_gwl_effects_absolute_days.png   change normalized by each
                                              region's baseline demand
                                              (days of baseline demand),
                                              same convention as fig45.py's
                                              _mmm_absolute_days

Inputs are the RL CSVs written by main_pipeline/make_rl_files.py (current
mix: STEP 1a, future mix: STEP 1b), the
renewable share CSV (current_ratio/future_ratio), the shapefile, and the
aggregated model-agreement mask.

Port of cell 11 of como24_group5/code_final/3.5 calculate_residual_load.ipynb
(previously fig45.py's plot_supp_mix_effect).
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from map_overlays import draw_discrepancy_mask_polygons, discrepancy_mask_legend_handle
from fig45 import (PATHS, MAIN_THR, MAIN_TOT_RE, FIG_WIDTH_IN,
                   MAP_LAT_SOUTH, MAP_LAT_NORTH, mask_poles,
                   load_gwl_dfs, load_hatch_agg, _save_fig)

NO_MIX_CHANGE_COLOR = "#8B4513"
QUADRANT_COLORS = {
    "both_up":      "#d73027",
    "both_down":    "#4575b4",
    "warming_up":   "#fee090",
    "warming_down": "#91bfdb",
}


# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Mix effect vs. global-warming effect on SWBDs "
                    "(supplementary figure 11), in % and in days of baseline demand.")
    p.add_argument("--rl_dir", default=PATHS["out_dir"],
                   help="Directory holding make_rl_files.py's rl_agg_adaptation_*_v2.csv files.")
    p.add_argument("--thr", type=float, default=MAIN_THR)
    p.add_argument("--tot_re", type=float, default=MAIN_TOT_RE)
    p.add_argument("--output_dir", default="../final_figs")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


# =============================================================================
# Data
# =============================================================================

def _load_shares(share_csv):
    df_share = pd.read_csv(share_csv)
    cur = xr.DataArray(df_share["current_ratio"].values,
                       coords=[df_share["poly_idx"].values], dims=["poly_idx"])
    if "future_ratio" in df_share.columns:
        fut = xr.DataArray(df_share["future_ratio"].values,
                           coords=[df_share["poly_idx"].values], dims=["poly_idx"])
    else:
        print("  [WARN] no future_ratio column in the share CSV -- "
              "every region is treated as 'no change in mix'.")
        fut = cur.copy()
    return cur, fut


def _mix_df_percent(df_gwl2_curr, df_gwl2_fut):
    """Relative change (%): cum_rl averaged over runs within each GCM, then
    over GCMs, then the ratio is taken on those multi-model means."""
    cols = ["cum_rl_ref", "cum_rl_gwl", "cum_rl_tas", "cum_rl_ds_cf"]
    df_c = (df_gwl2_curr[df_gwl2_curr["share_re"] == "current"]
            .groupby(["poly_idx", "GCM"], as_index=False)[cols].mean()
            .groupby("poly_idx", as_index=False)[cols].mean())
    df_f = (df_gwl2_fut[df_gwl2_fut["share_re"] == "future"]
            .groupby(["poly_idx", "GCM"], as_index=False)[["cum_rl_ref"]].mean()
            .groupby("poly_idx", as_index=False)[["cum_rl_ref"]].mean()
            .rename(columns={"cum_rl_ref": "cum_rl_ref_future"}))

    df_mix = df_c.merge(df_f, on="poly_idx", how="left")
    df_mix["mix_effect"] = (
        (df_mix["cum_rl_ref_future"] - df_mix["cum_rl_ref"]) / df_mix["cum_rl_ref"] * 100)
    df_mix["gwl_effect"] = (
        (df_mix["cum_rl_gwl"] - df_mix["cum_rl_ref"]) / df_mix["cum_rl_ref"] * 100)
    return df_mix[["poly_idx", "mix_effect", "gwl_effect"]]


def _mix_df_days(df_gwl2_curr, df_gwl2_fut):
    """Change in days of baseline demand: each (GCM, run, poly_idx) row is
    normalized by its own demand_bas first, then averaged over runs within
    each GCM and over GCMs (same convention as fig45.py's _mmm_absolute_days).
    demand_bas is identical in the current and future CSVs (fig45's
    _load_data always builds ds_cf_mean from the current share), so the
    current one is used."""
    keys = ["poly_idx", "GCM", "run"]
    df_c = df_gwl2_curr[df_gwl2_curr["share_re"] == "current"][
        keys + ["cum_rl_ref", "cum_rl_gwl", "demand_bas"]]
    df_f = (df_gwl2_fut[df_gwl2_fut["share_re"] == "future"][keys + ["cum_rl_ref"]]
            .rename(columns={"cum_rl_ref": "cum_rl_ref_future"}))
    df = df_c.merge(df_f, on=keys, how="left")

    df["mix_effect"] = (df["cum_rl_ref_future"] - df["cum_rl_ref"]) / df["demand_bas"]
    df["gwl_effect"] = (df["cum_rl_gwl"] - df["cum_rl_ref"]) / df["demand_bas"]
    df[["mix_effect", "gwl_effect"]] = (df[["mix_effect", "gwl_effect"]]
                                        .replace([np.inf, -np.inf], np.nan))
    return (df.groupby(["poly_idx", "GCM"], as_index=False)[["mix_effect", "gwl_effect"]].mean()
              .groupby("poly_idx", as_index=False)[["mix_effect", "gwl_effect"]].mean())


def _add_mix_change(df_mix, cur_share, fut_share):
    df_mix = df_mix.copy()
    poly_arr = df_mix["poly_idx"].values
    cur_vals = cur_share.sel(poly_idx=xr.DataArray(poly_arr, dims="z")).values
    fut_vals = fut_share.sel(poly_idx=xr.DataArray(poly_arr, dims="z")).values
    df_mix["scatter_color"] = np.where(cur_vals < fut_vals, "blue",
                                       np.where(cur_vals > fut_vals, "orange",
                                                NO_MIX_CHANGE_COLOR))
    df_mix["no_mix_change"] = (cur_vals == fut_vals)
    return df_mix


def _assign_color(row):
    if row["no_mix_change"]:                                return NO_MIX_CHANGE_COLOR
    elif row["mix_effect"] > 0 and row["gwl_effect"] > 0:   return QUADRANT_COLORS["both_up"]
    elif row["mix_effect"] < 0 and row["gwl_effect"] <= 0:  return QUADRANT_COLORS["both_down"]
    elif row["mix_effect"] < 0 and row["gwl_effect"] > 0:   return QUADRANT_COLORS["warming_up"]
    else:                                                    return QUADRANT_COLORS["warming_down"]


def _print_quadrant_stats(df_mix, unit_name):
    counts = df_mix["map_color"].value_counts()
    names = {**{v: k for k, v in QUADRANT_COLORS.items()},
             NO_MIX_CHANGE_COLOR: "no_mix_change"}
    summary = ", ".join(f"{names[c]}={int(counts.get(c, 0))}" for c in names)
    print(f"  [INFO] {unit_name}: {summary} (out of {len(df_mix)} regions)")


# =============================================================================
# Figure
# =============================================================================

def plot_mix_gwl_effects(df_mix, shapefile_path, idxs_to_hatch, out_path,
                         unit_label, linthresh, dpi=300):
    fig_w = FIG_WIDTH_IN
    fig_h = fig_w * (6 / 14)
    fig   = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs    = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.4])
    ax1   = fig.add_subplot(gs[0, 0])
    ax2   = fig.add_subplot(gs[0, 1], projection=ccrs.EqualEarth())

    ax1.annotate(
        "$\\mathbf{a}$",
        xy=(0.02, 1.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=8,
    )
    ax2.annotate(
        "$\\mathbf{b}$",
        xy=(0.02, 1.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=8,
        path_effects=[withStroke(linewidth=1.5, foreground="white")],
    )

    ax1.scatter(x=df_mix["mix_effect"], y=df_mix["gwl_effect"],
                c=df_mix["scatter_color"], marker="x", s=4, linewidths=0.4)
    ax1.set_xlabel(f"Mix effect on SWBD ({unit_label})", fontsize=5)
    ax1.set_ylabel(f"Global warming effect on SWBD ({unit_label})", fontsize=5)
    ax1.tick_params(labelsize=5)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xscale("symlog", linthresh=linthresh)
    ax1.set_yscale("symlog", linthresh=linthresh)
    ax1.legend(
        handles=[
            Line2D([0], [0], marker="x", linestyle="None",
                   markeredgecolor="blue",    color="blue",    label="Increase in wind share", markersize=3),
            Line2D([0], [0], marker="x", linestyle="None",
                   markeredgecolor="orange",  color="orange",  label="Increase in solar share", markersize=3),
            Line2D([0], [0], marker="x", linestyle="None",
                   markeredgecolor=NO_MIX_CHANGE_COLOR, color=NO_MIX_CHANGE_COLOR,
                   label="No change in mix", markersize=3),
        ],
        title="Renewable mix change", loc="upper center", frameon=True,
        fontsize=4, title_fontsize=4,
        bbox_to_anchor=(0.5, -0.25), bbox_transform=ax1.transAxes,
        handlelength=1.0, handletextpad=0.4, borderpad=0.4,
    )

    gdf_diff = gpd.read_file(shapefile_path)
    gdf_diff["poly_idx"] = gdf_diff.index
    gdf_diff = gdf_diff.cx[:, MAP_LAT_SOUTH:MAP_LAT_NORTH]
    color_map = dict(zip(df_mix["poly_idx"], df_mix["map_color"]))
    no_data_mask = gdf_diff["poly_idx"].map(color_map).isna()
    gdf_diff["color"] = gdf_diff["poly_idx"].map(color_map).fillna("white")
    no_data_set = set(gdf_diff[no_data_mask]["poly_idx"].tolist())
    hatch_set   = set(idxs_to_hatch.tolist() if hasattr(idxs_to_hatch, "tolist") else list(idxs_to_hatch))

    dot_geoms = []   # low model agreement -> black mask
    for geom, clr, pid in zip(gdf_diff.geometry, gdf_diff["color"], gdf_diff["poly_idx"]):
        if geom is None:
            continue
        ax2.add_geometries([geom], crs=ccrs.PlateCarree(),
                           facecolor=clr, edgecolor="black", linewidth=0.15, zorder=2)
        if pid in no_data_set:
            ax2.add_geometries([geom], crs=ccrs.PlateCarree(),
                               facecolor="none", edgecolor="black",
                               linewidth=0.0, hatch="\\" * 10, zorder=3)
        if pid in hatch_set:
            dot_geoms.append(geom)

    draw_discrepancy_mask_polygons(ax2, dot_geoms, zorder=4)
    ax2.set_global()
    mask_poles(ax2)
    try:
        ax2.spines["geo"].set_visible(False)
    except KeyError:
        ax2.outline_patch.set_visible(False)
    ax2.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.15)
    ax2.set_axis_off()

    ax2.legend(
        handles=[
            Patch(facecolor=QUADRANT_COLORS["both_up"],      edgecolor="none", label="Both increase SWBDs"),
            Patch(facecolor=QUADRANT_COLORS["both_down"],    edgecolor="none", label="Both decrease SWBDs"),
            Patch(facecolor=QUADRANT_COLORS["warming_up"],   edgecolor="none", label="Warming increases SWBDs, Mix decreases SWBDs"),
            Patch(facecolor=QUADRANT_COLORS["warming_down"], edgecolor="none", label="Warming decreases SWBDs, Mix increases SWBDs"),
            Patch(facecolor=NO_MIX_CHANGE_COLOR, edgecolor="none", label="No change in mix"),
            Patch(facecolor="white", edgecolor="black", hatch="\\" * 10, label="No RE capacities"),
            discrepancy_mask_legend_handle("Low model agreement"),
        ],
        loc="upper center", fontsize=4, ncol=2,
        bbox_to_anchor=(0.5, -0.08), bbox_transform=ax2.transAxes,
        handlelength=1.0, handletextpad=0.4, borderpad=0.4,
    )

    _save_fig(fig, out_path, dpi)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "supp"), exist_ok=True)

    csv_current = os.path.join(args.rl_dir,
        f"rl_agg_adaptation_Annual_{args.thr}_ren_pen_{args.tot_re}_current_v2.csv")
    csv_future = os.path.join(args.rl_dir,
        f"rl_agg_adaptation_Annual_{args.thr}_ren_pen_{args.tot_re}_future_v2.csv")
    missing = [p for p in (csv_current, csv_future) if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"  [ERROR] missing RL CSV: {p}")
        raise SystemExit("Run main_pipeline/make_rl_files.py first (STEP 1a writes the "
                         "current-mix CSV, STEP 1b the future-mix one).")

    print("\n=== Loading RL CSVs ===")
    _, df_gwl2_curr, _ = load_gwl_dfs(csv_current)
    _, df_gwl2_fut,  _ = load_gwl_dfs(csv_future)
    cur_share, fut_share = _load_shares(PATHS["df_share_csv"])

    print("\n=== Agreement mask ===")
    _, idxs_to_hatch = load_hatch_agg(
        PATHS["agreement_nc"], PATHS["shapefile"],
        agreement_aggregated_nc=PATHS["agreement_aggregated_nc"],
    )

    print("\n=== Figures ===")
    for df_mix, unit_name, unit_label, linthresh, fname in [
        (_mix_df_percent(df_gwl2_curr, df_gwl2_fut), "percent", "%", 10,
         "suppfig_mix_gwl_effects.png"),
        (_mix_df_days(df_gwl2_curr, df_gwl2_fut), "days", "days of baseline demand", 1,
         "suppfig11_mix_gwl_effects_absolute_days.png"),
    ]:
        df_mix = _add_mix_change(df_mix, cur_share, fut_share)
        df_mix["map_color"] = df_mix.apply(_assign_color, axis=1)
        _print_quadrant_stats(df_mix, unit_name)
        plot_mix_gwl_effects(df_mix, PATHS["shapefile"], idxs_to_hatch,
                             os.path.join(args.output_dir, "supp", fname),
                             unit_label, linthresh, dpi=args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()

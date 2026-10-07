# -*- coding: utf-8 -*-
"""
Figures 4 and 5: projected changes in SWBD (solar-wind budget droughts) at
1.5/2/3 degC warming, per country/region (Fig. 4, maps) and for selected
regions with their supply/demand decomposition (Fig. 5, dumbbell plot), in
days of baseline demand.

Reads the residual-load CSVs written by main_pipeline/make_rl_files.py (run it
first). The map / data helpers below are shared with the Extended Data scripts
supp_figs/suppfig8, 9, 11, 12 and 13 and aux_code/extra_figs_fig45.py.
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
from shapely.geometry import Polygon
import xarray as xr
import xagg as xa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Black mask for low model agreement (obs.-projection trend discrepancy),
# drawn in solid black on the aggregated polygon maps (see map_overlays.py).
from map_overlays import draw_discrepancy_mask_polygons

# Main-scenario constants and the run exclusions live with the RL computation.
from make_rl_files import MAIN_THR, MAIN_TOT_RE, MAIN_MIX, EXCLUDED_RUNS


# =============================================================================
# Figure size constants (LaTeX-compatible)
# =============================================================================
FIG_WIDTH_IN = 5.15   # single column width -- pt fontsizes match LaTeX

# Latitude band shown on EqualEarth maps in this module (matches the
# analysis's own poleward exclusion; see Methods: "Regions poleward of 68N
# and 58S were excluded due to artifacts in the duration metric"). Applied
# by masking data/shapefiles to this band and calling ax.set_global() --
# NOT ax.set_extent(), which miscalibrates on EqualEarth's curved meridians:
# it clips to the bounding rectangle of the extent box's own corners, whose
# right edge only touches the true 180 deg meridian at MAP_LAT_SOUTH/NORTH
# themselves, sitting well short of it at other latitudes -- slicing
# through real land (e.g. eastern Australia) even though it's nominally
# within +/-180 deg longitude.
MAP_LAT_SOUTH = -58
MAP_LAT_NORTH = 68


def mask_poles(ax, lat_south=MAP_LAT_SOUTH, lat_north=MAP_LAT_NORTH, zorder=12):
    """
    White out everything poleward of [lat_south, lat_north] on a set_global()
    EqualEarth map. Needed because cfeature.COASTLINE/ax.coastlines() draw
    the *entire* globe's coastlines (Antarctica, remote Arctic islands)
    regardless of how the data/shapefile were masked to this band, and
    shp.cx[:, lat_south:lat_north] keeps whole country geometries (e.g.
    Russia, Canada, Greenland) rather than clipping them at the band's edge
    -- both leak real content poleward of the intended crop (see
    MAP_LAT_SOUTH/NORTH above). Draws a white cap over each pole, in
    PlateCarree and densely sampled in longitude so it follows the
    projection's own curved boundary, on top of coastlines/boundaries but
    below panel labels/region boxes (zorder 20+ elsewhere in this module).
    """
    lons = np.linspace(-180.0, 180.0, 361)
    for lat_edge, lat_pole in ((lat_south, -90.0), (lat_north, 90.0)):
        cap = Polygon(
            list(zip(lons, np.full_like(lons, lat_edge))) +
            list(zip(lons[::-1], np.full_like(lons, lat_pole)))
        )
        ax.add_geometries([cap], crs=ccrs.PlateCarree(),
                          facecolor="white", edgecolor="none", zorder=zorder)


# =============================================================================
# PATHS
# =============================================================================
PATHS = {
    "path_preprocessed": config.PATH_PREPROCESSED,
    "temp_folder":        config.TEMP_FOLDER,
    "shapefile":          config.SHAPEFILE_PATH,
    "df_share_csv":       config.SHARE_RENEWABLE_CSV,
    "agreement_nc":       config.AGREEMENT_NC_PATH,
    "agreement_aggregated_nc": config.AGREEMENT_AGGREGATED_NC_PATH,
    "wasserstein_aggregated_nc": config.WASSERSTEIN_AGGREGATED_NC_PATH,
    "out_dir":            config.RL_OUT_DIR,
    "ssp":                config.SSP,
    "reanalysis":         config.REANALYSIS,
}
GWL_DISPLAY = {"GWL0-61": "0.61 C", "GWL1-5": "1.5 C",
               "GWL2": "2.0 C", "GWL3": "3.0 C"}

REGION_NAMES = [
    "Ecuador", "Ivory Coast", "Poland", "Parana",
    "Japan", "Andhra Pradesh", "Washington", "Queensland", "Egypt", "Florida",
]
REGION_LABELS = [
    "Ecuador", "Ivory Coast", "Poland", "Parana (BRA)", "Japan",
    "Andhra Pradesh (IND)", "Washington (USA)", "Queensland (AUS)", "Egypt", "Florida (USA)",
]
DICT_LABELS = dict(zip(REGION_NAMES, REGION_LABELS))


# =============================================================================
# DATA WRANGLING  (unchanged)
# =============================================================================

def build_gwl_df(df_source, gwl_label):
    keys = ["poly_idx", "GCM", "run", "share_re"]
    base = ((df_source["gwl_tas"] == "GWL0-61") & (df_source["gwl_ds_cf"] == "GWL0-61"))
    df_out = (df_source[base][keys + ["rl_cum", "demand_bas"]].reset_index(drop=True)
              .rename(columns={"rl_cum": "cum_rl_ref"}))
    for col, mask in {
        "cum_rl_gwl": ((df_source["gwl_tas"] == gwl_label) &
                       (df_source["gwl_ds_cf"] == gwl_label)),
        "cum_rl_tas": ((df_source["gwl_tas"] == gwl_label) &
                       (df_source["gwl_ds_cf"] == "GWL0-61")),
        "cum_rl_ds_cf": ((df_source["gwl_tas"] == "GWL0-61") &
                       (df_source["gwl_ds_cf"] == gwl_label)),
    }.items():
        tmp = (df_source[mask][keys + ["rl_cum"]].reset_index(drop=True)
               .rename(columns={"rl_cum": col}))
        df_out = df_out.merge(tmp, on=keys, how="left")
    return df_out


def load_gwl_dfs(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    mask_excl = pd.Series(False, index=df.index)
    for gcm, run in EXCLUDED_RUNS:
        mask_excl |= (df["GCM"] == gcm) & (df["run"] == run)
    df = df[~mask_excl]
    def _sub(label):
        mask = ((df["gwl_ds_cf"] == label) | (df["gwl_tas"] == label) |
                ((df["gwl_ds_cf"] == "GWL0-61") & (df["gwl_tas"] == "GWL0-61")))
        return build_gwl_df(df[mask].copy(), label)
    return _sub("GWL1-5"), _sub("GWL2"), _sub("GWL3")


def load_hatch_agg(agreement_nc, shapefile_path, agreement_threshold=None, agreement_aggregated_nc=None):
    """
    hatch_df/idxs_hatch: which shapefile polygons get hatched for low model
    agreement with ERA5's observed trend (agreement_pct < agreement_threshold,
    default config.AGREEMENT_THRESHOLD -- 50%% of models with an overlapping
    trend CI, see trend_sev_eval.py's build_agreement_mask()).

    `agreement_aggregated_nc` (recommended -- config.AGREEMENT_AGGREGATED_NC_PATH)
    is a poly_idx-native agreement_pct DataArray built by trend_sev_eval.py's
    aggregated pipeline (uncertainty_range_aggregated/make_ref_trend_ci_aggregated/
    build_agreement_mask) straight from each polygon's own aggregated wcf/scf
    trend, one value per polygon already -- pass it (once generated) to use
    that directly instead of the `agreement_nc` fallback below, which
    approximates a polygon score by area-weighting the *pixel*-level mask
    onto the shapefile via xagg.
    """
    agreement_threshold = config.AGREEMENT_THRESHOLD if agreement_threshold is None else agreement_threshold

    if agreement_aggregated_nc is not None and os.path.exists(agreement_aggregated_nc):
        agreement_pct = xr.open_dataarray(agreement_aggregated_nc)
        hatch_df = agreement_pct.to_dataframe(name="var").reset_index()
        idxs_hatch = hatch_df[hatch_df["var"] <= agreement_threshold]["poly_idx"].values
        return hatch_df, idxs_hatch

    hatchings  = xr.open_dataarray(agreement_nc)
    shapefile  = gpd.read_file(shapefile_path)
    weight_map = xa.pixel_overlaps(hatchings, shapefile)
    hatch_agg  = xa.aggregate(hatchings, weight_map).to_dataset()
    hatch_df   = hatch_agg[["poly_idx", "var"]].to_dataframe().reset_index()
    idxs_hatch = hatch_df[hatch_df["var"] <= agreement_threshold]["poly_idx"].values
    return hatch_df, idxs_hatch


# =============================================================================
# MAP / FIGURE HELPERS
# =============================================================================

def _make_cmap(vmin=-100, vmax=800):
    n_neg = 50
    n_pos = int(n_neg * abs(vmax) / 100)
    base  = plt.get_cmap("RdYlGn_r")
    cols  = ([base(v) for v in np.linspace(0.0, 0.45, n_neg)] +
             [base(v) for v in np.linspace(0.55, 1.0, n_pos)])
    return (LinearSegmentedColormap.from_list("custom", cols, N=300),
            mcolors.Normalize(vmin=vmin, vmax=vmax))


def _draw_map(ax, gdf, value_col, cmap, norm, hatch_df,
              title, panel_letter, density=7, title_fontsize=8):
    # Restrict to the analysis's latitude band here (rather than trusting
    # every caller to have already done so) -- this is the single chokepoint
    # all _draw_map() callers go through, and set_extent() below has been
    # replaced with set_global(), so nothing but this filter keeps
    # Antarctica (which has no data, so would draw as a hatched/white "no
    # data" polygon) out of the figure. See MAP_LAT_SOUTH/NORTH above.
    gdf2 = gdf.cx[:, MAP_LAT_SOUTH:MAP_LAT_NORTH].copy()
    if "var" not in gdf2.columns:
        gdf2 = gdf2.merge(hatch_df[["poly_idx", "var"]], on="poly_idx", how="left")
    gdf2["do_hatch"] = gdf2["var"].le(config.AGREEMENT_THRESHOLD).fillna(False) & config.SHOW_AGREEMENT_HATCHING
    vals     = gdf2[value_col].to_numpy()
    nan_mask = ~np.isfinite(vals)
    fcs      = [(1.0, 1.0, 1.0, 1.0) if n else cmap(norm(v))
                for v, n in zip(vals, nan_mask)]
    hpats = np.where(gdf2["do_hatch"].to_numpy(), "/" * density * 3, "")
    dot_geoms = []   # low model agreement -> black mask
    for geom, fc, hp, is_nan in zip(gdf2.geometry, fcs, hpats, nan_mask):
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
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.15)
    if panel_letter:
        ax.annotate(
            f"$\\mathbf{{{panel_letter}}}$",
            xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=8,
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=4)


def _add_colorbar(fig, cmap, norm, label,
                  pos=(0.25, 0.06, 0.5, 0.018), extend="neither"):
    ax_cb = fig.add_axes(pos)
    sm    = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar  = fig.colorbar(sm, cax=ax_cb, orientation="horizontal", extend=extend)
    cbar.set_label(label, fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    return cbar


def _build_gdf(shapefile_path, df_data, hatch_df):
    gdf = gpd.read_file(shapefile_path)
    gdf["poly_idx"] = gdf.index
    gdf = gdf.merge(df_data, on="poly_idx", how="left")
    gdf = gdf.merge(hatch_df[["poly_idx", "var"]], on="poly_idx", how="left")
    gdf["do_hatch"] = gdf["var"].le(config.AGREEMENT_THRESHOLD).fillna(False) & config.SHOW_AGREEMENT_HATCHING
    return gdf


def _mmm(df_gwl, effect_col, share_re, vmax, compute_fn):
    df = df_gwl[df_gwl["share_re"] == share_re].copy()
    compute_fn(df)
    out = (df[["poly_idx", "GCM", effect_col]]
           .groupby(["GCM", "poly_idx"])[effect_col].mean().reset_index()
           .groupby("poly_idx")[effect_col].mean().reset_index())
    if vmax is not None:
        out.loc[out[effect_col] > vmax, effect_col] = vmax
    return out


def _mmm_combined(df_gwl, share_re="current", vmax=800):
    def fn(df):
        df["Combined_Effect"] = (df["cum_rl_gwl"] - df["cum_rl_ref"]) / df["cum_rl_ref"] * 100
    return _mmm(df_gwl, "Combined_Effect", share_re, vmax, fn)


def _mmm_re(df_gwl, share_re="current", vmax=100):
    def fn(df):
        df["RE_Effect"] = (df["cum_rl_ds_cf"] - df["cum_rl_ref"]) / df["cum_rl_ref"] * 100
    return _mmm(df_gwl, "RE_Effect", share_re, vmax, fn)


def _mmm_tas(df_gwl, share_re="current", vmax=200):
    def fn(df):
        df["TAS_Effect"] = (df["cum_rl_tas"] - df["cum_rl_ref"]) / df["cum_rl_ref"] * 100
    return _mmm(df_gwl, "TAS_Effect", share_re, vmax, fn)


def _mmm_absolute_days(df_gwl, share_re="current"):
    # Absolute change in cumulative residual load (GWL2 - GWL0.61), normalized
    # by each region's non-thermosensitive baseline demand (demand_bas from
    # _calculate_rl) -- units are "days of baseline demand" rather than percent.
    def fn(df):
        df["Absolute_Days"] = (df["cum_rl_gwl"] - df["cum_rl_ref"]) / df["demand_bas"]
    return _mmm(df_gwl, "Absolute_Days", share_re, vmax=None, compute_fn=fn)


def _mmm_supply_days(df_gwl, share_re="current"):
    # Isolated RE-supply driver (cum_rl_ds_cf - cum_rl_ref), same
    # days-of-baseline-demand normalization as _mmm_absolute_days.
    def fn(df):
        df["Supply_Days"] = (df["cum_rl_ds_cf"] - df["cum_rl_ref"]) / df["demand_bas"]
    return _mmm(df_gwl, "Supply_Days", share_re, vmax=None, compute_fn=fn)


def _mmm_demand_days(df_gwl, share_re="current"):
    # Isolated TAS-demand driver (cum_rl_tas - cum_rl_ref), same
    # days-of-baseline-demand normalization as _mmm_absolute_days.
    def fn(df):
        df["Demand_Days"] = (df["cum_rl_tas"] - df["cum_rl_ref"]) / df["demand_bas"]
    return _mmm(df_gwl, "Demand_Days", share_re, vmax=None, compute_fn=fn)


def _print_supply_demand_stats(df_gwl, gwl_label, share_re="current"):
    """Diagnostic across all polygons at a given GWL: how many regions have a
    positive vs. negative isolated RE-supply driver, and how the average
    magnitude of the supply driver compares to the average magnitude of the
    demand driver, both in days of baseline demand."""
    supply = (_mmm_supply_days(df_gwl, share_re)["Supply_Days"]
              .replace([np.inf, -np.inf], np.nan).dropna())
    demand = (_mmm_demand_days(df_gwl, share_re)["Demand_Days"]
              .replace([np.inf, -np.inf], np.nan).dropna())
    n_pos = int((supply > 0).sum())
    n_neg = int((supply < 0).sum())
    print(f"  [INFO] {gwl_label}: supply effect > 0 in {n_pos} regions, "
          f"< 0 in {n_neg} regions (out of {len(supply)})")
    if len(demand) and demand.abs().mean() > 0:
        ratio = supply.abs().mean() / demand.abs().mean()
        print(f"  [INFO] {gwl_label}: avg |supply effect| / avg |demand effect| "
              f"(days of baseline demand) = {ratio:.3f}")

    # Joint sign of supply and demand drivers, per region (aligned on poly_idx).
    joint = (_mmm_supply_days(df_gwl, share_re)
             .merge(_mmm_demand_days(df_gwl, share_re), on="poly_idx")
             .replace([np.inf, -np.inf], np.nan)
             .dropna(subset=["Supply_Days", "Demand_Days"]))
    s, d = joint["Supply_Days"], joint["Demand_Days"]
    n_joint = len(joint)
    print(f"  [INFO] {gwl_label}: supply > 0 & demand > 0 in {int(((s > 0) & (d > 0)).sum())}, "
          f"supply < 0 & demand < 0 in {int(((s < 0) & (d < 0)).sum())}, "
          f"supply > 0 & demand < 0 in {int(((s > 0) & (d < 0)).sum())}, "
          f"supply < 0 & demand > 0 in {int(((s < 0) & (d > 0)).sum())} "
          f"regions (out of {n_joint})")


def _save_fig(fig, path, dpi):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def _three_panel_map(df_gwl15, df_gwl2, df_gwl3,
                     shapefile_path, hatch_df, cmap, norm,
                     value_fn, value_col,
                     title_gwl2, title_gwl15, title_gwl3,
                     cbar_label, dpi):
    proj = ccrs.EqualEarth()
    # 1 large top + 2 smaller bottom -> width = 2x FIG_WIDTH_IN, height proportional
    fig_w = FIG_WIDTH_IN
    fig_h = fig_w * (8 / 14)
    fig   = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs    = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.0, wspace=0.02,
                             bottom=0.14, top=0.97)
    ax2   = fig.add_subplot(gs[0, :], projection=proj)
    ax15  = fig.add_subplot(gs[1, 0], projection=proj)
    ax3   = fig.add_subplot(gs[1, 1], projection=proj)
    for ax, df_gwl, title, letter, tfs in [
        (ax2,  df_gwl2,  title_gwl2,  "a", 7),
        (ax15, df_gwl15, title_gwl15, "b", 6),
        (ax3,  df_gwl3,  title_gwl3,  "c", 6),
    ]:
        gdf = _build_gdf(shapefile_path, value_fn(df_gwl, MAIN_MIX), hatch_df)
        _draw_map(ax, gdf, value_col, cmap, norm, hatch_df, title, letter,
                  title_fontsize=tfs)
    _add_colorbar(fig, cmap, norm, cbar_label, pos=(0.25, 0.05, 0.5, 0.020))
    return fig


# =============================================================================
# FIGURES 4-5 -- Main: GWL maps + dumbbell
# =============================================================================


def plot_main_gwl_maps_absolute(df_gwl15, df_gwl2, df_gwl3,
                                shapefile_path, hatch_df, output_dir,
                                dpi=300, share_re="current"):
    """Same 3-panel layout as plot_main_gwl_maps, but each panel plots the
    absolute change in cumulative residual load (GWL - GWL0.61) normalized
    by the region's non-thermosensitive baseline demand, so the anomaly
    reads in units of "days of baseline demand" instead of percent."""
    abs_vals = []
    for df_gwl, gwl_label in ((df_gwl15, "1.5°C"), (df_gwl2, "2°C"), (df_gwl3, "3°C")):
        eff = (_mmm_absolute_days(df_gwl, share_re)["Absolute_Days"]
               .replace([np.inf, -np.inf], np.nan).dropna())
        if len(eff):
            abs_vals.append(np.abs(eff.values))
        _print_supply_demand_stats(df_gwl, gwl_label, share_re)
    vmax_days = (max(1.0, np.ceil(np.nanpercentile(np.concatenate(abs_vals), 95)))
                 if abs_vals else 1.0)
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mcolors.TwoSlopeNorm(vmin=-vmax_days, vcenter=0, vmax=vmax_days)
    fig = _three_panel_map(
        df_gwl15, df_gwl2, df_gwl3, shapefile_path, hatch_df, cmap, norm,
        value_fn=_mmm_absolute_days, value_col="Absolute_Days",
        title_gwl2="2°C",
        title_gwl15="1.5°C",
        title_gwl3="3°C",
        cbar_label="SWBDs change compared to 0.61°C (days of baseline demand)", dpi=dpi,
    )
    _save_fig(fig, os.path.join(output_dir, "main", "fig4_main_gwl_maps_absolute_days.png"), dpi)


def plot_main_dumbbell_absolute(df_gwl2, shapefile_path, dpi=300, share_re="current",
                                output_dir=None):
    """Same layout as plot_main_dumbbell, but each effect is the absolute
    change in cumulative residual load (GWL2 - GWL0.61) normalized by the
    region's non-thermosensitive baseline demand, so it reads in units of
    "days of baseline demand" instead of percent."""
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
    df_long = (df_db[["label", "Temp_Effect", "RE_Effect"]]
               .melt(id_vars="label", var_name="Effect", value_name="Value")
               .dropna())
    df_long["Effect"] = df_long["Effect"].map(
        {"Temp_Effect": "Demand driver", "RE_Effect": "Supply driver"})
    pal = plt.get_cmap("PuOr")
    demand_color, re_color = pal(0.8), pal(0.2)
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
                ax.scatter(r["Temp_Effect"],     i, color=demand_color, s=30, zorder=5, alpha=0.9)
                ax.scatter(r["RE_Effect"],       i, color=re_color,     s=30, zorder=5, alpha=0.9)
                ax.scatter(r["Combined_Effect"], i, color="black",      s=30, zorder=6, alpha=0.9)
        except Exception as exc:
            print(f"  [WARN] Absolute dumbbell failed: {exc}")

    ax.axvline(0, color="black", lw=1.2, alpha=0.6, linestyle="--")
    finite_vals = df_long["Value"].replace([np.inf, -np.inf], np.nan).dropna()
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
               color="black", label="Combined effect", markersize=5),
    ], title="Multi-model mean effect",
       title_fontproperties={"weight": "bold", "size": 6},
       loc="lower right", fontsize=5)

    _save_fig(fig, os.path.join(output_dir, "main", "fig5_main_dumbbell_absolute_days.png"), dpi)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Figures 4 and 5 (SWBD changes by region and warming level)."
    )
    p.add_argument("--output_dir", default="../final_figs")
    p.add_argument("--dpi",        type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "main"), exist_ok=True)

    print("\n=== STEP 1: Agreement mask ===")
    hatch_df, _ = load_hatch_agg(
        PATHS["agreement_nc"], PATHS["shapefile"],
        agreement_aggregated_nc=PATHS["agreement_aggregated_nc"],
    )

    print("\n=== STEP 2: Main figures ===")
    csv_current = os.path.join(PATHS["out_dir"],
        f"rl_agg_adaptation_Annual_{MAIN_THR}_ren_pen_{MAIN_TOT_RE}_{MAIN_MIX}_v2.csv")
    if not os.path.exists(csv_current):
        raise SystemExit(f"{csv_current} missing -- run main_pipeline/make_rl_files.py first.")
    df_gwl15, df_gwl2, df_gwl3 = load_gwl_dfs(csv_current)
    plot_main_gwl_maps_absolute(df_gwl15, df_gwl2, df_gwl3,
                                PATHS["shapefile"], hatch_df,
                                args.output_dir, dpi=args.dpi, share_re=MAIN_MIX)
    plot_main_dumbbell_absolute(df_gwl2, PATHS["shapefile"], dpi=args.dpi,
                                share_re=MAIN_MIX, output_dir=args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()

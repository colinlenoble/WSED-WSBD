# -*- coding: utf-8 -*-
import os
import config
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
os.environ["ESMFMKFILE"]       = config.ESMFMKFILE_XENV
os.environ["MPLBACKEND"]       = "Agg"

import argparse
import glob
import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import xarray as xr
import xagg as xa
from xclim import sdba

import matplotlib
matplotlib.use("Agg")
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
    "out_dir":            config.PATH_PREPROCESSED + "agg_datasets/rl_out/",
    "ssp":                config.SSP,
    "reanalysis":         config.REANALYSIS,
}

MAIN_THR    = 0.99
MAIN_TOT_RE = 0.5
MAIN_MIX    = "current"

GWL_LIST    = ["GWL1-5", "GWL2", "GWL3"]
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
# DEMAND SENSITIVITY CONFIGURATIONS
# =============================================================================

@dataclass
class DemandConfig:
    thr_cold  : float = 12.5
    thr_hot   : float = 19.6
    coef_cold : float = 0.026
    coef_hot  : float = 0.035

    @property
    def tag(self):
        return (f"tc{self.thr_cold}_th{self.thr_hot}"
                f"_cc{self.coef_cold}_ch{self.coef_hot}")

    def __str__(self):
        return (f"cold>{self.thr_cold}C x{self.coef_cold} | "
                f"hot>{self.thr_hot}C x{self.coef_hot}")


DEFAULT_DEMAND = DemandConfig()

DEMAND_CONFIGS = {
    "default"  : DemandConfig(),
    "cold_low" : DemandConfig(thr_cold=10.5),
    "cold_high": DemandConfig(thr_cold=14.5),
    "hot_low"  : DemandConfig(thr_hot=17.6),
    "hot_high" : DemandConfig(thr_hot=21.6),
    "coef_low" : DemandConfig(coef_cold=0.021, coef_hot=0.028),
    "coef_high": DemandConfig(coef_cold=0.031, coef_hot=0.042),
    "strict"   : DemandConfig(thr_cold=10.5, thr_hot=21.6,
                               coef_cold=0.021, coef_hot=0.028),
    "sensitive": DemandConfig(thr_cold=14.5, thr_hot=17.6,
                               coef_cold=0.031, coef_hot=0.042),
}


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute RL CSVs and generate all main / supplementary figures."
    )
    p.add_argument("--tot_re",     nargs="+", type=float, default=[0.25, 0.5, 0.75])
    p.add_argument("--thr",        nargs="+", type=float, default=[0.95, 0.99, 0.995])
    p.add_argument("--mix",        nargs="+", default=["current", "future"])
    p.add_argument("--output_dir", default="../final_figs")
    p.add_argument("--dpi",        type=int, default=300)
    p.add_argument("--skip_rl",    action="store_true", default=False,
                   help="Skip RL computation and CSV generation, only make figures.")
    return p.parse_args()


# =============================================================================
# CORE RL COMPUTATION  (unchanged)
# =============================================================================

def _calculate_rl(tas, ds_cf, ds_cf_mean, thr, period, tot_re,
                  threshold=None, demand_bas=None,
                  demand_cfg: DemandConfig = None, return_daily=False):
    if demand_cfg is None:
        demand_cfg = DEFAULT_DEMAND
    demand_temp = xr.where(
        tas < demand_cfg.thr_cold,
        (demand_cfg.thr_cold - tas) * demand_cfg.coef_cold,
        xr.where(tas > demand_cfg.thr_hot,
                 (tas - demand_cfg.thr_hot) * demand_cfg.coef_hot, 0.0),
    )
    if demand_bas is None:
        demand_bas = ds_cf_mean / (1.0 + demand_temp.mean(dim="time"))
    demand = demand_bas * (1.0 + demand_temp)
    demand["time"] = pd.to_datetime(demand["time"].dt.strftime("%Y-%m-%d").values)
    ds_cf["time"]    = pd.to_datetime(ds_cf["time"].dt.strftime("%Y-%m-%d").values)
    months = {"Annual": range(1, 13), "JJA": [6, 7, 8], "DJF": [12, 1, 2]}[period]
    rl = (demand - tot_re * ds_cf).sel(
        time=(demand - tot_re * ds_cf)["time.month"].isin(months))
    if threshold is None:
        threshold = rl.quantile(thr, dim="time")
    exceeds = rl > threshold
    cum_rl = xr.where(exceeds, rl - threshold, 0.0).sum(dim="time")
    if return_daily:
        # Day-level SWBD exceedance field, kept only for callers that need
        # individual-event duration (e.g. fig_duration_distribution_swed_swbd.py)
        # -- cum_rl above already discards this by summing over time, which
        # is all every other caller in this module has ever needed.
        return cum_rl, threshold, demand_bas, exceeds.astype(int)
    return cum_rl, threshold, demand_bas


def _align_histogram(source, target, kind="+", nquantiles=100):
    source.attrs["units"] = "K"
    target.attrs["units"] = "K"
    adj = sdba.EmpiricalQuantileMapping.train(
        ref=target, hist=source, kind=kind, nquantiles=nquantiles)
    return adj.adjust(source)


def _load_data(GCM, run, ssp, level, reanalysis, suffix_shp,
               path_preprocessed, df_share, mix="current"):
    base   = os.path.join(path_preprocessed, GCM)
    suffix = f"_{GCM}_{ssp}_{run}_{level}_{reanalysis}_{suffix_shp}.nc"
    tas = xr.open_dataset(os.path.join(base, f"tas_pop_agg{suffix}"))["tas"] - 273.15
    wcf = xr.open_dataset(os.path.join(base, f"wcf_agg{suffix}"))["wcf"]
    scf = xr.open_dataset(os.path.join(base, f"scf_agg{suffix}"))["scf"]
    cur_share = xr.DataArray(df_share["current_ratio"].values,
                             coords=[df_share["poly_idx"].values], dims=["poly_idx"])
    ds_cf_mean = (cur_share * scf + (1 - cur_share) * wcf).mean(dim="time")
    if mix == "future" and "future_ratio" in df_share.columns:
        fut_share = xr.DataArray(df_share["future_ratio"].values,
                                 coords=[df_share["poly_idx"].values], dims=["poly_idx"])
        ds_cf = fut_share * scf + (1 - fut_share) * wcf
    else:
        ds_cf = cur_share * scf + (1 - cur_share) * wcf
    return tas, ds_cf, ds_cf_mean


def compute_rl_one_gcm(GCM, run, ssp, gwl, thr, tot_re, mix,
                       path_preprocessed, df_share, reanalysis,
                       period="Annual", suffix_shp="v1",
                       demand_cfg: DemandConfig = None):
    if demand_cfg is None:
        demand_cfg = DEFAULT_DEMAND
    gwl_ref = "GWL0-61"
    dtas,     dds_cf,     _             = _load_data(GCM, run, ssp, gwl,     reanalysis,
                                                   suffix_shp, path_preprocessed, df_share, mix)
    dtas_ref, dds_cf_ref, dds_cf_ref_mean = _load_data(GCM, run, ssp, gwl_ref, reanalysis,
                                                   suffix_shp, path_preprocessed, df_share, mix)
    rows = []

    def _append(data, gwl_ds_cf_val, gwl_tas_val):
        rows.append(pd.DataFrame({
            "poly_idx": df_share["poly_idx"].values,
            "GCM": GCM, "run": run, "rl_cum": data.values,
            "gwl_ds_cf": gwl_ds_cf_val, "gwl_tas": gwl_tas_val,
            "period": period, "tot_re": tot_re, "share_re": mix,
            "demand_bas": demand_bas.values,
        }))

    cum_ref, rl_thr, demand_bas = _calculate_rl(
        dtas_ref, dds_cf_ref, dds_cf_ref_mean, thr, period, tot_re, demand_cfg=demand_cfg)
    _append(cum_ref, gwl_ref, gwl_ref)

    cum_gwl, _, _ = _calculate_rl(
        dtas, dds_cf, dds_cf_ref_mean, thr, period, tot_re,
        threshold=rl_thr, demand_bas=demand_bas, demand_cfg=demand_cfg)
    _append(cum_gwl, gwl, gwl)

    cum_tas, _, _ = _calculate_rl(
        _align_histogram(dtas_ref, dtas), dds_cf_ref, dds_cf_ref_mean,
        thr, period, tot_re,
        threshold=rl_thr, demand_bas=demand_bas, demand_cfg=demand_cfg)
    _append(cum_tas, gwl_ref, gwl)

    cum_ds_cf, _, _ = _calculate_rl(
        _align_histogram(dtas, dtas_ref), dds_cf, dds_cf_ref_mean,
        thr, period, tot_re,
        threshold=rl_thr, demand_bas=demand_bas, demand_cfg=demand_cfg)
    _append(cum_ds_cf, gwl, gwl_ref)

    return pd.concat(rows, ignore_index=True)


# =============================================================================
# PIPELINE HELPERS  (unchanged)
# =============================================================================

EXCLUDED_RUNS = {("EC-Earth3-Veg-LR", "r3i1p1f1"), ("NorESM2-MM", "r2i1p1f1")}

def _iter_gcm_runs(path_preprocessed, ssp, reanalysis, suffix_shp):
    pattern = os.path.join(path_preprocessed, "*",
                           f"wcf_agg_*GWL1-5*_{reanalysis}_{suffix_shp}.nc")
    for fpath in glob.glob(pattern):
        fname = os.path.basename(fpath)
        parts = fname.replace(".nc", "").split("_")
        try:
            idx = parts.index(ssp)
            gcm, run = "_".join(parts[2:idx]), parts[idx + 1]
            if (gcm, run) in EXCLUDED_RUNS:
                continue
            yield gcm, run
        except (ValueError, IndexError):
            print(f"  [WARN] Cannot parse: {fname}")


def _files_ready(GCM, run, ssp, gwl, reanalysis, suffix_shp, path_preprocessed):
    base = os.path.join(path_preprocessed, GCM)
    return all(os.path.exists(
        os.path.join(base, f"tas_pop_agg_{GCM}_{ssp}_{run}_{lv}_{reanalysis}_{suffix_shp}.nc"))
        for lv in (gwl, "GWL0-61"))


def _compute_and_save(thr, tot_re, mix, demand_cfg,
                      path_preprocessed, df_share, ssp, reanalysis, out_dir,
                      period, suffix_shp, demand_tag=None):
    if demand_tag is not None:
        fname = (f"rl_agg_adaptation_{period}_{thr}"
                 f"_ren_pen_{tot_re}_{mix}_demand-{demand_tag}_v2.csv")
    else:
        fname = f"rl_agg_adaptation_{period}_{thr}_ren_pen_{tot_re}_{mix}_v2.csv"
    out_path = os.path.join(out_dir, fname)
    required_cols = {"poly_idx", "GCM", "run", "rl_cum",
                     "gwl_ds_cf", "gwl_tas", "period", "tot_re", "share_re",
                     "demand_bas"}
    if os.path.exists(out_path):
        existing_cols = set(pd.read_csv(out_path, index_col=0, nrows=0).columns)
        missing = required_cols - existing_cols
        if not missing:
            print(f"  [SKIP] {fname}")
            return
        print(f"  [STALE] {fname} is missing columns {sorted(missing)} "
              "(schema changed since it was written) -- recomputing.")
    print(f"\n  Computing: {fname}")
    df_final = []
    for GCM, run in _iter_gcm_runs(path_preprocessed, ssp, reanalysis, suffix_shp):
        print(f"    {GCM}  {run}")
        try:
            for gwl in GWL_LIST:
                if not _files_ready(GCM, run, ssp, gwl, reanalysis, suffix_shp, path_preprocessed):
                    print(f"      Missing files for {gwl}, skipping.")
                    continue
                df_gcm = compute_rl_one_gcm(
                    GCM, run, ssp, gwl, thr, tot_re, mix,
                    path_preprocessed, df_share, reanalysis,
                    period=period, suffix_shp=suffix_shp, demand_cfg=demand_cfg)
                df_final.append(df_gcm)
        except Exception as exc:
            print(f"      ERROR {GCM} {run}: {exc}")
        gc.collect()
    if df_final:
        pd.concat(df_final, ignore_index=True).to_csv(out_path)
        print(f"  Saved: {out_path}")
    else:
        print(f"  [WARN] No data for {fname}")


def run_rl_pipeline(tot_re_list, thr_list, mix_list,
                    path_preprocessed, df_share, ssp, reanalysis, out_dir,
                    period="Annual", suffix_shp="v1"):
    for thr in thr_list:
        for tot_re in tot_re_list:
            for mix in mix_list:
                _compute_and_save(thr, tot_re, mix, demand_cfg=DEFAULT_DEMAND,
                                  path_preprocessed=path_preprocessed,
                                  df_share=df_share, ssp=ssp, reanalysis=reanalysis,
                                  out_dir=out_dir, period=period, suffix_shp=suffix_shp,
                                  demand_tag=None)


def run_demand_sensitivity_pipeline(path_preprocessed, df_share, ssp, reanalysis, out_dir,
                                    period="Annual", suffix_shp="v1"):
    for demand_name, demand_cfg in DEMAND_CONFIGS.items():
        _compute_and_save(thr=MAIN_THR, tot_re=MAIN_TOT_RE, mix=MAIN_MIX,
                          demand_cfg=demand_cfg,
                          path_preprocessed=path_preprocessed,
                          df_share=df_share, ssp=ssp, reanalysis=reanalysis,
                          out_dir=out_dir, period=period, suffix_shp=suffix_shp,
                          demand_tag=demand_name)


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
            ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor="black", edgecolor="black",
                              linewidth=0.0, zorder=4)
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


def _print_combined_effect_direction_stats(df_gwl, gwl_label, share_re="current"):
    """Diagnostic across all polygons at a given GWL: how many regions have a
    positive (increasing) vs. negative (decreasing) flat multi-model-mean
    Combined_Effect -- SWBDs change relative to 0.61 C, same metric/weighting
    as plot_main_gwl_maps -- out of how many regions have a finite value."""
    combined = (_mmm_combined(df_gwl, share_re, vmax=None)["Combined_Effect"]
                .replace([np.inf, -np.inf], np.nan).dropna())
    n_total = len(combined)
    n_up    = int((combined > 0).sum())
    n_down  = int((combined < 0).sum())
    print(f"  [INFO] {gwl_label}: multi-model mean SWBDs increasing in "
          f"{n_up}/{n_total} regions, decreasing in {n_down}/{n_total}")


# =============================================================================
# INVERSE-WASSERSTEIN-DISTANCE POLYGON WEIGHTING
#
# Aggregated-domain (poly_idx) twin of fig3.py's per-pixel inverse-W2
# weighting (_build_wasserstein_pixel_weight): instead of the flat multi-model
# mean used by _mmm() (equal weight per GCM, split evenly across its runs),
# a realization whose bootstrap trend distribution is closer to ERA5's own at
# a given polygon counts for more there. Source data is
# trend_sev_eval_wasserstein.py's wasserstein_empirical_agg() output
# (config.WASSERSTEIN_AGGREGATED_NC_PATH): dims (realization, poly_idx),
# variables w2_distance/w2_normalized, per-realization GCM/run values.
# =============================================================================

def _load_wasserstein_agg(wasserstein_path=None):
    """Open config.WASSERSTEIN_AGGREGATED_NC_PATH (or an override). Returns
    None (with a warning) if the file is missing, so callers can skip the
    inverse-W2-weighted companion figures gracefully rather than erroring."""
    path = wasserstein_path or config.WASSERSTEIN_AGGREGATED_NC_PATH
    if not os.path.exists(path):
        print(f"  [WARN] Wasserstein aggregated file not found: {path} "
              "-- inverse-W2-weighted companion figures skipped.")
        return None
    return xr.open_dataset(path)


def _wasserstein_weight_table(ds_wasserstein, var="w2_normalized", eps=1e-3):
    """
    Long-format (GCM, run, poly_idx) -> weight table combining the usual
    per-realization 1/n_gcm weight (de-duplicating multi-run GCMs, same
    convention as fig3.py's add_severity_and_weights/
    _build_wasserstein_pixel_weight) with the per-polygon inverse of that
    realization's normalized empirical Wasserstein trend distance to ERA5
    (ds_wasserstein's w2_normalized). eps floors w2_normalized so a
    near-zero distance cannot make a single realization dominate a
    polygon's weighted mean. Positional indexing (not a 'realization'
    coordinate) matches fig3.py's own handling of this dataset.
    """
    gcms = np.asarray(ds_wasserstein["GCM"].values).astype(str)
    runs = np.asarray(ds_wasserstein["run"].values).astype(str)
    wcount = pd.Series(gcms).value_counts()
    base_w = np.array([1.0 / wcount[g] / wcount.size for g in gcms])

    w2     = ds_wasserstein[var].values                     # (realization, poly_idx)
    inv_w2 = 1.0 / np.clip(w2, eps, None)
    weight = inv_w2 * base_w[:, None]                        # (realization, poly_idx)

    poly_idx     = ds_wasserstein["poly_idx"].values
    n_real, n_poly = weight.shape
    return pd.DataFrame({
        "GCM":       np.repeat(gcms, n_poly),
        "run":       np.repeat(runs, n_poly),
        "poly_idx":  np.tile(poly_idx, n_real),
        "w2_weight": weight.ravel(),
    })


def _mmm_wasserstein(df_gwl, effect_col, share_re, vmax, compute_fn, w2_table):
    """Inverse-W2-weighted twin of _mmm(): per-polygon weighted average of
    effect_col over (GCM, run) rows, using w2_table's weight instead of a
    flat per-GCM mean. Rows with no matching (GCM, run, poly_idx) weight are
    excluded, same as fig3.py's _build_wasserstein_pixel_weight."""
    df = df_gwl[df_gwl["share_re"] == share_re].copy()
    compute_fn(df)
    df["GCM"] = df["GCM"].astype(str)
    df["run"] = df["run"].astype(str)
    merged = df.merge(w2_table, on=["GCM", "run", "poly_idx"], how="inner")
    merged = merged[np.isfinite(merged[effect_col]) & (merged["w2_weight"] > 0)]
    if merged.empty:
        return pd.DataFrame(columns=["poly_idx", effect_col])
    merged["_wsum"] = merged[effect_col] * merged["w2_weight"]
    grp = merged.groupby("poly_idx").agg(_wsum=("_wsum", "sum"),
                                          _wtot=("w2_weight", "sum"))
    out = (grp["_wsum"] / grp["_wtot"]).rename(effect_col).reset_index()
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
# FIGURE 1 -- Main: GWL maps + dumbbell
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
    _save_fig(fig, os.path.join(output_dir, "main", "fig_main_gwl_maps_absolute_days.png"), dpi)


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
    Two-panel supplementary companion to plot_main_gwl_maps, at GWL2 (2 deg C)
    only. Aggregated (poly_idx) twin of fig3.py's
    plot_gwl_valuebyalpha_wasserstein, using the same SWBDs relative-change (%)
    metric as plot_main_gwl_maps (Combined_Effect) instead of fig3's pixel-level
    value-by-alpha map:
      a) average SWBDs change vs 0.61 deg C, weighted per polygon by each
         realization's inverse normalized Wasserstein trend distance to ERA5 at
         that polygon (_mmm_combined_wasserstein/_wasserstein_weight_table),
         instead of the flat multi-model mean plot_main_gwl_maps uses. Same
         colour scale as plot_main_gwl_maps so the two are directly comparable.
      b) the difference this reweighting makes: (inverse-W2-weighted change)
         minus (multi-model-mean change), in percentage points -- diverging
         blue/orange (cmo.cm.balance), matching fig3's diff panel. Agreement
         hatching is shown in (a) but, as in fig3's reference figure, omitted
         in (b) since it describes trend agreement, not this reweighting.
    """
    cmap, norm = _make_cmap(vmin=-100, vmax=800)

    df_flat = _mmm_combined(df_gwl2, share_re=share_re, vmax=None)
    df_w2   = _mmm_combined_wasserstein(df_gwl2, w2_table, share_re=share_re, vmax=None)

    diff = df_flat.merge(df_w2, on="poly_idx", suffixes=("_flat", "_w2"))
    diff["Diff_Effect"] = diff["Combined_Effect_w2"] - diff["Combined_Effect_flat"]
    diff = diff[["poly_idx", "Diff_Effect"]]

    finite = diff["Diff_Effect"].replace([np.inf, -np.inf], np.nan).dropna()
    diff_vmax = max(1.0, float(np.nanpercentile(np.abs(finite), 98))) if len(finite) else 1.0
    diff_cmap = cmo.cm.balance
    diff_norm = mcolors.TwoSlopeNorm(vmin=-diff_vmax, vcenter=0, vmax=diff_vmax)

    hatch_df_none = hatch_df.copy()
    hatch_df_none["var"] = np.nan

    proj  = ccrs.EqualEarth()
    fig_w = FIG_WIDTH_IN
    fig_h = fig_w * (15 / 14)
    fig   = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs    = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.55,
                             bottom=0.10, top=0.95)

    ax_a  = fig.add_subplot(gs[0, 0], projection=proj)
    gdf_a = _build_gdf(shapefile_path, df_w2, hatch_df)
    _draw_map(ax_a, gdf_a, "Combined_Effect", cmap, norm, hatch_df,
              "Inverse-Wasserstein-weighted SWBDs change under 2.0°C warming",
              "a", title_fontsize=7)
    _add_colorbar(fig, cmap, norm, "SWBDs change compared to 0.61°C (%)",
                  pos=(0.25, 0.545, 0.5, 0.016))

    ax_b  = fig.add_subplot(gs[1, 0], projection=proj)
    gdf_b = _build_gdf(shapefile_path, diff, hatch_df_none)
    _draw_map(ax_b, gdf_b, "Diff_Effect", diff_cmap, diff_norm, hatch_df_none,
              "Difference vs. multi-model mean", "b", title_fontsize=7)
    _add_colorbar(fig, diff_cmap, diff_norm,
                  "Difference in SWBDs change,\ninverse-W2 minus multi-model mean (pp)",
                  pos=(0.25, 0.035, 0.5, 0.016))

    _save_fig(fig, os.path.join(output_dir, "supp",
                                "suppfig_main_gwl_maps_GWL2_wasserstein.png"), dpi)


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

    _save_fig(fig, os.path.join(output_dir, "main", "fig_main_dumbbell_absolute_days.png"), dpi)


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
# FIGURE 2 -- Supp: three-panel GWL combined-effect maps
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
# FIGURE 3 -- Supp: isolated RE-supply effect
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
# FIGURE 4 -- Supp: isolated TAS-demand effect
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
# FIGURE 5 -- Supp: driver-decomposition ratio
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
            ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor="black", edgecolor="black",
                              linewidth=0.0, zorder=4)
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
        Patch(facecolor="black", edgecolor="black", label="Low model-agreement"),
    ], loc="lower right", bbox_to_anchor=(1.0, 0.0), bbox_transform=ax.transAxes,
       fontsize=5, framealpha=0.85, handlelength=1.0, handletextpad=0.4, borderpad=0.4)

    _save_fig(fig, os.path.join(output_dir, "supp",
                                f"suppfig_decomp_{tag}.png"), dpi)


# =============================================================================
# FIGURE 6 -- Supp: demand-sensitivity grid
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
            Patch(facecolor="black", edgecolor="black",
                  label="Low model agreement"),
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
# FIGURE 6b -- Supp: demand-sensitivity grid, absolute (days of baseline demand)
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
    diff_colors = ["#08519c", "#f7f7f7", "#d94801"]
    cmap_diff = LinearSegmentedColormap.from_list("diff_cmap_days", diff_colors, N=300)

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
            Patch(facecolor="black", edgecolor="black",
                  label="Low model agreement"),
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
                                "suppfig_demand_sensitivity_absolute_days.png"), dpi)


# =============================================================================
# FIGURE 7 -- Supp: mix effect vs GWL effect scatter + categorical map
# =============================================================================

def plot_supp_mix_effect(df_gwl2_curr, df_gwl2_fut,
                         shapefile_path, idxs_to_hatch,
                         output_dir, dpi=300):
    df_share = pd.read_csv(PATHS["df_share_csv"])
    real_current_share = xr.DataArray(
        df_share["current_ratio"].values,
        coords=[df_share["poly_idx"].values], dims=["poly_idx"])
    if "future_ratio" in df_share.columns:
        real_future_share = xr.DataArray(
            df_share["future_ratio"].values,
            coords=[df_share["poly_idx"].values], dims=["poly_idx"])
    else:
        real_future_share = real_current_share.copy()

    df_c = (df_gwl2_curr[df_gwl2_curr["share_re"] == "current"]
            .groupby(["poly_idx", "GCM"], as_index=False)
            .agg({"cum_rl_ref": "mean", "cum_rl_gwl": "mean",
                  "cum_rl_tas": "mean", "cum_rl_ds_cf": "mean"})
            .groupby("poly_idx", as_index=False)
            .agg({"cum_rl_ref": "mean", "cum_rl_gwl": "mean",
                  "cum_rl_tas": "mean", "cum_rl_ds_cf": "mean"}))
    df_f = (df_gwl2_fut[df_gwl2_fut["share_re"] == "future"]
            .groupby(["poly_idx", "GCM"], as_index=False).agg({"cum_rl_ref": "mean"})
            .groupby("poly_idx", as_index=False).agg({"cum_rl_ref": "mean"})
            .rename(columns={"cum_rl_ref": "cum_rl_ref_future"}))

    df_mix = df_c.merge(df_f, on="poly_idx", how="left")
    df_mix["mix_effect"] = (
        (df_mix["cum_rl_ref_future"] - df_mix["cum_rl_ref"]) / df_mix["cum_rl_ref"] * 100)
    df_mix["gwl_effect"] = (
        (df_mix["cum_rl_gwl"] - df_mix["cum_rl_ref"]) / df_mix["cum_rl_ref"] * 100)

    poly_arr = df_mix["poly_idx"].values
    cur_vals = real_current_share.sel(poly_idx=xr.DataArray(poly_arr, dims="z")).values
    fut_vals = real_future_share.sel(poly_idx=xr.DataArray(poly_arr, dims="z")).values
    color    = np.where(cur_vals < fut_vals, "blue",
                        np.where(cur_vals > fut_vals, "orange", "#8B4513"))
    df_mix["no_mix_change"] = (cur_vals == fut_vals)

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
                c=color, marker="x", s=4, linewidths=0.4)
    ax1.set_xlabel("Mix effect on SWBD (%)", fontsize=5)
    ax1.set_ylabel("Global warming effect on SWBD (%)", fontsize=5)
    ax1.tick_params(labelsize=5)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xscale("symlog", linthresh=10)
    ax1.set_yscale("symlog", linthresh=10)
    ax1.legend(
        handles=[
            Line2D([0], [0], marker="x", linestyle="None",
                   markeredgecolor="blue",    color="blue",    label="Increase in wind share", markersize=3),
            Line2D([0], [0], marker="x", linestyle="None",
                   markeredgecolor="orange",  color="orange",  label="Increase in solar share", markersize=3),
            Line2D([0], [0], marker="x", linestyle="None",
                   markeredgecolor="#8B4513", color="#8B4513", label="No change in mix", markersize=3),
        ],
        title="Renewable mix change", loc="upper center", frameon=True,
        fontsize=4, title_fontsize=4,
        bbox_to_anchor=(0.5, -0.25), bbox_transform=ax1.transAxes,
        handlelength=1.0, handletextpad=0.4, borderpad=0.4,
    )

    def _assign_color(row):
        if row["no_mix_change"]:                                 return "#8B4513"
        elif row["mix_effect"] > 0 and row["gwl_effect"] > 0:   return "#d73027"
        elif row["mix_effect"] < 0 and row["gwl_effect"] <= 0:  return "#4575b4"
        elif row["mix_effect"] < 0 and row["gwl_effect"] > 0:   return "#fee090"
        else:                                                     return "#91bfdb"

    df_mix["map_color"] = df_mix.apply(_assign_color, axis=1)
    gdf_diff = gpd.read_file(shapefile_path)
    gdf_diff["poly_idx"] = gdf_diff.index
    gdf_diff = gdf_diff.cx[:, MAP_LAT_SOUTH:MAP_LAT_NORTH]
    color_map = dict(zip(df_mix["poly_idx"], df_mix["map_color"]))
    no_data_mask = gdf_diff["poly_idx"].map(color_map).isna()
    gdf_diff["color"] = gdf_diff["poly_idx"].map(color_map).fillna("white")
    no_data_set = set(gdf_diff[no_data_mask]["poly_idx"].tolist())
    hatch_set   = set(idxs_to_hatch.tolist() if hasattr(idxs_to_hatch, "tolist") else list(idxs_to_hatch))

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
            ax2.add_geometries([geom], crs=ccrs.PlateCarree(),
                               facecolor="black", edgecolor="black",
                               linewidth=0.0, zorder=4)

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
            Patch(facecolor="#d73027", edgecolor="none", label="Both increase SWBDs"),
            Patch(facecolor="#4575b4", edgecolor="none", label="Both decrease SWBDs"),
            Patch(facecolor="#fee090", edgecolor="none", label="Warming increases SWBDs, Mix decreases SWBDs"),
            Patch(facecolor="#91bfdb", edgecolor="none", label="Warming decreases SWBDs, Mix increases SWBDs"),
            Patch(facecolor="#8B4513", edgecolor="none", label="No change in mix"),
            Patch(facecolor="white",   edgecolor="black", hatch="\\" * 10, label="No RE capacities"),
            Patch(facecolor="black",    edgecolor="black",
                  label="Low model agreement"),
        ],
        loc="upper center", fontsize=4, ncol=2,
        bbox_to_anchor=(0.5, -0.08), bbox_transform=ax2.transAxes,
        handlelength=1.0, handletextpad=0.4, borderpad=0.4,
    )

    _save_fig(fig, os.path.join(output_dir, "supp",
                                "suppfig_mix_gwl_effects.png"), dpi)


# =============================================================================
# FIGURE 8 -- Supp: inter-model uncertainty decomposition
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
            ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor="black", edgecolor="black",
                              linewidth=0.0, zorder=4)
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
        Patch(facecolor="black", edgecolor="black", label="Low model-agreement"),
    ], loc="lower right", bbox_to_anchor=(1.0, 0.0), bbox_transform=ax.transAxes,
       fontsize=5, framealpha=0.85, handlelength=1.0, handletextpad=0.4, borderpad=0.4)

    _save_fig(fig, os.path.join(output_dir, "supp",
                                f"suppfig_uncertainty_decomp_{tag}.png"), dpi)


# =============================================================================
# FIGURE 9 -- Supp: absolute inter-model spread in RE supply effect
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
            ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor="black", edgecolor="black",
                              linewidth=0.0, zorder=4)
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
        Patch(facecolor="black", edgecolor="black", label="Low model-agreement"),
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
        Patch(facecolor="black", edgecolor="black",
              label="Low model agreement"),
    ]
    last_panel = fig.axes[8]
    last_panel.legend(handles=legend_handles, loc="upper center",
                      bbox_to_anchor=(0.5, -0.08), bbox_transform=last_panel.transAxes,
                      fontsize=5, framealpha=0.85, handlelength=1.0, handletextpad=0.4,
                      borderpad=0.4)
    _save_fig(fig, os.path.join(output_dir, "supp",
                                f"suppfig_re_share_effect.png"), dpi)


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
    fig.text(0.5, 0.970, "Multi-model mean, current mix, threshold = 0.99",
             ha="center", va="top", fontsize=6, style="italic", color="#444444")

    legend_handles = [
        Patch(facecolor="white", edgecolor="black", hatch="\\" * 10, label="No solar-wind capacities"),
        Patch(facecolor="black", edgecolor="black",
              label="Low model agreement"),
    ]
    last_panel = fig.axes[8]
    last_panel.legend(handles=legend_handles, loc="upper center",
                      bbox_to_anchor=(0.5, -0.08), bbox_transform=last_panel.transAxes,
                      fontsize=5, framealpha=0.85, handlelength=1.0, handletextpad=0.4,
                      borderpad=0.4)
    _save_fig(fig, os.path.join(output_dir, "supp",
                                f"suppfig_re_share_effect_absolute_days.png"), dpi)


# =============================================================================
# FIGURE 10 -- Supp: combined 2x2 driver effects at GWL 2.0°C
# =============================================================================

def _driver_ratio_dfs(df_gwl2, share_re):
    """Per-region supply share of (a) the absolute multi-model-mean driver
    effect and (b) the inter-model spread. 0 = demand dominates, 1 = supply."""
    df = df_gwl2[df_gwl2["share_re"] == share_re].copy()
    df["RE_Effect"]   = df["cum_rl_ds_cf"] - df["cum_rl_ref"]
    df["Temp_Effect"] = df["cum_rl_gwl"] - df["cum_rl_ref"]
    per_gcm = (df[["poly_idx", "GCM", "RE_Effect", "Temp_Effect"]]
               .groupby(["GCM", "poly_idx"]).mean().reset_index())

    mean = per_gcm.groupby("poly_idx")[["RE_Effect", "Temp_Effect"]].mean().abs()
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
        Patch(facecolor="black", edgecolor="black", label="Low model agreement"),
    ], ncol=2, loc="center", bbox_to_anchor=(0.5, 0.09 / fig_h), bbox_transform=fig.transFigure,
       fontsize=5, framealpha=0.85, handlelength=1.0, handletextpad=0.4, borderpad=0.4)

    _save_fig(fig, out_path, dpi)


def plot_supp_combined_driver_effects(df_gwl2, shapefile_path, hatch_df,
                                      output_dir, share_re="current", dpi=300):
    _driver_effects_figure(
        _mmm_re(df_gwl2, share_re, vmax=None), "RE_Effect",
        _mmm_tas(df_gwl2, share_re, vmax=None), "TAS_Effect",
        "Effect on SWBDs (%)",
        df_gwl2, shapefile_path, hatch_df, share_re,
        os.path.join(output_dir, "supp", "suppfig_combined_driver_effects.png"), dpi)


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
                     "suppfig_combined_driver_effects_absolute_days.png"), dpi)


# =============================================================================
# MAIN  (unchanged logic)
# =============================================================================
def main():
    args = parse_args()
    for sub in ("main", "supp"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)
    ssp        = PATHS["ssp"]
    reanalysis = PATHS["reanalysis"]
    df_share   = pd.read_csv(PATHS["df_share_csv"])

    if not args.skip_rl:
        print("\n=== STEP 1a: RL CSVs -- tot_re sensitivity ===")
        run_rl_pipeline([0.25, 0.5, 0.75], [0.99], ["current"],
                        PATHS["path_preprocessed"], df_share, ssp, reanalysis, PATHS["out_dir"])
        print("\n=== STEP 1b: RL CSVs -- mix-effect baseline ===")
        run_rl_pipeline([0.5], [0.99], ["future"],
                        PATHS["path_preprocessed"], df_share, ssp, reanalysis, PATHS["out_dir"])
        print("\n=== STEP 1c: Demand-sensitivity CSVs ===")
        run_demand_sensitivity_pipeline(
            PATHS["path_preprocessed"], df_share, ssp, reanalysis, PATHS["out_dir"])
        print("\n=== STEP 1d: RL CSVs -- thr sensitivity ===")
        run_rl_pipeline([0.5], [0.95, 0.995], ["current"],
                        PATHS["path_preprocessed"], df_share, ssp, reanalysis, PATHS["out_dir"])
    else:
        print("\n[SKIP] RL computation skipped (--skip_rl).")

    print("\n=== STEP 2: Agreement mask ===")
    hatch_df, idxs_to_hatch = load_hatch_agg(
        PATHS["agreement_nc"], PATHS["shapefile"],
        agreement_aggregated_nc=PATHS["agreement_aggregated_nc"],
    )

    print("\n=== STEP 2b: Wasserstein aggregated weights ===")
    ds_wasserstein_agg = _load_wasserstein_agg(PATHS["wasserstein_aggregated_nc"])
    w2_table = (_wasserstein_weight_table(ds_wasserstein_agg)
                if ds_wasserstein_agg is not None else None)

    print("\n=== STEP 3: Main figure ===")
    csv_current = os.path.join(PATHS["out_dir"],
        f"rl_agg_adaptation_Annual_{MAIN_THR}_ren_pen_{MAIN_TOT_RE}_{MAIN_MIX}_v2.csv")
    if os.path.exists(csv_current):
        df_gwl15, df_gwl2, df_gwl3 = load_gwl_dfs(csv_current)
        plot_main_gwl_maps(df_gwl15, df_gwl2, df_gwl3,
                           PATHS["shapefile"], hatch_df,
                           args.output_dir, dpi=args.dpi, share_re=MAIN_MIX)
        plot_main_gwl_maps_absolute(df_gwl15, df_gwl2, df_gwl3,
                                    PATHS["shapefile"], hatch_df,
                                    args.output_dir, dpi=args.dpi, share_re=MAIN_MIX)
        plot_main_dumbbell(df_gwl2, PATHS["shapefile"], dpi=args.dpi,
                           share_re=MAIN_MIX, output_dir=args.output_dir)
        plot_main_dumbbell_absolute(df_gwl2, PATHS["shapefile"], dpi=args.dpi,
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

    if ds_wasserstein_agg is not None:
        ds_wasserstein_agg.close()

    print("\n=== STEP 4: Current vs Future mix comparison ===")
    csv_future = os.path.join(PATHS["out_dir"],
        f"rl_agg_adaptation_Annual_{MAIN_THR}_ren_pen_{MAIN_TOT_RE}_future_v2.csv")
    if os.path.exists(csv_current) and os.path.exists(csv_future):
        df_gwl15_curr, df_gwl2_curr, _ = load_gwl_dfs(csv_current)
        df_gwl15_fut,  df_gwl2_fut,  _ = load_gwl_dfs(csv_future)
        plot_supp_mix_effect(df_gwl2_curr, df_gwl2_fut,
                             PATHS["shapefile"], idxs_to_hatch,
                             args.output_dir, dpi=args.dpi)

    print("\n=== STEP 5: Demand-sensitivity figure ===")
    plot_supp_demand_sensitivity(
        shapefile_path=PATHS["shapefile"], hatch_df=hatch_df,
        output_dir=args.output_dir, agg_datasets_dir=PATHS["out_dir"], dpi=args.dpi)
    plot_supp_demand_sensitivity_absolute(
        shapefile_path=PATHS["shapefile"], hatch_df=hatch_df,
        output_dir=args.output_dir, agg_datasets_dir=PATHS["out_dir"], dpi=args.dpi)

    print("\n=== STEP 6: ren_tot sensitivity figures ===")
    for tot_re in [0.25, 0.5, 0.75]:
        csv = os.path.join(PATHS["out_dir"],
            f"rl_agg_adaptation_Annual_0.99_ren_pen_{tot_re}_current_v2.csv")
        if not os.path.exists(csv):
            print(f"  [SKIP] {csv} missing"); continue
        tag = f"re{tot_re}_thr0.99"
        print(f"  Generating figures for tot_re={tot_re} ...")
        df_gwl15, df_gwl2, df_gwl3 = load_gwl_dfs(csv)
        # plot_supp_gwl_maps(df_gwl15, df_gwl2, df_gwl3,
        #                    PATHS["shapefile"], hatch_df,
        #                    args.output_dir, tag, "current", dpi=args.dpi)
        # if tot_re != MAIN_TOT_RE:
        #     plot_supp_re_effect(df_gwl15, df_gwl2, df_gwl3,
        #                         PATHS["shapefile"], hatch_df,
        #                         args.output_dir, tag, "current", dpi=args.dpi)
        #     plot_supp_tas_effect(df_gwl15, df_gwl2, df_gwl3,
        #                          PATHS["shapefile"], hatch_df,
        #                          args.output_dir, tag, "current", dpi=args.dpi)
        #     plot_supp_decomp(df_gwl2, PATHS["shapefile"], hatch_df,
        #                      args.output_dir, tag, "current", dpi=args.dpi)
        #     plot_supp_uncertainty_decomp(df_gwl2, PATHS["shapefile"], hatch_df,
        #                                  args.output_dir, tag, "current", dpi=args.dpi)
        # plot_supp_re_variability(df_gwl2, PATHS["shapefile"], hatch_df,
        #                          args.output_dir, tag, "current", dpi=args.dpi)

    print("\n=== STEP 7: thr sensitivity figures ===")
    for thr in [0.95, 0.99, 0.995]:
        csv = os.path.join(PATHS["out_dir"],
            f"rl_agg_adaptation_Annual_{thr}_ren_pen_0.5_current_v2.csv")
        if not os.path.exists(csv):
            print(f"  [SKIP] {csv} missing"); continue
        tag = f"thr{thr}_re0.5"
        print(f"  Generating figures for thr={thr} ...")
        df_gwl15, df_gwl2, df_gwl3 = load_gwl_dfs(csv)
        # plot_supp_gwl_maps(df_gwl15, df_gwl2, df_gwl3,
        #                    PATHS["shapefile"], hatch_df,
        #                    args.output_dir, tag, "current", dpi=args.dpi)

    print("\n=== STEP 8: Supplementary figures for main thresholds ===")
    thr, tot_re = 0.99, 0.5
    tag = f"thr{thr}_re{tot_re}"
    csv = os.path.join(PATHS["out_dir"],
        f"rl_agg_adaptation_Annual_{thr}_ren_pen_{tot_re}_current_v2.csv")
    if os.path.exists(csv):
        df_gwl15, df_gwl2, df_gwl3 = load_gwl_dfs(csv)
        # plot_supp_gwl_maps(df_gwl15, df_gwl2, df_gwl3,
        #                    PATHS["shapefile"], hatch_df,
        #                    args.output_dir, tag, "current", dpi=args.dpi)
        plot_supp_combined_driver_effects(df_gwl2, PATHS["shapefile"], hatch_df,
                                          args.output_dir, share_re="current", dpi=args.dpi)
        plot_supp_combined_driver_effects_absolute(df_gwl2, PATHS["shapefile"], hatch_df,
                                                    args.output_dir, share_re="current", dpi=args.dpi)
        # plot_supp_re_variability(df_gwl2, PATHS["shapefile"], hatch_df,
        #                          args.output_dir, tag, "current", dpi=args.dpi)

    print("\n=== STEP 9: RE share effect figures ===")
    csv_25 = os.path.join(PATHS["out_dir"],
        "rl_agg_adaptation_Annual_0.99_ren_pen_0.25_current_v2.csv")
    csv_50 = os.path.join(PATHS["out_dir"],
        "rl_agg_adaptation_Annual_0.99_ren_pen_0.5_current_v2.csv")
    csv_75 = os.path.join(PATHS["out_dir"],
        "rl_agg_adaptation_Annual_0.99_ren_pen_0.75_current_v2.csv")
    if os.path.exists(csv_25) and os.path.exists(csv_50) and os.path.exists(csv_75):
        gwl_dfs_by_share = {
            0.25: load_gwl_dfs(csv_25),
            0.5:  load_gwl_dfs(csv_50),
            0.75: load_gwl_dfs(csv_75),
        }
        plot_re_share_effect(gwl_dfs_by_share, PATHS["shapefile"], hatch_df,
                             args.output_dir, "re_share_effect", dpi=args.dpi)
        plot_re_share_effect_absolute(gwl_dfs_by_share, PATHS["shapefile"], hatch_df,
                                      args.output_dir, "re_share_effect", dpi=args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
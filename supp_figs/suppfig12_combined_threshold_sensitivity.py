# -*- coding: utf-8 -*-
"""
Extended Data Fig. 12: sensitivity of SWED and SWBD changes to the
event-threshold choices (previously produced by fig1.py).

Top row: historical SWED change (value-by-alpha maps) for capacity-factor
quantile thresholds 0.05 / 0.10 / 0.20; bottom row: GWL2 SWBD change for
residual-load thresholds 0.95 / 0.99 / 0.995. Needs the RL CSVs written by
main_pipeline/make_rl_files.py. Shares data loading and CLI with
main_figs/fig1.py.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; also puts main_pipeline/, main_figs/, supp_figs/, aux_code/ on sys.path
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
os.environ['ESMFMKFILE'] = config.ESMFMKFILE_XENV

import cartopy.crs as ccrs
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patheffects import withStroke
from matplotlib import cm
from map_overlays import draw_wcf_zero_overlay

from fig1 import (build_land_mask, build_parser, FIG_WIDTH_IN, fit_to_width,
                  load_base_inputs, load_or_build_ds_final, MAP_LAT_NORTH,
                  MAP_LAT_SOUTH, mask_poles, rasterize_shapefile)


# =============================================================================
# Value-by-alpha colour helper (also used by aux_code/extra_figs_fig1.py)
# =============================================================================

def _compute_valuebyalpha_rgba(
    ds_final, mask,
    period_hist=(1982, 2001), period_comp=(2002, 2021),
    n_bins_change=5, n_bins_sev=5,
):
    da = (ds_final.frequency.where(mask == 1)
          * ds_final.severity.where(mask == 1)
          * ds_final.duration.where(mask == 1))
    da = da.sel(lat=slice(-60, 68))
    y0, y1 = period_hist
    y2, y3 = period_comp
    da_hist    = da.sel(year=slice(y0, y1)).mean("year", skipna=True)
    da_comp    = da.sel(year=slice(y2, y3)).mean("year", skipna=True)
    rel_change = 100.0 * (da_comp - da_hist) / da_hist
    rel_change = rel_change.where(np.isfinite(rel_change))
    sev        = da_hist

    change_edges = [-100, -25, -10, 10, 25, 100]
    change_bin   = np.digitize(rel_change.values, change_edges[1:-1])
    base_cmap    = cm.get_cmap("coolwarm")
    color_levels = base_cmap(np.linspace(0, 1, n_bins_change))
    max_sev      = 1.0
    sev_edges    = np.linspace(0, max_sev, n_bins_sev + 1) ** 2 / max_sev
    sev_bin      = np.digitize(sev.values, sev_edges[1:-1])
    alpha_levels = np.linspace(0.4, 1.0, n_bins_sev)

    nlat, nlon = rel_change.shape
    valid      = np.isfinite(rel_change.values) & np.isfinite(sev.values)
    rgba_map   = np.zeros((nlat, nlon, 4), dtype=float)
    cb = np.clip(change_bin, 0, n_bins_change - 1)
    sb = np.clip(sev_bin,    0, n_bins_sev    - 1)
    rgba_map[valid, :3] = color_levels[cb[valid], :3]
    rgba_map[valid,  3] = alpha_levels[sb[valid]]
    return rgba_map, rel_change, sev, color_levels, alpha_levels, sev_edges, change_edges


# =============================================================================
# Extended Data Fig. 12 -- combined threshold sensitivity
# =============================================================================

def rl_threshold_csv(rl_path, rl_thr):
    """make_rl_files.py's RL CSV for a given RL threshold at its main renewable
    penetration / mix (make_rl_files._compute_and_save naming, demand_tag=None)."""
    from make_rl_files import MAIN_TOT_RE, MAIN_MIX
    return os.path.join(
        rl_path, f"rl_agg_adaptation_Annual_{rl_thr}_ren_pen_{MAIN_TOT_RE}_{MAIN_MIX}_v2.csv")


def _load_wsbd_gwl2(csv_path):
    """
    GWL2 multi-model-mean SWBD change vs GWL0.61, in days of baseline demand
    -- delegates to main_figs/fig45.py's own loader (load_gwl_dfs, which also drops
    EXCLUDED_RUNS) and metric (_mmm_absolute_days) so these panels match
    fig45's Fig. 4 (fig4_main_gwl_maps_absolute_days) exactly. Returns poly_idx /
    Absolute_Days, with +/-inf (zero baseline demand) set to NaN.
    """
    from fig45 import load_gwl_dfs, _mmm_absolute_days, MAIN_MIX
    _, df_gwl2, _ = load_gwl_dfs(csv_path)
    mmm = _mmm_absolute_days(df_gwl2, share_re=MAIN_MIX)
    mmm["Absolute_Days"] = mmm["Absolute_Days"].replace([np.inf, -np.inf], np.nan)
    return mmm


def plot_combined_threshold_sensitivity(
    ds_005, mask_005, ds_01, mask_01, ds_02, mask_02,
    shapefile_path, csv_thr95, csv_thr99, csv_thr995,
    period_hist=(1982, 2001), period_comp=(2002, 2021),
    n_bins_change=5, n_bins_sev=5, dpi=300,
    wcf_zero_mask=None,
):
    shapefile = gpd.read_file(shapefile_path)
    shapefile_band = shapefile.cx[:, MAP_LAT_SOUTH:MAP_LAT_NORTH]

    rgba_005, _, sev_005, color_levels, alpha_levels, sev_edges, change_edges = \
        _compute_valuebyalpha_rgba(ds_005, mask_005, period_hist, period_comp,
                                   n_bins_change, n_bins_sev)
    rgba_01, _, sev_01, _, _, _, _ = \
        _compute_valuebyalpha_rgba(ds_01, mask_01, period_hist, period_comp,
                                   n_bins_change, n_bins_sev)
    rgba_02, _, sev_02, _, _, _, _ = \
        _compute_valuebyalpha_rgba(ds_02, mask_02, period_hist, period_comp,
                                   n_bins_change, n_bins_sev)

    mmm_95  = _load_wsbd_gwl2(csv_thr95)
    mmm_99  = _load_wsbd_gwl2(csv_thr99)
    mmm_995 = _load_wsbd_gwl2(csv_thr995)

    # Same colour scale construction as fig45.plot_main_gwl_maps_absolute:
    # symmetric around 0, capped at the 95th percentile of |change| pooled
    # over the panels shown (here the three RL thresholds).
    abs_vals = np.concatenate([np.abs(m["Absolute_Days"].dropna().values)
                               for m in (mmm_95, mmm_99, mmm_995)])
    vmax_days = (max(1.0, np.ceil(np.nanpercentile(abs_vals, 95)))
                 if abs_vals.size else 1.0)
    cmap_wsbd = plt.get_cmap("RdYlGn_r")
    norm_wsbd = mcolors.TwoSlopeNorm(vmin=-vmax_days, vcenter=0, vmax=vmax_days)
    gdf_re   = gpd.read_file(shapefile_path)
    gdf_re["poly_idx"] = gdf_re.index
    gdf_re_band = gdf_re.cx[:, MAP_LAT_SOUTH:MAP_LAT_NORTH]

    # 3-row 2-col layout left: value-by-alpha, right: WSBD
    fig_width_in  = FIG_WIDTH_IN        # same column width as all other figures
    fig_height_in = fig_width_in   # 3 rows: approx 3� single-map height
    proj = ccrs.EqualEarth()
    fig  = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    gs   = fig.add_gridspec(3, 2, hspace=0.25, wspace=0.05,
                            left=0.01, right=0.99, top=0.97, bottom=0.07)

    axes_left  = [fig.add_subplot(gs[i, 0], projection=proj) for i in range(3)]
    axes_right = [fig.add_subplot(gs[i, 1], projection=proj) for i in range(3)]

    REF_COLOR = "#c0392b"

    # --- Left column: value-by-alpha maps ---
    left_cfgs = [
        (axes_left[0],  rgba_005, sev_005, ds_005, "a", "thr = 0.05",              False),
        (axes_left[1],  rgba_01,  sev_01,  ds_01,  "b", "thr = 0.10  [reference]", True),
        (axes_left[2],  rgba_02,  sev_02,  ds_02,  "c", "thr = 0.20",              False),
    ]
    for ax, rgba_map, sev_da, ds_src, letter, thr_label, is_ref in left_cfgs:
        ax.imshow(
            rgba_map,
            extent=[sev_da.lon.min().item(), sev_da.lon.max().item(),
                    sev_da.lat.min().item(), sev_da.lat.max().item()],
            origin="lower", transform=ccrs.PlateCarree(),
            interpolation="nearest", rasterized=True,
        )
        da_m = ds_src.frequency.isel(year=0).sel(
            lat=slice(MAP_LAT_SOUTH, MAP_LAT_NORTH))
        t_m  = rasterio.transform.from_bounds(
            da_m.lon.min().item(), da_m.lat.min().item(),
            da_m.lon.max().item(), da_m.lat.max().item(),
            len(da_m.lon), len(da_m.lat),
        )
        land_m  = rasterize_shapefile(shapefile_band, da_m.shape, t_m)[::-1, :]
        ocean_m = land_m & (da_m.isnull())
        draw_wcf_zero_overlay(ax, wcf_zero_mask, land_m, da_m.lat, da_m.lon,
                              nan_data=ocean_m)
        shapefile_band.boundary.plot(ax=ax, color="black", linewidth=0.15,
                                     transform=ccrs.PlateCarree(), zorder=10)
        ax.annotate(
            f"$\\mathbf{{{letter}}}$",
            xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=8,
            color=REF_COLOR if is_ref else "black",
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )
        ax.set_title(f"{thr_label}", fontsize=6,
                     color=REF_COLOR if is_ref else "black")
        ax.set_global()
        mask_poles(ax)
        ax.spines["geo"].set_visible(False)
        if is_ref:
            try:
                ax.spines["geo"].set_edgecolor(REF_COLOR)
                ax.spines["geo"].set_linewidth(2.0)
                ax.spines["geo"].set_visible(True)
            except Exception:
                pass

    # Bivariate legend (lower-left of left column)
    legend_rgba = np.zeros((n_bins_change, n_bins_sev, 4))
    for ic in range(n_bins_change):
        legend_rgba[ic, :, :3] = color_levels[ic, :3]
        legend_rgba[ic, :,  3] = alpha_levels
    leg_ax = fig.add_axes([0.07, 0.06, 0.08, 0.10])   # labels stay inside the figure
    leg_ax.imshow(legend_rgba, origin="lower", aspect="equal")
    leg_ax.set_xticks([0, n_bins_sev // 2, n_bins_sev - 1])
    leg_ax.set_xticklabels(["low", "mid", "high"], fontsize=5, ha="center")
    leg_ax.set_yticks([0, n_bins_change // 2, n_bins_change - 1])
    leg_ax.set_yticklabels(
        [f"<{change_edges[1]:.0f}%",
         f"{change_edges[n_bins_change // 2]:.0f}%;{change_edges[n_bins_change // 2 + 1]:.0f}%",
         f">+{change_edges[-2]:.0f}%"],
        fontsize=5, va="center",
    )
    leg_ax.set_xlabel("Reference\nseverity", fontsize=5, labelpad=4)
    leg_ax.set_ylabel("Rel. change (%)",     fontsize=5, labelpad=4)
    leg_ax.tick_params(axis="both", which="both", length=0)

    # --- Right column: WSBD maps ---
    right_cfgs = [
        (axes_right[0], mmm_95,  "d", "RL thr = 0.95",              False),
        (axes_right[1], mmm_99,  "e", "RL thr = 0.99  [reference]", True),
        (axes_right[2], mmm_995, "f", "RL thr = 0.995",             False),
    ]
    for ax, mmm, letter, thr_label, is_ref in right_cfgs:
        gdf  = gdf_re_band.copy().merge(mmm, on="poly_idx", how="left")
        vals = gdf["Absolute_Days"].to_numpy()
        fcs  = [cmap_wsbd(norm_wsbd(v)) if np.isfinite(v) else (0.8, 0.8, 0.8, 1.0)
                for v in vals]
        for geom, fc in zip(gdf.geometry, fcs):
            if geom is None:
                continue
            ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                              facecolor=fc, edgecolor="none", zorder=2)
        shapefile_band.boundary.plot(ax=ax, color="black", linewidth=0.15,
                                     transform=ccrs.PlateCarree(), zorder=10)
        ax.set_global()
        mask_poles(ax)
        ax.annotate(
            f"$\\mathbf{{{letter}}}$",
            xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=8,
            color=REF_COLOR if is_ref else "black",
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )
        ax.set_title(f"{thr_label}",
                     fontsize=6, color=REF_COLOR if is_ref else "black")
        ax.spines["geo"].set_visible(False)
        if is_ref:
            try:
                ax.spines["geo"].set_edgecolor(REF_COLOR)
                ax.spines["geo"].set_linewidth(2.0)
                ax.spines["geo"].set_visible(True)
            except Exception:
                pass

    # WSBD shared colorbar (bottom of right column)
    cbar_ax = fig.add_axes([0.54, 0.025, 0.44, 0.012])
    sm = plt.cm.ScalarMappable(cmap=cmap_wsbd, norm=norm_wsbd)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="both")
    cb.set_label(
        "SWBDs change under 2°C compared to 0.61°C\n(days of baseline demand)",
        fontsize=5,
    )
    cb.ax.tick_params(labelsize=5)
    return fig


# =============================================================================
# Main
# =============================================================================

def build_s12_parser():
    parser = build_parser(__doc__.strip().splitlines()[0])
    parser.add_argument("--data_path_005", default=None,
                        help="Pre-computed ds_final at threshold 0.05; "
                             "computed from the reanalysis if omitted.")
    parser.add_argument("--data_path_02", default=None,
                        help="Pre-computed ds_final at threshold 0.20; "
                             "computed from the reanalysis if omitted.")
    parser.add_argument("--rl_path", default=config.RL_OUT_DIR,
                        help="Folder holding make_rl_files.py's RL CSVs.")
    return parser


def main():
    args = build_s12_parser().parse_args()
    ds_final, mask, wcf_zero_mask = load_base_inputs(args)

    csv_thr95, csv_thr99, csv_thr995 = (
        rl_threshold_csv(args.rl_path, t) for t in (0.95, 0.99, 0.995))
    missing = [p for p in (csv_thr95, csv_thr99, csv_thr995) if not os.path.exists(p)]
    if missing:
        raise SystemExit("RL-threshold CSVs not found (run main_pipeline/make_rl_files.py "
                         "first):\n  " + "\n  ".join(missing))

    ds_005   = load_or_build_ds_final(args, args.data_path_005, 0.05)
    ds_02    = load_or_build_ds_final(args, args.data_path_02,  0.2)
    mask_005 = build_land_mask(ds_005, args.shapefile)
    mask_02  = build_land_mask(ds_02,  args.shapefile)

    print("Plotting combined threshold-sensitivity figure  ")
    fig = plot_combined_threshold_sensitivity(
        ds_005, mask_005, ds_final, mask, ds_02, mask_02,
        shapefile_path=args.shapefile,
        csv_thr95=csv_thr95, csv_thr99=csv_thr99, csv_thr995=csv_thr995,
        dpi=args.dpi,
        wcf_zero_mask=wcf_zero_mask,
    )
    out = os.path.join(args.output_dir, "supp", "suppfig12_combined_threshold_sensitivity.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fit_to_width(fig)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

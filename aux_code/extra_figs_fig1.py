# -*- coding: utf-8 -*-
"""
Non-paper variants of the fig1.py figures: the per-pixel variability map of
annual SWED severity, and the value-by-alpha threshold-sensitivity maps
(0.05 / 0.20 quantile) without the SWBD row of Extended Data Fig. 12.
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
from matplotlib.patheffects import withStroke
from matplotlib import cm
import cmocean as cmo
from map_overlays import draw_wcf_zero_overlay

from fig1 import (build_land_mask, build_parser, FIG_WIDTH_IN, fit_to_width,
                  load_base_inputs, load_or_build_ds_final, MAP_LAT_NORTH,
                  MAP_LAT_SOUTH, mask_poles, rasterize_shapefile)
from suppfig12_combined_threshold_sensitivity import _compute_valuebyalpha_rgba


# =============================================================================
# Interannual variability map (not in the paper)
# =============================================================================

def plot_variability_map(ds_final, mask, shapefile_path, dpi=300, wcf_zero_mask=None):
    shapefile = gpd.read_file(shapefile_path)
    shapefile_band = shapefile.cx[:, MAP_LAT_SOUTH:MAP_LAT_NORTH]
    pdd     = (ds_final.frequency * ds_final.severity * ds_final.duration).where(mask)
    std_pdd = pdd.std(dim="year").sel(lat=slice(MAP_LAT_SOUTH, MAP_LAT_NORTH))

    fig_width_in  = FIG_WIDTH_IN
    fig_height_in = fig_width_in * (6 / 12)
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in),
                           subplot_kw={"projection": ccrs.EqualEarth()})
    im = ax.pcolormesh(
        std_pdd.lon, std_pdd.lat, std_pdd.values,
        transform=ccrs.PlateCarree(),
        cmap=cmo.cm.amp, vmin=0, vmax=0.25, rasterized=True,
    )
    da_mask = ds_final.frequency.isel(year=0).sel(
        lat=slice(MAP_LAT_SOUTH, MAP_LAT_NORTH))
    t_mask  = rasterio.transform.from_bounds(
        da_mask.lon.min().item(), da_mask.lat.min().item(),
        da_mask.lon.max().item(), da_mask.lat.max().item(),
        len(da_mask.lon), len(da_mask.lat),
    )
    land_plot = rasterize_shapefile(shapefile_band, da_mask.shape, t_mask)[::-1, :]
    mask_plot = land_plot & (da_mask.isnull())
    draw_wcf_zero_overlay(ax, wcf_zero_mask, land_plot, da_mask.lat, da_mask.lon,
                          nan_data=mask_plot)
    shapefile_band.boundary.plot(ax=ax, color="black", linewidth=0.15,
                                 transform=ccrs.PlateCarree(), zorder=10)
    ax.set_global()
    mask_poles(ax)
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6, aspect=40)
    cbar.set_label("Interannual variability", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    ax.set_title("Interannual variability of annual severity (1982-2021)",
                 fontsize=8, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_valuebyalpha_sensitivity(
    ds_005, mask_005, ds_02, mask_02, shapefile_path,
    period_hist=(1982, 2001), period_comp=(2002, 2021),
    n_bins_change=5, n_bins_sev=5,
    wcf_zero_mask=None,
):
    shapefile = gpd.read_file(shapefile_path)
    shapefile_band = shapefile.cx[:, MAP_LAT_SOUTH:MAP_LAT_NORTH]
    rgba_005, _, sev_005, color_levels, alpha_levels, sev_edges, change_edges = \
        _compute_valuebyalpha_rgba(ds_005, mask_005, period_hist, period_comp,
                                   n_bins_change, n_bins_sev)
    rgba_02,  _, sev_02,  _, _, _, _ = \
        _compute_valuebyalpha_rgba(ds_02,  mask_02,  period_hist, period_comp,
                                   n_bins_change, n_bins_sev)

    fig_width_in  = FIG_WIDTH_IN  # 2 side-by-side panels double width
    fig_height_in = FIG_WIDTH_IN * 1.5
    fig, axes = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in), dpi=300,
                             subplot_kw={"projection": ccrs.EqualEarth()})

    panel_configs = [
        (axes[0], rgba_005, sev_005, ds_005, "a",
         "Historical annual SWED severity change\nThreshold = 0.05"),
        (axes[1], rgba_02,  sev_02,  ds_02,  "b",
         "Historical annual SWED severity change\nThreshold = 0.20"),
    ]
    for ax, rgba_map, sev_da, ds_src, letter, title in panel_configs:
        ax.imshow(
            rgba_map,
            extent=[sev_da.lon.min().item(), sev_da.lon.max().item(),
                    sev_da.lat.min().item(), sev_da.lat.max().item()],
            origin="lower", transform=ccrs.PlateCarree(),
            interpolation="nearest", rasterized=True,
        )
        da_m  = ds_src.frequency.isel(year=0).sel(
            lat=slice(MAP_LAT_SOUTH, MAP_LAT_NORTH))
        t_m   = rasterio.transform.from_bounds(
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
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )
        ax.set_title(title, fontsize=8)
        ax.set_global()
        mask_poles(ax)
        ax.spines["geo"].set_visible(False)

    # Shared bivariate legend
    legend_rgba = np.zeros((n_bins_change, n_bins_sev, 4))
    for ic in range(n_bins_change):
        legend_rgba[ic, :, :3] = color_levels[ic, :3]
        legend_rgba[ic, :,  3] = alpha_levels
    legend_ax = fig.add_axes([0.07, 0.12, 0.08, 0.14])
    legend_ax.imshow(legend_rgba, origin="lower", aspect="equal")
    legend_ax.set_xticks([0, n_bins_sev // 2, n_bins_sev - 1])
    legend_ax.set_xticklabels(["low", "mid", "high"], fontsize=5, ha="center")
    legend_ax.set_yticks([0, n_bins_change // 2, n_bins_change - 1])
    legend_ax.set_yticklabels(
        [f"<{change_edges[1]:.0f}%",
         f"{change_edges[n_bins_change // 2]:.0f}%;{change_edges[n_bins_change // 2 + 1]:.0f}%",
         f">+{change_edges[-2]:.0f}%"],
        fontsize=5, va="center",
    )
    legend_ax.set_xlabel("Reference\nseverity", fontsize=6, labelpad=4)
    legend_ax.set_ylabel("Rel. change (%)",     fontsize=6, labelpad=4)
    legend_ax.tick_params(axis="both", which="both", length=0)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# =============================================================================
# Main
# =============================================================================

def build_aux_parser():
    parser = build_parser(__doc__.strip().splitlines()[0])
    parser.add_argument("--data_path_005", default=None)
    parser.add_argument("--data_path_02", default=None)
    return parser


def main():
    args = build_aux_parser().parse_args()
    ds_final, mask, wcf_zero_mask = load_base_inputs(args)

    print("Plotting variability map  ")
    fig = plot_variability_map(ds_final, mask, args.shapefile, dpi=args.dpi,
                               wcf_zero_mask=wcf_zero_mask)
    out = os.path.join(args.output_dir, "supp", "suppfig_pdd_std_map.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fit_to_width(fig)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    ds_005   = load_or_build_ds_final(args, args.data_path_005, 0.05)
    ds_02    = load_or_build_ds_final(args, args.data_path_02,  0.2)
    mask_005 = build_land_mask(ds_005, args.shapefile)
    mask_02  = build_land_mask(ds_02,  args.shapefile)
    print("Plotting value-by-alpha sensitivity figure  ")
    fig = plot_valuebyalpha_sensitivity(
        ds_005, mask_005, ds_02, mask_02, shapefile_path=args.shapefile,
        wcf_zero_mask=wcf_zero_mask)
    out = os.path.join(args.output_dir, "supp", "suppfig_valuebyalpha_sensitivity.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fit_to_width(fig)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

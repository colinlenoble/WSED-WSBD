# -*- coding: utf-8 -*-
"""
Extended Data Fig. 4: projected changes in annual SWED severity at 1.5, 2 and
3 degC warming, stacked value-by-alpha maps without the regional violin
panels (previously STEP 5 of fig3.py).

Shares its ensemble building, inverse-Wasserstein ensemble weighting, land
mask and CLI with main_figs/fig3.py.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; also puts main_pipeline/, main_figs/, supp_figs/, aux_code/ on sys.path
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XCLIM
os.environ['ESMFMKFILE'] = config.ESMFMKFILE_XCLIM

import gc
import cartopy.crs as ccrs
import numpy as np
import geopandas as gpd
import rasterio
from map_overlays import (draw_wcf_zero_overlay, draw_discrepancy_dots,
                          add_exclusion_legend)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from string import ascii_lowercase

from fig3 import (EQUAL_EARTH_ASPECT, FIG_WIDTH_IN, MAP_LAT_NORTH,
                  MAP_LAT_SOUTH, _compute_rgba_map, fit_to_width, mask_poles,
                  rasterize_shapefile)
from fig3 import build_parser, prepare_inputs, iter_gwl_decomp, ensemble_weight


def plot_supp_valuebyalpha_stacked(
    gwl_items,
    shapefile_path,
    da_mask_ref,
    hatchings=None,
    agreement_threshold=config.AGREEMENT_THRESHOLD,
    change_edges=None,
    sev_edges=None,
    n_bins_change=5,
    n_bins_sev=5,
    color_levels=None,
    alpha_levels=None,
    relchange_label="Relative change (%)",
    sev_label="Average annual\nseverity (0.61 °C)",
    wcf_zero_mask=None,
):
    """
    Supplementary figure: value-by-alpha maps for multiple GWL levels stacked
    vertically (one row per GWL), without violin plots.

    Parameters
    ----------
    gwl_items : list of dict with keys 'rgba_map', 'extent', 'gwl_label'
    da_mask_ref : 2-D DataArray (one realization) used for the wcf-zero contour
    """
    if change_edges is None:
        change_edges = [-100, -25, -10, 10, 25, 100]
    if color_levels is None:
        color_levels = cm.get_cmap("coolwarm")(np.linspace(0, 1, n_bins_change))
    if alpha_levels is None:
        alpha_levels = np.linspace(0.4, 1.0, n_bins_sev)
    if sev_edges is None:
        sev_edges = np.linspace(0, 1.0, n_bins_sev + 1) ** 2

    shp = gpd.read_file(shapefile_path)
    shp_band = shp.cx[:, MAP_LAT_SOUTH:MAP_LAT_NORTH]
    lat_ok = (da_mask_ref.lat >= MAP_LAT_SOUTH) & (da_mask_ref.lat <= MAP_LAT_NORTH)

    _transform_ref = rasterio.transform.from_bounds(
        da_mask_ref.lon.min().item(), da_mask_ref.lat.min().item(),
        da_mask_ref.lon.max().item(), da_mask_ref.lat.max().item(),
        len(da_mask_ref.lon), len(da_mask_ref.lat),
    )
    land_shp_band = rasterize_shapefile(shp, da_mask_ref.shape, _transform_ref)[::-1, :]
    land_shp_band = land_shp_band & lat_ok.values[:, None]
    n   = len(gwl_items)

    # Maps span the full column width; the height is derived from the
    # EqualEarth aspect so rows are not height-limited (a height-limited map
    # shrinks the tight crop and LaTeX then upscales every font).
    fig_width_in  = FIG_WIDTH_IN
    map_h_in      = fig_width_in * 0.98 / EQUAL_EARTH_ASPECT
    title_in, gap_in, bottom_in = 0.22, 0.30, 0.05
    fig_height_in = n * map_h_in + (n - 1) * gap_in + title_in + bottom_in

    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=300)
    gs  = GridSpec(n, 1, figure=fig, left=0.01, right=0.99,
                   top=1 - title_in / fig_height_in, bottom=bottom_in / fig_height_in,
                   hspace=gap_in / map_h_in)

    legend_rgba = np.zeros((n_bins_change, n_bins_sev, 4))
    for ic in range(n_bins_change):
        legend_rgba[ic, :, :3] = color_levels[ic, :3]
        legend_rgba[ic, :,  3] = alpha_levels

    for i, item in enumerate(gwl_items):
        ax        = fig.add_subplot(gs[i, 0], projection=ccrs.EqualEarth())
        rgba_map  = item["rgba_map"]
        extent    = item["extent"]
        gwl_label = item["gwl_label"]

        ax.imshow(
            rgba_map,
            extent=extent,
            origin="lower",
            transform=ccrs.PlateCarree(),
            interpolation="nearest",
            rasterized=True,
        )
        shp_band.boundary.plot(ax=ax, color="black", linewidth=0.15,
                               transform=ccrs.PlateCarree(), zorder=10)

        if hatchings is not None:
            # Dot hatching: land pixels that failed the trend-agreement
            # evaluation (agreement_pct <= agreement_threshold).
            _agree_float = hatchings.interp(
                lat=da_mask_ref.lat, lon=da_mask_ref.lon, method="nearest"
            )
            failed_eval_band = land_shp_band & (_agree_float.values <= agreement_threshold)
            draw_discrepancy_dots(ax, failed_eval_band, da_mask_ref.lat, da_mask_ref.lon)

        grey_drawn = draw_wcf_zero_overlay(ax, wcf_zero_mask, land_shp_band,
                                           da_mask_ref.lat, da_mask_ref.lon,
                                           nan_data=da_mask_ref.isnull().values)
        if i == n - 1:
            add_exclusion_legend(ax, show_discrepancy=hatchings is not None,
                                 show_wcf_zero=grey_drawn)

        panel_letter = ascii_lowercase[i]
        panel_gwl    = gwl_label.replace(".0°C", "°C")
        ax.annotate(
            f"$\\mathbf{{{panel_letter}}}$",
            xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=8,
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
            zorder=1000,
        )
        ax.set_title(panel_gwl, fontsize=8)
        ax.set_global()
        mask_poles(ax)
        ax.spines["geo"].set_visible(False)

        # Inset legend
        # Shifted right so its tick/axis labels stay inside the map frame
        legend_ax = inset_axes(ax, width="14%", height="50%", loc="center left", borderpad=0.5,
                               bbox_to_anchor=(0.07, 0, 1, 1), bbox_transform=ax.transAxes)
        legend_ax.imshow(legend_rgba, origin="lower", aspect="equal")
        legend_ax.set_xticks([0, n_bins_sev // 2, n_bins_sev - 1])
        legend_ax.set_xticklabels(["low", "mid", "high"], fontsize=5, ha="center")
        legend_ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        legend_ax.set_yticklabels(["-25%", "-10%", "10%", "25%"], fontsize=5, va="center")
        legend_ax.set_xlabel(sev_label, fontsize=5, labelpad=4)
        legend_ax.set_ylabel(relchange_label, fontsize=5, labelpad=4)
        legend_ax.tick_params(axis="both", which="both", length=0)

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    args = build_parser(__doc__.strip().splitlines()[0]).parse_args()
    inputs = prepare_inputs(args, with_regions=False)

    supp_items       = []
    supp_meta        = {}   # change_edges, sev_edges, color_levels, alpha_levels
    da_mask_ref_supp = None
    for level, gwl_key, gwl_label, fields in iter_gwl_decomp(args, inputs):
        (da_ref_freq, da_ref_int, da_ref_dur,
         da_proj_freq, da_proj_int, da_proj_dur, base_weight) = fields
        weight, _ = ensemble_weight(da_proj_freq, base_weight, inputs)
        print(f"  Collecting rgba map ...")
        _rgba, _extent, _cedges, _sedges, _clvl, _alvl = _compute_rgba_map(
            da_ref_freq, da_ref_int, da_ref_dur,
            da_proj_freq, da_proj_int, da_proj_dur,
            weight=weight, mask=inputs.mask, lat_min=-60, lat_max=68,
        )
        supp_items.append({"rgba_map": _rgba, "extent": _extent, "gwl_label": gwl_label})
        if not supp_meta:
            supp_meta = {
                "change_edges": _cedges, "sev_edges": _sedges,
                "color_levels": _clvl,   "alpha_levels": _alvl,
            }
        # Reference grid for the wcf-zero overlay (same grid for all GWLs)
        if da_mask_ref_supp is None:
            da_mask_ref_supp = da_ref_freq.isel(realization=0).load()
        del fields, da_ref_freq, da_ref_int, da_ref_dur
        del da_proj_freq, da_proj_int, da_proj_dur, weight, base_weight
        gc.collect()

    if not supp_items:
        raise SystemExit("No GWL level could be built.")
    fig_supp = plot_supp_valuebyalpha_stacked(
        gwl_items=supp_items,
        shapefile_path=args.shapefile,
        da_mask_ref=da_mask_ref_supp,
        hatchings=inputs.hatchings,
        agreement_threshold=args.agreement_threshold,
        wcf_zero_mask=inputs.wcf_zero_mask,
        **supp_meta,
    )
    out_supp = os.path.join(args.output_dir, "supp", "suppfig4_valuebyalpha_all_gwl.png")
    os.makedirs(os.path.dirname(out_supp), exist_ok=True)
    fit_to_width(fig_supp)
    fig_supp.savefig(out_supp, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_supp)
    print(f"  Saved -> {out_supp}")
    print("\nDone.")


if __name__ == "__main__":
    main()

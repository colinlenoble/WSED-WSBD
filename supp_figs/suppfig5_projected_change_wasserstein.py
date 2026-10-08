# -*- coding: utf-8 -*-
"""
Extended Data Fig. 5: inverse-Wasserstein-weighted vs. multi-model-mean
projections of annual SWED severity, one figure per GWL --
  a) relative change with each realization weighted by 1/n_gcm times the
     inverse of its normalized Wasserstein trend distance to ERA5 at each
     pixel (the weighting used in Fig. 3),
  b) the flat multi-model mean (1/n_gcm only),
  c) the difference a - b (percentage points).
a and b share one continuous colour scale so their differences read directly.

Needs the per-pixel Wasserstein file built by
main_pipeline/trend_sev_eval_wasserstein.py (config.WASSERSTEIN_NC_PATH).
Shares its ensemble building, weighting, land mask and CLI with
main_figs/fig3.py.
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
import cartopy.feature as cfeature
import numpy as np
import geopandas as gpd
import rasterio
from map_overlays import (draw_wcf_zero_overlay, draw_discrepancy_dots,
                          add_exclusion_legend)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patheffects import withStroke

from fig3 import (EQUAL_EARTH_ASPECT, FIG_WIDTH_IN, MAP_LAT_NORTH,
                  MAP_LAT_SOUTH, _compute_ensemble_rel_change,
                  compute_global_change_stats_gwl, ensemble_weight,
                  fit_to_width, mask_poles, rasterize_shapefile)
from fig3 import build_parser, prepare_inputs, iter_gwl_decomp


LAT_MIN, LAT_MAX = -60, 68   # same crop as fig3's maps/statistics


def _sym_vmax(*fields, pct=98):
    """Symmetric colour limit: the pct-th percentile of |value| over fields."""
    vals = np.concatenate([np.abs(f[np.isfinite(f)]) for f in fields])
    return max(float(np.nanpercentile(vals, pct)), 1e-6) if vals.size else 1.0


def plot_gwl_wasserstein_vs_mmm(
    rel_w2, rel_mmm, extent, gwl_label,
    shapefile_path,
    da_mask_ref,
    hatchings=None,
    agreement_threshold=config.AGREEMENT_THRESHOLD,
    wcf_zero_mask=None,
    change_label="Relative change in annual severity\ncompared to 0.61°C (%)",
    diff_label="Difference, historical-agreement-weighted\nminus multi-model mean (pp)",
):
    """
    Three-panel figure for one GWL level: a (inverse-W2 weighted) and b
    (multi-model mean) side by side on one shared continuous diverging scale,
    c (a - b) full width below on its own diverging scale. rel_w2/rel_mmm are
    2-D relative-change arrays on the cropped grid (from
    _compute_ensemble_rel_change), drawn at `extent`.
    """
    rel_diff = rel_w2 - rel_mmm

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
    failed_eval_band = None
    if hatchings is not None:
        # Dot hatching: land pixels that failed the trend-agreement evaluation.
        _agree = hatchings.interp(lat=da_mask_ref.lat, lon=da_mask_ref.lon, method="nearest")
        failed_eval_band = land_shp_band & (_agree.values <= agreement_threshold)

    # Fixed inch layout on a FIG_WIDTH_IN-wide canvas: row 1 two half-width
    # maps (a | b) + their shared colorbar, row 2 one full-width map (c) +
    # its colorbar.
    fig_w  = FIG_WIDTH_IN
    margin, hgap = 0.05, 0.10
    half_w = (fig_w - 2 * margin - hgap) / 2
    half_h = half_w / EQUAL_EARTH_ASPECT
    full_w = fig_w - 2 * margin
    full_h = full_w / EQUAL_EARTH_ASPECT
    title_h, cbar_zone, row_gap, suptitle_h = 0.24, 0.50, 0.10, 0.20
    fig_h  = (suptitle_h + title_h + half_h + cbar_zone + row_gap + title_h
              + full_h + cbar_zone)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    def _rect(x_in, y_in, w_in, h_in):
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    y_c   = cbar_zone
    y_top = y_c + full_h + title_h + row_gap + cbar_zone
    proj  = ccrs.EqualEarth()
    ax_a = fig.add_axes(_rect(margin, y_top, half_w, half_h), projection=proj)
    ax_b = fig.add_axes(_rect(margin + half_w + hgap, y_top, half_w, half_h), projection=proj)
    ax_c = fig.add_axes(_rect(margin, y_c, full_w, full_h), projection=proj)

    vmax_ab   = _sym_vmax(rel_w2, rel_mmm)
    norm_ab   = mcolors.Normalize(vmin=-vmax_ab, vmax=vmax_ab)
    cmap_ab   = plt.get_cmap("coolwarm")
    vmax_c    = _sym_vmax(rel_diff)
    norm_c    = mcolors.Normalize(vmin=-vmax_c, vmax=vmax_c)
    cmap_c    = plt.get_cmap("PuOr_r")  # non-red/blue so c is not read as a/b

    grey_drawn = False
    for ax, field, cmap, norm, letter, title, tfs in [
        (ax_a, rel_w2,   cmap_ab, norm_ab, "a", "Historical-agreement-weighted mean", 6),
        (ax_b, rel_mmm,  cmap_ab, norm_ab, "b", "Multi-model mean", 6),
        (ax_c, rel_diff, cmap_c,  norm_c,  "c", "Difference (a - b)", 7),
    ]:
        ax.imshow(field, extent=extent, origin="lower", transform=ccrs.PlateCarree(),
                  interpolation="nearest", cmap=cmap, norm=norm, rasterized=True)
        if failed_eval_band is not None:
            draw_discrepancy_dots(ax, failed_eval_band, da_mask_ref.lat, da_mask_ref.lon)
        grey_drawn = draw_wcf_zero_overlay(ax, wcf_zero_mask, land_shp_band,
                                           da_mask_ref.lat, da_mask_ref.lon,
                                           nan_data=da_mask_ref.isnull().values)
        shp_band.boundary.plot(ax=ax, color="black", linewidth=0.15,
                               transform=ccrs.PlateCarree(), zorder=10)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.15)
        ax.set_global()
        mask_poles(ax)
        ax.spines["geo"].set_visible(False)
        ax.annotate(
            f"$\\mathbf{{{letter}}}$", xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=8,
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )
        ax.set_title(title, fontsize=tfs, pad=4)
    add_exclusion_legend(ax_c, show_discrepancy=hatchings is not None,
                         show_wcf_zero=grey_drawn)

    fig.suptitle(f"Annual severity change under {gwl_label} warming",
                 fontsize=7, y=1 - 0.04 / fig_h, va="top")

    cbar_w = 0.5 * full_w
    for norm, cmap, label, y_in in [
        (norm_ab, cmap_ab, change_label, y_top - 0.20),
        (norm_c,  cmap_c,  diff_label,   y_c - 0.20),
    ]:
        cax = fig.add_axes(_rect((fig_w - cbar_w) / 2, y_in, cbar_w, 0.07))
        sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
        cb.set_label(label, fontsize=5, labelpad=2)
        cb.ax.tick_params(labelsize=5, length=2)

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    args = build_parser(__doc__.strip().splitlines()[0]).parse_args()
    if args.weighting != "w2":
        raise SystemExit("suppfig5 compares inverse-W2 weighting against the "
                         "multi-model mean; run it with --weighting w2.")
    inputs = prepare_inputs(args, with_regions=False)

    da_mask_ref_supp = None
    for level, gwl_key, gwl_label, fields in iter_gwl_decomp(args, inputs):
        (da_ref_freq, da_ref_int, da_ref_dur,
         da_proj_freq, da_proj_int, da_proj_dur, base_weight) = fields
        # Reference grid for the wcf-zero overlay (same grid for all GWLs)
        if da_mask_ref_supp is None:
            da_mask_ref_supp = da_ref_freq.isel(realization=0).load()

        print(f"  Building Wasserstein-vs-MMM figure ...")
        weight_pix, inv_w2 = ensemble_weight(da_proj_freq, base_weight, inputs)
        rel = {}
        for key, w in (("w2", weight_pix), ("mmm", base_weight)):
            rel[key] = _compute_ensemble_rel_change(
                da_ref_freq, da_ref_int, da_ref_dur,
                da_proj_freq, da_proj_int, da_proj_dur,
                weight=w, mask=inputs.mask, lat_min=LAT_MIN, lat_max=LAT_MAX,
            )
        lat_c  = da_ref_freq.lat.where((da_ref_freq.lat > LAT_MIN) &
                                       (da_ref_freq.lat < LAT_MAX), drop=True)
        extent = [float(da_ref_freq.lon.min()), float(da_ref_freq.lon.max()),
                  float(lat_c.min()), float(lat_c.max())]

        for key, w, iw in (("inverse-W2 weighted", weight_pix, inv_w2),
                           ("multi-model mean", base_weight, None)):
            (g, ci_lo, ci_hi, gci_lo, gci_hi) = compute_global_change_stats_gwl(
                da_ref_freq, da_ref_int, da_ref_dur,
                da_proj_freq, da_proj_int, da_proj_dur,
                weight=w, mask=inputs.mask, lat_min=LAT_MIN, lat_max=LAT_MAX,
                inv_w2=iw,
            )
            print(f"  Global mean change under {gwl_label} ({key}): {g:+.2f}% "
                  f"[spatial CI: {ci_lo:+.2f}%, {ci_hi:+.2f}%] "
                  f"[GCM CI: {gci_lo:+.2f}%, {gci_hi:+.2f}%]")

        fig = plot_gwl_wasserstein_vs_mmm(
            rel_w2=rel["w2"], rel_mmm=rel["mmm"], extent=extent, gwl_label=gwl_label,
            shapefile_path=args.shapefile,
            da_mask_ref=da_mask_ref_supp,
            hatchings=inputs.hatchings, agreement_threshold=args.agreement_threshold,
            wcf_zero_mask=inputs.wcf_zero_mask,
        )
        out = os.path.join(args.output_dir, "supp",
                           f"suppfig5_projected_change_{gwl_key}_wasserstein_vs_mmm.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fit_to_width(fig)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out}")

        del fields, da_ref_freq, da_ref_int, da_ref_dur
        del da_proj_freq, da_proj_int, da_proj_dur, base_weight, weight_pix, inv_w2
        gc.collect()

    inputs.ds_wasserstein.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

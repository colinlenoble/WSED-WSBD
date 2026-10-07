# -*- coding: utf-8 -*-
"""
Extended Data Fig. 5: skill-weighted projections of annual SWED severity --
each realization reweighted by the inverse of its normalized Wasserstein
trend distance to ERA5 at each pixel (previously part of fig3.py STEP 4).

Needs the per-pixel Wasserstein file built by
main_pipeline/trend_sev_eval_wasserstein.py (config.WASSERSTEIN_NC_PATH).
Shares its ensemble building, land mask and CLI with main_figs/fig3.py.
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
import xarray as xr
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
import cmocean as cmo

from fig3 import (EQUAL_EARTH_ASPECT, FIG_WIDTH_IN, MAP_LAT_NORTH,
                  MAP_LAT_SOUTH, _compute_ensemble_rel_change,
                  _compute_rgba_map, compute_global_change_stats_gwl,
                  fit_to_width, mask_poles, rasterize_shapefile)
from fig3 import build_parser, prepare_inputs, iter_gwl_decomp


# =============================================================================
# Inverse-Wasserstein-distance pixel weighting (supplementary figure)
# =============================================================================

def _build_wasserstein_pixel_weight(da_proj_freq, ds_wasserstein, base_weight,
                                     var="w2_normalized", eps=1e-3):
    """
    Combine the usual per-realization 1/n_gcm weight with a per-pixel weight
    equal to the inverse of that realization's normalized empirical
    Wasserstein trend distance to ERA5 at that pixel (ds_wasserstein's
    w2_normalized, from trend_sev_eval_wasserstein.wasserstein_empirical_grid()):
    a realization whose bootstrap trend distribution is closer to ERA5's at a
    given location counts for more there, on top of (not instead of) the
    existing 1/n_gcm de-duplication across multi-run GCMs.

    Matching is by (GCM, run) pair, since ds_wasserstein's realization axis
    (built from GWL1 files only) does not generally line up positionally with
    da_proj_freq's own realization axis (built per-GWL in from_ds_to_plot_decomp).
    Realizations in da_proj_freq with no Wasserstein match get weight 0
    (excluded) everywhere; eps floors w2_normalized so a near-zero distance
    cannot make a single realization dominate the pixel mean.

    Returns
    -------
    xr.DataArray with dims (realization, lat, lon), positionally aligned with
    da_proj_freq's realization axis (no 'realization' coordinate, matching
    base_weight's own convention), or None if no (GCM, run) pair matches.
    """
    n_real = da_proj_freq.sizes["realization"]
    proj_pairs = [(str(g), str(r)) for g, r in
                  zip(da_proj_freq.GCM.values, da_proj_freq.run.values)]
    w2_pairs = [(str(g), str(r)) for g, r in
                zip(ds_wasserstein.GCM.values, ds_wasserstein.run.values)]
    w2_index = {p: i for i, p in enumerate(w2_pairs)}

    matched_proj_idx, matched_w2_idx, missing = [], [], []
    for i, p in enumerate(proj_pairs):
        if p in w2_index:
            matched_proj_idx.append(i)
            matched_w2_idx.append(w2_index[p])
        else:
            missing.append(p)
    if missing:
        print(f"    [warn] no Wasserstein match for {len(missing)} realization(s), "
              f"excluded from the reweighted map: {missing}")
    if not matched_proj_idx:
        return None

    w2_sel = ds_wasserstein[var].isel(realization=matched_w2_idx)
    w2_sel = w2_sel.interp(lat=da_proj_freq.lat, lon=da_proj_freq.lon, method="nearest")
    inv_w2 = 1.0 / w2_sel.clip(min=eps)                       # (matched, lat, lon)

    base_sel = base_weight.values[matched_proj_idx]           # (matched,)
    weighted_matched = inv_w2.values * base_sel[:, None, None]

    full = np.zeros((n_real,) + weighted_matched.shape[1:], dtype=float)
    full[matched_proj_idx] = np.nan_to_num(weighted_matched, nan=0.0)

    return xr.DataArray(
        full, dims=("realization", "lat", "lon"),
        coords={"lat": da_proj_freq.lat, "lon": da_proj_freq.lon},
    )


def plot_gwl_valuebyalpha_wasserstein(
    rgba_map, extent, gwl_label,
    rel_diff, diff_extent,
    shapefile_path,
    da_mask_ref,
    hatchings=None,
    agreement_threshold=config.AGREEMENT_THRESHOLD,
    change_edges=None, sev_edges=None, n_bins_change=5, n_bins_sev=5,
    color_levels=None, alpha_levels=None,
    relchange_label="Relative change (%)",
    sev_label="Average annual\nseverity (0.61 °C)",
    diff_label="Difference in relative change,\ninverse-W2 minus multi-model mean (pp)",
    diff_vmax=None,
    wcf_zero_mask=None,
):
    """
    Two-panel supplementary figure for one GWL level:
      a) value-by-alpha map, weighted per-pixel by 1/n_gcm times the inverse
         of each realization's normalized Wasserstein trend distance to ERA5
         at that pixel (see _build_wasserstein_pixel_weight), instead of the
         flat multi-model-mean weighting used in the main figure. No region
         boxes.
      b) the difference this reweighting makes to the relative-change field:
         (inverse-W2-weighted relative change) minus (multi-model-mean
         relative change), in percentage points.
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

    # Two full-width map rows (see plot_supp_valuebyalpha_stacked); extra
    # bottom room for panel b's colorbar, which hangs below the map.
    fig_width_in  = FIG_WIDTH_IN
    map_h_in      = fig_width_in * 0.98 / EQUAL_EARTH_ASPECT
    title_in, gap_in, bottom_in = 0.22, 0.30, 0.50
    fig_height_in = 2 * map_h_in + gap_in + title_in + bottom_in
    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=300)
    gs  = GridSpec(2, 1, figure=fig, left=0.01, right=0.99,
                   top=1 - title_in / fig_height_in, bottom=bottom_in / fig_height_in,
                   hspace=gap_in / map_h_in)

    # --- Panel a: value-by-alpha map, inverse-W2 pixel weighting ---
    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.EqualEarth())
    ax_a.imshow(
        rgba_map, extent=extent, origin="lower", transform=ccrs.PlateCarree(),
        interpolation="nearest", rasterized=True,
    )
    shp_band.boundary.plot(ax=ax_a, color="black", linewidth=0.15,
                           transform=ccrs.PlateCarree(), zorder=10)
    if hatchings is not None:
        # Dot hatching: land pixels that failed the trend-agreement evaluation.
        _agree_float_a = hatchings.interp(
            lat=da_mask_ref.lat, lon=da_mask_ref.lon, method="nearest"
        )
        failed_eval_band_a = land_shp_band & (_agree_float_a.values <= agreement_threshold)
        draw_discrepancy_dots(ax_a, failed_eval_band_a, da_mask_ref.lat, da_mask_ref.lon)
    grey_drawn = draw_wcf_zero_overlay(ax_a, wcf_zero_mask, land_shp_band,
                                       da_mask_ref.lat, da_mask_ref.lon,
                                       nan_data=da_mask_ref.isnull().values)
    add_exclusion_legend(ax_a, show_discrepancy=hatchings is not None,
                         show_wcf_zero=grey_drawn)
    ax_a.annotate(
        "$\\mathbf{a}$", xy=(0.02, 1.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=8,
        path_effects=[withStroke(linewidth=1.5, foreground="white")],
    )
    ax_a.set_title(f"Inverse-Wasserstein-weighted annual severity change under {gwl_label} warming",
                   fontsize=7, pad=6)
    ax_a.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.15)
    ax_a.set_global()
    mask_poles(ax_a)
    ax_a.spines["geo"].set_visible(False)

    legend_rgba = np.zeros((n_bins_change, n_bins_sev, 4))
    for ic in range(n_bins_change):
        legend_rgba[ic, :, :3] = color_levels[ic, :3]
        legend_rgba[ic, :,  3] = alpha_levels
    # Shifted right so its tick/axis labels stay inside the map frame
    legend_ax = inset_axes(ax_a, width="14%", height="45%", loc="center left", borderpad=0.5,
                           bbox_to_anchor=(0.07, 0, 1, 1), bbox_transform=ax_a.transAxes)
    legend_ax.imshow(legend_rgba, origin="lower", aspect="equal")
    legend_ax.set_xticks([0, n_bins_sev // 2, n_bins_sev - 1])
    legend_ax.set_xticklabels(["low", "mid", "high"], fontsize=5, ha="center")
    legend_ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    legend_ax.set_yticklabels(["-25%", "-10%", "10%", "25%"], fontsize=5, va="center")
    legend_ax.set_xlabel(sev_label, fontsize=5, labelpad=4)
    legend_ax.set_ylabel(relchange_label, fontsize=5, labelpad=4)
    legend_ax.tick_params(axis="both", which="both", length=0)

    # --- Panel b: difference vs. multi-model mean ---
    ax_b = fig.add_subplot(gs[1, 0], projection=ccrs.EqualEarth())
    finite = rel_diff[np.isfinite(rel_diff)]
    if diff_vmax is None:
        diff_vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
        diff_vmax = max(diff_vmax, 1e-6)
    im = ax_b.imshow(
        rel_diff, extent=diff_extent, origin="lower", transform=ccrs.PlateCarree(),
        interpolation="nearest", cmap=cmo.cm.balance, vmin=-diff_vmax, vmax=diff_vmax,
        rasterized=True,
    )
    if hatchings is not None:
        # Same failed-evaluation dot hatching as panel a.
        draw_discrepancy_dots(ax_b, failed_eval_band_a, da_mask_ref.lat, da_mask_ref.lon)
    draw_wcf_zero_overlay(ax_b, wcf_zero_mask, land_shp_band, da_mask_ref.lat, da_mask_ref.lon,
                          nan_data=da_mask_ref.isnull().values)
    shp_band.boundary.plot(ax=ax_b, color="black", linewidth=0.15,
                           transform=ccrs.PlateCarree(), zorder=10)
    ax_b.annotate(
        "$\\mathbf{b}$", xy=(0.02, 1.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=8,
        path_effects=[withStroke(linewidth=1.5, foreground="white")],
    )
    ax_b.set_title("Difference vs. multi-model mean", fontsize=7, pad=6)
    ax_b.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.15)
    ax_b.set_global()
    mask_poles(ax_b)
    ax_b.spines["geo"].set_visible(False)

    cax = inset_axes(
        ax_b, width="30%", height="6%", loc="lower left",
        bbox_to_anchor=(0.0, -0.1, 1, 1), bbox_transform=ax_b.transAxes, borderpad=0,
    )
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(diff_label, fontsize=5, labelpad=2)
    cb.ax.tick_params(labelsize=5, length=2)

    return fig


# =============================================================================
# Main
# =============================================================================

def build_s5_parser():
    parser = build_parser(__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--wasserstein_path",
        default=config.WASSERSTEIN_NC_PATH,
        help=(
            "Path to a pre-computed per-(GCM, run), per-pixel empirical Wasserstein "
            "trend-distance DataArray (.nc), built by trend_sev_eval_wasserstein.py's "
            "wasserstein_empirical_grid() (variables w2_distance/w2_normalized, dims "
            "realization/lat/lon, coords GCM/run). Each realization is reweighted by "
            "1/n_gcm times the inverse of its normalized Wasserstein distance at each "
            "pixel instead of a flat multi-model mean."
        ),
    )
    return parser


def main():
    args = build_s5_parser().parse_args()
    if not os.path.exists(args.wasserstein_path):
        raise SystemExit(f"Wasserstein file not found: {args.wasserstein_path} -- run "
                         "main_pipeline/trend_sev_eval_wasserstein.py first.")
    inputs = prepare_inputs(args, with_regions=False)
    print(f"Loading Wasserstein distance dataset from {args.wasserstein_path} ...")
    ds_wasserstein = xr.open_dataset(args.wasserstein_path)

    da_mask_ref_supp = None
    for level, gwl_key, gwl_label, fields in iter_gwl_decomp(args, inputs):
        (da_ref_freq, da_ref_int, da_ref_dur,
         da_proj_freq, da_proj_int, da_proj_dur, weight) = fields
        # Reference grid for the wcf-zero overlay (same grid for all GWLs)
        if da_mask_ref_supp is None:
            da_mask_ref_supp = da_ref_freq.isel(realization=0).load()

        print(f"  Building Wasserstein-reweighted figure ...")
        weight_pix = _build_wasserstein_pixel_weight(da_proj_freq, ds_wasserstein, weight)
        if weight_pix is None:
            print(f"    [warn] No (GCM, run) overlap with the Wasserstein dataset "
                  f"for {gwl_label}, skipping.")
            continue
        _rgba_w, _extent_w, _cedges_w, _sedges_w, _clvl_w, _alvl_w = _compute_rgba_map(
            da_ref_freq, da_ref_int, da_ref_dur,
            da_proj_freq, da_proj_int, da_proj_dur,
            weight=weight_pix, mask=inputs.mask, lat_min=-60, lat_max=68,
        )
        rel_change_w = _compute_ensemble_rel_change(
            da_ref_freq, da_ref_int, da_ref_dur,
            da_proj_freq, da_proj_int, da_proj_dur,
            weight=weight_pix, mask=inputs.mask, lat_min=-60, lat_max=68,
        )
        rel_change_mmm = _compute_ensemble_rel_change(
            da_ref_freq, da_ref_int, da_ref_dur,
            da_proj_freq, da_proj_int, da_proj_dur,
            weight=weight, mask=inputs.mask, lat_min=-60, lat_max=68,
        )
        print(f"  Computing global change statistics (inverse-W2 weighted) ...")
        (global_chg_w, ci_lo_w, ci_hi_w,
         gcm_ci_lo_w, gcm_ci_hi_w) = compute_global_change_stats_gwl(
            da_ref_freq, da_ref_int, da_ref_dur,
            da_proj_freq, da_proj_int, da_proj_dur,
            weight=weight_pix, mask=inputs.mask,
            lat_min=-60, lat_max=68,
        )
        print(
            f"  Global mean change under {gwl_label} (inverse-W2 weighted): "
            f"{global_chg_w:+.2f}% "
            f"[spatial CI: {ci_lo_w:+.2f}%, {ci_hi_w:+.2f}%] "
            f"[GCM CI: {gcm_ci_lo_w:+.2f}%, {gcm_ci_hi_w:+.2f}%]"
        )
        fig_w = plot_gwl_valuebyalpha_wasserstein(
            rgba_map=_rgba_w, extent=_extent_w, gwl_label=gwl_label,
            rel_diff=rel_change_w - rel_change_mmm, diff_extent=_extent_w,
            shapefile_path=args.shapefile,
            da_mask_ref=da_mask_ref_supp,
            hatchings=inputs.hatchings, agreement_threshold=args.agreement_threshold,
            change_edges=_cedges_w, sev_edges=_sedges_w,
            color_levels=_clvl_w, alpha_levels=_alvl_w,
            wcf_zero_mask=inputs.wcf_zero_mask,
        )
        fname_w = f"suppfig5_projected_change_valuebyalpha_{gwl_key}_wasserstein.png"
        out_w = os.path.join(args.output_dir, "supp", fname_w)
        os.makedirs(os.path.dirname(out_w), exist_ok=True)
        fit_to_width(fig_w)
        fig_w.savefig(out_w, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_w)
        print(f"  Saved -> {out_w}")

        del fields, da_ref_freq, da_ref_int, da_ref_dur
        del da_proj_freq, da_proj_int, da_proj_dur, weight, weight_pix
        gc.collect()

    ds_wasserstein.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Compare wind / solar capacity-factor *tuning* choices against the default
model on real ERA5 data, reusing compare_wind_methods.py's compound
low-wind/low-solar "severity" metric (expected shortfall on compound-event
days) and comparison methodology -- R2 scatter of long-term mean severity,
discrepancy/relative-discrepancy maps, and Spearman rank correlation of the
reference-period -> rest-of-record relative change -- but generalized from
the 3 wind_method extrapolation strategies to a broader set of tuning
choices: the wind turbine power curve (cut-in/rated/cut-out speed) plus the
3 wind_method options, and the PVGIS solar technology/mounting presets
(compute_solar_cf.PVGIS_K_PRESETS / PVGIS_U_PRESETS).

For each family ("wind" or "solar"), "default" is the reference every other
tuning config is compared against (unlike compare_wind_methods.py, which
uses wind100 as reference); the *other* CF model (solar when sweeping wind,
wind when sweeping solar) stays fixed at its own default throughout, so
each family's comparison isolates that one model's tuning choice.

Source data / dependencies: identical to compare_wind_methods.py (see its
docstring) -- config.ERA5_REGRID_ZARR2_DIR, dependency-light imports
(wind_potential.py / compute_solar_cf.py only, no xesmf/xclim/xagg), plain
lat/lon pcolormesh maps (no cartopy).

Usage:
    python compare_cf_tuning.py
    python compare_cf_tuning.py --family wind
    python compare_cf_tuning.py --family solar --limit 48   # quick test
"""
import argparse
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

import config
from wind_potential import (
    DS_CFConfig, DEFAULT_DS_CF_CONFIG,
    get_hub_height_wind, compute_wind_potential_from_hub_wind,
)
from compute_solar_cf import (
    DEFAULT_PVGIS_COEFFICIENTS, compute_solar_cf, make_pvgis_coefficients,
)
from compare_wind_methods import (
    _HAS_MASKING, load_reanalysis, load_region_mask,
    local_shear_alpha_path, compute_severity, r2_score,
)

try:
    import geopandas as gpd
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False

REFERENCE = "default"

# =============================================================================
# Graphical config for the discrepancy maps -- mirrors fig45.py's
# FIG_WIDTH_IN / font sizes / panel-letter and shared-colorbar layout
# (fig45.plot_supp_demand_sensitivity), so these supplementary maps sit
# visually alongside the paper's other supp figures.
# =============================================================================
FIG_WIDTH_IN = 5.15          # LaTeX single-column width, matches fig45.FIG_WIDTH_IN
MAP_TITLE_FONTSIZE = 5.5
MAP_LETTER_FONTSIZE = 5.5
MAP_LETTER_COLOR = "#2c3e50"  # fig45.ALT_COLOR
SUPTITLE_FONTSIZE = 8
SUBTITLE_FONTSIZE = 6
CBAR_LABEL_FONTSIZE = 6
CBAR_TICK_FONTSIZE = 5

WIND_TUNING_CONFIGS = {
    "default"      : DEFAULT_DS_CF_CONFIG,
    "shear_uniform": replace(DEFAULT_DS_CF_CONFIG, wind_method="shear_uniform"),
    "wind100"      : replace(DEFAULT_DS_CF_CONFIG, wind_method="wind100"),
    "cutin_low"    : replace(DEFAULT_DS_CF_CONFIG, vci=2.5),
    "cutin_high"   : replace(DEFAULT_DS_CF_CONFIG, vci=4.5),
    "rated_low"    : replace(DEFAULT_DS_CF_CONFIG, vr=11.0),
    "rated_high"   : replace(DEFAULT_DS_CF_CONFIG, vr=15.0),
    "cutout_low"   : replace(DEFAULT_DS_CF_CONFIG, vco=22.0),
    "cutout_high"  : replace(DEFAULT_DS_CF_CONFIG, vco=28.0),
}

SOLAR_TUNING_CONFIGS = {
    "default"  : DEFAULT_PVGIS_COEFFICIENTS,
    "csi_2025" : make_pvgis_coefficients("csi_2025", "csi_free"),
    "csi_roof" : make_pvgis_coefficients("csi_current", "csi_roof"),
    "cigs_free": make_pvgis_coefficients("cigs", "cigs_free"),
    "cigs_roof": make_pvgis_coefficients("cigs", "cigs_roof"),
    "cdte_free": make_pvgis_coefficients("cdte", "cdte_free"),
    "cdte_roof": make_pvgis_coefficients("cdte", "cdte_roof"),
}

_FAMILY_CONFIGS = {"wind": WIND_TUNING_CONFIGS, "solar": SOLAR_TUNING_CONFIGS}


# =============================================================================
# Severity (reproduces compare_wind_methods.yearly_severity_for_method,
# split so wcf/scf can be supplied independently -- that version always
# recomputes wcf from cfg and takes scf as a fixed input, which only fits
# the wind sweep; here either side can be the one held fixed.)
# =============================================================================

def _yearly_severity(wcf, scf, ref_period, quantile):
    wcf_ref = wcf.sel(time=slice(*ref_period))
    scf_ref = scf.sel(time=slice(*ref_period))
    wcf_thr = wcf_ref.where(wcf_ref > 0).quantile(quantile, dim="time")
    scf_thr = scf_ref.where(scf_ref > 0).quantile(quantile, dim="time")

    low_wind = xr.where(wcf <= wcf_thr, 1, 0)
    low_solar = xr.where(scf <= scf_thr, 1, 0)
    compound = low_wind * low_solar

    severity = compute_severity(compound, scf, wcf, scf_thr, wcf_thr)
    severity["time"] = severity.time.dt.year
    return severity.rename({"time": "year"}).compute()


def compute_family_severity(family, ds, alpha, ref_period, quantile):
    """
    Yearly per-pixel severity for every tuning config in `family`. The
    non-swept CF model (solar when family='wind', wind when family='solar')
    is held at its own default throughout, computed once and reused.
    """
    configs = _FAMILY_CONFIGS[family]

    wind_hub_default = get_hub_height_wind(ds, DEFAULT_DS_CF_CONFIG, alpha=alpha)
    wcf_default = compute_wind_potential_from_hub_wind(wind_hub_default, DEFAULT_DS_CF_CONFIG).persist()
    scf_default = compute_solar_cf(ds["tas"], ds["rsds"], ds["sfcWind"],
                                   cfg=DEFAULT_PVGIS_COEFFICIENTS).persist()

    severities = {}
    for name, cfg in configs.items():
        print(f"Computing {family} severity for tuning config: {name!r}...")
        if family == "wind":
            if name == "default":
                wcf = wcf_default
            else:
                alpha_arg = alpha if cfg.wind_method == "shear_local" else None
                wind_hub = get_hub_height_wind(ds, cfg, alpha=alpha_arg)
                wcf = compute_wind_potential_from_hub_wind(wind_hub, cfg)
            scf = scf_default
        else:
            wcf = wcf_default
            scf = scf_default if name == "default" else compute_solar_cf(
                ds["tas"], ds["rsds"], ds["sfcWind"], cfg=cfg)

        severities[name] = _yearly_severity(wcf, scf, ref_period, quantile)
        print(f"  done ({severities[name].sizes['year']} years).")
    return severities


# =============================================================================
# Plotting (grid-generalized versions of compare_wind_methods.py's single-row
# comparison plots, so an arbitrary number of tuning configs -- not just 2 --
# fit on one figure)
# =============================================================================

def _grid_axes(n, ncols=3, figsize_per=(4.2, 3.6), **subplot_kw):
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False, **subplot_kw)
    axes_flat = axes.flatten()
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    return fig, axes_flat[:n]


def _map_grid_figure(n, ncols=3, suptitle=None, subtitle=None):
    """
    Compact map grid at fig45.plot_supp_demand_sensitivity's fixed
    FIG_WIDTH_IN width/font sizes/gridspec proportions, so the discrepancy
    maps read as one system with the rest of the supplementary figures.
    """
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig_w = FIG_WIDTH_IN
    fig_h = fig_w * ((5.5 * nrows + 1.2) / (8 * 3)) * 1.05
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.20, wspace=0.04,
                          left=0.01, right=0.99, top=0.90, bottom=0.07)
    axes = [fig.add_subplot(gs[divmod(i, ncols)]) for i in range(n)]
    for extra in range(n, nrows * ncols):
        fig.add_subplot(gs[divmod(extra, ncols)]).set_visible(False)
    if suptitle:
        fig.text(0.5, 0.962, suptitle, ha="center", va="bottom",
                 fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
    # if subtitle:
    #     fig.text(0.5, 0.933, subtitle, ha="center", va="bottom",
    #              fontsize=SUBTITLE_FONTSIZE, color="#555555")
    return fig, axes


def _style_map_panel(ax, name, idx):
    """Panel title + top-left bold letter, fig45 style (fontsize/position)."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(name, fontsize=MAP_TITLE_FONTSIZE, color=MAP_LETTER_COLOR, pad=2)
    ax.annotate(
        f"$\\mathbf{{{chr(97 + idx)}.}}$",
        xy=(0.02, 1.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=MAP_LETTER_FONTSIZE, color=MAP_LETTER_COLOR,
        path_effects=[withStroke(linewidth=1.5, foreground="white")], clip_on=False,
    )


def _overlay_shapefile(ax, shapefile_gdf):
    """Thin admin-boundary overlay from shp_re -- plain lon/lat, no cartopy
    reprojection needed since the regridded ERA5 archive is already on the
    W5E5 (-180..180) grid, same as load_region_mask's masking."""
    if shapefile_gdf is not None:
        shapefile_gdf.boundary.plot(ax=ax, color="black", linewidth=0.2, zorder=3)


def load_shapefile_gdf(shapefile_path):
    if _HAS_GPD and shapefile_path and os.path.exists(shapefile_path):
        return gpd.read_file(shapefile_path)
    if shapefile_path and not _HAS_GPD:
        print("geopandas unavailable -- discrepancy maps won't overlay the shapefile.")
    return None


def plot_r2_scatter(mean_severity, other_names, reference, out_path):
    """R2 (reference as ground truth) of long-term per-pixel mean severity."""
    fig, axes = _grid_axes(len(other_names))
    y_true = mean_severity[reference].values.ravel()
    r2_results = {}
    for ax, name in zip(axes, other_names):
        y_pred = mean_severity[name].values.ravel()
        r2, n = r2_score(y_true, y_pred)
        r2_results[name] = r2
        ax.scatter(y_true, y_pred, s=4, alpha=0.3)
        lims = [0, np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)])]
        ax.plot(lims, lims, "k--", lw=1, label="1:1")
        ax.set_xlabel(f"Mean severity ({reference})", fontsize=7)
        ax.set_ylabel(f"Mean severity ({name})", fontsize=7)
        ax.set_title(f"{name}\nR2={r2:.3f}, n={n}", fontsize=8)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Wrote", out_path)
    return r2_results


def plot_discrepancy_maps(mean_severity, other_names, reference, out_path,
                          shapefile_gdf=None, vmin=-5e-3, vmax=5e-3, axis_thr=1e-6):
    """
    (config - default) mean-severity maps, one shared colorbar fixed to
    [vmin, vmax] across every panel so magnitudes are comparable
    config-to-config. Axis-mismatch pixels (one side ~0 severity, the
    other isn't) are still counted for the summary CSV but no longer
    marked on the map.
    """
    ref = mean_severity[reference]
    lat, lon = ref["latitude"].values, ref["longitude"].values
    ref_vals = ref.values

    fig, axes = _map_grid_figure(
        len(other_names),
        suptitle="Mean-severity discrepancy vs default (absolute)",
        subtitle=f"shared scale, fixed to [{vmin:g}, {vmax:g}]",
    )
    counts = {}
    im = None
    for idx, (ax, name) in enumerate(zip(axes, other_names)):
        method_vals = mean_severity[name].values
        diff = method_vals - ref_vals

        axis_x = (ref_vals <= axis_thr) & (method_vals > axis_thr)
        axis_y = (ref_vals > axis_thr) & (method_vals <= axis_thr)
        counts[name] = (int(np.nansum(axis_x)), int(np.nansum(axis_y)))

        im = ax.pcolormesh(lon, lat, diff, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")
        _overlay_shapefile(ax, shapefile_gdf)
        _style_map_panel(ax, name, idx)

    cbar_ax = fig.add_axes([0.18, 0.025, 0.64, 0.020])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
    cb.set_label("severity diff", fontsize=CBAR_LABEL_FONTSIZE)
    cb.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_path)
    return counts


def plot_relative_discrepancy_maps(mean_severity, other_names, reference, out_path,
                                   shapefile_gdf=None, zero_thr=1e-6, pct_clip=99):
    """(config - default) / default mean-severity maps, one shared colorbar."""
    ref = mean_severity[reference]
    lat, lon = ref["latitude"].values, ref["longitude"].values
    ref_vals = ref.values

    rel_diff = {}
    for name in other_names:
        method_vals = mean_severity[name].values
        with np.errstate(divide="ignore", invalid="ignore"):
            rd = np.where(ref_vals > zero_thr, (method_vals - ref_vals) / ref_vals * 100.0, np.nan)
        rel_diff[name] = rd

    pooled = np.concatenate([rd[np.isfinite(rd)] for rd in rel_diff.values()])
    vmax = np.nanpercentile(np.abs(pooled), pct_clip) if pooled.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig, axes = _map_grid_figure(
        len(other_names),
        suptitle="Mean-severity discrepancy vs default (relative)",
        subtitle=f"shared scale, clipped at pct{pct_clip}=+-{vmax:.0f}%",
    )
    im = None
    for idx, (ax, name) in enumerate(zip(axes, other_names)):
        im = ax.pcolormesh(lon, lat, rel_diff[name], cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        _overlay_shapefile(ax, shapefile_gdf)
        _style_map_panel(ax, name, idx)

    cbar_ax = fig.add_axes([0.18, 0.025, 0.64, 0.020])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
    cb.set_label("relative diff (%)", fontsize=CBAR_LABEL_FONTSIZE)
    cb.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_path)
    return rel_diff


def plot_spearman_scatter(rel_change, other_names, reference, out_path):
    """Spearman rank correlation of the ref-period -> rest-of-record relative change."""
    x = rel_change[reference].values.ravel()
    fig, axes = _grid_axes(len(other_names))
    spearman_results = {}
    for ax, name in zip(axes, other_names):
        y = rel_change[name].values.ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        rho, pval = stats.spearmanr(x[valid], y[valid])
        spearman_results[name] = (float(rho), float(pval), int(valid.sum()))

        ax.scatter(x[valid], y[valid], s=4, alpha=0.3)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_xlabel(f"Rel. change ({reference})", fontsize=7)
        ax.set_ylabel(f"Rel. change ({name})", fontsize=7)
        ax.set_title(f"{name}\nSpearman rho={rho:.3f}, n={int(valid.sum())}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Wrote", out_path)
    return spearman_results


# =============================================================================
# Per-family analysis (mirrors compare_wind_methods.main()'s body)
# =============================================================================

def analyze_family(family, ds, alpha, ref_period, quantile, out_dir, shapefile_gdf=None):
    severities = compute_family_severity(family, ds, alpha, ref_period, quantile)
    other_names = [n for n in severities if n != REFERENCE]
    mean_severity = {n: s.mean(dim="year") for n, s in severities.items()}

    r2_results = plot_r2_scatter(
        mean_severity, other_names, REFERENCE,
        os.path.join(out_dir, f"{family}_severity_r2_scatter.png"))

    axis_counts = plot_discrepancy_maps(
        mean_severity, other_names, REFERENCE,
        os.path.join(out_dir, f"{family}_severity_discrepancy_map.png"),
        shapefile_gdf=shapefile_gdf)
    for name, (n_x, n_y) in axis_counts.items():
        print(f"{name}: {n_x} pixels with {name}>0/{REFERENCE}~0, "
              f"{n_y} pixels with {REFERENCE}>0/{name}~0")

    plot_relative_discrepancy_maps(
        mean_severity, other_names, REFERENCE,
        os.path.join(out_dir, f"{family}_severity_relative_discrepancy_map.png"),
        shapefile_gdf=shapefile_gdf)

    ref_start_year = int(pd.Timestamp(ref_period[0]).year)
    ref_end_year = int(pd.Timestamp(ref_period[1]).year)
    rel_change = {}
    for name, sev in severities.items():
        ref_mean = sev.sel(year=slice(ref_start_year, ref_end_year)).mean(dim="year")
        post_mean = sev.sel(year=slice(ref_end_year + 1, None)).mean(dim="year")
        rel_change[name] = xr.where(ref_mean > 0, (post_mean - ref_mean) / ref_mean, np.nan)

    spearman_results = plot_spearman_scatter(
        rel_change, other_names, REFERENCE,
        os.path.join(out_dir, f"{family}_severity_relchange_spearman_scatter.png"))

    summary = pd.DataFrame({
        "config": other_names,
        "R2_mean_severity_vs_default": [r2_results[n] for n in other_names],
        "spearman_rho_relchange_vs_default": [spearman_results[n][0] for n in other_names],
        "spearman_pvalue": [spearman_results[n][1] for n in other_names],
        "n_pixels": [spearman_results[n][2] for n in other_names],
        "n_axis_x_config_gt0_default_0": [axis_counts[n][0] for n in other_names],
        "n_axis_y_default_gt0_config_0": [axis_counts[n][1] for n in other_names],
    })
    summary_path = os.path.join(out_dir, f"{family}_tuning_comparison_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print("Wrote", summary_path)


# =============================================================================
# CLI
# =============================================================================

def main():
    default_out_dir = os.path.join(config.SUMMARY_FIGS_DIR, "cf_tuning_comparison")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["wind", "solar", "both"], default="both")
    ap.add_argument("--src_dir", default=config.ERA5_REGRID_ZARR2_DIR,
                    help="regridded ERA5 archive, W5E5 grid, Zarr format 2 (default: config.ERA5_REGRID_ZARR2_DIR)")
    ap.add_argument("--path_folder", default=config.PATH_FOLDER,
                    help="fallback root for the cached shear exponent (default: config.PATH_FOLDER)")
    ap.add_argument("--path_preprocessed", default=config.PATH_PREPROCESSED,
                    help="preferred root for the cached shear exponent (default: config.PATH_PREPROCESSED)")
    ap.add_argument("--alpha_path", default=None,
                    help="cached local shear exponent .nc; default derived from "
                         "--path_preprocessed/--path_folder + --ref_start/--ref_end")
    ap.add_argument("--shapefile", default=config.SHAPEFILE_PATH)
    ap.add_argument("--out_dir", default=default_out_dir)
    ap.add_argument("--ref_start", default=config.SHEAR_REF_PERIOD[0])
    ap.add_argument("--ref_end", default=config.SHEAR_REF_PERIOD[1])
    ap.add_argument("--quantile", type=float, default=0.1, help="low-wind/low-solar event threshold (fraction)")
    ap.add_argument("--limit", type=int, default=None, help="only load the first N monthly stores (for testing)")
    ap.add_argument("--no_mask", action="store_true", help="skip the land/region shapefile mask")
    args = ap.parse_args()
    ref_period = (args.ref_start, args.ref_end)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_reanalysis(args.src_dir, limit=args.limit)

    if not args.no_mask and _HAS_MASKING and os.path.exists(args.shapefile):
        print("Applying region mask:", args.shapefile)
        mask = load_region_mask(args.shapefile, ds["latitude"], ds["longitude"])
        ds = ds.where(mask)
    elif not args.no_mask:
        print("Shapefile masking unavailable/not found -- using the full grid.")

    alpha_path = args.alpha_path or local_shear_alpha_path(
        args.path_preprocessed, args.path_folder, ref_period)
    print("Loading alpha (local shear exponent):", alpha_path)
    alpha = xr.open_dataset(alpha_path)["alpha"]
    alpha = alpha.reindex(latitude=ds["latitude"], longitude=ds["longitude"],
                          method="nearest", tolerance=1e-6)

    shapefile_gdf = load_shapefile_gdf(args.shapefile)

    families = ["wind", "solar"] if args.family == "both" else [args.family]
    for family in families:
        print(f"\n=== {family} CF tuning comparison ===")
        analyze_family(family, ds, alpha, ref_period, args.quantile, args.out_dir,
                       shapefile_gdf=shapefile_gdf)


if __name__ == "__main__":
    main()

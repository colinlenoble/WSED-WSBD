# -*- coding: utf-8 -*-
"""
Compare wind / solar capacity-factor *tuning* choices against the default
model on real ERA5 data, reusing compare_wind_methods.py's compound
low-wind/low-solar "severity" metric (expected shortfall on compound-event
days) and comparison methodology -- generalized from the 3 wind_method
extrapolation strategies to a broader set of tuning choices: the wind
turbine power curve (cut-in/rated/cut-out speed) plus the 3 wind_method
options, the PVGIS solar technology/mounting presets
(compute_solar_cf.PVGIS_K_PRESETS / PVGIS_U_PRESETS), and (folded in from
the former analyze_zero_cf.py) the wind low-event threshold's zero-handling
("raw_threshold": the naive `wcf_ref.quantile(q)` with zeros included,
compared against the pipeline default `wcf_ref.where(wcf_ref>0).quantile(q)`).

Every tuning config's effect on the long-term mean severity is boiled down
to two comparable-scale numbers -- RMSE and relative RMSE (RMSE normalized
by the default's own mean severity, in %) across pixels, against that
config's own family default -- and reported in ONE multi-panel figure that
puts wind-CF tuning, the zero-threshold hypothesis, and solar-CF tuning
side by side (instead of the three separate per-family map/scatter figures
this script used to produce). R2, MAE and the Spearman rank correlation of
the reference-period -> rest-of-record relative change are kept as extra
columns in the summary table for anyone who wants to build a different view
from it. Figure width and font sizes are harmonized with fig1.py's
FIG_WIDTH_IN / panel-letter / title conventions so this sits visually
alongside the paper's other figures.

For each family ("wind" or "solar"), "default" is the reference every other
tuning config in that family is compared against (unlike
compare_wind_methods.py, which uses wind100 as reference); the *other* CF
model (solar when sweeping wind, wind when sweeping solar) stays fixed at
its own default throughout, so each family's comparison isolates that one
model's tuning choice. The zero-threshold hypothesis ("raw_threshold") is
wind-only and compared against the same wind "default".

Zero-inflation background (why "raw_threshold" exists): every
threshold-fitting step in this pipeline computes the "low wind"/"low solar"
event threshold as the q-th quantile of *non-zero* CF values only
(`x_ref.where(x_ref > 0).quantile(q)`). Solar is zero every night by
construction, so filtering zeros there just keeps the threshold meaningful
over daylight hours. Wind is zero whenever wind speed is below the
turbine's cut-in speed; a pixel where wind speed is *always* below cut-in
has wcf == 0 on every day of the reference period, so `.where(wcf_ref>0)`
turns the whole slice to NaN and the quantile comes out NaN -- `wcf <=
wcf_thr` is then False every day, and the pixel silently contributes 0 to
frequency/severity instead of being flagged high-risk. The "wind" family's
zero-fraction diagnostics (printed before the severity sweep) and the
"raw_threshold" config quantify how much this default zero-filtering
choice matters.

Source data / dependencies: identical to compare_wind_methods.py (see its
docstring) -- config.ERA5_REGRID_ZARR2_DIR, dependency-light imports
(wind_potential.py / compute_solar_cf.py only, no xesmf/xclim/xagg,
geopandas or cartopy -- this script has no maps).

Outputs (under --out_dir):
  cf_tuning_severity_summary.csv          -- one row per non-default config
                                              (group, R2, MAE, RMSE,
                                              relative_RMSE_pct, Spearman
                                              rho/p-value, n_pixels,
                                              axis-mismatch counts)
  cf_tuning_severity_effect_summary.png   -- 2-panel bar chart (RMSE,
                                              relative RMSE) of every config
                                              vs. its family default,
                                              grouped wind / zero-hypothesis
                                              / solar

Usage:
    python compare_cf_tuning.py
    python compare_cf_tuning.py --limit 48   # quick test
    python compare_cf_tuning.py --skip_zero_diagnostics
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
    DEFAULT_DS_CF_CONFIG,
    get_hub_height_wind, compute_wind_potential_from_hub_wind,
)
from compute_solar_cf import (
    DEFAULT_PVGIS_COEFFICIENTS, compute_solar_cf, make_pvgis_coefficients,
)
from compare_wind_methods import (
    _HAS_MASKING, load_reanalysis, load_region_mask,
    local_shear_alpha_path, compute_severity, r2_score,
)

REFERENCE = "default"

# =============================================================================
# Graphical config -- matches fig1.py's FIG_WIDTH_IN and the font sizes it
# uses for supplementary/summary figures (panel letters, titles, ticks),
# so this sits visually alongside the paper's other figures.
# =============================================================================
FIG_WIDTH_IN = 5.15          # LaTeX single-column width, matches fig1.FIG_WIDTH_IN
SUPTITLE_FONTSIZE = 8
PANEL_LETTER_FONTSIZE = 7
AXIS_LABEL_FONTSIZE = 6
TICK_FONTSIZE = 5
LEGEND_FONTSIZE = 6

GROUP_ORDER = ["wind", "zero_hypothesis", "solar"]
GROUP_COLORS = {"wind": "#2c7fb8", "zero_hypothesis": "#e6550d", "solar": "#31a354"}
GROUP_LABELS = {"wind": "Wind CF tuning", "zero_hypothesis": "Zero-threshold hypothesis",
                "solar": "Solar CF tuning"}

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
# Zero-inflation diagnostics (folded in from the former analyze_zero_cf.py).
# Wind-only: wind speed below a turbine's cut-in speed makes wcf exactly 0,
# and a pixel that's always below cut-in over the reference period turns the
# default `.where(wcf_ref>0).quantile(q)` threshold into NaN there (see
# module docstring). These are printed diagnostics only, run once before the
# wind severity sweep -- the "raw_threshold" config below is what actually
# quantifies the effect on severity (RMSE/relative RMSE, alongside every
# other tuning config, in the summary CSV/figure).
# =============================================================================

def zero_fraction(da, dim="time"):
    """Fraction of exactly-zero steps along `dim`, out of the non-NaN steps."""
    valid = da.notnull().sum(dim)
    n_zero = (da == 0).sum(dim)
    return xr.where(valid > 0, n_zero / valid, np.nan)


def summarize_zero_fraction(frac, label):
    """Print a distribution of per-pixel zero-fraction (NaNs, e.g. outside
    the region mask, excluded)."""
    vals = frac.values
    valid = np.isfinite(vals)
    n_valid = int(valid.sum())
    print(f"\n--- Zero-fraction summary: {label} ({n_valid} valid pixels) ---")
    if n_valid == 0:
        return
    bins = [(-1e-9, 1e-9, "== 0%  (never zero)"),
            (1e-9, 0.01, "0-1%"),
            (0.01, 0.10, "1-10%"),
            (0.10, 0.50, "10-50%"),
            (0.50, 0.90, "50-90%"),
            (0.90, 0.999, "90-99.9%"),
            (0.999, 1.0 + 1e-9, ">=99.9% (essentially always zero)")]
    for lo, hi, name in bins:
        n = int(np.sum(valid & (vals > lo) & (vals <= hi)))
        pct = 100.0 * n / n_valid
        print(f"  {name:35s}: {n:8d} pixels ({pct:5.1f}%)")
    print(f"  mean zero-fraction = {np.nanmean(vals):.4f}, median = {np.nanmedian(vals):.4f}")


def report_removal_cutoffs(frac_wcf_ref):
    """Show how many pixels get flagged as 'always-zero wind' at several
    candidate cutoffs, to make the removal criterion's sensitivity explicit."""
    vals = frac_wcf_ref.values
    n_valid = int(np.isfinite(vals).sum())
    print(f"\n--- 'Always-zero wind' pixel counts by cutoff ({n_valid} valid pixels) ---")
    for cutoff in (1.0, 0.99, 0.95, 0.90):
        n = int(np.nansum(vals >= cutoff))
        pct = 100.0 * n / n_valid if n_valid else np.nan
        print(f"  zero-fraction >= {cutoff:5.2f}: {n:6d} pixels ({pct:5.2f}% of land)")
    print("  These pixels have wind speed permanently (or almost permanently) below "
          "cut-in over the reference period: 'wcf_ref.where(wcf_ref>0).quantile(q)' sees "
          "an all-NaN slice there, returns NaN, and the resulting 'low_wind' flag is False "
          "every day -- i.e. they silently drop out of frequency/severity instead of being "
          "counted as high-risk. See the 'raw_threshold' row in the summary CSV/figure for "
          "how much this shifts the mean severity.")


def report_zero_diagnostics(wcf_default, ref_period):
    frac_full = zero_fraction(wcf_default).load()
    frac_ref = zero_fraction(wcf_default.sel(time=slice(*ref_period))).load()
    summarize_zero_fraction(frac_full, "wind_full_record")
    summarize_zero_fraction(frac_ref, "wind_reference_period")
    report_removal_cutoffs(frac_ref)


# =============================================================================
# Severity (reproduces compare_wind_methods.yearly_severity_for_method,
# split so wcf/scf can be supplied independently -- that version always
# recomputes wcf from cfg and takes scf as a fixed input, which only fits
# the wind sweep; here either side can be the one held fixed.)
# =============================================================================

def _yearly_severity(wcf, scf, ref_period, quantile, filter_zero_wind=True, filter_zero_solar=True):
    wcf_ref = wcf.sel(time=slice(*ref_period))
    scf_ref = scf.sel(time=slice(*ref_period))
    wcf_src = wcf_ref.where(wcf_ref > 0) if filter_zero_wind else wcf_ref
    scf_src = scf_ref.where(scf_ref > 0) if filter_zero_solar else scf_ref
    wcf_thr = wcf_src.quantile(quantile, dim="time")
    scf_thr = scf_src.quantile(quantile, dim="time")

    low_wind = xr.where(wcf <= wcf_thr, 1, 0)
    low_solar = xr.where(scf <= scf_thr, 1, 0)
    compound = low_wind * low_solar

    severity = compute_severity(compound, scf, wcf, scf_thr, wcf_thr)
    severity["time"] = severity.time.dt.year
    return severity.rename({"time": "year"}).compute()


def compute_family_severity(family, ds, alpha, ref_period, quantile, skip_zero_diagnostics=False):
    """
    Yearly per-pixel severity for every tuning config in `family`. The
    non-swept CF model (solar when family='wind', wind when family='solar')
    is held at its own default throughout, computed once and reused.

    For family='wind', an extra "raw_threshold" config is appended: same
    wcf/scf as "default", but the low-wind threshold is fit on the raw
    (zero-inclusive) quantile instead of the pipeline's zero-filtered one --
    see the module docstring and report_zero_diagnostics.
    """
    configs = _FAMILY_CONFIGS[family]

    wind_hub_default = get_hub_height_wind(ds, DEFAULT_DS_CF_CONFIG, alpha=alpha)
    wcf_default = compute_wind_potential_from_hub_wind(wind_hub_default, DEFAULT_DS_CF_CONFIG).persist()
    scf_default = compute_solar_cf(ds["tas"], ds["rsds"], ds["sfcWind"],
                                   cfg=DEFAULT_PVGIS_COEFFICIENTS).persist()

    if family == "wind" and not skip_zero_diagnostics:
        report_zero_diagnostics(wcf_default, ref_period)

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

    if family == "wind":
        print("Computing wind severity for tuning config: 'raw_threshold' "
              "(zero-inclusive low-wind quantile threshold)...")
        severities["raw_threshold"] = _yearly_severity(
            wcf_default, scf_default, ref_period, quantile, filter_zero_wind=False)
        print(f"  done ({severities['raw_threshold'].sizes['year']} years).")

    return severities


# =============================================================================
# Per-config metrics (mean-severity RMSE / relative RMSE / R2 / MAE against
# the family default, plus Spearman rank correlation of the reference-period
# -> rest-of-record relative change) -- everything the summary table/figure
# needs, computed once per config.
# =============================================================================

def _mae(y_true, y_pred):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return np.nan
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid])))


def _rmse(y_true, y_pred):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return np.nan
    return float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))


def _relative_rmse(y_true, y_pred):
    """RMSE normalized by the mean of the reference (y_true), as a
    percentage (a.k.a. NRMSE/CV(RMSE)) -- puts wind/zero-hypothesis/solar
    configs, whose severity units differ in typical magnitude, on one
    comparable scale, without the per-pixel divide-by-near-zero blowups a
    pixel-wise relative-difference map would have near-zero severity."""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return np.nan
    denom = np.mean(y_true[valid])
    if not np.isfinite(denom) or denom == 0:
        return np.nan
    rmse = np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2))
    return float(rmse / abs(denom) * 100.0)


def _rel_change(sev, ref_start_year, ref_end_year):
    """Reference-period -> rest-of-record relative change, per pixel."""
    ref_mean = sev.sel(year=slice(ref_start_year, ref_end_year)).mean(dim="year")
    post_mean = sev.sel(year=slice(ref_end_year + 1, None)).mean(dim="year")
    return xr.where(ref_mean > 0, (post_mean - ref_mean) / ref_mean, np.nan)


def compute_group_summary(group, severities, other_names, reference, ref_period, axis_thr=1e-6):
    """One summary row per non-default config in `other_names`, all compared
    against `severities[reference]` (that group's own family default)."""
    mean_severity = {n: severities[n].mean(dim="year") for n in [reference, *other_names]}
    ref_start_year = int(pd.Timestamp(ref_period[0]).year)
    ref_end_year = int(pd.Timestamp(ref_period[1]).year)
    ref_vals = mean_severity[reference].values.ravel()
    x = _rel_change(severities[reference], ref_start_year, ref_end_year).values.ravel()

    rows = []
    for name in other_names:
        pred_vals = mean_severity[name].values.ravel()
        r2, n = r2_score(ref_vals, pred_vals)

        axis_x = int(np.nansum((ref_vals <= axis_thr) & (pred_vals > axis_thr)))
        axis_y = int(np.nansum((ref_vals > axis_thr) & (pred_vals <= axis_thr)))

        y = _rel_change(severities[name], ref_start_year, ref_end_year).values.ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        rho, pval = stats.spearmanr(x[valid], y[valid]) if valid.sum() > 1 else (np.nan, np.nan)

        rows.append(dict(
            group=group, config=name,
            R2_mean_severity=r2,
            MAE_mean_severity=_mae(ref_vals, pred_vals),
            RMSE_mean_severity=_rmse(ref_vals, pred_vals),
            relative_RMSE_pct=_relative_rmse(ref_vals, pred_vals),
            spearman_rho_relchange=float(rho) if np.isfinite(rho) else np.nan,
            spearman_pvalue=float(pval) if np.isfinite(pval) else np.nan,
            n_pixels=n,
            n_axis_x_config_gt0_ref0=axis_x,
            n_axis_y_ref_gt0_config0=axis_y,
        ))
    return pd.DataFrame(rows)


# =============================================================================
# Consolidated multi-panel figure: every tuning config's RMSE / relative
# RMSE vs. its family default, grouped wind / zero-hypothesis / solar on one
# comparable scale. Width/fonts harmonized with fig1.py (FIG_WIDTH_IN,
# panel-letter style, title/tick font sizes).
# =============================================================================

def plot_severity_effect_summary(summary, out_path):
    configs = summary["config"].tolist()
    groups = summary["group"].tolist()
    colors = [GROUP_COLORS[g] for g in groups]
    x = np.arange(len(configs))

    boundaries = [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]

    fig_w = FIG_WIDTH_IN
    fig_h = fig_w * 0.85
    fig, (ax_rmse, ax_rel) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h), sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    for ax, col, ylabel, letter in (
        (ax_rmse, "RMSE_mean_severity", "RMSE\n(mean severity)", "a"),
        (ax_rel, "relative_RMSE_pct", "Relative RMSE (%)\n(mean severity)", "b"),
    ):
        ax.bar(x, summary[col], color=colors, width=0.7)
        for b in boundaries:
            ax.axvline(b - 0.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        ax.annotate(
            f"$\\mathbf{{{letter}}}$",
            xy=(0.01, 1.04), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=PANEL_LETTER_FONTSIZE,
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )

    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(configs, fontsize=TICK_FONTSIZE, rotation=45, ha="right")

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in GROUP_ORDER]
    ax_rmse.legend(legend_handles, [GROUP_LABELS[g] for g in GROUP_ORDER],
                   fontsize=LEGEND_FONTSIZE, loc="upper right", frameon=False)

    fig.suptitle("Effect of CF tuning choices on historical mean severity\n"
                 "(RMSE / relative RMSE vs. each family's default, across pixels)",
                 fontsize=SUPTITLE_FONTSIZE, y=1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_path)


# =============================================================================
# Full analysis: both families + the zero-threshold hypothesis, combined
# into one summary table and one summary figure.
# =============================================================================

def analyze_all(ds, alpha, ref_period, quantile, out_dir, skip_zero_diagnostics=False):
    print("\n=== wind CF tuning ===")
    severities_wind = compute_family_severity(
        "wind", ds, alpha, ref_period, quantile, skip_zero_diagnostics=skip_zero_diagnostics)
    print("\n=== solar CF tuning ===")
    severities_solar = compute_family_severity(
        "solar", ds, alpha, ref_period, quantile, skip_zero_diagnostics=True)

    wind_names = [n for n in severities_wind if n not in (REFERENCE, "raw_threshold")]
    solar_names = [n for n in severities_solar if n != REFERENCE]

    df_wind = compute_group_summary("wind", severities_wind, wind_names, REFERENCE, ref_period)
    df_zero = compute_group_summary(
        "zero_hypothesis", severities_wind, ["raw_threshold"], REFERENCE, ref_period)
    df_solar = compute_group_summary("solar", severities_solar, solar_names, REFERENCE, ref_period)

    summary = pd.concat([df_wind, df_zero, df_solar], ignore_index=True)
    summary_path = os.path.join(out_dir, "cf_tuning_severity_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n" + summary.to_string(index=False))
    print("Wrote", summary_path)

    plot_severity_effect_summary(
        summary, os.path.join(out_dir, "cf_tuning_severity_effect_summary.png"))

    return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    default_out_dir = os.path.join(config.SUMMARY_FIGS_DIR, "cf_tuning_comparison")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    ap.add_argument("--skip_zero_diagnostics", action="store_true",
                    help="skip the wind zero-fraction diagnostic report (printed by default)")
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

    analyze_all(ds, alpha, ref_period, args.quantile, args.out_dir,
               skip_zero_diagnostics=args.skip_zero_diagnostics)


if __name__ == "__main__":
    main()

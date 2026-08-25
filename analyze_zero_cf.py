# -*- coding: utf-8 -*-
"""
analyze_zero_cf.py

Diagnose zero-inflation in the wind (wcf) and solar (scf) capacity-factor
time series, and quantify its effect on the pipeline's compound
low-wind/low-solar "severity" metric (see make_grid_files.compute_severity /
fig1.compute_severity / compare_wind_methods.compute_severity).

Background
----------
Every threshold-fitting step in this pipeline (make_grid_files.py,
fig1.py, fig2.py/fig3.py, fig_persistent.py, compare_wind_methods.py,
compare_cf_tuning.py) computes the "low wind" / "low solar" event
threshold as the q-th quantile of *non-zero* CF values only:

    x_thr = x_ref.where(x_ref > 0).quantile(q, dim="time")

Solar is zero every night by construction, so filtering zeros there just
keeps the threshold meaningful over daylight hours -- otherwise the
quantile would collapse towards 0 for any q below the (roughly 50%) night
fraction. Wind is zero whenever wind speed is below the turbine's cut-in
speed. For most sites that's a handful of calm days; but a pixel where
wind speed is *always* below cut-in has wcf == 0 on every single day of
the reference period. Feeding an all-zero series into `.where(x > 0)`
turns every value to NaN, and `.quantile()` of an all-NaN slice is NaN.
Consequences, silently, downstream:
  - wcf_thr is NaN at that pixel,
  - `wcf <= wcf_thr` is False every day (comparisons against NaN are
    always False in numpy/xarray), so `low_wind` is 0 every day,
  - the pixel then never registers a compound event and contributes 0 to
    both frequency and severity -- indistinguishable, in the output
    statistics, from a genuinely windy site that simply never drops below
    its own threshold.
That conflation ("no wind resource" vs. "abundant, reliable wind") is why
these permanently-sub-cut-in pixels are excluded / grey-masked out of the
maps and severity statistics in fig2.py / fig3.py (see `no_wind_mask`,
built from a pixel-is-null test) rather than left in as if they carried 0
risk.

What this script does
----------------------
  1. Computes the per-pixel fraction of zero days in wcf and scf, over the
     full record and over the threshold-fitting reference period, and
     prints a distribution summary (counts/percentages of pixels falling
     in different zero-fraction bins).
  2. Flags the "always-zero wind" pixels at several candidate cutoffs
     (exactly 100% zero, >=99%, >=95%, >=90%) so you can see how sensitive
     the removal criterion is to the exact cutoff chosen.
  3. Recomputes the compound severity metric two ways for the surviving
     pixels:
       - "nonzero" : the pipeline default, `x_ref.where(x_ref > 0).quantile(q)`
       - "raw"     : the naive alternative, `x_ref.quantile(q)` (zeros included)
     and reports how much the threshold and the resulting severity differ:
     fraction of pixels whose "raw" threshold collapses to exactly 0
     (degenerate -- "low X" then only fires on exact-zero days instead of
     the intended bottom-q% of conditions), mean/median/percentile relative
     change in severity, total compound-day counts, and the R2 between the
     two severity maps.

Outputs (under --out_dir)
--------------------------
  zero_cf_diagnostics.nc   -- per-pixel fields (zero fractions, masks,
                               thresholds, mean severity) for further
                               plotting/inspection.
  zero_cf_summary.csv      -- one-row-per-metric table of the scalar
                               summary statistics printed to stdout.
  wcf_zero_fraction_map.png, severity_nonzero_vs_raw.png
                            -- quicklook maps (skip with --no_plots).

Usage
-----
    python analyze_zero_cf.py
    python analyze_zero_cf.py --reanalysis ERA5 --quantile 0.1 \\
        --ref_start 1982-01-01 --ref_end 2001-12-31 --out_dir ./zero_cf_diag
    python analyze_zero_cf.py --wcf_path /path/wcf_day_x.nc --scf_path /path/scf_day_x.nc
    python analyze_zero_cf.py --no_mask --no_plots   # skip shapefile land mask / figures
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

import config
from io_utils import match_files, open_dataset_any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    _HAS_MASKING = True
except ImportError:
    _HAS_MASKING = False


# =============================================================================
# Loading
# =============================================================================

def load_wcf_scf(args):
    """Open the reanalysis daily wcf/scf files (zarr preferred, .nc fallback),
    matching fig1.build_ds_final's lookup convention."""
    if args.wcf_path and args.scf_path:
        wcf_path, scf_path = args.wcf_path, args.scf_path
    else:
        base = os.path.join(args.path_preprocessed, args.reanalysis)
        wcf_files, _ = match_files(os.path.join(base, "wcf_day_*"))
        scf_files, _ = match_files(os.path.join(base, "scf_day_*"))
        if not wcf_files:
            raise FileNotFoundError(f"No wcf_day_* file found under {base}")
        if not scf_files:
            raise FileNotFoundError(f"No scf_day_* file found under {base}")
        wcf_path, scf_path = wcf_files[0], scf_files[0]

    print(f"Loading wcf: {wcf_path}")
    print(f"Loading scf: {scf_path}")
    chunks = {"time": -1, "lat": 50, "lon": 50}
    wcf = open_dataset_any(wcf_path, chunks=chunks)
    scf = open_dataset_any(scf_path, chunks=chunks)
    wcf = wcf.convert_calendar("standard")
    scf = scf.convert_calendar("standard")

    if args.lat_min is not None or args.lat_max is not None:
        wcf = wcf.sel(lat=slice(args.lat_min, args.lat_max))
        scf = scf.sel(lat=slice(args.lat_min, args.lat_max))
    return wcf, scf


def load_land_mask(args, lat, lon):
    """Optional shapefile land mask, True where the grid-cell center falls
    inside the shapefile (mirrors compare_wind_methods.load_region_mask)."""
    if args.no_mask or not _HAS_MASKING or not os.path.exists(args.shapefile):
        if not args.no_mask:
            print("Shapefile masking unavailable/not found -- using the full grid.")
        return None
    shp = gpd.read_file(args.shapefile)
    transform = rasterio.transform.from_bounds(
        float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()),
        len(lon), len(lat),
    )
    mask = geometry_mask(geometries=shp.geometry, out_shape=(len(lat), len(lon)),
                         transform=transform, invert=True, all_touched=True)
    if lat.values[0] < lat.values[-1]:
        mask = mask[::-1, :]
    return xr.DataArray(mask, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


# =============================================================================
# Zero-fraction diagnostics
# =============================================================================

def zero_fraction(da, dim="time"):
    """Fraction of exactly-zero steps along `dim`, out of the non-NaN steps."""
    valid = da.notnull().sum(dim)
    n_zero = (da == 0).sum(dim)
    return xr.where(valid > 0, n_zero / valid, np.nan), n_zero, valid


def summarize_zero_fraction(frac, label, land_mask=None):
    """Print a distribution of per-pixel zero-fraction, restricted to
    land_mask if given. Returns the summary rows as a list of dicts."""
    vals = frac.values
    if land_mask is not None:
        vals = np.where(land_mask.values, vals, np.nan)
    valid = np.isfinite(vals)
    n_valid = int(valid.sum())
    print(f"\n--- Zero-fraction summary: {label} ({n_valid} valid pixels) ---")
    if n_valid == 0:
        return []
    bins = [(-1e-9, 1e-9, "== 0%  (never zero)"),
            (1e-9, 0.01, "0-1%"),
            (0.01, 0.10, "1-10%"),
            (0.10, 0.50, "10-50%"),
            (0.50, 0.90, "50-90%"),
            (0.90, 0.999, "90-99.9%"),
            (0.999, 1.0 + 1e-9, ">=99.9% (essentially always zero)")]
    rows = []
    for lo, hi, name in bins:
        n = int(np.sum(valid & (vals > lo) & (vals <= hi)))
        pct = 100.0 * n / n_valid
        print(f"  {name:35s}: {n:8d} pixels ({pct:5.1f}%)")
        rows.append({"metric": f"{label}_zero_frac_bin[{name}]", "value": n,
                     "pct_of_valid": pct})
    mean_frac = float(np.nanmean(vals))
    median_frac = float(np.nanmedian(vals))
    print(f"  mean zero-fraction = {mean_frac:.4f}, median = {median_frac:.4f}")
    rows.append({"metric": f"{label}_zero_frac_mean", "value": mean_frac, "pct_of_valid": np.nan})
    rows.append({"metric": f"{label}_zero_frac_median", "value": median_frac, "pct_of_valid": np.nan})
    return rows


def report_removal_cutoffs(frac_wcf_ref, land_mask=None):
    """Show how many pixels get flagged as 'always-zero wind' at several
    candidate cutoffs, to make the removal criterion's sensitivity explicit."""
    vals = frac_wcf_ref.values
    if land_mask is not None:
        vals = np.where(land_mask.values, vals, np.nan)
    n_valid = int(np.isfinite(vals).sum())
    print(f"\n--- 'Always-zero wind' pixel counts by cutoff ({n_valid} valid land pixels) ---")
    rows = []
    for cutoff in (1.0, 0.99, 0.95, 0.90):
        n = int(np.nansum(vals >= cutoff))
        pct = 100.0 * n / n_valid if n_valid else np.nan
        print(f"  zero-fraction >= {cutoff:5.2f}: {n:6d} pixels ({pct:5.2f}% of land)")
        rows.append({"metric": f"n_pixels_wind_zero_frac_ge_{cutoff}", "value": n,
                     "pct_of_valid": pct})
    print("  These pixels have wind speed permanently (or almost permanently) below "
          "cut-in over the reference period: '.where(wcf_ref>0).quantile(q)' sees an "
          "all-NaN slice there, returns NaN, and the resulting 'low_wind' flag is False "
          "every day -- i.e. they silently drop out of frequency/severity instead of "
          "being counted as high-risk. That's why they're excluded/grey-masked in the "
          "figures rather than left in the statistics.")
    return rows


# =============================================================================
# Severity, computed two ways: pipeline-default ("nonzero") vs naive ("raw")
# =============================================================================

def compute_severity(compound_da, scf_da, wcf_da, scf_thr, wcf_thr):
    """Expected shortfall: mean positive deficit on compound-event days,
    aggregated yearly. Reproduces make_grid_files.compute_severity /
    compare_wind_methods.compute_severity."""
    deficit_scf = xr.where(scf_da < scf_thr, scf_thr - scf_da, 0)
    deficit_wcf = xr.where(wcf_da < wcf_thr, wcf_thr - wcf_da, 0)
    daily_deficit = deficit_scf + deficit_wcf
    masked = xr.where(compound_da == 1, daily_deficit, np.nan)
    return masked.resample(time="YE").mean().fillna(0)


def quantile_threshold(da_ref, q, filter_zero):
    """`filter_zero=True` reproduces the pipeline default
    (`da_ref.where(da_ref > 0).quantile(q)`); `filter_zero=False` is the
    naive alternative that keeps zeros in the quantile."""
    src = da_ref.where(da_ref > 0) if filter_zero else da_ref
    thr = src.quantile(q, dim="time")
    return thr.reset_coords("quantile", drop=True) if "quantile" in thr.coords else thr


def severity_for_method(wcf, scf, ref_period, q, filter_zero_wind, filter_zero_solar):
    wcf_ref = wcf.wcf.sel(time=slice(*ref_period))
    scf_ref = scf.scf.sel(time=slice(*ref_period))
    wcf_thr = quantile_threshold(wcf_ref, q, filter_zero_wind)
    scf_thr = quantile_threshold(scf_ref, q, filter_zero_solar)

    low_wind = xr.where(wcf.wcf <= wcf_thr, 1, 0)
    low_solar = xr.where(scf.scf <= scf_thr, 1, 0)
    compound = low_wind * low_solar

    severity = compute_severity(compound, scf.scf, wcf.wcf, scf_thr, wcf_thr)
    severity["time"] = severity.time.dt.year
    severity = severity.rename({"time": "year"})

    nb_days = compound.resample(time="YE").sum("time")
    nb_days["time"] = nb_days.time.dt.year
    nb_days = nb_days.rename({"time": "year"})

    return {
        "wcf_thr": wcf_thr.compute(),
        "scf_thr": scf_thr.compute(),
        "severity": severity.compute(),
        "nb_days": nb_days.compute(),
    }


def r2_score(y_true, y_pred):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if y_true.size < 2:
        return np.nan, int(y_true.size)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(r2), int(y_true.size)


def compare_methods(wcf, scf, ref_period, q, land_mask, all_zero_wind_mask):
    """Compute severity with the pipeline-default ('nonzero') and naive
    ('raw') thresholds, and quantify the difference. Solar keeps the
    default nonzero filter throughout (isolating the wind-side effect,
    which is what drives location removal); pass filter_zero_wind=False
    for the 'raw' wind comparison."""
    print("\nComputing severity with the pipeline-default threshold "
          "(wcf_ref.where(wcf_ref>0).quantile(q)) ...")
    default = severity_for_method(wcf, scf, ref_period, q,
                                  filter_zero_wind=True, filter_zero_solar=True)
    print("Computing severity with the naive threshold "
          "(wcf_ref.quantile(q), zeros included) ...")
    raw = severity_for_method(wcf, scf, ref_period, q,
                              filter_zero_wind=False, filter_zero_solar=True)

    valid = ~all_zero_wind_mask
    if land_mask is not None:
        valid = valid & land_mask

    rows = []

    # --- threshold degeneracy ---
    raw_thr_vals = np.where(valid.values, raw["wcf_thr"].values, np.nan)
    n_valid = int(np.isfinite(raw_thr_vals).sum())
    n_degenerate = int(np.nansum(raw_thr_vals <= 0))
    pct_degenerate = 100.0 * n_degenerate / n_valid if n_valid else np.nan
    print(f"\n--- Threshold comparison ({n_valid} valid, non-all-zero-wind, land pixels) ---")
    print(f"  raw (zeros-included) wcf threshold <= 0 at {n_degenerate} pixels "
          f"({pct_degenerate:.1f}%) -- at these, 'low_wind' under the raw method only "
          f"fires on exact-zero (dead-calm) days instead of the intended bottom-{q*100:.0f}% "
          "of wind conditions.")
    rows.append({"metric": "n_pixels_raw_wcf_thr_degenerate_le0", "value": n_degenerate,
                "pct_of_valid": pct_degenerate})

    default_thr_vals = np.where(valid.values, default["wcf_thr"].values, np.nan)
    thr_diff = raw_thr_vals - default_thr_vals
    print(f"  mean(raw_thr - default_thr) = {np.nanmean(thr_diff):.5f}  "
          f"(raw threshold is lower -> misses more genuinely-low-but-nonzero wind days)")
    rows.append({"metric": "mean_wcf_thr_raw_minus_default", "value": float(np.nanmean(thr_diff)),
                "pct_of_valid": np.nan})

    # --- severity impact ---
    mean_sev_default = default["severity"].mean(dim="year")
    mean_sev_raw = raw["severity"].mean(dim="year")
    y_true = np.where(valid.values, mean_sev_default.values, np.nan).ravel()
    y_pred = np.where(valid.values, mean_sev_raw.values, np.nan).ravel()

    r2, n = r2_score(y_true, y_pred)
    global_mean_default = float(np.nanmean(y_true))
    global_mean_raw = float(np.nanmean(y_pred))
    global_pct_change = (100.0 * (global_mean_raw - global_mean_default) / global_mean_default
                         if global_mean_default else np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff_pct = np.where(y_true > 1e-9, (y_pred - y_true) / y_true * 100.0, np.nan)
    finite_rel = rel_diff_pct[np.isfinite(rel_diff_pct)]

    print(f"\n--- Severity comparison (mean annual severity, {n} valid pixels) ---")
    print(f"  spatial-mean severity: default={global_mean_default:.5f}, "
          f"raw={global_mean_raw:.5f}  ({global_pct_change:+.1f}%)")
    print(f"  R2(raw vs default) = {r2:.4f}")
    if finite_rel.size:
        print(f"  per-pixel relative diff (raw vs default), pixels with default>0: "
              f"mean={np.mean(finite_rel):+.1f}%, median={np.median(finite_rel):+.1f}%, "
              f"p5={np.percentile(finite_rel, 5):+.1f}%, p95={np.percentile(finite_rel, 95):+.1f}%")

    total_days_default = float(default["nb_days"].sum().values)
    total_days_raw = float(raw["nb_days"].sum().values)
    pct_days_change = (100.0 * (total_days_raw - total_days_default) / total_days_default
                       if total_days_default else np.nan)
    print(f"  total compound-event days (all years, all pixels): "
          f"default={total_days_default:.0f}, raw={total_days_raw:.0f} ({pct_days_change:+.1f}%)")

    rows += [
        {"metric": "global_mean_severity_default", "value": global_mean_default, "pct_of_valid": np.nan},
        {"metric": "global_mean_severity_raw", "value": global_mean_raw, "pct_of_valid": np.nan},
        {"metric": "global_mean_severity_pct_change_raw_vs_default", "value": global_pct_change, "pct_of_valid": np.nan},
        {"metric": "R2_raw_vs_default_mean_severity", "value": r2, "pct_of_valid": np.nan},
        {"metric": "total_compound_days_default", "value": total_days_default, "pct_of_valid": np.nan},
        {"metric": "total_compound_days_raw", "value": total_days_raw, "pct_of_valid": np.nan},
        {"metric": "total_compound_days_pct_change_raw_vs_default", "value": pct_days_change, "pct_of_valid": np.nan},
    ]

    diag = xr.Dataset({
        "wcf_thr_default": default["wcf_thr"],
        "wcf_thr_raw": raw["wcf_thr"],
        "mean_severity_default": mean_sev_default,
        "mean_severity_raw": mean_sev_raw,
    })
    return rows, diag


# =============================================================================
# Plots
# =============================================================================

def save_plots(frac_wcf_ref, all_zero_wind_mask, diag, out_dir):
    if not _HAS_MPL:
        print("matplotlib unavailable -- skipping plots.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(frac_wcf_ref.lon, frac_wcf_ref.lat, frac_wcf_ref.values,
                       cmap="viridis", vmin=0, vmax=1, shading="auto")
    ax.contour(all_zero_wind_mask.lon, all_zero_wind_mask.lat,
              all_zero_wind_mask.astype(float).values, levels=[0.5], colors="red", linewidths=0.6)
    ax.set_title("Fraction of zero-wcf days over reference period\n"
                 "(red contour: pixels excluded as 'always-zero wind')")
    fig.colorbar(im, ax=ax, label="zero-day fraction")
    path = os.path.join(out_dir, "wcf_zero_fraction_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    y_true = diag["mean_severity_default"].values.ravel()
    y_pred = diag["mean_severity_raw"].values.ravel()
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    axes[0].scatter(y_true[valid], y_pred[valid], s=3, alpha=0.3)
    lims = [0, np.nanmax([y_true[valid].max() if valid.any() else 1,
                         y_pred[valid].max() if valid.any() else 1])]
    axes[0].plot(lims, lims, "k--", lw=1)
    axes[0].set_xlabel("mean severity (default, nonzero threshold)")
    axes[0].set_ylabel("mean severity (raw, zeros included)")
    axes[0].set_title("Per-pixel mean severity")

    diff = diag["mean_severity_raw"] - diag["mean_severity_default"]
    im1 = axes[1].pcolormesh(diff.lon, diff.lat, diff.values, cmap="RdBu_r",
                             vmin=-np.nanpercentile(np.abs(diff.values), 99),
                             vmax=np.nanpercentile(np.abs(diff.values), 99), shading="auto")
    axes[1].set_title("severity(raw) - severity(default)")
    fig.colorbar(im1, ax=axes[1])

    thr_diff = diag["wcf_thr_raw"] - diag["wcf_thr_default"]
    im2 = axes[2].pcolormesh(thr_diff.lon, thr_diff.lat, thr_diff.values, cmap="RdBu_r",
                             vmin=-np.nanpercentile(np.abs(thr_diff.values), 99),
                             vmax=np.nanpercentile(np.abs(thr_diff.values), 99), shading="auto")
    axes[2].set_title("wcf_thr(raw) - wcf_thr(default)")
    fig.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    path = os.path.join(out_dir, "severity_nonzero_vs_raw.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", path)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path_preprocessed", default=config.PATH_PREPROCESSED)
    ap.add_argument("--reanalysis", default=config.REANALYSIS)
    ap.add_argument("--wcf_path", default=None, help="explicit wcf file/store, overrides --path_preprocessed/--reanalysis lookup")
    ap.add_argument("--scf_path", default=None, help="explicit scf file/store, overrides --path_preprocessed/--reanalysis lookup")
    ap.add_argument("--quantile", type=float, default=0.1, help="low-wind/low-solar event quantile (default: 0.1, matching the rest of the pipeline)")
    ap.add_argument("--ref_start", default=config.SHEAR_REF_PERIOD[0])
    ap.add_argument("--ref_end", default=config.SHEAR_REF_PERIOD[1])
    ap.add_argument("--lat_min", type=float, default=None)
    ap.add_argument("--lat_max", type=float, default=None)
    ap.add_argument("--shapefile", default=config.SHAPEFILE_PATH)
    ap.add_argument("--no_mask", action="store_true", help="skip the land/region shapefile mask")
    ap.add_argument("--no_plots", action="store_true", help="skip PNG quicklooks")
    ap.add_argument("--out_dir", default="./zero_cf_diag")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ref_period = (args.ref_start, args.ref_end)

    wcf, scf = load_wcf_scf(args)
    land_mask = load_land_mask(args, wcf.lat, wcf.lon)

    wcf_ref = wcf.wcf.sel(time=slice(*ref_period))
    scf_ref = scf.scf.sel(time=slice(*ref_period))

    all_rows = []

    print("=" * 70)
    print("1) Zero-count / zero-fraction analysis")
    print("=" * 70)
    frac_wcf_full, n_zero_wcf_full, n_valid_wcf_full = zero_fraction(wcf.wcf)
    frac_scf_full, n_zero_scf_full, n_valid_scf_full = zero_fraction(scf.scf)
    frac_wcf_ref, n_zero_wcf_ref, n_valid_wcf_ref = zero_fraction(wcf_ref)
    frac_scf_ref, n_zero_scf_ref, n_valid_scf_ref = zero_fraction(scf_ref)
    for da in (frac_wcf_full, frac_scf_full, frac_wcf_ref, frac_scf_ref):
        da.load()

    all_rows += summarize_zero_fraction(frac_wcf_full, "wind_full_record", land_mask)
    all_rows += summarize_zero_fraction(frac_wcf_ref, "wind_reference_period", land_mask)
    all_rows += summarize_zero_fraction(frac_scf_full, "solar_full_record", land_mask)
    print("  (Solar's zero fraction is dominated by night-time steps -- expected/"
          "systematic, not a data-quality flag the way wind's is.)")

    print("\n" + "=" * 70)
    print("2) Why locations get removed: 'always-zero wind' pixels")
    print("=" * 70)
    all_rows += report_removal_cutoffs(frac_wcf_ref, land_mask)

    all_zero_wind_mask = (frac_wcf_ref >= 1.0 - 1e-9)

    print("\n" + "=" * 70)
    print("3) Impact of the non-zero threshold choice on severity")
    print("=" * 70)
    sev_rows, diag = compare_methods(wcf, scf, ref_period, args.quantile,
                                     land_mask, all_zero_wind_mask)
    all_rows += sev_rows

    diag["frac_zero_wcf_full"] = frac_wcf_full
    diag["frac_zero_wcf_ref"] = frac_wcf_ref
    diag["frac_zero_scf_full"] = frac_scf_full
    diag["frac_zero_scf_ref"] = frac_scf_ref
    diag["all_zero_wind_mask"] = all_zero_wind_mask
    if land_mask is not None:
        diag["land_mask"] = land_mask

    nc_path = os.path.join(args.out_dir, "zero_cf_diagnostics.nc")
    diag.to_netcdf(nc_path)
    print("\nWrote", nc_path)

    summary_df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.out_dir, "zero_cf_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print("Wrote", csv_path)

    if not args.no_plots:
        save_plots(frac_wcf_ref, all_zero_wind_mask, diag, args.out_dir)


if __name__ == "__main__":
    main()

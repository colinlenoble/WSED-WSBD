# -*- coding: utf-8 -*-
"""
Compare the three wind capacity-factor methods (wind_potential.WIND_METHODS)
against each other on real ERA5 data, treating wind100 -- the reanalysis
100 m wind read directly, no shear extrapolation -- as the reference that
shear_local (per-pixel fitted Hellmann exponent) and shear_uniform (single
global exponent, default 1/7) are checked against.

Source data: E:/climate_data/ERA5/daily_regrid_zarr2 (W5E5 0.5 deg grid,
Zarr format 2, has u10/v10/u100/v100/t2m/ssrd -- see
convert_regrid_to_zarr2.py), so all three methods see exactly the same
input and only the wcf formula differs.

Runs without xesmf/xclim/xagg. wind_potential.py and compute_solar_cf.py
are already dependency-light and imported directly; two short formulas
that otherwise only live inside xesmf-importing modules (calculate_cf.py's
ssrd unit conversion, make_grid_files.py's severity deficit) are
reproduced below instead, each commented with its source -- importing
those modules here would pull in xesmf at load time.

For each of shear_local / shear_uniform, against wind100:
  1. R2 (sklearn-style: 1 - SS_res/SS_tot, wind100 as ground truth) + a
     scatter plot of long-term per-pixel mean compound-event severity.
  2. Spearman rank correlation, per pixel, of the *relative change* in mean
     severity from the reference period (1982-01-01..2001-12-31, matching
     config.SHEAR_REF_PERIOD) to the rest of the record.

Usage:
    python compare_wind_methods.py
    python compare_wind_methods.py --limit 48   # ~4 years, quick test
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from wind_potential import DS_CFConfig, get_hub_height_wind, compute_wind_potential_from_hub_wind
from compute_solar_cf import compute_solar_cf, DEFAULT_PVGIS_COEFFICIENTS

try:
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    _HAS_MASKING = True
except ImportError:
    _HAS_MASKING = False

SRC_DIR = r"E:/climate_data/ERA5/daily_regrid_zarr2"
ALPHA_PATH = r"E:/climate_data/ERA5/shear_exponent_local_1982-01-01_2001-12-31.nc"
SHAPEFILE_PATH = r"C:/Users/colin/Documents/These/Recherche/Compound_ER/shapefiles/final_shp/shp_re.shp"
OUT_DIR = r"C:/Users/colin/Documents/These/Recherche/Compound_ER/preliminaries_figs/wind_method_comparison"

REF_PERIOD = ("1982-01-01", "2001-12-31")  # matches config.SHEAR_REF_PERIOD
# Seconds per daily ssrd accumulation step -- mirrors
# calculate_cf.ERA5_SSRD_ACCUM_SECONDS / _standardize_reanalysis_names.
ERA5_SSRD_ACCUM_SECONDS = 86400.0

METHODS = ["wind100", "shear_uniform", "shear_local"]
REFERENCE_METHOD = "wind100"


def load_reanalysis(src_dir, limit=None):
    """Open the regridded ERA5 archive and derive the variables the wcf/scf
    formulas need (rsds in W/m2, tas in degC, sfcWind = hypot(u10, v10))."""
    files = sorted(glob.glob(os.path.join(src_dir, "ERA5_daily_*.zarr")))
    if not files:
        raise FileNotFoundError(f"No ERA5_daily_*.zarr stores found in {src_dir}")
    if limit:
        files = files[:limit]
    print(f"Opening {len(files)} monthly zarr stores from {src_dir}")
    ds = xr.open_mfdataset(files, engine="zarr", combine="by_coords", chunks={})
    ds = ds.rename({"valid_time": "time"}).sortby("time")
    ds = ds.chunk({"time": -1, "latitude": 50, "longitude": 50})

    # ssrd (accumulated J/m2) -> rsds (W/m2 mean); mirrors
    # calculate_cf._standardize_reanalysis_names.
    ds["rsds"] = ds["ssrd"] / ERA5_SSRD_ACCUM_SECONDS
    # t2m (K) -> tas (degC); mirrors calculate_cf.calculate_ds_cf_reanalysis.
    ds["tas"] = ds["t2m"] - 273.15
    ds["sfcWind"] = np.hypot(ds["u10"], ds["v10"])
    return ds


def load_region_mask(shapefile_path, lat, lon):
    """Boolean DataArray, True where the grid-cell center falls inside the
    shapefile -- same approach as fit_local_shear.rasterize_region_mask."""
    shapefile = gpd.read_file(shapefile_path)
    transform = rasterio.transform.from_bounds(
        float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()),
        len(lon), len(lat),
    )
    mask = geometry_mask(
        geometries=shapefile.geometry, out_shape=(len(lat), len(lon)),
        transform=transform, invert=True, all_touched=True,
    )
    if lat.values[0] < lat.values[-1]:
        mask = mask[::-1, :]
    return xr.DataArray(mask, dims=(lat.dims[0], lon.dims[0]),
                         coords={lat.dims[0]: lat, lon.dims[0]: lon})


def compute_severity(compound_da, scf_da, wcf_da, scf_thr, wcf_thr):
    """
    Expected shortfall: mean positive deficit on compound-event days,
    aggregated yearly. Reproduces make_grid_files.compute_severity (that
    module imports xesmf at load time, so it can't be imported directly
    from this xesmf-free script).
    """
    deficit_scf = xr.where(scf_da < scf_thr, scf_thr - scf_da, 0)
    deficit_wcf = xr.where(wcf_da < wcf_thr, wcf_thr - wcf_da, 0)
    daily_deficit = deficit_scf + deficit_wcf
    masked = xr.where(compound_da == 1, daily_deficit, np.nan)
    return masked.resample(time="YE").mean().fillna(0)


def yearly_severity_for_method(ds, alpha, cfg, scf, ref_period, quantile):
    """
    Compute wcf for cfg.wind_method, derive the low-wind/low-solar compound
    day flag from each variable's own reference-period quantile (matching
    make_grid_files.load_gridded_data_compound's thresholding), and return
    yearly per-pixel severity.
    """
    alpha_arg = alpha if cfg.wind_method == "shear_local" else None
    wind_hub = get_hub_height_wind(ds, cfg, alpha=alpha_arg)
    wcf = compute_wind_potential_from_hub_wind(wind_hub, cfg)

    wcf_ref = wcf.sel(time=slice(*ref_period))
    scf_ref = scf.sel(time=slice(*ref_period))
    wcf_thr = wcf_ref.where(wcf_ref > 0).quantile(quantile, dim="time")
    scf_thr = scf_ref.where(scf_ref > 0).quantile(quantile, dim="time")

    low_wind = xr.where(wcf <= wcf_thr, 1, 0)
    low_solar = xr.where(scf <= scf_thr, 1, 0)
    compound = low_wind * low_solar

    severity = compute_severity(compound, scf, wcf, scf_thr, wcf_thr)
    severity["time"] = severity.time.dt.year
    severity = severity.rename({"time": "year"})
    return severity.compute()


def r2_score(y_true, y_pred):
    """Coefficient of determination of y_pred against y_true (sklearn
    convention: 1 - SS_res/SS_tot), computed over finite pairs only."""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if y_true.size < 2:
        return np.nan, 0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(r2), int(y_true.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_dir", default=SRC_DIR)
    ap.add_argument("--alpha_path", default=ALPHA_PATH)
    ap.add_argument("--shapefile", default=SHAPEFILE_PATH)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--ref_start", default=REF_PERIOD[0])
    ap.add_argument("--ref_end", default=REF_PERIOD[1])
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

    print("Loading alpha (local shear exponent):", args.alpha_path)
    alpha = xr.open_dataset(args.alpha_path)["alpha"]
    alpha = alpha.reindex(latitude=ds["latitude"], longitude=ds["longitude"],
                          method="nearest", tolerance=1e-6)

    print("Building solar potential (scf, shared across all wind methods)...")
    # Persisted once: the per-method loop below builds a separate wcf graph
    # on top of scf each time, so without persisting, dask would recompute
    # it (tas/rsds/sfcWind -> scf) from scratch for every wind_method.
    scf = compute_solar_cf(ds["tas"], ds["rsds"], ds["sfcWind"], cfg=DEFAULT_PVGIS_COEFFICIENTS)
    scf = scf.persist()

    severities = {}
    for method in METHODS:
        print(f"Computing severity for wind_method={method!r}...")
        cfg = DS_CFConfig(wind_method=method)
        severities[method] = yearly_severity_for_method(ds, alpha, cfg, scf, ref_period, args.quantile)
        print(f"  done ({severities[method].sizes['year']} years).")

    other_methods = [m for m in METHODS if m != REFERENCE_METHOD]

    # ------------------------------------------------------------------
    # 1. R2: long-term per-pixel mean severity, method vs wind100
    # ------------------------------------------------------------------
    mean_severity = {m: severities[m].mean(dim="year") for m in METHODS}

    fig, axes = plt.subplots(1, len(other_methods), figsize=(5.5 * len(other_methods), 5))
    axes = np.atleast_1d(axes)
    r2_results = {}
    for ax, method in zip(axes, other_methods):
        y_true = mean_severity[REFERENCE_METHOD].values.ravel()
        y_pred = mean_severity[method].values.ravel()
        r2, n = r2_score(y_true, y_pred)
        r2_results[method] = r2
        ax.scatter(y_true, y_pred, s=4, alpha=0.3)
        lims = [0, np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)])]
        ax.plot(lims, lims, "k--", lw=1, label="1:1")
        ax.set_xlabel(f"Mean severity ({REFERENCE_METHOD})")
        ax.set_ylabel(f"Mean severity ({method})")
        ax.set_title(f"{method} vs {REFERENCE_METHOD}\nR2={r2:.3f}, n={n}")
        ax.legend()
    fig.tight_layout()
    r2_fig_path = os.path.join(args.out_dir, "severity_r2_scatter.png")
    fig.savefig(r2_fig_path, dpi=150)
    print("Wrote", r2_fig_path)

    # ------------------------------------------------------------------
    # 2. Spearman: per-pixel relative change in mean severity,
    #    ref_period -> rest of record, method vs wind100
    # ------------------------------------------------------------------
    ref_start_year = int(pd.Timestamp(ref_period[0]).year)
    ref_end_year = int(pd.Timestamp(ref_period[1]).year)

    rel_change = {}
    for method in METHODS:
        sev = severities[method]
        ref_mean = sev.sel(year=slice(ref_start_year, ref_end_year)).mean(dim="year")
        post_mean = sev.sel(year=slice(ref_end_year + 1, None)).mean(dim="year")
        rel_change[method] = xr.where(ref_mean > 0, (post_mean - ref_mean) / ref_mean, np.nan)

    spearman_results = {}
    fig2, axes2 = plt.subplots(1, len(other_methods), figsize=(5.5 * len(other_methods), 5))
    axes2 = np.atleast_1d(axes2)
    for ax, method in zip(axes2, other_methods):
        x = rel_change[REFERENCE_METHOD].values.ravel()
        y = rel_change[method].values.ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        rho, pval = stats.spearmanr(x[valid], y[valid])
        spearman_results[method] = (float(rho), float(pval), int(valid.sum()))
        print(f"Spearman rho({method} rel. change, {REFERENCE_METHOD} rel. change) "
              f"= {rho:.3f} (p={pval:.2e}, n={int(valid.sum())})")

        ax.scatter(x[valid], y[valid], s=4, alpha=0.3)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_xlabel(f"Relative change in severity ({REFERENCE_METHOD})")
        ax.set_ylabel(f"Relative change in severity ({method})")
        ax.set_title(f"{method} vs {REFERENCE_METHOD}\nSpearman rho={rho:.3f}, n={int(valid.sum())}")
    fig2.tight_layout()
    spearman_fig_path = os.path.join(args.out_dir, "severity_relchange_spearman_scatter.png")
    fig2.savefig(spearman_fig_path, dpi=150)
    print("Wrote", spearman_fig_path)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    summary = pd.DataFrame({
        "wind_method": other_methods,
        "R2_mean_severity_vs_wind100": [r2_results[m] for m in other_methods],
        "spearman_rho_relchange_vs_wind100": [spearman_results[m][0] for m in other_methods],
        "spearman_pvalue": [spearman_results[m][1] for m in other_methods],
        "n_pixels": [spearman_results[m][2] for m in other_methods],
    })
    summary_path = os.path.join(args.out_dir, "wind_method_comparison_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()

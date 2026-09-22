# -*- coding: cp1252 -*-
"""
Standalone diagnostic: does excluding zero-capacity-factor days from the
threshold quantile change the annual compound-event indices much?

Production code (calculate_cf.py's _score_branch, make_grid_files.py's
process_single_gcm) always computes the low-wind/low-solar threshold as
    wcf.where(wcf > 0).quantile(q, dim='time')
i.e. the q-th percentile of *non-zero* days only. This script checks how
much that choice actually matters by recomputing everything a second way,
    wcf.quantile(q, dim='time')
(zeros included), and comparing the resulting freq/dur/int/sev annual
compound-event indices -- both globally and by 20-degree latitude band
(edges at -90..90, one band centered on the equator: -10..10) -- along with
how many zero days per pixel get filtered out of the quantile in the first
place (for both wcf and scf).

Uses ERA5 on its native grid (via calculate_ds_cf_reanalysis's wcf_day_ERA5*/
scf_day_ERA5* output) over the standard 1982-2001 reference period, so the
comparison is grid/GCM-independent. Run this on the server, in the same
environment as calculate_cf.py.
"""
import os

import numpy as np
import pandas as pd
import xarray as xr

import config
from io_utils import match_files, open_dataset_any
from calculate_cf import _compound_indices

REF_PERIOD = ('1982-01-01', '2001-12-31')
Q = 0.1

# 20-degree latitude bands, edges chosen so one band is centered on the
# equator (-10..10) rather than straddling it at an odd offset.
LAT_EDGES = [-90, -70, -50, -30, -10, 10, 30, 50, 70, 90]
LAT_LABELS = [f"{LAT_EDGES[i]}..{LAT_EDGES[i + 1]}" for i in range(len(LAT_EDGES) - 1)]


def load_era5_cf(path_preprocessed, reanalysis):
    base = os.path.join(path_preprocessed, reanalysis)
    wcf_files, _ = match_files(
        os.path.join(base, f"wcf_day_{reanalysis}_historical_reanalysis_19790101-20191231"))
    scf_files, _ = match_files(
        os.path.join(base, f"scf_day_{reanalysis}_historical_reanalysis_19790101-20191231"))
    if not wcf_files or not scf_files:
        raise FileNotFoundError(
            f"No wcf_day/scf_day files found under {base}. "
            "Run calculate_ds_cf_reanalysis (calculate_cf.py) first."
        )
    # Open lazily (cheap: just metadata) so .sel(ref_period) only reads the
    # ~7300 needed days off disk, then .load() that slice into plain numpy
    # arrays. _compound_indices -> duration_xr does eager boolean-mask
    # indexing (`.where(..., drop=True)`) that dask doesn't support
    # ("Indexing with a boolean dask array is not allowed") -- production
    # (make_grid_files.py's process_single_gcm) never hits this because it
    # opens wcf/scf with no chunks= at all, i.e. already eager.
    wcf = open_dataset_any(wcf_files[0], chunks={'time': -1}).sel(time=slice(*REF_PERIOD)).load()
    scf = open_dataset_any(scf_files[0], chunks={'time': -1}).sel(time=slice(*REF_PERIOD)).load()
    return wcf.sortby('lat').sortby('lon'), scf.sortby('lat').sortby('lon')


def quantile_threshold(da, q, exclude_zero):
    src = da.where(da > 0) if exclude_zero else da
    return src.quantile(q, dim='time')


def zero_day_stats(da):
    """
    Per-pixel count/fraction of exactly-zero time steps -- what
    exclude_zero=True removes from the quantile sample. Pixels with no
    valid (non-NaN) time steps at all (e.g. outside the reanalysis's masked
    domain) are returned as NaN rather than 0: `da == 0` is False on NaN, so
    a naive sum would silently count "no data here" as "never zero here"
    and pull the average down wherever the grid is masked out.
    """
    n_valid = da.notnull().sum('time')
    n_zero = ((da == 0) & da.notnull()).sum('time')
    n_zero = n_zero.where(n_valid > 0)
    frac_zero = n_zero / n_valid.where(n_valid > 0)
    return n_zero, frac_zero


def mean_by_band(da):
    """
    Unweighted mean of da (any combination of year/lat/lon dims) collapsed
    over every dim except lat, grouped into 20-degree latitude bands
    (LAT_EDGES/LAT_LABELS), plus the overall global mean over all dims.
    Grid cells are weighted equally (no cos(lat) area weighting) -- fine for
    comparing two threshold definitions on the same grid, not for an
    absolute climatology.
    """
    da = da.compute()
    global_mean = float(da.mean(skipna=True))
    df = da.rename('value').to_dataframe().reset_index()
    df = df.dropna(subset=['value'])
    df['band'] = pd.cut(df['lat'], bins=LAT_EDGES, labels=LAT_LABELS, include_lowest=True)
    band_means = df.groupby('band', observed=True)['value'].mean().reindex(LAT_LABELS)
    return global_mean, band_means


def safe_pct(diff_series, base_series):
    return (diff_series / base_series.replace(0, np.nan)) * 100


if __name__ == '__main__':
    path_preprocessed = config.PATH_PREPROCESSED
    reanalysis = config.REANALYSIS
    out_dir = getattr(config, 'TEMP_FOLDER', os.getcwd())
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading native-grid {reanalysis} wcf/scf, ref_period={REF_PERIOD}")
    wcf, scf = load_era5_cf(path_preprocessed, reanalysis)
    n_time = wcf.sizes['time']
    print(f"{n_time} daily time steps in ref_period\n")

    # ------------------------------------------------------------------
    # 1. How many zero days per pixel does exclude_zero actually filter
    #    out of the quantile, for wind and solar, globally and by band?
    # ------------------------------------------------------------------
    print("=== Zero-value days excluded from the quantile sample (per pixel, mean) ===")
    zero_rows = []
    for label, var, da in [('wind (wcf)', 'wcf', wcf['wcf']), ('solar (scf)', 'scf', scf['scf'])]:
        n_zero, frac_zero = zero_day_stats(da)
        g_n, band_n = mean_by_band(n_zero)
        g_f, band_f = mean_by_band(frac_zero * 100)

        print(f"\n{label}: global mean = {g_n:.1f} zero days / {n_time} ({g_f:.1f}%)")
        tbl = pd.DataFrame({'mean_zero_days': band_n, 'pct_zero_days': band_f})
        print(tbl.to_string(float_format=lambda x: f"{x:.1f}"))

        zero_rows.append({'var': var, 'scope': 'global', 'mean_zero_days': g_n, 'pct_zero_days': g_f})
        for band in LAT_LABELS:
            zero_rows.append({
                'var': var, 'scope': band,
                'mean_zero_days': band_n.get(band, np.nan),
                'pct_zero_days': band_f.get(band, np.nan),
            })

    zero_csv = os.path.join(out_dir, 'zero_quantile_excluded_days.csv')
    pd.DataFrame(zero_rows).to_csv(zero_csv, index=False)
    print(f"\nSaved zero-day stats to: {zero_csv}")

    # ------------------------------------------------------------------
    # 2. Threshold itself: how much does including zero shift wcf_thr/
    #    scf_thr (before even getting to freq/dur/int/sev)?
    # ------------------------------------------------------------------
    wcf_thr = {excl: quantile_threshold(wcf['wcf'], Q, excl) for excl in (True, False)}
    scf_thr = {excl: quantile_threshold(scf['scf'], Q, excl) for excl in (True, False)}

    print("\n=== Threshold shift from including zero (incl_zero - excl_zero) ===")
    for label, thr in [('wcf_thr', wcf_thr), ('scf_thr', scf_thr)]:
        diff = thr[False] - thr[True]
        rel = xr.where(thr[True] != 0, diff / thr[True], np.nan) * 100
        g_abs, band_abs = mean_by_band(diff)
        g_rel, band_rel = mean_by_band(rel)
        print(f"\n{label}: global mean diff = {g_abs:+.4f} ({g_rel:+.1f}%)")
        tbl = pd.DataFrame({'abs_diff': band_abs, 'pct_diff': band_rel})
        print(tbl.to_string(float_format=lambda x: f"{x:+.4f}"))

    # ------------------------------------------------------------------
    # 3. freq/dur/int/sev under both threshold definitions
    #    (mirrors calculate_cf.py's _compound_indices exactly).
    # ------------------------------------------------------------------
    results = {}
    for exclude_zero in (True, False):
        freq, dur, intensity, sev = _compound_indices(
            wcf, scf, wcf_thr[exclude_zero], scf_thr[exclude_zero])
        results[exclude_zero] = dict(freq=freq, dur=dur, int=intensity, sev=sev)

    print("\n=== freq/dur/int/sev: excl_zero (production) vs incl_zero ===")
    summary_rows = []
    for metric in ['freq', 'dur', 'int', 'sev']:
        g_excl, band_excl = mean_by_band(results[True][metric])
        g_incl, band_incl = mean_by_band(results[False][metric])
        g_diff = g_incl - g_excl
        g_pct = (g_diff / g_excl * 100) if g_excl else np.nan
        band_diff = band_incl - band_excl
        band_pct = safe_pct(band_diff, band_excl)

        print(f"\n--- {metric} ---")
        print(f"Global: excl_zero={g_excl:.4f}, incl_zero={g_incl:.4f}, "
              f"diff={g_diff:+.4f} ({g_pct:+.1f}%)")
        tbl = pd.DataFrame({
            'excl_zero': band_excl, 'incl_zero': band_incl,
            'abs_diff': band_diff, 'pct_diff': band_pct,
        })
        print(tbl.to_string(float_format=lambda x: f"{x:.4f}"))

        summary_rows.append({
            'metric': metric, 'scope': 'global',
            'excl_zero': g_excl, 'incl_zero': g_incl,
            'abs_diff': g_diff, 'pct_diff': g_pct,
        })
        for band in LAT_LABELS:
            summary_rows.append({
                'metric': metric, 'scope': band,
                'excl_zero': band_excl.get(band, np.nan),
                'incl_zero': band_incl.get(band, np.nan),
                'abs_diff': band_diff.get(band, np.nan),
                'pct_diff': band_pct.get(band, np.nan),
            })

    summary_csv = os.path.join(out_dir, 'zero_quantile_sensitivity_indices.csv')
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\nSaved freq/dur/int/sev sensitivity summary to: {summary_csv}")

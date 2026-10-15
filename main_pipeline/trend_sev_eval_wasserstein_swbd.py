# -*- coding: cp1252 -*-
"""
Empirical Wasserstein (W2) distance between each GCM run's bootstrap
distribution of SWBD-severity trends and ERA5's, per region (poly_idx).

SWBD analogue of trend_sev_eval_wasserstein.wasserstein_empirical_agg().
Annual SWBD severity is the yearly sum of the residual load above its
threshold (make_rl_files.py's demand and residual-load definitions):
  - GCM: 40-year GWL0-61 + GWL1 series; threshold (quantile `thr` of the
    daily residual load) and demand baseline from GWL0-61.
  - ERA5: full-period wcf/scf/pop-weighted tas_agg_ref_full_* (built by
    calculate_cf.aggregate_ref_full_GCM on the GCM grid), with that GCM run's
    GWL0-61 threshold and demand baseline.
Bootstrap and W2 helpers are imported from trend_sev_eval_wasserstein.py.

Output: agg_wasserstein_empirical_swbd_GCMs_aggregated_{reanalysis}_{suffix_shp}{tag}.nc,
dims (realization, poly_idx): w2_distance and w2_normalized (divided by
ERA5's bootstrap trend std; NaN below SIGMA_REF_MIN).
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; puts the code folders on sys.path

import xarray as xr
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from trend_sev_eval_wasserstein import (
    SIGMA_REF_MIN, _parse_exclude_gcm_run, _iter_gcm_runs_agg,
    stationary_bootstrap_samples_grid, empirical_w2,
)
from make_rl_files import (
    _load_data, _daily_rl, DEMAND_CONFIGS, MAIN_THR, MAIN_TOT_RE, MAIN_MIX,
)


# ---------------------------------------------------------------------------
# Annual SWBD severity
# ---------------------------------------------------------------------------

def _to_dates(da):
    da = da.copy()
    da['time'] = pd.to_datetime(da.time.dt.strftime('%Y-%m-%d').values)
    return da


def _swbd_annual(rl, threshold):
    """Yearly sum of the residual load above `threshold`, on a 'year' dim."""
    sev = xr.where(rl > threshold, rl - threshold, 0.0).resample(time='YE').sum()
    sev['time'] = sev.time.dt.year
    return sev.rename({'time': 'year'})


def _cf_mix(scf, wcf, df_share, mix):
    """Solar/wind mix as in make_rl_files._load_data."""
    col = "future_ratio" if (mix == "future" and "future_ratio" in df_share.columns) else "current_ratio"
    share = xr.DataArray(df_share[col].values, coords=[df_share["poly_idx"].values], dims=["poly_idx"])
    return share * scf + (1 - share) * wcf


def preprocess_single_swbd_agg(preprocessed_path, GCM, run, df_share, reanalysis=None,
                               suffix_shp=None, thr=MAIN_THR, tot_re=MAIN_TOT_RE, mix=MAIN_MIX,
                               period="Annual", demand_cfg=None):
    """40-year GWL0-61 + GWL1 annual SWBD severity of one GCM run, plus the
    GWL0-61 threshold, demand baseline and mean CF that define it."""
    reanalysis = reanalysis or config.REANALYSIS
    suffix_shp = suffix_shp or config.AGREEMENT_SUFFIX_SHP
    load = lambda gwl: _load_data(GCM, run, config.SSP, gwl, reanalysis, suffix_shp,
                                  preprocessed_path, df_share, mix)

    tas_061, ds_cf_061, ds_cf_mean = load('GWL0-61')
    tas_1, ds_cf_1, _ = load('GWL1')

    rl_061, demand_bas = _daily_rl(tas_061, ds_cf_061, ds_cf_mean, period, tot_re,
                                   demand_cfg=demand_cfg)
    threshold = rl_061.quantile(thr, dim="time").drop_vars('quantile')
    rl_1, _ = _daily_rl(tas_1, ds_cf_1, ds_cf_mean, period, tot_re,
                        demand_bas=demand_bas, demand_cfg=demand_cfg)

    sev_061 = _swbd_annual(rl_061, threshold)
    sev_1 = _swbd_annual(rl_1, threshold)

    # Same stitching as trend_sev_eval_wasserstein._severity_from_cf.
    n_years_061 = sev_061.year.size
    sev_1 = sev_1.isel(year=slice(0, 40 - n_years_061))
    sev_061['year'] = np.arange(n_years_061)
    sev_1['year'] = np.arange(sev_1.year.size) + n_years_061
    sev = xr.concat([sev_061, sev_1], dim='year')
    return sev, threshold, demand_bas, ds_cf_mean


def _build_ref_swbd_agg(preprocessed_path, GCM, df_share, threshold, demand_bas, ds_cf_mean,
                        reanalysis=None, suffix_shp=None, tot_re=MAIN_TOT_RE, mix=MAIN_MIX,
                        period="Annual", demand_cfg=None):
    """Full-period ERA5 annual SWBD severity, filtered with the GCM run's
    GWL0-61 threshold and demand baseline (as _build_ref_severity_agg does
    with the wcf/scf thresholds)."""
    reanalysis = reanalysis or config.REANALYSIS
    suffix_shp = suffix_shp or config.AGREEMENT_SUFFIX_SHP

    def _load(var, prefix):
        path = os.path.join(preprocessed_path, GCM,
                            f'{prefix}_ref_full_{GCM}_{reanalysis}_{suffix_shp}.nc')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} (run with --build-ref, or calculate_cf.aggregate_ref_full_GCM)")
        return _to_dates(xr.open_dataset(path)[var])

    tas = _load('tas', 'tas_pop_agg') - 273.15
    wcf = _load('wcf', 'wcf_agg')
    scf = _load('scf', 'scf_agg')
    tas, wcf, scf = xr.align(tas, wcf, scf, join='inner')

    rl, _ = _daily_rl(tas, _cf_mix(scf, wcf, df_share, mix), ds_cf_mean, period, tot_re,
                      demand_bas=demand_bas, demand_cfg=demand_cfg)
    return _swbd_annual(rl, threshold)


# ---------------------------------------------------------------------------
# Per-region empirical W2 pipeline
# ---------------------------------------------------------------------------

def _wasserstein_realization_swbd_agg(args):
    """Worker: one realization's W2 and normalized W2 per region."""
    (preprocessed_path, GCM, run, df_share, reanalysis, suffix_shp, thr, tot_re, mix,
     demand_cfg, n_boot, block_size, boot_batch, seed) = args
    rng = np.random.default_rng(seed)

    sev_gcm, threshold, demand_bas, ds_cf_mean = preprocess_single_swbd_agg(
        preprocessed_path, GCM, run, df_share, reanalysis, suffix_shp,
        thr=thr, tot_re=tot_re, mix=mix, demand_cfg=demand_cfg)
    sev_ref = _build_ref_swbd_agg(
        preprocessed_path, GCM, df_share, threshold, demand_bas, ds_cf_mean,
        reanalysis, suffix_shp, tot_re=tot_re, mix=mix, demand_cfg=demand_cfg)

    slopes_gcm = stationary_bootstrap_samples_grid(sev_gcm, n_boot=n_boot, block_size=block_size,
                                                    boot_batch=boot_batch, rng=rng)
    slopes_ref = stationary_bootstrap_samples_grid(sev_ref, n_boot=n_boot, block_size=block_size,
                                                    boot_batch=boot_batch, rng=rng)

    w2 = empirical_w2(slopes_gcm, slopes_ref)
    sigma_ref = slopes_ref.std(dim="boot", ddof=1)
    w2_norm = (w2 / sigma_ref.where(sigma_ref > SIGMA_REF_MIN)).rename("w2_normalized")

    ds = w2.to_dataset(name="w2_distance")
    ds["w2_normalized"] = w2_norm
    ds["GCM"] = GCM
    ds["run"] = run
    return GCM, run, ds


def _scenario_tag(thr, tot_re, mix, demand):
    if (thr, tot_re, mix, demand) == (MAIN_THR, MAIN_TOT_RE, MAIN_MIX, 'default'):
        return ''
    tag = f'_thr{thr}_re{tot_re}_{mix}'
    return tag if demand == 'default' else f'{tag}_demand-{demand}'


def wasserstein_empirical_swbd_agg(preprocessed_path, out_dir, df_share, reanalysis=None,
                                   exclude_gcm_run=None, suffix_shp=None,
                                   thr=MAIN_THR, tot_re=MAIN_TOT_RE, mix=MAIN_MIX,
                                   demand='default', n_boot=500, block_size=5,
                                   boot_batch=250, max_workers=4, seed=0):
    """Per-region W2 between each GCM run's and ERA5's bootstrap SWBD-severity
    trend distributions
    -> agg_wasserstein_empirical_swbd_GCMs_aggregated_{reanalysis}_{suffix_shp}{tag}.nc,
    dims (realization, poly_idx); tag is empty for the main scenario."""
    reanalysis = reanalysis or config.REANALYSIS
    suffix_shp = suffix_shp or config.AGREEMENT_SUFFIX_SHP
    exclude_gcm_run = _parse_exclude_gcm_run(exclude_gcm_run)
    demand_cfg = DEMAND_CONFIGS[demand]

    tasks = []
    for i, (GCM, run) in enumerate(
            _iter_gcm_runs_agg(preprocessed_path, config.SSP, reanalysis, suffix_shp, gwl='GWL1')):
        if (GCM, run) in exclude_gcm_run:
            print(f'    [excluded] {GCM} {run}')
            continue
        tasks.append((preprocessed_path, GCM, run, df_share, reanalysis, suffix_shp,
                      thr, tot_re, mix, demand_cfg, n_boot, block_size, boot_batch, seed + i))

    ds_final = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_wasserstein_realization_swbd_agg, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            GCM, run, ds = fut.result()
            print(f'[{i + 1}/{len(tasks)}] computed SWBD empirical W2 for {GCM} {run}')
            ds_final.append(ds.expand_dims({'realization': [i]}))

    out = xr.concat(ds_final, dim='realization')
    out.attrs.update({'thr': thr, 'tot_re': tot_re, 'mix': mix, 'demand': str(demand_cfg)})
    out_path = (f'{out_dir}/agg_wasserstein_empirical_swbd_GCMs_aggregated_'
                f'{reanalysis}_{suffix_shp}{_scenario_tag(thr, tot_re, mix, demand)}.nc')
    out.to_netcdf(out_path)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# ERA5 inputs
# ---------------------------------------------------------------------------

def build_ref_inputs(preprocessed_path, reanalysis=None, suffix_shp=None, exclude_gcm_run=None):
    """calculate_cf.aggregate_ref_full_GCM once per GCM (the grid is per GCM)."""
    from calculate_cf import aggregate_ref_full_GCM  # heavy, only needed here
    reanalysis = reanalysis or config.REANALYSIS
    suffix_shp = suffix_shp or config.AGREEMENT_SUFFIX_SHP
    exclude_gcm_run = _parse_exclude_gcm_run(exclude_gcm_run)

    done = set()
    for GCM, run in _iter_gcm_runs_agg(preprocessed_path, config.SSP, reanalysis,
                                       suffix_shp, gwl='GWL1'):
        if GCM in done or (GCM, run) in exclude_gcm_run:
            continue
        print(f'  Building ERA5 regional inputs on the {GCM} grid')
        aggregate_ref_full_GCM(GCM, run, config.SSP, os.path.join(preprocessed_path, ''),
                               config.PATH_FOLDER, config.TEMP_FOLDER, config.SHAPEFILE_PATH,
                               config.POP_PATH, reanalysis=reanalysis, suffix_shp=suffix_shp)
        done.add(GCM)


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--build-ref', action='store_true',
                         help="First build the full-period ERA5 regional inputs (*_agg_ref_full_*).")
    parser.add_argument('--thr', type=float, default=MAIN_THR,
                         help="Residual-load threshold quantile.")
    parser.add_argument('--tot-re', type=float, default=MAIN_TOT_RE,
                         help="Renewable penetration.")
    parser.add_argument('--mix', default=MAIN_MIX, choices=['current', 'future'])
    parser.add_argument('--demand', default='default', choices=list(DEMAND_CONFIGS),
                         help="Demand configuration (make_rl_files.DEMAND_CONFIGS).")
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--n-boot', type=int, default=500)
    parser.add_argument('--block-size', type=int, default=5)
    parser.add_argument('--boot-batch', type=int, default=250)
    parser.add_argument('--exclude_gcm_run', nargs='+', default=config.EXCLUDE_GCM_RUN,
                         help="GCM:run pairs to exclude.")
    parser.add_argument('--suffix-shp', default=config.AGREEMENT_SUFFIX_SHP,
                         help="Shapefile-version suffix of the *_agg_* files.")
    args = parser.parse_args()

    preprocessed_path = config.PATH_PREPROCESSED
    out_dir = os.path.join(config.PATH_PREPROCESSED, 'trend_evaluation')
    os.makedirs(out_dir, exist_ok=True)

    if args.build_ref:
        print("=== ERA5 full-period regional inputs (build_ref_inputs) ===")
        build_ref_inputs(preprocessed_path, suffix_shp=args.suffix_shp,
                         exclude_gcm_run=args.exclude_gcm_run)

    print("\n=== Aggregated-domain SWBD empirical W2 (wasserstein_empirical_swbd_agg) ===")
    wasserstein_empirical_swbd_agg(preprocessed_path, out_dir, pd.read_csv(config.SHARE_RENEWABLE_CSV),
                                   exclude_gcm_run=args.exclude_gcm_run, suffix_shp=args.suffix_shp,
                                   thr=args.thr, tot_re=args.tot_re, mix=args.mix, demand=args.demand,
                                   n_boot=args.n_boot, block_size=args.block_size,
                                   boot_batch=args.boot_batch, max_workers=args.max_workers)

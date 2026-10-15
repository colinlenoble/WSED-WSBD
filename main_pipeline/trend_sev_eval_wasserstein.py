# -*- coding: cp1252 -*-
"""
Empirical Wasserstein (W2) distance between each GCM run's bootstrap
distribution of severity trends and ERA5's, used to weight the ensemble.

Both distributions are computed on the GCM's native grid (ERA5 there is the
wcf_ref_*/scf_ref_* regridded by calculate_cf.py), keeping every bootstrap
replicate. Low-level helpers are copied from trend_sev_eval.py so this
module stays self-contained.

Pipelines:
  - wasserstein_empirical_grid(): per pixel, then put on the ERA5 grid
    (nearest_s2d); w2_distance and w2_normalized (divided by ERA5's
    bootstrap trend std).
  - wasserstein_empirical_agg(): the same per region (poly_idx).
  - region_empirical_trend_samples(): bootstrap trend samples of region-mean
    severity (ED Fig. 3).
"""
import os
import glob
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; puts the code folders on sys.path
os.environ['ESMFMKFILE'] = config.ESMFMKFILE_XENV

import xesmf as xe
import xarray as xr
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from io_utils import match_files, glob_any, open_dataset_any

# Below this ERA5 bootstrap-trend std (real values ~1e-3..2e-2; ~0 where
# severity is always zero), w2_normalized is NaN instead of exploding.
SIGMA_REF_MIN = 1e-6


# ---------------------------------------------------------------------------
# Helpers copied from trend_sev_eval.py
# ---------------------------------------------------------------------------

def _parse_exclude_gcm_run(exclude_gcm_run):
    if exclude_gcm_run is None:
        exclude_gcm_run = config.EXCLUDE_GCM_RUN
    return set(tuple(x.split(':')) for x in exclude_gcm_run)


def _pos_quantile(da, q):
    """q-th quantile of strictly positive values; returns 0 where no positive values exist."""
    pos = da.where(da > 0)
    thr = pos.quantile(q, dim="time")
    return thr.where(pos.notnull().any(dim="time"), 0)


def _target_grid(preprocessed_path, reanalysis):
    """Reanalysis wcf_day_* grid, onto which per-pixel outputs are regridded."""
    files, _ = match_files(os.path.join(preprocessed_path, reanalysis, 'wcf_day*'))
    if not files:
        raise FileNotFoundError(
            f"No wcf_day_* reanalysis file under {os.path.join(preprocessed_path, reanalysis)}"
        )
    return open_dataset_any(files[0]).isel(time=slice(0, 2))


def compute_severity(comp_da, scf_ds, wcf_ds, scf_thr, wcf_thr):
    """Mean deficit (intensity) on compound-event days, aggregated yearly."""
    deficit_scf = xr.where(scf_ds["scf"] <= scf_thr, scf_thr - scf_ds["scf"], 0)
    deficit_wcf = xr.where(wcf_ds["wcf"] <= wcf_thr, wcf_thr - wcf_ds["wcf"], 0)
    daily_deficit = deficit_scf + deficit_wcf
    masked = xr.where(comp_da == 1, daily_deficit, np.nan)
    return masked.resample(time="YE").mean().fillna(0)


def duration_xr(da):
    """Mean event duration and event frequency per (year, lat, lon). Returns ds, ds_freq."""
    da = da.convert_calendar('standard')
    da = da.sortby('lat').sortby('lon')
    da['lat'] = da['lat'].astype(float)
    da['lon'] = da['lon'].astype(float)

    first_time = pd.Timestamp(da.time[0].values)
    da_dur = xr.concat([
        xr.zeros_like(da.isel(time=0)).expand_dims(time=[first_time - pd.Timedelta(days=1)]),
        da,
    ], dim='time')

    start_event = (da_dur.diff(dim='time', label='lower') > 0)
    start_event['time'] = da.time
    start_event['year'] = start_event.time.dt.year
    id_event = start_event.cumsum(dim='time') * da
    id_event = id_event.where(id_event > 0)

    stacked = id_event.stack(z=('lat', 'lon', 'time'))
    stacked = stacked.where(stacked.notnull(), drop=True)

    event_ids = stacked.values.astype(int)
    lat_idxs = stacked['lat'].values
    lon_idxs = stacked['lon'].values
    year_idxs = stacked['year'].values.astype(int)

    df = pd.DataFrame({'event_id': event_ids, 'lat': lat_idxs, 'lon': lon_idxs, 'year': year_idxs})
    df['year'] = df.groupby(['event_id', 'lat', 'lon'])['year'].transform('min')

    combined_keys = (
        df['event_id'].astype(str) + ';' +
        df['lat'].astype(str) + ';' +
        df['lon'].astype(str) + ';' +
        df['year'].astype(str)
    )
    unique_keys, counts = np.unique(combined_keys.values, return_counts=True)
    ids, lats, lons, years = zip(*(k.split(';') for k in unique_keys))

    dur_da = xr.DataArray(
        counts,
        dims='event_instance',
        coords={
            'event_instance': np.arange(len(counts)),
            'event_id': ('event_instance', np.array(ids, dtype=int)),
            'lat': ('event_instance', np.array(lats, dtype=float)),
            'lon': ('event_instance', np.array(lons, dtype=float)),
            'year': ('event_instance', np.array(years, dtype=int)),
        },
    ).to_dataset(name='duration')

    df_dur = dur_da.to_dataframe()
    ds = df_dur.groupby(['year', 'lat', 'lon']).mean().to_xarray()[['duration']]
    ds_freq = df_dur.groupby(['year', 'lat', 'lon']).count().to_xarray()
    ds_freq = ds_freq['duration'].to_dataset(name='frequency')

    ds['duration'] = ds['duration'].fillna(0)
    ds_freq['frequency'] = ds_freq['frequency'].fillna(0)
    return ds, ds_freq


def duration_xr_by_poly(da):
    """duration_xr() for region-aggregated data (single 'poly_idx' dim)."""
    da = da.convert_calendar('standard')

    first_time = pd.Timestamp(da.time[0].values)
    da_dur = xr.concat([
        xr.zeros_like(da.isel(time=0)).expand_dims(time=[first_time - pd.Timedelta(days=1)]),
        da,
    ], dim='time')

    start_event = (da_dur.diff(dim='time', label='lower') > 0)
    start_event['time'] = da.time
    start_event['year'] = start_event.time.dt.year
    id_event = start_event.cumsum(dim='time') * da
    id_event = id_event.where(id_event > 0)

    stacked = id_event.stack(z=('poly_idx', 'time'))
    stacked = stacked.where(stacked.notnull(), drop=True)

    event_ids = stacked.values.astype(int)
    poly_idxs = stacked['poly_idx'].values
    year_idxs = stacked['year'].values.astype(int)

    df = pd.DataFrame({'event_id': event_ids, 'poly_idx': poly_idxs, 'year': year_idxs})
    df['year'] = df.groupby(['event_id', 'poly_idx'])['year'].transform('min')

    combined_keys = (
        df['event_id'].astype(str) + ';' +
        df['poly_idx'].astype(str) + ';' +
        df['year'].astype(str)
    )
    unique_keys, counts = np.unique(combined_keys.values, return_counts=True)
    ids, poly_idxs_u, years = zip(*(k.split(';') for k in unique_keys))

    dur_da = xr.DataArray(
        counts,
        dims='event_instance',
        coords={
            'event_instance': np.arange(len(counts)),
            'event_id': ('event_instance', np.array(ids, dtype=int)),
            'poly_idx': ('event_instance', np.array(poly_idxs_u, dtype=int)),
            'year': ('event_instance', np.array(years, dtype=int)),
        },
    ).to_dataset(name='duration')

    df_dur = dur_da.to_dataframe()
    ds = df_dur.groupby(['year', 'poly_idx']).mean().to_xarray()[['duration']]
    ds_freq = df_dur.groupby(['year', 'poly_idx']).count().to_xarray()
    ds_freq = ds_freq['duration'].to_dataset(name='frequency')

    ds['duration'] = ds['duration'].fillna(0)
    ds_freq['frequency'] = ds_freq['frequency'].fillna(0)
    return ds, ds_freq


def _severity_from_cf(wcf_gwl061, scf_gwl061, wcf_gwl1, scf_gwl1, duration_fn=duration_xr):
    """GWL0-61 thresholds -> compound days -> annual severity, as a 40-year
    GWL0-61 + GWL1 series (duration_fn: gridded or per-region)."""
    wcf_thr = _pos_quantile(wcf_gwl061.wcf, 0.1)
    scf_thr = _pos_quantile(scf_gwl061.scf, 0.1)

    for ds, wv, sv in [(wcf_gwl061, 'low_wind', None),
                       (wcf_gwl1, 'low_wind', None),
                       (scf_gwl061, None, 'low_solar'),
                       (scf_gwl1, None, 'low_solar')]:
        if wv:
            ds[wv] = xr.where(ds.wcf <= wcf_thr, 1, 0)
        else:
            ds[sv] = xr.where(ds.scf <= scf_thr, 1, 0)

    wcf_gwl061 = wcf_gwl061.convert_calendar('standard')
    wcf_gwl1 = wcf_gwl1.convert_calendar('standard')
    scf_gwl061 = scf_gwl061.convert_calendar('standard')
    scf_gwl1 = scf_gwl1.convert_calendar('standard')

    compound_gwl061 = (wcf_gwl061['low_wind'] * scf_gwl061['low_solar']).to_dataset(name='compound')
    compound_gwl1 = (wcf_gwl1['low_wind'] * scf_gwl1['low_solar']).to_dataset(name='compound')

    intensity_gwl061 = compute_severity(compound_gwl061.compound, scf_gwl061, wcf_gwl061, scf_thr, wcf_thr)
    intensity_gwl1 = compute_severity(compound_gwl1.compound, scf_gwl1, wcf_gwl1, scf_thr, wcf_thr)

    for da in (intensity_gwl061, intensity_gwl1):
        da['time'] = da.time.dt.year

    intensity_gwl061 = intensity_gwl061.rename({'time': 'year'})
    intensity_gwl1 = intensity_gwl1.rename({'time': 'year'})

    dur_gwl061, freq_gwl061 = duration_fn(compound_gwl061.compound)
    dur_gwl1, freq_gwl1 = duration_fn(compound_gwl1.compound)

    dur_gwl061 = dur_gwl061.reindex_like(intensity_gwl061, fill_value=0)
    dur_gwl1 = dur_gwl1.reindex_like(intensity_gwl1, fill_value=0)
    freq_gwl061 = freq_gwl061.reindex_like(intensity_gwl061, fill_value=0)
    freq_gwl1 = freq_gwl1.reindex_like(intensity_gwl1, fill_value=0)

    sev_gwl061 = (intensity_gwl061 * dur_gwl061.duration * freq_gwl061.frequency)
    sev_gwl1 = (intensity_gwl1 * dur_gwl1.duration * freq_gwl1.frequency)

    n_years_gwl061 = sev_gwl061.year.size
    n_years_gwl1 = 40 - n_years_gwl061
    sev_gwl1 = sev_gwl1.isel(year=slice(0, n_years_gwl1))

    sev_gwl1['year'] = sev_gwl1.year + n_years_gwl061
    return xr.concat([sev_gwl061, sev_gwl1], dim='year')


def preprocess_single_sev(preprocessed_path, GCM, run, reanalysis=None):
    """40-year GWL0-61 + GWL1 severity series on the GCM's native grid."""
    reanalysis = reanalysis or config.REANALYSIS

    def _load(var, gwl):
        files, _ = match_files(
            os.path.join(preprocessed_path, GCM, f'{var}_day_{GCM}_{config.SSP}_{run}_{gwl}_{reanalysis}'))
        if not files:
            raise FileNotFoundError(
                f"No {var}_day_{GCM}_{config.SSP}_{run}_{gwl}_{reanalysis}.[nc|zarr] found")
        return open_dataset_any(files[0])

    wcf_gwl061 = _load('wcf', 'GWL0-61')
    scf_gwl061 = _load('scf', 'GWL0-61')
    wcf_gwl1 = _load('wcf', 'GWL1')
    scf_gwl1 = _load('scf', 'GWL1')

    return _severity_from_cf(wcf_gwl061, scf_gwl061, wcf_gwl1, scf_gwl1, duration_fn=duration_xr)


def preprocess_single_sev_agg(preprocessed_path, GCM, run, reanalysis=None, suffix_shp=None):
    """preprocess_single_sev() from region-aggregated wcf_agg_*/scf_agg_*."""
    reanalysis = reanalysis or config.REANALYSIS
    suffix_shp = suffix_shp or config.AGREEMENT_SUFFIX_SHP

    def _load(var, gwl):
        path = os.path.join(
            preprocessed_path, GCM,
            f'{var}_agg_{GCM}_{config.SSP}_{run}_{gwl}_{reanalysis}_{suffix_shp}.nc')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return xr.open_dataset(path)

    wcf_gwl061 = _load('wcf', 'GWL0-61')
    scf_gwl061 = _load('scf', 'GWL0-61')
    wcf_gwl1 = _load('wcf', 'GWL1')
    scf_gwl1 = _load('scf', 'GWL1')

    return _severity_from_cf(wcf_gwl061, scf_gwl061, wcf_gwl1, scf_gwl1, duration_fn=duration_xr_by_poly)


def _build_ref_severity_agg(preprocessed_path, GCM, run, reanalysis, suffix_shp):
    """_build_ref_severity_native() from region-aggregated wcf/scf_agg_ref_*."""
    def _load(name):
        path = os.path.join(preprocessed_path, GCM, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return xr.open_dataset(path)

    wcf_gwl061 = _load(f'wcf_agg_{GCM}_{config.SSP}_{run}_GWL0-61_{reanalysis}_{suffix_shp}.nc')
    scf_gwl061 = _load(f'scf_agg_{GCM}_{config.SSP}_{run}_GWL0-61_{reanalysis}_{suffix_shp}.nc')
    wcf_ref = _load(f'wcf_agg_ref_{GCM}_{reanalysis}_{suffix_shp}.nc')
    scf_ref = _load(f'scf_agg_ref_{GCM}_{reanalysis}_{suffix_shp}.nc')

    wcf_ref['time'] = pd.to_datetime(wcf_ref.time.dt.strftime('%Y-%m-%d').values)
    scf_ref['time'] = pd.to_datetime(scf_ref.time.dt.strftime('%Y-%m-%d').values)

    wcf_thr = _pos_quantile(wcf_gwl061.wcf, 0.1)
    scf_thr = _pos_quantile(scf_gwl061.scf, 0.1)

    wcf_ref['low_wind'] = xr.where(wcf_ref.wcf <= wcf_thr, 1, 0)
    scf_ref['low_solar'] = xr.where(scf_ref.scf <= scf_thr, 1, 0)
    compound_ref = (wcf_ref.low_wind * scf_ref.low_solar).to_dataset(name='start_cooc')

    wcf_ref = wcf_ref.convert_calendar('standard')
    scf_ref = scf_ref.convert_calendar('standard')

    intensity_ref = compute_severity(compound_ref.start_cooc, scf_ref, wcf_ref, scf_thr, wcf_thr)
    intensity_ref['time'] = intensity_ref.time.dt.year
    intensity_ref = intensity_ref.rename({'time': 'year'})

    ds_dur, ds_freq = duration_xr_by_poly(compound_ref.start_cooc)
    ds_dur = ds_dur.reindex_like(intensity_ref, fill_value=0)
    ds_freq = ds_freq.reindex_like(intensity_ref, fill_value=0)

    return (intensity_ref * ds_dur.duration * ds_freq.frequency).rename('sev')


def _build_ref_severity_native(preprocessed_path, GCM, run, reanalysis):
    """ERA5 reference severity with this GCM run's GWL0-61 thresholds, on the
    GCM's native grid (wcf_ref_*/scf_ref_* are already ERA5 regridded there)."""
    wcf_gwl061_files, _ = match_files(
        os.path.join(preprocessed_path, GCM, f'wcf_day_{GCM}_{config.SSP}_{run}_GWL0-61_{reanalysis}'))
    scf_gwl061_files, _ = match_files(
        os.path.join(preprocessed_path, GCM, f'scf_day_{GCM}_{config.SSP}_{run}_GWL0-61_{reanalysis}'))
    wcf_ref_files, _ = match_files(
        os.path.join(preprocessed_path, GCM, f'wcf_ref_{GCM}_{reanalysis}'))
    scf_ref_files, _ = match_files(
        os.path.join(preprocessed_path, GCM, f'scf_ref_{GCM}_{reanalysis}'))

    wcf_gwl061 = open_dataset_any(wcf_gwl061_files[0])
    scf_gwl061 = open_dataset_any(scf_gwl061_files[0])
    wcf_ref = open_dataset_any(wcf_ref_files[0])
    scf_ref = open_dataset_any(scf_ref_files[0])

    wcf_ref['time'] = pd.to_datetime(wcf_ref.time.dt.strftime('%Y-%m-%d').values)
    scf_ref['time'] = pd.to_datetime(scf_ref.time.dt.strftime('%Y-%m-%d').values)

    wcf_thr = _pos_quantile(wcf_gwl061.wcf, 0.1)
    scf_thr = _pos_quantile(scf_gwl061.scf, 0.1)

    wcf_ref['low_wind'] = xr.where(wcf_ref.wcf <= wcf_thr, 1, 0)
    scf_ref['low_solar'] = xr.where(scf_ref.scf <= scf_thr, 1, 0)
    compound_ref = (wcf_ref.low_wind * scf_ref.low_solar).to_dataset(name='start_cooc')

    wcf_ref = wcf_ref.convert_calendar('standard')
    scf_ref = scf_ref.convert_calendar('standard')

    intensity_ref = compute_severity(compound_ref.start_cooc, scf_ref, wcf_ref, scf_thr, wcf_thr)
    intensity_ref['time'] = intensity_ref.time.dt.year
    intensity_ref = intensity_ref.rename({'time': 'year'})

    ds_dur, ds_freq = duration_xr(compound_ref.start_cooc)
    ds_dur = ds_dur.reindex_like(intensity_ref, fill_value=0)
    ds_freq = ds_freq.reindex_like(intensity_ref, fill_value=0)

    return (intensity_ref * ds_dur.duration * ds_freq.frequency).rename('sev')


# ---------------------------------------------------------------------------
# Vectorized stationary bootstrap (as in trend_sev_eval.py)
# ---------------------------------------------------------------------------

def _stationary_bootstrap_indices(n, n_boot, block_size, rng):
    p = 1.0 / float(block_size)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    if n > 1:
        new_block = rng.random((n_boot, n - 1)) < p
        new_starts = rng.integers(0, n, size=(n_boot, n - 1))
        for t in range(1, n):
            cont = (idx[:, t - 1] + 1) % n
            idx[:, t] = np.where(new_block[:, t - 1], new_starts[:, t - 1], cont)
    return idx


def _slopes_from_indices(y, idx):
    xb = idx.astype(np.float64)
    dx = xb - xb.mean(axis=1, keepdims=True)
    denom = np.sum(dx * dx, axis=1)

    was_1d = (y.ndim == 1)
    y2d = y[:, None] if was_1d else y

    yb = y2d[idx]
    yb_centered = yb - yb.mean(axis=1, keepdims=True)

    num = np.einsum('bn,bns->bs', dx, yb_centered)
    with np.errstate(invalid='ignore', divide='ignore'):
        slopes = num / denom[:, None]
    slopes[denom == 0] = np.nan
    return slopes[:, 0] if was_1d else slopes


def _stationary_bootstrap_slopes(y, n_boot=2000, block_size=5, rng=None):
    """All n_boot bootstrap OLS slopes of a 1-D series."""
    y = np.asarray(y, dtype=np.float64)
    if y.size < 2 or np.isfinite(y).sum() < 2:
        return np.full(n_boot, np.nan)
    rng = rng if rng is not None else np.random.default_rng()
    idx = _stationary_bootstrap_indices(y.size, n_boot, block_size, rng)
    return _slopes_from_indices(y, idx)


def stationary_bootstrap_samples_grid(da, n_boot=500, block_size=5, boot_batch=250, rng=None):
    """All bootstrap OLS slopes at every point of `da` (needs a 'year' dim), as
    a DataArray with a leading 'boot' dim. The full array is kept in memory."""
    if "year" not in da.dims:
        raise ValueError("Input DataArray must have a 'year' dimension.")
    rng = rng if rng is not None else np.random.default_rng()

    other_dims = [d for d in da.dims if d != "year"]
    da_t = da.transpose("year", *other_dims)
    shape_other = da_t.shape[1:]
    n = da_t.sizes["year"]
    y = da_t.values.reshape(n, -1)

    slope_batches = []
    done = 0
    while done < n_boot:
        b = min(boot_batch, n_boot - done)
        idx = _stationary_bootstrap_indices(n, b, block_size, rng)
        slope_batches.append(_slopes_from_indices(y, idx))
        done += b
    slopes_all = np.concatenate(slope_batches, axis=0).reshape((n_boot,) + shape_other)

    coords = {d: da_t.coords[d] for d in other_dims if d in da_t.coords}
    return xr.DataArray(slopes_all, dims=("boot", *other_dims), coords=coords, name="slope_boot")


def empirical_w2(samples_a, samples_b):
    """Empirical W2 between two equal-size samples along 'boot', vectorized over
    the other dims. For equal-weight samples the optimal pairing is by sorted
    order: W2 = sqrt(mean((sorted(a) - sorted(b))**2))."""
    if samples_a.sizes["boot"] != samples_b.sizes["boot"]:
        raise ValueError("samples_a and samples_b must share the same 'boot' size for the "
                          "sorted-order-statistic W2 formula to apply.")
    boot_axis = samples_a.get_axis_num("boot")
    a_sorted = np.sort(samples_a.values, axis=boot_axis)
    b_sorted = np.sort(samples_b.values, axis=samples_b.get_axis_num("boot"))
    w2 = np.sqrt(np.nanmean((a_sorted - b_sorted) ** 2, axis=boot_axis))

    other_dims = [d for d in samples_a.dims if d != "boot"]
    coords = {d: samples_a.coords[d] for d in other_dims if d in samples_a.coords}
    return xr.DataArray(w2, dims=other_dims, coords=coords, name="w2_distance")


# ---------------------------------------------------------------------------
# Full-grid empirical W2 pipeline
# ---------------------------------------------------------------------------

def _wasserstein_realization(args):
    """Worker: one realization's W2 and normalized W2 on its native grid,
    regridded (nearest_s2d) onto the reanalysis grid."""
    (preprocessed_path, GCM, run, reanalysis, grid, n_boot, block_size, boot_batch, seed) = args
    rng = np.random.default_rng(seed)

    sev_gcm = preprocess_single_sev(preprocessed_path, GCM, run, reanalysis)
    sev_ref = _build_ref_severity_native(preprocessed_path, GCM, run, reanalysis)

    slopes_gcm = stationary_bootstrap_samples_grid(sev_gcm, n_boot=n_boot, block_size=block_size,
                                                    boot_batch=boot_batch, rng=rng)
    slopes_ref = stationary_bootstrap_samples_grid(sev_ref, n_boot=n_boot, block_size=block_size,
                                                    boot_batch=boot_batch, rng=rng)

    w2 = empirical_w2(slopes_gcm, slopes_ref)
    sigma_ref = slopes_ref.std(dim="boot", ddof=1)
    w2_norm = (w2 / sigma_ref.where(sigma_ref > SIGMA_REF_MIN)).rename("w2_normalized")

    ds_native = w2.to_dataset(name="w2_distance")
    ds_native["w2_normalized"] = w2_norm

    ds_era5 = xe.Regridder(ds_native, grid, method="nearest_s2d")(ds_native)
    ds_era5["GCM"] = GCM
    ds_era5["run"] = run
    return GCM, run, ds_era5


def wasserstein_empirical_grid(preprocessed_path, out_dir, reanalysis=None, exclude_gcm_run=None,
                                n_boot=500, block_size=5, boot_batch=250, max_workers=4, seed=0):
    """Per-pixel W2 between each GCM run's and ERA5's bootstrap trend
    distributions (both on the GCM grid), regridded onto the ERA5 grid
    -> agg_wasserstein_empirical_GCMs_all_year_{reanalysis}.nc, dims
    (realization, lat, lon): w2_distance, and w2_normalized (divided by ERA5's
    trend std; NaN below SIGMA_REF_MIN).

    n_boot is 500 (vs 1000 elsewhere) because all replicates stay in memory."""
    reanalysis = reanalysis or config.REANALYSIS
    exclude_gcm_run = _parse_exclude_gcm_run(exclude_gcm_run)
    grid = _target_grid(preprocessed_path, reanalysis)

    wcf_paths = glob_any(os.path.join(preprocessed_path, f'*/wcf_day*_GWL1_{reanalysis}'))
    tasks = []
    for i, p in enumerate(wcf_paths):
        GCM, run = p.split('_')[-5], p.split('_')[-3]
        if (GCM, run) in exclude_gcm_run:
            print(f'    [excluded] {GCM} {run}')
            continue
        tasks.append((preprocessed_path, GCM, run, reanalysis, grid,
                      n_boot, block_size, boot_batch, seed + i))

    ds_final = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_wasserstein_realization, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            GCM, run, ds_era5 = fut.result()
            print(f'[{i + 1}/{len(tasks)}] computed empirical W2 for {GCM} {run}')
            ds_final.append(ds_era5.expand_dims({'realization': [i]}))

    xr.concat(ds_final, dim='realization').to_netcdf(
        f'{out_dir}/agg_wasserstein_empirical_GCMs_all_year_{reanalysis}.nc'
    )


# ---------------------------------------------------------------------------
# Per-region (poly_idx) empirical W2 pipeline
# ---------------------------------------------------------------------------

def _iter_gcm_runs_agg(preprocessed_path, ssp, reanalysis, suffix_shp, gwl):
    """Yield (GCM, run) for every wcf_agg_*_{gwl}_{reanalysis}_{suffix_shp}.nc."""
    pattern = os.path.join(preprocessed_path, '*',
                           f'wcf_agg_*{gwl}_{reanalysis}_{suffix_shp}.nc')
    for fpath in sorted(glob.glob(pattern)):
        parts = os.path.basename(fpath).replace('.nc', '').split('_')
        try:
            idx = parts.index(ssp)
        except ValueError:
            print(f"  [WARN] Cannot parse (no {ssp!r} token): {fpath}")
            continue
        GCM, run = '_'.join(parts[2:idx]), parts[idx + 1]
        yield GCM, run


def _wasserstein_realization_agg(args):
    """Worker: one realization's W2 and normalized W2 per region."""
    (preprocessed_path, GCM, run, reanalysis, suffix_shp, n_boot, block_size, boot_batch, seed) = args
    rng = np.random.default_rng(seed)

    sev_gcm = preprocess_single_sev_agg(preprocessed_path, GCM, run, reanalysis, suffix_shp)
    sev_ref = _build_ref_severity_agg(preprocessed_path, GCM, run, reanalysis, suffix_shp)

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


def wasserstein_empirical_agg(preprocessed_path, out_dir, reanalysis=None, exclude_gcm_run=None,
                               suffix_shp=None, n_boot=500, block_size=5, boot_batch=250,
                               max_workers=4, seed=0):
    """wasserstein_empirical_grid() per region
    -> agg_wasserstein_empirical_GCMs_aggregated_{reanalysis}_{suffix_shp}.nc,
    dims (realization, poly_idx)."""
    reanalysis = reanalysis or config.REANALYSIS
    suffix_shp = suffix_shp or config.AGREEMENT_SUFFIX_SHP
    exclude_gcm_run = _parse_exclude_gcm_run(exclude_gcm_run)

    tasks = []
    for i, (GCM, run) in enumerate(
            _iter_gcm_runs_agg(preprocessed_path, config.SSP, reanalysis, suffix_shp, gwl='GWL1')):
        if (GCM, run) in exclude_gcm_run:
            print(f'    [excluded] {GCM} {run}')
            continue
        tasks.append((preprocessed_path, GCM, run, reanalysis, suffix_shp,
                      n_boot, block_size, boot_batch, seed + i))

    ds_final = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_wasserstein_realization_agg, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            GCM, run, ds = fut.result()
            print(f'[{i + 1}/{len(tasks)}] computed aggregated empirical W2 for {GCM} {run}')
            ds_final.append(ds.expand_dims({'realization': [i]}))

    xr.concat(ds_final, dim='realization').to_netcdf(
        f'{out_dir}/agg_wasserstein_empirical_GCMs_aggregated_{reanalysis}_{suffix_shp}.nc'
    )


# ---------------------------------------------------------------------------
# Region-mean bootstrap trend samples
# ---------------------------------------------------------------------------

def _region_samples_realization(args):
    preprocessed_path, GCM, run, reanalysis, regions, n_boot, block_size, seed = args
    rng = np.random.default_rng(seed)

    sev_gcm = preprocess_single_sev(preprocessed_path, GCM, run, reanalysis)
    sev_ref = _build_ref_severity_native(preprocessed_path, GCM, run, reanalysis)

    gcm_samples, ref_samples, region_names = [], [], []
    for r in regions:
        lat_lo, lat_hi = r["lat"]
        lon_lo, lon_hi = r["lon"]
        s_gcm = sev_gcm.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi)).mean(dim=['lat', 'lon'])
        s_ref = sev_ref.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi)).mean(dim=['lat', 'lon'])
        gcm_samples.append(_stationary_bootstrap_slopes(s_gcm.values, n_boot=n_boot, block_size=block_size, rng=rng))
        ref_samples.append(_stationary_bootstrap_slopes(s_ref.values, n_boot=n_boot, block_size=block_size, rng=rng))
        region_names.append(r["name"])

    ds = xr.Dataset(
        {
            'gcm_trend_samples': (('region', 'boot'), np.stack(gcm_samples, axis=0)),
            'ref_trend_samples': (('region', 'boot'), np.stack(ref_samples, axis=0)),
        },
        coords={'region': region_names, 'boot': np.arange(n_boot), 'GCM': GCM, 'run': run},
    )
    return GCM, run, ds


def region_empirical_trend_samples(preprocessed_path, out_dir, reanalysis=None, exclude_gcm_run=None,
                                    regions=None, n_boot=2000, block_size=5, max_workers=4, seed=0):
    """All bootstrap trend samples of the region-mean severity of each GCM run
    and of ERA5 on its grid, for the four trend_sev_eval.py regions (ED Fig. 3)
    -> region_trend_samples_{reanalysis}.nc, dims (realization, region, boot)."""
    reanalysis = reanalysis or config.REANALYSIS
    exclude_gcm_run = _parse_exclude_gcm_run(exclude_gcm_run)
    if regions is None:
        regions = [
            {"name": "Guiana Shield", "lat": [-10, 10], "lon": [-70, -50]},
            {"name": "Western U.S.", "lat": [35, 50], "lon": [-125, -105]},
            {"name": "India", "lat": [10, 30], "lon": [70, 90]},
            {"name": "Kenya", "lat": [-5, 5], "lon": [33, 42]},
        ]

    wcf_paths = sorted(glob_any(os.path.join(preprocessed_path, f'*/wcf_day*_GWL1_{reanalysis}')))
    tasks = []
    for i, p in enumerate(wcf_paths):
        GCM, run = p.split('_')[-5], p.split('_')[-3]
        if (GCM, run) in exclude_gcm_run:
            print(f'    [excluded] {GCM} {run}')
            continue
        tasks.append((preprocessed_path, GCM, run, reanalysis, regions, n_boot, block_size, seed + i))

    ds_final = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_region_samples_realization, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            GCM, run, ds_region = fut.result()
            print(f'[{i + 1}/{len(tasks)}] computed regional trend samples for {GCM} {run}')
            ds_final.append(ds_region.expand_dims({'realization': [i]}))

    xr.concat(ds_final, dim='realization').to_netcdf(
        f'{out_dir}/region_trend_samples_{reanalysis}.nc'
    )


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--n-boot', type=int, default=500,
                         help="Bootstrap replicates for the per-pixel W2.")
    parser.add_argument('--n-boot-region', type=int, default=2000,
                         help="Bootstrap replicates for the regional trend samples.")
    parser.add_argument('--block-size', type=int, default=5)
    parser.add_argument('--boot-batch', type=int, default=250)
    parser.add_argument('--exclude_gcm_run', nargs='+', default=config.EXCLUDE_GCM_RUN,
                         help="GCM:run pairs to exclude.")
    parser.add_argument('--skip-grid', action='store_true',
                         help="Skip the per-pixel W2.")
    parser.add_argument('--skip-regions', action='store_true',
                         help="Skip the regional trend samples.")
    parser.add_argument('--aggregated', action='store_true',
                         help="Also compute the per-region (poly_idx) W2.")
    parser.add_argument('--suffix-shp', default=config.AGREEMENT_SUFFIX_SHP,
                         help="Shapefile-version suffix of the wcf_agg_*/scf_agg_* files.")
    args = parser.parse_args()

    preprocessed_path = config.PATH_PREPROCESSED
    out_dir = os.path.join(config.PATH_PREPROCESSED, 'trend_evaluation')
    os.makedirs(out_dir, exist_ok=True)

    if not args.skip_grid:
        print("=== Full-grid empirical W2 (wasserstein_empirical_grid) ===")
        wasserstein_empirical_grid(preprocessed_path, out_dir, exclude_gcm_run=args.exclude_gcm_run,
                                    n_boot=args.n_boot, block_size=args.block_size,
                                    boot_batch=args.boot_batch, max_workers=args.max_workers)

    if not args.skip_regions:
        print("\n=== Region-mean empirical trend samples (region_empirical_trend_samples) ===")
        region_empirical_trend_samples(preprocessed_path, out_dir, exclude_gcm_run=args.exclude_gcm_run,
                                        n_boot=args.n_boot_region, block_size=args.block_size,
                                        max_workers=args.max_workers)

    if args.aggregated:
        print("\n=== Aggregated-domain empirical W2 (wasserstein_empirical_agg) ===")
        wasserstein_empirical_agg(preprocessed_path, out_dir, exclude_gcm_run=args.exclude_gcm_run,
                                   suffix_shp=args.suffix_shp,
                                   n_boot=args.n_boot, block_size=args.block_size,
                                   boot_batch=args.boot_batch, max_workers=args.max_workers)

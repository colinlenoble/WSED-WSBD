# -*- coding: cp1252 -*-
"""GCM vs ERA5 trend evaluation of annual SWED severity (Fig. 2, agreement masks).

Trend mean/CI come from a stationary block bootstrap (Politis & Romano) of
the OLS slope: mean = mean of the bootstrap slopes, CI = 2.5/97.5th
percentiles. GCM fields are regridded (nearest neighbour) onto the
reanalysis grid only to line cells up; no downscaling is done here.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; puts the code folders on sys.path
os.environ['ESMFMKFILE'] = config.ESMFMKFILE_XENV

import glob
import xesmf as xe
import xarray as xr
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, as_completed

from io_utils import match_files, glob_any, open_dataset_any


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rasterize_shapefile(shapefile, shape, transform):
    """Rasterize shapefile geometries onto a grid."""
    return geometry_mask(
        geometries=shapefile['geometry'],
        all_touched=True,
        out_shape=shape,
        transform=transform,
        invert=True,
    )


def _land_mask(grid_coords, shapefile_path):
    """(lat, lon) boolean land mask on `grid_coords` from the shapefile."""
    shapefile = gpd.read_file(shapefile_path)
    transform = rasterio.transform.from_bounds(
        grid_coords.lon.min().item(), grid_coords.lat.min().item(),
        grid_coords.lon.max().item(), grid_coords.lat.max().item(),
        len(grid_coords.lon), len(grid_coords.lat),
    )
    mask = rasterize_shapefile(shapefile, (len(grid_coords.lat), len(grid_coords.lon)), transform)
    # rasterio row 0 = northernmost; flip to match an ascending lat coordinate
    if grid_coords.lat.values[0] < grid_coords.lat.values[-1]:
        mask = mask[::-1, :]
    return mask


def _parse_exclude_gcm_run(exclude_gcm_run):
    """'GCM:run' strings -> {(GCM, run)} set; None means config.EXCLUDE_GCM_RUN."""
    if exclude_gcm_run is None:
        exclude_gcm_run = config.EXCLUDE_GCM_RUN
    return set(tuple(x.split(':')) for x in exclude_gcm_run)


def _pos_quantile(da, q):
    """q-th quantile of strictly positive values; returns 0 where no positive values exist."""
    pos = da.where(da > 0)
    thr = pos.quantile(q, dim="time")
    return thr.where(pos.notnull().any(dim="time"), 0)


def _target_grid(preprocessed_path, reanalysis):
    """Common regrid target: the reanalysis wcf_day_* grid (as in make_grid_files.py)."""
    files, _ = match_files(os.path.join(preprocessed_path, reanalysis, 'wcf_day*'))
    if not files:
        raise FileNotFoundError(
            f"No wcf_day_* reanalysis file under {os.path.join(preprocessed_path, reanalysis)}"
        )
    return open_dataset_any(files[0]).isel(time=slice(0, 2))


# ---------------------------------------------------------------------------
# Severity: sev = intensity * duration * frequency (same <= threshold
# convention as make_grid_files.py)
# ---------------------------------------------------------------------------

def compute_severity(comp_da, scf_ds, wcf_ds, scf_thr, wcf_thr):
    """Mean deficit (intensity) on compound-event days, aggregated yearly."""
    deficit_scf = xr.where(scf_ds["scf"] <= scf_thr, scf_thr - scf_ds["scf"], 0)
    deficit_wcf = xr.where(wcf_ds["wcf"] <= wcf_thr, wcf_thr - wcf_ds["wcf"], 0)
    daily_deficit = deficit_scf + deficit_wcf
    masked = xr.where(comp_da == 1, daily_deficit, np.nan)
    return masked.resample(time="YE").mean().fillna(0)


def duration_xr(da):
    """Mean event duration and number of events per (year, lat, lon)."""
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

    event_ids  = stacked.values.astype(int)
    lat_idxs   = stacked['lat'].values
    lon_idxs   = stacked['lon'].values
    year_idxs  = stacked['year'].values.astype(int)

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
            'event_id': ('event_instance', np.array(ids,  dtype=int)),
            'lat':      ('event_instance', np.array(lats, dtype=float)),
            'lon':      ('event_instance', np.array(lons, dtype=float)),
            'year':     ('event_instance', np.array(years, dtype=int)),
        },
    ).to_dataset(name='duration')

    df_dur = dur_da.to_dataframe()
    ds      = df_dur.groupby(['year', 'lat', 'lon']).mean().to_xarray()[['duration']]
    ds_freq = df_dur.groupby(['year', 'lat', 'lon']).count().to_xarray()
    ds_freq = ds_freq['duration'].to_dataset(name='frequency')

    ds['duration']        = ds['duration'].fillna(0)
    ds_freq['frequency']  = ds_freq['frequency'].fillna(0)
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
    ds      = df_dur.groupby(['year', 'poly_idx']).mean().to_xarray()[['duration']]
    ds_freq = df_dur.groupby(['year', 'poly_idx']).count().to_xarray()
    ds_freq = ds_freq['duration'].to_dataset(name='frequency')

    ds['duration']       = ds['duration'].fillna(0)
    ds_freq['frequency'] = ds_freq['frequency'].fillna(0)
    return ds, ds_freq


# ---------------------------------------------------------------------------
# Reference (ERA5) severity: ERA5 on each GCM's grid, with that GCM's GWL0-61
# thresholds, regridded to the common grid. grid_ic_ref.nc holds its
# per-cell bootstrap trend CI, compared with each GCM's CI (Fig. 2 map).
# ---------------------------------------------------------------------------

def _build_ref_severity_grid(preprocessed_path, GCM, run, reanalysis, grid_coords):
    """ERA5 reference-period severity with one GCM's GWL0-61 thresholds,
    regridded onto `grid_coords`."""
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

    wcf_ref['low_wind']  = xr.where(wcf_ref.wcf <= wcf_thr, 1, 0)
    scf_ref['low_solar'] = xr.where(scf_ref.scf <= scf_thr, 1, 0)
    compound_ref = (wcf_ref.low_wind * scf_ref.low_solar).to_dataset(name='start_cooc')

    wcf_ref = wcf_ref.convert_calendar('standard')
    scf_ref = scf_ref.convert_calendar('standard')

    intensity_ref = compute_severity(compound_ref.start_cooc, scf_ref, wcf_ref, scf_thr, wcf_thr)
    intensity_ref['time'] = intensity_ref.time.dt.year
    intensity_ref = intensity_ref.rename({'time': 'year'})

    ds_dur, ds_freq = duration_xr(compound_ref.start_cooc)
    # Years without any event are missing from duration_xr's output: fill with 0.
    ds_dur  = ds_dur.reindex_like(intensity_ref, fill_value=0)
    ds_freq = ds_freq.reindex_like(intensity_ref, fill_value=0)

    sev_ref = (intensity_ref * ds_dur.duration * ds_freq.frequency).to_dataset(name='sev')
    return xe.Regridder(sev_ref, grid_coords, method='nearest_s2d')(sev_ref).sev


def _ref_realization_grid(args):
    """Worker: regridded (year, lat, lon) reference severity for one GCM-threshold."""
    preprocessed_path, GCM, run, reanalysis, grid_coords = args
    sev_ref = _build_ref_severity_grid(preprocessed_path, GCM, run, reanalysis, grid_coords)
    return GCM, run, sev_ref


def make_grid_ref_trend_ci(preprocessed_path, out_dir, reanalysis=None, max_workers=4, boot_batch=100):
    """Per-cell bootstrap trend CI of the ERA5 reference severity, averaged
    over the GCM-threshold realizations -> grid_ic_ref.nc.

    Realizations are accumulated as a running NaN-safe sum/count (holding
    them all in memory caused OOM) and computed over `max_workers`
    processes. Lower `boot_batch` if the bootstrap runs out of memory."""
    reanalysis = reanalysis or config.REANALYSIS
    grid = _target_grid(preprocessed_path, reanalysis)
    grid_coords = xr.Dataset(coords={'lat': grid.lat.values, 'lon': grid.lon.values})

    wcf_paths = glob_any(os.path.join(preprocessed_path, f'*/wcf_day_*ssp*GWL0-61_{reanalysis}'))
    gcm_list = [x.split('_')[-5] for x in wcf_paths]
    run_list = [x.split('_')[-3] for x in wcf_paths]

    tasks = [(preprocessed_path, GCM, run, reanalysis, grid_coords) for GCM, run in zip(gcm_list, run_list)]

    sum_sev = None
    count_sev = None
    template = None
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_ref_realization_grid, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            GCM, run, sev_ref = fut.result()
            print(f'[{i + 1}/{len(tasks)}] accumulated reference realization {GCM} {run}')
            values = sev_ref.values
            if sum_sev is None:
                sum_sev = np.zeros_like(values)
                count_sev = np.zeros_like(values)
                template = sev_ref
            finite = np.isfinite(values)
            sum_sev[finite]   += values[finite]
            count_sev[finite] += 1

    composite_values = np.where(count_sev > 0, sum_sev / np.maximum(count_sev, 1), np.nan)
    composite = xr.DataArray(composite_values, dims=template.dims, coords=template.coords, name='sev')

    low, up, mean = stationary_bootstrap_ci_grid(composite, boot_batch=boot_batch)
    ds = low.to_dataset(name='low_trend')
    ds['up_trend']   = up
    ds['mean_trend'] = mean
    ds.to_netcdf(f'{out_dir}/grid_ic_ref.nc')


def _severity_from_cf(wcf_gwl061, scf_gwl061, wcf_gwl1, scf_gwl1, duration_fn=duration_xr):
    """GWL0-61 thresholds -> compound days -> annual severity, GWL0-61 and GWL1
    concatenated into a 40-year series. duration_fn handles gridded
    (duration_xr) or region-aggregated (duration_xr_by_poly) data."""
    wcf_thr = _pos_quantile(wcf_gwl061.wcf, 0.1)
    scf_thr = _pos_quantile(scf_gwl061.scf, 0.1)

    for ds, wv, sv in [(wcf_gwl061, 'low_wind', None),
                       (wcf_gwl1,   'low_wind', None),
                       (scf_gwl061, None, 'low_solar'),
                       (scf_gwl1,   None, 'low_solar')]:
        if wv:
            ds[wv] = xr.where(ds.wcf <= wcf_thr, 1, 0)
        else:
            ds[sv] = xr.where(ds.scf <= scf_thr, 1, 0)

    wcf_gwl061 = wcf_gwl061.convert_calendar('standard')
    wcf_gwl1   = wcf_gwl1.convert_calendar('standard')
    scf_gwl061 = scf_gwl061.convert_calendar('standard')
    scf_gwl1   = scf_gwl1.convert_calendar('standard')

    compound_gwl061 = (wcf_gwl061['low_wind'] * scf_gwl061['low_solar']).to_dataset(name='compound')
    compound_gwl1   = (wcf_gwl1['low_wind']   * scf_gwl1['low_solar']  ).to_dataset(name='compound')

    intensity_gwl061 = compute_severity(compound_gwl061.compound, scf_gwl061, wcf_gwl061, scf_thr, wcf_thr)
    intensity_gwl1   = compute_severity(compound_gwl1.compound,   scf_gwl1,   wcf_gwl1,   scf_thr, wcf_thr)

    for da in (intensity_gwl061, intensity_gwl1):
        da['time'] = da.time.dt.year

    intensity_gwl061 = intensity_gwl061.rename({'time': 'year'})
    intensity_gwl1   = intensity_gwl1.rename({'time': 'year'})

    dur_gwl061, freq_gwl061 = duration_fn(compound_gwl061.compound)
    dur_gwl1,   freq_gwl1   = duration_fn(compound_gwl1.compound)

    # Years without any event are missing from duration_fn's output: fill with 0.
    dur_gwl061  = dur_gwl061.reindex_like(intensity_gwl061, fill_value=0)
    dur_gwl1    = dur_gwl1.reindex_like(intensity_gwl1, fill_value=0)
    freq_gwl061 = freq_gwl061.reindex_like(intensity_gwl061, fill_value=0)
    freq_gwl1   = freq_gwl1.reindex_like(intensity_gwl1, fill_value=0)

    sev_gwl061 = (intensity_gwl061 * dur_gwl061.duration * freq_gwl061.frequency)
    sev_gwl1   = (intensity_gwl1   * dur_gwl1.duration   * freq_gwl1.frequency  )

    n_years_gwl061 = sev_gwl061.year.size
    n_years_gwl1 = 40 - n_years_gwl061
    sev_gwl1 = sev_gwl1.isel(year=slice(0, n_years_gwl1))

    sev_gwl1['year'] = sev_gwl1.year + n_years_gwl061
    return xr.concat([sev_gwl061, sev_gwl1], dim='year')


def preprocess_single_sev(preprocessed_path, GCM, run, reanalysis=None):
    """40-year gridded severity series (GWL0-61 then GWL1) of one realization,
    for trend estimation (see Methods)."""
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
    wcf_gwl1   = _load('wcf', 'GWL1')
    scf_gwl1   = _load('scf', 'GWL1')

    return _severity_from_cf(wcf_gwl061, scf_gwl061, wcf_gwl1, scf_gwl1, duration_fn=duration_xr)


# ---------------------------------------------------------------------------
# Bootstrap trend estimation (vectorized). The resampling indices depend only
# on the series length, so one (n_boot, n_years) index array is shared by
# all grid cells; each cell is still resampled from its own values.
# ---------------------------------------------------------------------------

def _stationary_bootstrap_indices(n, n_boot, block_size, rng):
    """Stationary-bootstrap index sequences, shape (n_boot, n): each step
    continues the current block or, with probability 1/block_size, starts a
    new one at a random position."""
    p = 1.0 / float(block_size)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    if n > 1:
        new_block  = rng.random((n_boot, n - 1)) < p
        new_starts = rng.integers(0, n, size=(n_boot, n - 1))
        for t in range(1, n):
            cont = (idx[:, t - 1] + 1) % n
            idx[:, t] = np.where(new_block[:, t - 1], new_starts[:, t - 1], cont)
    return idx


def _slopes_from_indices(y, idx):
    """OLS slope of y[idx[b]] against idx[b] (each point keeps its true year)
    for every replicate b. y: (n,) or (n, n_series).
    Returns (n_boot,) or (n_boot, n_series)."""
    xb = idx.astype(np.float64)                          # (n_boot, n)
    dx = xb - xb.mean(axis=1, keepdims=True)              # (n_boot, n)
    denom = np.sum(dx * dx, axis=1)                       # (n_boot,)

    was_1d = (y.ndim == 1)
    y2d = y[:, None] if was_1d else y

    yb = y2d[idx]                                          # (n_boot, n, n_series)
    yb_centered = yb - yb.mean(axis=1, keepdims=True)

    num = np.einsum('bn,bns->bs', dx, yb_centered)         # (n_boot, n_series)
    with np.errstate(invalid='ignore', divide='ignore'):
        slopes = num / denom[:, None]
    slopes[denom == 0] = np.nan
    return slopes[:, 0] if was_1d else slopes


def _stationary_bootstrap_slopes(y, n_boot=1000, block_size=5, ci=95, rng=None):
    """Bootstrap OLS slope of a 1-D series: (low_ci, up_ci, mean, slopes)."""
    y = np.asarray(y, dtype=np.float64)
    nan_result = np.nan, np.nan, np.nan, np.full(n_boot, np.nan)

    if y.size < 2 or np.isfinite(y).sum() < 2:
        return nan_result

    rng = rng if rng is not None else np.random.default_rng()
    idx = _stationary_bootstrap_indices(y.size, n_boot, block_size, rng)
    slopes = _slopes_from_indices(y, idx)

    alpha = (100.0 - ci) / 2.0
    low  = np.nanpercentile(slopes, alpha)
    up   = np.nanpercentile(slopes, 100.0 - alpha)
    mean = np.nanmean(slopes)
    return low, up, mean, slopes


def stationary_bootstrap_ci_grid(da, n_boot=1000, block_size=5, ci=95,
                                 boot_batch=200, rng=None):
    """Bootstrap trend CI (low, up, mean) at every point of `da` (needs a 'year'
    dim). boot_batch replicates are processed at a time to bound memory."""
    if "year" not in da.dims:
        raise ValueError("Input DataArray must have a 'year' dimension.")
    rng = rng if rng is not None else np.random.default_rng()

    other_dims = [d for d in da.dims if d != "year"]
    da_t = da.transpose("year", *other_dims)
    shape_other = da_t.shape[1:]
    n = da_t.sizes["year"]
    y = da_t.values.reshape(n, -1)          # (n_years, n_cells)

    slope_batches = []
    done = 0
    while done < n_boot:
        b = min(boot_batch, n_boot - done)
        idx = _stationary_bootstrap_indices(n, b, block_size, rng)
        slope_batches.append(_slopes_from_indices(y, idx))   # (b, n_cells)
        done += b
    slopes_all = np.concatenate(slope_batches, axis=0)       # (n_boot, n_cells)

    alpha = (100.0 - ci) / 2.0
    low  = np.nanpercentile(slopes_all, alpha,         axis=0).reshape(shape_other)
    up   = np.nanpercentile(slopes_all, 100.0 - alpha, axis=0).reshape(shape_other)
    mean = np.nanmean(slopes_all, axis=0).reshape(shape_other)

    coords = {d: da_t.coords[d] for d in other_dims if d in da_t.coords}
    low_da  = xr.DataArray(low,  dims=other_dims, coords=coords, name="slope_ci_low")
    up_da   = xr.DataArray(up,   dims=other_dims, coords=coords, name="slope_ci_up")
    mean_da = xr.DataArray(mean, dims=other_dims, coords=coords, name="slope_boot_mean")
    return low_da, up_da, mean_da


# ---------------------------------------------------------------------------
# Spatial uncertainty aggregation
# ---------------------------------------------------------------------------

def uncertainty_range(preprocessed_path, out_dir, reanalysis=None, exclude_gcm_run=None):
    """Compute per-GCM bootstrap trend CI on the reanalysis target grid."""
    reanalysis = reanalysis or config.REANALYSIS
    exclude_gcm_run = _parse_exclude_gcm_run(exclude_gcm_run)
    wcf_paths = glob_any(os.path.join(preprocessed_path, f'*/wcf_day*_GWL1_{reanalysis}'))
    grid = _target_grid(preprocessed_path, reanalysis)

    ds_final = []
    for i, p in enumerate(wcf_paths):
        GCM, run = p.split('_')[-5], p.split('_')[-3]
        if (GCM, run) in exclude_gcm_run:
            print(f'    [excluded] {GCM} {run}')
            continue
        sev = preprocess_single_sev(preprocessed_path, GCM, run, reanalysis)
        low, up, mean = stationary_bootstrap_ci_grid(sev)
        ds = low.to_dataset(name='low_trend')
        ds['up_trend']   = up
        ds['mean_trend'] = mean
        ds = xe.Regridder(ds, grid, 'nearest_s2d')(ds)
        ds['GCM'] = GCM
        ds['run'] = run
        ds_final.append(ds.expand_dims({'realization': [i]}))

    xr.concat(ds_final, dim='realization').to_netcdf(
        f'{out_dir}/agg_ic_ann_sev_GCMs_all_year_{reanalysis}.nc'
    )


def slopes_samples(preprocessed_path, out_dir, shapefile_path, reanalysis=None, exclude_gcm_run=None):
    """Bootstrap slope samples for each GCM * region combination."""
    reanalysis = reanalysis or config.REANALYSIS
    exclude_gcm_run = _parse_exclude_gcm_run(exclude_gcm_run)
    wcf_paths = sorted(glob_any(os.path.join(preprocessed_path, f'*/wcf_day*_GWL1_{reanalysis}')))
    grid = _target_grid(preprocessed_path, reanalysis)

    regions = [
        {"name": "Guiana Shield", "lat": [-10, 10],  "lon": [-70, -50]},
        {"name": "Western U.S.",  "lat": [35,  50],  "lon": [-125, -105]},
        {"name": "India",         "lat": [10,  30],  "lon": [70,   90]},
        {"name": "Kenya",         "lat": [-5,   5],  "lon": [33,   42]},
    ]

    ds_final = []
    for i, p in enumerate(wcf_paths):
        GCM, run = p.split('_')[-5], p.split('_')[-3]
        if (GCM, run) in exclude_gcm_run:
            print(f'    [excluded] {GCM} {run}')
            continue
        sev = preprocess_single_sev(preprocessed_path, GCM, run, reanalysis)

        for r in regions:
            lat_lo, lat_hi = r["lat"]
            lon_lo, lon_hi = r["lon"]
            sev_region = (
                sev
                .sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
                .mean(dim=['lat', 'lon'])
            )
            # Only the CI summary is saved, not the 2000 bootstrap slopes.
            low, up, mean, _ = _stationary_bootstrap_slopes(
                sev_region.values, n_boot=2000, block_size=5, ci=95
            )
            ds_final.append(xr.Dataset(
                {
                    'slope_ci_low': ((), low),
                    'slope_ci_up':  ((), up),
                    'slope_mean':   ((), mean),
                },
                coords={
                    'region': r['name'],
                    'GCM':    GCM,
                    'run':    run,
                },
            ))

    xr.concat(ds_final, dim='realization').to_netcdf(
        f'{out_dir}/trend_ci_bootstrap_wcf_GWL1_{reanalysis}.nc'
    )


def _ref_realization_regional_series(args):
    """Worker: one GCM-threshold's reference severity reduced to land-masked
    regional annual series. Returns (GCM, run, years, {region: values})."""
    preprocessed_path, GCM, run, reanalysis, grid_coords, mask, regions = args

    sev_ref = _build_ref_severity_grid(preprocessed_path, GCM, run, reanalysis, grid_coords)
    sev_ref = sev_ref.where(xr.DataArray(mask, dims=['lat', 'lon'], coords={'lat': sev_ref.lat, 'lon': sev_ref.lon}))

    years = sev_ref.year.values
    region_out = {}
    for r in regions:
        lat_lo, lat_hi = r["lat"]
        lon_lo, lon_hi = r["lon"]
        series = sev_ref.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi)).mean(dim=['lat', 'lon'])
        region_out[r["name"]] = series.values

    return GCM, run, years, region_out


def preprocess_ref_boot(preprocessed_path, out_dir, shapefile_path, reanalysis=None, max_workers=4):
    """Bootstrap trend CI of the ERA5 reference severity per region (Fig. 2
    joyplots): regional means per realization, averaged across realizations
    (same as averaging grids first, since the means commute), in parallel
    over `max_workers` processes."""
    reanalysis = reanalysis or config.REANALYSIS
    grid = _target_grid(preprocessed_path, reanalysis)
    grid_coords = xr.Dataset(coords={'lat': grid.lat.values, 'lon': grid.lon.values})

    wcf_paths = glob_any(os.path.join(preprocessed_path, f'*/wcf_day_*ssp*GWL0-61_{reanalysis}'))
    gcm_list = [x.split('_')[-5] for x in wcf_paths]
    run_list = [x.split('_')[-3] for x in wcf_paths]

    regions = [
        {"name": "Guiana Shield", "lat": [-10, 10],  "lon": [-70, -50]},
        {"name": "Western U.S.",  "lat": [35,  50],  "lon": [-125, -105]},
        {"name": "India",         "lat": [10,  30],  "lon": [70,   90]},
        {"name": "Kenya",         "lat": [-5,   5],  "lon": [33,   42]},
    ]

    mask = _land_mask(grid_coords, shapefile_path)

    tasks = [
        (preprocessed_path, GCM, run, reanalysis, grid_coords, mask, regions)
        for GCM, run in zip(gcm_list, run_list)
    ]

    region_series = {r["name"]: [] for r in regions}
    years_ref = None
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_ref_realization_regional_series, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            GCM, run, years, region_out = fut.result()
            print(f'[{i + 1}/{len(tasks)}] processed reference realization {GCM} {run}')
            years_ref = years
            for name, series in region_out.items():
                region_series[name].append(series)

    ds_final = []
    for r in regions:
        composite = np.nanmean(np.stack(region_series[r["name"]], axis=0), axis=0)
        low, up, mean, _ = _stationary_bootstrap_slopes(
            composite, n_boot=2000, block_size=5, ci=95
        )
        ds_final.append(xr.Dataset(
            {
                'slope_ci_low': ((), low),
                'slope_ci_up':  ((), up),
                'slope_mean':   ((), mean),
            },
            coords={'region': r['name']},
        ))

    xr.concat(ds_final, dim='realization').to_netcdf(
        f'{out_dir}/slopes_ic_ref_region_{reanalysis}.nc'
    )


# ---------------------------------------------------------------------------
# Region-aggregated trend evaluation: the same recipe computed from each
# region's aggregated wcf/scf series (wcf_agg_*/scf_agg_*, dim poly_idx), as
# used by fig45.py's regional maps. No regridding needed.
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
    wcf_gwl1   = _load('wcf', 'GWL1')
    scf_gwl1   = _load('scf', 'GWL1')

    return _severity_from_cf(wcf_gwl061, scf_gwl061, wcf_gwl1, scf_gwl1,
                             duration_fn=duration_xr_by_poly)


def _build_ref_severity_agg(preprocessed_path, GCM, run, reanalysis, suffix_shp):
    """_build_ref_severity_grid() from region-aggregated wcf/scf_agg_ref_*."""
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

    wcf_ref['low_wind']  = xr.where(wcf_ref.wcf <= wcf_thr, 1, 0)
    scf_ref['low_solar'] = xr.where(scf_ref.scf <= scf_thr, 1, 0)
    compound_ref = (wcf_ref.low_wind * scf_ref.low_solar).to_dataset(name='start_cooc')

    wcf_ref = wcf_ref.convert_calendar('standard')
    scf_ref = scf_ref.convert_calendar('standard')

    intensity_ref = compute_severity(compound_ref.start_cooc, scf_ref, wcf_ref, scf_thr, wcf_thr)
    intensity_ref['time'] = intensity_ref.time.dt.year
    intensity_ref = intensity_ref.rename({'time': 'year'})

    ds_dur, ds_freq = duration_xr_by_poly(compound_ref.start_cooc)
    ds_dur  = ds_dur.reindex_like(intensity_ref, fill_value=0)
    ds_freq = ds_freq.reindex_like(intensity_ref, fill_value=0)

    return (intensity_ref * ds_dur.duration * ds_freq.frequency)


def make_ref_trend_ci_aggregated(preprocessed_path, out_dir, reanalysis=None, suffix_shp=None):
    """make_grid_ref_trend_ci() per region (poly_idx)
    -> grid_ic_ref_aggregated_{reanalysis}_{suffix_shp}.nc"""
    reanalysis = reanalysis or config.REANALYSIS
    suffix_shp = suffix_shp or config.AGREEMENT_SUFFIX_SHP

    sum_sev = count_sev = template = None
    n = 0
    for GCM, run in _iter_gcm_runs_agg(preprocessed_path, config.SSP, reanalysis, suffix_shp, gwl='GWL0-61'):
        sev_ref = _build_ref_severity_agg(preprocessed_path, GCM, run, reanalysis, suffix_shp)
        values = sev_ref.values
        if sum_sev is None:
            sum_sev = np.zeros_like(values)
            count_sev = np.zeros_like(values)
            template = sev_ref
        finite = np.isfinite(values)
        sum_sev[finite]   += values[finite]
        count_sev[finite] += 1
        n += 1
        print(f'[{n}] accumulated aggregated reference realization {GCM} {run}')

    composite_values = np.where(count_sev > 0, sum_sev / np.maximum(count_sev, 1), np.nan)
    composite = xr.DataArray(composite_values, dims=template.dims, coords=template.coords, name='sev')

    low, up, mean = stationary_bootstrap_ci_grid(composite)
    ds = low.to_dataset(name='low_trend')
    ds['up_trend']   = up
    ds['mean_trend'] = mean
    ds.to_netcdf(f'{out_dir}/grid_ic_ref_aggregated_{reanalysis}_{suffix_shp}.nc')


def uncertainty_range_aggregated(preprocessed_path, out_dir, reanalysis=None, exclude_gcm_run=None, suffix_shp=None):
    """uncertainty_range() per region (poly_idx)
    -> agg_ic_ann_sev_GCMs_aggregated_{reanalysis}_{suffix_shp}.nc"""
    reanalysis = reanalysis or config.REANALYSIS
    suffix_shp = suffix_shp or config.AGREEMENT_SUFFIX_SHP
    exclude_gcm_run = _parse_exclude_gcm_run(exclude_gcm_run)

    ds_final = []
    for i, (GCM, run) in enumerate(
            _iter_gcm_runs_agg(preprocessed_path, config.SSP, reanalysis, suffix_shp, gwl='GWL1')):
        if (GCM, run) in exclude_gcm_run:
            print(f'    [excluded] {GCM} {run}')
            continue
        sev = preprocess_single_sev_agg(preprocessed_path, GCM, run, reanalysis, suffix_shp)
        low, up, mean = stationary_bootstrap_ci_grid(sev)
        ds = low.to_dataset(name='low_trend')
        ds['up_trend']   = up
        ds['mean_trend'] = mean
        ds['GCM'] = GCM
        ds['run'] = run
        ds_final.append(ds.expand_dims({'realization': [i]}))

    xr.concat(ds_final, dim='realization').to_netcdf(
        f'{out_dir}/agg_ic_ann_sev_GCMs_aggregated_{reanalysis}_{suffix_shp}.nc'
    )


# ---------------------------------------------------------------------------
# Model-agreement mask (per pixel or per region)
# ---------------------------------------------------------------------------

def build_agreement_mask(ref_path, gcm_path, out_path, exclude_gcm_run=None, shapefile_path=None):
    """Save agreement_pct: % of GCM realizations whose trend CI overlaps the ERA5
    CI. Cells below config.AGREEMENT_THRESHOLD are flagged on the maps.
    shapefile_path (gridded files only) sets ocean cells to NaN."""
    exclude_gcm_run = _parse_exclude_gcm_run(exclude_gcm_run)
    ds_ref = xr.open_dataset(ref_path)
    ds_gcm = xr.open_dataset(gcm_path)

    keep = np.array([(g, r) not in exclude_gcm_run for g, r in zip(ds_gcm.GCM.values, ds_gcm.run.values)])
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"Dropping {n_dropped} excluded realization(s): "
              f"{list(zip(ds_gcm.GCM.values[~keep], ds_gcm.run.values[~keep]))}")
    ds_gcm = ds_gcm.isel(realization=keep)

    overlap = (ds_ref.low_trend < ds_gcm.up_trend) & (ds_ref.up_trend > ds_gcm.low_trend)
    within = overlap.sum(dim='realization')
    agreement_pct = (within / ds_gcm.sizes['realization'] * 100)
    # overlap is False (not NaN) wherever the reference CI is NaN, so restore NaN there
    agreement_pct = agreement_pct.where(ds_ref.low_trend.notnull() & ds_ref.up_trend.notnull())
    if shapefile_path is not None:
        land = _land_mask(xr.Dataset(coords={'lat': ds_ref.lat.values, 'lon': ds_ref.lon.values}),
                          shapefile_path)
        agreement_pct = agreement_pct.where(
            xr.DataArray(land, dims=['lat', 'lon'], coords={'lat': ds_ref.lat, 'lon': ds_ref.lon}))
    agreement_pct.name = 'agreement_pct'
    agreement_pct.attrs['description'] = (
        "% of GCM realizations whose bootstrap trend CI overlaps the ERA5 reference "
        "trend CI at each cell/polygon. fig3.py/fig45.py hatch wherever this is "
        "below config.AGREEMENT_THRESHOLD (50)."
    )
    agreement_pct.to_netcdf(out_path)
    print(f"Wrote {out_path} (mean agreement: {float(agreement_pct.mean(skipna=True)):.1f}%)")
    return agreement_pct


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--era5-ref-only', action='store_true',
        help="Only compute the ERA5 reference outputs.",
    )
    parser.add_argument(
        '--max-workers', type=int, default=4,
        help="Worker processes for the ERA5 reference steps.",
    )
    parser.add_argument(
        '--exclude_gcm_run', nargs='+', default=config.EXCLUDE_GCM_RUN,
        help="GCM:run pairs to exclude.",
    )
    parser.add_argument(
        '--aggregated', action='store_true',
        help="Also run the region-aggregated pipeline and its agreement mask.",
    )
    parser.add_argument(
        '--suffix-shp', default=config.AGREEMENT_SUFFIX_SHP,
        help="Shapefile-version suffix of the wcf_agg_*/scf_agg_* files.",
    )
    parser.add_argument(
        '--skip-mask', action='store_true',
        help="Skip building the agreement mask(s).",
    )
    parser.add_argument(
        '--mask-only', action='store_true',
        help="Only rebuild the agreement mask(s) from existing CI files.",
    )
    args = parser.parse_args()

    preprocessed_path = config.PATH_PREPROCESSED
    shapefile_path    = config.SHAPEFILE_PATH
    out_dir            = os.path.join(config.PATH_PREPROCESSED, 'trend_evaluation')
    os.makedirs(out_dir, exist_ok=True)

    if not args.mask_only:
        make_grid_ref_trend_ci(preprocessed_path, out_dir, max_workers=args.max_workers)
        preprocess_ref_boot(preprocessed_path, out_dir, shapefile_path, max_workers=args.max_workers)

    if not args.era5_ref_only:
        if not args.mask_only:
            uncertainty_range(preprocessed_path, out_dir, exclude_gcm_run=args.exclude_gcm_run)
            slopes_samples(preprocessed_path, out_dir, shapefile_path, exclude_gcm_run=args.exclude_gcm_run)

        if not args.skip_mask:
            print("\n=== Building per-pixel agreement mask (< 50% CI overlap with ERA5, land only) ===")
            build_agreement_mask(
                ref_path=os.path.join(out_dir, 'grid_ic_ref.nc'),
                gcm_path=os.path.join(out_dir, f'agg_ic_ann_sev_GCMs_all_year_{config.REANALYSIS}.nc'),
                out_path=config.AGREEMENT_NC_PATH,
                exclude_gcm_run=args.exclude_gcm_run,
                shapefile_path=shapefile_path,
            )

    if args.aggregated:
        print("\n=== Aggregated-domain pipeline (wcf_agg_*/scf_agg_*, per poly_idx) ===")
        if not args.mask_only:
            make_ref_trend_ci_aggregated(preprocessed_path, out_dir,
                                         reanalysis=config.REANALYSIS, suffix_shp=args.suffix_shp)
        if not args.era5_ref_only:
            if not args.mask_only:
                uncertainty_range_aggregated(preprocessed_path, out_dir,
                                             reanalysis=config.REANALYSIS,
                                             exclude_gcm_run=args.exclude_gcm_run,
                                             suffix_shp=args.suffix_shp)
            if not args.skip_mask:
                print("\n=== Building aggregated (per-polygon) agreement mask (< 50% CI overlap) ===")
                build_agreement_mask(
                    ref_path=os.path.join(
                        out_dir, f'grid_ic_ref_aggregated_{config.REANALYSIS}_{args.suffix_shp}.nc'),
                    gcm_path=os.path.join(
                        out_dir, f'agg_ic_ann_sev_GCMs_aggregated_{config.REANALYSIS}_{args.suffix_shp}.nc'),
                    out_path=config.AGREEMENT_AGGREGATED_NC_PATH,
                    exclude_gcm_run=args.exclude_gcm_run,
                )

# -*- coding: cp1252 -*-
import os
import config
os.environ['ESMFMKFILE'] = config.ESMFMKFILE_XENV

import xesmf as xe
import xarray as xr
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Zarr/NetCDF-agnostic file lookup + opener, shared with calculate_cf.py /
# make_grid_files.py (prefers a .zarr store when present, falls back to
# .nc) -- calculate_cf.py's calculate_ds_cf_GCM writes wcf_day_*/scf_day_*
# as Zarr, so a plain glob('*.nc') + xr.open_dataset never finds them.
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


def _pos_quantile(da, q):
    """q-th quantile of strictly positive values; returns 0 where no positive values exist."""
    pos = da.where(da > 0)
    thr = pos.quantile(q, dim="time")
    return thr.where(pos.notnull().any(dim="time"), 0)


def _target_grid(preprocessed_path, reanalysis):
    """
    Common regrid target: the reanalysis's own daily wcf record
    (wcf_day_{reanalysis}_historical_reanalysis_..., written by
    calculate_cf.calculate_ds_cf_reanalysis), same target grid
    make_grid_files.py regrids every GCM/realization onto.
    """
    files, _ = match_files(os.path.join(preprocessed_path, reanalysis, 'wcf_day*'))
    if not files:
        raise FileNotFoundError(
            f"No wcf_day_* reanalysis file under {os.path.join(preprocessed_path, reanalysis)}"
        )
    return open_dataset_any(files[0]).isel(time=slice(0, 2))


# ---------------------------------------------------------------------------
# Severity (sev) components: sev = intensity * duration * frequency
# (mirrors calculate_cf.py's _compound_indices / make_grid_files.py's
# compute_severity+duration_xr exactly: <= threshold comparison, so a day
# exactly at the threshold is treated the same way here as by the
# compound-day flag itself and by every other script in the pipeline).
# ---------------------------------------------------------------------------

def compute_severity(comp_da, scf_ds, wcf_ds, scf_thr, wcf_thr):
    """Mean deficit (intensity) on compound-event days, aggregated yearly."""
    deficit_scf = xr.where(scf_ds["scf"] <= scf_thr, scf_thr - scf_ds["scf"], 0)
    deficit_wcf = xr.where(wcf_ds["wcf"] <= wcf_thr, wcf_thr - wcf_ds["wcf"], 0)
    daily_deficit = deficit_scf + deficit_wcf
    masked = xr.where(comp_da == 1, daily_deficit, np.nan)
    return masked.resample(time="YE").mean().fillna(0)


def duration_xr(da):
    """
    Compute mean event duration and event frequency per (year, lat, lon).

    Returns ds (mean duration), ds_freq (event count).
    """
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


# ---------------------------------------------------------------------------
# Reference period (reanalysis-based): one sev series per GCM's own grid +
# GWL0-61 threshold (same per-model threshold every other GWL is scored
# against), regridded onto the common reanalysis target grid and
# concatenated over 'realization' -- mirrors make_grid_files.py's
# load_gridded_data_compound reference branch.
# ---------------------------------------------------------------------------

def make_annual_freq_ref(preprocessed_path, out_dir, reanalysis=None):
    """Build annual sev for the reanalysis reference period, once per GCM grid."""
    reanalysis = reanalysis or config.REANALYSIS
    grid = _target_grid(preprocessed_path, reanalysis)

    wcf_paths = glob_any(os.path.join(preprocessed_path, f'*/wcf_day_*ssp*GWL0-61_{reanalysis}'))
    gcm_list = [x.split('_')[-5] for x in wcf_paths]
    run_list = [x.split('_')[-3] for x in wcf_paths]

    dfinal = []
    for i, (GCM, run) in enumerate(zip(gcm_list, run_list)):
        print(f'Processing GCM: {GCM}, Run: {run}')

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
        # duration_xr's groupby-based output only carries (year, lat, lon)
        # combinations that had at least one compound-event day, so a year
        # with zero events anywhere is entirely absent from its 'year'
        # coordinate -- reindex onto intensity_ref's full (always-complete,
        # one row per calendar year from resample) year/lat/lon grid and
        # fill those zero-event years with 0 rather than leaving them NaN.
        ds_dur  = ds_dur.reindex_like(intensity_ref, fill_value=0)
        ds_freq = ds_freq.reindex_like(intensity_ref, fill_value=0)

        sev_ref = (intensity_ref * ds_dur.duration * ds_freq.frequency).to_dataset(name='sev')
        sev_ref = xe.Regridder(sev_ref, grid, method='nearest_s2d')(sev_ref)
        sev_ref['GCM'] = f'{GCM}_{reanalysis}'
        sev_ref['run'] = run
        dfinal.append(sev_ref.expand_dims({'realization': [i]}))

    xr.concat(dfinal, dim='realization').to_netcdf(
        os.path.join(out_dir, f'grid_ref_annual_sev_{reanalysis}_all_year.nc')
    )


def preprocess_ref(preprocessed_path, out_dir, reanalysis=None):
    """Compute bootstrap CI of trend for the reanalysis reference sev."""
    reanalysis = reanalysis or config.REANALYSIS
    ds_ref = xr.open_dataset(os.path.join(preprocessed_path, f'grid_ref_annual_sev_{reanalysis}_all_year.nc'))
    low, up, mean = stationary_bootstrap_ci_grid(ds_ref.sev)
    ds = low.to_dataset(name='low_trend')
    ds['up_trend']   = up
    ds['mean_trend'] = mean
    ds.to_netcdf(f'{out_dir}/grid_ic_ref.nc')


def preprocess_single_sev(preprocessed_path, GCM, run, reanalysis=None):
    """
    Concatenate GWL0-61 and GWL1 sev into a single 40-year series for trend
    estimation (the "40-year window whose first 20 years were centered on
    GWL0.61" described in the trend-evaluation Methods).
    """
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

    dur_gwl061, freq_gwl061 = duration_xr(compound_gwl061.compound)
    dur_gwl1,   freq_gwl1   = duration_xr(compound_gwl1.compound)

    # duration_xr's groupby-based output only carries (year, lat, lon)
    # combinations that had at least one compound-event day, so a year with
    # zero events anywhere is entirely absent from its 'year' coordinate --
    # reindex onto intensity's full (always-complete, one row per calendar
    # year from resample) year/lat/lon grid and fill those zero-event years
    # with 0 rather than leaving them NaN.
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


# ---------------------------------------------------------------------------
# Bootstrap trend estimation -- vectorized.
#
# The original implementation ran, per grid cell, a Python `for b in
# range(n_boot)` loop that itself built each stationary-bootstrap index
# sequence with a `while total < n` loop of np.random.geometric/randint
# calls -- i.e. O(n_cells * n_boot) sequential Python-level steps (via
# xr.apply_ufunc(..., vectorize=True), which is still a per-cell Python
# loop under the hood).
#
# Below, index generation is vectorized across bootstrap replicates: the
# stationary-bootstrap block structure only depends on n (number of years,
# ~40 here) and block_size, not on the data, so one array of shape
# (n_boot, n) is built with a loop over the n time steps (vectorized across
# n_boot) instead of a loop over n_boot. The same index array is then used
# to gather + fit *every* grid cell/series at once via array ops. This
# turns the per-cell cost from O(n_boot) sequential Python iterations into
# a handful of vectorized numpy calls shared across all cells, and drops
# the total step count from O(n_cells * n_boot) to O(n_boot) (batched).
#
# Sharing bootstrap index sequences across grid cells does not bias any
# single cell's CI (each cell is still resampled from its own values --
# only the *pattern* of which years get picked together is shared instead
# of independently redrawn per cell), it just avoids paying for
# independent block draws at every one of possibly 10^5+ cells.
# ---------------------------------------------------------------------------

def _stationary_bootstrap_indices(n, n_boot, block_size, rng):
    """
    Vectorized stationary-bootstrap (Politis & Romano) index sequences.

    At each of the n time steps, every replicate either continues its
    current block (previous position + 1, wrapped) or -- w.p. 1/block_size
    -- starts a new block at a uniform random position. Loops over n (the
    number of years) instead of n_boot, vectorized across all replicates.

    Returns idx, shape (n_boot, n), int64.
    """
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
    """
    OLS slope of y[idx[b]] vs. idx[b] itself, for every bootstrap replicate
    b at once -- matches the original implementation's `xb, yb =
    np.arange(n)[idx], y[idx]` exactly: the regressor is the *actual*
    resampled position of each point (so a duplicated/omitted block keeps
    its true year label), not a fixed re-numbering of the resampled series.
    Since the regressor depends on idx (not on which series/cell is being
    fit), dx/denom are computed once per replicate and shared across every
    cell for that replicate.

    y   : (n,) or (n, n_series) array of values.
    idx : (n_boot, n) index matrix from _stationary_bootstrap_indices.

    Returns slopes, shape (n_boot,) if y is 1-D else (n_boot, n_series).
    """
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
    """
    Stationary bootstrap distribution of the OLS slope of a single 1-D
    series y. Returns (low_ci, up_ci, mean, slopes_array).
    """
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
    """
    Stationary-bootstrap trend CI at every point of `da` (must have a
    'year' dim, plus any number of other dims, e.g. lat/lon).

    boot_batch caps how many bootstrap replicates are gathered at once
    (the gathered array is boot_batch * n_years * n_cells floats) to bound
    memory; batching only changes how the RNG stream is drawn from (not
    the resulting CI's statistical properties), so results are equivalent
    to -- not bit-identical to -- a single boot_batch=n_boot pass. Replaces
    the old per-pixel xr.apply_ufunc(..., vectorize=True) + Python double
    loop.
    """
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

def uncertainty_range(preprocessed_path, out_dir, reanalysis=None):
    """Compute per-GCM bootstrap trend CI on the reanalysis target grid."""
    reanalysis = reanalysis or config.REANALYSIS
    wcf_paths = glob_any(os.path.join(preprocessed_path, f'*/wcf_day*_GWL1_{reanalysis}'))
    grid = _target_grid(preprocessed_path, reanalysis)

    ds_final = []
    for i, p in enumerate(wcf_paths):
        GCM, run = p.split('_')[-5], p.split('_')[-3]
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


def slopes_samples(preprocessed_path, out_dir, shapefile_path, reanalysis=None):
    """Bootstrap slope samples for each GCM * region combination."""
    reanalysis = reanalysis or config.REANALYSIS
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
        sev = preprocess_single_sev(preprocessed_path, GCM, run, reanalysis)

        for r in regions:
            lat_lo, lat_hi = r["lat"]
            lon_lo, lon_hi = r["lon"]
            sev_region = (
                sev
                .sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
                .mean(dim=['lat', 'lon'])
            )
            # Only the summary (low/up/mean) is kept -- the full 2000-sample
            # bootstrap distribution is computed (needed for the percentile
            # CI) but not written to disk, to keep this file small across
            # 32 realizations x 4 regions.
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
    """
    Worker: build one GCM-threshold's ERA5-based reference severity grid
    (same recipe as make_annual_freq_ref's inner loop), then immediately
    reduce it to a masked per-region annual time series and discard the
    (year, lat, lon) grid -- only a handful of small 1-D arrays per
    GCM/region ever have to leave this worker process.

    Returns (GCM, run, {region_name: (years, values)}).
    """
    preprocessed_path, GCM, run, reanalysis, grid_coords, mask, regions = args

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
    ds_dur  = ds_dur.reindex_like(intensity_ref, fill_value=0)
    ds_freq = ds_freq.reindex_like(intensity_ref, fill_value=0)

    sev_ref = (intensity_ref * ds_dur.duration * ds_freq.frequency).to_dataset(name='sev')
    sev_ref = xe.Regridder(sev_ref, grid_coords, method='nearest_s2d')(sev_ref).sev
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
    """
    Bootstrap trend CI of the ERA5 reference severity, per region.

    Streams straight from each GCM-threshold's own reference-period severity
    grid to a masked regional-mean annual series, instead of going through
    make_annual_freq_ref's grid_ref_annual_sev_*.nc (a (realization, year,
    lat, lon) grid for every GCM, concatenated and held in memory all at
    once before being written to disk) -- that intermediate is what was
    running this out of memory. No (realization, year, lat, lon) grid is
    ever built, on disk or off: each realization's grid is reduced to a
    handful of per-region 1-D series and discarded immediately.

    make_annual_freq_ref()/preprocess_ref() average across GCM-threshold
    realizations *before* the regional spatial mean; here that order is
    flipped (regional mean first, then averaged across realizations below),
    which gives the same composite regional series since both steps are
    plain means over disjoint axes (grid cells vs. realizations) and so
    commute -- but flipping the order is exactly what lets each realization
    be reduced to a few numbers immediately instead of needing every
    realization's full grid in memory at once.

    Each realization's grid+regrid work is independent of every other's, so
    it is embarrassingly parallel across GCMs -- distributed across
    `max_workers` processes (keep this modest; each worker still holds one
    GCM's full grid transiently during regridding).
    """
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

    shapefile = gpd.read_file(shapefile_path)
    transform = rasterio.transform.from_bounds(
        grid_coords.lon.min().item(), grid_coords.lat.min().item(),
        grid_coords.lon.max().item(), grid_coords.lat.max().item(),
        len(grid_coords.lon), len(grid_coords.lat),
    )
    mask = rasterize_shapefile(shapefile, (len(grid_coords.lat), len(grid_coords.lon)), transform)[::-1, :]

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
        # Average across GCM-threshold realizations (see docstring) --
        # equivalent to averaging the full grid across realizations first,
        # but computed from the already-tiny per-region series.
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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--era5-ref-only', action='store_true',
        help="Only compute the ERA5 reference regional trend CI (preprocess_ref_boot -> "
             "slopes_ic_ref_region_*.nc); skip the per-GCM uncertainty_range/slopes_samples passes.",
    )
    parser.add_argument(
        '--max-workers', type=int, default=4,
        help="Worker processes for preprocess_ref_boot's per-GCM regridding (default: 4).",
    )
    args = parser.parse_args()

    preprocessed_path = config.PATH_PREPROCESSED
    shapefile_path    = config.SHAPEFILE_PATH
    out_dir            = os.path.join(config.PATH_PREPROCESSED, 'trend_evaluation')
    os.makedirs(out_dir, exist_ok=True)

    # make_annual_freq_ref(preprocessed_path, out_dir)  # full-grid path -- only needed for preprocess_ref()'s map-panel grid_ic_ref.nc, not for the regional CI below
    # preprocess_ref(out_dir, out_dir)
    preprocess_ref_boot(preprocessed_path, out_dir, shapefile_path, max_workers=args.max_workers)

    if not args.era5_ref_only:
        uncertainty_range(preprocessed_path, out_dir)
        slopes_samples(preprocessed_path, out_dir, shapefile_path)

import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; also puts main_pipeline/, main_figs/, supp_figs/, aux_code/ on sys.path
os.environ['ESMFMKFILE'] = config.ESMFMKFILE_XENV
import xesmf as xe
import xarray as xr
import dask.array as da
import numpy as np
import pandas as pd
import glob as glob
import dask.array as da
import gc
from xclim import sdba
from dask import compute
import geopandas as gpd
import xagg as xa
from rasterio.features import geometry_mask
import rasterio
from string import ascii_lowercase
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patheffects import withStroke
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
import cmocean as cmo
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from itertools import groupby


def compute_severity(comp_da, scf_ds, wcf_ds, scf_threshold, wcf_threshold, freq='year'):
    """
    Compute severity of the compound event as the expected shortfall.
    For each day where a compound event occurs (i.e. compound_occurrence==1),
    compute the deficit for scf and wcf as (threshold - actual value) and then
    take their average. Finally, compute the mean of these deficits over time.

    Parameters:
      comp_ds: Dataset with the binary 'compound_occurrence' variable.
      scf_ds: Dataset with the 'scf' variable.
      wcf_ds: Dataset with the 'wcf' variable.
      scf_threshold: DataArray of scf threshold per region.
      wcf_threshold: DataArray of wcf threshold per region.
      freq: 'year' (default, annual buckets) or 'month' (monthly buckets --
        used by load_agg_data_compound(..., freq='month') for the variance
        -decomposition pipeline, suppfig7.load_variability()).

    Returns:
      severity: DataArray of average severity per region and per `freq`
    """

    deficit_scf = (scf_threshold - scf_ds["scf"])
    deficit_wcf = (wcf_threshold - wcf_ds["wcf"])

    daily_deficit = ((deficit_scf + deficit_wcf)) * comp_da

    resample_rule = '1Y' if freq == 'year' else '1M'
    total_deficit = daily_deficit.resample(time=resample_rule).sum()

    severity = total_deficit

    return severity




def duration_xr(da, freq='year'):
    """
    Compute event durations for each (event_id, poly_idx), handling cases where
    same event_id occurs in different regions (poly_idx).

    Parameters:
    da (xr.DataArray): DataArray of 0 and 1 for compound event days
    freq: 'year' (default) or 'month' -- the period each event is assigned
      to (by its start day), and the name of the returned time dimension.
      With freq='month', periods are absolute calendar months
      (year*12 + month) so the same calendar month in different years
      doesn't collide.

    Returns:
    ds (xr.Dataset): DataSet of mean duration of RES waves per `freq` and poly_idx.
    ds_freq (xr.Dataset): Number of events per poly_idx and `freq`.
    """
    dim = freq

    # Add dummy time at start to detect events starting at first step
    da_dur = xr.concat([
        xr.zeros_like(da.isel(time=0)).expand_dims(time=[pd.Timestamp('2000-01-01')]),
        da
        ], dim='time')

    # Detect start of new events (transition from 0 to event_id)
    start_event = (da_dur.diff(dim='time', label='lower') > 0)  # transition to non-zero
    start_event['time'] = da.time
    if freq == 'year':
        start_event[dim] = start_event.time.dt.year
    else:
        start_event[dim] = start_event.time.dt.year * 12 + start_event.time.dt.month
    # Build cumulative event counter for each poly_idx
    id_event = start_event.cumsum(dim='time') * da
    id_event = id_event.where(id_event > 0)  # Mask non-events

    nb_event = start_event.groupby(dim).sum(dim='time')

    stacked_bis = start_event.stack(z=('poly_idx', 'time'))
    valid_period = (stacked_bis.where(stacked_bis > 0)).dropna('z')

    # Now, to compute durations for each unique (id_event, poly_idx, period)
    # Stack dimensions to flatten for easier manipulation
    stacked = id_event.stack(z=('poly_idx', 'time'))

    # Drop NaNs (non-event locations)
    valid = stacked.dropna('z')

    # Extract corresponding event IDs and poly_idx
    event_ids = valid.values.astype(int)  # event ids
    poly_idxs = valid['poly_idx'].values  # poly_idx associated

    combined_keys = np.core.defchararray.add(
            event_ids.astype(str),
            np.core.defchararray.add('-', poly_idxs.astype(str))
        )

    # Use numpy unique to get counts of each unique (event_id, poly_idx)
    unique_keys, counts = np.unique(combined_keys, return_counts=True)

    # Split combined keys back to event_id and poly_idx
    event_ids_split, poly_idxs_split = zip(*(key.split('-') for key in unique_keys))
    event_ids_split = np.array(event_ids_split, dtype=int)
    poly_idxs_split = np.array(poly_idxs_split, dtype=int)
    # Build final DataArray for durations
    dur_da = xr.DataArray(
    counts,
    dims='event_instance',
    coords={'event_instance': np.arange(len(counts)),
        'event_id': ('event_instance', event_ids_split),
        'poly_idx': ('event_instance', poly_idxs_split)
    })

    dur_da = dur_da.to_dataset(name='duration')
    valid_period = valid_period.rename({'z':'event_instance'})
    dur_da[dim] = valid_period[dim]

    #create a new dataset that gives the mean duration for every period and every poly_idx

    ds = []
    ds_freq = []
    for p in np.unique(dur_da[dim].values):
        ds.append(dur_da.where(dur_da[dim]==p,drop=True).duration.groupby('poly_idx').mean().assign_coords(**{dim: p}))
        ds_freq.append(dur_da.where(dur_da[dim]==p,drop=True).duration.groupby('poly_idx').count().assign_coords(**{dim: p}))
    ds = xr.concat(ds, dim=dim)
    ds_freq = xr.concat(ds_freq, dim=dim)
    ds_freq = ds_freq.to_dataset(name='frequency')
    ds = ds.to_dataset(name='duration')

    # Reindex onto the full period range of `da`: a period with zero events
    # at every single poly_idx never becomes a level value coming out of the
    # groupby loop above, so it's missing entirely (not just NaN). This also
    # keeps duration/frequency calendar-complete like compute_severity's
    # resample(time=...) output, so every realization ends up with the same
    # number of periods before load_agg_data_compound() concatenates them
    # over 'realization' -- otherwise realizations with different sets of
    # zero-event periods would concat into a huge, mostly-NaN axis (the
    # union of every realization's own distinct periods).
    if freq == 'year':
        full_periods = np.unique(da.time.dt.year.values)
    else:
        full_periods = np.unique(da.time.dt.year.values * 12 + da.time.dt.month.values)
    ds = ds.reindex(**{dim: full_periods}, fill_value=0)
    ds_freq = ds_freq.reindex(**{dim: full_periods}, fill_value=0)

    return ds, ds_freq





def load_agg_data_compound(preprocessed_path, freq='year'):
    '''
    ### Load aggregated data for compound events
    ### Parameters:
    - preprocessed_path: path to the preprocessed data (per-GCM
      wcf_agg_*/scf_agg_* aggregates written by calculate_cf.py, the same
      files fig45.py/trend_sev_eval.py read for their own aggregated-domain
      pipelines -- see config.AGREEMENT_AGGREGATED_NC_PATH's docstring)
    - freq: 'year' (default; the yearly RED indicators consumed by
      suppfig7.load_indicators()) or 'month' (monthly buckets, consumed by
      suppfig7.load_variability() for the variance-decomposition panels --
      port of the never-saved pipeline behind
      compound_monthly_agg_freq_sev_dur.nc in
      3.2 Variability decomposition.ipynb)

    ### Returns:
    - data: the dataset with the aggregated data for compound events for every GCM, run, ssp and gwl
    that returns the duration, frequency and severity of the compound events per `freq` and poly_idx

  '''
    dim = freq
    #the wcf_ref/scf_ref and GWL0-61 paths below are built by raw string
    #concatenation, not os.path.join -- ensure a trailing separator so that
    #works regardless of whether the caller passed one
    preprocessed_path = os.path.join(preprocessed_path, '')

    wcf_paths = glob.glob(os.path.join(preprocessed_path , '*/wcf_agg_*ssp*_ERA5_v1.nc'))
    scf_paths = glob.glob(os.path.join(preprocessed_path , '*/scf_agg_*ssp*_ERA5_v1.nc'))

    wcf_paths.sort()
    scf_paths.sort()

    gcm_list = [x.split('_')[-6] for x in wcf_paths]
    run_list = [x.split('_')[-4] for x in wcf_paths]
    ssp_list = [x.split('_')[-5] for x in wcf_paths]
    gwl_list = [x.split('_')[-3] for x in wcf_paths]
    print(gcm_list)
    data = []

    assert run_list == [x.split('_')[-4] for x in scf_paths]
    print('Reading data...')

    realization_idx = 0
    for i, GCM in enumerate(gcm_list):
        run = run_list[i]
        gwl = gwl_list[i]
        ssp = ssp_list[i]

        if gwl == 'GWL1':
            print(f"Skipping GWL1: {GCM} {ssp} {run}")
            continue
        if GCM == 'EC-Earth3-Veg-LR' and run == 'r3i1p1f1':
            print(f"Skipping EC-Earth3-Veg-LR r3i1p1f1")
            continue

        wcf = xr.open_dataset(wcf_paths[i])
        scf = xr.open_dataset(scf_paths[i])

        if gwl=='GWL0-61':
            wcf_ref = xr.open_dataset(preprocessed_path + GCM + '/wcf_agg_ref_' +GCM+'_ERA5_v1.nc')
            scf_ref = xr.open_dataset(preprocessed_path + GCM + '/scf_agg_ref_' +GCM+'_ERA5_v1.nc')
            wcf_ref = wcf_ref.sel(time=slice('1982-01-01','2001-12-31'))
            scf_ref = scf_ref.sel(time=slice('1982-01-01','2001-12-31'))
        else:
            wcf_path = glob.glob(preprocessed_path + GCM + '/wcf_agg_*ssp*'+run_list[i]+'_GWL0-61_ERA5_v1.nc')
            scf_path = glob.glob(preprocessed_path + GCM + '/scf_agg_*ssp*'+run_list[i]+'_GWL0-61_ERA5_v1.nc')
            wcf_ref = xr.open_dataset(wcf_path[0])
            scf_ref = xr.open_dataset(scf_path[0])


        wcf_thr = wcf_ref.where(wcf_ref.wcf>0).wcf.quantile(0.1, dim='time')
        scf_thr = scf_ref.where(scf_ref.scf>0).scf.quantile(0.1, dim='time')

        wcf['low_wind'] = xr.where(wcf.wcf <= wcf_thr,1 , 0)
        scf['low_solar'] = xr.where(scf.scf <= scf_thr,1 , 0)

        compound = wcf.low_wind * scf.low_solar
        compound = compound.to_dataset(name='start_cooc')

        #here severity_ds has already been averaged over each `freq` period;
        #compute_severity returns a bare DataArray, wrap it so
        #severity_ds.severity below works
        severity_ds = compute_severity(compound.start_cooc, scf, wcf, scf_thr, wcf_thr, freq=freq).to_dataset(name='severity')
        ds_dur, ds_freq = duration_xr(compound.start_cooc, freq=freq)

        #integer period labels, matching duration_xr()'s ds_dur/ds_freq
        #`dim` coordinate -- a string label here would silently turn every
        #'severity' value into NaN below, since ds_final['severity'] =
        #severity_ds.severity reindexes onto ds_final's existing int-labeled
        #index, and no string label ever matches an int one
        if freq == 'year':
            severity_ds['time'] = severity_ds.time.dt.year
        else:
            severity_ds['time'] = severity_ds.time.dt.year * 12 + severity_ds.time.dt.month
        severity_ds = severity_ds.rename({'time': dim})

        ds_final = ds_dur.copy()
        ds_final['frequency'] = ds_freq.frequency
        ds_final['severity'] = severity_ds.severity

        # Relabel `dim` from absolute calendar periods to a positional index
        # (1..N, periods since this realization's own 20-year window starts)
        # before concatenating over 'realization' below -- different GWLs
        # (and the same GWL across different GCMs) span different, largely
        # non-overlapping absolute calendar periods, so concatenating on the
        # raw calendar labels would align by (disjoint) label instead of by
        # within-window position, blowing `dim` up to the union of every
        # realization's own distinct periods instead of a common N.
        ds_final = ds_final.assign_coords(**{dim: np.arange(1, ds_final.sizes[dim] + 1)})

        ds_final = ds_final.expand_dims({'realization': [realization_idx]})
        realization_idx += 1
        ds_final['GCM'] = GCM
        ds_final['run'] = run_list[i]
        ds_final['ssp'] = ssp_list[i]
        ds_final['gwl'] = gwl_list[i]

        data.append(ds_final)
    data = xr.concat(data, dim='realization')
    data = data.drop_dims('time', errors='ignore')
    #each realization already carries a clean positional 1..N `dim` index
    #(see above), so this is now just a defensive final relabel -- sized to
    #whatever `dim` actually came out as, instead of a hardcoded 20
    data[dim] = np.arange(1, data.sizes[dim] + 1)

    return data




# -------------------------------

if __name__ == '__main__':
    path_preprocessed = config.PATH_PREPROCESSED
    shapefile_path = config.SHAPEFILE_PATH_LIGHT
    figs_dir = config.SUMMARY_FIGS_DIR


    ssp = config.SSP
    gwl_list = config.GWL_LIST

    reanalysis = config.REANALYSIS
    delete = False
    unbias = True
    
    data = load_agg_data_compound(path_preprocessed)

    data.attrs = {
        'description': 'Annual statistics of compound energy drought events (simultaneous low-wind and low-solar days).',
        'threshold': '10th percentile of non-zero days over GWL0-61 reference window (ssp245)',
        'excluded': 'MIROC6; EC-Earth3-Veg-LR r3i1p1f1; GWL1',
        'variables': 'duration [days], frequency [count/year], severity [wcf+scf deficit on event days], low_wind [days/year], low_solar [days/year]',
        'source': 'ERA5-bias-corrected ISIMIP3b projections',
        'creation_date': '2026-03-30',
    }
    data['duration'].attrs   = {'long_name': 'Mean compound event duration'}
    data['frequency'].attrs  = {'long_name': 'Number of compound events per year', 'units': 'count year-1'}
    data['severity'].attrs   = {'long_name': 'Mean compound event severity (total wcf+scf deficit / nb events)', 'units': 'equivalent full load days'}

    data.to_netcdf(path_preprocessed + 'agg_datasets/compound_years_agg_freq_sev_dur.nc')
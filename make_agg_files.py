import os
import config
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


def compute_severity(comp_da, spp_ds, wpp_ds, spp_threshold, wpp_threshold):
    """
    Compute severity of the compound event as the expected shortfall.
    For each day where a compound event occurs (i.e. compound_occurrence==1), 
    compute the deficit for spp and wpp as (threshold - actual value) and then 
    take their average. Finally, compute the mean of these deficits over time.
    
    Parameters:
      comp_ds: Dataset with the binary 'compound_occurrence' variable.
      spp_ds: Dataset with the 'spp' variable.
      wpp_ds: Dataset with the 'wpp' variable.
      spp_threshold: DataArray of spp threshold per region.
      wpp_threshold: DataArray of wpp threshold per region.
    
    Returns:
      severity: DataArray of average severity per region and per year
    """

    deficit_spp = (spp_threshold - spp_ds["spp"])
    deficit_wpp = (wpp_threshold - wpp_ds["wpp"])
    
    daily_deficit = ((deficit_spp + deficit_wpp)) * comp_da
    
    total_deficit = daily_deficit.resample(time='1Y').sum()

    severity = total_deficit
    
    return severity




def duration_xr(da):
    """
    Compute event durations for each (event_id, poly_idx), handling cases where
    same event_id occurs in different regions (poly_idx).
    
    Parameters:
    da (xr.DataArray): DataArray of 0 and 1 for compound event days
    
    Returns:
    ds (xr.Dataset): DataSet of mean duration of RES waves per year and poly_idx.
    ds_freq (xr.Dataset): Number of events per poly_idx and year.
    """
    
    # Add dummy time at start to detect events starting at first step
    da_dur = xr.concat([
        xr.zeros_like(da.isel(time=0)).expand_dims(time=[pd.Timestamp('2000-01-01')]),
        da
        ], dim='time')
        
    # Detect start of new events (transition from 0 to event_id)
    start_event = (da_dur.diff(dim='time', label='lower') > 0)  # transition to non-zero
    start_event['time'] = da.time
    start_event['year'] = start_event.time.dt.year
    # Build cumulative event counter for each poly_idx
    id_event = start_event.cumsum(dim='time') * da
    id_event = id_event.where(id_event > 0)  # Mask non-events

    nb_event = start_event.groupby('year').sum(dim='time')

    stacked_bis = start_event.stack(z=('poly_idx', 'time'))
    valid_year = (stacked_bis.where(stacked_bis > 0)).dropna('z')

    # Now, to compute durations for each unique (id_event, poly_idx, year)
    # Stack dimensions to flatten for easier manipulation
    stacked = id_event.stack(z=('poly_idx', 'time'))

    # Drop NaNs (non-event locations)
    valid = stacked.dropna('z')

    # Extract corresponding event IDs, poly_idx, and year
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
    valid_year = valid_year.rename({'z':'event_instance'})
    dur_da['year'] = valid_year['year']

    #create a new dataset that gives the mean duration for every year and every poly_idx
    
    ds = []
    ds_freq = []
    for y in np.unique(dur_da.year.values):
        ds.append(dur_da.where(dur_da.year==y,drop=True).duration.groupby('poly_idx').mean().assign_coords(year=y))
        ds_freq.append(dur_da.where(dur_da.year==y,drop=True).duration.groupby('poly_idx').count().assign_coords(year=y))
    ds = xr.concat(ds, dim='year')
    ds_freq = xr.concat(ds_freq, dim='year')
    ds_freq = ds_freq.to_dataset(name='frequency')
    ds = ds.to_dataset(name='duration')

    return ds, ds_freq





def load_agg_data_compound(preprocessed_path):
    '''
    ### Load aggregated data for compound events
    ### Parameters:
    - preprocessed_path: path to the preprocessed data
    - gwl: the global warming level
    - rolling: the rolling window for the data

    ### Returns:
    - data: the dataset with the aggregated data for compound events for every GCM, run, ssp and gwl
    that returns the duration, frequency and severity of the compound events per year and poly_idx

  '''
    wpp_paths = glob.glob(os.path.join(preprocessed_path , '*/wpp_agg_*ssp*_W5E5_v1.nc'))
    spp_paths = glob.glob(os.path.join(preprocessed_path , '*/spp_agg_*ssp*_W5E5_v1.nc'))

    wpp_paths.sort()
    spp_paths.sort()

    gcm_list = [x.split('_')[-6] for x in wpp_paths]
    run_list = [x.split('_')[-4] for x in wpp_paths]
    ssp_list = [x.split('_')[-5] for x in wpp_paths]
    gwl_list = [x.split('_')[-3] for x in wpp_paths]
    print(gcm_list)    
    data = []

    assert run_list == [x.split('_')[-4] for x in spp_paths]
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

        wpp = xr.open_dataset(wpp_paths[i])
        spp = xr.open_dataset(spp_paths[i])

        if gwl=='GWL0-61':
            wpp_ref = xr.open_dataset(preprocessed_path + GCM + '/wpp_agg_ref_' +GCM+'_W5E5_v1.nc')
            spp_ref = xr.open_dataset(preprocessed_path + GCM + '/spp_agg_ref_' +GCM+'_W5E5_v1.nc')
            wpp_ref = wpp_ref.sel(time=slice('1982-01-01','2001-12-31'))
            spp_ref = spp_ref.sel(time=slice('1982-01-01','2001-12-31'))
        else:
            wpp_path = glob.glob(preprocessed_path + GCM + '/wpp_agg_*ssp*'+run_list[i]+'_GWL0-61_W5E5_v1.nc')
            spp_path = glob.glob(preprocessed_path + GCM + '/spp_agg_*ssp*'+run_list[i]+'_GWL0-61_W5E5_v1.nc')
            wpp_ref = xr.open_dataset(wpp_path[0])
            spp_ref = xr.open_dataset(spp_path[0])
        

        wpp_thr = wpp_ref.where(wpp_ref.wpp>0).wpp.quantile(0.1, dim='time')
        spp_thr = spp_ref.where(spp_ref.spp>0).spp.quantile(0.1, dim='time')

        wpp['low_wind'] = xr.where(wpp.wpp <= wpp_thr,1 , 0)
        spp['low_solar'] = xr.where(spp.spp <= spp_thr,1 , 0)

        compound = wpp.low_wind * spp.low_solar
        compound = compound.to_dataset(name='start_cooc')

        severity_ds = compute_severity(compound.start_cooc, spp, wpp, spp_thr, wpp_thr) #here severity_ds has already been averaged over years
        ds_dur, ds_freq = duration_xr(compound.start_cooc)

        severity_ds['time'] = severity_ds.time.dt.year.astype(str)
        severity_ds = severity_ds.rename({'time':'year'})

        ds_final = ds_dur.copy()
        ds_final['frequency'] = ds_freq.frequency
        ds_final['severity'] = severity_ds.severity

        ds_final = ds_final.expand_dims({'realization': [realization_idx]})
        realization_idx += 1
        ds_final['GCM'] = GCM 
        ds_final['run'] = run_list[i] 
        ds_final['ssp'] = ssp_list[i]
        ds_final['gwl'] = gwl_list[i]
        
        data.append(ds_final)
    data = xr.concat(data, dim='realization')
    data = data.drop_dims('time')
    data['year'] = np.arange(1,21,1)

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
        'variables': 'duration [days], frequency [count/year], severity [wpp+spp deficit on event days], low_wind [days/year], low_solar [days/year]',
        'source': 'W5E5-bias-corrected ISIMIP3b projections',
        'creation_date': '2026-03-30',
    }
    data['duration'].attrs   = {'long_name': 'Mean compound event duration'}
    data['frequency'].attrs  = {'long_name': 'Number of compound events per year', 'units': 'count year-1'}
    data['severity'].attrs   = {'long_name': 'Mean compound event severity (total wcf+scf deficit / nb events)', 'units': 'equivalent full load days'}

    data.to_netcdf(path_preprocessed + 'agg_datasets/compound_years_agg_freq_sev_dur.nc')
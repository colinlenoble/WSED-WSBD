import os
import config
os.environ['ESMFMKFILE'] = config.ESMFMKFILE_XENV
import xesmf as xe
import xarray as xr
import numpy as np
import pandas as pd
import glob as glob

# Zarr/NetCDF-agnostic file lookup + opener, shared with calculate_cf.py
# (prefers a .zarr store when present, falls back to .nc).
from io_utils import match_files, glob_any, open_dataset_any

def compute_severity(comp_da, scf_ds, wcf_ds, scf_thr, wcf_thr):
    """
    Expected shortfall: mean positive deficit on compound-event days,
    aggregated yearly.
    """

    # Deficits (clip at 0 so negatives can't creep in). <= to match the
    # compound-day definition in calculate_cf.py's _compound_indices
    # (low_wind/low_solar use wcf/scf <= thr), so a day exactly at the
    # threshold is treated consistently by both the event flag and the
    # deficit used for severity.
    deficit_scf = xr.where(scf_ds["scf"] <= scf_thr, scf_thr - scf_ds["scf"], 0)
    deficit_wcf = xr.where(wcf_ds["wcf"] <= wcf_thr, wcf_thr - wcf_ds["wcf"], 0)
    daily_deficit = deficit_scf + deficit_wcf

    # Mask to compound days
    masked = xr.where(comp_da == 1, daily_deficit, np.nan)

    # Expected shortfall per calendar year end
    severity = masked.resample(time="YE").mean().fillna(0)
    
    return severity



def duration_xr(da):
    """
    Compute event durations for each (event_id, poly_idx), handling cases where
    same event_id occurs in different regions (poly_idx).

    Returns:
      ds: Dataset of mean duration per year and lat-lon.
      ds_freq: Dataset with the number of events per year and lat-lon.
    """
    # Add a dummy time at the start to detect events starting at the first step
    print(da)
    da = da.convert_calendar('standard')
    da = da.sortby('lat').sortby('lon')
    da['lat'] = da['lat'].astype(float)
    da['lon'] = da['lon'].astype(float)
    print(da)
    # Use pd.Timestamp if da.time is not a pandas.Timestamp
    first_time = da.time[0].values
    if not isinstance(first_time, (np.datetime64, pd.Timestamp)):
        first_time = pd.Timestamp(first_time)
    else:
        first_time = pd.Timestamp(first_time)
    da_dur = xr.concat([
        xr.zeros_like(da.isel(time=0)).expand_dims(time=[first_time - pd.Timedelta(days=1)]),
        da
    ], dim='time')
        
    # Detect the start of events (transition from 0 to 1)
    start_event = (da_dur.diff(dim='time', label='lower') > 0)
    start_event['time'] = da.time
    start_event['year'] = start_event.time.dt.year
    id_event = start_event.cumsum(dim='time') * da
    id_event = id_event.where(id_event > 0)

    stacked = id_event.stack(z=('lat','lon','time'))
    valid = stacked.notnull()
    stacked = stacked.where(valid, drop=True)

    event_ids = stacked.values.astype(int)
    lat_idxs = stacked['lat'].values
    lon_idxs = stacked['lon'].values
    year_idxs = stacked['year'].values
    year_idxs = year_idxs.astype(int)

    #for the same event_id, lat_id, lon_id put the minimal year
    df = pd.DataFrame({
        'event_id': event_ids,
        'lat': lat_idxs,
        'lon': lon_idxs,
        'year': year_idxs
    })
    # Instead of aggregating, just update the 'year' column to the minimal year for each (event_id, lat, lon)
    min_years = df.groupby(['event_id', 'lat', 'lon'])['year'].transform('min')
    df['year'] = min_years
    event_ids = df['event_id'].values
    lat_idxs = df['lat'].values
    lon_idxs = df['lon'].values
    year_idxs = df['year'].values

    # Create unique keys for each event instance (combining event_id, spatial indices, and year)
    combined_keys = np.core.defchararray.add(
        np.core.defchararray.add(
            np.core.defchararray.add(event_ids.astype(str), ';'),
            np.core.defchararray.add(lat_idxs.astype(str), ';')
        ),
        np.core.defchararray.add(lon_idxs.astype(str), np.core.defchararray.add(';', year_idxs.astype(str)))
    )
    unique_keys, counts = np.unique(combined_keys, return_counts=True)
    event_ids_split, lat_idxs_split, lon_idxs_split, year_idxs_split = zip(*(key.split(';') for key in unique_keys))
    event_ids_split = np.array(event_ids_split, dtype=int)
    lat_idxs_split = np.array(lat_idxs_split, dtype=float)
    lon_idxs_split = np.array(lon_idxs_split, dtype=float)
    year_idxs_split = np.array(year_idxs_split, dtype=int)
    

    
    # Build a DataArray for the event durations
    dur_da = xr.DataArray(
        counts,
        dims='event_instance',
        coords={'event_instance': np.arange(len(counts)),
                'event_id': ('event_instance', event_ids_split),
                'lat': ('event_instance', lat_idxs_split),
                'lon': ('event_instance', lon_idxs_split),
                'year': ('event_instance', year_idxs_split)})
    
    dur_da = dur_da.to_dataset(name='duration')

    ds = dur_da.to_dataframe().groupby(['year', 'lat', 'lon']).mean().to_xarray()
    ds_freq = dur_da.to_dataframe().groupby(['year', 'lat', 'lon']).count().to_xarray()
    ds = ds[['duration']]
    ds_freq = ds_freq['duration'].to_dataset(name='frequency')
    ds_freq['frequency'] = ds_freq['frequency'].fillna(0)
    ds['duration'] = ds['duration'].fillna(0)

    # Reindex onto the full (year, lat, lon) grid from da: a year with zero
    # events at every single pixel never becomes a level value coming out of
    # groupby/to_xarray above, so it's missing entirely (not just NaN) --
    # the fillna(0) calls above can't catch that. This also keeps freq/dur
    # on the same coordinates as compute_severity's intensity output, which
    # is already calendar-complete via its .resample(...).fillna(0).
    full_years = np.unique(da.time.dt.year.values)
    ds = ds.reindex(year=full_years, lat=da.lat.values, lon=da.lon.values, fill_value=0)
    ds_freq = ds_freq.reindex(year=full_years, lat=da.lat.values, lon=da.lon.values, fill_value=0)

    return ds, ds_freq
    
def load_gridded_data_compound(preprocessed_path, gwl, reanalysis=False):
    '''
    Load gridded data for compound events and compute severity and duration in parallel.
    
    Parameters:
      preprocessed_path: path to the preprocessed data files.
      rolling: the rolling window size to use.
    
    Returns:
      data: xarray.Dataset with duration, frequency, and severity per year and spatial point.
    '''
    # Find and sort file paths (zarr preferred, falls back to .nc)
    wcf_paths = glob_any(os.path.join(preprocessed_path, '*/wcf_day_*ssp*'+gwl+'_ERA5'))
    scf_paths = glob_any(os.path.join(preprocessed_path, '*/scf_day_*ssp*'+gwl+'_ERA5'))

    #wcf_paths = wcf_paths[16:]
    #scf_paths = scf_paths[16:]
    print(wcf_paths)
    
    chunks = {'time': -1, 'lat': 50, 'lon': 50}

    # Extract metadata from file names
    gcm_list = [x.split('_')[-5] for x in wcf_paths]
    run_list = [x.split('_')[-3] for x in wcf_paths]
    ssp_list = [x.split('_')[-4] for x in wcf_paths]
    gwl_list = [x.split('_')[-2] for x in wcf_paths]
    print(gcm_list)

    wcf_rea_files, _ = match_files(preprocessed_path + 'ERA5/wcf_day*')
    wcf_rea = open_dataset_any(wcf_rea_files[0])
    wcf_rea = wcf_rea.isel(time=slice(0,2))
    # Wrap the per-GCM processing in a delayed function
    def process_single_gcm(i, preprocessed_path, reanalysis):
        GCM = gcm_list[i]
        run = run_list[i]
        ssp = ssp_list[i]
        gwl = gwl_list[i]
        
        wcf_files, _ = match_files(preprocessed_path+ GCM+'/wcf_day_' +GCM + '_' + ssp + '_' + run + '_'+gwl+'_ERA5')
        scf_files, _ = match_files(preprocessed_path+ GCM+'/scf_day_' +GCM + '_' + ssp + '_' + run + '_'+gwl+'_ERA5')
        wcf = open_dataset_any(wcf_files[0])
        scf = open_dataset_any(scf_files[0])
        wcf['time'] = pd.to_datetime(wcf.time.dt.strftime('%Y-%m-%d').values)
        scf['time'] = pd.to_datetime(scf.time.dt.strftime('%Y-%m-%d').values)

        wcf_path_ref = glob_any(os.path.join(preprocessed_path, GCM+'/wcf_day_' +GCM + '*ssp*' + run + '_GWL0-61_ERA5'))
        scf_path_ref = glob_any(os.path.join(preprocessed_path, GCM+'/scf_day_' +GCM + '*ssp*' + run + '_GWL0-61_ERA5'))
        wcf_ref = open_dataset_any(wcf_path_ref[0])
        scf_ref = open_dataset_any(scf_path_ref[0])
        wcf_ref['time'] = pd.to_datetime(wcf_ref.time.dt.strftime('%Y-%m-%d').values)
        scf_ref['time'] = pd.to_datetime(scf_ref.time.dt.strftime('%Y-%m-%d').values)
        # Compute thresholds (10th percentile)
        wcf_thr = wcf_ref.wcf.where(wcf_ref.wcf>0)
        wcf_thr = wcf_thr.quantile(0.1, dim='time')
        
        scf_thr = scf_ref.scf.where(scf_ref.scf>0)
        scf_thr = scf_thr.quantile(0.1, dim='time')
        
        wcf['low_wind'] = xr.where(wcf.wcf <= wcf_thr,1,0)
        scf['low_solar'] = xr.where(scf.scf <= scf_thr,1,0)

        compound = (wcf.low_wind * scf.low_solar)
        compound = compound.to_dataset(name='start_cooc')

        compound = compound.convert_calendar('standard')
        wcf = wcf.convert_calendar('standard')
        scf = scf.convert_calendar('standard')
    
        # Compute severity (this operation is lazy if using dask-backed arrays)
        severity_ds = compute_severity(compound.start_cooc, scf, wcf, scf_thr, wcf_thr)
        # Apply duration_xr over each grid cell (lat/lon) independently, working along time
        
        ds_dur, ds_freq = duration_xr(compound.start_cooc)
        if GCM=='IPSL-CM6A-LR':
            print(ds_freq.mean(dim='lat').mean(dim='year').compute())
        ds_dur = ds_dur.reindex({'lat': scf.lat, 'lon': scf.lon})
        ds_freq = ds_freq.reindex({'lat': scf.lat, 'lon': scf.lon})
            
        severity_ds['time'] = severity_ds.time.dt.year
        severity_ds = severity_ds.rename({'time':'year'})
        pdd = severity_ds * ds_dur.duration * ds_freq.frequency
 
        ds_final = ds_dur.copy()
        ds_final['frequency'] = ds_freq.frequency
        ds_final['severity'] = severity_ds
        ds_final['pdd'] = pdd
                
        compound = compound.resample(time='YE').sum()
        
        compound['time'] = compound.time.dt.year
        compound = compound.rename({'time':'year'})
        ds_final['nb_days'] = compound.start_cooc
        

        ds_final = ds_final.expand_dims({'realization': [i]})
        regrid = xe.Regridder(ds_final, wcf_rea, method='nearest_s2d')
        ds_final = regrid(ds_final)
        
        ds_final['GCM'] = xr.DataArray([GCM], dims='realization')
        ds_final['run'] = xr.DataArray([run], dims='realization')
        ds_final['ssp'] = xr.DataArray([ssp], dims='realization')
        ds_final['gwl'] = xr.DataArray([gwl], dims='realization')
       
        ds_final = ds_final.load()
        if not os.path.exists(preprocessed_path + 'agg_datasets/gridded_'+gwl+'/'):
            os.makedirs(preprocessed_path + 'agg_datasets/gridded_'+gwl+'/')
        ds_final.to_netcdf(preprocessed_path + 'agg_datasets/gridded_'+gwl+'/agg_'+GCM+'_'+run+'_'+ssp+'_'+gwl+'_ERA5.nc')
        
        ### REANALYSIS
        if reanalysis and not os.path.exists(preprocessed_path + 'agg_datasets/gridded_ref/agg_'+GCM+'_ref_regrid_ERA5.nc'):
            wcf_bis_files, _ = match_files(preprocessed_path+ GCM+'/wcf_ref_' +GCM + '_ERA5')
            scf_bis_files, _ = match_files(preprocessed_path+ GCM+'/scf_ref_' +GCM + '_ERA5')
            wcf_bis = open_dataset_any(wcf_bis_files[0])
            scf_bis = open_dataset_any(scf_bis_files[0])
            wcf_bis = wcf_bis.convert_calendar('standard')
            scf_bis = scf_bis.convert_calendar('standard')
            wcf_bis['time'] = pd.to_datetime(wcf_bis['time'].values)
            scf_bis['time'] = pd.to_datetime(scf_bis['time'].values)

            
            wcf_bis['low_wind'] = xr.where(wcf_bis.wcf <= wcf_thr,1,0)
            scf_bis['low_solar'] = xr.where(scf_bis.scf <= scf_thr,1,0)

            compound_bis = (wcf_bis.low_wind * scf_bis.low_solar)
            compound_bis = compound_bis.to_dataset(name='start_cooc')
            compound_bis = compound_bis.convert_calendar('standard')
            
            # Compute severity
            severity_ds = compute_severity(compound_bis.start_cooc, scf_bis, wcf_bis, scf_thr, wcf_thr)

            # Apply duration_xr over each grid cell (lat/lon) independently, working along time
            ds_dur, ds_freq = duration_xr(compound_bis.start_cooc)
            ds_dur = ds_dur.reindex({'lat': scf_bis.lat, 'lon': scf_bis.lon})
            ds_freq = ds_freq.reindex({'lat': scf_bis.lat, 'lon': scf_bis.lon})
                
            severity_ds['time'] = severity_ds.time.dt.year
            severity_ds = severity_ds.rename({'time':'year'})

            pdd_ref = severity_ds * ds_dur.duration * ds_freq.frequency

            ds_final = ds_dur.copy()
            ds_final['frequency'] = ds_freq.frequency
            ds_final['severity'] = severity_ds
            ds_final['pdd'] = pdd_ref
            
            compound_bis = compound_bis.resample(time='YE').sum()
            compound_bis['time'] = compound_bis.time.dt.year
            compound_bis = compound_bis.rename({'time':'year'})
            ds_final['nb_days'] = compound_bis.start_cooc
            
            ds_final = ds_final.expand_dims({'realization': [i]})
            regrid = xe.Regridder(ds_final, wcf_rea, method='nearest_s2d')
            ds_final = regrid(ds_final)
            
            ds_final['GCM'] = xr.DataArray([GCM], dims='realization')
            ds_final['run'] = xr.DataArray([run], dims='realization')
            ds_final['ssp'] = xr.DataArray([ssp], dims='realization')
            ds_final['gwl'] = xr.DataArray([gwl], dims='realization')
           
            ds_final = ds_final.load()
            if not os.path.exists(preprocessed_path + 'agg_datasets/gridded_ref/'):
                os.makedirs(preprocessed_path + 'agg_datasets/gridded_ref/')
            ds_final.to_netcdf(preprocessed_path + 'agg_datasets/gridded_ref/agg_'+GCM+'_ref_regrid_ERA5.nc')
        
        

    for i, GCM in enumerate(gcm_list):
        process_single_gcm(i, preprocessed_path, reanalysis)
        
        
def load_gridded_data_ds_cf(preprocessed_path, gwl, rolling=1):
    '''
    Load gridded data for compound events and compute severity and duration in parallel.
    
    Parameters:
      preprocessed_path: path to the preprocessed data files.
      rolling: the rolling window size to use.
    
    Returns:
      data: xarray.Dataset with duration, frequency, and severity per year and spatial point.
    '''
    # Find and sort file paths (zarr preferred, falls back to .nc)
    wcf_paths = glob_any(os.path.join(preprocessed_path, '*/wcf_day_*ssp*'+gwl+'_ERA5'))
    scf_paths = glob_any(os.path.join(preprocessed_path, '*/scf_day_*ssp*'+gwl+'_ERA5'))

    #wcf_paths = wcf_paths[0:4]
    #scf_paths = scf_paths[0:4]
    print(wcf_paths)
    
    chunks = {'time': -1, 'lat': 50, 'lon': 50}

    # Extract metadata from file names
    gcm_list = [x.split('_')[-5] for x in wcf_paths]
    run_list = [x.split('_')[-3] for x in wcf_paths]
    ssp_list = [x.split('_')[-4] for x in wcf_paths]
    gwl_list = [x.split('_')[-2] for x in wcf_paths]
    print(gcm_list)

    wcf_rea_files, _ = match_files(preprocessed_path + 'ERA5/wcf_day*')
    wcf_rea = open_dataset_any(wcf_rea_files[0])
    wcf_rea = wcf_rea.isel(time=slice(0,2))
    # Wrap the per-GCM processing in a delayed function
    def process_single_gcm(i, preprocessed_path, rolling):
        GCM = gcm_list[i]
        run = run_list[i]
        ssp = ssp_list[i]
        gwl = gwl_list[i]
        
        wcf_files, _ = match_files(preprocessed_path+ GCM+'/wcf_day_' +GCM + '_' + ssp + '_' + run + '_'+gwl+'_ERA5')
        scf_files, _ = match_files(preprocessed_path+ GCM+'/scf_day_' +GCM + '_' + ssp + '_' + run + '_'+gwl+'_ERA5')
        wcf = open_dataset_any(wcf_files[0])
        scf = open_dataset_any(scf_files[0])
        wcf = wcf.mean(dim='time')
        scf = scf.mean(dim='time')
        ds_final = wcf.copy()
        ds_final['scf'] = scf.scf
        
        ds_final = ds_final.expand_dims({'realization': [i]})
        regrid = xe.Regridder(ds_final, wcf_rea, method='nearest_s2d')
        ds_final = regrid(ds_final)
        
        ds_final['GCM'] = xr.DataArray([GCM], dims='realization')
        ds_final['run'] = xr.DataArray([run], dims='realization')
        ds_final['ssp'] = xr.DataArray([ssp], dims='realization')
        ds_final['gwl'] = xr.DataArray([gwl], dims='realization')
       
        ds_final = ds_final.load()
        if not os.path.exists(preprocessed_path + 'agg_datasets/gridded_'+gwl+'_ds_cf/'):
            os.makedirs(preprocessed_path + 'agg_datasets/gridded_'+gwl+'_ds_cf/')
        ds_final.to_netcdf(preprocessed_path + 'agg_datasets/gridded_'+gwl+'_ds_cf/agg_ds_cf_'+GCM+'_'+run+'_'+ssp+'_'+gwl+'_ERA5.nc')

        
        

    for i, GCM in enumerate(gcm_list):
        process_single_gcm(i, preprocessed_path, rolling)





# -------------------------------

if __name__ == '__main__':
    path_preprocessed = config.PATH_PREPROCESSED
    ssp = config.SSP
    gwl_list = ['GWL1-5', 'GWL2', 'GWL3']

    for gwl in gwl_list:
        load_gridded_data_compound(path_preprocessed, gwl, reanalysis=False)

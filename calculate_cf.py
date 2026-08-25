# -*- coding: cp1252 -*-
import os
import config
os.environ['ESMFMKFILE'] = config.ESMFMKFILE_XENV
import xesmf as xe
import xarray as xr
import numpy as np
import pandas as pd
import glob as glob
import gc
from xclim import sdba
from dask import delayed, compute
import geopandas as gpd
import xagg as xa
import rioxarray as rxr
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
import rasterio

# Zarr/NetCDF-agnostic file lookup, opener, and atomic-write helpers
# (kept dependency-light so fig*.py scripts can import them without
# pulling in xesmf/xclim/xagg the way importing this module would).
from io_utils import (
    match_files as _match_files, open_dataset_any, open_mfdataset_any,
    safe_to_netcdf, safe_to_zarr,
)

from fit_local_shear import fit_local_shear
from compute_solar_cf import compute_solar_cf, PVGISCoefficients, DEFAULT_PVGIS_COEFFICIENTS

# Wind capacity-factor physics (power curve + the three wind_method
# extrapolation strategies) -- kept dependency-light (numpy/xarray only,
# same rationale as io_utils.py) in its own module so it can be imported
# by scripts that don't want xesmf/xclim/xagg (e.g. compare_wind_methods.py).
from wind_potential import (
    WIND_METHODS, DS_CFConfig, DEFAULT_DS_CF_CONFIG,
    compute_wind_potential_from_hub_wind, get_hub_height_wind, compute_wind_potential,
)


# -------------------------
# Helper functions
# -------------------------
# (_match_files / open_dataset_any / open_mfdataset_any / safe_to_netcdf /
# safe_to_zarr now live in io_utils.py, imported above, so fig*.py scripts
# can reuse them without importing this whole module.)

def load_variable(var, GCM, ssp, run, path_folder, gwl, chunks):
    """
    Load a single variable dataset from a file matching a pattern.
    Only keep the variable and lat, lon, time coordinates.
    Drop 'height' variable if present.
    Accepts either NetCDF (.nc) or Zarr (.zarr) stores.
    """
    pattern_base = f"{path_folder}{GCM}/{var}_day_{GCM}_{ssp}_{run}*{gwl}"
    files, _ = _match_files(pattern_base)
    if len(files) == 0:
        raise FileNotFoundError(
            f"No files found for pattern: {pattern_base}.[nc|zarr]")
    try:
        ds = open_dataset_any(files[0], chunks=chunks)
    except Exception:
        ds = open_dataset_any(files[0], chunks=chunks, decode_times=False)
        ds['time'] = pd.to_datetime(ds['time'], unit='D', origin='2015-01-01')
    ds['time'] = ds['time'].dt.floor('D')
    ds = ds[[var, 'lat', 'lon', 'time']]
    if 'height' in ds:
        ds = ds.drop_vars('height')
    return ds


def grids_match(ds1, ds2):
    """Check whether the lat/lon grids of two datasets match."""
    lat_match = (ds1.lat.shape[0] == ds2.lat.shape[0]
                 and np.array_equal(ds1.lat.values, ds2.lat.values))
    lon_match = (ds1.lon.shape[0] == ds2.lon.shape[0]
                 and np.array_equal(ds1.lon.values, ds2.lon.values))
    return lat_match and lon_match


def choose_target_grid(datasets, method="min"):
    """
    Choose a target grid dataset from a dictionary of datasets.
    'min' dataset with the smallest lat/lon dimensions.
    'mode' dataset whose dimensions are the most common.
    """
    dims = [(ds.lat.shape[0], ds.lon.shape[0]) for ds in datasets.values()]
    if method == "min":
        target_lat = min(d[0] for d in dims)
        target_lon = min(d[1] for d in dims)
    elif method == "mode":
        lat_list = [d[0] for d in dims]
        lon_list = [d[1] for d in dims]
        target_lat = max(set(lat_list), key=lat_list.count)
        target_lon = max(set(lon_list), key=lon_list.count)
    else:
        raise ValueError(f"Unknown method for target grid selection: {method!r}")

    for ds in datasets.values():
        if ds.lat.shape[0] == target_lat and ds.lon.shape[0] == target_lon:
            return ds
    return list(datasets.values())[0]


def regrid_to_target(ds, target_ds, var_name):
    """Regrid ds to the target_ds grid if needed."""
    if not grids_match(ds, target_ds):
        print(f"Regridding {var_name}")
        regridder = xe.Regridder(ds, target_ds, method='bilinear')
        return regridder(ds)
    return ds


def rasterize_shapefile(shapefile, coords, shape, transform):
    """Rasterize shapefile geometries onto a grid defined by shape/transform."""
    geometries = shapefile['geometry']
    mask = geometry_mask(
        geometries=geometries,
        all_touched=True,
        out_shape=shape,
        transform=transform,
        invert=True
    )
    return mask


# -------------------------
# Local (per-pixel) wind shear exponent
# -------------------------

def get_local_shear_exponent(era5_file_pattern, path_preprocessed,
                             ref_period=('1982-01-01', '2001-12-31'),
                             overwrite=False):
    """
    Per-pixel Hellmann shear exponent (ref_height -> hub_height), fit from
    reanalysis daily u10/v10/u100/v100 over `ref_period` (see
    fit_local_shear.fit_local_shear), cached to
    {path_preprocessed}/ERA5/shear_exponent_local_{start}_{end}.nc.

    era5_file_pattern : glob pattern for the reanalysis wind files (.nc or
                        .zarr). Only needed if no cached alpha file is found
                        -- checked first at {path_preprocessed}/ERA5/, then
                        at {config.PATH_FOLDER}/ERA5/ (in case it was fit and
                        left alongside the raw reanalysis archive instead of
                        the preprocessed one). Pass overwrite=True to refit
                        from era5_file_pattern regardless of either cache.
    """
    fname = f"shear_exponent_local_{ref_period[0]}_{ref_period[1]}.nc"
    out_dir = os.path.join(path_preprocessed, 'ERA5')
    out_path = os.path.join(out_dir, fname)
    fallback_path = os.path.join(config.PATH_FOLDER, 'ERA5', fname)

    if not overwrite and os.path.exists(out_path):
        alpha = xr.open_dataset(out_path)['alpha']
    elif not overwrite and os.path.exists(fallback_path):
        alpha = xr.open_dataset(fallback_path)['alpha']
    else:
        if not era5_file_pattern:
            raise FileNotFoundError(
                f"No cached local shear exponent at {out_path} or "
                f"{fallback_path}, and no era5_file_pattern given to fit one."
            )
        os.makedirs(out_dir, exist_ok=True)
        print(f"Fitting local shear exponent from {era5_file_pattern} over {ref_period}")
        alpha_ds = fit_local_shear(era5_file_pattern, time_slice=ref_period)
        safe_to_netcdf(alpha_ds, out_path)
        alpha = alpha_ds['alpha']

    rename = {}
    if 'latitude' in alpha.dims:
        rename['latitude'] = 'lat'
    if 'longitude' in alpha.dims:
        rename['longitude'] = 'lon'
    if rename:
        alpha = alpha.rename(rename)
    return alpha


def regrid_alpha_to_grid(alpha, target_grid, interp_method='linear'):
    """Interpolate a per-pixel shear-exponent DataArray onto target_grid's lat/lon."""
    return alpha.interp(lat=target_grid['lat'], lon=target_grid['lon'], method=interp_method)


def get_gcm_shear_exponent(GCM, shear_by_gcm_dir, target_grid,
                           ref_period=('1982-01-01', '2001-12-31')):
    """
    Per-pixel local Hellmann shear exponent for `GCM`, already fit on GCM's
    own native grid (see shear_by_gcm/compute_shear_by_gcm.py, which
    regrids ERA5 u10/v10/u100/v100 to the GCM grid *before* fitting alpha,
    rather than fitting on the ERA5 grid and interpolating alpha itself as
    get_local_shear_exponent / regrid_alpha_to_grid do). Sea pixels are
    already NaN (masked with shp_re.shp at fit time).

    No interpolation needed here -- unlike get_local_shear_exponent, the
    cached alpha sits on GCM's native grid at the same resolution as
    target_grid, just possibly over a wider lat/lon extent (e.g. the shear
    cache was fit once on a looser domain crop than whatever shapefile-
    cropped grid is in use now). Select target_grid's lat/lon out of the
    cache (nearest, with a tight tolerance for float round-trip noise)
    rather than requiring an exact shape match, and reuse target_grid's own
    coordinate values so downstream alignment (compute_wind_potential's
    xr.where) is exact.
    """
    path = os.path.join(shear_by_gcm_dir, f"shear_exponent_{GCM}_{ref_period[0]}_{ref_period[1]}.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No per-GCM shear exponent file for {GCM} at {path}. "
            "Run shear_by_gcm/compute_shear_by_gcm.py first."
        )
    alpha = xr.open_dataset(path)['alpha']

    # tolerance must be applied per-dimension via separate .sel() calls --
    # passing a single .sel(lat=..., lon=..., tolerance=(lat_tol, lon_tol))
    # does NOT apply lat_tol/lon_tol per axis (verified: it raises KeyError
    # even for points within each axis's own tolerance).
    lat_res = float(np.median(np.abs(np.diff(alpha['lat'].values))))
    lon_res = float(np.median(np.abs(np.diff(alpha['lon'].values))))
    try:
        alpha = alpha.sel(lat=target_grid['lat'], method='nearest', tolerance=lat_res / 4)
        alpha = alpha.sel(lon=target_grid['lon'], method='nearest', tolerance=lon_res / 4)
    except KeyError:
        raise ValueError(
            f"Cached shear exponent grid for {GCM} "
            f"(lat {alpha['lat'].values.min():.3f}..{alpha['lat'].values.max():.3f}, "
            f"lon {alpha['lon'].values.min():.3f}..{alpha['lon'].values.max():.3f}) "
            "does not cover the target grid "
            f"(lat {target_grid['lat'].values.min():.3f}..{target_grid['lat'].values.max():.3f}, "
            f"lon {target_grid['lon'].values.min():.3f}..{target_grid['lon'].values.max():.3f}) "
            "-- refit with compute_shear_by_gcm.py."
        )
    return alpha.assign_coords(lat=target_grid['lat'], lon=target_grid['lon'])


# -------------------------
# Main loading function
# -------------------------

def load_ds(GCM, ssp, run, path_folder, gwl):
    """
    Load tas, rsds, and either (uas, vas) or sfcWind.
    Regrid to a common grid, merge, and convert the calendar.
    """
    chunks = {'time': -1, 'lat': 100, 'lon': 100}

    dtas    = load_variable('tas',    GCM, ssp, run, path_folder, gwl, chunks)
    drsds   = load_variable('rsds',   GCM, ssp, run, path_folder, gwl, chunks)

    datasets = {'tas': dtas, 'rsds': drsds}

    uas_pattern_base = f"{path_folder}{GCM}/uas_*{GCM}_{ssp}_{run}*{gwl}"
    uas_files, _ = _match_files(uas_pattern_base)
    if uas_files:
        duas = load_variable('uas', GCM, ssp, run, path_folder, gwl, chunks)
        dvas = load_variable('vas', GCM, ssp, run, path_folder, gwl, chunks)
        datasets.update({'uas': duas, 'vas': dvas})
        target_ds = choose_target_grid(datasets, method="min")
        for key, ds in datasets.items():
            datasets[key] = regrid_to_target(ds, target_ds, key)
        ds = xr.merge([datasets['tas'],
                       datasets['rsds'], datasets['uas'], datasets['vas']])
        ds['sfcWind'] = np.sqrt(ds.uas**2 + ds.vas**2)
        ds = ds.drop_vars(['uas', 'vas'])
    else:
        dsfcWind = load_variable('sfcWind', GCM, ssp, run, path_folder, gwl, chunks)
        datasets['sfcWind'] = dsfcWind
        target_ds = choose_target_grid(datasets, method="mode")
        for key, ds_item in datasets.items():
            datasets[key] = regrid_to_target(ds_item, target_ds, key)
        ds = xr.merge(list(datasets.values()))

    try:
        ds = ds.convert_calendar('standard')
    except Exception:
        ds = ds.convert_calendar('standard', align_on='year')
        ds = ds.interp(time=pd.date_range(ds.time.values[0], ds.time.values[-1], freq='D'))

    ds = ds.sortby('lat').sortby('lon').sortby('time')
    ds = ds.chunk(chunks)
    return ds


def set_variable_units(ds, var_units):
    """Set the 'units' attribute for each variable listed in var_units."""
    for var, units in var_units.items():
        if var in ds:
            ds[var].attrs['units'] = units
    return ds


def filter_domain(ds, lat_range, lon_range):
    """Subset ds to the given lat/lon ranges."""
    return ds.sel(lat=slice(lat_range[0], lat_range[1]),
                  lon=slice(lon_range[0], lon_range[1]))


def get_output_filename(path_preprocessed, GCM, ssp, run, gwl, reanalysis):
    """Return the expected bias-corrected output filename."""
    folder = os.path.join(path_preprocessed, GCM)
    fname = f"dadjusted_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}.nc"
    return os.path.join(folder, fname)


# -------------------------
# Bias correction
# -------------------------

def unbias_GCM(GCM, run, ssp, path_preprocessed, shapefile_path, path_folder, gwl_list,
               reanalysis='ERA5', overwrite=False):
    """
    Train an MBCn adjustment on historical data and apply it to each future GWL.

    Each GWL is processed as a true Dask delayed task; all tasks are submitted
    before any computation starts, so they can run in parallel on the cluster.
    """
    print("Starting unbias_GCM function")

    gwl_unbias = []
    for gwl in gwl_list:
        outfile = get_output_filename(path_preprocessed, GCM, ssp, run, gwl, reanalysis)
        if os.path.exists(outfile) and not overwrite:
            print(f"{gwl}: File already exists")
        else:
            gwl_unbias.append(gwl)

    if not gwl_unbias:
        print("All files already exist. Exiting function.")
        return

    # ------------------------------------------------------------------
    # 1. Historical simulation (GWL0-61)
    # ------------------------------------------------------------------
    dhist = load_ds(GCM, ssp, run, path_folder, 'GWL0-61').dropna('time', how='all')
    chunk_loc = 60

    files_ref, _ = _match_files(os.path.join(path_folder, reanalysis, f"*{reanalysis}*"))
    if not files_ref:
        raise FileNotFoundError(
            f"No reanalysis files found in {os.path.join(path_folder, reanalysis)}")
    dref = open_mfdataset_any(files_ref)
    dref = _standardize_reanalysis_names(dref)

    if 'sfcWind' not in dref:
        if not {'u10', 'v10'}.issubset(dref.data_vars):
            raise KeyError("Need either 'sfcWind' or both 'u10' and 'v10' in reanalysis")
        print("Computing sfcWind from u10/v10")
        dref['sfcWind'] = np.hypot(dref['u10'], dref['v10'])

    dref = dref.sortby('lat').sortby('lon').sortby('time')
    dhist = dhist.sortby('lat').sortby('lon').sortby('time')
    dref = dref.chunk({'time': -1, 'lat': 50, 'lon': 50})

    lat_range = (dref.lat.values[0], dref.lat.values[-1])
    lon_range = (dref.lon.values[0], dref.lon.values[-1])

    dhist = filter_domain(dhist, lat_range, lon_range)
    dhist = dhist.chunk({'time': -1, 'lat': 20, 'lon': 20})

    # --- Build the raster mask from the shapefile ---
    mask_template = dref.tas.isel(time=0).load()
    shapefile = gpd.read_file(shapefile_path)
    lons, lats = np.meshgrid(mask_template.lon, mask_template.lat)
    coords = np.array([lons.flatten(), lats.flatten()]).T
    transform = rasterio.transform.from_bounds(
        mask_template.lon.min().item(), mask_template.lat.min().item(),
        mask_template.lon.max().item(), mask_template.lat.max().item(),
        len(mask_template.lon), len(mask_template.lat)
    )
    mask = rasterize_shapefile(shapefile, coords, mask_template.shape, transform)
    mask = mask[::-1, :]

    dref = dref.where(mask == 1, np.nan)
    dref['mask'] = xr.where(~np.isnan(dref.isel(time=0).tas), 1, 0)
    print("dref mask coverage:", float(dref['mask'].sum() / dref['mask'].count()))

    # Regrid reference to dhist grid. skipna=True matters specifically for
    # ssrd: it comes from ERA5-Land (ocean masked as NaN, native coverage
    # only down to -57.1 S), unlike tas/sfcWind which come from full-globe
    # ERA5. Without skipna, xesmf NaNs out any target cell that overlaps
    # even one ocean source pixel, wiping out coastal land cells that do
    # have valid ssrd data, not just genuinely NaN open-ocean cells.
    regridder = xe.Regridder(dref, dhist, method='conservative_normed')
    dref = regridder(dref, skipna=True, output_chunks={'lat': 50, 'lon': 50})
    dref = dref.convert_calendar('noleap').convert_calendar('standard')
    dhist = dhist.convert_calendar('noleap').convert_calendar('standard')

    # Second, finer mask on the regridded grid
    ref_grid = dref.tas.isel(time=0)

    def create_mask_from_shapefile(grid, shapefile):
        transform = rasterio.transform.from_bounds(
            grid.lon.min().item(), grid.lat.min().item(),
            grid.lon.max().item(), grid.lat.max().item(),
            len(grid.lon), len(grid.lat)
        )
        shape = (len(grid.lat), len(grid.lon))
        mask = geometry_mask(
            geometries=shapefile.geometry,
            all_touched=True,
            out_shape=shape,
            transform=transform,
            invert=True
        )
        return xr.DataArray(
            mask[::-1, :], dims=("lat", "lon"),
            coords={"lat": grid.lat, "lon": grid.lon}
        )

    mask_array = create_mask_from_shapefile(ref_grid, shapefile)

    var_units = {'sfcWind': 'm s-1', 'tas': 'K', 'rsds': 'W m-2'}
    dref = dref.where(mask_array)
    dhist = dhist.where(mask_array)
    dref = set_variable_units(dref, var_units)
    dhist = set_variable_units(dhist, var_units)

    dref = dref.sel(time=slice('1982-01-01', '2001-12-31'))
    dref = dref[['sfcWind', 'tas', 'rsds']]

    lon_ori = dref.lon
    lat_ori = dref.lat

    dref = dref.stack(location=("lat", "lon"))
    dhist = dhist.stack(location=("lat", "lon"))

    # Jitter lower bounds set to a fixed safe value
    rsds_low = 1e-6
    wind_low = 1e-6

    def remove_constant_locations(da, dim='time'):
        """Drop locations where any single variable is constant or entirely
        NaN across `dim`. An all-NaN slice has std == NaN, which `std == 0`
        alone does not catch (NaN == 0 is False), so it's checked explicitly
        -- otherwise a location with just one dead variable (e.g. rsds NaN
        everywhere while tas/sfcWind are fine) silently survives and poisons
        the multivariate (stacked) sample used downstream."""
        for v in da.data_vars:
            if dim in da[v].dims:
                std = da[v].std(dim=dim)
                is_allnan = da[v].isnull().all(dim=dim)
                is_const = ((std == 0) | is_allnan).compute()
                if is_const.any():
                    idx_to_remove = da.location[is_const]
                    print(f"Variable '{v}': removing {len(idx_to_remove)} constant/all-NaN locations.")
                    da = da.drop_sel(location=idx_to_remove)
        return da

    valid_mask = (~dref.tas.isnull().all('time')).compute()
    nb = valid_mask.count()
    print("Valid fraction before cleaning:", float(valid_mask.sum() / nb))
    dref = remove_constant_locations(dref)
    dref = dref.dropna(dim='location', how='all')
    print("Remaining locations:", dref.location.shape[0])

    # dref's own constant/all-NaN scrub above says nothing about dhist: a
    # location can have real variance in the reanalysis but be constant or
    # all-NaN in the GCM historical run (e.g. after masking/regridding).
    # MBCn's per-location energy-score training (ref vs. hist) divides by
    # zero on such a location, so dhist needs the same scrub, and both
    # sides must be re-intersected on the result.
    dhist_matched = dhist.sel(location=dref.location).sortby('location')
    dhist_matched = remove_constant_locations(dhist_matched)
    dhist_matched = dhist_matched.dropna(dim='location', how='all')

    dref = dref.sel(location=dhist_matched.location).sortby('location')
    dhist = dhist_matched
    print("Remaining locations after dhist scrub:", dref.location.shape[0])

    for ds_name, ds_obj in [('dref', dref), ('dhist', dhist)]:
        ds_obj = ds_obj.assign(
            rsds=sdba.processing.to_additive_space(
                sdba.processing.jitter(ds_obj.rsds,
                                       lower=f"{rsds_low} W m-2", minimum="0 W m-2"),
                lower_bound="0 W m-2", trans="log",
            ),
            sfcWind=sdba.processing.to_additive_space(
                sdba.processing.jitter(ds_obj.sfcWind,
                                       lower=f"{wind_low} m s-1", minimum="0 m s-1"),
                lower_bound="0 m s-1", trans="log",
            )
        )
        if ds_name == 'dref':
            dref = ds_obj
        else:
            dhist = ds_obj

    loc_values = dref.location.to_dataframe().reset_index(drop=True)
    loc_values['location_index'] = loc_values.index

    ref  = dref.drop_vars(['lat', 'lon', 'location'])
    hist = dhist.drop_vars(['lat', 'lon', 'location'])
    ref  = sdba.processing.stack_variables(ref)
    hist = sdba.processing.stack_variables(hist)

    # Align historical time axis onto the reference period
    hist = hist.assign_coords(time=hist.time - hist.time.values[-1] + ref.time.values[-1])
    common_times = np.intersect1d(ref.time.values, hist.time.values)
    print(f"[Diag] ref.time size={ref.time.size}, hist.time size={hist.time.size}, "
          f"common_times={common_times.size} "
          f"(ref-only={ref.time.size - common_times.size}, "
          f"hist-only={hist.time.size - common_times.size})")
    hist = hist.where(hist.time.isin(common_times))
    ref  = ref.where(ref.time.isin(common_times))

    # .where() masks non-matching steps to NaN rather than dropping them, so
    # a calendar-label mismatch between ref/hist shows up as NaNs scattered
    # across time for every location, not as fully-NaN/constant locations
    # (which is all remove_constant_locations can catch). Quantify that here.
    for name, da in [('ref', ref), ('hist', hist)]:
        nan_frac = float(da.isnull().mean().compute())
        nan_per_loc = da.isnull().any('multivar').sum('time').compute()
        print(f"[Diag] {name}: overall NaN fraction={nan_frac:.4%}, "
              f"locations with >=1 NaN time step={int((nan_per_loc > 0).sum())} "
              f"(out of {da.sizes['location']}), "
              f"max NaN time steps at one location={int(nan_per_loc.max())}")

    ref  = ref.chunk({'time': -1, 'location': chunk_loc})
    hist = hist.chunk({'time': -1, 'location': chunk_loc})
    print(f"Training window time steps: ref={ref.sizes['time']}, hist={hist.sizes['time']}")

    # Train once
    ADJ = sdba.MBCn.train(
        ref, hist,
        base_kws={"nquantiles": 30, "group": "time"},
        adj_kws={"interp": "nearest", "extrapolation": "constant"},
        n_iter=20,
        n_escore=1000,
        pts_dim='multivar',
    )
    # xclim's TrainAdjust objects subclass dict (via Parametrizable), so
    # dask.delayed would otherwise traverse ADJ like a plain mapping and
    # rebuild it as a bare dict inside the delayed tasks, dropping .adjust().
    ADJ = delayed(ADJ, traverse=False)

    # Load all future datasets eagerly so they are available inside delayed tasks
    dfut_datasets = {
        gwl: load_ds(GCM, ssp, run, path_folder, gwl) for gwl in gwl_unbias
    }

    # ------------------------------------------------------------------
    # 2. Process each future GWL as a *true* Dask delayed task
    # ------------------------------------------------------------------

    @delayed
    def process_gwl_delayed(dfut, dref, gwl, ADJ, ref, hist, loc_values, mask_array,
                             GCM, ssp, run, path_preprocessed, reanalysis,
                             lat_ori, lon_ori, rsds_low, wind_low, chunk_loc):
        """
        Bias-correct a single future GWL dataset and write it to disk.
        Decorated with @delayed so all GWLs can be submitted simultaneously
        and executed in parallel by the Dask scheduler.

        Note: the time axis of `fut` is shifted onto the reference period
        because MBCn requires training and simulation data to share the same
        calendar positions (the adjustment is purely distributional, not
        chronological).
        """
        print(f"[Delayed] Processing GWL: {gwl}")

        dfut = filter_domain(dfut, (lat_ori[0], lat_ori[-1]), (lon_ori[0], lon_ori[-1]))
        dfut = set_variable_units(dfut,
                                  {'sfcWind': 'm s-1', 'tas': 'K', 'rsds': 'W m-2'})
        n_raw = dfut.sizes['time']
        dfut = dfut.convert_calendar('noleap').convert_calendar('standard')
        print(f"[Delayed] GWL {gwl}: dfut time steps raw={n_raw}, "
              f"after noleap/standard round-trip={dfut.sizes['time']}")
        dfut = dfut.sortby('lat').sortby('lon').sortby('time')
        dfut = dfut.stack(location=("lat", "lon"))

        dfut = dfut.assign(
            rsds=sdba.processing.to_additive_space(
                sdba.processing.jitter(dfut.rsds,
                                       lower=f"{rsds_low} W m-2", minimum="0 W m-2"),
                lower_bound="0 W m-2", trans="log",
            ),
            sfcWind=sdba.processing.to_additive_space(
                sdba.processing.jitter(dfut.sfcWind,
                                       lower=f"{wind_low} m s-1", minimum="0 m s-1"),
                lower_bound="0 m s-1", trans="log",
            )
        )

        # MBCn was trained on ref/hist at exactly this length (group="time"
        # builds a single block spanning the whole series, purely
        # positional -- it does not use calendar labels), so sim must match
        # that length positionally. ref.sizes['time'] is the authoritative
        # number: `ref`/`hist` keep their full length through the NaN-mask
        # step above (`.where(...)` doesn't drop rows). A calendar-label
        # intersection (np.intersect1d on ref.time/hist.time) is NOT the
        # same number -- ref and hist come from different source calendars
        # and only get a constant-offset shift, so their labels drift out
        # of exact agreement over a multi-year window even though both
        # arrays are still length ref.sizes['time']. Using that smaller
        # intersection count here previously caused
        # "IndexError: ... size 7295" (5 short of the 7300 MBCn trained on).
        n = ref.sizes['time']
        if dfut.sizes['time'] < n:
            raise ValueError(
                f"GWL {gwl}: future window has only {dfut.sizes['time']} "
                f"time steps, need at least {n} to match the training period."
            )

        # Match locations only (independent of the time-length fix above);
        # dref/dfut can have different surviving locations after masking.
        _, dfut = xr.align(dref.isel(time=0, drop=True), dfut, join="inner", copy=False)

        dfut = dfut.isel(time=slice(-n, None))
        real_future_times = dfut.time.values  # restore onto adj before saving
        dfut = dfut.assign_coords(time=ref.time.values)

        # Diagnostic: MBCn's per-location energy-score step can divide by
        # zero if a location's block is all-NaN or exactly constant. dref
        # already gets a constant/all-NaN location scrub (remove_constant_
        # locations + dropna), dfut never has, so check it explicitly here.
        for var in ['tas', 'rsds', 'sfcWind']:
            da = dfut[var].compute()
            n_allnan = int(da.isnull().all('time').sum())
            n_const = int((da.std('time', skipna=True) == 0).sum())
            print(f"[Delayed] GWL {gwl}: {var} -- all-NaN locations={n_allnan}, "
                  f"zero-variance locations={n_const} (out of {da.sizes['location']})")

        dfut = dfut.drop_vars(['lat', 'lon', 'location'], errors='ignore')
        fut  = sdba.processing.stack_variables(dfut)
        fut = fut.chunk({'time': -1, 'location': chunk_loc})

        adj = ADJ.adjust(
            ref=ref,
            hist=hist,
            sim=fut,
            base=sdba.QuantileDeltaMapping,
            adj_kws={"interp": "linear", "extrapolation": "constant"},
        )

        adj = sdba.unstack_variables(adj).compute()
        adj = adj.assign_coords(time=real_future_times)
        adj = adj.assign(
            rsds=sdba.processing.from_additive_space(adj.rsds),
            sfcWind=sdba.processing.from_additive_space(adj.sfcWind)
        )

        adj = adj.assign_coords(location=loc_values['location'])
        adj = (adj
               .assign_coords(lat=("location", loc_values["lat"].values),
                               lon=("location", loc_values["lon"].values))
               .set_index(location=["lat", "lon"])
               .unstack("location"))

        adj = adj.reindex(lat=lat_ori, lon=lon_ori)
        adj.rsds.values[adj.rsds.values < 1e-5] = 0
        adj.sfcWind.values[adj.sfcWind.values < 1e-5] = 0

        for var in adj.data_vars:
            adj[var] = adj[var].astype(np.float32)

        out_file = get_output_filename(path_preprocessed, GCM, ssp, run, gwl, reanalysis)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        adj.to_netcdf(out_file)
        print(f"[Delayed] Saved: {out_file}")

    # Build the list of delayed tasks nothing runs yet
    tasks = [
        process_gwl_delayed(
            dfut_datasets[gwl], dref, gwl, ADJ, ref, hist,
            loc_values, mask_array, GCM, ssp, run,
            path_preprocessed, reanalysis,
            lat_ori, lon_ori, rsds_low, wind_low, chunk_loc
        )
        for gwl in gwl_unbias
    ]

    # Trigger parallel execution of all tasks
    print(f"Submitting {len(tasks)} delayed task(s) to the Dask scheduler...")
    compute(*tasks)
    print("All GWL tasks completed.")


# -------------------------
# DS_CF calculation
# -------------------------

# Seconds represented by one ssrd accumulation step, for converting raw
# ERA5 ssrd (accumulated J/m2) to rsds (W/m2 mean). The wcf_day_*/scf_day_*
# naming and config.ERA5_WIND_PATTERN ("ERA5_daily_*.zarr") indicate the raw
# files here are daily sums of 24 hourly J/m2 values (see the sibling
# como24_group5/code_review/calculate_wind_solar_cf.py:load_era5, which uses
# the same 86400 divisor for its "daily" file family) -- if your raw ERA5
# ssrd is still hourly-accumulated, pass ssrd_accum_seconds=3600 instead.
ERA5_SSRD_ACCUM_SECONDS = 86400.0


def _standardize_reanalysis_names(ds, ssrd_accum_seconds=ERA5_SSRD_ACCUM_SECONDS):
    """
    Rename raw ERA5 dims/vars (latitude/longitude/valid_time/t2m/ssrd) to
    the CF-ish names used downstream (lat/lon/time/tas/rsds), and convert
    ssrd from accumulated J/m2 to a W/m2 mean (divide by ssrd_accum_seconds)
    since compute_solar_cf expects rsds in W/m2. u10/v10 are left as-is
    since they're combined into sfcWind later. No-op for datasets that
    already use the target names.
    """
    rename = {}
    for src, dst in (
        ('latitude', 'lat'), ('longitude', 'lon'), ('valid_time', 'time'),
        ('t2m', 'tas'), ('ssrd', 'rsds'),
    ):
        if src in ds.dims or src in ds.variables:
            rename[src] = dst
    had_ssrd = 'ssrd' in ds.variables
    ds = ds.rename(rename) if rename else ds
    if had_ssrd:
        ds['rsds'] = ds['rsds'] / ssrd_accum_seconds
        ds['rsds'].attrs['units'] = 'W m-2'
    return ds


def calculate_ds_cf_reanalysis_grid_GCM(
    GCM, run, ssp,
    path_preprocessed, path_folder,
    reanalysis='ERA5',
    shapefile_path=None,
    cfg: DS_CFConfig = DEFAULT_DS_CF_CONFIG,
    pv_cfg: PVGISCoefficients = DEFAULT_PVGIS_COEFFICIENTS,
    shear_ref_period=('1982-01-01', '2001-12-31'),
    shear_by_gcm_dir=None,
):
    """
    Compute wcf/scf reference files from reanalysis regridded to the GCM grid.

    shear_by_gcm_dir : folder of precomputed per-GCM shear exponent files
                        (see get_gcm_shear_exponent / shear_by_gcm/compute_shear_by_gcm.py).
                        Defaults to config.SHEAR_BY_GCM_DIR.
    """
    if shear_by_gcm_dir is None:
        shear_by_gcm_dir = config.SHEAR_BY_GCM_DIR

    out_folder = os.path.join(path_preprocessed, GCM)
    os.makedirs(out_folder, exist_ok=True)
    # wind_method is tagged onto the wcf filename so the three methods don't
    # overwrite each other; 'shear_local' (the original default) keeps the
    # untagged name for backward compatibility -- see calculate_ds_cf_reanalysis.
    wcf_suffix = '' if cfg.wind_method == 'shear_local' else f'_{cfg.wind_method}'
    path_wcf_ref = os.path.join(out_folder, f"wcf_ref_{GCM}_{reanalysis}{wcf_suffix}.zarr")
    path_scf_ref = os.path.join(out_folder, f"scf_ref_{GCM}_{reanalysis}.zarr")

    if os.path.exists(path_wcf_ref) and os.path.exists(path_scf_ref):
        print("DS_CF ref files already exist, skipping.")
        return

    # 1. Load GCM grid template. Uses dadjusted_* (unbias_GCM's output) rather
    # than wcf_day_* (calculate_ds_cf_GCM's output, derived from dadjusted_*)
    # because only the lat/lon grid is needed here, and calculate_ds_cf_GCM
    # -- which produces wcf_day_* -- runs later in the pipeline than this
    # function; requiring wcf_day_* here would make GWL0-61 a circular
    # dependency for any GCM/run seen for the first time.
    gcm_file_base = os.path.join(
        path_preprocessed, GCM,
        f"dadjusted_{GCM}_{ssp}_{run}_GWL0-61_{reanalysis}"
    )
    gcm_files, _ = _match_files(gcm_file_base)
    if not gcm_files:
        raise FileNotFoundError(f"No GCM file found for pattern: {gcm_file_base}.[nc|zarr]")
    print("Loading GCM file:", gcm_files[0])
    ds_gcm = open_dataset_any(gcm_files[0], chunks={'time': 100})
    gcm_grid = xr.Dataset(coords={"lat": ds_gcm["lat"], "lon": ds_gcm["lon"]})

    # 2. Load reanalysis
    files_ref, _ = _match_files(os.path.join(path_folder, reanalysis, f"*{reanalysis}*"))
    if not files_ref:
        raise FileNotFoundError(
            f"No reanalysis files found in {os.path.join(path_folder, reanalysis)}")

    print(f"Found {len(files_ref)} reanalysis files for {reanalysis}")
    dref = open_mfdataset_any(
        files_ref, combine='by_coords', chunks={}, parallel=True
    )
    dref = _standardize_reanalysis_names(dref)
    dref = dref.sortby('lat').sortby('lon').sortby('time')

    for v in ('tas', 'rsds'):
        if v not in dref:
            raise KeyError(f"{v} not found in reanalysis dataset")

    if 'sfcWind' not in dref:
        if not {'u10', 'v10'}.issubset(dref.data_vars):
            raise KeyError("Need either 'sfcWind' or both 'u10' and 'v10' in reanalysis")
        print("Computing sfcWind from u10/v10")
        dref['sfcWind'] = np.hypot(dref['u10'], dref['v10'])

    keep_vars = ['tas', 'rsds', 'sfcWind']
    if cfg.wind_method == 'wind100':
        if not {'u100', 'v100'}.issubset(dref.data_vars):
            raise KeyError(
                "wind_method='wind100' requires 'u100'/'v100' in the reanalysis "
                f"files matched under {os.path.join(path_folder, reanalysis)}"
            )
        keep_vars += ['u100', 'v100']
    dref = dref[keep_vars]
    dref = dref.chunk({'time': -1, 'lat': 50, 'lon': 50})

    # 3. Optional shapefile mask (before regrid)
    if shapefile_path is not None:
        print("Applying shapefile mask:", shapefile_path)
        shapefile = gpd.read_file(shapefile_path)
        mask_template = dref.tas.isel(time=0).load()
        lons, lats = np.meshgrid(mask_template.lon, mask_template.lat)
        coords = np.array([lons.flatten(), lats.flatten()]).T
        transform = rasterio.transform.from_bounds(
            float(mask_template.lon.min()), float(mask_template.lat.min()),
            float(mask_template.lon.max()), float(mask_template.lat.max()),
            len(mask_template.lon), len(mask_template.lat)
        )
        mask = rasterize_shapefile(shapefile, coords, mask_template.shape, transform)
        mask = mask[::-1, :]
        dref = dref.where(mask == 1, np.nan)
        mask_da = xr.where(~np.isnan(mask_template), mask, 0)
        dref['mask'] = mask_da
        print("Fraction of grid kept after mask:",
              float(dref['mask'].sum() / dref['mask'].count()))
        dref = dref.drop_vars('mask')

    # 4. Regrid to GCM grid
    print("Regridding reanalysis to GCM grid...")
    regridder = xe.Regridder(dref, gcm_grid, method='conservative_normed', reuse_weights=False)
    dref_rg = regridder(dref, output_chunks={'lat': 50, 'lon': 50})
    ds_gcm.close()

    dref_rg = dref_rg.convert_calendar('noleap').convert_calendar('standard')
    dref_rg['tas'] = dref_rg['tas'] - 273.15

    std_mask = dref_rg.tas.std(dim='time')
    if hasattr(std_mask, 'compute'):
        std_mask = std_mask.compute()
    dref_rg = dref_rg.where(~std_mask.isnull() & (std_mask != 0), drop=True)

    # Persist once here: the solar and wind branches below both build a
    # separate graph on top of dref_rg (open + regrid + calendar-convert +
    # mask), so without persisting, Dask would redo all of that work twice.
    dref_rg = dref_rg.persist()

    # 5. Solar potential (scf), PVGIS relative-efficiency + Faiman
    #    module-temperature model.
    if os.path.exists(path_scf_ref):
        print("scf file already exists, skipping:", path_scf_ref)
    else:
        print("Computing solar potential (scf)...")
        scf = compute_solar_cf(dref_rg['tas'], dref_rg['rsds'], dref_rg['sfcWind'], cfg=pv_cfg)

        solar_potential = scf.to_dataset(name='scf').convert_calendar('noleap')
        solar_potential = solar_potential.chunk({'time': 100, 'lat': -1, 'lon': -1})
        solar_potential['scf'] = solar_potential['scf'].astype('f4')
        solar_potential.attrs.update({
            'DESCRIPTION': f'scf reference for {reanalysis} regridded to {GCM} grid',
            'units': 'dimensionless',
            'long_name': 'PVtot potential',
            'SOURCE': 'calculate_ds_cf_reanalysis_grid_GCM',
            'AUTHOR': 'Colin Lenoble',
            'MODEL': 'PVGIS relative efficiency + Faiman module temperature (calculate_wind_solar_cf.py)',
        })
        solar_potential = solar_potential.compute()
        safe_to_zarr(solar_potential, path_scf_ref)
        print("Written scf to", path_scf_ref)
        solar_potential.close()

    # 6. Wind potential (wcf). cfg.wind_method selects how ref_height wind is
    #    turned into hub_height wind (see get_hub_height_wind): a per-pixel
    #    shear exponent precomputed on GCM's own native grid (default,
    #    'shear_local' -- see get_gcm_shear_exponent), a single global
    #    exponent ('shear_uniform'), or the reanalysis 100 m wind read
    #    directly, regridded to the GCM grid alongside sfcWind ('wind100').
    if os.path.exists(path_wcf_ref):
        print("wcf file already exists, skipping:", path_wcf_ref)
    else:
        print(f"Computing wind potential (wcf), wind_method={cfg.wind_method!r}...")
        if cfg.wind_method == 'shear_local':
            alpha = get_gcm_shear_exponent(GCM, shear_by_gcm_dir, gcm_grid, shear_ref_period)
        else:
            alpha = None
        wind_hub = get_hub_height_wind(dref_rg, cfg, alpha=alpha)
        wind_pot = compute_wind_potential_from_hub_wind(wind_hub, cfg)

        wind_potential = wind_pot.to_dataset(name='wcf')
        wind_potential = wind_potential.chunk({'time': 100, 'lat': -1, 'lon': -1})
        wind_potential['wcf'] = wind_potential['wcf'].astype('f4')
        wind_potential.attrs.update({
            'DESCRIPTION': f'wcf reference for {reanalysis} regridded to {GCM} grid',
            'units': 'dimensionless',
            'long_name': 'Wind potential',
            'SOURCE': 'calculate_ds_cf_reanalysis_grid_GCM',
            'AUTHOR': 'Colin Lenoble',
            'wind_method': cfg.wind_method,
            'uniform_shear_exponent': cfg.uniform_shear_exponent if cfg.wind_method == 'shear_uniform' else 'n/a',
        })
        wind_potential = wind_potential.compute()
        safe_to_zarr(wind_potential, path_wcf_ref)
        print("Written wcf to", path_wcf_ref)
        wind_potential.close()

    dref_rg.close()
    dref.close()
    gc.collect()
    print("DS_CF reference files saved:", path_scf_ref, path_wcf_ref)


def calculate_ds_cf_GCM(GCM, run, ssp, path_preprocessed, gwl,
                      reanalysis='ERA5', cfg: DS_CFConfig = DEFAULT_DS_CF_CONFIG,
                      pv_cfg: PVGISCoefficients = DEFAULT_PVGIS_COEFFICIENTS,
                      shear_ref_period=('1982-01-01', '2001-12-31'),
                      shear_by_gcm_dir=None):
    """
    Compute wcf/scf from a bias-corrected GCM file.

    shear_by_gcm_dir : folder of precomputed per-GCM shear exponent files
                        (see get_gcm_shear_exponent / shear_by_gcm/compute_shear_by_gcm.py).
                        Defaults to config.SHEAR_BY_GCM_DIR.
    """
    if shear_by_gcm_dir is None:
        shear_by_gcm_dir = config.SHEAR_BY_GCM_DIR

    ds_path_base = os.path.join(path_preprocessed, GCM,
                                f"dadjusted_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}")
    ds_files, _ = _match_files(ds_path_base)
    if not ds_files:
        raise FileNotFoundError(f"No dadjusted file found for pattern: {ds_path_base}.[nc|zarr]")
    ds = open_dataset_any(ds_files[0])

    # wind_method is tagged onto the wcf filename so the three methods don't
    # overwrite each other; 'shear_local' (the original default) keeps the
    # untagged name for backward compatibility -- see calculate_ds_cf_reanalysis.
    wcf_suffix = '' if cfg.wind_method == 'shear_local' else f'_{cfg.wind_method}'
    wcf_path = os.path.join(path_preprocessed, GCM,
                            f"wcf_day_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}{wcf_suffix}.zarr")
    scf_path = os.path.join(path_preprocessed, GCM,
                            f"scf_day_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}.zarr")

    if os.path.exists(wcf_path) and os.path.exists(scf_path):
        print('DS_CF files already exist')
        return

    print('Calculating DS_CF')
    ds = ds.convert_calendar('noleap')
    ds['tas'] = ds['tas'] - 273.15

    # Solar potential, PVGIS relative-efficiency + Faiman module-temperature model.
    solar_potential = compute_solar_cf(ds['tas'], ds['rsds'], ds['sfcWind'], cfg=pv_cfg)

    solar_xr = solar_potential.to_dataset(name='scf').convert_calendar('noleap')
    solar_xr = solar_xr.chunk({'time': 100, 'lat': -1, 'lon': -1})
    solar_xr['scf'] = solar_xr['scf'].astype('f4')
    solar_xr.attrs.update({
        'DESCRIPTION': f"{GCM} solar potential",
        'units': 'dimensionless', 'long_name': 'PVtot potential',
        'SOURCE': 'calculate_ds_cf_GCM_dask.py', 'AUTHOR': 'Colin Lenoble', 'corrected': 1,
        'MODEL': 'PVGIS relative efficiency + Faiman module temperature (calculate_wind_solar_cf.py)',
    })
    safe_to_zarr(solar_xr, scf_path)

    # Wind potential. cfg.wind_method selects how sfcWind is turned into
    # hub-height wind (see get_hub_height_wind): a per-pixel shear exponent
    # precomputed on GCM's own native grid (default, 'shear_local' -- see
    # get_gcm_shear_exponent) or a single global exponent ('shear_uniform').
    # 'wind100' isn't available here -- the bias-corrected dadjusted_* file
    # only carries tas/rsds/sfcWind, no 100 m wind -- and raises accordingly.
    if cfg.wind_method == 'shear_local':
        alpha = get_gcm_shear_exponent(GCM, shear_by_gcm_dir, ds, shear_ref_period)
    else:
        alpha = None
    wind_hub = get_hub_height_wind(ds, cfg, alpha=alpha)
    wind_pot = compute_wind_potential_from_hub_wind(wind_hub, cfg)

    wind_xr = wind_pot.to_dataset(name='wcf')
    wind_xr = wind_xr.chunk({'time': 100, 'lat': -1, 'lon': -1})
    wind_xr['wcf'] = wind_xr['wcf'].astype('f4')
    wind_xr.attrs.update({
        'DESCRIPTION': f"{GCM} wind potential",
        'units': 'dimensionless', 'long_name': 'Wind potential',
        'SOURCE': 'calculate_ds_cf_GCM_dask.py', 'AUTHOR': 'Colin Lenoble', 'corrected': 1,
        'wind_method': cfg.wind_method,
        'uniform_shear_exponent': cfg.uniform_shear_exponent if cfg.wind_method == 'shear_uniform' else 'n/a',
    })
    safe_to_zarr(wind_xr, wcf_path)

    print('DS_CF saved')
    solar_xr.close()
    wind_xr.close()
    del solar_xr, wind_xr
    gc.collect()


def calculate_ds_cf_reanalysis(
    path_folder,
    path_preprocessed,
    era5_file_pattern=None,
    reanalysis='ERA5',
    shapefile_path=None,
    cfg: DS_CFConfig = DEFAULT_DS_CF_CONFIG,
    pv_cfg: PVGISCoefficients = DEFAULT_PVGIS_COEFFICIENTS,
    shear_ref_period=('1982-01-01', '2001-12-31'),
):
    """
    Compute wcf/scf on the **native reanalysis grid** (no GCM regridding).

    Loads the raw reanalysis files, optionally applies a shapefile mask,
    converts units, and writes the DS_CF datasets to disk.

    era5_file_pattern : glob pattern for the reanalysis 10 m/100 m wind files
                        used to fit the local shear exponent (see
                        get_local_shear_exponent). Only needed for
                        cfg.wind_method='shear_local' (the default), and
                        ignored once that fit is cached; 'shear_uniform' and
                        'wind100' don't need it.

    Outputs
    -------
    {path_preprocessed}/{reanalysis}/wcf_{reanalysis}.nc
    {path_preprocessed}/{reanalysis}/scf_{reanalysis}.nc
    """
    out_folder = os.path.join(path_preprocessed, reanalysis)
    os.makedirs(out_folder, exist_ok=True)
    # wind_method is tagged onto the wcf filename so the three methods don't
    # overwrite each other; 'shear_local' (the original default) keeps the
    # untagged name for backward compatibility with already-cached files and
    # downstream globs (e.g. make_grid_files.py's wcf_day_* pattern).
    wcf_suffix = '' if cfg.wind_method == 'shear_local' else f'_{cfg.wind_method}'
    path_wcf = os.path.join(out_folder, f"wcf_day_{reanalysis}_historical_reanalysis_19790101-20191231{wcf_suffix}.zarr")
    path_scf = os.path.join(out_folder, f"scf_day_{reanalysis}_historical_reanalysis_19790101-20191231.zarr")

    if os.path.exists(path_wcf) and os.path.exists(path_scf):
        print("Reanalysis DS_CF files already exist, skipping.")
        return

    # 1. Load reanalysis files
    files_ref, _ = _match_files(os.path.join(path_folder, reanalysis, f"*{reanalysis}*"))
    if not files_ref:
        raise FileNotFoundError(
            f"No reanalysis files found in {os.path.join(path_folder, reanalysis)}")
    print(f"Found {len(files_ref)} reanalysis files for {reanalysis}")

    dref = open_mfdataset_any(
        files_ref, combine='by_coords', chunks={}, parallel=True,
    )
    dref = _standardize_reanalysis_names(dref)
    dref = dref.sortby('lat').sortby('lon').sortby('time')

    for v in ('tas', 'rsds'):
        if v not in dref:
            raise KeyError(f"'{v}' not found in reanalysis dataset")

    if 'sfcWind' not in dref:
        if not {'u10', 'v10'}.issubset(dref.data_vars):
            raise KeyError("Need either 'sfcWind' or both 'u10' and 'v10' in reanalysis")
        print("Computing sfcWind from u10/v10")
        dref['sfcWind'] = np.hypot(dref['u10'], dref['v10'])

    keep_vars = ['tas', 'rsds', 'sfcWind']
    if cfg.wind_method == 'wind100':
        if not {'u100', 'v100'}.issubset(dref.data_vars):
            raise KeyError(
                "wind_method='wind100' requires 'u100'/'v100' in the reanalysis "
                f"files matched under {os.path.join(path_folder, reanalysis)}"
            )
        keep_vars += ['u100', 'v100']
    dref = dref[keep_vars]
    dref = dref.chunk({'time': -1, 'lat': 50, 'lon': 50})

    # 2. Optional shapefile mask
    if shapefile_path is not None:
        print("Applying shapefile mask:", shapefile_path)
        shapefile = gpd.read_file(shapefile_path)
        mask_template = dref.tas.isel(time=0).load()
        transform = rasterio.transform.from_bounds(
            float(mask_template.lon.min()), float(mask_template.lat.min()),
            float(mask_template.lon.max()), float(mask_template.lat.max()),
            len(mask_template.lon), len(mask_template.lat),
        )
        lons, lats = np.meshgrid(mask_template.lon, mask_template.lat)
        coords = np.array([lons.flatten(), lats.flatten()]).T
        mask = rasterize_shapefile(shapefile, coords, mask_template.shape, transform)
        mask = mask[::-1, :]
        dref = dref.where(mask == 1, np.nan)
        print("Fraction of grid kept after mask:",
              float((mask == 1).sum() / mask.size))

    # 3. Unit conversion (K °C)
    dref = dref.convert_calendar('noleap').convert_calendar('standard')
    dref['tas'] = dref['tas'] - 273.15

    # Persist once here: the solar and wind branches below both build a
    # separate graph on top of dref (open_mfdataset + mask + calendar-
    # convert), so without persisting, Dask would redo all of that work
    # (i.e. re-read every reanalysis file from disk) twice.
    dref = dref.persist()

    # 4. Solar potential (scf), PVGIS relative-efficiency + Faiman
    #    module-temperature model.
    if os.path.exists(path_scf):
        print("scf file already exists, skipping:", path_scf)
    else:
        print("Computing solar potential (scf)...")
        scf = compute_solar_cf(dref['tas'], dref['rsds'], dref['sfcWind'], cfg=pv_cfg)

        solar_potential = scf.to_dataset(name='scf').convert_calendar('noleap')
        solar_potential = solar_potential.chunk({'time': 100, 'lat': -1, 'lon': -1})
        solar_potential['scf'] = solar_potential['scf'].astype('f4')
        solar_potential.attrs.update({
            'DESCRIPTION': f'scf on native {reanalysis} grid',
            'units': 'dimensionless',
            'long_name': 'PVtot potential',
            'SOURCE': 'calculate_ds_cf_reanalysis',
            'AUTHOR': 'Colin Lenoble',
            'MODEL': 'PVGIS relative efficiency + Faiman module temperature (calculate_wind_solar_cf.py)',
        })
        solar_potential = solar_potential.compute()
        safe_to_zarr(solar_potential, path_scf)
        print("Written scf to", path_scf)
        solar_potential.close()

    # 5. Wind potential (wcf). cfg.wind_method selects how ref_height wind is
    #    turned into hub_height wind before the power curve (see
    #    get_hub_height_wind): a per-pixel local shear exponent fit from
    #    reanalysis 10 m/100 m wind (default, 'shear_local'), a single global
    #    Hellmann exponent ('shear_uniform'), or the reanalysis 100 m wind
    #    read directly, no extrapolation ('wind100').
    if os.path.exists(path_wcf):
        print("wcf file already exists, skipping:", path_wcf)
    else:
        print(f"Computing wind potential (wcf), wind_method={cfg.wind_method!r}...")
        if cfg.wind_method == 'shear_local':
            alpha_native = get_local_shear_exponent(era5_file_pattern, path_preprocessed, shear_ref_period)
            alpha = regrid_alpha_to_grid(alpha_native, dref)
        else:
            alpha = None
        wind_hub = get_hub_height_wind(dref, cfg, alpha=alpha)
        wind_pot = compute_wind_potential_from_hub_wind(wind_hub, cfg)

        wind_potential = wind_pot.to_dataset(name='wcf')
        wind_potential = wind_potential.chunk({'time': 100, 'lat': -1, 'lon': -1})
        wind_potential['wcf'] = wind_potential['wcf'].astype('f4')
        wind_potential.attrs.update({
            'DESCRIPTION': f'wcf on native {reanalysis} grid',
            'units': 'dimensionless',
            'long_name': 'Wind potential',
            'SOURCE': 'calculate_ds_cf_reanalysis',
            'AUTHOR': 'Colin Lenoble',
            'wind_method': cfg.wind_method,
            'uniform_shear_exponent': cfg.uniform_shear_exponent if cfg.wind_method == 'shear_uniform' else 'n/a',
        })
        wind_potential = wind_potential.compute()
        safe_to_zarr(wind_potential, path_wcf)
        print("Written wcf to", path_wcf)
        wind_potential.close()

    dref.close()
    gc.collect()
    print("Reanalysis DS_CF files saved:", path_scf, path_wcf)


# -------------------------
# Population-weighted temperature aggregation
# -------------------------
def align_pop_to_GCM_sum(pop_path, GCM, run, ssp, gwl, reanalysis, path_preprocessed, temp_folder):
    """Reproject population raster onto the GCM grid (sum resampling)."""
    da_path = get_output_filename(path_preprocessed, GCM, ssp, run, gwl, reanalysis)
    tas = xr.open_dataset(da_path)["tas"]

    pop = rxr.open_rasterio(pop_path, masked=True).squeeze("band", drop=True)

    lat = tas["lat"].values
    lon = tas["lon"].values
    lat_asc = lat if lat[0] < lat[-1] else lat[::-1]
    lon_asc = lon if lon[0] < lon[-1] else lon[::-1]

    dlat = float(np.median(np.diff(lat_asc)))
    dlon = float(np.median(np.diff(lon_asc)))
    dst_transform = from_bounds(
        lon_asc[0] - dlon / 2, lat_asc[0] - dlat / 2,
        lon_asc[-1] + dlon / 2, lat_asc[-1] + dlat / 2,
        len(lon_asc), len(lat_asc),
    )

    dst = np.zeros((len(lat_asc), len(lon_asc)), dtype=np.float64)
    reproject(
        source=np.asarray(pop.data, dtype=np.float64),
        destination=dst,
        src_transform=pop.rio.transform(),
        src_crs=pop.rio.crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.sum,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    if lat[0] > lat[-1]:
        dst = dst[::-1, :]
    if lon[0] > lon[-1]:
        dst = dst[:, ::-1]

    pop_on_GCM = xr.DataArray(
        dst,
        coords={"lat": tas["lat"], "lon": tas["lon"]},
        dims=("lat", "lon"),
        name="pop",
        attrs={"long_name": "Population aggregated on GCM grid", "aggregation": "sum"},
    )
    pop_on_GCM['lat'] = -pop_on_GCM['lat']
    pop_on_GCM.to_netcdf(f"{temp_folder}{GCM}/pop_on_{GCM}.nc")
    return pop_on_GCM


def aggregate_tas(GCM, run, ssp, gwl, path_preprocessed, temp_folder, shapefile_path,
                  reanalysis='W5E5', weight=True, suffix_shp=''):
    tas = xr.open_dataset(
        get_output_filename(path_preprocessed, GCM, ssp, run, gwl, reanalysis)
    )['tas']

    shapefile = gpd.read_file(shapefile_path)
    if weight:
        pop_gcm = xr.open_dataset(f"{temp_folder}{GCM}/pop_on_{GCM}.nc")['pop']
        weight_map = xa.pixel_overlaps(tas, shapefile, weights=pop_gcm)
    else:
        weight_map = xa.pixel_overlaps(tas, shapefile)

    agg_tas = xa.aggregate(tas.load(), weight_map).to_dataset()
    agg_tas.attrs.update({
        'units': 'dimensionless',
        'long_name': 'Temperature by country/region',
        'SOURCE': 'calculate_cf.py: aggregate_tas',
        'AUTHOR': 'Colin Lenoble',
    })
    tag = 'tas_pop_agg' if weight else 'tas_agg'
    agg_tas.to_netcdf(
        f"{path_preprocessed}{GCM}/{tag}_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}_{suffix_shp}.nc"
    )


# -------------------------
# Spatial aggregation
# -------------------------
def aggregate_ds_cf_ref_GCM(GCM, path_preprocessed, temp_folder, shapefile_path,
                            reanalysis='ERA5', suffix_shp='v1',
                            cfg: DS_CFConfig = DEFAULT_DS_CF_CONFIG):
    """
    Aggregate the GCM-grid reference wcf_ref/scf_ref (calculate_ds_cf_reanalysis_grid_GCM's
    output) by region -- independent of any per-GWL wcf_day/scf_day file, so
    it can run right after calculate_ds_cf_reanalysis_grid_GCM instead of
    being tied to the per-GWL aggregate_ds_cf loop.

    wcf_ref/scf_ref are already masked to shapefile_path *before* being
    regridded onto the GCM grid (see calculate_ds_cf_reanalysis_grid_GCM,
    step 3 runs before step 4) -- this function only aggregates that
    already-masked-and-regridded data, it does not mask or regrid again.

    suffix_shp controls the weighting scheme (see aggregate_ds_cf):
      'v1' : weighted by climate reference capacity factors
      'v2' : weighted by grid-cell area only
    """
    if suffix_shp not in ('v1', 'v2'):
        raise ValueError(
            f"Unknown suffix_shp {suffix_shp!r}. "
            "Expected 'v1' (capacity-factor weighted) or 'v2' (area weighted)."
        )

    scf_ref_agg_path = f"{path_preprocessed}{GCM}/scf_agg_ref_{GCM}_{reanalysis}_{suffix_shp}.nc"
    wcf_ref_agg_path = f"{path_preprocessed}{GCM}/wcf_agg_ref_{GCM}_{reanalysis}_{suffix_shp}.nc"
    if os.path.exists(scf_ref_agg_path) and os.path.exists(wcf_ref_agg_path):
        print("Reference aggregation files already exist, skipping:",
              scf_ref_agg_path, wcf_ref_agg_path)
        return

    wcf_suffix = '' if cfg.wind_method == 'shear_local' else f'_{cfg.wind_method}'
    wcf_ref_files, _ = _match_files(f"{path_preprocessed}{GCM}/wcf_ref_{GCM}_{reanalysis}{wcf_suffix}")
    scf_ref_files, _ = _match_files(f"{path_preprocessed}{GCM}/scf_ref_{GCM}_{reanalysis}")
    if not wcf_ref_files:
        raise FileNotFoundError(
            f"No wcf_ref file found for {GCM}/{reanalysis}. "
            "Run calculate_ds_cf_reanalysis_grid_GCM first."
        )
    if not scf_ref_files:
        raise FileNotFoundError(
            f"No scf_ref file found for {GCM}/{reanalysis}. "
            "Run calculate_ds_cf_reanalysis_grid_GCM first."
        )
    wcf_ref = open_dataset_any(wcf_ref_files[0]).sel(time=slice('1982-01-01', '2001-12-31'))
    scf_ref = open_dataset_any(scf_ref_files[0]).sel(time=slice('1982-01-01', '2001-12-31'))

    shapefile = gpd.read_file(shapefile_path)
    os.makedirs(f"{temp_folder}{GCM}/", exist_ok=True)

    # Weight map filenames match aggregate_ds_cf's -- whichever of the two
    # functions runs first builds the cache, the other reuses it.
    scf_wm_path = f"{temp_folder}{GCM}/{GCM}_weightmap_scf_{suffix_shp}"
    wcf_wm_path = f"{temp_folder}{GCM}/{GCM}_weightmap_wcf_{suffix_shp}"

    if suffix_shp == 'v1':
        if not os.path.exists(scf_wm_path):
            scf_weight_map = xa.pixel_overlaps(scf_ref, shapefile,
                                               weights=scf_ref.scf.mean(dim='time'))
            scf_weight_map.to_file(scf_wm_path)
        else:
            scf_weight_map = xa.read_wm(scf_wm_path)

        if not os.path.exists(wcf_wm_path):
            wcf_weight_map = xa.pixel_overlaps(wcf_ref, shapefile,
                                               weights=wcf_ref.wcf.mean(dim='time'))
            wcf_weight_map.to_file(wcf_wm_path)
        else:
            wcf_weight_map = xa.read_wm(wcf_wm_path)
    else:
        if not os.path.exists(scf_wm_path):
            scf_weight_map = xa.pixel_overlaps(scf_ref, shapefile)
            scf_weight_map.to_file(scf_wm_path)
        else:
            scf_weight_map = xa.read_wm(scf_wm_path)

        if not os.path.exists(wcf_wm_path):
            wcf_weight_map = xa.pixel_overlaps(wcf_ref, shapefile)
            wcf_weight_map.to_file(wcf_wm_path)
        else:
            wcf_weight_map = xa.read_wm(wcf_wm_path)

    weighting_desc = (
        'weighted by mean solar/wind capacity factor over reference period 1982-2001'
        if suffix_shp == 'v1' else
        'weighted by grid-cell area only (no capacity factor)'
    )

    if not os.path.exists(scf_ref_agg_path):
        agg_solar_ref = xa.aggregate(scf_ref.load(), scf_weight_map).to_dataset()
        agg_solar_ref.attrs.update({
            'units': 'dimensionless',
            'long_name': 'PVtot potential by country/region (reference)',
            'weighting': weighting_desc,
            'SOURCE': 'aggregate_ds_cf_ref_GCM',
            'AUTHOR': 'Colin Lenoble',
        })
        agg_solar_ref.to_netcdf(scf_ref_agg_path)
        print("Written", scf_ref_agg_path)

    if not os.path.exists(wcf_ref_agg_path):
        agg_wind_ref = xa.aggregate(wcf_ref.load(), wcf_weight_map).to_dataset()
        agg_wind_ref.attrs.update({
            'units': 'dimensionless',
            'long_name': 'Wind potential by country/region (reference)',
            'weighting': weighting_desc,
            'SOURCE': 'aggregate_ds_cf_ref_GCM',
            'AUTHOR': 'Colin Lenoble',
        })
        agg_wind_ref.to_netcdf(wcf_ref_agg_path)
        print("Written", wcf_ref_agg_path)


def aggregate_ds_cf(GCM, run, ssp, path_preprocessed, temp_folder, gwl, shapefile_path,
                  reanalysis='ERA5', suffix_shp='v1'):
    """
    Aggregate wind and solar potential by region.

    suffix_shp controls the weighting scheme:
      'v1' : weighted by climate reference capacity factors
             (mean scf/wcf over the reference period 1982-2001)
      'v2' : weighted by grid-cell area only (pure geographic weighting,
             xa.pixel_overlaps without explicit weights)

    Any other value raises a ValueError.
    """
    if suffix_shp not in ('v1', 'v2'):
        raise ValueError(
            f"Unknown suffix_shp {suffix_shp!r}. "
            "Expected 'v1' (capacity-factor weighted) or 'v2' (area weighted)."
        )

    wcf_path_base = f"{path_preprocessed}{GCM}/wcf_day_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}"
    scf_path_base = f"{path_preprocessed}{GCM}/scf_day_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}"
    wcf_files, _ = _match_files(wcf_path_base)
    scf_files, _ = _match_files(scf_path_base)
    if not wcf_files:
        raise FileNotFoundError(f"No wcf file found for pattern: {wcf_path_base}.[nc|zarr]")
    if not scf_files:
        raise FileNotFoundError(f"No scf file found for pattern: {scf_path_base}.[nc|zarr]")
    wcf = open_dataset_any(wcf_files[0])
    scf = open_dataset_any(scf_files[0])
    shapefile = gpd.read_file(shapefile_path)

    wcf_ref_files, _ = _match_files(f"{path_preprocessed}{GCM}/wcf_ref_{GCM}_{reanalysis}")
    scf_ref_files, _ = _match_files(f"{path_preprocessed}{GCM}/scf_ref_{GCM}_{reanalysis}")
    if not wcf_ref_files:
        raise FileNotFoundError(f"No wcf_ref file found for {GCM}/{reanalysis}")
    if not scf_ref_files:
        raise FileNotFoundError(f"No scf_ref file found for {GCM}/{reanalysis}")
    wcf_ref = open_dataset_any(wcf_ref_files[0]).sel(time=slice('1982-01-01', '2001-12-31'))
    scf_ref = open_dataset_any(scf_ref_files[0]).sel(time=slice('1982-01-01', '2001-12-31'))

    wcf_ref = wcf_ref.sel(lat=slice(wcf.lat.values[0], wcf.lat.values[-1]),
                          lon=slice(wcf.lon.values[0], wcf.lon.values[-1]))
    scf_ref = scf_ref.sel(lat=slice(scf.lat.values[0], scf.lat.values[-1]),
                          lon=slice(scf.lon.values[0], scf.lon.values[-1]))

    os.makedirs(f"{temp_folder}{GCM}/", exist_ok=True)

    # ------------------------------------------------------------------
    # Build or load weight maps one pair per weighting scheme
    # Weight map filenames include the suffix so v1 and v2 never collide
    # ------------------------------------------------------------------
    scf_wm_path = f"{temp_folder}{GCM}/{GCM}_weightmap_scf_{suffix_shp}"
    wcf_wm_path = f"{temp_folder}{GCM}/{GCM}_weightmap_wcf_{suffix_shp}"

    if suffix_shp == 'v1':
        # v1: weight each pixel by its mean capacity factor over 1982-2001
        if not os.path.exists(scf_wm_path):
            scf_weight_map = xa.pixel_overlaps(scf_ref, shapefile,
                                               weights=scf_ref.scf.mean(dim='time'))
            scf_weight_map.to_file(scf_wm_path)
        else:
            scf_weight_map = xa.read_wm(scf_wm_path)

        if not os.path.exists(wcf_wm_path):
            wcf_weight_map = xa.pixel_overlaps(wcf_ref, shapefile,
                                               weights=wcf_ref.wcf.mean(dim='time'))
            wcf_weight_map.to_file(wcf_wm_path)
        else:
            wcf_weight_map = xa.read_wm(wcf_wm_path)

    elif suffix_shp == 'v2':
        # v2: weight pixels by grid-cell area only (no capacity factor)
        if not os.path.exists(scf_wm_path):
            scf_weight_map = xa.pixel_overlaps(scf_ref, shapefile)
            scf_weight_map.to_file(scf_wm_path)
        else:
            scf_weight_map = xa.read_wm(scf_wm_path)

        if not os.path.exists(wcf_wm_path):
            wcf_weight_map = xa.pixel_overlaps(wcf_ref, shapefile)
            wcf_weight_map.to_file(wcf_wm_path)
        else:
            wcf_weight_map = xa.read_wm(wcf_wm_path)

    # ------------------------------------------------------------------
    # Attrs description shared across solar and wind blocks
    # ------------------------------------------------------------------
    weighting_desc = (
        'weighted by mean solar/wind capacity factor over reference period 1982-2001'
        if suffix_shp == 'v1' else
        'weighted by grid-cell area only (no capacity factor)'
    )

    # ------------------------------------------------------------------
    # Solar aggregation
    # ------------------------------------------------------------------
    agg_solar = xa.aggregate(scf.load(), scf_weight_map).to_dataset()
    agg_solar.attrs.update({
        'units': 'dimensionless',
        'long_name': 'PVtot potential by country/region',
        'weighting': weighting_desc,
        'SOURCE': 'aggregate_ds_cf',
        'AUTHOR': 'Colin Lenoble',
    })
    agg_solar.to_netcdf(
        f"{path_preprocessed}{GCM}/scf_agg_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}_{suffix_shp}.nc")

    # ------------------------------------------------------------------
    # Wind aggregation
    # ------------------------------------------------------------------
    agg_wind = xa.aggregate(wcf.load(), wcf_weight_map).to_dataset()
    agg_wind.attrs.update({
        'units': 'dimensionless',
        'long_name': 'Wind potential by country/region',
        'weighting': weighting_desc,
        'SOURCE': 'aggregate_ds_cf',
        'AUTHOR': 'Colin Lenoble',
    })
    agg_wind.to_netcdf(
        f"{path_preprocessed}{GCM}/wcf_agg_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}_{suffix_shp}.nc")


def aggregate_ds_cf_reanalysis(
    path_preprocessed, temp_folder, shapefile_path,
    reanalysis='ERA5', suffix_shp='v1',
):
    """
    Aggregate reanalysis wcf/scf (native grid) by region.

    Reads the files produced by ``calculate_ds_cf_reanalysis`` and applies
    the same xagg-based weighting as ``aggregate_ds_cf``.

    suffix_shp controls the weighting scheme:
      'v1' : weighted by mean capacity factor over the full reanalysis period
      'v2' : weighted by grid-cell area only (pure geographic weighting)

    Outputs
    -------
    {path_preprocessed}/{reanalysis}/scf_agg_{reanalysis}_{suffix_shp}.nc
    {path_preprocessed}/{reanalysis}/wcf_agg_{reanalysis}_{suffix_shp}.nc
    """
    if suffix_shp not in ('v1', 'v2'):
        raise ValueError(
            f"Unknown suffix_shp {suffix_shp!r}. "
            "Expected 'v1' (capacity-factor weighted) or 'v2' (area weighted)."
        )

    wcf_path_base = os.path.join(path_preprocessed, reanalysis, f"wcf_day_{reanalysis}_historical_reanalysis_19790101-20191231")
    scf_path_base = os.path.join(path_preprocessed, reanalysis, f"scf_day_{reanalysis}_historical_reanalysis_19790101-20191231")

    wcf_files, _ = _match_files(wcf_path_base)
    scf_files, _ = _match_files(scf_path_base)
    if not wcf_files or not scf_files:
        raise FileNotFoundError(
            f"Reanalysis DS_CF files not found in "
            f"{os.path.join(path_preprocessed, reanalysis)}. "
            "Run calculate_ds_cf_reanalysis first."
        )

    wcf = open_dataset_any(wcf_files[0])
    scf = open_dataset_any(scf_files[0])
    shapefile = gpd.read_file(shapefile_path)

    wm_dir = os.path.join(temp_folder, reanalysis)
    os.makedirs(wm_dir, exist_ok=True)
    scf_wm_path = os.path.join(wm_dir, f"{reanalysis}_weightmap_scf_{suffix_shp}")
    wcf_wm_path = os.path.join(wm_dir, f"{reanalysis}_weightmap_wcf_{suffix_shp}")

    if suffix_shp == 'v1':
        if not os.path.exists(scf_wm_path):
            scf_weight_map = xa.pixel_overlaps(scf, shapefile,
                                               weights=scf.scf.mean(dim='time'))
            scf_weight_map.to_file(scf_wm_path)
        else:
            scf_weight_map = xa.read_wm(scf_wm_path)

        if not os.path.exists(wcf_wm_path):
            wcf_weight_map = xa.pixel_overlaps(wcf, shapefile,
                                               weights=wcf.wcf.mean(dim='time'))
            wcf_weight_map.to_file(wcf_wm_path)
        else:
            wcf_weight_map = xa.read_wm(wcf_wm_path)

    else:  # suffix_shp == 'v2'
        if not os.path.exists(scf_wm_path):
            scf_weight_map = xa.pixel_overlaps(scf, shapefile)
            scf_weight_map.to_file(scf_wm_path)
        else:
            scf_weight_map = xa.read_wm(scf_wm_path)

        if not os.path.exists(wcf_wm_path):
            wcf_weight_map = xa.pixel_overlaps(wcf, shapefile)
            wcf_weight_map.to_file(wcf_wm_path)
        else:
            wcf_weight_map = xa.read_wm(wcf_wm_path)

    weighting_desc = (
        'weighted by mean solar/wind capacity factor over the reanalysis period'
        if suffix_shp == 'v1' else
        'weighted by grid-cell area only (no capacity factor)'
    )

    # Solar aggregation
    agg_solar = xa.aggregate(scf.load(), scf_weight_map).to_dataset()
    agg_solar.attrs.update({
        'units': 'dimensionless',
        'long_name': 'PVtot potential by country/region',
        'weighting': weighting_desc,
        'SOURCE': 'aggregate_ds_cf_reanalysis',
        'AUTHOR': 'Colin Lenoble',
    })
    scf_out = os.path.join(path_preprocessed, reanalysis,
                           f"scf_agg_{reanalysis}_{suffix_shp}.nc")
    agg_solar.to_netcdf(scf_out)
    print("Written aggregated scf to", scf_out)

    # Wind aggregation
    agg_wind = xa.aggregate(wcf.load(), wcf_weight_map).to_dataset()
    agg_wind.attrs.update({
        'units': 'dimensionless',
        'long_name': 'Wind potential by country/region',
        'weighting': weighting_desc,
        'SOURCE': 'aggregate_ds_cf_reanalysis',
        'AUTHOR': 'Colin Lenoble',
    })
    wcf_out = os.path.join(path_preprocessed, reanalysis,
                           f"wcf_agg_{reanalysis}_{suffix_shp}.nc")
    agg_wind.to_netcdf(wcf_out)
    print("Written aggregated wcf to", wcf_out)


def build_available_df(path_preprocessed, ssp, reanalysis='ERA5',
                       gwl_list=('GWL0-61', 'GWL1', 'GWL1-5', 'GWL2', 'GWL3')):
    """
    Scan path_preprocessed and return a DataFrame of all GCM-run pairs
    with a boolean column per GWL indicating whether the wcf_day file exists.

    Detection is based on: wcf_day_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}.nc
    or the equivalent .zarr store.

    Parameters
    ----------
    path_preprocessed : str
    ssp               : str   e.g. 'ssp245'
    reanalysis        : str   e.g. 'ERA5'
    gwl_list          : sequence of GWL strings to check

    Returns
    -------
    pd.DataFrame with columns: GCM, run, ssp, <one bool col per GWL>, n_gwl_available
    """
    pattern_base = os.path.join(path_preprocessed, '*',
                                f"wcf_day_*_{ssp}_*_{reanalysis}")
    all_files = glob.glob(pattern_base + '.nc') + glob.glob(pattern_base + '.zarr')

    if not all_files:
        print(f"No wcf files found under {path_preprocessed}")
        return pd.DataFrame()

    records = {}
    for fpath in all_files:
        fname = os.path.basename(fpath.rstrip('/\\'))
        parts = fname.replace('.zarr', '').replace('.nc', '').split('_')
        # Filename format: wcf_day_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}.nc
        # Anchor on ssp and reanalysis to handle GCM names with underscores
        # e.g. EC-Earth3-Veg-LR -> parts between 'day' and ssp = GCM
        try:
            ssp_idx = parts.index(ssp)
            rea_idx = parts.index(reanalysis)
            gcm = '_'.join(parts[2:ssp_idx])
            run = parts[ssp_idx + 1]
            gwl = '_'.join(parts[ssp_idx + 2:rea_idx])
        except ValueError:
            print(f"Could not parse: {fname}, skipping.")
            continue

        key = (gcm, run)
        if key not in records:
            records[key] = {'GCM': gcm, 'run': run, 'ssp': ssp}
        records[key][gwl] = True

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(list(records.values()))

    # Ensure every expected GWL has a column (False if no file was found)
    for gwl in gwl_list:
        if gwl not in df.columns:
            df[gwl] = False
        else:
            df[gwl] = df[gwl].fillna(False)

    df['n_gwl_available'] = df[list(gwl_list)].sum(axis=1).astype(int)
    df = df[['GCM', 'run', 'ssp'] + list(gwl_list) + ['n_gwl_available']]
    df = df.sort_values(['GCM', 'run']).reset_index(drop=True)

    return df


if __name__ == "__main__":
    path_folder       = config.PATH_FOLDER
    path_preprocessed = config.PATH_PREPROCESSED
    shapefile_path    = config.SHAPEFILE_PATH
    temp_folder       = config.TEMP_FOLDER
    era5_file_pattern = config.ERA5_WIND_PATTERN  # used by calculate_ds_cf_reanalysis only
    shear_by_gcm_dir  = config.SHEAR_BY_GCM_DIR    # used by the *_GCM functions below
    pop_path          = config.POP_PATH            # used by align_pop_to_GCM_sum below

    ssp             = config.SSP
    gwl_list        = config.GWL_LIST
    reanalysis      = config.REANALYSIS
    shear_ref_period = config.SHEAR_REF_PERIOD

    # --- Inventory ---
    df_available = build_available_df(path_preprocessed, ssp, reanalysis, gwl_list)
    print(df_available.to_string())
    df_to_process = df_available.copy()
    

    path_list_base = f"{path_preprocessed}*/wcf_day_*_ssp245_*_GWL0-61_ERA5"
    path_list = glob.glob(path_list_base + '.nc') + glob.glob(path_list_base + '.zarr')
    GCM_list  = [os.path.basename(p.rstrip('/\\')).split('_')[-5] for p in path_list]
    run_list  = [os.path.basename(p.rstrip('/\\')).split('_')[-3] for p in path_list]

    

    # Load physical constants
    cfg = DEFAULT_DS_CF_CONFIG
    pv_cfg = DEFAULT_PVGIS_COEFFICIENTS

    GCM, run = 'MRI-ESM2-0', 'r1i1p1f1'

    
    calculate_ds_cf_reanalysis(
        path_folder,
        path_preprocessed,
        era5_file_pattern,
        reanalysis='ERA5',
        shapefile_path=shapefile_path,
        cfg=cfg,
        pv_cfg=pv_cfg,
        shear_ref_period=shear_ref_period
    )

  
    aggregate_ds_cf_reanalysis(path_preprocessed, temp_folder, shapefile_path,reanalysis='ERA5', suffix_shp='v1')
    aggregate_ds_cf_reanalysis(path_preprocessed, temp_folder, shapefile_path,reanalysis='ERA5', suffix_shp='v2')

    unbias_GCM(GCM, run, ssp, path_preprocessed, shapefile_path,
                path_folder, gwl_list, reanalysis)
    calculate_ds_cf_reanalysis_grid_GCM(GCM, run, ssp, path_preprocessed,
                                       path_folder, reanalysis,
                                       shapefile_path, cfg=cfg, pv_cfg=pv_cfg,
                                       shear_ref_period=shear_ref_period,
                                       shear_by_gcm_dir=shear_by_gcm_dir)
    aggregate_ds_cf_ref_GCM(GCM, path_preprocessed, temp_folder, shapefile_path,
                            reanalysis=reanalysis, suffix_shp='v1', cfg=cfg)
    # --- Loop over gwl_list for the GCM/run just unbiased above, gated on
    # dadjusted_* existence rather than df_to_process/row[gwl]. df_to_process
    # is a wcf_day_* inventory snapshot taken at the very start of __main__,
    # before unbias_GCM produced anything -- so a GCM/run processed for the
    # first time in this same script run is never in it (see the earlier
    # "Empty DataFrame" case), and gating on it would skip every GWL just
    # unbiased. dadjusted_* is unbias_GCM's own direct output, so it's
    # authoritative regardless of when this run started. ---
    print(f"\n--- Processing {GCM} {run} ---")
    for gwl in gwl_list:
        dadj_path = get_output_filename(path_preprocessed, GCM, ssp, run, gwl, reanalysis)
        if not os.path.exists(dadj_path):
            print(f"  Skipping {gwl} (dadjusted file not available)")
            continue
        print(f"  Processing {gwl}")
        calculate_ds_cf_GCM(GCM, run, ssp, path_preprocessed, gwl,
                           cfg=cfg, pv_cfg=pv_cfg, shear_ref_period=shear_ref_period,
                           shear_by_gcm_dir=shear_by_gcm_dir)
        aggregate_ds_cf(GCM, run, ssp, path_preprocessed, temp_folder,
                     gwl, shapefile_path, reanalysis, suffix_shp='v1')

        if not os.path.exists(f"{temp_folder}{GCM}/pop_on_{GCM}.nc"):
            print("  Aligning population to GCM grid...")
            align_pop_to_GCM_sum(pop_path, GCM, run, ssp, gwl, reanalysis,
                                 path_preprocessed, temp_folder)

        tas_pop_agg_path = (f"{path_preprocessed}{GCM}/tas_pop_agg_"
                           f"{GCM}_{ssp}_{run}_{gwl}_{reanalysis}_v1.nc")
        if not os.path.exists(tas_pop_agg_path):
            print("  Aggregating population-weighted temperature...")
            aggregate_tas(GCM, run, ssp, gwl, path_preprocessed, temp_folder,
                         shapefile_path, reanalysis=reanalysis, weight=True,
                         suffix_shp='v1')
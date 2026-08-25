# -*- coding: cp1252 -*-
"""
Standalone diagnostic for the ZeroDivisionError seen in unbias_GCM's MBCn
training (xclim's _escore dividing by zero).

Reconstructs dref/dhist through exactly the same masking/regridding/unit
steps as unbias_GCM (calculate_cf.py, up to just before the jitter /
additive-space transform), then prints, per variable and per location, how
many time steps are NaN. This is meant to answer one question concretely:
is the NaN problem "a little bit of NaN spread across every location"
(harmless) or "a subset of locations that are entirely NaN in one
variable" (fatal for the multivariate MBCn training)?

Run this on the server, in the same environment as calculate_cf.py.
"""
import numpy as np
import xarray as xr
import xesmf as xe
import geopandas as gpd
import rasterio

import config
from calculate_cf import (
    load_ds, filter_domain, set_variable_units, rasterize_shapefile,
    _standardize_reanalysis_names,
)
from io_utils import match_files as _match_files, open_mfdataset_any

GCM, run, ssp = 'CanESM5', 'r10i1p1f1', config.SSP
path_folder = config.PATH_FOLDER
shapefile_path = config.SHAPEFILE_PATH
reanalysis = config.REANALYSIS

print(f"Reconstructing dref/dhist for GCM={GCM}, run={run}, ssp={ssp}, reanalysis={reanalysis}")

# ------------------------------------------------------------------
# Same steps as unbias_GCM up to the point ref/hist locations are fixed
# (calculate_cf.py, unbias_GCM, roughly lines 312-398)
# ------------------------------------------------------------------
dhist = load_ds(GCM, ssp, run, path_folder, 'GWL0-61').dropna('time', how='all')

files_ref, _ = _match_files(__import__('os').path.join(path_folder, reanalysis, f"*{reanalysis}*"))
if not files_ref:
    raise FileNotFoundError(f"No reanalysis files found in {path_folder}{reanalysis}")
dref = open_mfdataset_any(files_ref)
dref = _standardize_reanalysis_names(dref)

if 'sfcWind' not in dref:
    print("Computing sfcWind from u10/v10")
    dref['sfcWind'] = np.hypot(dref['u10'], dref['v10'])

dref = dref.sortby('lat').sortby('lon').sortby('time')
dhist = dhist.sortby('lat').sortby('lon').sortby('time')
dref = dref.chunk({'time': -1, 'lat': 50, 'lon': 50})

lat_range = (dref.lat.values[0], dref.lat.values[-1])
lon_range = (dref.lon.values[0], dref.lon.values[-1])
dhist = filter_domain(dhist, lat_range, lon_range)
dhist = dhist.chunk({'time': -1, 'lat': 20, 'lon': 20})

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

regridder = xe.Regridder(dref, dhist, method='conservative_normed')
dref = regridder(dref, output_chunks={'lat': 50, 'lon': 50})
dref = dref.convert_calendar('noleap').convert_calendar('standard')
dhist = dhist.convert_calendar('noleap').convert_calendar('standard')

ref_grid = dref.tas.isel(time=0)


def create_mask_from_shapefile(grid, shapefile):
    transform = rasterio.transform.from_bounds(
        grid.lon.min().item(), grid.lat.min().item(),
        grid.lon.max().item(), grid.lat.max().item(),
        len(grid.lon), len(grid.lat)
    )
    from rasterio.features import geometry_mask
    shape = (len(grid.lat), len(grid.lon))
    mask = geometry_mask(
        geometries=shapefile.geometry, all_touched=True,
        out_shape=shape, transform=transform, invert=True
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

dref = dref.stack(location=("lat", "lon"))
dhist = dhist.stack(location=("lat", "lon"))

# ------------------------------------------------------------------
# Diagnostics: is NaN spread thin, or concentrated in dead locations?
# ------------------------------------------------------------------
n_time_ref = dref.sizes['time']
n_time_hist = dhist.sizes['time']
print(f"\ndref time steps={n_time_ref}, dhist time steps={n_time_hist}\n")

for name, ds in [('dref', dref), ('dhist', dhist)]:
    print(f"--- {name} ---")
    for v in ['tas', 'rsds', 'sfcWind']:
        nan_per_loc = ds[v].isnull().sum('time').compute()
        n_loc = nan_per_loc.sizes['location']
        n_time = n_time_ref if name == 'dref' else n_time_hist

        n_zero = int((nan_per_loc == 0).sum())
        n_full = int((nan_per_loc == n_time).sum())
        n_partial = n_loc - n_zero - n_full

        print(f"  {v}: {n_loc} locations total")
        print(f"    - fully valid (0 NaN steps):      {n_zero}")
        print(f"    - fully NaN (all {n_time} steps):  {n_full}"
              f"  <-- these poison the multivariate MBCn sample entirely")
        print(f"    - partially NaN (some but not all): {n_partial}")
        if n_partial > 0:
            partial_counts = nan_per_loc.values[
                (nan_per_loc.values > 0) & (nan_per_loc.values < n_time)
            ]
            print(f"      partial-NaN counts: min={partial_counts.min()}, "
                  f"median={int(np.median(partial_counts))}, "
                  f"max={partial_counts.max()} (out of {n_time})")

        # zero-variance among the fully-valid locations (what
        # remove_constant_locations's std==0 check actually catches)
        std = ds[v].std(dim='time').compute()
        n_const = int(((std == 0) & (nan_per_loc == 0)).sum())
        print(f"    - constant but not NaN (std==0):    {n_const}")
    print()

print(
    "Interpretation:\n"
    "  'fully NaN' locations are invisible to a std==0 check (NaN std != 0)\n"
    "  and survive dropna(how='all') if only ONE variable is dead there\n"
    "  while the others are fine. Those are the ones that blow up xclim's\n"
    "  _escore (n=0 samples -> division by zero). 'partially NaN' locations\n"
    "  with only a handful of NaN steps (e.g. from the ref/hist calendar-\n"
    "  label mismatch) are harmless by comparison."
)

# ------------------------------------------------------------------
# Map the fully-NaN locations (headless server -> save PNG, no display)
# ------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

out_dir = getattr(config, 'TEMP_FOLDER', os.getcwd())
os.makedirs(out_dir, exist_ok=True)

datasets = [('dref', dref, n_time_ref), ('dhist', dhist, n_time_hist)]
variables = ['tas', 'rsds', 'sfcWind']

fig, axes = plt.subplots(len(datasets), len(variables),
                          figsize=(5 * len(variables), 5 * len(datasets)),
                          squeeze=False)

for row, (name, ds, n_time) in enumerate(datasets):
    lat_vals = ds.location.lat.values
    lon_vals = ds.location.lon.values
    for col, v in enumerate(variables):
        ax = axes[row][col]
        nan_per_loc = ds[v].isnull().sum('time').compute().values
        is_full_nan = nan_per_loc == n_time
        is_valid = nan_per_loc == 0

        shapefile.boundary.plot(ax=ax, color='black', linewidth=0.5)
        ax.scatter(lon_vals[is_valid], lat_vals[is_valid],
                   s=6, c='lightgray', label=f'valid (n={int(is_valid.sum())})')
        ax.scatter(lon_vals[~is_valid & ~is_full_nan], lat_vals[~is_valid & ~is_full_nan],
                   s=6, c='orange', label=f'partial NaN (n={int((~is_valid & ~is_full_nan).sum())})')
        ax.scatter(lon_vals[is_full_nan], lat_vals[is_full_nan],
                   s=10, c='red', label=f'fully NaN (n={int(is_full_nan.sum())})')

        ax.set_title(f"{name}.{v}")
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        ax.legend(fontsize=7, loc='upper right', markerscale=2)

fig.tight_layout()
out_path = os.path.join(out_dir, 'nan_locations_diagnostic.png')
fig.savefig(out_path, dpi=150)
print(f"\nSaved NaN-location map to: {out_path}")

# Also dump the fully-NaN dref locations (the ones that matter) to CSV
for v in variables:
    nan_per_loc = dref[v].isnull().sum('time').compute().values
    is_full_nan = nan_per_loc == n_time_ref
    if is_full_nan.any():
        import pandas as pd
        df_out = pd.DataFrame({
            'lat': dref.location.lat.values[is_full_nan],
            'lon': dref.location.lon.values[is_full_nan],
        })
        csv_path = os.path.join(out_dir, f'fully_nan_locations_dref_{v}.csv')
        df_out.to_csv(csv_path, index=False)
        print(f"Saved {len(df_out)} fully-NaN dref.{v} locations to: {csv_path}")

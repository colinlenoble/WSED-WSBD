# -*- coding: utf-8 -*-
"""
Wind capacity factor: turbine power curve and the three ways of getting wind
speed to hub height. Depends only on numpy/xarray, so scripts can use it
without xesmf/xclim.
"""
from dataclasses import dataclass

import numpy as np
import xarray as xr

WIND_METHODS = ('shear_local', 'shear_uniform', 'wind100')


@dataclass
class DS_CFConfig:
    """
    Wind capacity-factor settings.

    vr, vci, vco : rated, cut-in and cut-out wind speeds (m/s)
    ref_height   : height of the input wind (m, 10 m)
    hub_height   : turbine hub height (m, 100 m)
    wind_method  : how hub-height wind is obtained (see get_hub_height_wind):
        'shear_local'   per-pixel Hellmann exponent fit from reanalysis
                        10 m/100 m wind (default)
        'shear_uniform' single exponent, uniform_shear_exponent (1/7, open
                        terrain)
        'wind100'       100 m wind (u100/v100) used directly
    """
    # Wind turbine curve
    vr:  float = 13.0
    vci: float = 3.5
    vco: float = 25.0
    ref_height: float = 10.0
    hub_height: float = 100.0

    # Wind extrapolation method
    wind_method: str = 'shear_local'
    uniform_shear_exponent: float = 1.0 / 7.0

    def __post_init__(self):
        if self.wind_method not in WIND_METHODS:
            raise ValueError(
                f"wind_method must be one of {WIND_METHODS}, got {self.wind_method!r}")


# Default config instance can be overridden at call sites
DEFAULT_DS_CF_CONFIG = DS_CFConfig()


def compute_wind_potential_from_hub_wind(wind_hub, cfg):
    """Power curve on hub-height wind: 0 below cut-in, cubic ramp up to rated,
    1 up to cut-out, 0 above."""
    wind_pot = xr.where(wind_hub < cfg.vci, 0, wind_hub)
    wind_pot = xr.where(wind_pot >= cfg.vco, 0, wind_pot)
    wind_pot = xr.where((wind_pot >= cfg.vr) & (wind_pot < cfg.vco), 1, wind_pot)
    wind_pot = xr.where(
        (wind_pot >= cfg.vci) & (wind_pot < cfg.vr),
        (wind_pot**3 - cfg.vci**3) / (cfg.vr**3 - cfg.vci**3),
        wind_pot
    )
    return wind_pot


def get_hub_height_wind(ds, cfg, alpha=None):
    """Wind speed at cfg.hub_height: hypot(u100, v100) for 'wind100', else
    sfcWind * (hub/ref)**alpha with the uniform exponent ('shear_uniform') or
    the per-pixel `alpha` given ('shear_local')."""
    if cfg.wind_method == 'wind100':
        if cfg.hub_height != 100.0:
            raise ValueError(
                "wind_method='wind100' reads the reanalysis 100 m wind directly, "
                f"so cfg.hub_height must be 100.0, got {cfg.hub_height}"
            )
        if not {'u100', 'v100'}.issubset(ds.data_vars):
            raise KeyError(
                "wind_method='wind100' requires 'u100'/'v100' in the dataset "
                "(e.g. don't drop them when loading the reanalysis archive)"
            )
        return np.hypot(ds['u100'], ds['v100'])
    elif cfg.wind_method == 'shear_uniform':
        return ds['sfcWind'] * (cfg.hub_height / cfg.ref_height) ** cfg.uniform_shear_exponent
    elif cfg.wind_method == 'shear_local':
        if alpha is None:
            raise ValueError(
                "wind_method='shear_local' requires a per-pixel alpha "
                "(see calculate_cf.get_local_shear_exponent / get_gcm_shear_exponent)"
            )
        return ds['sfcWind'] * (cfg.hub_height / cfg.ref_height) ** alpha
    else:
        raise ValueError(f"Unknown wind_method {cfg.wind_method!r}")


def compute_wind_potential(sfcwind, alpha, cfg):
    """Wind capacity factor with per-pixel shear exponent alpha (shorthand for
    get_hub_height_wind + compute_wind_potential_from_hub_wind)."""
    wind_hub = sfcwind * (cfg.hub_height / cfg.ref_height) ** alpha
    return compute_wind_potential_from_hub_wind(wind_hub, cfg)

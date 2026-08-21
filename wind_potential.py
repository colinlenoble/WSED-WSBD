# -*- coding: utf-8 -*-
"""
Wind capacity-factor physics: the power curve and the three ways of getting
wind speed up to hub height. Split out of calculate_cf.py (which imports
xesmf/xclim/xagg at module load) so scripts that only need the wcf formula
-- e.g. compare_wind_methods.py, or any local script run outside the HPC
env where xesmf/ESMF isn't installed -- can import it without dragging
those in. Same dependency-light rationale as io_utils.py.
"""
from dataclasses import dataclass

import numpy as np
import xarray as xr

WIND_METHODS = ('shear_local', 'shear_uniform', 'wind100')


@dataclass
class DS_CFConfig:
    """
    Physical constants for the wind potential calculation.

    Wind power curve parameters
    ---------------------------
    vr  : rated wind speed (m/s) turbine reaches full output above this
    vci : cut-in wind speed (m/s) turbine starts generating below this
    vco : cut-out wind speed (m/s) turbine shuts down above this
    ref_height : height (m) of the input wind speed (reanalysis/GCM 10 m wind)
    hub_height : height (m) the wind speed is extrapolated to (or read
                 directly from, for wind_method='wind100'). Defaults to
                 100 m.

    Wind extrapolation method
    --------------------------
    wind_method : how ref_height wind is turned into hub_height wind before
                  the power curve is applied (see get_hub_height_wind).
        'shear_local'   (default) -- per-pixel Hellmann shear exponent fit
                         from reanalysis 100 m/10 m wind (see fit_local_shear
                         / get_local_shear_exponent / get_gcm_shear_exponent
                         in calculate_cf.py). Backward-compatible default --
                         unchanged behavior.
        'shear_uniform' -- single global Hellmann exponent
                         (uniform_shear_exponent) applied everywhere instead
                         of a per-pixel fit.
        'wind100'       -- skip extrapolation entirely and use the
                         reanalysis/GCM 100 m wind (u100/v100) directly.
                         Requires hub_height == 100 and u100/v100 present in
                         the input dataset.
    uniform_shear_exponent : Hellmann exponent used when
                 wind_method='shear_uniform'. Defaults to 1/7 (~0.143), the
                 standard atmospheric-boundary-layer power-law value over
                 open terrain.
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
    """
    Cubic power curve applied to a wind speed already at cfg.hub_height
    (0 below cut-in, cubic ramp between cut-in and rated, 1 between rated
    and cut-out, 0 above cut-out). Shared by every wind_method in
    get_hub_height_wind -- only how wind_hub is obtained differs.
    """
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
    """
    Wind speed at cfg.hub_height, dispatching on cfg.wind_method:

      'wind100'       : ws100 = hypot(u100, v100) read directly from ds --
                        no extrapolation, so no shear-exponent error, but
                        only available where the archive actually has 100 m
                        wind components. Requires cfg.hub_height == 100.
      'shear_uniform' : ds['sfcWind'] extrapolated from cfg.ref_height to
                        cfg.hub_height with a single global Hellmann
                        exponent, cfg.uniform_shear_exponent (default 1/7).
      'shear_local'   : ds['sfcWind'] extrapolated with a per-pixel alpha
                        (must be supplied by the caller -- see
                        calculate_cf.get_local_shear_exponent /
                        get_gcm_shear_exponent).

    Feed the result to compute_wind_potential_from_hub_wind to get the
    capacity factor.
    """
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
    """
    Wind capacity factor: extrapolate sfcwind from cfg.ref_height to
    cfg.hub_height using the per-pixel Hellmann exponent alpha, then apply
    the cubic power curve between cut-in and rated speed.

    Kept as a thin shear-only wrapper for existing call sites; new code
    computing wcf under any of the three wind_method options should call
    get_hub_height_wind(ds, cfg, alpha) + compute_wind_potential_from_hub_wind
    instead (see calculate_cf.calculate_ds_cf_reanalysis).
    """
    wind_hub = sfcwind * (cfg.hub_height / cfg.ref_height) ** alpha
    return compute_wind_potential_from_hub_wind(wind_hub, cfg)

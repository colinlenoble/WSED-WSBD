# -*- coding: utf-8 -*-
"""
Solar capacity factor from the PVGIS PV performance model: a relative-
efficiency polynomial (k1-k7) of normalized irradiance and module
temperature, with the Faiman module-temperature model (u0, u1). Source:
https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/general-information/data-sources-calculation-methods_en

PVGIS gives P = (G/1000) * A * eff_nom * eff_rel(G', T'm); module area A and
nominal efficiency eff_nom are constants, so the capacity factor is
(G/1000) * eff_rel(G', T'm).
"""
from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass
class PVGISCoefficients:
    """
    PVGIS relative-efficiency and Faiman module-temperature coefficients.

        G'  = G / 1000                      (G: in-plane irradiance, W/m2)
        T'm = Tm - 25                       (Tm: module temperature, degC)
        Tm  = Ta + G / (u0 + u1 * W)         (Faiman module temperature model;
                                              Ta: air temp degC, W: wind m/s)

        eff_rel(G', T'm) = 1 + k1*ln(G') + k2*ln(G')**2 + k3*T'm
                              + k4*T'm*ln(G') + k5*T'm*ln(G')**2 + k6*T'm**2

    k7 (optional ln(G')**3 term, default 0) only allows 7-parameter sets;
    published PVGIS sets use k1-k6.

    The polynomial is fitted for G' ~ 0.03-1.2 and diverges as G' -> 0
    (spurious CFs at high latitude, where daily-mean irradiance is often tiny
    but nonzero), so irradiance below g_min_wm2 gives CF = 0, like wind
    below cut-in.
    """
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    k7: float = 0.0
    u0: float = 26.9
    u1: float = 6.2
    g_min_wm2: float = 10.0


# Relative-efficiency coefficient sets from the PVGIS "Data sources and
# calculation methods" page (crystalline silicon, CIGS and CdTe modules).
PVGIS_K_PRESETS = {
    "csi_current": dict(k1=-0.017237, k2=-0.040465, k3=-0.004702,
                        k4=0.000149, k5=0.000170, k6=0.000005),
    "csi_2025":    dict(k1=-0.006756, k2=-0.016444, k3=-0.003015,
                        k4=-0.000045, k5=-0.000043, k6=0.000000),
    "cigs":        dict(k1=-0.005554, k2=-0.038724, k3=-0.003723,
                        k4=-0.000905, k5=-0.001256, k6=0.000001),
    "cdte":        dict(k1=-0.046689, k2=-0.072844, k3=-0.002262,
                        k4=0.000276, k5=0.000159, k6=-0.000006),
}

# Faiman module-temperature coefficient sets (U0 [W/(degC.m2)], U1 [W.s/(degC.m3)]),
# by technology and mounting.
PVGIS_U_PRESETS = {
    "csi_free":  dict(u0=26.9,  u1=6.2),
    "csi_roof":  dict(u0=20.0,  u1=3.2),
    "cigs_free": dict(u0=22.64, u1=3.6),
    "cigs_roof": dict(u0=20.0,  u1=2.0),
    "cdte_free": dict(u0=23.37, u1=5.44),
    "cdte_roof": dict(u0=20.0,  u1=3.2),
}


def make_pvgis_coefficients(k_preset="csi_current", u_preset="csi_free", **overrides):
    """PVGISCoefficients from a k_preset (efficiency) and an independent u_preset
    (module temperature); any field can be overridden, e.g.
    make_pvgis_coefficients("cdte", "cdte_roof")."""
    if k_preset not in PVGIS_K_PRESETS:
        raise ValueError(f"Unknown k_preset {k_preset!r}. Options: {list(PVGIS_K_PRESETS)}")
    if u_preset not in PVGIS_U_PRESETS:
        raise ValueError(f"Unknown u_preset {u_preset!r}. Options: {list(PVGIS_U_PRESETS)}")
    params = {**PVGIS_K_PRESETS[k_preset], **PVGIS_U_PRESETS[u_preset]}
    params.update(overrides)
    return PVGISCoefficients(**params)


DEFAULT_PVGIS_COEFFICIENTS = make_pvgis_coefficients()


def compute_solar_cf(tas, rsds, sfcwind, cfg: PVGISCoefficients = DEFAULT_PVGIS_COEFFICIENTS,
                      valid_mask=None):
    """Solar capacity factor (~0-1) from tas (degC), rsds (W/m2) and sfcwind
    (m/s). Pixels where the optional valid_mask is False are set to NaN (see
    build_valid_mask)."""
    finite_inputs = np.isfinite(tas) & np.isfinite(rsds) & np.isfinite(sfcwind)
    above_floor = finite_inputs & (rsds > cfg.g_min_wm2)
    G = xr.where(above_floor, rsds, 0.0)
    G_prime = G / 1000.0

    Tm = tas + G / (cfg.u0 + cfg.u1 * sfcwind)
    T_prime = Tm - 25.0

    ln_G = xr.where(above_floor, np.log(xr.where(above_floor, G_prime, 1.0)), 0.0)

    eff_rel = (1
              + cfg.k1 * ln_G
              + cfg.k2 * ln_G**2
              + cfg.k3 * T_prime
              + cfg.k4 * T_prime * ln_G
              + cfg.k5 * T_prime * ln_G**2
              + cfg.k6 * T_prime**2
              + cfg.k7 * ln_G**3)

    cf = xr.where(above_floor, G_prime * eff_rel, 0.0)  # 0 below g_min_wm2
    # NaN inputs stay NaN rather than becoming 0.
    cf =xr.where(finite_inputs, cf, np.nan)
    if valid_mask is not None:
        cf = xr.where(valid_mask, cf, np.nan)
    return cf


def build_valid_mask(reference, buffer_cells=1):
    """True where `reference` (lat, lon) is finite, eroded by `buffer_cells` so
    neighbours of invalid cells (e.g. coastal ERA5-Land gaps) are excluded
    too."""
    valid = np.isfinite(reference)
    if buffer_cells > 0:
        window = 2 * buffer_cells + 1
        # Window minimum: False if any cell is invalid or at the domain edge.
        windowed = valid.rolling(lat=window, lon=window, center=True).construct(
            lat="lat_win", lon="lon_win")
        valid = windowed.min(dim=("lat_win", "lon_win")).fillna(0).astype(bool)
    return valid

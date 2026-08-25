# -*- coding: utf-8 -*-
"""
Solar capacity factor via the PVGIS PV performance model -- a
relative-efficiency polynomial (k1-k7) driven by normalized irradiance and
module temperature, plus the Faiman module-temperature model (u0, u1) --
instead of the NOCT/Huld cell-temperature model (c_1..c_4) used in
`2.1 calculate_epp_GCM_clean.py`. Source:
https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/general-information/data-sources-calculation-methods_en

The published PVGIS power equation is
    P = (G/1000) * A * eff_nom * eff_rel(G', T'm)
`A` (module area) and `eff_nom` (nominal STC efficiency) are module-scale
constants unrelated to weather variability, so they are dropped here -- the
capacity factor (P / P_stc) is just (G/1000) * eff_rel(G', T'm).

Trimmed down from the sibling como24_group5/code_review project's
calculate_wind_solar_cf.py to just the solar piece calculate_cf.py uses --
that file's wind (WindConfig/compute_wind_cf) and raw-ERA5-loading drivers
belong to a separate validation pipeline and aren't needed here.
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

    Only k1-k6 are used by the coefficient sets currently published by
    PVGIS. k7 is exposed (as an optional ln(G')**3 term, see
    compute_solar_cf) purely so a user-supplied 7-parameter coefficient set
    can be plugged in; it defaults to 0, which reduces to the standard
    PVGIS formula above.

    eff_rel is a polynomial fit over PVGIS's normal operating range of G'
    (roughly 0.03-1.2); it is not valid as G' -> 0. The ln(G')**2 / T'*ln(G')**2
    terms diverge there, and since k1/k2/k5 vary a lot by technology (e.g.
    cdte's |k1|,|k2| are ~3x csi's), that divergence shows up as large,
    technology-dependent spurious capacity factors -- worst at high
    latitude, where many days have a tiny but nonzero daily-mean irradiance
    (long dawn/dusk, near-polar-night) instead of a clean zero. g_min_wm2
    floors this the same way wind speeds below cut-in are floored to 0.
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
    g_min_wm2: float = 20.0


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
    """
    Build a PVGISCoefficients instance from a named k_preset (efficiency
    polynomial) and u_preset (Faiman module-temperature model). The two are
    independent -- mix and match technology/mounting freely -- and any field,
    including k7, can be overridden, e.g.:

        make_pvgis_coefficients("cdte", "cdte_roof")
        make_pvgis_coefficients("csi_current", "csi_free", k7=1.2e-4)
    """
    if k_preset not in PVGIS_K_PRESETS:
        raise ValueError(f"Unknown k_preset {k_preset!r}. Options: {list(PVGIS_K_PRESETS)}")
    if u_preset not in PVGIS_U_PRESETS:
        raise ValueError(f"Unknown u_preset {u_preset!r}. Options: {list(PVGIS_U_PRESETS)}")
    params = {**PVGIS_K_PRESETS[k_preset], **PVGIS_U_PRESETS[u_preset]}
    params.update(overrides)
    return PVGISCoefficients(**params)


DEFAULT_PVGIS_COEFFICIENTS = make_pvgis_coefficients()


def compute_solar_cf(tas, rsds, sfcwind, cfg: PVGISCoefficients = DEFAULT_PVGIS_COEFFICIENTS):
    """
    Solar (PV) capacity factor (dimensionless, ~0-1) using the PVGIS
    relative-efficiency + Faiman module-temperature model.

    tas     : air temperature (degC)
    rsds    : surface irradiance (W/m2)
    sfcwind : wind speed at module height (m/s)
    """
    above_floor = rsds > cfg.g_min_wm2
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

    # Below g_min_wm2, eff_rel is extrapolating the log-polynomial outside
    # its fitted range and can diverge (see PVGISCoefficients docstring) --
    # treat those steps as producing 0, same as a below-cut-in wind day.
    cf = xr.where(above_floor, G_prime * eff_rel, 0.0)
    return cf

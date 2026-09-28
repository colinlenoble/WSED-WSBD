# -*- coding: utf-8 -*-
"""
Synthetic version of fig3 (value-by-alpha map + regional violins), built
entirely from ad-hoc variables so it runs locally without the HPC data.

It calls fig3.plot_gwl_valuebyalpha_discrete() unchanged, so the layout,
colour/alpha bins and overlay logic are exactly the production ones. Only
the inputs are synthetic:

  - da_ref_* / da_proj_*  : per-realization frequency / intensity / duration
                            on a 1 deg global grid (smooth large-scale
                            patterns + GCM-specific noise), for GWL0-61,
                            GWL1-5, GWL2 and GWL3.
  - wcf_zero_mask         : GREY layer, i.e. land pixels excluded because the
                            wind capacity factor is 0 over the whole reference
                            period (ad-hoc blobs over the Amazon, Congo basin
                            and Maritime Continent).
  - agreement (hatchings) : DOT-HATCHED layer, i.e. land pixels excluded because
                            observed (ERA5) and projected (GCM) trends
                            disagree: agreement_pct <= AGREEMENT_THRESHOLD
                            (ad-hoc blobs over the Sahel, Central Asia,
                            interior Australia and Patagonia).
  - df_regions            : regional box means of the synthetic fields.

Land geometry comes from the Natural Earth countries cached by cartopy
(simplified and written once to test_data/shapefile/).

Usage (needs cartopy + geopandas + rasterio, e.g. the xesmf_env env):
    python fig3_synthetic.py [--gwl GWL2] [--output_dir test_data/output/main]
"""
import os
import sys
import types
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

# fig3 imports xesmf at module level, but only uses it when building the
# real datasets. Stub it so this script runs in envs without ESMF.
try:
    import xesmf  # noqa: F401
except Exception:
    sys.modules["xesmf"] = types.ModuleType("xesmf")

import fig3  # noqa: E402  (sets CARTOPY_DATA_DIR to the HPC path)
import config  # noqa: E402
import cartopy  # noqa: E402

# fig3 points cartopy at the HPC cache; fall back to the local one.
if not os.path.isdir(config.CARTOPY_DATA_DIR_XCLIM):
    cartopy.config["data_dir"] = os.path.join(
        os.path.expanduser("~"), ".local", "share", "cartopy")

import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SHP_OUT = os.path.join(HERE, "test_data", "shapefile", "ne_countries_simplified.shp")

GWL_LABELS = ["GWL0-61", "GWL1-5", "GWL2", "GWL3"]
GWL_WARMING = {"GWL0-61": 0.0, "GWL1-5": 1.0, "GWL2": 1.6, "GWL3": 2.6}   # scales the change
GCMS = {  # GCM -> runs (uneven, so the 1/n_run/n_gcm weights matter)
    "GCM-A": ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
    "GCM-B": ["r1i1p1f1"],
    "GCM-C": ["r1i1p1f1", "r2i1p1f1"],
    "GCM-D": ["r1i1p1f1"],
    "GCM-E": ["r1i1p1f1", "r2i1p1f1"],
    "GCM-F": ["r1i1p1f1"],
    "GCM-G": ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1", "r4i1p1f1"],
    "GCM-H": ["r1i1p1f1"],
}

# Ad-hoc exclusion zones: (lat_c, lon_c, lat_radius, lon_radius) ellipses.
# Grey: no wind capacity (wcf == 0 over the whole reference period).
WCF_ZERO_BLOBS = [
    (-4.0, -63.0, 7.0, 11.0),    # Amazon
    (0.0, 21.0, 5.0, 7.0),       # Congo basin
    (0.5, 113.5, 3.5, 4.0),      # Borneo
    (-1.0, 102.0, 3.0, 3.0),     # Sumatra
]
# Black: observation-projection trend discrepancy (agreement <= threshold).
DISCREPANCY_BLOBS = [
    (14.0, 5.0, 5.0, 16.0),      # Sahel
    (45.0, 68.0, 6.0, 14.0),     # Central Asia
    (-25.0, 130.0, 6.0, 9.0),    # interior Australia
    (-44.0, -69.0, 5.0, 3.0),    # Patagonia
    (58.0, -100.0, 5.0, 12.0),   # central Canada
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gwl", default="GWL2", choices=GWL_LABELS[1:],
                   help="Projected GWL shown on the map (default: GWL2).")
    p.add_argument("--output_dir", default=os.path.join(HERE, "test_data", "output", "main"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nan_only", action="store_true",
                   help="Do not pass wcf_zero_mask: the grey then comes only from the "
                        "NaN data inside the wcf == 0 zones.")
    return p.parse_args()


# =============================================================================
# Ad-hoc inputs
# =============================================================================

def make_shapefile(path=SHP_OUT):
    """Simplified Natural Earth countries (from cartopy's cache), written once."""
    if os.path.exists(path):
        return path
    from cartopy.io import shapereader
    src = shapereader.natural_earth("50m", "cultural", "admin_0_countries")
    shp = gpd.read_file(src)[["ADMIN", "geometry"]]
    shp = shp[shp["ADMIN"] != "Antarctica"]
    shp["geometry"] = shp.geometry.simplify(0.1, preserve_topology=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shp.to_file(path)
    return path


def make_grid():
    lat = np.arange(-89.5, 90.0, 1.0)     # ascending, as build_land_mask expects
    lon = np.arange(-179.5, 180.0, 1.0)
    return lat, lon


def blob_field(lat, lon, blobs):
    """Boolean (lat, lon) field that is True inside any of the ellipses."""
    LON, LAT = np.meshgrid(lon, lat)
    out = np.zeros(LAT.shape, dtype=bool)
    for lat_c, lon_c, r_lat, r_lon in blobs:
        out |= ((LAT - lat_c) / r_lat) ** 2 + ((LON - lon_c) / r_lon) ** 2 <= 1.0
    return out


def smooth_noise(rng, shape, scale=8):
    """Cheap large-scale noise: coarse random field, upsampled + box-smoothed."""
    from scipy.ndimage import zoom, uniform_filter
    coarse = rng.normal(size=(shape[0] // scale + 2, shape[1] // scale + 2))
    fine = zoom(coarse, scale, order=3)[: shape[0], : shape[1]]
    fine = uniform_filter(fine, size=scale // 2, mode="wrap")
    return fine / fine.std()


def make_wcf_zero_mask(lat, lon):
    da = xr.DataArray(blob_field(lat, lon, WCF_ZERO_BLOBS),
                      coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    return da.rename("wcf_zero")


def make_agreement(lat, lon, rng):
    """agreement_pct in [0, 100]: high everywhere, <= threshold in the blobs."""
    agree = 75 + 10 * smooth_noise(rng, (lat.size, lon.size))
    agree = np.clip(agree, 55, 100)
    agree[blob_field(lat, lon, DISCREPANCY_BLOBS)] = 30.0
    return xr.DataArray(agree, coords={"lat": lat, "lon": lon},
                        dims=("lat", "lon"), name="agreement_pct")


def make_datasets(lat, lon, rng):
    """
    One Dataset per GWL with frequency / intensity / duration, dims
    (realization, lat, lon), and GCM / run as per-realization variables --
    the same structure as fig3.build_gridded_datasets() returns.

    Baseline severity (freq * int * dur) lies in ~[0, 1] so it spans the
    fixed alpha bins; the projected relative change follows a smooth
    pattern (drying subtropics up, mid-latitudes down) scaled by warming.
    """
    no_wind = blob_field(lat, lon, WCF_ZERO_BLOBS)
    LON, LAT = np.meshgrid(lon, lat)
    shape = LAT.shape

    # Baseline climatology
    sev_base = 0.15 + 0.55 * np.exp(-((np.abs(LAT) - 25) / 18) ** 2) \
        + 0.12 * smooth_noise(rng, shape)
    sev_base = np.clip(sev_base, 0.02, 0.98)

    # Relative-change pattern per degree of warming (fraction)
    change_pat = (0.20 * np.exp(-((np.abs(LAT) - 22) / 12) ** 2)
                  - 0.16 * np.exp(-((np.abs(LAT) - 52) / 12) ** 2)
                  + 0.10 * smooth_noise(rng, shape, scale=12))

    realizations = [(g, r) for g, runs in GCMS.items() for r in runs]
    gcm_bias = {g: rng.normal(1.0, 0.08) for g in GCMS}

    datasets = {}
    for gwl in GWL_LABELS:
        dT = GWL_WARMING[gwl]
        freq, inten, dur = [], [], []
        for g, _ in realizations:
            noise = 1 + 0.05 * rng.normal(size=shape)
            sev = sev_base * gcm_bias[g] * (1 + dT * change_pat * gcm_bias[g]) * noise
            sev = np.clip(sev, 1e-3, None)
            sev[no_wind] = np.nan   # wcf == 0 -> no data, as in the real files
            # split severity into freq * int * dur (cube-root split + wobble)
            f = sev ** (1 / 3) * (1 + 0.05 * rng.normal(size=shape))
            d = sev ** (1 / 3) * (1 + 0.05 * rng.normal(size=shape))
            freq.append(f * 4.0)            # ~events / yr
            dur.append(d * 3.0)             # ~days / event
            inten.append(sev / (f * d) / 12.0)
        coords = {"realization": np.arange(len(realizations)), "lat": lat, "lon": lon}
        dims = ("realization", "lat", "lon")
        ds = xr.Dataset(
            {
                "frequency": (dims, np.stack(freq)),
                "intensity": (dims, np.stack(inten)),
                "duration": (dims, np.stack(dur)),
                "GCM": ("realization", [g for g, _ in realizations]),
                "run": ("realization", [r for _, r in realizations]),
            },
            coords=coords,
        )
        datasets[gwl] = ds
    return datasets


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    shp_path = make_shapefile()
    lat, lon = make_grid()
    built = make_datasets(lat, lon, rng)
    wcf_zero_mask = make_wcf_zero_mask(lat, lon)
    agreement = make_agreement(lat, lon, rng)

    ref_2d = built["GWL0-61"]["frequency"].isel(realization=0)
    mask = fig3.build_land_mask(ref_2d, shp_path)

    df_regions = fig3.add_severity_and_weights(
        fig3.create_dataframe_regional(built, mask))

    (da_ref_freq, da_ref_int, da_ref_dur,
     da_proj_freq, da_proj_int, da_proj_dur, weight) = fig3.from_ds_to_plot_decomp(
        built[args.gwl], built["GWL0-61"])

    gwl_txt = {"GWL1-5": "1.5 °C", "GWL2": "2 °C", "GWL3": "3 °C"}[args.gwl]
    fig = fig3.plot_gwl_valuebyalpha_discrete(
        da_ref_freq, da_ref_int, da_ref_dur,
        da_proj_freq, da_proj_int, da_proj_dur,
        weight, mask, shp_path, df_regions,
        gwl_label=gwl_txt,
        hatchings=agreement,
        agreement_threshold=config.AGREEMENT_THRESHOLD,
        map_title="SYNTHETIC - Projected change in annual severity under {gwl_label} warming",
        wcf_zero_mask=None if args.nan_only else wcf_zero_mask,
    )

    fig3.fit_to_width(fig)
    suffix = "_nan_only" if args.nan_only else ""
    out = os.path.join(args.output_dir, f"fig3_SYNTHETIC_{args.gwl}{suffix}.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

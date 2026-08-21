# -*- coding: cp1252 -*-
"""
Diagnostic script for the WSED/WSBD pipeline (build_ds_final in fig1.py).

Loads the raw wcf/scf zarr stores the same way fig1.py does and prints
sanity-check stats at every stage (raw data -> thresholds -> compound flag
-> severity -> duration/frequency -> masked compound index), to find where
huge/garbage values (e.g. ~1e12 in annual severity) are entering the
pipeline.

Run on the HPC (needs access to config.PATH_PREPROCESSED). By default it
restricts the expensive duration/frequency step to a single small region
(Western U.S., matching one of fig1's region boxes) so it runs fast; pass
--full to run duration_xr on the whole global grid instead.

Usage:
    python test.py
    python test.py --reanalysis ERA5 --full
    python test.py --lat 35 50 --lon -125 -105
"""
import argparse
import os

import config
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
os.environ['ESMFMKFILE'] = config.ESMFMKFILE_XENV

import numpy as np
import xarray as xr

from io_utils import match_files, open_dataset_any
from fig1 import (
    compute_severity, duration_xr, build_land_mask_from_grid, build_land_mask,
)


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose the WSED/WSBD pipeline for garbage values.")
    p.add_argument("--path_preprocessed", default=config.PATH_PREPROCESSED)
    p.add_argument("--reanalysis", default=config.REANALYSIS)
    p.add_argument("--shapefile", default=config.SHAPEFILE_PATH)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--ref_start", default=config.SHEAR_REF_PERIOD[0])
    p.add_argument("--ref_end", default=config.SHEAR_REF_PERIOD[1])
    p.add_argument("--lat", type=float, nargs=2, default=[35, 50],
                    help="lat_lo lat_hi for the region subset used in the fast path")
    p.add_argument("--lon", type=float, nargs=2, default=[-125, -105],
                    help="lon_lo lon_hi for the region subset used in the fast path")
    p.add_argument("--sane_bound", type=float, default=5.0,
                    help="abs value beyond which raw wcf/scf are flagged as suspicious")
    p.add_argument("--full", action="store_true", default=False,
                    help="run duration_xr on the whole grid instead of a region subset")
    return p.parse_args()


def describe(da, name, top_n=5):
    """Print min/max/mean/#nan/#inf and the top-N most extreme finite cells."""
    vals = da.compute() if hasattr(da.data, "chunks") else da
    v = vals.values
    finite = np.isfinite(v)
    n_total = v.size
    n_nan = int(np.isnan(v).sum())
    n_inf = int(np.isinf(v).sum())
    print(f"\n--- {name} ---")
    print(f"  shape={v.shape} dtype={v.dtype} total={n_total} nan={n_nan} inf={n_inf}")
    if not finite.any():
        print("  all values are NaN/inf -- nothing finite to summarize")
        return
    vf = v[finite]
    print(f"  min={vf.min():.6g} max={vf.max():.6g} mean={vf.mean():.6g} std={vf.std():.6g}")

    flat = vals.values
    order = np.argsort(-np.abs(np.where(np.isfinite(flat), flat, 0)).ravel())[:top_n]
    idx_tuples = np.unravel_index(order, flat.shape)
    print(f"  top {top_n} |value| cells:")
    for k in range(len(order)):
        coord_idx = tuple(int(idx_tuples[d][k]) for d in range(len(idx_tuples)))
        sel = {dim: coord_idx[d] for d, dim in enumerate(vals.dims)}
        cell = vals.isel(**sel)
        coord_str = ", ".join(f"{d}={float(cell[d].values):.3f}" for d in vals.dims if d in cell.coords)
        print(f"    value={float(cell.values):.6g}  ({coord_str})")


def main():
    args = parse_args()

    print(f"Loading wcf/scf for reanalysis={args.reanalysis}")
    wcf_files, wfmt = match_files(os.path.join(args.path_preprocessed, args.reanalysis, "wcf_day_*"))
    scf_files, sfmt = match_files(os.path.join(args.path_preprocessed, args.reanalysis, "scf_day_*"))
    if not wcf_files:
        raise FileNotFoundError(f"No wcf_day_* file found under {os.path.join(args.path_preprocessed, args.reanalysis)}")
    if not scf_files:
        raise FileNotFoundError(f"No scf_day_* file found under {os.path.join(args.path_preprocessed, args.reanalysis)}")
    print(f"  wcf: {wcf_files[0]} ({wfmt})")
    print(f"  scf: {scf_files[0]} ({sfmt})")

    chunks = {"time": 1000, "lat": -1, "lon": -1}
    wcf = open_dataset_any(wcf_files[0], chunks=chunks).sel(lat=slice(-58, 68))
    scf = open_dataset_any(scf_files[0], chunks=chunks).sel(lat=slice(-58, 68))
    wcf = wcf.convert_calendar("standard")
    scf = scf.convert_calendar("standard")

    print("\nEncoding (check for an un-decoded _FillValue/missing_value):")
    print(f"  wcf.wcf.encoding: {wcf.wcf.encoding}")
    print(f"  scf.scf.encoding: {scf.scf.encoding}")

    # --- 1. Raw data sanity ---------------------------------------------------
    describe(wcf.wcf, "raw wcf.wcf")
    describe(scf.scf, "raw scf.scf")

    n_wcf_extreme = int((np.abs(wcf.wcf) > args.sane_bound).sum().compute())
    n_scf_extreme = int((np.abs(scf.scf) > args.sane_bound).sum().compute())
    print(f"\nCells with |wcf.wcf| > {args.sane_bound}: {n_wcf_extreme}")
    print(f"Cells with |scf.scf| > {args.sane_bound}: {n_scf_extreme}")
    if n_wcf_extreme or n_scf_extreme:
        print("  -> raw data has out-of-range values; this is the likely source of the")
        print("     ~1e12 severity blow-up (unmasked fill/sentinel values surviving arithmetic).")

    # --- 2. Thresholds ----------------------------------------------------------
    print(f"\nComputing thresholds (quantile={args.threshold}, ref={args.ref_start}-{args.ref_end})")
    wcf_ref = wcf.sel(time=slice(args.ref_start, args.ref_end))
    scf_ref = scf.sel(time=slice(args.ref_start, args.ref_end))
    wcf_thr = wcf_ref.wcf.where(wcf_ref.wcf > 0).quantile(args.threshold, dim="time")
    scf_thr = scf_ref.scf.where(scf_ref.scf > 0).quantile(args.threshold, dim="time")
    describe(wcf_thr, "wcf_thr")
    describe(scf_thr, "scf_thr")

    # --- 3. Compound flag direction / frequency ---------------------------------
    print("\nDetecting compound events")
    wcf["low_wind"] = xr.where(wcf.wcf >= wcf_thr, 1, 0)
    scf["low_solar"] = xr.where(scf.scf >= scf_thr, 1, 0)
    frac_low_wind = float(wcf["low_wind"].mean().compute())
    frac_low_solar = float(scf["low_solar"].mean().compute())
    print(f"  fraction of (day,pixel) flagged low_wind:  {frac_low_wind:.3f}")
    print(f"  fraction of (day,pixel) flagged low_solar: {frac_low_solar:.3f}")
    if frac_low_wind > 0.5 or frac_low_solar > 0.5:
        print("  -> with threshold=quantile(0.1), a 'low' flag should fire on ~10% of days,")
        print("     not the majority. Check the comparison direction (`>=` vs `<=`) against")
        print("     wcf_thr/scf_thr in fig1.py's build_ds_final.")

    compound = (wcf.low_wind * scf.low_solar).to_dataset(name="start_cooc")
    frac_compound = float(compound.start_cooc.mean().compute())
    print(f"  fraction of (day,pixel) flagged compound (start_cooc==1): {frac_compound:.3f}")

    land_mask = build_land_mask_from_grid(compound.lat.values, compound.lon.values, args.shapefile)
    land_mask_da = xr.DataArray(land_mask, dims=("lat", "lon"),
                                coords={"lat": compound.lat, "lon": compound.lon})
    compound["start_cooc"] = compound["start_cooc"].where(land_mask_da)

    # --- 4. Severity --------------------------------------------------------
    print("\nComputing severity")
    severity_da = compute_severity(compound.start_cooc, scf, wcf, scf_thr, wcf_thr)
    severity_da["time"] = severity_da.time.dt.year
    severity_da = severity_da.rename({"time": "year"})
    describe(severity_da, "severity (annual)")

    # --- 5. Duration / frequency (region subset by default, --full for global) --
    if args.full:
        print("\nComputing duration and frequency (FULL GRID)")
        da_for_duration = compound.start_cooc
    else:
        lat_lo, lat_hi = args.lat
        lon_lo, lon_hi = args.lon
        print(f"\nComputing duration and frequency (region subset lat={args.lat} lon={args.lon}; pass --full for the whole grid)")
        da_for_duration = compound.start_cooc.sel(
            lat=slice(min(lat_lo, lat_hi), max(lat_lo, lat_hi)),
            lon=slice(min(lon_lo, lon_hi), max(lon_lo, lon_hi)),
        )
        severity_da = severity_da.sel(
            lat=slice(min(lat_lo, lat_hi), max(lat_lo, lat_hi)),
            lon=slice(min(lon_lo, lon_hi), max(lon_lo, lon_hi)),
        )

    ds_dur, ds_freq = duration_xr(da_for_duration)
    describe(ds_dur.duration, "duration")
    describe(ds_freq.frequency, "frequency")

    # --- 6. Assemble ds_final + land mask, exactly like build_ds_final ----------
    ds_final = ds_dur.copy()
    ds_final["frequency"] = ds_freq.frequency
    ds_final["severity"] = severity_da
    mask = build_land_mask(ds_final, args.shapefile)

    da_index = (ds_final.frequency.where(mask == 1)
                * ds_final.severity.where(mask == 1)
                * ds_final.duration.where(mask == 1))
    describe(da_index, "compound index (frequency * severity * duration), masked")

    ts = da_index.mean(("lat", "lon"), skipna=True)
    print("\nRegion-mean annual compound index (this is what the timeseries subplot draws):")
    print(ts.to_series())


if __name__ == "__main__":
    main()

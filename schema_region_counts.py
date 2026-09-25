# -*- coding: utf-8 -*-
"""
Per-region counts over the whole reference window, for every region of the
aggregated files used by schema_explanation_era5.py:
  - SWED days, "global" and "seasonal" threshold definitions
  - SWBD days
  - common days (SWED & SWBD on the same day), for each SWED definition

All thresholds and event definitions come from schema_explanation_era5.py
(compute_swed / compute_swbd), so the numbers match the sandbox notebook.
"""
import os

# netCDF4 must be imported before anything that pulls in rasterio/GDAL.
import netCDF4  # noqa: F401

import numpy as np
import pandas as pd
import xarray as xr

import schema_explanation_era5 as sch

OUT_CSV = os.path.join(sch.OUT_DIR, "swed_swbd_counts_by_region.csv")
SWED_METHODS = ("global", "seasonal")


def _fix_mojibake(s):
    """Region names are stored as UTF-8 bytes decoded as latin-1
    ("France mÃ©tropolitaine"); undo that, leave clean names untouched."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def main():
    ds_w = xr.open_dataset(os.path.join(sch.DATA_DIR, f"wcf{sch.FILE_SUFFIX}")).load()
    ds_s = xr.open_dataset(os.path.join(sch.DATA_DIR, f"scf{sch.FILE_SUFFIX}")).load()
    ds_t = xr.open_dataset(os.path.join(sch.DATA_DIR, f"tas_pop{sch.FILE_SUFFIX}")).load()
    share_idx = set(pd.read_csv(sch.SHARE_RENEWABLE_CSV)["poly_idx"])

    rows = []
    n = ds_w.sizes["poly_idx"]
    for i, poly_idx in enumerate(ds_w["poly_idx"].values):
        poly_idx = int(poly_idx)
        name = _fix_mojibake(str(ds_w["name"].values[i]))
        wcf = ds_w["wcf"].sel(poly_idx=poly_idx)
        scf = ds_s["scf"].sel(poly_idx=poly_idx)
        tas = ds_t["tas"].sel(poly_idx=poly_idx) - 273.15
        row = {"poly_idx": poly_idx, "name": name, "n_days": wcf.sizes["time"]}

        # All-NaN, or wcf and scf both 0 on every day (small islands the grid
        # misses): no usable resource data, so leave the counts empty, not 0.
        if (wcf.isnull().all() or scf.isnull().all()
                or (not (wcf > 0).any() and not (scf > 0).any())):
            print(f"[{i + 1}/{n}] {name}: no wcf/scf data, skipped")
            rows.append(row)
            continue

        swbd = None
        if poly_idx in share_idx and not tas.isnull().all():
            *_, swbd, _ = sch.compute_swbd(wcf, scf, tas, poly_idx)
            row["swbd_days"] = int(swbd.sum())

        for m in SWED_METHODS:
            _, _, swed = sch.compute_swed(wcf, scf, method=m)
            row[f"swed_days_{m}"] = int(swed.sum())
            if swbd is not None:
                row[f"common_days_{m}"] = int((swed & swbd).sum())

        rows.append(row)
        print(f"[{i + 1}/{n}] {name}: " +
              ", ".join(f"{k}={v}" for k, v in row.items() if k.endswith(("global", "seasonal", "swbd_days"))))

    cols = (["poly_idx", "name", "n_days", "swbd_days"] +
            [f"swed_days_{m}" for m in SWED_METHODS] +
            [f"common_days_{m}" for m in SWED_METHODS])
    df = pd.DataFrame(rows).reindex(columns=cols)
    for c in cols[3:]:
        df[c] = df[c].astype("Int64")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")  # BOM so Excel reads accents
    print(f"Saved {OUT_CSV} ({len(df)} regions)")


if __name__ == "__main__":
    main()

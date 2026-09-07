# -*- coding: utf-8 -*-
"""
Scan every scf_day_* file across all GCMs/runs/GWLs for the kind of
collapse-episode anomaly found in EC-Earth3-Veg-LR r2i1p1f1's GWL1 scf:
domain-mean solar capacity factor crashing to near-zero for a multi-month
stretch (2013-10-05 .. 2014-01-29), traced to that window's MBCn
bias-adjusted rsds collapsing to ~1/6 of its normal level while the raw GCM
rsds for the same days was completely normal -- i.e. a bias-adjustment bug,
not a GCM data problem (see diagnostics/check_ec_earth_scf_wcf.py, and the
conversation that led to this script).

No xesmf import needed -- this only reads already-computed scf_day_* files
via io_utils (xesmf-free), so it's safe to run standalone on the server
without the full xesmf_env pipeline.

For every scf file found under PREPROCESSED_PATH:
  1. Reduce to a domain-mean daily series (skipna).
  2. Flag "collapse days": domain-mean scf below FLOOR_FRAC of that run's
     own median. A global-domain solar CF should never be a tiny fraction
     of its own typical level -- there's always daylight somewhere -- so
     this is a physically motivated floor, not just a statistical outlier
     test. Consecutive/near-consecutive flagged days (within GAP_DAYS of
     each other) are grouped into episodes instead of listed individually.
  3. Flag years whose annual mean is a robust (median/MAD) outlier relative
     to that *same run's* other years -- catches partial collapses too
     mild to trip the absolute floor in (2) but still well outside that
     run's own normal year-to-year spread.

Prints one block per flagged (GCM, run, GWL), then a final summary list.
Tune FLOOR_FRAC/MIN_EPISODE_DAYS/YEAR_Z_THRESH below if the real data's
normal day-to-day spread makes this too strict or too lax -- these were
calibrated against one known-bad case, not the whole ensemble's spread.

Usage: edit PREPROCESSED_PATH/REANALYSIS/GWLS below if config.py's values
aren't what you want to scan, then: python scan_cf_collapse_episodes.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + r"/..")
import config
from io_utils import glob_any, open_dataset_any

VAR = "scf"
REANALYSIS = config.REANALYSIS
PREPROCESSED_PATH = config.PATH_PREPROCESSED
GWLS = config.GWL_LIST  # ['GWL0-61', 'GWL1', 'GWL1-5', 'GWL2', 'GWL3']

FLOOR_FRAC = 0.2       # a day below 20% of the run's own median scf is "collapsed"
MIN_EPISODE_DAYS = 2   # ignore single isolated low days (ordinary weather/cloud noise)
GAP_DAYS = 3           # allow up to this many non-flagged days inside one episode
YEAR_Z_THRESH = 4.0    # robust z-score threshold to flag a whole anomalous year


def robust_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / max(mad, 1e-12)


def find_episodes(dates, is_low):
    """Group a boolean is_low array into (start, end, n_days) episodes,
    merging flagged days separated by up to GAP_DAYS non-flagged days."""
    idx = np.where(is_low)[0]
    if idx.size == 0:
        return []
    episodes, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if (dates[i] - dates[prev]).days > GAP_DAYS:
            episodes.append((start, prev))
            start = i
        prev = i
    episodes.append((start, prev))
    return [(dates[s], dates[e], e - s + 1) for s, e in episodes if (e - s + 1) >= MIN_EPISODE_DAYS]


def check_file(path):
    da = open_dataset_any(path, chunks={})[VAR]
    spatial_dims = [d for d in da.dims if d != "time"]
    s = da.mean(dim=spatial_dims, skipna=True).compute().to_series()
    dates = s.index

    median = s.median()
    is_low = (s < FLOOR_FRAC * median).values
    episodes = find_episodes(dates, is_low)

    annual = s.groupby(s.index.year).mean()
    z = robust_z(annual.values)
    bad_years = [(yr, annual.loc[yr], z[i]) for i, yr in enumerate(annual.index) if abs(z[i]) > YEAR_Z_THRESH]

    return median, episodes, bad_years


if __name__ == "__main__":
    flagged = []
    for gwl in GWLS:
        pattern = os.path.join(PREPROCESSED_PATH, "*", f"{VAR}_day_*_{gwl}_{REANALYSIS}")
        files = sorted(glob_any(pattern))
        print(f"\n=== {gwl}: {len(files)} {VAR} file(s) found ===")
        for f in files:
            # {var}_day_{GCM}_{ssp}_{run}_{gwl}_{reanalysis}.{nc,zarr} -- GCM
            # names use hyphens not underscores, so counting from the end
            # (as trend_sev_eval.py does) is robust to that.
            stem = os.path.basename(f.rstrip("/\\")).rsplit(".", 1)[0]
            parts = stem.split("_")
            gcm, run = parts[-5], parts[-3]
            try:
                median, episodes, bad_years = check_file(f)
            except Exception as exc:
                print(f"  ERROR checking {os.path.basename(f)}: {exc}")
                continue
            if episodes or bad_years:
                print(f"\n<<<< FLAGGED: {gcm} {run} {gwl} (median {VAR}={median:.3f}) >>>>")
                for start, end, n in episodes:
                    print(f"    collapse episode: {start.date()} .. {end.date()} "
                          f"({n} days, floor={FLOOR_FRAC * median:.3f})")
                for yr, mean_val, zval in bad_years:
                    print(f"    anomalous year {yr}: mean={mean_val:.3f}  (robust z={zval:.1f})")
                flagged.append((gcm, run, gwl))

    print("\n" + "=" * 60)
    if flagged:
        print(f"{len(flagged)} (GCM, run, GWL) combination(s) flagged:")
        for gcm, run, gwl in flagged:
            print(f"  {gcm:20s} {run:12s} {gwl}")
    else:
        print("No collapse episodes or anomalous years found.")

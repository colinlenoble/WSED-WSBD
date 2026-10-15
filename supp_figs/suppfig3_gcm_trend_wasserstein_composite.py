# -*- coding: utf-8 -*-
"""
Extended Data Fig. 3: model skill in reproducing observed annual SWED severity
trends (empirical, normalized Wasserstein distance to ERA5's bootstrap trend
distribution).

Loads the empirical W2 grids and regional trend samples written by
trend_sev_eval_wasserstein.py, drops pixels with a near-zero ERA5
bootstrap-trend std and draws the normalized-W2 composite (helpers in
gcm_trend_agreement.py). Panel a maps the per-pixel ensemble median of
w2_normalized (weights 1/n_runs(GCM)).
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; puts the code folders on sys.path

import argparse

import numpy as np

import gcm_trend_agreement as cgta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--w2_path", default=cgta.IC_EMPIRICAL_W2_PATH,
                   help="Per-pixel empirical Wasserstein file (trend_sev_eval_wasserstein.py).")
    p.add_argument("--region_samples_path", default=cgta.REGION_TREND_SAMPLES_PATH,
                   help="Per-region bootstrap trend samples (trend_sev_eval_wasserstein.py).")
    p.add_argument("--shapefile", default=cgta.SHAPEFILE_PATH, help="Land-mask shapefile.")
    p.add_argument("--output_dir", default="../final_figs")
    return p.parse_args()


def drop_near_zero_reference_std(ds_w2, sigma_ref_min=cgta.SIGMA_REF_MIN):
    """Set w2_normalized to NaN where ERA5's bootstrap-trend std (w2_distance /
    w2_normalized) is below sigma_ref_min."""
    wd, wn_old = ds_w2.w2_distance, ds_w2.w2_normalized
    sigma_ref_recovered = wd / wn_old
    keep = (wd == 0) | (sigma_ref_recovered > sigma_ref_min)
    n_total = int(np.isfinite(wn_old.values).sum())
    n_dropped = int((np.isfinite(wn_old.values) & ~keep.values).sum())
    print(f"Dropping {n_dropped}/{n_total} pixel-realizations ({n_dropped / n_total:.1%}) with a "
          f"near-zero ERA5 bootstrap-trend std (< {sigma_ref_min:g}).")
    return ds_w2.assign(w2_normalized=wn_old.where(keep))


def main():
    args = parse_args()
    ds_w2, land_mask = cgta.load_empirical_grids(args.w2_path, args.shapefile)
    region_samples = cgta.load_region_trend_samples(args.region_samples_path)
    ds_w2 = drop_near_zero_reference_std(ds_w2)
    df_empirical = cgta.build_table_empirical(ds_w2, land_mask)

    out_path = os.path.join(args.output_dir, "supp",
                            "suppfig3_gcm_trend_wasserstein_composite_w2_normalized.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cgta.suppfig3_gcm_trend_wasserstein_composite_w2_normalized(
        df_empirical, ds_w2, region_samples, land_mask, out_path=out_path,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

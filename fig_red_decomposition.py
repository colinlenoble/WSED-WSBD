# -*- coding: cp1252 -*-
"""
Persistent compound wind-solar Renewable Energy Drought (RED): value-by-alpha
decomposition by event-duration class.

Produces two 2x2 figures (colour = relative change between period_comp and
period_hist, opacity = period_hist baseline severity), each decomposing the
persistent-drought severity index (frequency * mean duration * severity)
into:
  (a) all events combined (the unrestricted index)
  (b) events lasting exactly 1 day
  (c) events lasting exactly 2 days
  (d) events lasting 3 days or more

Figure 1 uses the standard weekly-rolling-mean-smoothed wcf/scf with the
1st-percentile low-week threshold (quantile=0.01). Figure 2 redoes the same
duration decomposition on raw (unsmoothed) daily wcf/scf -- no rolling
mean -- with the low-week threshold relaxed to the 10th percentile
(quantile=0.10, "q10"), matching the classic daily coincidence-below-
threshold definition instead of the persistent/rolling one.

Reuses the event-detection pipeline from fig_persistent.py.
"""
import os
import config
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
os.environ["ESMFMKFILE"] = config.ESMFMKFILE_XENV

import argparse

import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig_persistent import (
    build_duration_decomposition_persistent, build_land_mask,
    plot_valuebyalpha_decomposition_persistent,
)

PERIOD_HIST = (1982, 2001)
PERIOD_COMP = (2002, 2021)


# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Value-by-alpha decomposition of persistent compound WSE droughts "
                     "by event-duration class, at two low-week thresholds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--path_preprocessed", default=config.PATH_PREPROCESSED)
    parser.add_argument("--reanalysis", default=config.REANALYSIS)
    parser.add_argument("--threshold", type=float, default=0.01,
                         help="Low-week quantile threshold for figure 1 (default: 0.01).")
    parser.add_argument("--threshold_q10", type=float, default=0.10,
                         help="Low-week quantile threshold for figure 2 (default: 0.10).")
    parser.add_argument("--roll_window", type=int, default=7,
                         help="Rolling-mean window (days) applied to daily wcf/scf before "
                              "thresholding for figure 1 (default: 7, i.e. weekly). Figure 2 "
                              "always uses raw daily data (roll_window=1, i.e. no averaging).")
    parser.add_argument("--ref_start", default=config.SHEAR_REF_PERIOD[0])
    parser.add_argument("--ref_end", default=config.SHEAR_REF_PERIOD[1])
    parser.add_argument("--shapefile", default=config.SHAPEFILE_PATH)
    parser.add_argument("--output_dir",
                         default=os.path.join(config.SUMMARY_FIGS_DIR, "persistance"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def _run_decomposition(args, threshold, roll_window, out_suffix, suptitle):
    thr_str = str(threshold).replace(".", "")
    print(f"Computing duration-class decomposition (threshold={threshold}, "
          f"roll_window={roll_window})")
    indices, resource_valid = build_duration_decomposition_persistent(
        path_preprocessed=args.path_preprocessed,
        reanalysis=args.reanalysis,
        threshold=threshold,
        ref_start=args.ref_start,
        ref_end=args.ref_end,
        roll_window=roll_window,
    )

    print("Building land/resource mask")
    ds_for_mask = xr.Dataset({"resource_valid": resource_valid})
    mask = build_land_mask(ds_for_mask, args.shapefile)

    print(f"Plotting decomposition figure ({out_suffix})")
    fig = plot_valuebyalpha_decomposition_persistent(
        indices=indices, mask=mask, shapefile_path=args.shapefile,
        period_hist=PERIOD_HIST, period_comp=PERIOD_COMP,
        lat_min=-60, lat_max=75,
        suptitle=suptitle,
    )
    out_path = os.path.join(
        args.output_dir, "main",
        f"fig_red_decomposition_{out_suffix}_q{thr_str}_roll{roll_window}.png",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    _run_decomposition(
        args, threshold=args.threshold, roll_window=args.roll_window, out_suffix="hist",
        suptitle="Persistent compound WSE drought decomposition by event duration (ERA5)",
    )
    _run_decomposition(
        args, threshold=args.threshold_q10, roll_window=1, out_suffix="q10",
        suptitle="Compound WSE drought decomposition by event duration (ERA5)\n"
                 f"raw daily wcf/scf (no rolling mean), low-week threshold = "
                 f"{args.threshold_q10:.2f} quantile",
    )
    print("Done.")


if __name__ == "__main__":
    main()

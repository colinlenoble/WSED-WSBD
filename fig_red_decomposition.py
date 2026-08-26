# -*- coding: cp1252 -*-
"""
Compound wind-solar Renewable Energy Drought (RED): value-by-alpha
decomposition by event-duration class.

Produces a 2x2 figure (colour = relative change between period_comp and
period_hist, opacity = period_hist baseline severity) decomposing the
annual drought-severity index (frequency * mean duration * severity) into:
  (a) all events combined (the unrestricted index)
  (b) events lasting exactly 1 day
  (c) events lasting exactly 2 days
  (d) events lasting 3 days or more
-- to diagnose which event-duration class is driving the change in the
overall index.

Uses fig1.py's own event-detection pipeline (raw, un-smoothed daily wcf/scf;
default --threshold 0.01, the classic per-day coincidence-below-threshold
definition) -- i.e. this decomposes fig1.py's own headline severity index
(build_ds_final / plot_reanalysis_disagg_timeseries_valuebyalpha_discrete),
telling us whether fig1's reported change is mostly a single-day-coincidence
effect or is driven by longer (2+ day) spells.

Reuses the generic event-table / duration-class machinery
(duration_decomposition.py) and fig_persistent.py's value-by-alpha plotting
(plot_valuebyalpha_decomposition).
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

from fig_persistent import build_land_mask, plot_valuebyalpha_decomposition
from fig1 import build_duration_decomposition_daily

PERIOD_HIST = (1982, 2001)
PERIOD_COMP = (2002, 2021)


# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Value-by-alpha decomposition of compound WSE droughts by "
                     "event-duration class (fig1.py's raw daily wcf/scf pipeline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--path_preprocessed", default=config.PATH_PREPROCESSED)
    parser.add_argument("--reanalysis", default=config.REANALYSIS)
    parser.add_argument("--threshold", type=float, default=0.1,
                         help="Low-day quantile threshold (default: 0.1).")
    parser.add_argument("--ref_start", default=config.SHEAR_REF_PERIOD[0])
    parser.add_argument("--ref_end", default=config.SHEAR_REF_PERIOD[1])
    parser.add_argument("--shapefile", default=config.SHAPEFILE_PATH)
    parser.add_argument("--output_dir",
                         default=os.path.join(config.SUMMARY_FIGS_DIR, "persistance"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


# =============================================================================
# Classic daily-coincidence decomposition (fig1.py's own pipeline)
# =============================================================================

def run_daily_decomposition(args):
    print(f"Duration-class decomposition (threshold={args.threshold})")
    indices, resource_valid, freq_all = build_duration_decomposition_daily(
        path_preprocessed=args.path_preprocessed,
        reanalysis=args.reanalysis,
        threshold=args.threshold,
        ref_start=args.ref_start,
        ref_end=args.ref_end,
    )

    thr_str = str(args.threshold).replace(".", "")

    print("Building land/resource mask")
    ds_for_mask = xr.Dataset({"resource_valid": resource_valid})
    mask = build_land_mask(ds_for_mask, args.shapefile)
    # Match fig1.py's build_land_mask convention: also exclude pixels with
    # zero events in the first on-record year (fig1.py checks
    # duration.isnull() there; frequency/duration are 0-filled here rather
    # than NaN, so the equivalent check is == 0).
    no_event_year0 = (freq_all.isel(year=0) == 0).values
    mask = mask & ~no_event_year0

    print("Plotting decomposition figure")
    fig = plot_valuebyalpha_decomposition(
        indices=indices, mask=mask, shapefile_path=args.shapefile,
        period_hist=PERIOD_HIST, period_comp=PERIOD_COMP,
        lat_min=-60, lat_max=75,
        suptitle="Compound WSE drought decomposition by event duration (ERA5)\n"
                 f"raw daily wcf/scf (no rolling mean, fig1.py pipeline), "
                 f"low-day threshold = {args.threshold:.2f} quantile",
    )
    out_path = os.path.join(
        args.output_dir, "main",
        f"fig_red_decomposition_daily_q{thr_str}.png",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_daily_decomposition(args)
    print("Done.")


if __name__ == "__main__":
    main()

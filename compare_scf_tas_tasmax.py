# -*- coding: utf-8 -*-
"""
Sensitivity check: solar capacity factor (scf) computed with `tas` (daily
mean 2m air temperature) vs. with `tasmax` (daily maximum 2m air
temperature) feeding the Faiman module-temperature term in
compute_solar_cf.compute_solar_cf.

Runs directly on the raw CMIP6 files for one GCM/run/ssp (no bias
adjustment, no regridding -- native model grid), across every available GWL
window, and reports both per-pixel spatial error maps and domain-aggregated
scalar/time-series comparisons (bias, RMSE, %RMSE, MAE, Pearson r/R2),
each computed over "all time steps" and over "daytime only" (rsds above the
compute_solar_cf.g_min_wm2 floor) since every night both variants are
identically 0 and would dilute the daytime signal.

GWL windows can overlap in calendar time (e.g. GWL0-61 and GWL1-5 both
cover 2000-2002) because each is independently "the 20 years surrounding
the year this run reaches that warming level" -- so windows are processed
one at a time and combined via running sums (bias/RMSE/correlation
sufficient statistics), not via a single concatenated time axis.

Usage:
    python compare_scf_tas_tasmax.py --gcm CanESM5 --run r10i1p1f1 --ssp ssp245 \
        --raw_dir "E:/climate_data/climate_raw" --out_dir <output folder>
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_solar_cf import compute_solar_cf, DEFAULT_PVGIS_COEFFICIENTS

GWL_LIST = ["GWL0-61", "GWL1-5", "GWL2", "GWL3", "GWL4"]


def load_gwl_window(raw_dir, gcm, ssp, run, gwl):
    """Load tas/tasmax/rsds/sfcWind for one GWL window on the model's native grid."""
    das = {}
    for var in ("tas", "tasmax", "rsds"):
        pattern = os.path.join(raw_dir, gcm, f"{var}_day_{gcm}_{ssp}_{run}_*_{gwl}.nc")
        files = sorted(glob.glob(pattern))
        if not files:
            return None
        da = xr.open_dataset(files[0])[var]
        if "height" in da.coords:
            da = da.drop_vars("height")
        das[var] = da

    uas_files = sorted(glob.glob(os.path.join(raw_dir, gcm, f"uas_day_{gcm}_{ssp}_{run}_*_{gwl}.nc")))
    vas_files = sorted(glob.glob(os.path.join(raw_dir, gcm, f"vas_day_{gcm}_{ssp}_{run}_*_{gwl}.nc")))
    if not uas_files or not vas_files:
        return None
    uas = xr.open_dataset(uas_files[0])["uas"]
    vas = xr.open_dataset(vas_files[0])["vas"]
    if "height" in uas.coords:
        uas = uas.drop_vars("height")
    if "height" in vas.coords:
        vas = vas.drop_vars("height")
    das["sfcWind"] = np.hypot(uas, vas)

    ds = xr.Dataset(das).sortby("lat").sortby("lon").sortby("time")
    try:
        ds = ds.convert_calendar("standard")
    except Exception:
        ds = ds.convert_calendar("standard", align_on="year")
    ds["tas"] = ds["tas"] - 273.15
    ds["tasmax"] = ds["tasmax"] - 273.15
    return ds


def area_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return w / w.sum()


class RunningStats:
    """Accumulates the sufficient statistics needed for bias/RMSE/Pearson r
    across several GWL windows without concatenating mismatched time axes."""

    def __init__(self, shape):
        self.n = np.zeros(shape)
        self.sum_x = np.zeros(shape)
        self.sum_y = np.zeros(shape)
        self.sum_x2 = np.zeros(shape)
        self.sum_y2 = np.zeros(shape)
        self.sum_xy = np.zeros(shape)
        self.sum_err = np.zeros(shape)
        self.sum_err2 = np.zeros(shape)

    def update(self, x, y, valid):
        x = np.where(valid, x, 0.0)
        y = np.where(valid, y, 0.0)
        err = y - x
        v = valid.astype(float)
        self.n += v
        self.sum_x += x
        self.sum_y += y
        self.sum_x2 += x * x
        self.sum_y2 += y * y
        self.sum_xy += x * y
        self.sum_err += np.where(valid, err, 0.0)
        self.sum_err2 += np.where(valid, err * err, 0.0)

    def finalize(self):
        n = np.where(self.n > 0, self.n, np.nan)
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        bias = self.sum_err / n
        rmse = np.sqrt(self.sum_err2 / n)
        mae = None  # not accumulable exactly from sums; see scalar pooled pass
        cov = self.sum_xy / n - mean_x * mean_y
        var_x = self.sum_x2 / n - mean_x ** 2
        var_y = self.sum_y2 / n - mean_y ** 2
        r = cov / np.sqrt(var_x * var_y)
        pct_rmse = 100.0 * rmse / mean_x
        return dict(n=self.n, mean_x=mean_x, mean_y=mean_y, bias=bias,
                    rmse=rmse, pct_rmse=pct_rmse, r=r, r2=r ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gcm", default="CanESM5")
    ap.add_argument("--run", default="r10i1p1f1")
    ap.add_argument("--ssp", default="ssp245")
    ap.add_argument("--raw_dir", default="E:/climate_data/climate_raw")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stats_all = None
    stats_day = None
    lat = lon = None
    domain_series = []  # list of (time, dom_mean_tas, dom_mean_tasmax) per GWL, daily
    pooled = {"all": {"sum_abs": 0.0, "sum_sq": 0.0, "sum_err": 0.0, "n": 0},
              "day": {"sum_abs": 0.0, "sum_sq": 0.0, "sum_err": 0.0, "n": 0}}

    for gwl in GWL_LIST:
        print(f"Loading {args.gcm}/{args.run}/{args.ssp} {gwl} ...")
        ds = load_gwl_window(args.raw_dir, args.gcm, args.ssp, args.run, gwl)
        if ds is None:
            print(f"  skipping {gwl}: files not found")
            continue

        scf_tas = compute_solar_cf(ds["tas"], ds["rsds"], ds["sfcWind"], cfg=DEFAULT_PVGIS_COEFFICIENTS)
        scf_tasmax = compute_solar_cf(ds["tasmax"], ds["rsds"], ds["sfcWind"], cfg=DEFAULT_PVGIS_COEFFICIENTS)
        scf_tas = scf_tas.compute()
        scf_tasmax = scf_tasmax.compute()

        daytime = ds["rsds"].values > DEFAULT_PVGIS_COEFFICIENTS.g_min_wm2
        valid_all = np.isfinite(scf_tas.values) & np.isfinite(scf_tasmax.values)
        valid_day = valid_all & daytime

        if lat is None:
            lat = scf_tas["lat"].values
            lon = scf_tas["lon"].values
            stats_all = RunningStats(shape=(len(lat), len(lon)))
            stats_day = RunningStats(shape=(len(lat), len(lon)))

        x_all, y_all = scf_tas.values, scf_tasmax.values
        for t in range(x_all.shape[0]):
            stats_all.update(x_all[t], y_all[t], valid_all[t])
            stats_day.update(x_all[t], y_all[t], valid_day[t])

        err = y_all - x_all
        for key, mask in (("all", valid_all), ("day", valid_day)):
            e = err[mask]
            pooled[key]["sum_abs"] += np.sum(np.abs(e))
            pooled[key]["sum_sq"] += np.sum(e * e)
            pooled[key]["sum_err"] += np.sum(e)
            pooled[key]["n"] += e.size

        w = area_weights(lat)
        weights_2d = xr.DataArray(np.broadcast_to(w[:, None], (len(lat), len(lon))),
                                   dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
        dom_tas = (scf_tas * weights_2d).sum(dim=("lat", "lon")) / weights_2d.where(np.isfinite(scf_tas)).sum(dim=("lat", "lon"))
        dom_tasmax = (scf_tasmax * weights_2d).sum(dim=("lat", "lon")) / weights_2d.where(np.isfinite(scf_tasmax)).sum(dim=("lat", "lon"))
        domain_series.append(pd.DataFrame({
            "tas_scf": dom_tas.values,
            "tasmax_scf": dom_tasmax.values,
        }, index=pd.to_datetime(scf_tas["time"].values)))

        ds.close()
        print(f"  done ({scf_tas.sizes['time']} days).")

    if stats_all is None:
        raise SystemExit("No GWL windows found -- check --raw_dir/--gcm/--run/--ssp.")

    res_all = stats_all.finalize()
    res_day = stats_day.finalize()

    def pooled_metrics(key):
        p = pooled[key]
        n = p["n"]
        mae = p["sum_abs"] / n
        rmse = np.sqrt(p["sum_sq"] / n)
        bias = p["sum_err"] / n
        return dict(n=n, mae=mae, rmse=rmse, bias=bias)

    pm_all = pooled_metrics("all")
    pm_day = pooled_metrics("day")

    mean_scf_tas_all = np.nansum(res_all["mean_x"] * res_all["n"]) / np.nansum(res_all["n"])
    mean_scf_tas_day = np.nansum(res_day["mean_x"] * res_day["n"]) / np.nansum(res_day["n"])

    print("\n================ Pooled scalar comparison: scf(tasmax) vs scf(tas) ================")
    print(f"{'':22s} {'all timesteps':>18s} {'daytime only':>18s}")
    print(f"{'n samples':22s} {pm_all['n']:18,d} {pm_day['n']:18,d}")
    print(f"{'mean scf (tas)':22s} {mean_scf_tas_all:18.4f} {mean_scf_tas_day:18.4f}")
    print(f"{'bias':22s} {pm_all['bias']:18.5f} {pm_day['bias']:18.5f}")
    print(f"{'MAE':22s} {pm_all['mae']:18.5f} {pm_day['mae']:18.5f}")
    print(f"{'RMSE':22s} {pm_all['rmse']:18.5f} {pm_day['rmse']:18.5f}")
    print(f"{'%RMSE (vs mean scf)':22s} {100*pm_all['rmse']/mean_scf_tas_all:17.2f}% {100*pm_day['rmse']/mean_scf_tas_day:17.2f}%")

    summary = pd.DataFrame({
        "subset": ["all_timesteps", "daytime_only"],
        "n": [pm_all["n"], pm_day["n"]],
        "mean_scf_tas": [mean_scf_tas_all, mean_scf_tas_day],
        "bias": [pm_all["bias"], pm_day["bias"]],
        "MAE": [pm_all["mae"], pm_day["mae"]],
        "RMSE": [pm_all["rmse"], pm_day["rmse"]],
        "pct_RMSE": [100*pm_all["rmse"]/mean_scf_tas_all, 100*pm_day["rmse"]/mean_scf_tas_day],
    })
    summary_path = os.path.join(args.out_dir, "scf_tas_vs_tasmax_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\nWrote", summary_path)

    # ---------------- Spatial maps (daytime-only, the physically meaningful subset) ----------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        (res_day["bias"], "Bias: scf(tasmax) - scf(tas)\n(daytime mean)", "RdBu_r", None),
        (res_day["rmse"], "RMSE (daytime)", "viridis", None),
        (res_day["pct_rmse"], "%RMSE vs mean scf(tas) (daytime)", "viridis", None),
        (res_day["r"], "Pearson r (daytime)", "viridis", (0, 1)),
    ]
    for ax, (data, title, cmap, vrange) in zip(axes.flat, panels):
        vmax = np.nanmax(np.abs(data)) if vrange is None and cmap == "RdBu_r" else None
        vmin = -vmax if vmax is not None else (vrange[0] if vrange else None)
        vmax = vmax if vmax is not None else (vrange[1] if vrange else None)
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"{args.gcm} {args.run} {args.ssp}: scf sensitivity to tas vs tasmax "
                f"(native grid, all GWL windows pooled)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    map_path = os.path.join(args.out_dir, "scf_tas_vs_tasmax_spatial_maps.png")
    fig.savefig(map_path, dpi=150)
    plt.close(fig)
    print("Wrote", map_path)

    # ---------------- Domain-aggregated time series ----------------
    dom = pd.concat(domain_series).sort_index()
    dom = dom[~dom.index.duplicated(keep="first")]
    dom_annual = dom.resample("YE").mean()
    dom_annual["diff"] = dom_annual["tasmax_scf"] - dom_annual["tas_scf"]
    dom_annual["pct_diff"] = 100 * dom_annual["diff"] / dom_annual["tas_scf"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(dom_annual.index, dom_annual["tas_scf"], label="scf(tas)", lw=1.5)
    ax1.plot(dom_annual.index, dom_annual["tasmax_scf"], label="scf(tasmax)", lw=1.5, ls="--")
    ax1.set_ylabel("domain-mean scf")
    ax1.legend(fontsize=8)
    ax1.set_title(f"{args.gcm} {args.run} {args.ssp}: annual domain-mean scf, GWL windows pooled "
                  "(gaps = calendar years not covered by any GWL window)")

    ax2.plot(dom_annual.index, dom_annual["pct_diff"], color="firebrick", lw=1.5)
    ax2.axhline(0, color="grey", lw=0.5)
    ax2.set_ylabel("% diff (tasmax - tas) / tas")
    ax2.set_xlabel("year")
    fig.tight_layout()
    ts_path = os.path.join(args.out_dir, "scf_tas_vs_tasmax_domain_timeseries.png")
    fig.savefig(ts_path, dpi=150)
    plt.close(fig)
    print("Wrote", ts_path)

    dom_annual.to_csv(os.path.join(args.out_dir, "scf_tas_vs_tasmax_domain_annual.csv"))
    print("Wrote", os.path.join(args.out_dir, "scf_tas_vs_tasmax_domain_annual.csv"))


if __name__ == "__main__":
    main()

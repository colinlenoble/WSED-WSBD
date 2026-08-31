# -*- coding: utf-8 -*-
"""
Three-way comparison of solar capacity-factor methodologies used across this
project, run directly on the raw CMIP6 files for one GCM/run/ssp (no bias
adjustment, no regridding -- native model grid), across every available GWL
window:

  pvgis_tas    -- compute_solar_cf.compute_solar_cf: PVGIS relative-efficiency
                  polynomial + Faiman module-temperature model, driven by
                  `tas` (daily mean 2m air temperature).
  pvgis_tasmax -- same PVGIS/Faiman model, driven by `tasmax` (daily max)
                  instead of `tas` -- the sensitivity check from
                  compare_scf_tas_tasmax.py.
  noct_huld    -- the older NOCT-based cell-temperature model from
                  como24_group5/code_final/2.1 calculate_epp_GCM_clean.py
                  (EPPConfig c1..c4/gamma/T_ref/G_stc), driven by the
                  midpoint (tasmax+tas)/2:
                      T_cell = c1 + c2*((tasmax+tas)/2) + c3*rsds + c4*sfcWind
                      P_R    = 1 + gamma*(T_cell - T_ref)
                      spp    = P_R * (rsds / G_stc)
                  This is a linear temperature correction on a fixed
                  irradiance ratio, rather than PVGIS's log-irradiance
                  polynomial -- a structurally different model, not just a
                  different temperature input.

Reports pairwise bias/MAE/RMSE/%RMSE/Pearson r for all 3 pairs, each over
"all timesteps" and "daytime only" (rsds > compute_solar_cf.g_min_wm2),
since at night pvgis_* are floored to exactly 0 while noct_huld is not
floored (it goes to 0 too, but via rsds/G_stc -> 0 rather than an explicit
floor) -- pooling both regimes together would mostly just measure how often
it's night.

GWL windows can overlap in calendar time (e.g. GWL0-61 and GWL1-5 both cover
2000-2002) because each is independently "the 20 years surrounding the year
this run reaches that warming level" -- so windows are processed one at a
time and combined via running sums, not via a single concatenated time axis.

Usage:
    python compare_scf_methods.py --gcm CanESM5 --run r10i1p1f1 --ssp ssp245 \
        --raw_dir "E:/climate_data/climate_raw" --out_dir <output folder>
"""
import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_solar_cf import compute_solar_cf, DEFAULT_PVGIS_COEFFICIENTS

GWL_LIST = ["GWL0-61", "GWL1-5", "GWL2", "GWL3", "GWL4"]

# NOCT/Huld cell-temperature coefficients, copied from
# como24_group5/code_final/2.1 calculate_epp_GCM_clean.py's EPPConfig
# defaults (this script only reuses the 5 solar-relevant constants, not the
# wind-turbine curve also defined there).
NOCT_HULD = dict(gamma=-0.005, T_ref=25.0, G_stc=1000.0,
                 c_1=4.3, c_2=0.943, c_3=0.028, c_4=-1.528)

METHOD_LABELS = {
    "pvgis_tas": "PVGIS/Faiman (tas)",
    "pvgis_tasmax": "PVGIS/Faiman (tasmax)",
    "noct_huld": "NOCT/Huld ((tas+tasmax)/2)",
}
METHODS = list(METHOD_LABELS)


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


def compute_noct_huld(tas, tasmax, rsds, sfcwind, cfg=NOCT_HULD):
    """spp per como24_group5/code_final/2.1 calculate_epp_GCM_clean.py."""
    T_cell = (cfg["c_1"]
              + cfg["c_2"] * ((tasmax + tas) / 2)
              + cfg["c_3"] * rsds
              + cfg["c_4"] * sfcwind)
    P_R = 1 + cfg["gamma"] * (T_cell - cfg["T_ref"])
    return P_R * (rsds / cfg["G_stc"])


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
        self.n += valid.astype(float)
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
        cov = self.sum_xy / n - mean_x * mean_y
        var_x = self.sum_x2 / n - mean_x ** 2
        var_y = self.sum_y2 / n - mean_y ** 2
        r = cov / np.sqrt(var_x * var_y)
        pct_rmse = 100.0 * rmse / mean_x
        return dict(n=self.n, mean_x=mean_x, mean_y=mean_y, bias=bias,
                    rmse=rmse, pct_rmse=pct_rmse, r=r, r2=r ** 2)


class PooledScalar:
    """Same sufficient statistics as RunningStats, but as plain scalars
    (pooled over every pixel and timestep) -- used for the summary table."""

    def __init__(self):
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_err = 0.0

    def update(self, x, y, valid):
        xv, yv = x[valid], y[valid]
        err = yv - xv
        self.n += xv.size
        self.sum_x += xv.sum()
        self.sum_y += yv.sum()
        self.sum_x2 += np.sum(xv * xv)
        self.sum_y2 += np.sum(yv * yv)
        self.sum_xy += np.sum(xv * yv)
        self.sum_abs += np.sum(np.abs(err))
        self.sum_sq += np.sum(err * err)
        self.sum_err += err.sum()

    def finalize(self):
        n = self.n
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        bias = self.sum_err / n
        mae = self.sum_abs / n
        rmse = np.sqrt(self.sum_sq / n)
        cov = self.sum_xy / n - mean_x * mean_y
        var_x = self.sum_x2 / n - mean_x ** 2
        var_y = self.sum_y2 / n - mean_y ** 2
        r = cov / np.sqrt(var_x * var_y)
        pct_rmse = 100.0 * rmse / mean_x
        return dict(n=n, mean_x=mean_x, mean_y=mean_y, bias=bias, mae=mae,
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

    pairs = list(itertools.combinations(METHODS, 2))
    lat = lon = None
    spatial_stats = {("all", p): None for p in pairs}
    spatial_stats.update({("day", p): None for p in pairs})
    pooled = {("all", p): PooledScalar() for p in pairs}
    pooled.update({("day", p): PooledScalar() for p in pairs})
    domain_series = []

    for gwl in GWL_LIST:
        print(f"Loading {args.gcm}/{args.run}/{args.ssp} {gwl} ...")
        ds = load_gwl_window(args.raw_dir, args.gcm, args.ssp, args.run, gwl)
        if ds is None:
            print(f"  skipping {gwl}: files not found")
            continue

        cf = {
            "pvgis_tas": compute_solar_cf(ds["tas"], ds["rsds"], ds["sfcWind"], cfg=DEFAULT_PVGIS_COEFFICIENTS).compute(),
            "pvgis_tasmax": compute_solar_cf(ds["tasmax"], ds["rsds"], ds["sfcWind"], cfg=DEFAULT_PVGIS_COEFFICIENTS).compute(),
            "noct_huld": compute_noct_huld(ds["tas"], ds["tasmax"], ds["rsds"], ds["sfcWind"]).compute(),
        }

        daytime = ds["rsds"].values > DEFAULT_PVGIS_COEFFICIENTS.g_min_wm2
        finite_all3 = np.isfinite(cf["pvgis_tas"].values) & np.isfinite(cf["pvgis_tasmax"].values) \
            & np.isfinite(cf["noct_huld"].values)
        valid_all = finite_all3
        valid_day = finite_all3 & daytime

        if lat is None:
            lat = cf["pvgis_tas"]["lat"].values
            lon = cf["pvgis_tas"]["lon"].values
            for key in spatial_stats:
                spatial_stats[key] = RunningStats(shape=(len(lat), len(lon)))

        n_time = cf["pvgis_tas"].sizes["time"]
        for subset, mask in (("all", valid_all), ("day", valid_day)):
            for pair in pairs:
                x_all = cf[pair[0]].values
                y_all = cf[pair[1]].values
                stat = spatial_stats[(subset, pair)]
                pstat = pooled[(subset, pair)]
                for t in range(n_time):
                    stat.update(x_all[t], y_all[t], mask[t])
                pstat.update(x_all, y_all, mask)

        w = area_weights(lat)
        weights_2d = xr.DataArray(np.broadcast_to(w[:, None], (len(lat), len(lon))),
                                   dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
        dom = {}
        for name, da in cf.items():
            dom[name] = ((da * weights_2d).sum(dim=("lat", "lon"))
                        / weights_2d.where(np.isfinite(da)).sum(dim=("lat", "lon"))).values
        domain_series.append(pd.DataFrame(dom, index=pd.to_datetime(cf["pvgis_tas"]["time"].values)))

        ds.close()
        print(f"  done ({n_time} days).")

    if lat is None:
        raise SystemExit("No GWL windows found -- check --raw_dir/--gcm/--run/--ssp.")

    # ---------------- Pairwise pooled summary table ----------------
    rows = []
    for subset in ("all", "day"):
        for pair in pairs:
            m = pooled[(subset, pair)].finalize()
            rows.append({
                "subset": "all_timesteps" if subset == "all" else "daytime_only",
                "method_A": METHOD_LABELS[pair[0]],
                "method_B": METHOD_LABELS[pair[1]],
                "n": m["n"],
                "mean_A": m["mean_x"],
                "mean_B": m["mean_y"],
                "bias_B_minus_A": m["bias"],
                "MAE": m["mae"],
                "RMSE": m["rmse"],
                "pct_RMSE_vs_mean_A": m["pct_rmse"],
                "pearson_r": m["r"],
            })
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "scf_methods_pairwise_summary.csv")
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:,.5f}")
    print("\n================ Pairwise comparison: 3 solar-CF methodologies ================")
    print(summary.to_string(index=False))
    print("\nWrote", summary_path)

    # ---------------- Spatial maps (daytime only) ----------------
    fig, axes = plt.subplots(len(pairs), 3, figsize=(14, 4 * len(pairs)))
    for row, pair in enumerate(pairs):
        res = spatial_stats[("day", pair)].finalize()
        panels = [
            (res["bias"], f"Bias: {METHOD_LABELS[pair[1]]}\n- {METHOD_LABELS[pair[0]]}", "RdBu_r"),
            (res["rmse"], "RMSE (daytime)", "viridis"),
            (res["pct_rmse"], "%RMSE vs mean(A) (daytime)", "viridis"),
        ]
        for col, (data, title, cmap) in enumerate(panels):
            ax = axes[row, col]
            if cmap == "RdBu_r":
                vmax = np.nanmax(np.abs(data))
                vmin = -vmax
            else:
                vmin, vmax = None, None
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            ax.set_title(title, fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"{args.gcm} {args.run} {args.ssp}: pairwise solar-CF methodology differences "
                f"(native grid, all GWL windows pooled, daytime only)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    map_path = os.path.join(args.out_dir, "scf_methods_spatial_maps.png")
    fig.savefig(map_path, dpi=150)
    plt.close(fig)
    print("Wrote", map_path)

    # ---------------- Domain-aggregated annual time series ----------------
    dom = pd.concat(domain_series).sort_index()
    dom = dom[~dom.index.duplicated(keep="first")]
    dom_annual = dom.resample("YE").mean()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    styles = {"pvgis_tas": ("-", None), "pvgis_tasmax": ("--", None), "noct_huld": (":", None)}
    for name in METHODS:
        ls, _ = styles[name]
        ax.plot(dom_annual.index, dom_annual[name], ls, lw=1.5, label=METHOD_LABELS[name])
    ax.set_ylabel("domain-mean capacity factor")
    ax.set_xlabel("year")
    ax.legend(fontsize=8)
    ax.set_title(f"{args.gcm} {args.run} {args.ssp}: annual domain-mean solar CF, 3 methods "
                "(gaps = years not covered by any GWL window)", fontsize=9)
    fig.tight_layout()
    ts_path = os.path.join(args.out_dir, "scf_methods_domain_timeseries.png")
    fig.savefig(ts_path, dpi=150)
    plt.close(fig)
    print("Wrote", ts_path)

    dom_annual.to_csv(os.path.join(args.out_dir, "scf_methods_domain_annual.csv"))
    print("Wrote", os.path.join(args.out_dir, "scf_methods_domain_annual.csv"))


if __name__ == "__main__":
    main()

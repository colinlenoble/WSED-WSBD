# -*- coding: utf-8 -*-
"""
Residual-load (SWBD) CSVs for the country/region-aggregated pipeline
(previously STEP 1 of main_figs/fig45.py).

For every GCM/run with aggregated wcf/scf/tas_pop files (calculate_cf.py's
aggregate_ds_cf), computes the cumulative residual load above its reference
quantile at each GWL -- plus the supply-only and demand-only counterfactuals
used for the driver decomposition -- and writes one CSV per
(threshold, renewable penetration, mix[, demand configuration]):

    config.RL_OUT_DIR/rl_agg_adaptation_{period}_{thr}_ren_pen_{tot_re}_{mix}[_demand-{tag}]_v2.csv

Read by main_figs/fig45.py (Figs. 4-5) and supp_figs/suppfig6, 8, 9, 11, 12
and 13.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; also puts main_pipeline/, main_figs/, supp_figs/, aux_code/ on sys.path

import argparse
import glob
import gc
from dataclasses import dataclass
import pandas as pd
import xarray as xr
from xclim import sdba


# Main scenario: RL threshold quantile, renewable penetration, renewable mix.
MAIN_THR    = 0.99
MAIN_TOT_RE = 0.5
MAIN_MIX    = "current"

GWL_LIST    = ["GWL1-5", "GWL2", "GWL3"]


# =============================================================================
# DEMAND SENSITIVITY CONFIGURATIONS
# =============================================================================

@dataclass
class DemandConfig:
    thr_cold  : float = 12.5
    thr_hot   : float = 19.6
    coef_cold : float = 0.026
    coef_hot  : float = 0.035

    @property
    def tag(self):
        return (f"tc{self.thr_cold}_th{self.thr_hot}"
                f"_cc{self.coef_cold}_ch{self.coef_hot}")

    def __str__(self):
        return (f"cold>{self.thr_cold}C x{self.coef_cold} | "
                f"hot>{self.thr_hot}C x{self.coef_hot}")


DEFAULT_DEMAND = DemandConfig()


DEMAND_CONFIGS = {
    "default"  : DemandConfig(),
    "cold_low" : DemandConfig(thr_cold=10.5),
    "cold_high": DemandConfig(thr_cold=14.5),
    "hot_low"  : DemandConfig(thr_hot=17.6),
    "hot_high" : DemandConfig(thr_hot=21.6),
    "coef_low" : DemandConfig(coef_cold=0.021, coef_hot=0.028),
    "coef_high": DemandConfig(coef_cold=0.031, coef_hot=0.042),
    "strict"   : DemandConfig(thr_cold=10.5, thr_hot=21.6,
                               coef_cold=0.021, coef_hot=0.028),
    "sensitive": DemandConfig(thr_cold=14.5, thr_hot=17.6,
                               coef_cold=0.031, coef_hot=0.042),
}


# =============================================================================
# CORE RL COMPUTATION  (unchanged)
# =============================================================================

def _calculate_rl(tas, ds_cf, ds_cf_mean, thr, period, tot_re,
                  threshold=None, demand_bas=None,
                  demand_cfg: DemandConfig = None, return_daily=False):
    if demand_cfg is None:
        demand_cfg = DEFAULT_DEMAND
    demand_temp = xr.where(
        tas < demand_cfg.thr_cold,
        (demand_cfg.thr_cold - tas) * demand_cfg.coef_cold,
        xr.where(tas > demand_cfg.thr_hot,
                 (tas - demand_cfg.thr_hot) * demand_cfg.coef_hot, 0.0),
    )
    if demand_bas is None:
        demand_bas = ds_cf_mean / (1.0 + demand_temp.mean(dim="time"))
    demand = demand_bas * (1.0 + demand_temp)
    demand["time"] = pd.to_datetime(demand["time"].dt.strftime("%Y-%m-%d").values)
    ds_cf["time"]    = pd.to_datetime(ds_cf["time"].dt.strftime("%Y-%m-%d").values)
    months = {"Annual": range(1, 13), "JJA": [6, 7, 8], "DJF": [12, 1, 2]}[period]
    rl = (demand - tot_re * ds_cf).sel(
        time=(demand - tot_re * ds_cf)["time.month"].isin(months))
    if threshold is None:
        threshold = rl.quantile(thr, dim="time")
    exceeds = rl > threshold
    cum_rl = xr.where(exceeds, rl - threshold, 0.0).sum(dim="time")
    if return_daily:
        # Day-level SWBD exceedance field, kept only for callers that need
        # individual-event duration (e.g. suppfig6_duration_distribution_swed_swbd.py)
        # -- cum_rl above already discards this by summing over time, which
        # is all every other caller in this module has ever needed.
        return cum_rl, threshold, demand_bas, exceeds.astype(int)
    return cum_rl, threshold, demand_bas


def _align_histogram(source, target, kind="+", nquantiles=100):
    source.attrs["units"] = "K"
    target.attrs["units"] = "K"
    adj = sdba.EmpiricalQuantileMapping.train(
        ref=target, hist=source, kind=kind, nquantiles=nquantiles)
    return adj.adjust(source)


def _load_data(GCM, run, ssp, level, reanalysis, suffix_shp,
               path_preprocessed, df_share, mix="current"):
    base   = os.path.join(path_preprocessed, GCM)
    suffix = f"_{GCM}_{ssp}_{run}_{level}_{reanalysis}_{suffix_shp}.nc"
    tas = xr.open_dataset(os.path.join(base, f"tas_pop_agg{suffix}"))["tas"] - 273.15
    wcf = xr.open_dataset(os.path.join(base, f"wcf_agg{suffix}"))["wcf"]
    scf = xr.open_dataset(os.path.join(base, f"scf_agg{suffix}"))["scf"]
    cur_share = xr.DataArray(df_share["current_ratio"].values,
                             coords=[df_share["poly_idx"].values], dims=["poly_idx"])
    ds_cf_mean = (cur_share * scf + (1 - cur_share) * wcf).mean(dim="time")
    if mix == "future" and "future_ratio" in df_share.columns:
        fut_share = xr.DataArray(df_share["future_ratio"].values,
                                 coords=[df_share["poly_idx"].values], dims=["poly_idx"])
        ds_cf = fut_share * scf + (1 - fut_share) * wcf
    else:
        ds_cf = cur_share * scf + (1 - cur_share) * wcf
    return tas, ds_cf, ds_cf_mean


def compute_rl_one_gcm(GCM, run, ssp, gwl, thr, tot_re, mix,
                       path_preprocessed, df_share, reanalysis,
                       period="Annual", suffix_shp="v1",
                       demand_cfg: DemandConfig = None):
    if demand_cfg is None:
        demand_cfg = DEFAULT_DEMAND
    gwl_ref = "GWL0-61"
    dtas,     dds_cf,     _             = _load_data(GCM, run, ssp, gwl,     reanalysis,
                                                   suffix_shp, path_preprocessed, df_share, mix)
    dtas_ref, dds_cf_ref, dds_cf_ref_mean = _load_data(GCM, run, ssp, gwl_ref, reanalysis,
                                                   suffix_shp, path_preprocessed, df_share, mix)
    rows = []

    def _append(data, gwl_ds_cf_val, gwl_tas_val):
        rows.append(pd.DataFrame({
            "poly_idx": df_share["poly_idx"].values,
            "GCM": GCM, "run": run, "rl_cum": data.values,
            "gwl_ds_cf": gwl_ds_cf_val, "gwl_tas": gwl_tas_val,
            "period": period, "tot_re": tot_re, "share_re": mix,
            "demand_bas": demand_bas.values,
        }))

    cum_ref, rl_thr, demand_bas = _calculate_rl(
        dtas_ref, dds_cf_ref, dds_cf_ref_mean, thr, period, tot_re, demand_cfg=demand_cfg)
    _append(cum_ref, gwl_ref, gwl_ref)

    cum_gwl, _, _ = _calculate_rl(
        dtas, dds_cf, dds_cf_ref_mean, thr, period, tot_re,
        threshold=rl_thr, demand_bas=demand_bas, demand_cfg=demand_cfg)
    _append(cum_gwl, gwl, gwl)

    cum_tas, _, _ = _calculate_rl(
        _align_histogram(dtas_ref, dtas), dds_cf_ref, dds_cf_ref_mean,
        thr, period, tot_re,
        threshold=rl_thr, demand_bas=demand_bas, demand_cfg=demand_cfg)
    _append(cum_tas, gwl_ref, gwl)

    cum_ds_cf, _, _ = _calculate_rl(
        _align_histogram(dtas, dtas_ref), dds_cf, dds_cf_ref_mean,
        thr, period, tot_re,
        threshold=rl_thr, demand_bas=demand_bas, demand_cfg=demand_cfg)
    _append(cum_ds_cf, gwl, gwl_ref)

    return pd.concat(rows, ignore_index=True)


# =============================================================================
# PIPELINE HELPERS  (unchanged)
# =============================================================================

EXCLUDED_RUNS = {("EC-Earth3-Veg-LR", "r3i1p1f1"), ("NorESM2-MM", "r2i1p1f1")}


def _iter_gcm_runs(path_preprocessed, ssp, reanalysis, suffix_shp):
    pattern = os.path.join(path_preprocessed, "*",
                           f"wcf_agg_*GWL1-5*_{reanalysis}_{suffix_shp}.nc")
    for fpath in glob.glob(pattern):
        fname = os.path.basename(fpath)
        parts = fname.replace(".nc", "").split("_")
        try:
            idx = parts.index(ssp)
            gcm, run = "_".join(parts[2:idx]), parts[idx + 1]
            if (gcm, run) in EXCLUDED_RUNS:
                continue
            yield gcm, run
        except (ValueError, IndexError):
            print(f"  [WARN] Cannot parse: {fname}")


def _files_ready(GCM, run, ssp, gwl, reanalysis, suffix_shp, path_preprocessed):
    base = os.path.join(path_preprocessed, GCM)
    return all(os.path.exists(
        os.path.join(base, f"tas_pop_agg_{GCM}_{ssp}_{run}_{lv}_{reanalysis}_{suffix_shp}.nc"))
        for lv in (gwl, "GWL0-61"))


def _compute_and_save(thr, tot_re, mix, demand_cfg,
                      path_preprocessed, df_share, ssp, reanalysis, out_dir,
                      period, suffix_shp, demand_tag=None):
    if demand_tag is not None:
        fname = (f"rl_agg_adaptation_{period}_{thr}"
                 f"_ren_pen_{tot_re}_{mix}_demand-{demand_tag}_v2.csv")
    else:
        fname = f"rl_agg_adaptation_{period}_{thr}_ren_pen_{tot_re}_{mix}_v2.csv"
    out_path = os.path.join(out_dir, fname)
    required_cols = {"poly_idx", "GCM", "run", "rl_cum",
                     "gwl_ds_cf", "gwl_tas", "period", "tot_re", "share_re",
                     "demand_bas"}
    if os.path.exists(out_path):
        existing_cols = set(pd.read_csv(out_path, index_col=0, nrows=0).columns)
        missing = required_cols - existing_cols
        if not missing:
            print(f"  [SKIP] {fname}")
            return
        print(f"  [STALE] {fname} is missing columns {sorted(missing)} "
              "(schema changed since it was written) -- recomputing.")
    print(f"\n  Computing: {fname}")
    df_final = []
    for GCM, run in _iter_gcm_runs(path_preprocessed, ssp, reanalysis, suffix_shp):
        print(f"    {GCM}  {run}")
        try:
            for gwl in GWL_LIST:
                if not _files_ready(GCM, run, ssp, gwl, reanalysis, suffix_shp, path_preprocessed):
                    print(f"      Missing files for {gwl}, skipping.")
                    continue
                df_gcm = compute_rl_one_gcm(
                    GCM, run, ssp, gwl, thr, tot_re, mix,
                    path_preprocessed, df_share, reanalysis,
                    period=period, suffix_shp=suffix_shp, demand_cfg=demand_cfg)
                df_final.append(df_gcm)
        except Exception as exc:
            print(f"      ERROR {GCM} {run}: {exc}")
        gc.collect()
    if df_final:
        pd.concat(df_final, ignore_index=True).to_csv(out_path)
        print(f"  Saved: {out_path}")
    else:
        print(f"  [WARN] No data for {fname}")


def run_rl_pipeline(tot_re_list, thr_list, mix_list,
                    path_preprocessed, df_share, ssp, reanalysis, out_dir,
                    period="Annual", suffix_shp="v1"):
    for thr in thr_list:
        for tot_re in tot_re_list:
            for mix in mix_list:
                _compute_and_save(thr, tot_re, mix, demand_cfg=DEFAULT_DEMAND,
                                  path_preprocessed=path_preprocessed,
                                  df_share=df_share, ssp=ssp, reanalysis=reanalysis,
                                  out_dir=out_dir, period=period, suffix_shp=suffix_shp,
                                  demand_tag=None)


def run_demand_sensitivity_pipeline(path_preprocessed, df_share, ssp, reanalysis, out_dir,
                                    period="Annual", suffix_shp="v1"):
    for demand_name, demand_cfg in DEMAND_CONFIGS.items():
        _compute_and_save(thr=MAIN_THR, tot_re=MAIN_TOT_RE, mix=MAIN_MIX,
                          demand_cfg=demand_cfg,
                          path_preprocessed=path_preprocessed,
                          df_share=df_share, ssp=ssp, reanalysis=reanalysis,
                          out_dir=out_dir, period=period, suffix_shp=suffix_shp,
                          demand_tag=demand_name)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute the residual-load (SWBD) CSVs read by fig45.py and "
                    "the Extended Data figure scripts."
    )
    p.add_argument("--out_dir", default=config.RL_OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    path_preprocessed = config.PATH_PREPROCESSED
    ssp        = config.SSP
    reanalysis = config.REANALYSIS
    df_share   = pd.read_csv(config.SHARE_RENEWABLE_CSV)

    print("\n=== STEP 1a: RL CSVs -- tot_re sensitivity ===")
    run_rl_pipeline([0.25, 0.5, 0.75], [0.99], ["current"],
                    path_preprocessed, df_share, ssp, reanalysis, args.out_dir)
    print("\n=== STEP 1b: RL CSVs -- mix-effect baseline ===")
    # Future-mix CSV, consumed by suppfig11_mix_gwl_effects.py (mix vs GWL effects).
    run_rl_pipeline([0.5], [0.99], ["future"],
                    path_preprocessed, df_share, ssp, reanalysis, args.out_dir)
    print("\n=== STEP 1c: Demand-sensitivity CSVs ===")
    run_demand_sensitivity_pipeline(
        path_preprocessed, df_share, ssp, reanalysis, args.out_dir)
    print("\n=== STEP 1d: RL CSVs -- thr sensitivity ===")
    run_rl_pipeline([0.5], [0.95, 0.995], ["current"],
                    path_preprocessed, df_share, ssp, reanalysis, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()

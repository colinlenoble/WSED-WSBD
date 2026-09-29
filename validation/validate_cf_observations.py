# -*- coding: utf-8 -*-
"""
Validation of the regional reanalysis capacity factors (DS_CF) against
observed / independent national generation series.

For every country, the daily reanalysis scf / wcf series is correlated
(Pearson and Spearman) with the matching observed series over their whole
overlapping period -- one coefficient per country, no per-year split:

  ninja : Renewables.ninja v1.1 national capacity factors
          (wind: current fleet, 1980-2016; solar: MERRA-2, 1985-2016)
  entso : ENTSO-E actual generation per production type (MW; Solar and
          Wind Onshore), one .xlsx per country and year

Both weighting schemes of calculate_cf.aggregate_ds_cf_reanalysis() are
evaluated:
  v1 : weighted by the mean capacity factor over the reanalysis period
  v2 : weighted by grid-cell area only

Inputs are {preprocessed_path}/{reanalysis}/{scf,wcf}_agg_{reanalysis}_{v1,v2}.nc
(one value per poly_idx and day, no Feb 29) and config.NINJA_WIND_CSV,
config.NINJA_PV_CSV, config.ENTSOE_DIR.

Outputs (in --output_dir):
  validation_cf_{reanalysis}.csv            one row per (temporal extent,
                                             dataset, variable, weighting):
                                             n_countries and mean / median /
                                             min / max of both coefficients
  validation_cf_{reanalysis}_by_country.csv  one row per country

Port of como24_group5/code_final/1.4 validation_energy_potential.ipynb
(which used the W5E5 wpp / spp aggregates instead of the reanalysis CF).
"""
import os
import re
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
import pycountry
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from io_utils import open_dataset_any

MIN_DAYS = 365          # countries with a shorter overlap are skipped

# var: (short label, ninja key / ENTSO-E column)
VARIABLES = {
    "scf": ("SCF", "Solar"),
    "wcf": ("WCF", "Wind Onshore"),
}
VERSION_LABELS = {"v1": "potential-weighted", "v2": "area-weighted"}


# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Correlation of the regional reanalysis capacity factors "
                    "with Renewables.ninja and ENTSO-E national series.")
    p.add_argument("--preprocessed_path", default=config.PATH_PREPROCESSED,
                   help="Root holding <reanalysis>/{scf,wcf}_agg_<reanalysis>_{v1,v2}.nc.")
    p.add_argument("--reanalysis", default=config.REANALYSIS)
    p.add_argument("--ninja_wind", default=config.NINJA_WIND_CSV)
    p.add_argument("--ninja_pv", default=config.NINJA_PV_CSV)
    p.add_argument("--entsoe_dir", default=config.ENTSOE_DIR,
                   help="Folder of 'Actual Generation per Production Type_*_<ISO3>.xlsx' files.")
    p.add_argument("--output_dir", default="../final_figs")
    return p.parse_args()


# =============================================================================
# Data
# =============================================================================

def drop_feb29(obj):
    return obj[~((obj.index.month == 2) & (obj.index.day == 29))]


def load_reanalysis_cf(preprocessed_path, reanalysis, var, version):
    """{ISO_A3: daily pd.Series} from {var}_agg_{reanalysis}_{version}.nc.

    One polygon per country; France is metropolitan France only (the other
    FRA polygon is French Guiana).
    """
    path = os.path.join(preprocessed_path, reanalysis, f"{var}_agg_{reanalysis}_{version}.nc")
    with open_dataset_any(path) as ds:
        time_dim = "valid_time" if "valid_time" in ds[var].dims else "time"
        iso = ds["ISO_A3"].values
        names = ds["name"].values
        time = pd.to_datetime(ds[time_dim].values)
        data = ds[var].transpose("poly_idx", time_dim).values

    series = {}
    for code in np.unique(iso):
        if code == "FRA":
            idx = [k for k, n in enumerate(names) if "tropolitaine" in n]
        else:
            idx = np.flatnonzero(iso == code)
        if len(idx) == 1:
            series[code] = pd.Series(data[idx[0]], index=time)
    return series


def load_ninja(path):
    """Daily mean ninja capacity factors, columns renamed alpha-2 -> ISO_A3."""
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
    df.columns = [pycountry.countries.get(alpha_2=c).alpha_3 for c in df.columns]
    return drop_feb29(df.resample("D").mean())


def read_entsoe_file(path):
    """Daily mean generation (MW) of one ENTSO-E export.

    Each day starts with a 'dd.mm.yyyy' row in the MTU column (except the
    first, taken from the file-name start stamp), followed by hourly or
    15-min 'hh:mm - hh:mm' rows. The daily mean (not sum) keeps DST days and
    hourly vs. 15-min countries comparable.
    """
    df = pd.read_excel(path, skiprows=4).iloc[3:]          # drop the 3 unit rows
    start = re.search(r"_(\d{8})\d{4}-", os.path.basename(path)).group(1)
    mtu = df["MTU"].astype(str)
    is_date = mtu.str.fullmatch(r"\d{2}\.\d{2}\.\d{4}")
    date = pd.Series(pd.NaT, index=df.index)
    date[is_date] = pd.to_datetime(df.loc[is_date, "MTU"], format="%d.%m.%Y")
    date.iloc[0] = pd.to_datetime(start, format="%Y%m%d")
    df["date"] = date.ffill()
    df = df[mtu.str.contains(" - ")]
    cols = [c for _, c in VARIABLES.values() if c in df.columns]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")   # 'n/e', '-' -> NaN
    return df.groupby("date")[cols].mean()


def load_entsoe(entsoe_dir, end):
    """{ISO_A3: daily DataFrame} over all yearly exports, clipped to `end`."""
    per_country = {}
    for path in sorted(glob.glob(os.path.join(entsoe_dir, "Actual Generation*.xlsx"))):
        country = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
        per_country.setdefault(country, []).append(read_entsoe_file(path))

    out = {}
    for country, dfs in per_country.items():
        df = pd.concat(dfs)
        df = df[~df.index.duplicated()].sort_index()
        out[country] = drop_feb29(df[df.index <= end])
    return out


def observed_series(dataset, var, ninja, entsoe):
    """{ISO_A3: daily pd.Series} of the observed counterpart of `var`."""
    column = VARIABLES[var][1]
    if dataset == "ninja":
        return {c: ninja[var][c] for c in ninja[var].columns}
    return {c: df[column] for c, df in entsoe.items() if column in df.columns}


# =============================================================================
# Correlations
# =============================================================================

def correlate(observed, reanalysis_cf):
    """One row per country with a >= MIN_DAYS non-constant overlap."""
    rows, skipped = [], []
    for country, obs in observed.items():
        if country not in reanalysis_cf:
            skipped.append((country, "no polygon"))
            continue
        pair = pd.concat([obs.rename("obs"), reanalysis_cf[country].rename("cf")],
                         axis=1, join="inner").dropna()
        if len(pair) < MIN_DAYS or pair["obs"].nunique() < 2:
            skipped.append((country, f"{len(pair)} valid days"))
            continue
        rows.append({
            "country": country,
            "start": pair.index.min().date(),
            "end": pair.index.max().date(),
            "n_days": len(pair),
            "pearson": pearsonr(pair["obs"], pair["cf"])[0],
            "spearman": spearmanr(pair["obs"], pair["cf"])[0],
        })
    return rows, skipped


def summarize(by_country):
    keys = ["dataset", "variable", "weighting"]
    summary = by_country.groupby(keys, sort=False).agg(
        start=("start", "min"), end=("end", "max"), n_countries=("country", "count"),
        pearson_mean=("pearson", "mean"), pearson_median=("pearson", "median"),
        pearson_min=("pearson", "min"), pearson_max=("pearson", "max"),
        spearman_mean=("spearman", "mean"), spearman_median=("spearman", "median"),
        spearman_min=("spearman", "min"), spearman_max=("spearman", "max"),
    ).reset_index()
    extent = (summary["start"].map(lambda d: str(d.year)) + "-"
              + summary["end"].map(lambda d: str(d.year)))
    summary.insert(0, "temporal_extent", extent)
    return summary.drop(columns=["start", "end"])


# =============================================================================
# Main
# =============================================================================

def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cf = {(var, version): load_reanalysis_cf(args.preprocessed_path, args.reanalysis, var, version)
          for var in VARIABLES for version in VERSION_LABELS}
    cf_end = max(s.index.max() for s in next(iter(cf.values())).values())

    ninja = {"wcf": load_ninja(args.ninja_wind), "scf": load_ninja(args.ninja_pv)}
    entsoe = load_entsoe(args.entsoe_dir, cf_end)

    rows = []
    for dataset in ["ninja", "entso"]:
        for var in VARIABLES:
            observed = observed_series(dataset, var, ninja, entsoe)
            for version, label in VERSION_LABELS.items():
                country_rows, skipped = correlate(observed, cf[(var, version)])
                for country, reason in skipped:
                    print(f"  skipped {dataset} {var} {version} {country}: {reason}")
                for r in country_rows:
                    rows.append({"dataset": dataset, "variable": var,
                                 "weighting": f"{version} ({label})", **r})

    by_country = pd.DataFrame(rows)
    summary = summarize(by_country)

    keys = ["dataset", "variable", "weighting"]
    grouped = by_country.groupby(keys)
    by_country.insert(0, "temporal_extent",
                      grouped["start"].transform("min").map(lambda d: str(d.year)) + "-"
                      + grouped["end"].transform("max").map(lambda d: str(d.year)))

    out_summary = os.path.join(args.output_dir, f"validation_cf_{args.reanalysis}.csv")
    out_country = os.path.join(args.output_dir, f"validation_cf_{args.reanalysis}_by_country.csv")
    summary.to_csv(out_summary, index=False, float_format="%.4f")
    by_country.to_csv(out_country, index=False, float_format="%.4f")
    print(summary.to_string(index=False))
    print(f"Saved {out_summary}\nSaved {out_country}")


if __name__ == "__main__":
    main()

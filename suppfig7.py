# -*- coding: utf-8 -*-
"""
Supplementary figure 7: model agreement on projected RED severity changes
(panels a-b) and variability decomposition of the projections (panels c-d).

  a - agreement across simulations on significant severity changes at GWL 2°C
  b - same at GWL 3°C
  c - total projection spread of RED severity under 2°C warming
  d - fraction of that spread explained by internal variability

Terminology: frequency (events/year), duration (days/event) and intensity
(mean deficit on event days) are the three RED components; severity is
their product, frequency x duration x intensity. make_agg_files.py calls
intensity "severity"; it is renamed on load.

The yearly indicators are always rebuilt from scratch via
make_agg_files.load_agg_data_compound(), which reads every per-GCM
wpp_agg_*/spp_agg_* aggregate under --preprocessed_path -- there is no
cached compound_years_agg_freq_sev_dur.nc to read instead (it no longer
exists on disk). Significance of the severity change (each GWL vs GWL0-61,
per GCM/run) is then computed on that rebuilt dataset with a paired
permutation test + Benjamini-Hochberg FDR. Panels c-d read a precomputed
variance decomposition (--variability_nc).

Port of cell 26 of como24_group5/code_final/3.1.3 disagreements.ipynb.
"""
import os
import config
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
os.environ["ESMFMKFILE"] = config.ESMFMKFILE_XENV

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
import inspect
from scipy.stats import permutation_test

# scipy renamed permutation_test's RNG-seed parameter from 'random_state' to
# 'rng' in 1.15; detect once so this runs on either an older HPC scipy
# (random_state) or a newer one (rng) without an explicit version check.
_PERMUTATION_TEST_RNG_KW = (
    "rng" if "rng" in inspect.signature(permutation_test).parameters else "random_state"
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm

from map_overlays import draw_discrepancy_mask_polygons, add_exclusion_legend
from make_agg_files import load_agg_data_compound

FIG_WIDTH_IN = 5.15
MAP_EXTENT = [-180, 180, -58, 68]

# Share of (GCM-weighted) simulations with a significant increase / decrease
# that separates the three classes of each bivariate axis.
AGREEMENT_CLASS_THRESHOLDS = [0.25, 0.5]
# 3x3 bivariate palette, index = decrease_class * 3 + increase_class:
# reds = increase agreement, blues = decrease agreement, yellow = disagreement.
BIVARIATE_COLORS = [
    "#e8e8e8", "#eeaeae", "#f47474",
    "#aeaed9", "#ffd166", "#ffd166",
    "#7474c9", "#ffd166", "#ffffff",
]

GWL_LABELS = {"GWL1-5": "1.5°C", "GWL2": "2°C", "GWL3": "3°C"}


# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    agg_dir = os.path.join(config.PATH_PREPROCESSED, "agg_datasets")
    parser = argparse.ArgumentParser(
        description="Model agreement on RED severity changes and variability "
                    "decomposition (supplementary figure 7).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preprocessed_path", default=config.PATH_PREPROCESSED,
                        help="Root of the per-GCM wpp_agg_*/spp_agg_* aggregates "
                             "(make_agg_files.load_agg_data_compound()'s input). The "
                             "yearly indicators (frequency, duration, intensity/severity) "
                             "are always rebuilt from these from scratch -- there is no "
                             "cached compound_years_agg_freq_sev_dur.nc to read instead.")
    parser.add_argument("--save_significance_nc", default=None,
                        help="Optional path to also save the recomputed severity trend "
                             "significance to (e.g. for reuse by other figures). Not read "
                             "back in -- the significance is always recomputed from the "
                             "rebuilt indicators.")
    parser.add_argument("--variability_nc",
                        default=os.path.join(agg_dir, "custom_regional_analysis_v1.nc"),
                        help="Precomputed variance decomposition (variables 'total' and 'I').")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ssp", default=config.SSP)
    parser.add_argument("--reference_gwl", default="GWL0-61")
    parser.add_argument("--gwls", nargs=2, default=["GWL2", "GWL3"],
                        help="The two GWLs shown in panels a and b.")
    parser.add_argument("--shapefile_disag", default=config.SHAPEFILE_PATH_LIGHT,
                        help="Shapefile matching the poly_idx of the rebuilt indicators.")
    parser.add_argument("--shapefile_var", default=config.SHAPEFILE_PATH,
                        help="Shapefile matching the regions of --variability_nc.")
    parser.add_argument("--agreement_aggregated_nc", default=config.AGREEMENT_AGGREGATED_NC_PATH)
    parser.add_argument("--agreement_threshold", type=float, default=config.AGREEMENT_THRESHOLD)
    parser.add_argument("--output_dir", default=config.SUMMARY_FIGS_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


# =============================================================================
# Trend significance (permutation test + FDR), inlined from
# trend_significance_from_nc.py
# =============================================================================

def comparison_triplets(ds, ssp, reference_gwl):
    """Return unique (GCM, run, comparison GWL) values in dataset order."""
    triplets, seen = [], set()
    for gcm, run, scenario, gwl in zip(
        ds["GCM"].values, ds["run"].values, ds["ssp"].values, ds["gwl"].values,
    ):
        triplet = (str(gcm), str(run), str(gwl))
        if str(scenario) == ssp and str(gwl) != reference_gwl and triplet not in seen:
            seen.add(triplet)
            triplets.append(triplet)
    return triplets


def fdr_mask(pvalues, fdr):
    """Benjamini-Hochberg field-significance mask over poly_idx."""
    stacked = pvalues.stack(location=("poly_idx",))
    ranks = stacked.rank("location")
    thresholds = ranks / ranks.max("location") * fdr
    cutoff = stacked.where(stacked <= thresholds).max("location")
    return (stacked <= cutoff).where(stacked.notnull()).unstack("location")


def trend_for_metric(data_by_gwl, metric, comparison_gwl, reference_gwl,
                     alpha, n_resamples, seed):
    """Sign of the significant change of `metric` at `comparison_gwl` vs
    `reference_gwl`: paired permutation test on yearly values + Benjamini-
    Hochberg FDR at 2*alpha."""
    reference = data_by_gwl[metric].sel(gwl=reference_gwl).transpose("year", "poly_idx")
    comparison = data_by_gwl[metric].sel(gwl=comparison_gwl).transpose("year", "poly_idx")
    reference_values = reference.values
    comparison_values = comparison.values

    def mean_difference(x, y, axis):
        return np.mean(x - y, axis=axis)

    test = permutation_test(
        (comparison_values, reference_values),
        mean_difference,
        permutation_type="samples",
        alternative="two-sided",
        n_resamples=n_resamples,
        vectorized=True,
        axis=0,
        **{_PERMUTATION_TEST_RNG_KW: np.random.default_rng(seed)},
    )
    coords = {"poly_idx": data_by_gwl.poly_idx.values}
    pvalues = xr.DataArray(test.pvalue, coords=coords, dims="poly_idx")
    difference = xr.DataArray(
        np.mean(comparison_values - reference_values, axis=0), coords=coords, dims="poly_idx")
    significant = fdr_mask(pvalues, fdr=2 * alpha)
    return xr.where(significant, np.sign(difference), 0).astype(int)


# =============================================================================
# Data loading
# =============================================================================

def _squeeze_degenerate_dims(da, keep=("realization", "poly_idx", "year", "time")):
    """
    Drop any size-1 dimension of `da` besides `keep`. On the raw aggregated
    file, intensity (severity) can carry a stray length-1 'year' or 'time'
    dim alongside its real one -- a leftover from make_agg_files.py's
    coordinate bookkeeping (e.g. an auxiliary year label promoted to its
    own dimension), not a genuine second data axis.
    """
    degenerate = [d for d in da.dims if d not in ("realization", "poly_idx")
                 and da.sizes[d] == 1]
    return da.squeeze(degenerate, drop=True) if degenerate else da


def _year_dim(da, var_name):
    """The one dim of `da` besides realization/poly_idx -- its annual axis."""
    candidates = [d for d in da.dims if d not in ("realization", "poly_idx")]
    if len(candidates) != 1:
        sizes = {d: da.sizes[d] for d in candidates}
        raise ValueError(
            f"{var_name} has unexpected dims {da.dims} (sizes {sizes} besides "
            f"realization/poly_idx): expected exactly one non-degenerate "
            f"dimension for the annual axis after squeezing size-1 dims")
    return candidates[0]


def _align_year_axis(ds, variables=("frequency", "duration", "intensity")):
    """
    Normalise frequency/duration/intensity onto a single, positionally
    -indexed 'year' dimension before combining them. On the raw aggregated
    file (make_agg_files.py) frequency/duration come out of duration_xr()
    already on a 'year' dim, while intensity (severity) keeps
    resample(time=...)'s 'time' dim, renamed to 'year' with its labels
    turned into strings -- so even after the rename, its 'year' coordinate
    doesn't match frequency/duration's integer one, and it can also retain
    a stray degenerate ('time' or 'year') axis on top of the real one. Left
    as-is, multiplying the three together silently broadcasts the
    mismatched axes into a spurious extra dimension instead of erroring
    loudly.
    """
    for var in variables:
        ds[var] = _squeeze_degenerate_dims(ds[var])
    dims = {var: _year_dim(ds[var], var) for var in variables}
    lengths = {var: ds[var].sizes[dims[var]] for var in variables}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"frequency/duration/intensity have mismatched year "
                         f"axis lengths: {lengths}")
    for var in variables:
        da = ds[var].drop_vars(dims[var], errors="ignore")
        if dims[var] != "year":
            da = da.rename({dims[var]: "year"})
        ds[var] = da
    return ds.assign_coords(year=np.arange(1, lengths[variables[0]] + 1))


def load_indicators(preprocessed_path):
    """
    Yearly indicators with the current terminology: frequency, duration,
    intensity, and severity = frequency x duration x intensity. Rebuilt from
    scratch every call via make_agg_files.load_agg_data_compound() (reads
    every wpp_agg_*/spp_agg_* aggregate under `preprocessed_path`) -- there
    is no cached compound_years_agg_freq_sev_dur.nc to read instead. That
    function names intensity "severity"; it is renamed here.
    """
    ds = load_agg_data_compound(preprocessed_path)
    if "intensity" not in ds:
        if "severity" not in ds:
            raise ValueError("load_agg_data_compound() returned neither "
                             "'intensity' nor legacy 'severity'")
        ds = ds.rename({"severity": "intensity"})
    ds = _align_year_axis(ds)
    for var in ("frequency", "duration", "intensity"):
        ds[var] = ds[var].fillna(0)
    ds["severity"] = ds["frequency"] * ds["duration"] * ds["intensity"]
    return ds


def compute_severity_significance(ds, gwls, alpha, ssp, reference_gwl, n_resamples, seed):
    """
    Sign of the significant severity change of each GWL in `gwls` vs
    `reference_gwl`, per (GCM, run): +1 / -1 / 0 per poly_idx. Paired
    permutation test on yearly values, Benjamini-Hochberg FDR at 2*alpha,
    restricted to severity, one alpha and the plotted GWLs.
    """
    triplets = [t for t in comparison_triplets(ds, ssp, reference_gwl) if t[2] in gwls]
    if not triplets:
        raise ValueError(f"No (GCM, run) found for ssp={ssp!r} and GWLs {gwls}")

    results, metadata = [], []
    for i, (gcm, run, gwl) in enumerate(triplets, start=1):
        print(f"[{i}/{len(triplets)}] GCM={gcm}, run={run}, GWL={gwl}", flush=True)
        selected = ds.where(
            (ds["GCM"] == gcm) & (ds["run"] == run) & (ds["ssp"] == ssp),
            drop=True,
        ).groupby("gwl").mean(dim="realization")
        if reference_gwl not in set(map(str, selected.gwl.values)):
            raise ValueError(f"Missing {reference_gwl} for GCM={gcm}, run={run}")
        results.append(trend_for_metric(
            selected, "severity", gwl, reference_gwl, alpha, n_resamples, seed))
        metadata.append((gcm, run, gwl))

    gcms, runs, gwl_comp = map(np.asarray, zip(*metadata))
    trend = xr.concat(results, dim="realization").assign_coords(
        GCM=("realization", gcms),
        run=("realization", runs),
        GWL_comp=("realization", gwl_comp),
        alpha=("realization", np.full(len(metadata), alpha)),
    )
    out = trend.to_dataset(name="severity_trend")
    out["severity_trend"].attrs = {
        "long_name": "Sign of significant change in RED severity (frequency x duration x intensity)",
        "reference_gwl": reference_gwl,
        "test": f"two-sided paired permutation test, {n_resamples} resamples, "
                f"Benjamini-Hochberg FDR at {2 * alpha}",
    }
    return out


def agreement_fractions(sig, gwl):
    """
    GCM-weighted share of simulations with a significant increase and a
    significant decrease in severity at `gwl` (each GCM weighs 1/n_GCM,
    split evenly across its runs).
    """
    sub = sig.where(sig.GWL_comp == gwl, drop=True)
    counts = pd.Series(sub.GCM.values).value_counts()
    weights = xr.DataArray(
        [1.0 / counts[g] / counts.size for g in sub.GCM.values], dims="realization")
    trend = sub["severity_trend"]
    incr = (trend > 0).weighted(weights).mean(dim="realization")
    decr = (trend < 0).weighted(weights).mean(dim="realization")
    return incr, decr


def load_discrepancy_idx(agreement_aggregated_nc, agreement_threshold):
    """
    poly_idx where observed (ERA5) and projected trends disagree
    (agreement_pct <= threshold), as in fig45.load_hatch_agg. Returns an
    empty array (no mask) if the file is missing.
    """
    if agreement_aggregated_nc is None or not os.path.exists(agreement_aggregated_nc):
        print(f"  Agreement file not found ({agreement_aggregated_nc}) -- no discrepancy mask")
        return np.array([], dtype=int)
    agreement_pct = xr.open_dataarray(agreement_aggregated_nc)
    df = agreement_pct.to_dataframe(name="var").reset_index()
    return df[df["var"] <= agreement_threshold]["poly_idx"].values


# =============================================================================
# Figure
# =============================================================================

def _panel_letter(ax, letter):
    ax.text(0.01, 0.98, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=8, fontweight="bold")


def _draw_bivariate(ax, shp, incr, decr, discrepancy_idx, letter, center_label):
    class_incr = np.digitize(incr.values, AGREEMENT_CLASS_THRESHOLDS, right=True)
    class_decr = np.digitize(decr.values, AGREEMENT_CLASS_THRESHOLDS, right=True)
    shp = shp.copy()
    shp["var"] = class_decr * 3 + class_incr

    ax.coastlines(resolution="50m", color="black", linewidth=0.4, zorder=1)
    shp.boundary.plot(ax=ax, color="black", linewidth=0.15, transform=ccrs.PlateCarree())
    shp.plot(ax=ax, column="var", cmap=ListedColormap(BIVARIATE_COLORS),
             norm=BoundaryNorm(np.arange(-0.5, 9, 1), ncolors=9),
             linewidth=0, transform=ccrs.PlateCarree(), legend=False)
    draw_discrepancy_mask_polygons(
        ax, list(shp.loc[shp["poly_idx"].isin(discrepancy_idx)].geometry))

    ax.spines["geo"].set_visible(False)
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    _panel_letter(ax, letter)
    ax.text(0.5, 1.03, center_label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7)


def _draw_bivariate_legend(fig):
    leg_ax = fig.add_axes([0.12, 0.6, 0.11, 0.11])
    leg_ax.imshow(np.arange(9).reshape(3, 3), cmap=ListedColormap(BIVARIATE_COLORS),
                  norm=BoundaryNorm(np.arange(-0.5, 9, 1), ncolors=9),
                  origin="lower", extent=[-0.5, 2.5, -0.5, 2.5])
    leg_ax.set_xticks([0.4, 1.6])
    leg_ax.set_xticklabels(["25", "50"], fontsize=5)
    leg_ax.set_yticks([0.5, 1.5])
    leg_ax.set_yticklabels(["25", "50"], fontsize=5)
    leg_ax.set_xlabel("Increasing\nsimulations (%)", fontsize=5, labelpad=3)
    leg_ax.set_ylabel("Decreasing\nsimulations (%)", fontsize=5, labelpad=3)
    leg_ax.tick_params(length=0, pad=2)
    for spine in leg_ax.spines.values():
        spine.set_visible(False)


def _style_last_colorbar(fig, map_ax, labelsize):
    # GeoPandas adds the colorbar as the most recent axes.
    cbar_ax = fig.axes[-1]
    if cbar_ax is not map_ax:
        cbar_ax.tick_params(labelsize=labelsize, length=2, pad=1)
        cbar_ax.xaxis.label.set_size(5)


def plot_suppfig7(agreement_panels, shapefile_disag, discrepancy_idx,
                  ds_var, shapefile_var):
    """
    agreement_panels: two (incr, decr, center_label) tuples for panels a, b.
    ds_var: variance decomposition with 'total' (projection spread) and 'I'
    (internal-variability fraction), one value per region of shapefile_var.
    """
    shp_disag = gpd.read_file(shapefile_disag)
    shp_disag["poly_idx"] = shp_disag.index
    n_poly = agreement_panels[0][0].sizes["poly_idx"]
    if len(shp_disag) != n_poly:
        raise ValueError(f"{shapefile_disag} has {len(shp_disag)} polygons, "
                         f"significance data has {n_poly}")

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.65), dpi=300)
    gs = GridSpec(2, 2, hspace=0.0, wspace=0.0, figure=fig)
    proj = ccrs.Robinson()
    ax_a = fig.add_subplot(gs[0, 0], projection=proj)
    ax_b = fig.add_subplot(gs[0, 1], projection=proj)
    ax_c = fig.add_subplot(gs[1, 0], projection=proj)
    ax_d = fig.add_subplot(gs[1, 1], projection=proj)

    # -- a, b: agreement on significant severity changes ---------------------
    for ax, letter, (incr, decr, label) in zip((ax_a, ax_b), "ab", agreement_panels):
        _draw_bivariate(ax, shp_disag, incr, decr, discrepancy_idx, letter, label)
    _draw_bivariate_legend(fig)
    if len(discrepancy_idx):
        add_exclusion_legend(ax_b, show_discrepancy=True, show_wcf_zero=False,
                             discrepancy_style="mask")

    # -- c: total projection spread ------------------------------------------
    shp_var = gpd.read_file(shapefile_var)
    shp_var["total"] = ds_var["total"].values
    shp_var["I"] = ds_var["I"].values

    total_positive = shp_var["total"].where(shp_var["total"] > 0).min()
    ax_c.coastlines(linewidth=0.25)
    shp_var.boundary.plot(ax=ax_c, color="black", linewidth=0.15, transform=ccrs.PlateCarree())
    shp_var.plot(
        column="total", ax=ax_c, legend=True, transform=ccrs.PlateCarree(), cmap="Reds",
        norm=LogNorm(vmin=float(total_positive), vmax=float(shp_var["total"].max())),
        legend_kwds={"label": "Total projection spread of RED severity under 2°C warming",
                     "orientation": "horizontal", "shrink": 0.6, "pad": 0.02},
    )
    ax_c.spines["geo"].set_visible(False)
    _panel_letter(ax_c, "c")
    _style_last_colorbar(fig, ax_c, labelsize=4)

    # -- d: fraction of internal variability ---------------------------------
    ax_d.coastlines(linewidth=0.25)
    shp_var.boundary.plot(ax=ax_d, color="black", linewidth=0.15, transform=ccrs.PlateCarree())
    shp_var.plot(
        column="I", ax=ax_d, legend=True, transform=ccrs.PlateCarree(), cmap="PiYG",
        vmin=0, vmax=1,
        legend_kwds={"label": "Fraction of internal variability",
                     "orientation": "horizontal", "shrink": 0.6, "pad": 0.02},
    )
    arrow_props = dict(facecolor="black", width=0.15, headwidth=6, headlength=4)
    arrow_y = -0.10
    for x_head, x_tail, x_text, text in ((0.08, 0.18, 0.07, "Model"),
                                         (0.92, 0.82, 0.93, "Internal")):
        ax_d.annotate("", xy=(x_head, arrow_y), xytext=(x_tail, arrow_y),
                      xycoords="axes fraction", textcoords="axes fraction",
                      arrowprops=arrow_props)
        ax_d.text(x_text, arrow_y - 0.06, text, transform=ax_d.transAxes,
                  ha="center", va="top", fontsize=5)
    ax_d.spines["geo"].set_visible(False)
    _panel_letter(ax_d, "d")
    _style_last_colorbar(fig, ax_d, labelsize=5)

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print(f"Rebuilding yearly RED indicators from {args.preprocessed_path}")
    ds = load_indicators(args.preprocessed_path)
    print("Computing severity trend significance")
    sig = compute_severity_significance(
        ds, gwls=args.gwls, alpha=args.alpha, ssp=args.ssp,
        reference_gwl=args.reference_gwl, n_resamples=args.n_resamples, seed=args.seed,
    )
    if args.save_significance_nc:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_significance_nc)), exist_ok=True)
        sig.to_netcdf(args.save_significance_nc)
        print(f"  Saved {args.save_significance_nc}")

    agreement_panels = []
    for gwl in args.gwls:
        incr, decr = agreement_fractions(sig, gwl)
        agreement_panels.append((incr, decr, GWL_LABELS.get(gwl, gwl)))

    discrepancy_idx = load_discrepancy_idx(args.agreement_aggregated_nc, args.agreement_threshold)
    ds_var = xr.open_dataset(args.variability_nc)

    print("Plotting supplementary figure 7")
    fig = plot_suppfig7(agreement_panels, args.shapefile_disag, discrepancy_idx,
                        ds_var, args.shapefile_var)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "suppfig7_agreement_variability_combined.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

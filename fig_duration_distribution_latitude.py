# -*- coding: utf-8 -*-
"""
Distribution of WSED (wind-solar energy drought) event duration by latitude
zone, decomposed by global warming level (GWL) and pooled across every
available GCM/run realization.

Latitude zones (5), using this project's own poleward exclusion band
(MAP_LAT_SOUTH/MAP_LAT_NORTH -- see fig1.py/fig3.py: "Regions poleward of
68N and 58S were excluded due to artifacts in the duration metric"):
    Tropical          -23.5 .. 23.5
    Subtropical (N)     23.5 .. 35
    Subtropical (S)    -35 .. -23.5
    Midlatitude (N)     35 .. MAP_LAT_NORTH
    Midlatitude (S)     MAP_LAT_SOUTH .. -35

Event durations come from duration_decomposition.compute_event_table, the
same gap-free run-length encoding of the classic daily wind+solar
coincidence field used by fig1.py/fig3.py -- one row per contiguous compound
low-production spell (event) at one pixel, not a per-pixel-year mean, so the
resulting distribution reflects actual individual event lengths pooled over
every land pixel in the zone and every available GCM/run realization.

Two figures are produced:
  1. fig_duration_distribution_by_latitude.png
     One KDE curve per GWL (pooled over every GCM/run), one panel per zone,
     with a vertical dashed line at each GWL's mean duration.
  2. fig_duration_distribution_by_latitude_with_simulations.png
     Same, with every individual GCM/run's own KDE drawn faintly in the
     background (same colour as its GWL, low alpha) behind the pooled curve.
"""
import os
import config

import argparse
import gc

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Zarr/NetCDF-agnostic file lookup + opener, shared with calculate_cf.py /
# every other fig*.py script.
from io_utils import match_files, glob_any, open_dataset_any

# Generic event-table builder (gap-free run-length encoding of a boolean/0-1
# compound-day field into one row per event), shared with fig1.py/fig3.py's
# duration-class decomposition.
from duration_decomposition import compute_event_table

# =============================================================================
# Figure size / latitude-band constants (match fig1.py/fig3.py conventions)
# =============================================================================
FIG_WIDTH_IN = 5.15   # single column width -- fontsizes match LaTeX

MAP_LAT_SOUTH = -58.0
MAP_LAT_NORTH = 68.0

LAT_ZONE_EDGES = [MAP_LAT_SOUTH, -35.0, -23.5, 23.5, 35.0, MAP_LAT_NORTH]
LAT_ZONE_LABELS = [
    "Midlatitude (S)", "Subtropical (S)", "Tropical", "Subtropical (N)", "Midlatitude (N)",
]

GWL_KEYS = ["GWL0-61", "GWL1-5", "GWL2", "GWL3"]
GWL_LABELS = {
    "GWL0-61": "Baseline (~0.61°C)",
    "GWL1-5":  "+1.5°C",
    "GWL2":    "+2.0°C",
    "GWL3":    "+3.0°C",
}
# Cool (baseline) -> warm (highest GWL) progression, ColorBrewer RdYlBu-4.
GWL_COLORS = {
    "GWL0-61": "#2c7bb6",
    "GWL1-5":  "#abd9e9",
    "GWL2":    "#fdae61",
    "GWL3":    "#d7191c",
}


# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Distribution of WSED event duration by latitude zone, "
            "decomposed by GWL and pooled over every available GCM/run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preprocessed_path", default=config.PATH_PREPROCESSED)
    parser.add_argument("--ssp", default=config.SSP)
    parser.add_argument(
        "--gwl_list", nargs="+", default=GWL_KEYS,
        help="GWL keys to include (default: GWL0-61 GWL1-5 GWL2 GWL3).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Quantile defining a low-production day, from the GWL0-61 reference "
             "period (default: 0.1, matching fig1.py/fig3.py).",
    )
    parser.add_argument("--exclude_gcm", nargs="+", default=[])
    parser.add_argument("--exclude_gcm_run", nargs="+", default=config.EXCLUDE_GCM_RUN)
    parser.add_argument("--shapefile", default=config.SHAPEFILE_PATH)
    parser.add_argument("--output_dir", default="../final_figs")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--max_duration_days", type=float, default=None,
        help="X-axis cap shared by every panel (default: 99th percentile of "
             "every pooled event duration across all requested GWLs).",
    )
    parser.add_argument(
        "--min_events_for_kde", type=int, default=5,
        help="Minimum events a single GCM/run needs in a zone to get its own "
             "background KDE curve in the 'with_simulations' figure.",
    )
    return parser.parse_args()


# =============================================================================
# Land mask (same rasterize-from-grid approach as fig1.py/fig3.py)
# =============================================================================

def rasterize_shapefile(shapefile, shape, transform):
    return geometry_mask(
        geometries=shapefile["geometry"], all_touched=True,
        out_shape=shape, transform=transform, invert=True,
    )


def build_land_mask_from_grid(lat, lon, shapefile_path):
    shapefile = gpd.read_file(shapefile_path)
    shape = (len(lat), len(lon))
    transform = rasterio.transform.from_bounds(
        float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat)),
        len(lon), len(lat),
    )
    mask = rasterize_shapefile(shapefile, shape, transform)
    return mask[::-1, :]


# =============================================================================
# Latitude-zone assignment
# =============================================================================

def assign_lat_zone(lat_values):
    """Vectorized zone label for an array of latitudes, per LAT_ZONE_EDGES."""
    idx = np.digitize(np.asarray(lat_values, dtype=float), LAT_ZONE_EDGES[1:-1])
    return np.array(LAT_ZONE_LABELS, dtype=object)[idx]


def compute_pixel_counts_per_zone(preprocessed_path, shapefile_path):
    """
    Number of land pixels per latitude zone on the common ERA5 reference grid
    (the grid every GCM/run is ultimately compared against elsewhere in this
    project), independent of which GCM's own native grid produced the event
    durations -- so the "n pixels" shown per zone is a single, well-defined
    spatial property of the analysis domain, not a per-GCM count.
    """
    rea_files, _ = match_files(os.path.join(preprocessed_path, "ERA5", "wcf_day*"))
    if not rea_files:
        raise FileNotFoundError(
            f"No ERA5 reference file found under {os.path.join(preprocessed_path, 'ERA5')}. "
            "Cannot count reference-grid pixels per zone."
        )
    da = open_dataset_any(rea_files[0]).isel(time=0).wcf
    da = da.sortby("lat").sortby("lon")
    da = da.sel(lat=slice(MAP_LAT_SOUTH, MAP_LAT_NORTH))

    land_mask = build_land_mask_from_grid(da.lat.values, da.lon.values, shapefile_path)
    land_mask = land_mask & da.notnull().values

    zone_of_row = assign_lat_zone(da.lat.values)
    counts = {}
    for zlabel in LAT_ZONE_LABELS:
        row_sel = zone_of_row == zlabel
        counts[zlabel] = int(land_mask[row_sel, :].sum())
    return counts


# =============================================================================
# Discover available GCM/run realizations for one GWL
# =============================================================================

def discover_realizations(preprocessed_path, gwl, ssp, exclude_gcm, exclude_gcm_run):
    """
    Glob every wcf_day_*<ssp>*<gwl>_ERA5(.zarr|.nc) file under
    preprocessed_path/*/ and parse out (GCM, run) from the filename -- same
    convention as fig3.py's build_gridded_datasets (filenames are always
    wcf_day_{GCM}_{ssp}_{run}_{gwl}_ERA5.ext).
    """
    exclude_gcm = set(exclude_gcm or [])
    exclude_pairs = set(tuple(x.split(":")) for x in (exclude_gcm_run or []))

    wcf_paths = glob_any(os.path.join(preprocessed_path, "*", f"wcf_day_*{ssp}*{gwl}_ERA5"))
    realizations = []
    for p in wcf_paths:
        gcm = p.split("_")[-5]
        run = p.split("_")[-3]
        if gcm in exclude_gcm or (gcm, run) in exclude_pairs:
            print(f"    [excluded] {gcm} {run}")
            continue
        realizations.append((gcm, run))
    return realizations


# =============================================================================
# Per-realization daily compound field + event table
# =============================================================================

def build_daily_compound(preprocessed_path, gwl, gcm, run, ssp, threshold):
    """
    Classic daily wind+solar coincidence field (no rolling mean), same
    definition as fig1.py's build_daily_pipeline / fig3.py's
    _build_single_gcm: a day is "low wind" / "low solar" if wcf/scf falls at
    or below the `threshold` quantile of positive values over the GWL0-61
    reference period; "compound" is both at once. Kept at native GCM
    resolution (no regridding -- we only need per-pixel event lengths, not a
    cross-GCM common grid).
    """
    wcf_files, _ = match_files(
        os.path.join(preprocessed_path, gcm, f"wcf_day_{gcm}_{ssp}_{run}_{gwl}_ERA5"))
    scf_files, _ = match_files(
        os.path.join(preprocessed_path, gcm, f"scf_day_{gcm}_{ssp}_{run}_{gwl}_ERA5"))
    if not wcf_files or not scf_files:
        raise FileNotFoundError(f"Missing wcf/scf files for {gcm}/{run}/{gwl}")
    wcf = open_dataset_any(wcf_files[0]).convert_calendar("standard")
    scf = open_dataset_any(scf_files[0]).convert_calendar("standard")

    wcf_ref_paths = glob_any(
        os.path.join(preprocessed_path, gcm, f"wcf_day_{gcm}*{ssp}*{run}_GWL0-61_ERA5"))
    scf_ref_paths = glob_any(
        os.path.join(preprocessed_path, gcm, f"scf_day_{gcm}*{ssp}*{run}_GWL0-61_ERA5"))
    if not wcf_ref_paths or not scf_ref_paths:
        raise FileNotFoundError(f"Missing GWL0-61 reference files for {gcm}/{run}")
    wcf_ref = open_dataset_any(wcf_ref_paths[0]).convert_calendar("standard")
    scf_ref = open_dataset_any(scf_ref_paths[0]).convert_calendar("standard")

    wcf_thr = wcf_ref.wcf.where(wcf_ref.wcf > 0).quantile(threshold, dim="time")
    scf_thr = scf_ref.scf.where(scf_ref.scf > 0).quantile(threshold, dim="time")

    low_wind  = xr.where(wcf.wcf <= wcf_thr, 1, 0)
    low_solar = xr.where(scf.scf <= scf_thr, 1, 0)
    compound  = (low_wind * low_solar).astype(int)
    compound  = compound.sortby("lat").sortby("lon")
    compound["lat"] = compound["lat"].astype(float)
    compound["lon"] = compound["lon"].astype(float)
    compound  = compound.sel(lat=slice(MAP_LAT_SOUTH, MAP_LAT_NORTH))
    compound  = compound.load()

    del wcf, scf, wcf_ref, scf_ref
    gc.collect()
    return compound


def build_events_for_realization(preprocessed_path, gwl, gcm, run, ssp, threshold, shapefile_path):
    """
    One row per WSED event (contiguous compound low-production spell) at one
    land pixel, with its total duration (days), the year it started, and its
    latitude zone. Ocean pixels are zeroed out (not NaN) before event
    detection so they simply generate zero events, rather than needing
    NaN-aware handling downstream.
    """
    compound = build_daily_compound(preprocessed_path, gwl, gcm, run, ssp, threshold)

    land_mask = build_land_mask_from_grid(compound.lat.values, compound.lon.values, shapefile_path)
    land_mask_da = xr.DataArray(land_mask, dims=("lat", "lon"),
                                 coords={"lat": compound.lat, "lon": compound.lon})
    compound = compound.where(land_mask_da, 0).astype(int)

    df = compute_event_table(compound)
    df = df.drop_duplicates(["event_id", "lat", "lon"])[["lat", "year", "duration"]].copy()
    df["zone"] = assign_lat_zone(df["lat"].to_numpy())
    df["GCM"] = gcm
    df["run"] = run
    df["gwl"] = gwl

    del compound, land_mask_da
    gc.collect()
    return df


def build_all_events(preprocessed_path, gwl_list, ssp, threshold, shapefile_path,
                      exclude_gcm, exclude_gcm_run):
    """
    {gwl: DataFrame} of pooled per-event rows (lat, year, duration, zone,
    GCM, run, gwl) across every available (GCM, run) realization for that GWL.
    """
    events_by_gwl = {}
    empty_cols = ["lat", "year", "duration", "zone", "GCM", "run", "gwl"]
    for gwl in gwl_list:
        print(f"\n  -- {gwl} --")
        realizations = discover_realizations(preprocessed_path, gwl, ssp, exclude_gcm, exclude_gcm_run)
        if not realizations:
            print(f"    No files found for {gwl}, skipping.")
            events_by_gwl[gwl] = pd.DataFrame(columns=empty_cols)
            continue

        dfs = []
        for gcm, run in realizations:
            print(f"    {gcm} / {run}")
            try:
                dfs.append(build_events_for_realization(
                    preprocessed_path, gwl, gcm, run, ssp, threshold, shapefile_path))
            except Exception as exc:
                print(f"      [ERROR] {gcm}/{run}/{gwl}: {exc}")
            gc.collect()

        events_by_gwl[gwl] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=empty_cols)
    return events_by_gwl


# =============================================================================
# Plotting
# =============================================================================

def _kde_curve(values, x_grid, min_events):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < min_events or np.allclose(values, values[0]):
        return None
    try:
        kde = gaussian_kde(values)
    except np.linalg.LinAlgError:
        return None
    return kde(x_grid)


def plot_distributions(events_by_gwl, pixel_counts, gwl_list, output_path, dpi=300,
                        max_duration_days=None, show_individual=False,
                        min_events_for_kde=5, title=None):
    all_durations = np.concatenate([
        events_by_gwl[g]["duration"].to_numpy() for g in gwl_list if not events_by_gwl[g].empty
    ]) if any(not events_by_gwl[g].empty for g in gwl_list) else np.array([])
    if max_duration_days is None:
        max_duration_days = float(max(5.0, np.percentile(all_durations, 99))) if all_durations.size else 20.0
    x_grid = np.linspace(0.5, max_duration_days, 300)

    fig, axes = plt.subplots(
        len(LAT_ZONE_LABELS), 1, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 1.7), sharex=True,
    )

    for ax, zlabel in zip(axes, LAT_ZONE_LABELS):
        for gwl in gwl_list:
            color = GWL_COLORS.get(gwl, "gray")
            df_gwl = events_by_gwl[gwl]
            df_zone = df_gwl[df_gwl["zone"] == zlabel] if not df_gwl.empty else df_gwl

            if show_individual and not df_zone.empty:
                for (_gcm, _run), df_sim in df_zone.groupby(["GCM", "run"]):
                    y_sim = _kde_curve(df_sim["duration"].to_numpy(), x_grid, min_events_for_kde)
                    if y_sim is not None:
                        ax.plot(x_grid, y_sim, color=color, alpha=0.15, linewidth=0.6, zorder=1)

            if df_zone.empty:
                continue
            durations = df_zone["duration"].to_numpy()
            y_pooled = _kde_curve(durations, x_grid, min_events=2)
            if y_pooled is not None:
                ax.plot(x_grid, y_pooled, color=color, linewidth=1.8, zorder=3,
                        label=GWL_LABELS.get(gwl, gwl))
            ax.axvline(durations.mean(), color=color, linestyle="--", linewidth=1.2, zorder=4)

        n_px = pixel_counts.get(zlabel)
        ylabel = zlabel + (f"\nDensity (n={n_px:,} px)" if n_px is not None else "\nDensity")
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, max_duration_days)
        ax.grid(True, linestyle="--", alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

    axes[-1].set_xlabel("WSED event duration (days)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(gwl_list), fontsize=7,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    if title:
        fig.suptitle(title, fontsize=8, y=1.06)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1 - Counting land pixels per latitude zone (ERA5 reference grid)")
    print("=" * 60)
    pixel_counts = compute_pixel_counts_per_zone(args.preprocessed_path, args.shapefile)
    for z, n in pixel_counts.items():
        print(f"  {z}: {n} pixels")

    print("\n" + "=" * 60)
    print("STEP 2 - Building WSED event tables for every GCM/run x GWL")
    print("=" * 60)
    events_by_gwl = build_all_events(
        args.preprocessed_path, args.gwl_list, args.ssp, args.threshold,
        args.shapefile, args.exclude_gcm, args.exclude_gcm_run,
    )
    for gwl, df in events_by_gwl.items():
        n_sims = df[["GCM", "run"]].drop_duplicates().shape[0] if not df.empty else 0
        print(f"  {gwl}: {len(df)} events from {n_sims} GCM-run realizations")

    print("\n" + "=" * 60)
    print("STEP 3 - Plotting")
    print("=" * 60)
    out1 = os.path.join(args.output_dir, "fig_duration_distribution_by_latitude.png")
    plot_distributions(
        events_by_gwl, pixel_counts, args.gwl_list, out1, dpi=args.dpi,
        max_duration_days=args.max_duration_days, show_individual=False,
        title="WSED event-duration distribution by latitude zone and GWL",
    )
    print(f"  Saved -> {out1}")

    out2 = os.path.join(
        args.output_dir, "fig_duration_distribution_by_latitude_with_simulations.png")
    plot_distributions(
        events_by_gwl, pixel_counts, args.gwl_list, out2, dpi=args.dpi,
        max_duration_days=args.max_duration_days, show_individual=True,
        min_events_for_kde=args.min_events_for_kde,
        title=("WSED event-duration distribution by latitude zone and GWL\n"
               "(faint lines: individual GCM-run realizations)"),
    )
    print(f"  Saved -> {out2}")


if __name__ == "__main__":
    main()

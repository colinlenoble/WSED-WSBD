"""
Helpers for suppfig3_gcm_trend_wasserstein_composite.py (ED Fig. 3): load the
empirical Wasserstein grids and regional trend samples, tabulate W2 per
region, and draw the composite figure.
"""
import os

# Import these first on Windows: they ship their own GEOS/PROJ/GDAL DLLs and
# a conflicting earlier load fails with "DLL load failed".
import cartopy  # noqa: F401 (import ordering only)
import geopandas as gpd
from shapely.geometry import mapping
import rioxarray  # noqa: F401 (registers the .rio accessor)

import numpy as np
import pandas as pd
import xarray as xr

CODE_REVIEWED_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_DIR = os.path.join(CODE_REVIEWED_DIR, "test_data")
SHAPEFILE_PATH = os.path.join(CODE_REVIEWED_DIR, "..", "shapefiles", "final_shp", "ne_mix_adm0_adm1_greenland.shp")

# Ice sheets dropped from the land mask (Greenland, Antarctica).
EXCLUDE_ISO_A3 = {"GRL", "ATA"}

# Same four regions as trend_sev_eval.py.
REGIONS = [
    {"name": "Guiana Shield", "lat": [-10, 10], "lon": [-70, -50]},
    {"name": "Western U.S.",  "lat": [35, 50],  "lon": [-125, -105]},
    {"name": "India",         "lat": [10, 30],  "lon": [70, 90]},
    {"name": "Kenya",         "lat": [-5, 5],   "lon": [33, 42]},
]

# Known-bad realization (truncated rsds input).
EXCLUDE_GCM_RUN = {("EC-Earth3-Veg-LR", "r3i1p1f1")}

# As in trend_sev_eval_wasserstein.py: below this ERA5 bootstrap-trend std,
# normalized W2 is set to NaN instead of blowing up.
SIGMA_REF_MIN = 1e-6


def _land_mask_from_da(da, shapefile_path=SHAPEFILE_PATH):
    """Boolean land mask on da's (lat, lon) grid, without EXCLUDE_ISO_A3."""
    shapefile = gpd.read_file(shapefile_path)
    shapefile = shapefile[~shapefile["ISO_A3"].isin(EXCLUDE_ISO_A3)].reset_index(drop=True)

    land_da = da.copy()
    land_da = land_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    land_da = land_da.rio.write_crs("epsg:4326", inplace=True)
    clipped = land_da.rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=False, all_touched=True)
    return clipped.notnull()


# Display names of the regions ("Guiana Shield" box shown as "Northern Amazon").
SCOPE_DISPLAY_LABELS = {
    "Global": "World",
    "Guiana Shield": "Northern Amazon",
    "Western U.S.": "Western US",
    "India": "India",
    "Kenya": "Kenya",
}

# Letter linking each region's map box to its row in the dot plot.
SCOPE_LETTERS = {
    "Guiana Shield": "N",
    "Western U.S.": "W",
    "India": "I",
    "Kenya": "K",
}

# Regions with their own trend-distribution panels (b, c).
PANEL_REGIONS = ["Guiana Shield", "India"]

# Default inputs (outputs of trend_sev_eval_wasserstein.py).
IC_EMPIRICAL_W2_PATH = os.path.join(TEST_DATA_DIR, "agg_wasserstein_empirical_GCMs_all_year_ERA5.nc")
REGION_TREND_SAMPLES_PATH = os.path.join(TEST_DATA_DIR, "region_trend_samples_ERA5.nc")


def plot_wasserstein_by_scope(df, scope_order=None, scope_labels=None, scope_letters=None,
                               ax=None, out_path=None, seed=0, vertical=False, show_title=True,
                               show_legend=True, value_col="avg_wasserstein",
                               value_label="Mean per-pixel W2 distance to ERA5", y_rotation=0,
                               marker_size=40, jitter_width=0.18):
    """Dot plot of `value_col` per scope (World + regions), one jittered point per
    GCM run colored by GCM; `vertical=True` puts the scopes on x. Draws into
    `ax` if given (out_path then ignored).
    Returns (fig, ax, point_positions) with point_positions[(scope, GCM, run)]
    = (x, y) in data coordinates."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    if scope_order is None:
        scope_order = ["Global"] + [r["name"] for r in REGIONS]
    if scope_labels is None:
        scope_labels = SCOPE_DISPLAY_LABELS
    if scope_letters is None:
        scope_letters = SCOPE_LETTERS

    gcms = sorted(df["GCM"].unique())
    cmap = plt.get_cmap("tab20")
    color_map = {g: cmap(i % 20) for i, g in enumerate(gcms)}

    rng = np.random.default_rng(seed)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 0.9 * len(scope_order) + 1.8))
    else:
        fig = ax.figure

    point_positions = {}
    for row_idx, scope in enumerate(scope_order):
        sub = df[df.scope == scope]
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(sub))
        colors = [color_map[g] for g in sub["GCM"]]
        if vertical:
            xs = row_idx + jitter
            ax.scatter(xs, sub[value_col], c=colors, s=marker_size, edgecolor="black",
                       linewidth=0.3, alpha=0.85, zorder=3)
            for (_, row), xx in zip(sub.iterrows(), xs):
                point_positions[(row.scope, row.GCM, row.run)] = (float(xx), float(row[value_col]))
        else:
            ys = row_idx + jitter
            ax.scatter(sub[value_col], ys, c=colors, s=marker_size, edgecolor="black",
                       linewidth=0.3, alpha=0.85, zorder=3)
            for (_, row), yy in zip(sub.iterrows(), ys):
                point_positions[(row.scope, row.GCM, row.run)] = (float(row[value_col]), float(yy))

    def _label(s):
        name = scope_labels.get(s, s)
        letter = scope_letters.get(s)
        return f"{name} ({letter})" if letter else name

    if vertical:
        ax.set_xticks(range(len(scope_order)))
        ax.set_xticklabels([_label(s) for s in scope_order], rotation=45, ha="right")
        ax.set_ylabel(value_label)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    else:
        ax.set_yticks(range(len(scope_order)))
        ax.set_yticklabels([_label(s) for s in scope_order], rotation=y_rotation,
                            ha="right" if y_rotation else "center")
        ax.invert_yaxis()
        ax.set_xlabel(value_label)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        # At most 5 ticks, with as many decimals as the tick step needs.
        locator = MaxNLocator(nbins=5)
        ax.xaxis.set_major_locator(locator)
        ticks = locator.tick_values(*ax.get_xlim())
        step = ticks[1] - ticks[0] if len(ticks) > 1 else 0.0
        decimals = max(0, int(np.ceil(-np.log10(step)))) if step > 0 else 3
        ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))

    if show_title:
        ax.set_title("Per-GCM-run W2 distance to ERA5, by region")

    if show_legend:
        handles = [
            mlines.Line2D([0], [0], marker="o", linestyle="", color=color_map[g],
                          markeredgecolor="black", markeredgewidth=0.3, label=g)
            for g in gcms
        ]
        ax.legend(handles=handles, title="GCM", bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=7, title_fontsize=8, frameon=False)

    if standalone:
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig, ax, point_positions


# ---------------------------------------------------------------------------
# Empirical-distribution composite figure
# ---------------------------------------------------------------------------

def _empirical_w2_1d(samples_a, samples_b):
    """Empirical 2-Wasserstein distance between two 1-D sample arrays (sorted
    order statistics), used to label the curves of panels b/c."""
    a = np.sort(samples_a[np.isfinite(samples_a)])
    b = np.sort(samples_b[np.isfinite(samples_b)])
    n = min(a.size, b.size)
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def load_empirical_grids(w2_path=IC_EMPIRICAL_W2_PATH, shapefile_path=SHAPEFILE_PATH):
    """Per-pixel empirical W2 file without EXCLUDE_GCM_RUN, and its land mask."""
    ds = xr.open_dataset(w2_path)

    keep = np.array([(g, r) not in EXCLUDE_GCM_RUN for g, r in zip(ds.GCM.values, ds.run.values)])
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"Dropping {n_dropped} excluded realization(s): "
              f"{list(zip(ds.GCM.values[~keep], ds.run.values[~keep]))}")
    ds = ds.isel(realization=keep)

    land_mask = _land_mask_from_da(ds.w2_distance.isel(realization=0), shapefile_path)
    print(f"land fraction of grid: {float(land_mask.mean()):.1%}")
    return ds, land_mask


def load_region_trend_samples(path=REGION_TREND_SAMPLES_PATH):
    """Regional bootstrap trend samples (realization, region, boot) without
    EXCLUDE_GCM_RUN."""
    ds = xr.open_dataset(path)
    keep = np.array([(g, r) not in EXCLUDE_GCM_RUN for g, r in zip(ds.GCM.values, ds.run.values)])
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"Dropping {n_dropped} excluded realization(s) from region trend samples: "
              f"{list(zip(ds.GCM.values[~keep], ds.run.values[~keep]))}")
    return ds.isel(realization=keep)


def build_table_empirical(ds_w2, land_mask, scopes=None):
    """Per-realization table of w2_distance/w2_normalized averaged over each
    scope's land pixels (cos(lat)-weighted for Global, plain mean for the
    REGIONS boxes). This averages the per-pixel W2 field; compare
    build_table_aggregated, which takes W2 of the region-mean series."""
    if scopes is None:
        scopes = [("Global", None)] + [(r["name"], r) for r in REGIONS]

    rows = []
    for scope_name, box in scopes:
        w2 = ds_w2.w2_distance
        w2n = ds_w2.w2_normalized
        mask = land_mask
        if box is not None:
            w2 = w2.sel(lat=slice(*box["lat"]), lon=slice(*box["lon"]))
            w2n = w2n.sel(lat=slice(*box["lat"]), lon=slice(*box["lon"]))
            mask = mask.sel(lat=slice(*box["lat"]), lon=slice(*box["lon"]))
            avg_w2 = w2.where(mask).mean(dim=["lat", "lon"], skipna=True)
            avg_w2n = w2n.where(mask).mean(dim=["lat", "lon"], skipna=True)
        else:
            weights = np.cos(np.deg2rad(mask.lat))
            avg_w2 = w2.where(mask).weighted(weights).mean(dim=["lat", "lon"], skipna=True)
            avg_w2n = w2n.where(mask).weighted(weights).mean(dim=["lat", "lon"], skipna=True)

        for i in range(ds_w2.sizes["realization"]):
            rows.append({
                "scope": scope_name,
                "GCM": str(ds_w2.GCM.values[i]),
                "run": str(ds_w2.run.values[i]),
                "w2_distance": float(avg_w2.values[i]),
                "w2_normalized": float(avg_w2n.values[i]),
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["scope", "w2_distance"]).reset_index(drop=True)


def build_table_aggregated(region_samples, regions=None):
    """Per-realization table of region-aggregated W2: W2 between the bootstrap
    trend distributions of the region-mean GCM and ERA5 severity (no Global
    row). Used by panels b-d."""
    if regions is None:
        regions = [str(r) for r in region_samples.region.values]

    rows = []
    for region_name in regions:
        sub = region_samples.sel(region=region_name)
        for i in range(sub.sizes["realization"]):
            gcm_vals = sub.gcm_trend_samples.isel(realization=i).values
            ref_vals = sub.ref_trend_samples.isel(realization=i).values
            w2 = _empirical_w2_1d(gcm_vals, ref_vals)
            sigma_ref = float(np.nanstd(ref_vals[np.isfinite(ref_vals)], ddof=1))
            w2_normalized = w2 / sigma_ref if sigma_ref > SIGMA_REF_MIN else np.nan
            rows.append({
                "scope": region_name,
                "GCM": str(sub.GCM.values[i]),
                "run": str(sub.run.values[i]),
                "w2_distance": w2,
                "w2_normalized": w2_normalized,
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["scope", "w2_distance"]).reset_index(drop=True)


def empirical_ensemble_median_map(ds_w2, land_mask, value_var="w2_distance", weighted=True):
    """Per-pixel lower weighted median of `value_var` across realizations (always
    an actual value), with weights 1/n_runs(GCM) if `weighted`; NaNs ignored."""
    da = ds_w2[value_var].transpose("realization", ...)
    gcm = np.asarray(ds_w2.GCM.values).astype(str)
    if weighted:
        _, inv, counts = np.unique(gcm, return_inverse=True, return_counts=True)
        w_real = 1.0 / counts[inv]
    else:
        w_real = np.ones(gcm.size)

    vals = da.values
    flat = vals.reshape(vals.shape[0], -1)
    w = np.where(np.isfinite(flat), w_real[:, None], 0.0)
    order = np.argsort(np.where(np.isfinite(flat), flat, np.inf), axis=0)  # NaNs sorted last
    sorted_vals = np.take_along_axis(flat, order, axis=0)
    cum_w = np.cumsum(np.take_along_axis(w, order, axis=0), axis=0)
    total = cum_w[-1]
    idx = np.argmax(cum_w >= 0.5 * total, axis=0)
    med = np.take_along_axis(sorted_vals, idx[None, :], axis=0)[0]
    med = np.where(total > 0, med, np.nan).reshape(vals.shape[1:])

    result = da.isel(realization=0, drop=True).copy(data=med)
    return result.where(land_mask)


def build_wasserstein_composite_figure_empirical(df, ds_w2, region_samples, land_mask,
                                                  metric="w2_distance", weighted=True, out_path=None):
    """ED Fig. 3 composite for `metric` (w2_distance or w2_normalized):
      a: map of the per-pixel ensemble-median W2, with the REGIONS boxes;
      b, c: bootstrap trend distributions (KDEs) in Northern Amazon and India,
         all realizations in grey, the lowest- and highest-W2 ones in colour,
         ERA5 dashed;
      d: region-aggregated W2 per realization and region (same values as b, c).
    Panel a is per pixel, so it can rank realizations differently from b-d."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Rectangle
    import matplotlib.patheffects as pe
    import matplotlib.lines as mlines
    import cartopy.crs as ccrs
    from scipy.stats import gaussian_kde
    import colorsys

    if metric not in ("w2_distance", "w2_normalized"):
        raise ValueError(f"metric must be 'w2_distance' or 'w2_normalized', got {metric!r}")
    value_label = ("Ensemble-median empirical W2 distance to ERA5" if metric == "w2_distance"
                   else "Ensemble-median normalized empirical W2 distance to ERA5")
    value_label_agg = ("Region-aggregated W2 distance to ERA5" if metric == "w2_distance"
                        else "Region-aggregated normalized W2 distance to ERA5")

    FIG_WIDTH_IN = 5.15  # same column width as fig1.py
    LETTER_FONTSIZE = 8
    LABEL_FONTSIZE = 6

    gcms = sorted(df["GCM"].unique())
    gcm_colors = {g: plt.get_cmap("tab20")(i % 20) for i, g in enumerate(gcms)}

    def label_color(gcm):
        h, l, s = colorsys.rgb_to_hls(*gcm_colors[gcm][:3])
        return colorsys.hls_to_rgb(h, min(l, 0.55), s)

    text_halo = [pe.withStroke(linewidth=2, foreground="white")]
    box_halo = [pe.withStroke(linewidth=3, foreground="white")]

    # Region-aggregated W2 for panels b-d.
    df_agg = build_table_aggregated(region_samples)

    # ---- Panel a data ----
    # Median, not mean: W2 is heavy-tailed, one bad run would shift a pixel's mean.
    ens_med = empirical_ensemble_median_map(ds_w2, land_mask, value_var=metric, weighted=weighted)
    vmax = float(ens_med.max(skipna=True))
    lat_min, lat_max = -58, 68
    ens_view = ens_med.where((ens_med.lat >= lat_min) & (ens_med.lat <= lat_max))

    # Layout: map (a) on top; below, b/c on the left and d/legend on the right.
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 1.35), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 1.0], width_ratios=[1, 1],
                  hspace=0.16, wspace=0.55)
    ax_a = fig.add_subplot(gs[0, :], projection=ccrs.EqualEarth())

    gs_bc = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=1.0)
    ax_b = fig.add_subplot(gs_bc[0])
    ax_c = fig.add_subplot(gs_bc[1])

    gs_d = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], height_ratios=[0.72, 0.28], hspace=1.1)
    ax_d = fig.add_subplot(gs_d[0])
    ax_legend = fig.add_subplot(gs_d[1])
    ax_legend.axis("off")

    # ---- Panel a: ensemble-median map ----
    # w2_normalized (in units of ERA5's trend std) gets classes 0-1, 1-3, 3-5,
    # 5-7, 7+; w2_distance a continuous scale.
    discrete = metric == "w2_normalized"
    if discrete:
        from matplotlib.colors import BoundaryNorm, ListedColormap
        W2N_BOUNDS = [0, 1, 3, 5, 7]
        map_cmap = ListedColormap(plt.get_cmap("cividis")(np.linspace(0, 1, len(W2N_BOUNDS))))
        map_norm = BoundaryNorm(W2N_BOUNDS, map_cmap.N, extend="max")
        mb = ens_view.plot(ax=ax_a, cmap=map_cmap, norm=map_norm, transform=ccrs.PlateCarree(),
                           add_colorbar=False, add_labels=False)
    else:
        mb = ens_view.plot(ax=ax_a, cmap="cividis", vmin=0, vmax=vmax, transform=ccrs.PlateCarree(),
                            add_colorbar=False, add_labels=False)
    ax_a.coastlines(linewidth=0.4)
    ax_a.set_global()
    if "geo" in ax_a.spines:
        ax_a.spines["geo"].set_visible(False)
    eq, pc = ccrs.EqualEarth(), ccrs.PlateCarree()
    y_bottom = eq.transform_point(0, lat_min, pc)[1]
    y_top = eq.transform_point(0, lat_max, pc)[1]
    ax_a.set_ylim(y_bottom, y_top)
    if discrete:
        # Equal-width class boxes, ticks on the class boundaries.
        cbar = fig.colorbar(mb, ax=ax_a, orientation="horizontal", pad=0.1, shrink=0.85,
                            spacing="uniform", extend="max", extendfrac=0.12)
        cbar.set_ticks(W2N_BOUNDS)
        cbar.minorticks_off()
    else:
        cbar = fig.colorbar(mb, ax=ax_a, orientation="horizontal", pad=0.1, shrink=0.85)
    cbar.set_label(value_label, fontsize=LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=LABEL_FONTSIZE)
    ax_a.text(-0.03, 1.1, "a", transform=ax_a.transAxes, fontsize=LETTER_FONTSIZE, fontweight="bold")

    # Map boxes tagged with the letter of the panel showing their curves.
    box_panel_letters = {"Guiana Shield": "b", "India": "c"}

    for r in REGIONS:
        is_focus = r["name"] in PANEL_REGIONS
        rect = Rectangle((r["lon"][0], r["lat"][0]), r["lon"][1] - r["lon"][0], r["lat"][1] - r["lat"][0],
                          transform=ccrs.PlateCarree(), fill=False, edgecolor="black",
                          linewidth=1.8 if is_focus else 0.9, zorder=6, path_effects=box_halo)
        ax_a.add_patch(rect)

        letter = box_panel_letters.get(r["name"])
        if letter:
            # Letter above the top-left corner with a leader line (ax.plot: annotate's
            # arrowprops draws nothing on this GeoAxes).
            lon0, lat0 = r["lon"][0], r["lat"][1]
            text_lon, text_lat = lon0 - 5, lat0 + 5
            ax_a.plot([lon0, text_lon], [lat0, text_lat], transform=ccrs.PlateCarree(),
                      color="black", linewidth=0.8, alpha=0.9, zorder=7)
            ax_a.text(text_lon, text_lat, letter, transform=ccrs.PlateCarree(), ha="right", va="bottom",
                      fontsize=LETTER_FONTSIZE, fontweight="bold", color="black", zorder=8,
                      path_effects=text_halo)

    # ---- Panels b/c: empirical trend distributions for one region each ----
    def draw_region_panel(ax, region_name, tag, labels_right=False):
        sub = region_samples.sel(region=region_name)

        # Lowest/highest W2 from df_agg, matching panel d's extremes.
        region_rows = df_agg[df_agg.scope == region_name].sort_values(metric)
        best_row, worst_row = region_rows.iloc[0], region_rows.iloc[-1]

        def realization(row):
            i = int(np.where((sub.GCM.values == row.GCM) & (sub.run.values == row.run))[0][0])
            return {"GCM": row.GCM, "run": row.run, "w2_agg": float(row[metric]), "i": i}

        best, worst = realization(best_row), realization(worst_row)

        all_vals = sub.gcm_trend_samples.values
        finite = all_vals[np.isfinite(all_vals)]
        xlim = (float(finite.min()), float(finite.max()))
        xs = np.linspace(*xlim, 300)

        for i in range(sub.sizes["realization"]):
            vals = sub.gcm_trend_samples.isel(realization=i).values
            vals = vals[np.isfinite(vals)]
            if vals.size < 2 or np.ptp(vals) == 0:
                continue
            ax.plot(xs, gaussian_kde(vals)(xs), color="lightgray", linewidth=0.7, alpha=0.6, zorder=1)

        # One ERA5 curve (black, dashed): each realization's copy is ERA5 on its
        # GCM grid, so they differ only by regridding noise.
        ref_vals = sub.ref_trend_samples.isel(realization=best["i"]).values
        ref_curve = gaussian_kde(ref_vals)(xs)
        ax.plot(xs, ref_curve, color="black", linewidth=1.4, linestyle="--", zorder=5, alpha=0.8)
        ylim_top = float(ref_curve.max())

        # GCM labels in one top corner, ERA5 in the other (labels_right swaps them
        # when the peaks sit on the left).
        text_x, text_ha = (0.97, "right") if labels_right else (0.03, "left")
        era5_x, era5_ha = (0.03, "left") if labels_right else (0.97, "right")
        TEXT_ZORDER = 8

        w2_decimals = 1 if metric == "w2_normalized" else 4  # w2_distance is ~1e-4..1e-2
        for row, y in ((best, 0.97), (worst, 0.86)):
            i = row["i"]
            gcm_vals = sub.gcm_trend_samples.isel(realization=i).values
            color = gcm_colors[row["GCM"]]
            gcm_curve = gaussian_kde(gcm_vals)(xs)
            ax.plot(xs, gcm_curve, color=color, linewidth=2.0, zorder=6)
            ax.fill_between(xs, gcm_curve, color=color, alpha=0.15, zorder=2)
            ylim_top = max(ylim_top, float(gcm_curve.max()))

            lcolor = label_color(row["GCM"])
            ax.text(text_x, y, f"{row['GCM']}: W2 = {row['w2_agg']:.{w2_decimals}f}", transform=ax.transAxes,
                    ha=text_ha, va="top", fontsize=LABEL_FONTSIZE, color=lcolor, fontweight="bold",
                    path_effects=text_halo, zorder=TEXT_ZORDER)

        # ERA5 label on its own row, below the (possibly wide) GCM labels.
        ax.text(era5_x, 0.72, "ERA5", transform=ax.transAxes, ha=era5_ha, va="top",
                fontsize=LABEL_FONTSIZE, color="black", fontweight="bold", path_effects=text_halo,
                zorder=TEXT_ZORDER)

        ax.set_xlim(xlim)
        ax.set_ylim(0, ylim_top * 1.35 if ylim_top > 0 else 1)
        ax.set_yticks([])
        ax.set_xlabel("Severity trend", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Probability density", fontsize=LABEL_FONTSIZE)
        ax.tick_params(labelsize=LABEL_FONTSIZE)
        ax.text(-0.1, 1.1, tag, transform=ax.transAxes, fontsize=LETTER_FONTSIZE, fontweight="bold")
        ax.set_title(SCOPE_DISPLAY_LABELS.get(region_name, region_name), fontsize=LABEL_FONTSIZE + 1.5)

    draw_region_panel(ax_b, "Guiana Shield", "b")
    draw_region_panel(ax_c, "India", "c", labels_right=True)

    # ---- Panel d: region-aggregated W2 per region (rows Kenya ... Northern Amazon) ----
    scope_order = [r["name"] for r in REGIONS][::-1]
    _, ax_d, _ = plot_wasserstein_by_scope(df_agg, scope_order=scope_order, ax=ax_d, vertical=False,
                                            show_title=False, show_legend=False, scope_letters={},
                                            value_col=metric, value_label=value_label_agg,
                                            y_rotation=30, marker_size=14,
                                            jitter_width=0.3)
    ax_d.text(-0.12, 1.04, "d", transform=ax_d.transAxes, fontsize=LETTER_FONTSIZE, fontweight="bold")
    ax_d.xaxis.label.set_fontsize(LABEL_FONTSIZE)
    ax_d.tick_params(labelsize=LABEL_FONTSIZE)

    # ---- Shared GCM legend, above panel d ----
    gcm_handles = [
        mlines.Line2D([0], [0], marker="o", linestyle="", color=gcm_colors[g],
                      markeredgecolor="black", markeredgewidth=0.3, label=g)
        for g in gcms
    ]
    ncol = 2
    ax_legend.legend(handles=gcm_handles, title="GCM", ncol=ncol, loc="center",
                      fontsize=LABEL_FONTSIZE, title_fontsize=LABEL_FONTSIZE, frameon=False)

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def suppfig3_gcm_trend_wasserstein_composite_w2_normalized(df_empirical, ds_w2, region_samples, land_mask,
                                                            out_path, weighted=True):
    """ED Fig. 3: the composite figure for normalized W2."""
    return build_wasserstein_composite_figure_empirical(
        df_empirical, ds_w2, region_samples, land_mask,
        metric="w2_normalized", weighted=weighted, out_path=out_path,
    )

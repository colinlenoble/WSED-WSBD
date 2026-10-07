# -*- coding: utf-8 -*-
"""
Extended Data Fig. 2: reference-period (ERA5) mean of the meteorological
inputs, capacity factors and SWED indicators, six map panels (previously
produced by fig1.py).

Shares its data loading (ds_final, land mask, wcf-zero mask) and CLI with
main_figs/fig1.py.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # repo-root config.py; also puts main_pipeline/, main_figs/, supp_figs/, aux_code/ on sys.path
os.environ["CARTOPY_DATA_DIR"] = config.CARTOPY_DATA_DIR_XENV
os.environ['ESMFMKFILE'] = config.ESMFMKFILE_XENV

import cartopy.crs as ccrs
import pandas as pd
import geopandas as gpd
import rasterio
from io_utils import match_files, open_dataset_any
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from matplotlib import cm
from string import ascii_lowercase
import cmocean as cmo
from map_overlays import draw_wcf_zero_overlay

from fig1 import (build_parser, FIG_WIDTH_IN, fit_to_width, load_base_inputs,
                  mask_poles, rasterize_shapefile)


# =============================================================================
# Figure 3 Mean solar & wind capacity factor maps (reference period)
# =============================================================================

def plot_mean_variables_6panel(
    ds_final, mask, shapefile_path, path_preprocessed, reanalysis,
    ref_start="1982-01-01", ref_end="2001-12-31",
    wcf_zero_mask=None,
):
    lat_south, lat_north = -60, 68
    shapefile      = gpd.read_file(shapefile_path)
    shapefile_band = shapefile.cx[:, lat_south:lat_north]
    ref_label      = f"{pd.Timestamp(ref_start).year}-{pd.Timestamp(ref_end).year}"
    ref_start_year = pd.Timestamp(ref_start).year
    ref_end_year   = pd.Timestamp(ref_end).year

    freq_mean    = ds_final.frequency.where(mask == 1).sel(
        year=slice(ref_start_year, ref_end_year)).mean("year").sel(
        lat=slice(lat_south, lat_north))
    dur_mean     = ds_final.duration.where(mask == 1).sel(
        year=slice(ref_start_year, ref_end_year)).mean("year").sel(
        lat=slice(lat_south, lat_north))
    int_mean     = ds_final.severity.where(mask == 1).sel(
        year=slice(ref_start_year, ref_end_year)).mean("year").sel(
        lat=slice(lat_south, lat_north))
    ann_sev_mean = (ds_final.frequency * ds_final.severity * ds_final.duration).where(
        mask == 1).sel(year=slice(ref_start_year, ref_end_year)).mean("year").sel(
        lat=slice(lat_south, lat_north))
    std_pdd = (ds_final.frequency * ds_final.severity * ds_final.duration).where(
        mask == 1).std(dim="year").sel(lat=slice(lat_south, lat_north))

    da_comp = ds_final.frequency.isel(year=0).sel(lat=slice(lat_south, lat_north))
    t_comp  = rasterio.transform.from_bounds(
        da_comp.lon.min().item(), da_comp.lat.min().item(),
        da_comp.lon.max().item(), da_comp.lat.max().item(),
        len(da_comp.lon), len(da_comp.lat),
    )
    land_mask_comp  = rasterize_shapefile(shapefile_band, da_comp.shape, t_comp)[::-1, :]
    ocean_mask_comp = land_mask_comp & (da_comp.isnull())

    print(f"  Loading wcf/scf for 6-panel map (ref: {ref_start}-{ref_end})  ")
    wcf_files, _ = match_files(os.path.join(path_preprocessed, reanalysis, "wcf_day_*"))
    scf_files, _ = match_files(os.path.join(path_preprocessed, reanalysis, "scf_day_*"))
    chunks   = {"time": 1000, "lat": -1, "lon": -1}
    wcf_ref  = open_dataset_any(wcf_files[0], chunks=chunks).sel(
        lat=slice(lat_south, lat_north), time=slice(ref_start, ref_end))
    scf_ref  = open_dataset_any(scf_files[0], chunks=chunks).sel(
        lat=slice(lat_south, lat_north), time=slice(ref_start, ref_end))
    da_wcf   = wcf_ref.wcf.isel(time=0)
    t_wcf    = rasterio.transform.from_bounds(
        da_wcf.lon.min().item(), da_wcf.lat.min().item(),
        da_wcf.lon.max().item(), da_wcf.lat.max().item(),
        len(da_wcf.lon), len(da_wcf.lat),
    )
    land_mask_wcf = rasterize_shapefile(shapefile_band, da_wcf.shape, t_wcf)[::-1, :]
    wcf_mean = wcf_ref.wcf.mean(dim="time").where(land_mask_wcf)
    scf_mean = scf_ref.scf.mean(dim="time").where(land_mask_wcf)

    datasets   = [freq_mean, dur_mean, int_mean, ann_sev_mean, wcf_mean, scf_mean, std_pdd]
    title_list = [
        "Frequency", "Duration",
        "Intensity", "Annual SWED severity",
        "Wind Capacity\nFactor",     "Solar Capacity\nFactor",
        "Interannual variability of\nannual severity",
    ]
    legend_list = [
        "Events/yr", "Days/event", "Intensity/day of event",
        "Annual SWED severity", "Wind Capacity Factor", "Solar Capacity Factor", "Std of annual severity",
    ]
    cmap_list = [
        cmo.cm.solar.reversed(), cmo.cm.matter, cmo.cm.dense,
        cmo.cm.balance, cmo.cm.speed, cmo.cm.thermal, cmo.cm.amp,
    ]
    vmin_list = [0,  1,  0,    0, 0,   0,   0]
    vmax_list = [36, 3,  0.05, 1, 0.5, 0.5, 0.25]
    panellabels = list(ascii_lowercase[:7])

    # 2-column layout (4 rows), same width; height scales proportionally
    # so each map panel is bigger than the previous 3-column layout
    fig_width_in  = FIG_WIDTH_IN
    fig_height_in = fig_width_in * 1.2
    fig, axes = plt.subplots(4, 2, figsize=(fig_width_in, fig_height_in), dpi=300,
                             subplot_kw={"projection": ccrs.EqualEarth()})
    axes_flat = axes.flatten()


    for idx, ax in enumerate(axes_flat):
        if idx >= 7:
            ax.set_visible(False)
            continue
        ds = datasets[idx]
        if hasattr(ds, "load"):
            ds = ds.load()
        ax.set_global()
        mask_poles(ax, lat_south, lat_north)
        ax.coastlines(resolution="50m", linewidth=0.15, color="black")
        land_for_panel = land_mask_comp if idx in (0, 1, 2, 3, 6) else land_mask_wcf
        nan_for_panel  = ocean_mask_comp if idx in (0, 1, 2, 3, 6) else None
        draw_wcf_zero_overlay(ax, wcf_zero_mask, land_for_panel, ds.lat, ds.lon,
                              nan_data=nan_for_panel)
        ds.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap=cmap_list[idx], vmin=vmin_list[idx], vmax=vmax_list[idx],
            add_colorbar=True, add_labels=False,
            cbar_kwargs={"orientation": "horizontal", "shrink": 0.7,
                         "pad": 0.05, "label": legend_list[idx]},
            rasterized=True, linewidth=0,
        )
        shapefile_band.boundary.plot(ax=ax, color="black", linewidth=0.1,
                                     transform=ccrs.PlateCarree())
        ax.annotate(
            f"$\\mathbf{{{panellabels[idx]}}}$",
            xy=(0.02, 1.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=8,
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )
        # ax.set_title(title_list[idx], fontsize=5)
        cbar_ax = fig.axes[-1]
        cbar_ax.set_xlabel(legend_list[idx], fontsize=5)
        cbar_ax.tick_params(labelsize=5)
        ax.spines["geo"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    args = build_parser(__doc__.strip().splitlines()[0]).parse_args()
    ds_final, mask, wcf_zero_mask = load_base_inputs(args)

    print("Plotting mean variable maps (6 panels)")
    fig = plot_mean_variables_6panel(
        ds_final=ds_final, mask=mask, shapefile_path=args.shapefile,
        path_preprocessed=args.path_preprocessed,
        reanalysis=args.reanalysis,
        ref_start=args.ref_start, ref_end=args.ref_end,
        wcf_zero_mask=wcf_zero_mask,
    )
    out = os.path.join(args.output_dir, "supp", "suppfig2_mean_variables_6panel.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fit_to_width(fig)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

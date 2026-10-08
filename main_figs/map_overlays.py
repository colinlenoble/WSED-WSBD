# -*- coding: utf-8 -*-
"""
Exclusion layers shared by the map figures (fig1.py, fig3.py, fig45.py), so
every figure uses the same style:

  - grey fill : land pixels with no wind capacity (ERA5 wcf == 0 over the
                whole reference period, and the NaN data they produce), see
                draw_wcf_zero_overlay(). The only grey used on the maps.
  - black dots: pixels / polygons where observed (ERA5) and projected (GCM)
                trends disagree (agreement_pct <= config.AGREEMENT_THRESHOLD),
                see draw_discrepancy_dots() (gridded maps). Dots rather
                than a solid fill, so the colour underneath still shows.
  - black mask: same discrepancy criterion on the aggregated (polygon)
                maps (fig45.py, suppfig7_agreement_variability.py): polygons filled solid black,
                see draw_discrepancy_mask_polygons().

Kept free of config / heavy imports so any figure script can use it.
"""
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Mid grey (not lightgrey) so wcf-zero pixels stay distinct from the pale
# grey of coolwarm's neutral change bin.
WCF_ZERO_COLOR = "#8c8c8c"

# Dots are drawn with scatter on a staggered lon/lat lattice rather than a
# hatch, since matplotlib's "." hatch has a fixed dot size (thicker hatch
# lines only turn it into rings).
DISCREPANCY_DOT_SPACING = (1.5, 1.0)   # (lon, lat) deg between dots
DISCREPANCY_DOT_SIZE = 0.5             # scatter marker area (pt^2)
DISCREPANCY_DOT_COLOR = "black"


# =============================================================================
# Grey: no wind capacity
# =============================================================================

def draw_wcf_zero_overlay(ax, wcf_zero_mask, land_shp, target_lat, target_lon,
                          nan_data=None, zorder=7):
    """
    Single grey layer for excluded land pixels, drawn in WCF_ZERO_COLOR:
    land pixels where ERA5 wcf is exactly 0 over the whole reference period
    (wcf_zero_mask, built by fig1.py's build_wcf_zero_mask()) OR where the
    plotted data is NaN (nan_data, 2-D bool on target_lat x target_lon --
    those NaNs come from the same wcf == 0 pixels). Drawn above the
    discrepancy dots (zorder 6) so it stays visible regardless of their
    state at the same pixel. No-op if both masks are None/empty.

    Returns True if any grey pixel was drawn (e.g. to show its legend entry).
    """
    land = np.asarray(land_shp, dtype=bool)
    grey = np.zeros(land.shape, dtype=bool)
    if wcf_zero_mask is not None:
        wcf0 = wcf_zero_mask.astype(float).interp(
            lat=target_lat, lon=target_lon, method="nearest").values
        grey |= wcf0 > 0.5
    if nan_data is not None:
        grey |= np.asarray(nan_data, dtype=bool)
    grey &= land
    if not grey.any():
        return False
    ax.contourf(
        target_lon, target_lat, grey.astype(float),
        levels=[0.5, 1], colors=[WCF_ZERO_COLOR],
        transform=ccrs.PlateCarree(), zorder=zorder,
    )
    return True


# =============================================================================
# Dots: observation-projection trend discrepancy
# =============================================================================

def _dot_lattice(lon_min, lon_max, lat_min, lat_max):
    """Staggered lattice: every other row shifted by half a lon step."""
    step_lon, step_lat = DISCREPANCY_DOT_SPACING
    lat_pts = np.arange(lat_min, lat_max + step_lat / 2, step_lat)
    dot_lon, dot_lat = [], []
    for k, la in enumerate(lat_pts):
        lo = np.arange(lon_min + (step_lon / 2) * (k % 2), lon_max + step_lon / 2, step_lon)
        dot_lon.append(lo)
        dot_lat.append(np.full_like(lo, la))
    return np.concatenate(dot_lon), np.concatenate(dot_lat)


def _scatter_dots(ax, lon, lat, zorder):
    ax.scatter(
        lon, lat, s=DISCREPANCY_DOT_SIZE, c=DISCREPANCY_DOT_COLOR,
        marker="o", linewidths=0, transform=ccrs.PlateCarree(),
        zorder=zorder, rasterized=True,
    )


def draw_discrepancy_dots(ax, failed_mask, target_lat, target_lon, zorder=6):
    """
    Gridded maps: black dots over pixels where failed_mask (2-D, on
    target_lat x target_lon) is True. Each lattice dot is kept if its
    nearest grid pixel is True.
    """
    failed_mask = np.asarray(failed_mask)
    if not failed_mask.any():
        return
    lat = np.asarray(target_lat, dtype=float)
    lon = np.asarray(target_lon, dtype=float)
    dot_lon, dot_lat = _dot_lattice(lon.min(), lon.max(), lat.min(), lat.max())
    i_lat = np.abs(lat[:, None] - dot_lat[None, :]).argmin(axis=0)
    i_lon = np.abs(lon[:, None] - dot_lon[None, :]).argmin(axis=0)
    keep = failed_mask[i_lat, i_lon]
    _scatter_dots(ax, dot_lon[keep], dot_lat[keep], zorder)


def draw_discrepancy_dots_polygons(ax, geoms, zorder=4):
    """
    Polygon maps: black dots inside each geometry in geoms (lon/lat
    shapely geometries). Polygons too small to contain a lattice dot get
    one dot at their representative point, so none goes unmarked.
    """
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return
    polys = gpd.GeoDataFrame(geometry=geoms)
    lon_min, lat_min, lon_max, lat_max = polys.total_bounds
    dot_lon, dot_lat = _dot_lattice(lon_min, lon_max, lat_min, lat_max)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(dot_lon, dot_lat))
    hit = gpd.sjoin(pts, polys, how="inner", predicate="within")
    kept = pts.loc[np.unique(hit.index)]
    lon, lat = list(kept.geometry.x), list(kept.geometry.y)
    for i in set(range(len(polys))) - set(hit["index_right"]):
        rp = polys.geometry.iloc[i].representative_point()
        lon.append(rp.x)
        lat.append(rp.y)
    _scatter_dots(ax, lon, lat, zorder)


# =============================================================================
# Black mask: observation-projection trend discrepancy (aggregated maps)
# =============================================================================

DISCREPANCY_MASK_COLOR = "black"


def draw_discrepancy_mask_polygons(ax, geoms, zorder=4):
    """
    Aggregated (polygon) maps: fill each geometry in geoms (lon/lat shapely
    geometries) in solid black, hiding the colour underneath.
    """
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return
    ax.add_geometries(geoms, crs=ccrs.PlateCarree(),
                      facecolor=DISCREPANCY_MASK_COLOR,
                      edgecolor=DISCREPANCY_MASK_COLOR,
                      linewidth=0.15, zorder=zorder)


# =============================================================================
# Legend handles
# =============================================================================

class DiscrepancyDotsHandle(Line2D):
    """Legend handle drawn as three dots (see the handler registered below)."""


Legend.update_default_handler_map({DiscrepancyDotsHandle: HandlerLine2D(numpoints=3)})


def discrepancy_legend_handle(label="Obs.-projection trend discrepancy"):
    return DiscrepancyDotsHandle(
        [], [], linestyle="none", marker="o", color=DISCREPANCY_DOT_COLOR,
        markersize=2 * np.sqrt(DISCREPANCY_DOT_SIZE / np.pi), label=label)


def discrepancy_mask_legend_handle(label="Obs.-projection trend discrepancy"):
    return Patch(facecolor=DISCREPANCY_MASK_COLOR, edgecolor="none", label=label)


def wcf_zero_legend_handle(label="Excluded: no wind capacity"):
    return Patch(facecolor=WCF_ZERO_COLOR, edgecolor="none", label=label)


def add_exclusion_legend(ax, show_discrepancy=True, show_wcf_zero=True,
                         discrepancy_style="dots", wcf_zero_label=None, **kwargs):
    """
    Legend for the two exclusion layers, lower right of a map axes.
    discrepancy_style: "dots" (gridded maps) or "mask" (aggregated maps).
    wcf_zero_label overrides wcf_zero_legend_handle's default label.
    """
    handles = []
    if show_wcf_zero:
        handles.append(wcf_zero_legend_handle() if wcf_zero_label is None
                       else wcf_zero_legend_handle(wcf_zero_label))
    if show_discrepancy:
        handles.append(discrepancy_mask_legend_handle() if discrepancy_style == "mask"
                       else discrepancy_legend_handle())
    if not handles:
        return None
    opts = dict(loc="lower right", bbox_to_anchor=(1.0, -0.02), fontsize=4.5,
                frameon=False, handlelength=1.6, handleheight=0.9)
    opts.update(kwargs)
    leg = ax.legend(handles=handles, **opts)
    leg.set_zorder(30)   # above mask_poles' white caps (zorder 12)
    return leg

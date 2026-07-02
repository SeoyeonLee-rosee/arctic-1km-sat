#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import cartopy.io.shapereader as shpreader
from shapely.geometry import Polygon

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset


# ------------------ USER PATHS ------------------
ML_DIR_TILES = "../data/MODIS/"
ML_05_AVE    = "../data/MODIS/ave_0.5.npy"
ML_05_TRD    = "../data/MODIS/trend_0.5.npy"

CRU_05_AVE   = "../data/CRU/grid/ave_gr.npy"
CRU_05_TRD   = "../data/CRU/grid/trend.npy"
CRU_NC_LATLON= "../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc"

OUTDIR       = "../FIG_fin/"
os.makedirs(OUTDIR, exist_ok=True)
OUT_PNG      = os.path.join(OUTDIR, "Fig1_a-h_ML_and_MLminusCRU.png")


# ------------------ SETTINGS ------------------
LEVEL_IDX = 5
WARM = [4, 5, 6, 7, 8, 9, 10]
COLD = [11, 12, 1, 2, 3]
PER_YEAR_TO_PER_DECADE = 10.0

# --- palettes ---
MEAN_CMAP_A    = "Spectral_r"
MEAN_CMAP_C    = "YlGnBu_r"
TREND_CMAP     = "RdBu_r"
DIFF_MEAN_CMAP = "bwr"
DIFF_TREND_CMAP= "bwr"

# --- fixed ranges ---
A_RANGE  = (-10.0, 10.0)
C_RANGE  = (-40.0,  0.0)
B_RANGE  = (-10.0, 10.0)
D_RANGE  = (-10.0, 10.0)
EG_RANGE = (-2.0,   2.0)
FH_RANGE = (-0.2,   0.2)

BD_BOUNDS = np.array([
    -10.0, -7.0,
    -5.0, -4.0, -3.0, -2.0, -1.0,
     0.0,
     1.0,  2.0,  3.0,  4.0,  5.0,
     7.0, 10.0
], dtype=float)

FIG_DPI = 300


# ------------------ PANEL a WINDOWS ------------------
windows = [
    {"key": "U",  "name": "UR",  "lon": (58, 63),         "lat": (62, 67)},
    {"key": "CS", "name": "CS", "lon": (89, 94),         "lat": (65, 70)},
    {"key": "ES", "name": "ES", "lon": (125, 130),   "lat": (65, 70)},
    {"key": "AK", "name": "AK", "lon": (-155, -150),     "lat": (63.5, 68.5)},
]


# ============================================================
# Helpers
# ============================================================
def log(msg):
    print(f"[figure1] {msg}")

def month_to_index(mm):
    return int(mm) - 1

def seasonal_mean_from_12(arr12, months):
    idx = [month_to_index(m) for m in months]
    return np.nanmean(arr12[idx, ...], axis=0).astype(np.float32)

def _extract_level(arr, level_idx=LEVEL_IDX):
    if arr.ndim == 4:
        return arr[:, level_idx, :, :]
    if arr.ndim == 3:
        if arr.shape[0] > 20:
            return arr
        return arr[level_idx, :, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected shape: {arr.shape}")

def centers_to_edges_1d(c1d: np.ndarray) -> np.ndarray:
    c = np.asarray(c1d, dtype=float)
    if c.size < 2:
        raise ValueError("Need >=2 centers to build edges.")
    e = np.empty(c.size + 1, dtype=float)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0]    = c[0] - 0.5 * (c[1] - c[0])
    e[-1]   = c[-1] + 0.5 * (c[-1] - c[-2])
    return e

def make_north_polar_axes(ax):
    ax.set_extent([0, 359, 60, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)

    theta  = np.linspace(0, 2 * np.pi, 256)
    center = np.array([0.5, 0.5])
    radius = 0.5
    circle = mpath.Path(np.vstack([np.cos(theta), np.sin(theta)]).T * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    return ax

def add_polar_labels_outside(ax):
    lat_color = "0.6"
    fs_lat = 7.5
    ax.text(0.50, 0.02, "60°N", color=lat_color, fontsize=fs_lat,
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.50, 0.19, "70°N", color=lat_color, fontsize=fs_lat,
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.50, 0.36, "80°N", color=lat_color, fontsize=fs_lat,
            ha="center", va="center", transform=ax.transAxes)

    lon_color = "black"
    fs_lon = 9.0

    ax.text(0.50, 1.01, "180°", color=lon_color, fontsize=fs_lon,
            ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.50, -0.01, "0°", color=lon_color, fontsize=fs_lon,
            ha="center", va="top", transform=ax.transAxes)

    y120 = 0.78
    y60  = 0.23

    ax.text(-0.05, y120, "120°W", color=lon_color, fontsize=fs_lon,
            ha="left", va="center", transform=ax.transAxes)
    ax.text(-0.04, y60,  "60°W",  color=lon_color, fontsize=fs_lon,
            ha="left", va="center", transform=ax.transAxes)
    ax.text(1.05, y120, "120°E", color=lon_color, fontsize=fs_lon,
            ha="right", va="center", transform=ax.transAxes)
    ax.text(1.03, y60,  "60°E",  color=lon_color, fontsize=fs_lon,
            ha="right", va="center", transform=ax.transAxes)

def add_panel_colorbar(ax, mappable, ticks, fmt="%.1f"):
    cax = ax.inset_axes([0.10, -0.10, 0.80, 0.04])
    cb  = plt.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
    cb.ax.tick_params(labelsize=9)
    cb.set_label("")
    return cb

def plot_ml_tiles_scatter(ax, lon_tiles, lat_tiles, fld_tiles, cmap, norm, s=1.15):
    for ti in range(fld_tiles.shape[0]):
        lo = lon_tiles[ti].ravel()
        la = lat_tiles[ti].ravel()
        vv = fld_tiles[ti].ravel()
        m = np.isfinite(lo) & np.isfinite(la) & np.isfinite(vv)
        if not np.any(m):
            continue
        ax.scatter(lo[m], la[m], c=vv[m],
                   s=s, cmap=cmap, norm=norm,
                   transform=ccrs.PlateCarree(),
                   linewidths=0, rasterized=True)
    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])
    return sm

def plot_gridded_pcolormesh(ax, lon1d, lat1d, fld2d, cmap, norm):
    lon_e = centers_to_edges_1d(lon1d)
    lat_e = centers_to_edges_1d(lat1d)
    pm = ax.pcolormesh(lon_e, lat_e, fld2d,
                       cmap=cmap, norm=norm,
                       shading="auto",
                       transform=ccrs.PlateCarree(),
                       rasterized=True)
    return pm

def shade_greenland(ax, color="0.85", zorder=1):
    shp = shpreader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries"
    )
    reader = shpreader.Reader(shp)
    for rec in reader.records():
        name = rec.attributes.get("NAME_LONG") or rec.attributes.get("NAME")
        if name == "Greenland":
            ax.add_geometries(
                [rec.geometry],
                crs=ccrs.PlateCarree(),
                facecolor=color,
                edgecolor="none",
                zorder=zorder
            )

def add_region_boxes_and_labels(ax, windows,
                                edgecolor="blue",
                                linewidth=1.0,
                                textcolor="black",
                                fontsize=7,
                                fontweight="bold",
                                zorder_box=8,
                                zorder_text=9):
    """
    Draw transparent rectangular windows with blue outlines
    and put labels at the center of each box on panel a.
    """
    for w in windows:
        lon0, lon1 = w["lon"]
        lat0, lat1 = w["lat"]

        poly = Polygon([
            (lon0, lat0),
            (lon1, lat0),
            (lon1, lat1),
            (lon0, lat1),
            (lon0, lat0),
        ])

        ax.add_geometries(
            [poly],
            crs=ccrs.PlateCarree(),
            facecolor="none",          # transparent background
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder_box
        )

        lon_c = 0.5 * (lon0 + lon1)
        lat_c = 0.5 * (lat0 + lat1)

        ax.text(
            lon_c, lat_c, w["name"],
            transform=ccrs.PlateCarree(),
            ha="center", va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            color=textcolor,
            zorder=zorder_text
        )


# ============================================================
# Loaders
# ============================================================
def load_ml_latlon_tiles():
    lat = np.load(os.path.join(ML_DIR_TILES, "lat.npy")).astype(np.float32)
    lon = np.load(os.path.join(ML_DIR_TILES, "lon.npy")).astype(np.float32)
    if lat.shape != lon.shape:
        raise ValueError(f"ML lat/lon shape mismatch: {lat.shape} vs {lon.shape}")
    return lat, lon

def load_ml_month_tile(mm, fname):
    fp = os.path.join(ML_DIR_TILES, f"{mm:02d}", fname)
    if not os.path.exists(fp):
        return None
    arr = np.load(fp)
    return _extract_level(arr, LEVEL_IDX).astype(np.float32)

def build_ml_season_tiles(months):
    mean_list, trd_list = [], []
    for m in months:
        a = load_ml_month_tile(m, "ave_gr.npy")
        t = load_ml_month_tile(m, "trend_gr.npy")
        if a is not None:
            mean_list.append(a)
        if t is not None:
            trd_list.append(t)

    if len(mean_list) == 0:
        return None, None

    mean = np.nanmean(np.stack(mean_list, axis=0), axis=0).astype(np.float32)

    if len(trd_list) == 0:
        trd = np.full_like(mean, np.nan, dtype=np.float32)
    else:
        trd = np.nanmean(np.stack(trd_list, axis=0), axis=0).astype(np.float32) * PER_YEAR_TO_PER_DECADE

    return mean, trd

def load_cru_latlon_from_nc(lat_min=60.0):
    with Dataset(CRU_NC_LATLON) as nc:
        lat_all = nc.variables["lat"][:].astype(float)
        lon_all = nc.variables["lon"][:].astype(float)

    lon_all = np.where(lon_all > 180, lon_all - 360, lon_all)
    lat_mask = lat_all >= lat_min
    return lat_all[lat_mask].astype(np.float32), lon_all.astype(np.float32), lat_mask

def load_05_arrays():
    mod_ave12 = np.load(ML_05_AVE).astype(np.float32)
    mod_trd12 = np.load(ML_05_TRD).astype(np.float32)

    cru_ave12 = np.load(CRU_05_AVE).astype(np.float32)
    cru_trd12 = np.load(CRU_05_TRD).astype(np.float32)

    if mod_ave12.shape != cru_ave12.shape:
        raise ValueError(f"Shape mismatch ave: MOD {mod_ave12.shape} vs CRU {cru_ave12.shape}")
    if mod_trd12.shape != cru_trd12.shape:
        raise ValueError(f"Shape mismatch trend: MOD {mod_trd12.shape} vs CRU {cru_trd12.shape}")

    lat, lon, lat_mask = load_cru_latlon_from_nc(lat_min=60.0)

    if mod_ave12.shape[1] == 360:
        mod_ave12 = mod_ave12[:, lat_mask, :]
        mod_trd12 = mod_trd12[:, lat_mask, :]
        cru_ave12 = cru_ave12[:, lat_mask, :]
        cru_trd12 = cru_trd12[:, lat_mask, :]

    if (lat.size, lon.size) != (mod_ave12.shape[1], mod_ave12.shape[2]):
        raise ValueError(
            f"CRU nc lat/lon ({lat.size},{lon.size}) != npy grid ({mod_ave12.shape[1]},{mod_ave12.shape[2]})"
        )

    return mod_ave12, mod_trd12, cru_ave12, cru_trd12, lat, lon

def build_diff_season(months):
    mod_ave12, mod_trd12, cru_ave12, cru_trd12, lat, lon = load_05_arrays()

    mod_mean = seasonal_mean_from_12(mod_ave12, months)
    cru_mean = seasonal_mean_from_12(cru_ave12, months)

    mod_trd  = seasonal_mean_from_12(mod_trd12, months)
    cru_trd  = seasonal_mean_from_12(cru_trd12, months)

    return (mod_mean - cru_mean).astype(np.float32), (mod_trd - cru_trd).astype(np.float32), lat, lon


# ============================================================
# Main
# ============================================================
def main():
    log("Loading ML tile lat/lon...")
    lat_tiles, lon_tiles = load_ml_latlon_tiles()

    log("Building ML warm/cold seasonal tiles...")
    ML_mean_w, ML_trd_w = build_ml_season_tiles(WARM)
    ML_mean_c, ML_trd_c = build_ml_season_tiles(COLD)

    if ML_mean_w is None:
        ML_mean_w = np.full_like(lat_tiles, np.nan, dtype=np.float32)
        ML_trd_w  = np.full_like(lat_tiles, np.nan, dtype=np.float32)
    if ML_mean_c is None:
        ML_mean_c = np.full_like(lat_tiles, np.nan, dtype=np.float32)
        ML_trd_c  = np.full_like(lat_tiles, np.nan, dtype=np.float32)

    log("Building (ML-CRU) diff on 0.5° grid...")
    DIFF_mean_w, DIFF_trd_w, lat05, lon05 = build_diff_season(WARM)
    DIFF_mean_c, DIFF_trd_c, _, _         = build_diff_season(COLD)

    # ── norms ──────────────────────────────────────────────────────────
    norm_a = Normalize(vmin=A_RANGE[0], vmax=A_RANGE[1])
    norm_c = Normalize(vmin=C_RANGE[0], vmax=C_RANGE[1])

    cmap_diff = plt.get_cmap(DIFF_MEAN_CMAP)
    norm_b = mcolors.BoundaryNorm(BD_BOUNDS, ncolors=cmap_diff.N, clip=True)
    norm_d = mcolors.BoundaryNorm(BD_BOUNDS, ncolors=cmap_diff.N, clip=True)

    norm_e = TwoSlopeNorm(vmin=EG_RANGE[0], vcenter=0.0, vmax=EG_RANGE[1])
    norm_g = norm_e
    norm_f = TwoSlopeNorm(vmin=FH_RANGE[0], vcenter=0.0, vmax=FH_RANGE[1])
    norm_h = norm_f

    # ── ticks ──────────────────────────────────────────────────────────
    ticks_a = np.linspace(A_RANGE[0], A_RANGE[1], 5)
    ticks_c = np.linspace(C_RANGE[0], C_RANGE[1], 5)

    ticks_b = [-10, -7, -5, -3, -1, 1, 3, 5, 7, 10]
    ticks_d = ticks_b

    ticks_e = np.linspace(EG_RANGE[0], EG_RANGE[1], 5)
    ticks_g = ticks_e
    ticks_f = np.linspace(FH_RANGE[0], FH_RANGE[1], 5)
    ticks_h = ticks_f

    # ── layout ─────────────────────────────────────────────────────────
    proj = ccrs.NorthPolarStereo(central_longitude=0)
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.6), subplot_kw=dict(projection=proj))
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.03, top=0.99,
                        wspace=0.06, hspace=0.26)

    panel_order = [
        # row 0
        ("a", "tiles",  ML_mean_w,   MEAN_CMAP_A,     norm_a, ticks_a),
        ("b", "tiles",  ML_mean_c,   MEAN_CMAP_C,     norm_c, ticks_c),
        ("c", "tiles",  ML_trd_w,    TREND_CMAP,      norm_e, ticks_e),
        ("d", "tiles",  ML_trd_c,    TREND_CMAP,      norm_g, ticks_g),
        # row 1
        ("e", "grid05", DIFF_mean_w, DIFF_MEAN_CMAP,  norm_b, ticks_b),
        ("f", "grid05", DIFF_mean_c, DIFF_MEAN_CMAP,  norm_d, ticks_d),
        ("g", "grid05", DIFF_trd_w,  DIFF_TREND_CMAP, norm_f, ticks_f),
        ("h", "grid05", DIFF_trd_c,  DIFF_TREND_CMAP, norm_h, ticks_h),
    ]

    for i, (tag, kind, field, cmap, norm, ticks) in enumerate(panel_order):
        r = 0 if i < 4 else 1
        c = i % 4
        ax = axes[r, c]

        make_north_polar_axes(ax)
        shade_greenland(ax, color="0.95")
        add_polar_labels_outside(ax)

        if kind == "tiles":
            mappable = plot_ml_tiles_scatter(ax, lon_tiles, lat_tiles, field,
                                             cmap=cmap, norm=norm, s=1.12)
        else:
            mappable = plot_gridded_pcolormesh(ax, lon05, lat05, field,
                                               cmap=cmap, norm=norm)

        # panel a 에만 영역 표시
        if tag == "a":
            add_region_boxes_and_labels(
                ax, windows,
                edgecolor="blue",
                linewidth=1.0,
                textcolor="black",
                fontsize=8,
                fontweight="bold"
            )

        ax.text(0.02, 0.98, tag, transform=ax.transAxes,
                ha="left", va="top", fontsize=15, fontweight="bold")

        if tag in ("e", "f"):
            add_panel_colorbar(ax, mappable, ticks=ticks, fmt="%d")
        else:
            add_panel_colorbar(ax, mappable, ticks=ticks, fmt="%.1f")

    fig.savefig(OUT_PNG, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
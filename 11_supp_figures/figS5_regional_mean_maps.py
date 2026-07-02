#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Supplementary Fig. 5-style regional SAT maps
Layout: 4 rows x 3 columns

Columns:
    ML | CRU | ERA5

Rows:
    Ural
    Central Siberia
    Eastern Siberia
    Northern Alaska

Field:
    warm-season mean SAT (Apr-Oct)

Rendering:
    ML   : scatter
    CRU  : pcolormesh
    ERA5 : pcolormesh

Colorbars:
    - fixed color range: -8 to 8
    - one shared horizontal colorbar at the bottom
"""

import os
import warnings
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from matplotlib.cm import ScalarMappable

# =========================================================
# Paths
# =========================================================
ML_DIR     = "../data/MODIS/"
CRU_NC     = "../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc"
CRU_AVE    = "../data/CRU/grid/ave_gr.npy"
ERA_NC     = "../data/ERA5_t2m_mon_1940-2025.nc"

OUTDIR = "../FIG_fin"
os.makedirs(OUTDIR, exist_ok=True)

OUT_PNG = os.path.join(OUTDIR, "FigS5_regional_warm_mean_ML_CRU_ERA5_4x3_fixedrange_new.png")
OUT_PDF = os.path.join(OUTDIR, "FigS5_regional_warm_mean_ML_CRU_ERA5_4x3_fixedrange_new.pdf")

# =========================================================
# Options
# =========================================================
WARM = [4, 5, 6, 7, 8, 9, 10]
LEVEL_IDX = 5
MARGIN_CRU = 0.5

# fixed color range
FIXED_VMIN = -8.0
FIXED_VMAX = 8.0

REGIONS = [
    dict(key="U",  name="Ural",             lon=(58.0, 63.0),     lat=(62.0, 67.0)),
    dict(key="CS", name="Central Siberia",  lon=(89.0, 94.0),     lat=(65.0, 70.0)),
    dict(key="ES", name="Eastern Siberia",  lon=(125, 130),   lat=(65.0, 70.0)),
    dict(key="NA", name="Northern Alaska",  lon=(-155.0, -150.0), lat=(63.5, 68.5)),
]

COLUMN_TITLES = ["ML", "CRU", "ERA5"]

# =========================================================
# Layout / typography controls
# =========================================================
FIG_W = 11.5
FIG_H = 14.8

LEFT   = 0.10
RIGHT  = 0.97
BOTTOM = 0.11
TOP    = 0.93

# tighter column spacing
WSPACE = 0.015
HSPACE = 0.14

# room for external row labels
ROW_LABEL_X = 0.055

# column titles
COL_TITLE_Y_OFFSET = 0.012
COL_TITLE_SIZE = 17
COL_TITLE_WEIGHT = "bold"

# row titles
ROW_TITLE_SIZE = 16
ROW_TITLE_WEIGHT = "bold"

# panel tags
PANEL_TAG_SIZE = 16

# colorbar: narrower and thinner
CB_LEFT   = 0.28
CB_BOTTOM = 0.060
CB_WIDTH  = 0.44
CB_HEIGHT = 0.012
CB_LABEL_SIZE = 12
CB_UNIT_TEXT = "[°C]"
CB_UNIT_SIZE = 12
CB_UNIT_DX = 0.010
CB_UNIT_DY = 0.008

# global fonts
BASE_FONT_SIZE = 13
AXES_TITLE_SIZE = 17
AXES_LABEL_SIZE = 13
TICK_LABEL_SIZE = 12

# =========================================================
# Style
# =========================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": AXES_TITLE_SIZE,
    "axes.labelsize": AXES_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# =========================================================
# Helpers
# =========================================================
def season_mean(arr12, months):
    idx = np.array(months, dtype=int) - 1
    sub = arr12[idx, ...]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmean(sub, axis=0)
    valid_count = np.sum(np.isfinite(sub), axis=0)
    out[valid_count == 0] = np.nan
    return out.astype(np.float32)

def extract_ml_level(arr, level_idx=LEVEL_IDX):
    if arr.ndim == 4:
        return arr[:, level_idx, :, :]
    if arr.ndim == 3:
        if arr.shape[0] > 20:
            return arr
        return arr[level_idx, :, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected shape: {arr.shape}")

def set_geo_axes(ax, lon_min, lon_max, lat_min, lat_max, show_ylabels=True):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.45)
    ax.add_feature(cfeature.BORDERS, linewidth=0.25)

    xt0 = int(np.ceil(lon_min / 2.0) * 2)
    xt1 = int(np.floor(lon_max / 2.0) * 2)
    xt = np.arange(xt0, xt1 + 1, 2, dtype=int)
    ax.set_xticks(xt, crs=ccrs.PlateCarree())

    yt0 = int(np.ceil(lat_min / 2.0) * 2)
    yt1 = int(np.floor(lat_max / 2.0) * 2)
    yt = np.arange(yt0, yt1 + 1, 2, dtype=int)
    ax.set_yticks(yt, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter(number_format=".0f"))
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter(number_format=".0f"))
    ax.tick_params(labelsize=TICK_LABEL_SIZE, direction="out", pad=2)

    if not show_ylabels:
        ax.tick_params(axis="y", labelleft=False)
        ax.set_yticklabels([])

def add_panel_tag(ax, tag, color="black", outline=False):
    txt = ax.text(
        0.02, 0.03, tag,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=PANEL_TAG_SIZE, fontweight="bold",
        color=color, zorder=10
    )
    if outline:
        txt.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

def centers_to_edges(c1d):
    c = np.asarray(c1d, dtype=float)
    if c.size < 2:
        raise ValueError("Need at least 2 coordinates.")
    e = np.empty(c.size + 1, dtype=float)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - 0.5 * (c[1] - c[0])
    e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return e

def shared_norm():
    return Normalize(vmin=FIXED_VMIN, vmax=FIXED_VMAX)

def add_column_titles(fig, axes_top_row, titles):
    fig.canvas.draw()
    for ax, title in zip(axes_top_row, titles):
        bb = ax.get_position()
        x = 0.5 * (bb.x0 + bb.x1)
        y = bb.y1 + COL_TITLE_Y_OFFSET
        fig.text(
            x, y, title,
            ha="center", va="bottom",
            fontsize=COL_TITLE_SIZE,
            fontweight=COL_TITLE_WEIGHT
        )

def add_row_titles(fig, axes_left_col, row_names):
    fig.canvas.draw()
    for ax, row_name in zip(axes_left_col, row_names):
        bb = ax.get_position()
        x = ROW_LABEL_X
        y = 0.5 * (bb.y0 + bb.y1)
        fig.text(
            x, y, row_name,
            ha="center", va="center",
            rotation=90,
            fontsize=ROW_TITLE_SIZE,
            fontweight=ROW_TITLE_WEIGHT
        )

# =========================================================
# Load ML
# =========================================================
print("Loading ML lat/lon ...")
lat_ml = np.load(os.path.join(ML_DIR, "lat.npy"))
lon_ml = np.load(os.path.join(ML_DIR, "lon.npy"))
if lat_ml.shape != lon_ml.shape:
    raise ValueError("lat.npy and lon.npy shapes differ")

print("Loading ML monthly ave_gr.npy ...")
ml_mon = []
for mm in range(1, 13):
    fp = os.path.join(ML_DIR, f"{mm:02d}", "ave_gr.npy")
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    arr = np.load(fp)
    arr = extract_ml_level(arr, LEVEL_IDX).astype(np.float32)
    ml_mon.append(arr)
ml_mon = np.stack(ml_mon, axis=0)
ML_WARM = season_mean(ml_mon, WARM)

# =========================================================
# Load CRU
# =========================================================
print("Loading CRU ...")
with xr.open_dataset(CRU_NC) as ds:
    cru_lat = ds["lat"].values.astype(float)
    cru_lon = ds["lon"].values.astype(float)

cru_lon = np.where(cru_lon > 180, cru_lon - 360, cru_lon)

cru_mon = np.load(CRU_AVE).astype(np.float32)
if cru_mon.shape[1] == cru_lat.size:
    cru_lat_vals = cru_lat
else:
    cru_lat_vals = cru_lat[cru_lat >= 60]

CRU_WARM = season_mean(cru_mon, WARM)

# =========================================================
# Load ERA5
# =========================================================
print("Loading ERA5 ...")
with xr.open_dataset(ERA_NC) as ds:
    tname = "valid_time" if "valid_time" in ds.variables else "time"
    latn = "latitude" if "latitude" in ds.variables else "lat"
    lonn = "longitude" if "longitude" in ds.variables else "lon"
    era_da = (ds["t2m"] - 273.15).rename({tname: "time", latn: "lat", lonn: "lon"})

era_lon = era_da["lon"].values
if float(np.nanmax(era_lon)) > 180:
    era_da = era_da.assign_coords(lon=((era_da["lon"] + 180) % 360) - 180)

ERA_WARM = era_da.sel(time=era_da["time"].dt.month.isin(WARM)).mean("time", skipna=True).sortby(["lat", "lon"])

# =========================================================
# Plotters
# =========================================================
def plot_ml_panel(ax, R, field, norm, cmap_name="RdYlBu_r", tag=None, show_ylabels=True):
    lon_min, lon_max = R["lon"]
    lat_min, lat_max = R["lat"]

    set_geo_axes(ax, lon_min, lon_max, lat_min, lat_max, show_ylabels=show_ylabels)

    la = lat_ml.ravel()
    lo = lon_ml.ravel()
    vv = field.ravel()

    m = (
        np.isfinite(la) & np.isfinite(lo) & np.isfinite(vv) &
        (la >= lat_min) & (la <= lat_max) &
        (lo >= lon_min) & (lo <= lon_max)
    )

    sc = ax.scatter(
        lo[m], la[m], c=vv[m], s=2.0,
        cmap=cmap_name, norm=norm,
        transform=ccrs.PlateCarree()
    )

    if tag is not None:
        add_panel_tag(ax, tag, color="black", outline=True)

    return sc

def plot_cru_panel(ax, R, field2d, norm, cmap_name="RdYlBu_r", tag=None, show_ylabels=True):
    lon_min, lon_max = R["lon"]
    lat_min, lat_max = R["lat"]

    set_geo_axes(ax, lon_min, lon_max, lat_min, lat_max, show_ylabels=show_ylabels)

    lon_mask = (cru_lon >= lon_min - MARGIN_CRU) & (cru_lon <= lon_max + MARGIN_CRU)
    lat_mask = (cru_lat_vals >= lat_min - MARGIN_CRU) & (cru_lat_vals <= lat_max + MARGIN_CRU)

    sub_lon = cru_lon[lon_mask]
    sub_lat = cru_lat_vals[lat_mask]
    sub_val = field2d[np.ix_(lat_mask, lon_mask)]

    pm = ax.pcolormesh(
        sub_lon, sub_lat, sub_val,
        cmap=cmap_name, norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree()
    )

    if tag is not None:
        add_panel_tag(ax, tag, color="black", outline=True)

    return pm

def plot_era_panel(ax, R, da2d, norm, cmap_name="RdYlBu_r", tag=None, show_ylabels=True):
    lon_min, lon_max = R["lon"]
    lat_min, lat_max = R["lat"]

    set_geo_axes(ax, lon_min, lon_max, lat_min, lat_max, show_ylabels=show_ylabels)

    sub = da2d.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    lon_vals = sub["lon"].values
    lat_vals = sub["lat"].values
    sub_val = sub.values

    lon_e = centers_to_edges(lon_vals)
    lat_e = centers_to_edges(lat_vals)

    pm = ax.pcolormesh(
        lon_e, lat_e, sub_val,
        cmap=cmap_name, norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree()
    )

    if tag is not None:
        add_panel_tag(ax, tag, color="black", outline=True)

    return pm

# =========================================================
# Main plot
# =========================================================
def main():
    norm = shared_norm()
    cmap_name = "RdYlBu_r"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(
        nrows=4, ncols=3,
        left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP,
        wspace=WSPACE, hspace=HSPACE
    )

    panel_tags = list("abcdefghijkl")
    tag_idx = 0

    axes_grid = []

    for i, R in enumerate(REGIONS):
        row_axes = []

        show_y_ml = True
        show_y_cru = False
        show_y_era = False

        ax1 = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
        plot_ml_panel(
            ax1, R, ML_WARM, norm=norm, cmap_name=cmap_name,
            tag=panel_tags[tag_idx], show_ylabels=show_y_ml
        )
        tag_idx += 1
        row_axes.append(ax1)

        ax2 = fig.add_subplot(gs[i, 1], projection=ccrs.PlateCarree())
        plot_cru_panel(
            ax2, R, CRU_WARM, norm=norm, cmap_name=cmap_name,
            tag=panel_tags[tag_idx], show_ylabels=show_y_cru
        )
        tag_idx += 1
        row_axes.append(ax2)

        ax3 = fig.add_subplot(gs[i, 2], projection=ccrs.PlateCarree())
        plot_era_panel(
            ax3, R, ERA_WARM, norm=norm, cmap_name=cmap_name,
            tag=panel_tags[tag_idx], show_ylabels=show_y_era
        )
        tag_idx += 1
        row_axes.append(ax3)

        axes_grid.append(row_axes)

    # -----------------------------------------------------
    # Column titles and row titles
    # -----------------------------------------------------
    top_row_axes = axes_grid[0]
    left_col_axes = [row[0] for row in axes_grid]
    row_names = [R["name"] for R in REGIONS]

    add_column_titles(fig, top_row_axes, COLUMN_TITLES)
    add_row_titles(fig, left_col_axes, row_names)

    # -----------------------------------------------------
    # Shared bottom colorbar
    # -----------------------------------------------------
    cax = fig.add_axes([CB_LEFT, CB_BOTTOM, CB_WIDTH, CB_HEIGHT])
    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
    sm.set_array([])

    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="neither")
    cb.set_ticks(np.arange(-8, 9, 4))
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    cb.ax.tick_params(labelsize=CB_LABEL_SIZE)
    cb.outline.set_linewidth(0.8)

    cax.text(
        1.0 + CB_UNIT_DX, 1.0 + CB_UNIT_DY,
        CB_UNIT_TEXT,
        transform=cax.transAxes,
        ha="left", va="bottom",
        fontsize=CB_UNIT_SIZE
    )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", OUT_PNG)
    print("Saved:", OUT_PDF)

if __name__ == "__main__":
    main()
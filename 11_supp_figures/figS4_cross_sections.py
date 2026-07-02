#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import math
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

from netCDF4 import Dataset
from scipy.spatial import cKDTree
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

# =========================================================
# Paths
# =========================================================
ML_DIR      = "../data/MODIS/"
CRU_NC      = "../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc"
CRU_AVE     = "../data/CRU/grid/ave_gr.npy"
ERA_NC      = "../data/ERA5_t2m_mon_1940-2025.nc"
ETOPO_DIR   = "/data1/DATA_ARCHIVE/ETOPO2022/"

OUTDIR = "../FIG_fin"
os.makedirs(OUTDIR, exist_ok=True)

OUT_PNG = os.path.join(OUTDIR, "Figs4_warm_cross_section_4x2_redraw_swapped.png")
OUT_PDF = os.path.join(OUTDIR, "Figs4_warm_cross_section_4x2_redraw_swapped.pdf")

# =========================================================
# Fixed options
# =========================================================
WARM = [4, 5, 6, 7, 8, 9, 10]
ERA_START = "2000-01-01"
ERA_END   = "2021-12-31"
STEP_FINE = 0.001
LEVEL_IDX = 5

ALT_NORM  = Normalize(vmin=0, vmax=1400)
ALT_TICKS = np.arange(0, 1500, 200)
ALT_CMAP  = plt.get_cmap("viridis").copy()
ALT_CMAP.set_bad("white")

# =========================================================
# Style
# =========================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# =========================================================
# Region definitions
# =========================================================
regions = [
    dict(
        key="U",
        name="Ural",
        cs_type="lat",
        lat=65.0,
        lon_min=57.5,
        lon_max=63.5,
        x_plot_min=58.0,
        x_plot_max=63.0,
        xticks=np.arange(58, 64, 1),
        map_window=dict(lon=(58.0, 63.0), lat=(62.0, 67.0)),
    ),
    dict(
        key="CS",
        name="Central Siberia",
        cs_type="lat",
        lat=69.0,
        lon_min=89.0,
        lon_max=94.0,
        cru_lon_min=88.75,
        cru_lon_max=94.25,
        x_plot_min=89.0,
        x_plot_max=94.0,
        xticks=np.arange(89, 95, 1),
        map_window=dict(lon=(89.0, 94.0), lat=(65.0, 70.0)),
    ),
    dict(
        key="ES",
        name="Eastern Siberia",
        cs_type="lat",
        lat=67.0,
        lon_min=125.0,
        lon_max=130.0,
        cru_lon_min=124.75,
        cru_lon_max=130.25,
        x_plot_min=125.0,
        x_plot_max=130.0,
        xticks=np.arange(126, 131, 1),
        map_window=dict(lon=(125.0, 130.0), lat=(65.0, 70.0)),
    ),
    dict(
        key="NA",
        name="Northern Alaska",
        cs_type="lon",
        lon=-152.0,
        lat_min=63.0,
        lat_max=69.0,
        x_plot_min=63.5,
        x_plot_max=68.5,
        xticks=np.arange(64, 69, 1),
        map_window=dict(lon=(-155.0, -150.0), lat=(63.5, 68.5)),
    ),
]

# =========================================================
# Helpers
# =========================================================
def int_bounds(y_min, y_max, min_span=4.0):
    lo = math.floor(y_min)
    hi = math.ceil(y_max)
    if hi - lo < min_span:
        mid = 0.5 * (lo + hi)
        half = max(min_span / 2.0, 1.0)
        lo = math.floor(mid - half)
        hi = math.ceil(mid + half)
    return lo, hi

def _lat_str(d):
    return f"N{abs(int(d))//15*15:02d}" if d >= 0 else f"S{abs(int(d))//15*15:02d}"

def _lon_str(d):
    return f"E{abs(int(d))//15*15:03d}" if d >= 0 else f"W{abs(int(d))//15*15:03d}"

def format_lat_label(lat):
    return f"{abs(lat):.0f}°N" if lat >= 0 else f"{abs(lat):.0f}°S"

def format_lon_label(lon):
    return f"{abs(lon):.0f}°E" if lon >= 0 else f"{abs(lon):.0f}°W"

def get_etopo_tile_names(lat_range, lon_range):
    la0 = int(np.floor(min(lat_range)))
    la1 = int(np.ceil(max(lat_range)))
    lo0 = int(np.floor(min(lon_range)))
    lo1 = int(np.ceil(max(lon_range)))
    lat_vals = range(la0 // 15 * 15, la1 + 15, 15)
    lon_vals = range(lo0 // 15 * 15, lo1 + 15, 15)
    return [
        f"ETOPO_2022_v1_15s_{_lat_str(la)}{_lon_str(lo)}_surface.nc"
        for la in lat_vals for lo in lon_vals
    ]

def load_etopo_tiles(tile_dir, names):
    lat_all, lon_all, z_all = [], [], []
    for nm in names:
        p = os.path.join(tile_dir, nm)
        if not os.path.exists(p):
            continue
        with Dataset(p) as nc:
            lat = nc.variables["lat"][:]
            lon = nc.variables["lon"][:]
            lon = np.where(lon > 180, lon - 360, lon)
            z = nc.variables["z"][:].astype(np.float32)
            z[z <= 0] = np.nan
        lat_all.append(lat)
        lon_all.append(lon)
        z_all.append(z)
    return lat_all, lon_all, z_all

def build_kdtree(lat_arrs, lon_arrs, z_arrs):
    pts, vals = [], []
    for la, lo, zz in zip(lat_arrs, lon_arrs, z_arrs):
        LA, LO = np.meshgrid(la, lo, indexing="ij")
        pts.append(np.stack([LA.ravel(), LO.ravel()], axis=1))
        vals.append(zz.ravel())
    if not pts:
        return None, None
    P = np.concatenate(pts, axis=0)
    V = np.concatenate(vals, axis=0)
    return cKDTree(P), V

def sample_etopo(tree, vals, lat_vec, lon_vec):
    if tree is None or vals is None:
        return np.full(lat_vec.shape, np.nan, dtype=np.float32)
    xy = np.stack([lat_vec, lon_vec], axis=1)
    ok = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    out = np.full(lat_vec.shape[0], np.nan, dtype=np.float32)
    if np.any(ok):
        _, idx = tree.query(xy[ok], k=1)
        out[ok] = vals[idx]
    return out

def etopo_along_section(R, x_vec):
    if R["cs_type"] == "lat":
        lat_vec = np.full_like(x_vec, R["lat"], dtype=float)
        lon_vec = x_vec
        tiles = get_etopo_tile_names((R["lat"], R["lat"]), (R["lon_min"], R["lon_max"]))
    else:
        lat_vec = x_vec
        lon_vec = np.full_like(x_vec, R["lon"], dtype=float)
        tiles = get_etopo_tile_names((R["lat_min"], R["lat_max"]), (R["lon"], R["lon"]))

    la, lo, zz = load_etopo_tiles(ETOPO_DIR, tiles)
    tree, vals = build_kdtree(la, lo, zz)
    return sample_etopo(tree, vals, lat_vec, lon_vec)

def load_merged_etopo(tile_dir, names):
    lat_all, lon_all, z_all = [], [], []
    for nm in names:
        p = os.path.join(tile_dir, nm)
        if not os.path.exists(p):
            continue
        with Dataset(p) as nc:
            lat = nc.variables["lat"][:]
            lon = nc.variables["lon"][:]
            lon = np.where(lon > 180, lon - 360, lon)
            z = nc.variables["z"][:].astype(np.float32)
            z[z <= 0] = np.nan
            LA, LO = np.meshgrid(lat, lon, indexing="ij")
            lat_all.append(LA.ravel())
            lon_all.append(LO.ravel())
            z_all.append(z.ravel())
    if not lat_all:
        return None, None, None
    return np.concatenate(lat_all), np.concatenate(lon_all), np.concatenate(z_all)

def colored_line(ax, x, y, c_vals, cmap="viridis", norm=None, lw=2.0, ls="solid", alpha=0.95):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c_vals)
    x, y, c_vals = x[m], y[m], c_vals[m]
    if x.size < 2:
        return None
    ii = np.argsort(x)
    x, y, c_vals = x[ii], y[ii], c_vals[ii]
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    c_seg = 0.5 * (c_vals[:-1] + c_vals[1:])
    if norm is None:
        norm = Normalize(vmin=np.nanmin(c_seg), vmax=np.nanmax(c_seg))
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw, linestyles=ls, alpha=alpha)
    lc.set_array(c_seg)
    ax.add_collection(lc)
    return lc

# =========================================================
# Load datasets
# =========================================================
print("Loading ML lat/lon ...")
lat_ml = np.load(os.path.join(ML_DIR, "lat.npy"))
lon_ml = np.load(os.path.join(ML_DIR, "lon.npy"))

print("Loading ML monthly ave_gr.npy ...")
mave = np.full((12, lat_ml.shape[0], lat_ml.shape[1], lat_ml.shape[2]), np.nan, dtype=np.float32)
for mm in range(1, 13):
    fp = os.path.join(ML_DIR, f"{mm:02d}", "ave_gr.npy")
    if not os.path.exists(fp):
        print(f"Missing ML file: {fp}")
        continue
    arr = np.load(fp)
    if arr.ndim != 4:
        raise ValueError(f"Unexpected shape in {fp}: {arr.shape}")
    mave[mm - 1] = arr[:, LEVEL_IDX]

print("Loading CRU ...")
with xr.open_dataset(CRU_NC) as ds:
    cru_lat_all = ds["lat"].values
    cru_lon_all = ds["lon"].values
    cru_lon_all = np.where(cru_lon_all > 180, cru_lon_all - 360, cru_lon_all)

cru_mon = np.load(CRU_AVE).astype(np.float32)
if cru_mon.shape[1] == cru_lat_all.size:
    cru_lat_vals = cru_lat_all
else:
    cru_lat_vals = cru_lat_all[cru_lat_all >= 60]

print("Loading ERA5 ...")
with xr.open_dataset(ERA_NC) as ds:
    time_era = ds["valid_time"].to_index() if "valid_time" in ds.variables else ds["time"].to_index()
    lat_era  = ds["latitude"].values if "latitude" in ds.variables else ds["lat"].values
    lon_era  = ds["longitude"].values if "longitude" in ds.variables else ds["lon"].values
    t2m      = ds["t2m"].values - 273.15

lon_era = np.where(lon_era > 180, lon_era - 360, lon_era)
mask_period = (time_era >= ERA_START) & (time_era <= ERA_END)

# =========================================================
# Section extractors
# =========================================================
def ml_section(R, months):
    ml_season = np.nanmean(mave[np.array(months) - 1], axis=0)
    LAT_TOL = 0.1
    PAD = 0.5

    if R["cs_type"] == "lat":
        x = np.arange(R["lon_min"], R["lon_max"] + STEP_FINE / 2, STEP_FINE)
        tgt_lat = np.full_like(x, R["lat"], dtype=float)
        cand_mask = (
            (np.abs(lat_ml - R["lat"]) <= LAT_TOL) &
            (lon_ml >= (R["lon_min"] - PAD)) &
            (lon_ml <= (R["lon_max"] + PAD))
        )
        lat_c = lat_ml[cand_mask].ravel()
        lon_c = lon_ml[cand_mask].ravel()
        val_c = ml_season[cand_mask].ravel()
        y = np.full_like(x, np.nan, dtype=np.float32)
        if lat_c.size > 0:
            tree = cKDTree(np.column_stack([lat_c, lon_c]))
            _, idx = tree.query(np.column_stack([tgt_lat, x]), k=1)
            y = val_c[idx]
        return x, y

    x = np.arange(R["lat_min"], R["lat_max"] + STEP_FINE / 2, STEP_FINE)
    tgt_lon = np.full_like(x, R["lon"], dtype=float)
    cand_mask = (
        (np.abs(lon_ml - R["lon"]) <= LAT_TOL) &
        (lat_ml >= (R["lat_min"] - PAD)) &
        (lat_ml <= (R["lat_max"] + PAD))
    )
    lat_c = lat_ml[cand_mask].ravel()
    lon_c = lon_ml[cand_mask].ravel()
    val_c = ml_season[cand_mask].ravel()
    y = np.full_like(x, np.nan, dtype=np.float32)
    if lat_c.size > 0:
        tree = cKDTree(np.column_stack([lat_c, lon_c]))
        _, idx = tree.query(np.column_stack([x, tgt_lon]), k=1)
        y = val_c[idx]
    return x, y

def cru_section(R, months):
    if R["cs_type"] == "lat":
        lat_idx = int(np.argmin(np.abs(cru_lat_vals - R["lat"])))
        lon_min = R.get("cru_lon_min", R["lon_min"])
        lon_max = R.get("cru_lon_max", R["lon_max"])
        lon_mask = (cru_lon_all >= lon_min) & (cru_lon_all <= lon_max)
        x = cru_lon_all[lon_mask]
        y = np.nanmean(cru_mon[np.array(months) - 1, lat_idx, :], axis=0)[lon_mask]
        return x, y

    lon_idx = int(np.argmin(np.abs(cru_lon_all - R["lon"])))
    lat_mask = (cru_lat_vals >= R["lat_min"]) & (cru_lat_vals <= R["lat_max"])
    x = cru_lat_vals[lat_mask]
    y_all = np.nanmean(cru_mon[np.array(months) - 1, :, lon_idx], axis=0)
    y = y_all[lat_mask]
    return x, y

def era_section(R, months):
    time_sel = time_era[mask_period]
    t2m_sel  = t2m[mask_period, :, :]
    m_mask = time_sel.month.isin(months)
    t2m_mean = np.nanmean(t2m_sel[m_mask, :, :], axis=0)

    if R["cs_type"] == "lat":
        lat_idx = int(np.argmin(np.abs(lat_era - R["lat"])))
        lon_mask = (lon_era >= R["lon_min"]) & (lon_era <= R["lon_max"])
        x = lon_era[lon_mask]
        y = t2m_mean[lat_idx, :][lon_mask]
    else:
        lon_idx = int(np.argmin(np.abs(lon_era - R["lon"])))
        lat_mask = (lat_era >= min(R["lat_min"], R["lat_max"])) & (lat_era <= max(R["lat_min"], R["lat_max"]))
        x = lat_era[lat_mask]
        y = t2m_mean[:, lon_idx][lat_mask]

    if x.size > 1 and np.any(np.diff(x) < 0):
        ii = np.argsort(x)
        x, y = x[ii], y[ii]
    return x, y

# =========================================================
# Plot helpers
# =========================================================
def draw_section_panel(ax, R, months, tag=None, unit_on_top=False, show_legend=False):
    x_ml, y_ml = ml_section(R, months)
    x_cru, y_cru = cru_section(R, months)
    x_era, y_era = era_section(R, months)
    alt = etopo_along_section(R, x_ml)

    colored_line(ax, x_ml, y_ml, alt, cmap=ALT_CMAP, norm=ALT_NORM, lw=2.0)

    m = np.isfinite(x_era) & np.isfinite(y_era)
    if m.sum() > 1:
        ii = np.argsort(x_era[m])
        ax.plot(x_era[m][ii], y_era[m][ii], color="0.25", lw=1.7, label="ERA5")

    m = np.isfinite(x_cru) & np.isfinite(y_cru)
    if m.sum() > 1:
        ii = np.argsort(x_cru[m])
        ax.plot(x_cru[m][ii], y_cru[m][ii], color="0.45", lw=1.6, ls="--", label="CRU")

    ax.grid(True, alpha=0.25)
    ax.set_xlim(R["x_plot_min"], R["x_plot_max"])
    ax.set_xticks(R["xticks"])

    series = []
    for arr in (y_ml, y_cru, y_era):
        if arr is not None and np.any(np.isfinite(arr)):
            series.append(arr)
    if series:
        y_min = float(np.nanmin([np.nanmin(s) for s in series]))
        y_max = float(np.nanmax([np.nanmax(s) for s in series]))
        lo, hi = int_bounds(y_min, y_max, min_span=4.0)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5], integer=True))

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    if tag is not None:
        ax.text(0.02, 0.04, tag, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=17, fontweight="bold")

    if unit_on_top:
        ax.text(
            -0.045, 1.065, "[°C]",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=10.5
        )

    if show_legend:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="tab:green", lw=2.0, label="ML"),
            Line2D([0], [0], color="0.25", lw=1.8, label="ERA5"),
            Line2D([0], [0], color="0.45", lw=1.8, ls="--", label="CRU"),
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            frameon=False,
            ncol=3,
            bbox_to_anchor=(0.67, 1.14),
            columnspacing=1.0,
            handlelength=2.8
        )

def draw_section_line_on_map(ax, R, add_label=True):
    lon0, lon1 = R["map_window"]["lon"]
    lat0, lat1 = R["map_window"]["lat"]

    if R["cs_type"] == "lat":
        y = R["lat"]
        ax.plot([lon0, lon1], [y, y], transform=ccrs.PlateCarree(),
                color="0.78", lw=2.2, ls=":", zorder=10)

        if add_label:
            frac = (y - lat0) / (lat1 - lat0)
            frac = min(max(frac, 0.04), 0.96)
            ax.text(
                -0.06, frac, format_lat_label(y),
                transform=ax.transAxes,
                ha="right", va="center",
                fontsize=10, color="0.35",
                zorder=11, clip_on=False
            )

    else:
        x = R["lon"]
        ax.plot([x, x], [lat0, lat1], transform=ccrs.PlateCarree(),
                color="0.78", lw=2.2, ls=":", zorder=10)

        if add_label:
            frac = (x - lon0) / (lon1 - lon0)
            frac = min(max(frac, 0.04), 0.96)
            ax.text(
                frac, 1.02,
                format_lon_label(x),
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=10, color="0.35",
                zorder=11, clip_on=False
            )

def draw_map_panel(ax, R, tag=None):
    lon_min, lon_max = R["map_window"]["lon"]
    lat_min, lat_max = R["map_window"]["lat"]

    tiles = get_etopo_tile_names((lat_min, lat_max), (lon_min, lon_max))
    la, lo, zz = load_merged_etopo(ETOPO_DIR, tiles)
    if la is None:
        ax.text(0.5, 0.5, "No ETOPO data", transform=ax.transAxes,
                ha="center", va="center")
        return None

    m = (
        (la >= lat_min) & (la <= lat_max) &
        (lo >= lon_min) & (lo <= lon_max)
    )
    if np.sum(m) == 0:
        ax.text(0.5, 0.5, "No points in window", transform=ax.transAxes,
                ha="center", va="center")
        return None

    ax.set_extent([lon_min + 0.01, lon_max - 0.01, lat_min + 0.01, lat_max - 0.01],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.45)
    ax.add_feature(cfeature.BORDERS, linewidth=0.25)

    if R["key"] == "CS":
        xticks = [90, 92, 94]
        yticks = [66, 68, 70]
    elif R["key"] == "ES":
        xticks = [126, 128, 130]
        yticks = [66, 68, 70]
    elif R["key"] == "NA":
        xticks = [-154, -152, -150]
        yticks = [64, 66, 68]
    else:
        xticks = np.arange(np.ceil(lon_min), np.floor(lon_max) + 0.1, 2)
        if R["key"] == "NA":
            yticks = [64, 66, 68]
        else:
            yticks = np.arange(np.ceil(lat_min), np.floor(lat_max) + 0.1, 2)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(labelsize=11, direction="out")

    sc = ax.scatter(lo[m], la[m], c=zz[m], cmap=ALT_CMAP, norm=ALT_NORM,
                    s=1, linewidths=0, transform=ccrs.PlateCarree(), zorder=1)

    draw_section_line_on_map(ax, R, add_label=True)

    if tag is not None:
        txt_color = "w"
        effects = None
        if tag in ["a", "b", "c", "d"]:
            txt_color = "black"
            effects = [pe.withStroke(linewidth=2.0, foreground="white")]

        t = ax.text(0.02, 0.04, tag, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=17, fontweight="bold",
                    color=txt_color)
        if effects is not None:
            t.set_path_effects(effects)

    return sc

# =========================================================
# Main
# =========================================================
def main():
    months = WARM

    fig = plt.figure(figsize=(10.8, 14.2))
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        width_ratios=[0.92, 1.34],
        hspace=0.24,
        wspace=0.025
    )

    map_tags = list("abcd")
    sec_tags = list("efgh")

    ts_axes = []
    map_axes = []
    last_sc = None

    for i, R in enumerate(regions):
        ax_mp = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())

        pos = ax_mp.get_position()
        ax_mp.set_position([pos.x0 + 0.01, pos.y0, pos.width, pos.height])

        ax_ts = fig.add_subplot(gs[i, 1])

        last_sc = draw_map_panel(ax_mp, R, tag=map_tags[i])

        draw_section_panel(
            ax_ts, R, months,
            tag=sec_tags[i],
            unit_on_top=(i == 0),
            show_legend=(i == 0)
        )

        # region name | map | timeseries
        pos_mp = ax_mp.get_position()
        fig.text(
            pos_mp.x0 - 0.055,
            0.5 * (pos_mp.y0 + pos_mp.y1),
            R["name"],
            rotation=90,
            ha="center",
            va="center",
            fontsize=13
        )

        map_axes.append(ax_mp)
        ts_axes.append(ax_ts)
        ax_mp.tick_params(axis="x", labelbottom=True)

    # Longitude label for ES section panel
    pos_es = ts_axes[2].get_position()
    fig.text(
        0.5 * (pos_es.x0 + pos_es.x1),
        pos_es.y0 - 0.020,
        "Longitude [°E]",
        ha="center", va="top", fontsize=12
    )

    # Latitude label for NA section panel
    pos_na = ts_axes[3].get_position()
    fig.text(
        0.5 * (pos_na.x0 + pos_na.x1),
        pos_na.y0 - 0.022,
        "Latitude [°N]",
        ha="center", va="top", fontsize=12
    )

    # Colorbar on far right
    if last_sc is not None:
        pos_r2 = map_axes[1].get_position()
        pos_r3 = map_axes[2].get_position()

        cax_y = pos_r3.y0
        cax_h = pos_r2.y1 - pos_r3.y0

        cax = fig.add_axes([0.92, cax_y, 0.015, cax_h])
        cb = fig.colorbar(last_sc, cax=cax, orientation="vertical", ticks=ALT_TICKS)
        cb.ax.tick_params(labelsize=11)
        cb.ax.text(
            0.5, 1.005, "[m]",
            transform=cb.ax.transAxes,
            ha="center", va="bottom", fontsize=12
        )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", OUT_PNG)
    print("Saved:", OUT_PDF)

if __name__ == "__main__":
    main()
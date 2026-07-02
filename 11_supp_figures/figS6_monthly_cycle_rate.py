#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from scipy.spatial import cKDTree
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import MultipleLocator

# =========================================================
# User paths
# =========================================================
TIME_START = "2000-03-01"
TIME_END   = "2022-04-30"

ML_DIR     = "../data/MODIS/"
CRU_NC     = "../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc"
ERA_NC     = "../data/ERA5_t2m_mon_1940-2025.nc"
ETOPO_DIR  = "/data1/DATA_ARCHIVE/ETOPO2022/"
OUTDIR     = "../FIG_fin"
os.makedirs(OUTDIR, exist_ok=True)

OUT_PNG = os.path.join(OUTDIR, "FigS6_monthly_cycle_rate_of_change_4x2_faster.png")
OUT_PDF = os.path.join(OUTDIR, "FigS6_monthly_cycle_rate_of_change_4x2_faster.pdf")

# =========================================================
# Regions
# =========================================================
REGIONS = [
    dict(
        key="U",
        name="Ural",
        coord_label="[65°N, 59.75–60.25°E]",
        kind="lat",
        lat=65.0,
        x_start=59.75,
        x_end=60.25,
        x_step=0.01,
    ),
    dict(
        key="CS",
        name="Central Siberia",
        coord_label="[69°N, 91.75–92.25°E]",
        kind="lat",
        lat=69.0,
        x_start=91.75,
        x_end=92.25,
        x_step=0.01,
    ),
    dict(
        key="ES",
        name="Eastern Siberia",
        coord_label="[67°N, 127.75–128.25°E]",
        kind="lat",
        lat=67.0,
        x_start=127.75,
        x_end=128.25,
        x_step=0.01,
    ),
    dict(
        key="NA",
        name="Northern Alaska",
        coord_label="[67.75–68.25°N, 152°W]",
        kind="lon",
        lon=-152.0,
        x_start=67.75,
        x_end=68.25,
        x_step=0.01,
    ),
]

# =========================================================
# Plot styling
# =========================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COL_ERA = "0.25"
COL_CRU = "0.45"
ERA_LW = 1.0
CRU_LW = 1.0
ML_LW  = 1.1
ALPHA_ERA = 0.35
ALPHA_CRU = 0.35
ALPHA_ML  = 0.75

CMAP = get_cmap("viridis")
ALT_NORM = Normalize(vmin=0, vmax=1400)

MON_NUMS = np.arange(1, 13)
MON_LABS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

LEVEL_IDX = 5
ML_TREND_IS_PER_YEAR = True
ML_RADIUS_DEG = 0.005

# =========================================================
# Helpers
# =========================================================
def nearest_index(coord_vals, target):
    return int(np.nanargmin(np.abs(np.asarray(coord_vals) - target)))

def nearest_indices(coord_vals, targets):
    coord_vals = np.asarray(coord_vals)
    return np.array([nearest_index(coord_vals, t) for t in np.asarray(targets)], dtype=int)

def lon_to_0360_if_needed(lon_vals, query_lon):
    return query_lon % 360 if np.nanmax(lon_vals) > 180 else query_lon

def lons_to_0360_if_needed(lon_vals, query_lons):
    return np.mod(query_lons, 360.0) if np.nanmax(lon_vals) > 180 else np.asarray(query_lons)

def _extract_level(arr, level_idx):
    if arr.ndim == 4:
        return arr[:, level_idx, :, :]
    if arr.ndim == 3:
        if arr.shape[0] <= 12:
            return arr[level_idx, :, :]
        return arr
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected array shape: {arr.shape}")

def decimal_years_from_datetimeindex(times):
    return (times.year.values + (times.dayofyear.values - 1) / 365.25).astype(float)

def compute_monthly_climatology_and_trend_full(data3d, times):
    """
    data3d: (time, lat, lon)
    returns:
        clim12  : (12, lat, lon)
        trend12 : (12, lat, lon)  [°C/decade]
    """
    months = times.month.values
    years = decimal_years_from_datetimeindex(times)

    nt, ny, nx = data3d.shape
    clim12 = np.full((12, ny, nx), np.nan, dtype=np.float32)
    trend12 = np.full((12, ny, nx), np.nan, dtype=np.float32)

    for m in range(1, 13):
        mm = (months == m)
        if mm.sum() == 0:
            continue

        Ym = data3d[mm, :, :]
        clim12[m - 1] = np.nanmean(Ym, axis=0).astype(np.float32)

        t = years[mm]
        if t.size < 3:
            continue

        t0 = t - t.mean()
        sxx = np.sum(t0 ** 2)
        if not np.isfinite(sxx) or sxx <= 0:
            continue

        ymean = np.nanmean(Ym, axis=0)
        num = np.nansum(t0[:, None, None] * (Ym - ymean[None, :, :]), axis=0)
        slope_year = num / sxx
        trend12[m - 1] = (slope_year * 10.0).astype(np.float32)

    return clim12, trend12

def sample_transect_from_monthly_fields(monthly_fields, lat_vals, lon_vals, region):
    """
    monthly_fields: (12, lat, lon)
    return: (npoint, 12)
    """
    Xs = np.round(np.arange(region["x_start"], region["x_end"] + 1e-9, region["x_step"]), 5)

    if region["kind"] == "lat":
        iy = nearest_index(lat_vals, region["lat"])
        qlons = lons_to_0360_if_needed(lon_vals, Xs)
        ix = nearest_indices(lon_vals, qlons)
        out = monthly_fields[:, iy, ix].T
    else:
        ix = nearest_index(lon_vals, lon_to_0360_if_needed(lon_vals, region["lon"]))
        iy = nearest_indices(lat_vals, Xs)
        out = monthly_fields[:, iy, ix].T

    return Xs, out.astype(np.float32)

# =========================================================
# ETOPO helpers
# =========================================================
def _lat_str(d):
    return f"N{abs(int(d))//15*15:02d}" if d >= 0 else f"S{abs(int(d))//15*15:02d}"

def _lon_str(d):
    return f"E{abs(int(d))//15*15:03d}" if d >= 0 else f"W{abs(int(d))//15*15:03d}"

def get_etopo_tile_names(lat_range, lon_range):
    lat_vals = range(int(lat_range[0]) // 15 * 15, int(lat_range[1]) + 15, 15)
    lon_vals = range(int(lon_range[0]) // 15 * 15, int(lon_range[1]) + 15, 15)
    return [
        f"ETOPO_2022_v1_15s_{_lat_str(la)}{_lon_str(lo)}_surface.nc"
        for la in lat_vals for lo in lon_vals
    ]

def load_etopo_tiles(tile_dir, names):
    lat_all, lon_all, z_all = [], [], []
    for nm in names:
        p = os.path.join(tile_dir, nm)
        if not os.path.exists(p):
            print(f"Missing ETOPO tile: {p}")
            continue
        with Dataset(p) as nc:
            lat = nc.variables["lat"][:]
            lon = nc.variables["lon"][:]
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
    P = np.concatenate(pts, axis=0)
    V = np.concatenate(vals, axis=0)
    return cKDTree(P), V

def sample_etopo(tree, vals, lat_vec, lon_vec):
    xy = np.stack([lat_vec, lon_vec], axis=1)
    ok = np.all(np.isfinite(xy), axis=1)
    out = np.full(lat_vec.shape[0], np.nan, dtype=np.float32)
    if np.any(ok):
        _, idx = tree.query(xy[ok])
        out[ok] = vals[idx]
    return out

# =========================================================
# Load ML
# =========================================================
print("Loading ML lat/lon ...")
lat_ml = np.load(os.path.join(ML_DIR, "lat.npy"))
lon_ml = np.load(os.path.join(ML_DIR, "lon.npy"))
if lat_ml.shape != lon_ml.shape:
    raise ValueError("ML lat/lon shape mismatch")

print("Loading ML monthly mean/trend grids ...")
ml_mean_mon = [None] * 12
ml_trend_mon = [None] * 12

for m in range(1, 13):
    fp_mean = os.path.join(ML_DIR, f"{m:02d}", "ave_gr.npy")
    fp_tr   = os.path.join(ML_DIR, f"{m:02d}", "trend_gr.npy")

    arr_mean = _extract_level(np.load(fp_mean), LEVEL_IDX).astype(np.float32)
    arr_tr   = _extract_level(np.load(fp_tr), LEVEL_IDX).astype(np.float32)

    if ML_TREND_IS_PER_YEAR:
        arr_tr *= 10.0

    ml_mean_mon[m - 1] = arr_mean
    ml_trend_mon[m - 1] = arr_tr

latv = lat_ml.ravel()
lonv = lon_ml.ravel()

def build_ml_transect_masks(region, radius_deg=ML_RADIUS_DEG):
    Xs = np.round(np.arange(region["x_start"], region["x_end"] + 1e-9, region["x_step"]), 5)
    masks = []

    for x in Xs:
        if region["kind"] == "lat":
            lat0, lon0 = region["lat"], x
        else:
            lat0, lon0 = x, region["lon"]

        mask = (np.abs(latv - lat0) <= radius_deg) & (np.abs(lonv - lon0) <= radius_deg)
        masks.append(np.where(mask)[0])

    return Xs, masks

def sample_ml_from_masks(monthly_fields_list, masks):
    """
    monthly_fields_list: list length 12, each flattened later
    returns: (npoint, 12)
    """
    npoint = len(masks)
    out = np.full((npoint, 12), np.nan, dtype=np.float32)

    flat_months = [fld.ravel() for fld in monthly_fields_list]

    for i, idx in enumerate(masks):
        if idx.size == 0:
            continue
        for m in range(12):
            vals = flat_months[m][idx]
            if vals.size > 0:
                out[i, m] = np.nanmean(vals)

    return out

# =========================================================
# Load ERA / CRU once, then precompute 12 monthly fields
# =========================================================
print("Loading ERA5 once ...")
with xr.open_dataset(ERA_NC) as ds:
    tname = "valid_time" if "valid_time" in ds.variables else "time"
    latn  = "latitude" if "latitude" in ds.coords else "lat"
    lonn  = "longitude" if "longitude" in ds.coords else "lon"

    era_da = (ds["t2m"] - 273.15).sel({tname: slice(TIME_START, TIME_END)})
    era_da = era_da.rename({tname: "time", latn: "lat", lonn: "lon"})
    era_times = era_da["time"].to_index()
    era_lat = era_da["lat"].values
    era_lon = era_da["lon"].values
    era_data = era_da.load().values.astype(np.float32)

print("Computing ERA5 monthly climatology/trend fields ...")
ERA_CLIM12, ERA_TREND12 = compute_monthly_climatology_and_trend_full(era_data, era_times)

print("Loading CRU once ...")
with xr.open_dataset(CRU_NC) as ds:
    var = [v for v in ds.data_vars if v not in ("lat", "lon")][0]
    cru_da = ds[var].sel(time=slice(TIME_START, TIME_END))
    cru_times = cru_da["time"].to_index()
    cru_lat = ds["lat"].values
    cru_lon = ds["lon"].values
    cru_data = cru_da.load().values.astype(np.float32)

print("Computing CRU monthly climatology/trend fields ...")
CRU_CLIM12, CRU_TREND12 = compute_monthly_climatology_and_trend_full(cru_data, cru_times)

# =========================================================
# Compute per region
# =========================================================
def compute_region_series(region):
    Xs = np.round(np.arange(region["x_start"], region["x_end"] + 1e-9, region["x_step"]), 5)

    if region["kind"] == "lat":
        lat_fixed = region["lat"]
        tiles = get_etopo_tile_names((lat_fixed, lat_fixed), (Xs.min(), Xs.max()))
        la, lo, zz = load_etopo_tiles(ETOPO_DIR, tiles)
        tree_z, zvals = build_kdtree(la, lo, zz)
        altitudes = sample_etopo(tree_z, zvals, np.full_like(Xs, lat_fixed), Xs)
    else:
        lon_fixed = region["lon"]
        tiles = get_etopo_tile_names((Xs.min(), Xs.max()), (lon_fixed, lon_fixed))
        la, lo, zz = load_etopo_tiles(ETOPO_DIR, tiles)
        tree_z, zvals = build_kdtree(la, lo, zz)
        altitudes = sample_etopo(tree_z, zvals, Xs, np.full_like(Xs, lon_fixed))

    if not np.any(np.isfinite(altitudes)):
        altitudes = np.zeros_like(Xs)

    _, masks = build_ml_transect_masks(region, radius_deg=ML_RADIUS_DEG)
    ml_cycle = sample_ml_from_masks(ml_mean_mon, masks)
    ml_trend = sample_ml_from_masks(ml_trend_mon, masks)

    _, era_cycle = sample_transect_from_monthly_fields(ERA_CLIM12, era_lat, era_lon, region)
    _, era_trend = sample_transect_from_monthly_fields(ERA_TREND12, era_lat, era_lon, region)

    _, cru_cycle = sample_transect_from_monthly_fields(CRU_CLIM12, cru_lat, cru_lon, region)
    _, cru_trend = sample_transect_from_monthly_fields(CRU_TREND12, cru_lat, cru_lon, region)

    return {
        "x": Xs,
        "alt": altitudes,
        "ml_cycle": ml_cycle,
        "era_cycle": era_cycle,
        "cru_cycle": cru_cycle,
        "ml_trend": ml_trend,
        "era_trend": era_trend,
        "cru_trend": cru_trend,
    }

# =========================================================
# Plot helpers
# =========================================================
def plot_region_panel(ax, y_ml, y_era, y_cru, altitudes, panel_tag=None,
                      show_xticklabels=False, ylim=None):
    for i in range(y_ml.shape[0]):
        col = CMAP(ALT_NORM(altitudes[i] if np.isfinite(altitudes[i]) else 0.0))
        ax.plot(MON_NUMS, y_ml[i], color=col, lw=ML_LW, alpha=ALPHA_ML)

    for i in range(y_era.shape[0]):
        ax.plot(MON_NUMS, y_era[i], color=COL_ERA, lw=ERA_LW, alpha=ALPHA_ERA)
    for i in range(y_cru.shape[0]):
        ax.plot(MON_NUMS, y_cru[i], color=COL_CRU, lw=CRU_LW, alpha=ALPHA_CRU, linestyle="--")

    ax.set_xlim(1, 12)
    ax.set_xticks(MON_NUMS)
    ax.set_xticklabels(MON_LABS if show_xticklabels else [])

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.20)

    if panel_tag is not None:
        ax.text(0.01, 0.98, panel_tag, transform=ax.transAxes,
                ha="left", va="top", fontsize=12, fontweight="bold")

def add_row_title(fig, ax, region_name, coord_label):
    pos = ax.get_position()
    x = pos.x0 + 0.005
    y = pos.y1 + 0.002

    fig.text(
        x, y,
        f"{region_name} {coord_label}",
        ha="left", va="bottom",
        fontsize=11, fontweight="bold"
    )

# =========================================================
# Main
# =========================================================
def main():
    print("Computing regional series ...")
    data = [compute_region_series(r) for r in REGIONS]

    fig = plt.figure(figsize=(11.5, 11.8))
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        left=0.08, right=0.90, bottom=0.08, top=0.95,
        wspace=0.08, hspace=0.10
    )

    panel_tags = list("abcdefgh")
    left_axes = []
    right_axes = []

    for i, (R, D) in enumerate(zip(REGIONS, data)):
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])

        left_axes.append(ax1)
        right_axes.append(ax2)

        show_xt = (i == 3)

        plot_region_panel(
            ax1, D["ml_cycle"], D["era_cycle"], D["cru_cycle"], D["alt"],
            panel_tag=panel_tags[2 * i], show_xticklabels=show_xt
        )

        plot_region_panel(
            ax2, D["ml_trend"], D["era_trend"], D["cru_trend"], D["alt"],
            panel_tag=panel_tags[2 * i + 1], show_xticklabels=show_xt
        )

        add_row_title(fig, ax1, R["name"], R["coord_label"])

        # custom y-tick intervals
        if i == 0:   # a, b
            ax1.yaxis.set_major_locator(MultipleLocator(10))
            ax2.yaxis.set_major_locator(MultipleLocator(1))

        if i == 1:   # d
            ax2.yaxis.set_major_locator(MultipleLocator(1))

        if i == 3:   # g
            ax1.yaxis.set_major_locator(MultipleLocator(10))

    pos_l = left_axes[0].get_position()
    pos_r = right_axes[0].get_position()

    fig.text(
        0.5 * (pos_l.x0 + pos_l.x1), 0.972,
        "Seasonal cycle [°C]",
        ha="center", va="bottom", fontsize=14, fontweight="bold"
    )
    fig.text(
        0.5 * (pos_r.x0 + pos_r.x1), 0.972,
        "Rate of change [°C 10-yr$^{-1}$]",
        ha="center", va="bottom", fontsize=14, fontweight="bold"
    )

    fig.text(0.46, 0.035, "Month", ha="center", va="center", fontsize=16)

    handles = [
        Line2D([0], [0], color=CMAP(0.65), lw=1.8, label="ML"),
        Line2D([0], [0], color=COL_ERA, lw=1.6, label="ERA5"),
        Line2D([0], [0], color=COL_CRU, lw=1.6, linestyle="--", label="CRU"),
    ]
    fig.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.905, 0.028),
        frameon=False,
        ncol=3,
        columnspacing=1.0,
        handlelength=2.8,
        borderaxespad=0.0
    )

    pos_r2 = right_axes[1].get_position()
    pos_r3 = right_axes[2].get_position()

    cax = fig.add_axes([0.915, pos_r3.y0, 0.012, pos_r2.y1 - pos_r3.y0])
    cb = ColorbarBase(cax, cmap=CMAP, norm=ALT_NORM, orientation="vertical")
    cb.ax.tick_params(labelsize=8)
    cb.ax.text(0.5, 1.02, "[m]", transform=cb.ax.transAxes,
               ha="center", va="bottom", fontsize=8)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", OUT_PNG)
    print("Saved:", OUT_PDF)

if __name__ == "__main__":
    main()
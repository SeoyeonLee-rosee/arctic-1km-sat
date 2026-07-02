#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Summer (Apr–Oct): ML − CRU (CRU interpolated to ML 1 km grid)  2×4 panels

Top row  : mean difference   -> a, b, c, d
Bottom row: trend difference -> e, f, g, h

Panels:
  a/e : U
  b/f : CS
  c/g : ES
  d/h : NA

- Mean  : ML summer mean  − CRU summer mean interpolated to ML grid
- Trend : ML summer trend − CRU summer trend interpolated to ML grid
- ML: tile-wise pcolormesh if rectilinear; otherwise scatter fallback
- CRU: interpolated onto ML points with bilinear interpolation
- Station overlay (NAME.csv): ▲ only
- ETOPO >=500 m overlay as dots on same plane, after diff draw
- Shared colorbars:
    * one for top row (mean)
    * one for bottom row (rate of change)

Outputs:
  ../FIG_fin/SUMMER_MLminusCRU_mean_trend_2x4.png
  ../FIG_fin/SUMMER_MLminusCRU_mean_trend_2x4.pdf
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


# --------------------------
# Paths
# --------------------------
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/ 기준

ML_DIR      = os.path.join(SCRIPT_DIR, "..", "data", "MODIS")
CRU_NC      = os.path.join(SCRIPT_DIR, "..", "data", "CRU", "grid", "cru_ts4.06.1901.2021.tmp.dat.nc")
CRU_AVE     = os.path.join(SCRIPT_DIR, "..", "data", "CRU", "grid", "ave_gr.npy")
CRU_TREND   = os.path.join(SCRIPT_DIR, "..", "data", "CRU", "grid", "trend.npy")
ETOPO_DIR   = "/data1/DATA_ARCHIVE/ETOPO2022/"

OUTDIR  = os.path.join(SCRIPT_DIR, "..", "FIG_fin")
os.makedirs(OUTDIR, exist_ok=True)
OUT_PNG = os.path.join(OUTDIR, "SUMMER_MLminusCRU_mean_trend_2x4.png")
OUT_PDF = os.path.join(OUTDIR, "SUMMER_MLminusCRU_mean_trend_2x4.pdf")


# --------------------------
# Style
# --------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# --------------------------
# Fixed settings
# --------------------------
WARM_MONTHS = [4, 5, 6, 7, 8, 9, 10]   # Apr–Oct
LEVEL_IDX   = 5
CRU_LATMIN  = 60.0

# unit conversion for trend (if ML trend is per year and you want per decade)
PER_YEAR_TO_PER_DECADE = 10.0

# mean diff colormap
MEAN_VMIN, MEAN_VMAX = -4.0, 4.0
MEAN_TICKS = [-4, -2, 0, 2, 4]
MEAN_CMAP  = "bwr"

# trend diff colormap
TREND_VMIN, TREND_VMAX = -0.1, 0.1
TREND_TICKS = [-0.1, -0.05, 0, 0.05, 0.1]
TREND_CMAP  = "bwr"

# station style (▲ only)
STA_TRI_MARKER = "^"
STA_TRI_SIZE   = 120
STA_TRI_COLOR  = "k"
STA_TRI_ALPHA  = 0.95

# scatter fallback for non-rect tiles
SCATTER_S = 1.0
RASTERIZE_SCATTER = True

# topo dots
TOPO_ON         = True
TOPO_PAD_DEG    = 0.25
TOPO_MIN_M      = 500.0
TOPO_GRID_DEG   = 0.25
TOPO_DOTS_SIZE  = 15.0
TOPO_DOTS_ALPHA = 0.60
TOPO_DOTS_COLOR = "0.10"

# panels
windows = [
    {"key": "U",  "name": "U",  "lon": (58, 63),         "lat": (62, 67)},
    {"key": "CS", "name": "CS", "lon": (89, 94),         "lat": (65, 70)},
    {"key": "ES", "name": "ES", "lon": (125, 130),   "lat": (65, 70)},
    {"key": "NA", "name": "NA", "lon": (-155, -150),     "lat": (63.5, 68.5)},
]

PANEL_TAGS_TOP    = ["a", "b", "c", "d"]
PANEL_TAGS_BOTTOM = ["e", "f", "g", "h"]


# --------------------------
# Helpers
# --------------------------
def np_load_robust(path):
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return np.load(path, allow_pickle=True)

def _extract_level(arr, level_idx=LEVEL_IDX):
    if arr.ndim == 4:      # (tile, level, y, x)
        return arr[:, level_idx, :, :]
    if arr.ndim == 3:
        if arr.shape[0] >= 20:  # (tile,y,x)
            return arr
        return arr[level_idx, :, :]  # (level,y,x)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected array shape: {arr.shape}")

def load_ml_latlon():
    lat = np_load_robust(os.path.join(ML_DIR, "lat.npy")).astype(np.float32)
    lon = np_load_robust(os.path.join(ML_DIR, "lon.npy")).astype(np.float32)
    if lat.shape != lon.shape:
        raise ValueError(f"ML lat/lon shape mismatch: {lat.shape} vs {lon.shape}")
    if lat.ndim != 3:
        raise ValueError(f"Expected ML lat/lon shape (tile,y,x), got {lat.shape}")
    return lat, lon

def load_ml_month(month, fname, ref_shape):
    fp = os.path.join(ML_DIR, f"{month:02d}", fname)
    if not os.path.exists(fp):
        return None
    arr = np_load_robust(fp)
    out = _extract_level(arr, LEVEL_IDX).astype(np.float32)
    if out.shape != ref_shape:
        raise ValueError(f"ML {fname} {month:02d} shape {out.shape} != {ref_shape}")
    return out

def build_ml_summer_mean():
    lat_ml, lon_ml = load_ml_latlon()
    ref_shape = lat_ml.shape  # (tile,y,x)

    good = []
    used_months = []
    for m in WARM_MONTHS:
        a = load_ml_month(m, "ave_gr.npy", ref_shape)
        if a is None:
            print(f"  ⚠ ML mean month {m:02d} missing -> skip")
            continue
        nfin = int(np.isfinite(a).sum())
        if nfin < 1000:
            print(f"  ⚠ ML mean month {m:02d} mostly NaN (finite={nfin}) -> skip")
            continue
        good.append(a)
        used_months.append(m)

    if len(good) == 0:
        raise RuntimeError("No valid ML warm-month mean fields found.")

    ml_summer = np.nanmean(np.stack(good, axis=0), axis=0).astype(np.float32)
    print(f"  ML mean warm months used: {used_months}")
    return ml_summer, lat_ml, lon_ml

def build_ml_summer_trend():
    lat_ml, lon_ml = load_ml_latlon()
    ref_shape = lat_ml.shape  # (tile,y,x)

    good = []
    used_months = []
    for m in WARM_MONTHS:
        a = load_ml_month(m, "trend_gr.npy", ref_shape)
        if a is None:
            print(f"  ⚠ ML trend month {m:02d} missing -> skip")
            continue
        nfin = int(np.isfinite(a).sum())
        if nfin < 1000:
            print(f"  ⚠ ML trend month {m:02d} mostly NaN (finite={nfin}) -> skip")
            continue
        good.append(a)
        used_months.append(m)

    if len(good) == 0:
        raise RuntimeError("No valid ML warm-month trend fields found.")

    # convert to per decade if original is per year
    ml_trend = np.nanmean(np.stack(good, axis=0), axis=0).astype(np.float32) #* PER_YEAR_TO_PER_DECADE
    print(f"  ML trend warm months used: {used_months}")
    return ml_trend, lat_ml, lon_ml

def load_cru_latlon():
    with xr.open_dataset(CRU_NC) as ds:
        lat_all = ds["lat"].values.astype(np.float64)
        lon_all = ds["lon"].values.astype(np.float64)

    lon_all = np.where(lon_all > 180, lon_all - 360, lon_all)
    latmask = (lat_all >= CRU_LATMIN)
    lat = lat_all[latmask].astype(np.float64)
    lon = lon_all.astype(np.float64)
    return lat, lon, latmask

def load_cru_summer_mean():
    lat, lon, latmask = load_cru_latlon()

    arr12 = np_load_robust(CRU_AVE).astype(np.float32)  # (12, lat, lon)
    if arr12.shape[1] == latmask.size:
        arr12 = arr12[:, latmask, :]

    idx = [m - 1 for m in WARM_MONTHS]
    good = []
    used = []
    for ii in idx:
        a = arr12[ii, :, :]
        if np.isfinite(a).sum() < 1000:
            continue
        good.append(a)
        used.append(ii + 1)

    if len(good) == 0:
        raise RuntimeError("No valid CRU warm-month mean slices found.")

    cru_summer = np.nanmean(np.stack(good, axis=0), axis=0).astype(np.float32)

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        cru_summer = cru_summer[::-1, :]

    order = np.argsort(lon)
    lon = lon[order]
    cru_summer = cru_summer[:, order]

    print(f"  CRU mean warm months used: {used}")
    return cru_summer, lat, lon

def load_cru_summer_trend():
    lat, lon, latmask = load_cru_latlon()

    arr12 = np_load_robust(CRU_TREND).astype(np.float32)  # expected (12, lat, lon)
    if arr12.shape[1] == latmask.size:
        arr12 = arr12[:, latmask, :]

    idx = [m - 1 for m in WARM_MONTHS]
    good = []
    used = []
    for ii in idx:
        a = arr12[ii, :, :]
        if np.isfinite(a).sum() < 1000:
            continue
        good.append(a)
        used.append(ii + 1)

    if len(good) == 0:
        raise RuntimeError("No valid CRU warm-month trend slices found.")

    cru_trend = np.nanmean(np.stack(good, axis=0), axis=0).astype(np.float32)

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        cru_trend = cru_trend[::-1, :]

    order = np.argsort(lon)
    lon = lon[order]
    cru_trend = cru_trend[:, order]

    print(f"  CRU trend warm months used: {used}")
    return cru_trend, lat, lon

def make_cru_interpolator(cru2d, lat1d, lon1d):
    cru = cru2d.copy()
    if np.any(~np.isfinite(cru)):
        m = np.isfinite(cru)
        if m.sum() > 0:
            yy, xx = np.where(m)
            tree = cKDTree(np.c_[yy, xx])
            y0, x0 = np.where(~m)
            _, idx = tree.query(np.c_[y0, x0], k=1)
            cru[y0, x0] = cru[yy[idx], xx[idx]]
        else:
            cru[:] = 0.0

    return RegularGridInterpolator(
        (lat1d, lon1d),
        cru.astype(np.float64),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

def tile_rectilinear_axes(lat2d, lon2d, tol=5e-4):
    m = np.isfinite(lat2d) & np.isfinite(lon2d)
    if m.sum() < 0.5 * lat2d.size:
        return None, None, False
    lat1 = np.nanmean(lat2d, axis=1)
    lon1 = np.nanmean(lon2d, axis=0)
    dlat = np.nanmax(np.abs(lat2d - lat1[:, None]))
    dlon = np.nanmax(np.abs(lon2d - lon1[None, :]))
    if np.isfinite(dlat) and np.isfinite(dlon) and dlat < tol and dlon < tol:
        return lat1.astype(np.float64), lon1.astype(np.float64), True
    return None, None, False

def centers_to_edges_1d(c):
    c = np.asarray(c, dtype=np.float64)
    if c.size < 2 or not np.all(np.isfinite(c)):
        return None
    e = np.empty(c.size + 1, dtype=np.float64)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - 0.5 * (c[1] - c[0])
    e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return e

def set_ticks_2deg(ax, win, step=2):
    lon0, lon1 = win["lon"]
    lat0, lat1 = win["lat"]

    xt = np.arange(np.ceil(lon0 / step) * step, np.floor(lon1 / step) * step + 0.1, step).astype(int)
    yt = np.arange(np.ceil(lat0 / step) * step, np.floor(lat1 / step) * step + 0.1, step).astype(int)

    ax.set_xticks(xt, crs=ccrs.PlateCarree())
    ax.set_yticks(yt, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(direction="out")
    ax.gridlines(draw_labels=False, linewidth=0.35, color="gray", alpha=0.45, linestyle="--")

def add_panel_tag(ax, tag, region_name):
    ax.text(
        0.02, 0.98, f"{tag}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13, fontweight="bold",
        color="k",
        zorder=20
    )
    ax.text(
        0.50, 1.02, region_name,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=11, fontweight="bold",
        color="k",
        zorder=20
    )


# --------------------------
# ETOPO KDTree
# --------------------------
def _lat_str(d):
    return f"N{abs(int(d))//15*15:02d}" if d >= 0 else f"S{abs(int(d))//15*15:02d}"

def _lon_str(d):
    return f"E{abs(int(d))//15*15:03d}" if d >= 0 else f"W{abs(int(d))//15*15:03d}"

def get_etopo_tile_names(lat_range, lon_range):
    lat_vals = range(int(np.floor(lat_range[0]/15))*15, int(np.ceil(lat_range[1]/15))*15 + 15, 15)
    lon_vals = range(int(np.floor(lon_range[0]/15))*15, int(np.ceil(lon_range[1]/15))*15 + 15, 15)
    return [f"ETOPO_2022_v1_15s_{_lat_str(la)}{_lon_str(lo)}_surface.nc"
            for la in lat_vals for lo in lon_vals]

def load_etopo_tiles(tile_dir, names):
    lat_all, lon_all, z_all = [], [], []
    for nm in names:
        p = os.path.join(tile_dir, nm)
        if not os.path.exists(p):
            continue
        with Dataset(p) as nc:
            lat = nc.variables["lat"][:].astype(np.float64)
            lon = nc.variables["lon"][:].astype(np.float64)
            lon = np.where(lon > 180, lon - 360, lon)
            z   = nc.variables["z"][:].astype(np.float32)
            z[z < 0] = np.nan
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

ETOPO_CACHE = {}
def get_etopo_tree(lat_range, lon_range):
    key = (tuple(np.round(lat_range, 3)), tuple(np.round(lon_range, 3)))
    if key in ETOPO_CACHE:
        return ETOPO_CACHE[key]
    names = get_etopo_tile_names(lat_range, lon_range)
    la_e, lo_e, zz_e = load_etopo_tiles(ETOPO_DIR, names)
    if len(la_e) == 0:
        ETOPO_CACHE[key] = (None, None)
        return None, None
    tree, zvals = build_kdtree(la_e, lo_e, zz_e)
    ETOPO_CACHE[key] = (tree, zvals)
    return tree, zvals

def etopo_sample_points(latp, lonp, lat_range, lon_range):
    tree, zvals = get_etopo_tree(lat_range, lon_range)
    if tree is None:
        return np.full(latp.shape[0], np.nan, dtype=np.float32)
    xy = np.stack([latp.astype(np.float64), lonp.astype(np.float64)], axis=1)
    ok = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    out = np.full(latp.shape[0], np.nan, dtype=np.float32)
    if np.any(ok):
        _, idx = tree.query(xy[ok], k=1)
        out[ok] = zvals[idx].astype(np.float32)
    return out

def overlay_topo_dots_regular(ax, lat_range, lon_range, zorder=6):
    if not TOPO_ON:
        return

    latv = np.arange(lat_range[0], lat_range[1] + 1e-9, TOPO_GRID_DEG, dtype=np.float64)
    lonv = np.arange(lon_range[0], lon_range[1] + 1e-9, TOPO_GRID_DEG, dtype=np.float64)
    if latv.size == 0 or lonv.size == 0:
        return

    LON, LAT = np.meshgrid(lonv, latv)
    latp = LAT.ravel()
    lonp = LON.ravel()

    z = etopo_sample_points(latp, lonp, lat_range, lon_range)
    m = np.isfinite(z) & (z >= TOPO_MIN_M)
    if not np.any(m):
        return

    ax.scatter(
        lonp[m], latp[m],
        s=TOPO_DOTS_SIZE,
        c=TOPO_DOTS_COLOR,
        alpha=TOPO_DOTS_ALPHA,
        marker="o",
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=zorder,
        rasterized=True
    )


# --------------------------
# Stations
# --------------------------
def find_station_csv():
    cands = []
    cands += glob.glob(os.path.join(SCRIPT_DIR, "..", "data", "CRU", "station", "*.csv"))
    pref = ["NAME", "station", "rmse", "RMSE"]
    for p in pref:
        for c in cands:
            if p.lower() in os.path.basename(c).lower():
                return c
    return cands[0] if cands else None

def read_station_table(path):
    if path is None or (not os.path.exists(path)):
        return None, None

    last_err = None
    df = None
    for enc in ["cp949", "euc-kr", "utf-8-sig", "utf-8"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception as e:
            last_err = e
    if df is None:
        print(f"⚠ Station read failed: {path} ({last_err})")
        return None, None

    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_lat = pick("lat", "latitude", "LAT")
    c_lon = pick("lon", "longitude", "LON")
    if c_lat is None or c_lon is None:
        print(f"⚠ Station columns not found (need LAT/LON): {path}")
        return None, None

    lat = pd.to_numeric(df[c_lat], errors="coerce").values.astype(float)
    lon = pd.to_numeric(df[c_lon], errors="coerce").values.astype(float)
    lon = np.where(lon > 180, lon - 360, lon)

    m = np.isfinite(lat) & np.isfinite(lon)
    return lat[m], lon[m]


# --------------------------
# Drawing core
# --------------------------
def draw_diff_panel(ax, win, ml_field, lat_ml, lon_ml, cru_interp, cmap, norm, sta_lat, sta_lon):
    lon0, lon1 = win["lon"]
    lat0, lat1 = win["lat"]

    ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35)
    set_ticks_2deg(ax, win, step=2)

    lat_rng_pad = (lat0 - TOPO_PAD_DEG, lat1 + TOPO_PAD_DEG)
    lon_rng_pad = (lon0 - TOPO_PAD_DEG, lon1 + TOPO_PAD_DEG)

    n_rect_drawn = 0
    n_scat_drawn = 0
    last_mappable = None

    for t in range(lat_ml.shape[0]):
        lat_t = lat_ml[t]
        lon_t = lon_ml[t]
        ml_t  = ml_field[t]

        if not (np.nanmax(lon_t) >= lon0 and np.nanmin(lon_t) <= lon1 and
                np.nanmax(lat_t) >= lat0 and np.nanmin(lat_t) <= lat1):
            continue

        mwin = np.isfinite(lat_t) & np.isfinite(lon_t) & np.isfinite(ml_t) & \
               (lat_t >= lat0) & (lat_t <= lat1) & (lon_t >= lon0) & (lon_t <= lon1)
        if mwin.sum() < 500:
            continue

        lat1d, lon1d, ok = tile_rectilinear_axes(lat_t, lon_t, tol=5e-4)

        if ok:
            iy = np.where((lat1d >= lat0) & (lat1d <= lat1))[0]
            ix = np.where((lon1d >= lon0) & (lon1d <= lon1))[0]
            if iy.size < 2 or ix.size < 2:
                ok = False
            else:
                lat_sub = lat1d[iy]
                lon_sub = lon1d[ix]
                ml_sub  = ml_t[np.ix_(iy, ix)]

                LON2, LAT2 = np.meshgrid(lon_sub.astype(np.float64), lat_sub.astype(np.float64))
                pts = np.c_[LAT2.ravel(), LON2.ravel()]
                cru_on_ml = cru_interp(pts).reshape(LAT2.shape).astype(np.float32)
                diff = (ml_sub - cru_on_ml).astype(np.float32)

                lat_e = centers_to_edges_1d(lat_sub)
                lon_e = centers_to_edges_1d(lon_sub)
                if lat_e is None or lon_e is None:
                    ok = False
                else:
                    im = ax.pcolormesh(
                        lon_e, lat_e, diff,
                        cmap=cmap, norm=norm,
                        shading="auto",
                        transform=ccrs.PlateCarree(),
                        zorder=1
                    )
                    last_mappable = im
                    n_rect_drawn += 1

        if not ok:
            latp = lat_t[mwin].ravel().astype(np.float64)
            lonp = lon_t[mwin].ravel().astype(np.float64)
            mlp  = ml_t[mwin].ravel().astype(np.float32)

            pts = np.c_[latp, lonp]
            crup = cru_interp(pts).astype(np.float32)
            diffp = (mlp - crup).astype(np.float32)

            sc = ax.scatter(
                lonp, latp, c=diffp,
                s=SCATTER_S, cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree(),
                linewidths=0,
                zorder=1,
                rasterized=RASTERIZE_SCATTER
            )
            last_mappable = sc
            n_scat_drawn += 1

    if (n_rect_drawn + n_scat_drawn) == 0:
        ax.text(0.5, 0.5, "No ML pixels drawn",
                ha="center", va="center", transform=ax.transAxes, color="crimson")
    else:
        print(f"  [{win['key']}] tiles drawn: rect={n_rect_drawn}, scatter_fallback={n_scat_drawn}")

    if TOPO_ON:
        overlay_topo_dots_regular(ax, lat_rng_pad, lon_rng_pad, zorder=6)

    if sta_lat is not None:
        m = (sta_lon >= lon0) & (sta_lon <= lon1) & (sta_lat >= lat0) & (sta_lat <= lat1)
        if np.any(m):
            ax.scatter(
                sta_lon[m], sta_lat[m],
                transform=ccrs.PlateCarree(),
                marker=STA_TRI_MARKER,
                s=STA_TRI_SIZE,
                c=STA_TRI_COLOR,
                alpha=STA_TRI_ALPHA,
                linewidths=0,
                zorder=9
            )

    return last_mappable


# --------------------------
# Main
# --------------------------
def main():
    print("• Build ML summer mean ...")
    ML_summer_mean, lat_ml, lon_ml = build_ml_summer_mean()

    print("• Build ML summer trend ...")
    ML_summer_trend, _, _ = build_ml_summer_trend()

    print("• Load CRU summer mean ...")
    cru_summer_mean, cru_lat_mean, cru_lon_mean = load_cru_summer_mean()
    cru_interp_mean = make_cru_interpolator(cru_summer_mean, cru_lat_mean, cru_lon_mean)

    print("• Load CRU summer trend ...")
    cru_summer_trend, cru_lat_trend, cru_lon_trend = load_cru_summer_trend()
    cru_interp_trend = make_cru_interpolator(cru_summer_trend, cru_lat_trend, cru_lon_trend)

    print("• Find & read station CSV ...")
    sta_csv = find_station_csv()
    if sta_csv:
        print(f"  Station CSV: {sta_csv}")
    sta_lat, sta_lon = read_station_table(sta_csv)
    if sta_lat is None:
        print("  (No valid station table -> proceed without stations)")

    fig, axes = plt.subplots(
        2, 4, figsize=(18.0, 8.8),
        subplot_kw=dict(projection=ccrs.PlateCarree())
    )
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.14, top=0.90, wspace=0.22, hspace=0.18)

    mean_norm  = mpl.colors.TwoSlopeNorm(vmin=MEAN_VMIN,  vcenter=0.0, vmax=MEAN_VMAX)
    trend_norm = mpl.colors.TwoSlopeNorm(vmin=TREND_VMIN, vcenter=0.0, vmax=TREND_VMAX)

    top_mappable = None
    bot_mappable = None

    # top row: mean
    for i, win in enumerate(windows):
        ax = axes[0, i]
        top_mappable = draw_diff_panel(
            ax=ax,
            win=win,
            ml_field=ML_summer_mean,
            lat_ml=lat_ml,
            lon_ml=lon_ml,
            cru_interp=cru_interp_mean,
            cmap=MEAN_CMAP,
            norm=mean_norm,
            sta_lat=sta_lat,
            sta_lon=sta_lon
        )
        add_panel_tag(ax, PANEL_TAGS_TOP[i], win["name"])

    # bottom row: trend
    for i, win in enumerate(windows):
        ax = axes[1, i]
        bot_mappable = draw_diff_panel(
            ax=ax,
            win=win,
            ml_field=ML_summer_trend,
            lat_ml=lat_ml,
            lon_ml=lon_ml,
            cru_interp=cru_interp_trend,
            cmap=TREND_CMAP,
            norm=trend_norm,
            sta_lat=sta_lat,
            sta_lon=sta_lon
        )
        add_panel_tag(ax, PANEL_TAGS_BOTTOM[i], win["name"])

    # shared colorbar for top row (mean)
    if top_mappable is not None:
        cax1 = fig.add_axes([0.12, 0.08, 0.33, 0.028])
        cb1 = fig.colorbar(top_mappable, cax=cax1, orientation="horizontal", ticks=MEAN_TICKS)
        cb1.set_label("ML − CRU mean (°C)")
        cb1.ax.tick_params(labelsize=9)

    # shared colorbar for bottom row (trend)
    if bot_mappable is not None:
        cax2 = fig.add_axes([0.56, 0.08, 0.33, 0.028])
        cb2 = fig.colorbar(bot_mappable, cax=cax2, orientation="horizontal", ticks=TREND_TICKS)
        cb2.set_label("ML − CRU rate of change (°C decade$^{-1}$)")
        cb2.ax.tick_params(labelsize=9)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("✅ Saved:", OUT_PNG)
    print("✅ Saved:", OUT_PDF)


if __name__ == "__main__":
    main()
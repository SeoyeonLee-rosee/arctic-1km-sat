#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
CRU vs MODIS – SUMMER ONLY – DIFF ONLY (MOD-CRU) – 2x2 mosaic → PNG

추가:
- ETOPO 고도 500m 이상 영역을 점(도트)로 overlay  (현재는 해칭으로 구현)
- station을 검은색 삼각형(▲)으로 크게 표시

추가 요청(패치):
- 해칭 영역 도메인을 window보다 ±0.25° 넓게 잡아서 "안쪽 틈" 없애기
  (축 extent는 원래 window 유지 → 자동으로 밖은 clip됨)
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

from matplotlib.colors import TwoSlopeNorm
from netCDF4 import Dataset
from scipy.spatial import cKDTree
import matplotlib as mpl

# =========================
# User options
# =========================
DRAW_STATION   = True
DEADBAND_EPS   = 0  # 0.0이면 deadband 끔

ETOPO_DIR      = "/data1/DATA_ARCHIVE/ETOPO2022/"
ETOPO_LEVELS   = []#[500, 1000, 1500, 2000]
ETOPO_LW       = 0.6
ETOPO_ALPHA    = 0.85

# ✅ altitude hatch overlay
ALT_HATCH_ON    = True
ALT_HATCH_MIN   = 500.0
ALT_HATCH       = "."     # 또는 "\\\\", "xx", "..", "++" 등
ALT_HATCH_EC    = 0.6       # hatch 선색 (edgecolor)
ALT_HATCH_LW    = 0.0       # 0이면 테두리 안그림(권장)
ALT_HATCH_ALPHA = 0.0       # face 투명 (해칭만 보이게)
mpl.rcParams["hatch.color"] = "0.3"   # 회색 (0=black, 1=white)
mpl.rcParams["hatch.linewidth"] = 0.3

# ✅ station style
STATION_MARKER = "^"
STATION_SIZE   = 55
STATION_COLOR  = "k"
STATION_ALPHA  = 0.95

# ✅ hatch 도메인 padding (요청: ±0.25°)
WIN_PAD_HATCH  = 0.25

# =========================
# Paths
# =========================
CRU_NC   = "../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc"
CRU_AVE  = "../data/CRU/grid/ave_gr.npy"
MOD_AVE  = "../data/MODIS/ave_0.5.npy"
STA_CSV  = "../data/CRU/station/NAME.csv"
OUTDIR   = "../FIG_newest/"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------- Windows  -----------------
windows = [
#    {"key":"T2", "name":"T2","lon":(9,14), "lat":(60,65)},
#    {"key":"T2", "name":"T2","lon":(14,19), "lat":(65,70)},
    {"key":"T2",   "name":"T2",              "lon":(25,30),       "lat":(60,65)},
    {"key":"Ural", "name":"Ural",            "lon":(58,63),       "lat":(62,67)},
    {"key":"Verkhoyansk","name":"Verkhoyansk","lon":(128.5,133.5),"lat":(65,70)},
    {"key":"Northern Alaska","name":"Northern Alaska","lon":(-155,-150),"lat":(63.5,68.5)},
]

# =========================
# Utils
# =========================
def apply_deadband(a, eps):
    if eps <= 0:
        return a
    a = a.copy()
    m = np.isfinite(a) & (np.abs(a) < eps)
    a[m] = 0.0
    return a

def set_ticks_2deg(ax, win, step=2):
    lon0, lon1 = win["lon"]
    lat0, lat1 = win["lat"]

    xt = np.arange(np.ceil(lon0/step)*step, np.floor(lon1/step)*step + 0.1, step).astype(int)

    # ✅ Verkhoyansk만 ytick 강제: 65, 67, 69
    if win.get("key", "") == "Verkhoyansk":
        yt = np.array([65, 67, 69], dtype=int)
    else:
        yt = np.arange(np.ceil(lat0/step)*step, np.floor(lat1/step)*step + 0.1, step).astype(int)

    ax.set_xticks(xt, crs=ccrs.PlateCarree())
    ax.set_yticks(yt, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(labelsize=8, direction="out")

    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")


def pad_window(win, pad=0.25):
    lon0, lon1 = win["lon"]
    lat0, lat1 = win["lat"]
    return {"lon": (lon0 - pad, lon1 + pad), "lat": (lat0 - pad, lat1 + pad)}

# =========================
# ETOPO helpers
# =========================
def _lat_str(d):
    return f"N{abs(int(d))//15*15:02d}" if d >= 0 else f"S{abs(int(d))//15*15:02d}"

def _lon_str(d):
    return f"E{abs(int(d))//15*15:03d}" if d >= 0 else f"W{abs(int(d))//15*15:03d}"

def get_etopo_tile_names(lat_range, lon_range):
    lat_vals = range(int(lat_range[0])//15*15, int(lat_range[1]) + 15, 15)
    lon_vals = range(int(lon_range[0])//15*15, int(lon_range[1]) + 15, 15)
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

def sample_etopo_vec(tree, vals, lat_vec, lon_vec):
    xy = np.stack([lat_vec, lon_vec], axis=1)
    ok = np.all(np.isfinite(xy), axis=1)
    out = np.full(lat_vec.shape[0], np.nan, dtype=np.float32)
    if np.any(ok):
        _, idx = tree.query(xy[ok])
        out[ok] = vals[idx]
    return out

ETOPO_CACHE = {}

def get_etopo_tree_for_window(win):
    key = (tuple(win["lat"]), tuple(win["lon"]))
    if key in ETOPO_CACHE:
        return ETOPO_CACHE[key]

    names = get_etopo_tile_names(win["lat"], win["lon"])
    la_e, lo_e, zz_e = load_etopo_tiles(ETOPO_DIR, names)
    if len(la_e) == 0:
        ETOPO_CACHE[key] = (None, None)
        return None, None

    tree, zvals = build_kdtree(la_e, lo_e, zz_e)
    ETOPO_CACHE[key] = (tree, zvals)
    return tree, zvals

def add_etopo_contours(ax, win, lon2d, lat2d):
    tree, zvals = get_etopo_tree_for_window(win)
    if tree is None:
        return None

    z = sample_etopo_vec(tree, zvals, lat2d.ravel().astype(np.float32), lon2d.ravel().astype(np.float32))
    z = z.reshape(lat2d.shape)
    z_masked = np.ma.masked_invalid(z)

    cs = ax.contour(
        lon2d, lat2d, z_masked,
        levels=ETOPO_LEVELS,
        colors="k",
        linewidths=ETOPO_LW,
        alpha=ETOPO_ALPHA,
        transform=ccrs.PlateCarree(),
        zorder=5
    )
    return cs

def add_altitude_hatch(ax, win, lon2d, lat2d):
    """✅ z>=ALT_HATCH_MIN 영역을 해칭으로 overlay."""
    if not ALT_HATCH_ON:
        return

    tree, zvals = get_etopo_tree_for_window(win)
    if tree is None:
        return

    z = sample_etopo_vec(
        tree, zvals,
        lat2d.ravel().astype(np.float32),
        lon2d.ravel().astype(np.float32)
    ).reshape(lat2d.shape)

    m = np.isfinite(z) & (z >= ALT_HATCH_MIN)
    if not np.any(m):
        return

    mask01 = np.zeros_like(z, dtype=np.float32)
    mask01[m] = 1.0

    cf = ax.contourf(
        lon2d, lat2d, mask01,
        levels=[0.5, 1.5],
        colors="none",
        hatches=[ALT_HATCH],
        transform=ccrs.PlateCarree(),
        zorder=6
    )

    try:
        for col in cf.collections:
            col.set_edgecolor(ALT_HATCH_EC)
            col.set_linewidth(ALT_HATCH_LW)
            col.set_alpha(ALT_HATCH_ALPHA)
    except Exception:
        pass

# =========================
# Load CRU grid (lat/lon)
# =========================
with xr.open_dataset(CRU_NC) as ds:
    lat_all = ds["lat"].values
    lon_all = ds["lon"].values

lon_all = np.where(lon_all > 180, lon_all - 360, lon_all)
lat_60_mask = (lat_all >= 60)
lat_60 = lat_all[lat_60_mask]

# =========================
# Stations (optional)
# =========================
if DRAW_STATION:
    NAME = pd.read_csv(STA_CSV, encoding="cp949")
    WO   = NAME[NAME["W"] == 0]
    ALL  = NAME[NAME["W"] >= 1]

# =========================
# Load data and compute seasonal means
# =========================
def load_season(fp):
    arr = np.load(fp).astype(np.float32)  # (12, lat, lon)
    if arr.shape[1] == lat_all.size:
        arr = arr[:, lat_60_mask, :]
    idx = {"winter":[10,11,0,1,2], "summer":list(range(3,10))}  # Apr-Oct
    return {s: np.nanmean(arr[i], axis=0) for s, i in idx.items()}

cru = load_season(CRU_AVE)
mod = load_season(MOD_AVE)
diff = {s: mod[s] - cru[s] for s in cru}

# =========================
# Plot settings
# =========================
season = "summer"
norm_diff  = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
ticks_diff = [-3, -1.5, 0, 1.5, 3]
cmap_diff  = "bwr"

# =========================
# Plot 2x2 mosaic
# =========================
fig, axes = plt.subplots(
    2, 2,
    figsize=(9.2, 7.2),
    subplot_kw=dict(projection=ccrs.PlateCarree()),
    constrained_layout=True
)

mappables = []

for i, win in enumerate(windows):
    r, c = divmod(i, 2)
    ax = axes[r, c]

    lat_m = (lat_60 >= win["lat"][0]) & (lat_60 <= win["lat"][1])
    lon_m = (lon_all >= win["lon"][0]) & (lon_all <= win["lon"][1])

    fld = diff[season][np.ix_(lat_m, lon_m)]
    fld = apply_deadband(fld, DEADBAND_EPS)

    ax.set_extent([*win["lon"], *win["lat"]], ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.set_title(win["name"], fontsize=11, pad=5)

    set_ticks_2deg(ax, win, step=2)

    if np.isnan(fld).all():
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="red")
        continue

    # pcolormesh (데이터는 원래 window 서브셋으로)
    im = ax.pcolormesh(
        lon_all[lon_m], lat_60[lat_m], fld,
        cmap=cmap_diff, norm=norm_diff,
        shading="auto", transform=ccrs.PlateCarree(),
        zorder=1
    )
    mappables.append(im)

    # === ETOPO contours (원래 grid로) ===
    lons_1d = lon_all[lon_m]
    lats_1d = lat_60[lat_m]
    LON2, LAT2 = np.meshgrid(lons_1d, lats_1d)
    add_etopo_contours(ax, win, LON2, LAT2)

    # ✅ altitude hatch: window보다 ±0.25° 확장 격자로 생성 → 안쪽 틈 제거
    win_pad = pad_window(win, WIN_PAD_HATCH)

    lat_m2 = (lat_60 >= win_pad["lat"][0]) & (lat_60 <= win_pad["lat"][1])
    lon_m2 = (lon_all >= win_pad["lon"][0]) & (lon_all <= win_pad["lon"][1])

    lons_1d2 = lon_all[lon_m2]
    lats_1d2 = lat_60[lat_m2]
    LON2_pad, LAT2_pad = np.meshgrid(lons_1d2, lats_1d2)

    add_altitude_hatch(ax, win_pad, LON2_pad, LAT2_pad)

    # ✅ stations overlay (bigger black triangles)
    if DRAW_STATION:
        flt = lambda d: d["LON"].between(*win["lon"]) & d["LAT"].between(*win["lat"])
        ax.scatter(WO.loc[flt(WO), "LON"], WO.loc[flt(WO), "LAT"],
                   transform=ccrs.PlateCarree(),
                   marker=STATION_MARKER, c=STATION_COLOR,
                   s=STATION_SIZE, alpha=STATION_ALPHA,
                   linewidths=0, zorder=7)
        ax.scatter(ALL.loc[flt(ALL), "LON"], ALL.loc[flt(ALL), "LAT"],
                   transform=ccrs.PlateCarree(),
                   marker=STATION_MARKER, c=STATION_COLOR,
                   s=STATION_SIZE, alpha=STATION_ALPHA,
                   linewidths=0, zorder=7)

# shared colorbar
im0 = next((m for m in mappables if m is not None), None)
if im0 is not None:
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), orientation="horizontal",
                        fraction=0.05, pad=0.07, ticks=ticks_diff)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("MODIS − CRU (°C)", fontsize=11)

postfix = "withStation" if DRAW_STATION else "noStation"
out = os.path.join(
    OUTDIR,
    f"SUMMER_MODminusCRU_2x2_mesh_tick2deg_deadband{DEADBAND_EPS:g}_"
    f"altHatch{int(ALT_HATCH_MIN)}m_pad{WIN_PAD_HATCH:.2f}deg_{postfix}.png"
)

fig.suptitle("Summer (Apr–Oct) MODIS − CRU", fontsize=14)
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", out)

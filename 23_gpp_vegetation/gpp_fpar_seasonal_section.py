#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
ML(GPP/FPAR) seasonal longitudinal section along a fixed latitude (U or V),
colored by elevation (ETOPO 2022).

입력
  ../data/MOD/lat.npy, lon.npy
  ../data/MOD/GPP/01/ave.npy ... 12/ave.npy
  ../data/MOD/FPAR/01/ave.npy ... 12/ave.npy
  /data1/DATA_ARCHIVE/ETOPO2022/* (ETOPO_2022_v1_15s_*_surface.nc)

출력
  ../FIG/scatter/ML_<VAR>_<REGION>_<season>.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from netCDF4 import Dataset
from scipy.spatial import cKDTree

# ───────────────────────────────────────────────
# 사용자 설정
# ───────────────────────────────────────────────
REGION_KEY = 'V'       # 'U' 또는 'V'
DO_WINTER  = True     # True: Cold(Nov–Mar), False: Warm(Apr–Oct)
VAR_LIST   = ['GPP', 'FPAR']  # 그리고 싶은 변수

# 고도 색상 설정 (요청사항 유지)
ALT_CMAP = plt.get_cmap('viridis')
ALT_NORM = Normalize(vmin=0, vmax=1400)  # 0–1400 m

# 파일/디렉토리
DATA_BASE_PATH = '../data/MOD'
FIG_DIR        = '../FIG/scatter'
ETOPO_DIR      = '/data1/DATA_ARCHIVE/ETOPO2022/'
os.makedirs(FIG_DIR, exist_ok=True)

# 계절 정의 (월=1..12)
WARM = [4, 5, 6, 7, 8, 9, 10]          # Apr–Oct
COLD = [11, 12, 1, 2, 3]               # Nov–Mar
MONS = COLD if DO_WINTER else WARM
SEASON_NAME = "Cold (Nov–Mar)" if DO_WINTER else "Warm (Apr–Oct)"
SEASON_KEY  = "cold" if DO_WINTER else "warm"

# 지역 설정
REGIONS = {
    'U': dict(lat=65.0, lon_min=57.5, lon_max=62.5, plot_min=58.0, plot_max=62.0,
              xticks=np.arange(58, 63, 1), title="Ural [65°N]"),
    'V': dict(lat=67.0, lon_min=129.5, lon_max=135.5, plot_min=130.0, plot_max=135.0,
              xticks=np.arange(130, 136, 1), title="Verkhoyansk [67°N]"),
}
CONF = REGIONS[REGION_KEY]

# ── 요청하신 y축 범위(변수×지역×계절) ─────────────────────────────
YLIM_BY_CASE = {
    # GPP
    ('GPP', 'U', 'warm'): (0.0, 0.1),
    ('GPP', 'V', 'warm'): (0.0, 0.1),
    ('GPP', 'U', 'cold'): (0.0, 0.01),
    ('GPP', 'V', 'cold'): (0.0, 0.01),
    # FPAR
    ('FPAR', 'U', 'warm'): (0.0, 0.42),
    ('FPAR', 'V', 'warm'): (0.0, 0.40),
    ('FPAR', 'U', 'cold'): (0.0, 0.18),
    ('FPAR', 'V', 'cold'): (0.0, 0.15),
}
# ───────────────────────────────────────────────────────────────

# 샘플링/후처리 파라미터
LON_STEP = 0.01      # 경도 샘플 간격(도)
LAT_TOL  = 0.10      # 위도 후보 픽셀 허용 오차(도)
LON_PAD  = 0.50      # 후보 픽셀 경도 여유(도)

# ───────────────────────────────────────────────
# 유틸: ETOPO 타일 로딩 및 샘플링
# ───────────────────────────────────────────────
def _lat_str(d): return f"N{abs(int(d))//15*15:02d}" if d >= 0 else f"S{abs(int(d))//15*15:02d}"
def _lon_str(d): return f"E{abs(int(d))//15*15:03d}" if d >= 0 else f"W{abs(int(d))//15*15:03d}"

def get_etopo_tile_names(lat_range, lon_range):
    la0 = int(np.floor(min(lat_range)))
    la1 = int(np.ceil (max(lat_range)))
    lo0 = int(np.floor(min(lon_range)))
    lo1 = int(np.ceil (max(lon_range)))
    lat_vals = range(la0//15*15, la1 + 15, 15)
    lon_vals = range(lo0//15*15, lo1 + 15, 15)
    return [f"ETOPO_2022_v1_15s_{_lat_str(la)}{_lon_str(lo)}_surface.nc"
            for la in lat_vals for lo in lon_vals]

def load_etopo_tiles(tile_dir, names):
    lat_all, lon_all, z_all = [], [], []
    for nm in names:
        p = os.path.join(tile_dir, nm)
        if not os.path.exists(p):
            print(f"  ⚠️ 누락 ETOPO 타일: {p}")
            continue
        with Dataset(p) as nc:
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
            z   = nc.variables['z'][:].astype(np.float32)
            z[z <= 0] = np.nan  # 바다/수면 NaN
            lat_all.append(lat); lon_all.append(lon); z_all.append(z)
    if not lat_all:
        raise FileNotFoundError("요청 범위를 덮는 ETOPO 타일이 없습니다.")
    return lat_all, lon_all, z_all

def build_kdtree(lat_arrs, lon_arrs, z_arrs):
    pts, vals = [], []
    for la, lo, zz in zip(lat_arrs, lon_arrs, z_arrs):
        LA, LO = np.meshgrid(la, lo, indexing='ij')  # (nlat,nlon)
        pts.append(np.stack([LA.ravel(), LO.ravel()], axis=1))
        vals.append(zz.ravel())
    P = np.concatenate(pts, axis=0)
    V = np.concatenate(vals, axis=0)
    return cKDTree(P), V

def sample_etopo(tree, vals, lat_vec, lon_vec):
    xy = np.column_stack([lat_vec, lon_vec])
    ok = np.all(np.isfinite(xy), axis=1)
    out = np.full(lat_vec.shape[0], np.nan, dtype=np.float32)
    if np.any(ok):
        _, idx = tree.query(xy[ok])
        out[ok] = vals[idx]
    return out

# ───────────────────────────────────────────────
# 유틸: ML 월평균 로드 → 계절평균 2D 필드
# ───────────────────────────────────────────────
def load_lat_lon():
    lat = np.load(os.path.join(DATA_BASE_PATH, 'lat.npy')).astype(np.float32)
    lon = np.load(os.path.join(DATA_BASE_PATH, 'lon.npy')).astype(np.float32)
    return lat, lon

def seasonal_mean_field(var_name, months):
    """../data/MOD/<var>/<MM>/ave.npy 읽어서 months 평균"""
    arrs = []
    for m in months:
        fp = os.path.join(DATA_BASE_PATH, var_name, f"{m:02d}", "ave.npy")
        if not os.path.exists(fp):
            print(f"  ⚠️ {var_name} {m:02d}월 없음: {fp}")
            continue
        arrs.append(np.load(fp).astype(np.float32))
    if not arrs:
        raise FileNotFoundError(f"{var_name} {months} 월 자료가 없습니다.")
    return np.nanmean(np.stack(arrs, axis=0), axis=0)  # 2D

# ───────────────────────────────────────────────
# 유틸: 위도선 단면 최근접 샘플링
# ───────────────────────────────────────────────
def extract_line_nearest(field2d, lat2d, lon2d, lat_line, lon_min, lon_max,
                         lon_step=LON_STEP, lat_tol=LAT_TOL, lon_pad=LON_PAD):
    """주변 후보 픽셀 → KDTree → target 경도선 최근접 샘플"""
    target_lons = np.arange(lon_min, lon_max + lon_step/2.0, lon_step)
    target_lats = np.full_like(target_lons, lat_line, dtype=float)

    cand = (np.abs(lat2d - lat_line) <= lat_tol) & \
           (lon2d >= (lon_min - lon_pad)) & (lon2d <= (lon_max + lon_pad))
    lat_cand = lat2d[cand].ravel()
    lon_cand = lon2d[cand].ravel()
    val_cand = field2d[cand].ravel()

    out = np.full_like(target_lons, np.nan, dtype=np.float32)
    if lat_cand.size > 0:
        tree = cKDTree(np.column_stack([lat_cand, lon_cand]))
        _, idx = tree.query(np.column_stack([target_lats, target_lons]))
        out = val_cand[idx].astype(np.float32)
    else:
        print("  ⚠️ 후보 픽셀 없음: lat_tol/lon_pad를 늘려보세요.")
    return target_lons, out

# ───────────────────────────────────────────────
# 유틸: 색상=고도 선(LineCollection)
# ───────────────────────────────────────────────
def colored_line(ax, x, y, c_vals, cmap=ALT_CMAP, norm=ALT_NORM, lw=1.8, ls='solid', alpha=0.95):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c_vals)
    x, y, c_vals = x[m], y[m], c_vals[m]
    if x.size < 2:
        return None
    ii = np.argsort(x)
    x, y, c_vals = x[ii], y[ii], c_vals[ii]
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    c_seg = 0.5 * (c_vals[:-1] + c_vals[1:])
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw, linestyles=ls, alpha=alpha)
    lc.set_array(c_seg)
    ax.add_collection(lc)
    return lc

# ───────────────────────────────────────────────
# 메인
# ───────────────────────────────────────────────
def main():
    lat2d, lon2d = load_lat_lon()

    # ETOPO: 위도선 고도
    tiles = get_etopo_tile_names((CONF['lat'], CONF['lat']), (CONF['lon_min'], CONF['lon_max']))
    la, lo, zz = load_etopo_tiles(ETOPO_DIR, tiles)
    tree_z, zvals = build_kdtree(la, lo, zz)

    # target 경도 좌표(고도 샘플용)
    x_plot = np.arange(CONF['lon_min'], CONF['lon_max'] + LON_STEP/2.0, LON_STEP)
    z_line = sample_etopo(tree_z, zvals, np.full_like(x_plot, CONF['lat']), x_plot)

    for var in VAR_LIST:
        print(f"• Build seasonal field: {var} / {SEASON_NAME}")
        fld = seasonal_mean_field(var, MONS)  # 2D
        lons, vals = extract_line_nearest(fld, lat2d, lon2d,
                                          CONF['lat'], CONF['lon_min'], CONF['lon_max'])

        fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.5))
        lc = colored_line(ax, lons, vals, z_line, lw=1.8)

        ax.grid(True, alpha=0.3)
        ax.set_xlim(CONF['plot_min'], CONF['plot_max'])
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel(var)

        # ▶▶ 요청 y축 범위 적용
        ylim = YLIM_BY_CASE.get((var, REGION_KEY, SEASON_KEY))
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_xticks(CONF['xticks'])
        title = f"{CONF['title']}  —  {SEASON_NAME}"
        ax.set_title(title, fontsize=11)

        if lc is not None:
            cbar = fig.colorbar(lc, ax=ax, orientation='horizontal', fraction=0.10, pad=0.22)
            cbar.set_label("Altitude (m)")

        season_tag = "winter" if DO_WINTER else "summer"
        out_png = os.path.join(FIG_DIR, f"ML_{var}_{REGION_KEY}_{season_tag}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close(fig)
        print("✅ Saved:", out_png)

if __name__ == "__main__":
    main()

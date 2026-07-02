#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Generate longitudinal temperature profiles along 65°N or 67°N
depending on region (Ural or Verkhoyansk) and season (winter or summer).
Supports:
  - MODIS LST (ML)
  - CRU TS
  - ERA5 reanalysis
  - Colored elevation line using ETOPO
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# ───────────────────────────────────────────────
# ✅ 사용자 설정
# ───────────────────────────────────────────────
region_key = 'V'       # 'V' = Verkhoyansk, 'U' = Ural
DO_WINTER  = True     # True = winter, False = summer

# ───────────────────────────────────────────────
# 🌡 계절별 월 설정
# ───────────────────────────────────────────────
WARM = [4, 5, 6, 7, 8, 9, 10]       # Apr–Oct (여름)
COLD = [10, 12, 1, 2, 3]           # Nov–Mar (겨울)
MONS = COLD if DO_WINTER else WARM

# ───────────────────────────────────────────────
# 🔧 지역별 설정 자동 적용
# ───────────────────────────────────────────────
if region_key == 'V':
    LAT_LINE     = 67.0
    LON_MIN      = 129.5
    LON_MAX      = 135.5
    LON_PLOT_MIN = 130.0
    LON_PLOT_MAX = 135.0
    TEMP_YLIM    = (-36, -26) if DO_WINTER else (1, 6)
    XTICKS       = np.arange(130, 136, 1)
    YTICKS       = np.arange(-36, -25, 2) if DO_WINTER else np.arange(1, 7, 1)
    TITLE_FMT    = "Verkhoyansk [{lat:.0f}°N]"
    FNAME_FMT    = "Verkhoyansk_winter.png" if DO_WINTER else "Verkhoyansk_summer.png"
elif region_key == 'U':
    LAT_LINE     = 65.0
    LON_MIN      = 57.5
    LON_MAX      = 62.5
    LON_PLOT_MIN = 58.0
    LON_PLOT_MAX = 62.0
    TEMP_YLIM    = (-17, -10) if DO_WINTER else (1, 8)
    XTICKS       = np.arange(58, 63, 1)
    YTICKS       = np.arange(-17, -9, 2) if DO_WINTER else np.arange(1, 9, 1)
    TITLE_FMT    = "Ural [{lat:.0f}°N]"
    FNAME_FMT    = "Ural_winter.png" if DO_WINTER else "Ural_summer.png"
else:
    raise ValueError(f"Unknown region_key: {region_key}")

SEASON_NAME = "Cold (Nov–Mar)" if DO_WINTER else "Warm (Apr–Oct)"
ML_LON_STEP = 0.001
ERA_START = "2000-01-01"
ERA_END   = "2021-12-31"

# 📁 경로 설정
ml_data_path = '../data/MODIS/'
cru_nc_path  = '../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc'
cru_ave_path = '../data/CRU/grid/ave_gr.npy'
era_nc_path  = '../data/ERA5_t2m_mon_1940-2025.nc'
etopo_dir    = '/data1/DATA_ARCHIVE/ETOPO2022/'
fig_dir      = '../FIG/scatter/'
os.makedirs(fig_dir, exist_ok=True)

# ------------------------------
# ETOPO 유틸
# ------------------------------
def _lat_str(d): return f"N{abs(int(d))//15*15:02d}" if d >= 0 else f"S{abs(int(d))//15*15:02d}"
def _lon_str(d): return f"E{abs(int(d))//15*15:03d}" if d >= 0 else f"W{abs(int(d))//15*15:03d}"

def get_etopo_tile_names(lat_range, lon_range):
    lat_vals = range(int(lat_range[0])//15*15, int(lat_range[1])+15, 15)
    lon_vals = range(int(lon_range[0])//15*15, int(lon_range[1])+15, 15)
    return [f"ETOPO_2022_v1_15s_{_lat_str(la)}{_lon_str(lo)}_surface.nc"
            for la in lat_vals for lo in lon_vals]

def load_etopo_tiles(tile_dir, names):
    lat_all, lon_all, z_all = [], [], []
    for nm in names:
        p = os.path.join(tile_dir, nm)
        if not os.path.exists(p):
            print(f"❗ 누락: {p}")
            continue
        with Dataset(p) as nc:
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
            if 'z' not in nc.variables: raise KeyError(f"'z' 없음: {p}")
            z = nc.variables['z'][:].astype(np.float32)
            z[z <= 0] = np.nan  # 바다 마스크
            lat_all.append(lat); lon_all.append(lon); z_all.append(z)
    return lat_all, lon_all, z_all

def build_kdtree(lat_arrs, lon_arrs, z_arrs):
    pts, vals = [], []
    for la, lo, zz in zip(lat_arrs, lon_arrs, z_arrs):
        LA, LO = np.meshgrid(la, lo, indexing='ij')
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

# ------------------------------
# 색상=고도 선(LineCollection) 도우미
# ------------------------------
def colored_line(ax, x, y, c_vals, cmap='viridis', norm=None, lw=1.8, ls='solid', alpha=0.95):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c_vals)
    x, y, c_vals = x[m], y[m], c_vals[m]
    if x.size < 2:
        print("⚠️ colored_line: 유효 포인트 부족:", x.size)
        return None
    idx = np.argsort(x)
    x, y, c_vals = x[idx], y[idx], c_vals[idx]
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    c_seg = 0.5 * (c_vals[:-1] + c_vals[1:])
    if norm is None: norm = Normalize(vmin=np.nanmin(c_seg), vmax=np.nanmax(c_seg))
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw, linestyles=ls, alpha=alpha)
    lc.set_array(c_seg)
    ax.add_collection(lc)
    return lc

# ==============================
# 1) ML: 월평균 로드 → 여름평균 → 65N 단면 (고해상 경도 샘플)
# ==============================
lat_ml = np.load(os.path.join(ml_data_path, 'lat.npy'))   # (34,1200,1200) sinusoidal
lon_ml = np.load(os.path.join(ml_data_path, 'lon.npy'))

mave = np.full((12, 34, 1200, 1200), np.nan, dtype=np.float32)
for mm in range(1, 13):
    fp = os.path.join(ml_data_path, f"{str(mm).zfill(2)}/ave_gr.npy")
    if not os.path.exists(fp):
        print(f"❗ ML 월파일 없음: {fp}")
        continue
    arr = np.load(fp)                                    # (34, 6, 1200, 1200)
    mave[mm-1] = arr[:, 5]                               # 6번째 레벨

ml_warm = np.nanmean(mave[np.array(MONS) - 1], axis=0)  # (34,1200,1200)

# 65N 단면 대상 경도(고해상 샘플)
target_lons = np.arange(LON_MIN, LON_MAX + ML_LON_STEP/2, ML_LON_STEP)
target_lats = np.full_like(target_lons, LAT_LINE, dtype=float)

# 65N 주변 후보 픽셀 추출 후 최근접 샘플
LAT_TOL = 0.1
LON_PAD = 0.5
cand_mask = (np.abs(lat_ml - LAT_LINE) <= LAT_TOL) & \
            (lon_ml >= (LON_MIN - LON_PAD)) & (lon_ml <= (LON_MAX + LON_PAD))
lat_cand = lat_ml[cand_mask].ravel()
lon_cand = lon_ml[cand_mask].ravel()
ml_warm_cand = ml_warm[cand_mask].ravel()

ml_line_warm = np.full_like(target_lons, np.nan, dtype=np.float32)
if lat_cand.size > 0:
    tree_ml = cKDTree(np.column_stack([lat_cand, lon_cand]))
    _, idx = tree_ml.query(np.column_stack([target_lats, target_lons]))
    ml_line_warm = ml_warm_cand[idx]
else:
    print("⚠️ ML 후보 픽셀이 없습니다. LAT_TOL/LON_PAD를 넓혀보세요.")

# ==============================
# 2) CRU: 여름평균 → 65N 최근접 단면 (57.5–62.5°E)
# ==============================
with xr.open_dataset(cru_nc_path) as ds:
    cru_lat_all = ds['lat'].values     # (360,)
    cru_lon_all = ds['lon'].values     # (720,)
cru_lon_all = np.where(cru_lon_all > 180, cru_lon_all - 360, cru_lon_all)

cru_mon = np.load(cru_ave_path).astype(np.float32)  # (12, 360, 720) or (12, 60, 720)
# 위도 축 결정 (파일이 lat>=60 잘린 버전인지 감지)
if cru_mon.shape[1] == cru_lat_all.size:       # 360
    cru_lat_vals = cru_lat_all
else:                                          # 60 (lat>=60만 존재)
    cru_lat_vals = cru_lat_all[cru_lat_all >= 60]

lat_idx_cru   = int(np.argmin(np.abs(cru_lat_vals - LAT_LINE)))
lon_mask_cru  = (cru_lon_all >= LON_MIN) & (cru_lon_all <= LON_MAX)
lons_cru      = cru_lon_all[lon_mask_cru]
cru_warm_line = np.nanmean(cru_mon[np.array(MONS) - 1, lat_idx_cru, :], axis=0)[lon_mask_cru]  # 회색 점선

# ==============================
# 3) ERA5: 2000–2021 필터 → 여름평균 → 65N 단면
# ==============================
with xr.open_dataset(era_nc_path) as ds:
    # 변수/축 이름은 파일에 따라 다를 수 있으니 아래 키 확인 후 필요시 수정
    time_era = ds['valid_time'].to_index() if 'valid_time' in ds else ds['time'].to_index()
    lat_era  = ds['latitude'   ].values if 'latitude'    in ds else ds['lat'].values
    lon_era  = ds['longitude'  ].values if 'longitude'   in ds else ds['lon'].values
    t2m      = ds['t2m'].values -273.15  # (time, lat, lon)

# 경도 0-360 → -180~180
lon_era = np.where(lon_era > 180, lon_era - 360, lon_era)

# 2000-01 ~ 2021-12만
mask_period = (time_era >= ERA_START) & (time_era <= ERA_END)
time_sel = time_era[mask_period]
t2m_sel  = t2m[mask_period, :, :]  # (T, lat, lon)

warm_mask = time_sel.month.isin(MONS)
t2m_warm  = np.nanmean(t2m_sel[warm_mask, :, :], axis=0)  # (lat, lon)

# ERA 위도/경도 인덱싱 (ERA 위도는 내림차순일 가능성이 큼)
lat_idx_era  = int(np.argmin(np.abs(lat_era - LAT_LINE)))
lon_mask_era = (lon_era >= LON_MIN) & (lon_era <= LON_MAX)
lons_era     = lon_era[lon_mask_era]
era_warm_line = t2m_warm[lat_idx_era, :][lon_mask_era]    # 회색 실선

# 혹시 ERA 경도가 내림차순이면 오름차순으로 정렬
if lons_era.size > 1 and np.any(np.diff(lons_era) < 0):
    ii = np.argsort(lons_era)
    lons_era = lons_era[ii]
    era_warm_line = era_warm_line[ii]

# ==============================
# 4) ETOPO: 65N 단면 고도 → ML 색상에 사용
# ==============================
tiles = get_etopo_tile_names((LAT_LINE, LAT_LINE), (LON_MIN, LON_MAX))
la, lo, zz = load_etopo_tiles(etopo_dir, tiles)
tree_z, zvals = build_kdtree(la, lo, zz)
alt_ml  = sample_etopo(tree_z, zvals, np.full_like(target_lons, LAT_LINE), target_lons)

# 고도 색 스케일(ML 기준)
if np.any(np.isfinite(alt_ml)):
    alt_norm = Normalize(vmin=0.0, vmax=float(np.nanpercentile(alt_ml[np.isfinite(alt_ml)], 99.5)))
else:
    alt_norm = Normalize(vmin=0.0, vmax=1000.0)

# 고도 색 스케일 고정: 0–1400m
alt_norm = Normalize(vmin=0, vmax=1400)

# ==============================
# 5) 플롯: ML(색=고도, 실선) + CRU(회색 점선) + ERA(회색 실선)
# ==============================
fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

# ML = 색이 있는 실선
lc_ml = colored_line(ax, target_lons, ml_line_warm, alt_ml, norm=alt_norm, ls='solid', lw=1.8)

# CRU = 회색 점선
m_cru = np.isfinite(lons_cru) & np.isfinite(cru_warm_line)
if m_cru.sum() > 1:
    ii = np.argsort(lons_cru[m_cru])
    ax.plot(lons_cru[m_cru][ii], cru_warm_line[m_cru][ii],
            color='0.4', linestyle=(0,(5,3)), linewidth=1.6, label='CRU')

# ERA = 회색 실선
m_era = np.isfinite(lons_era) & np.isfinite(era_warm_line)
if m_era.sum() > 1:
    ii = np.argsort(lons_era[m_era])
    ax.plot(lons_era[m_era][ii], era_warm_line[m_era][ii],
            color='0.25', linestyle='-', linewidth=1.6, label='ERA5 (2000–2021)')

# 제목/축 범위/레이블
title = TITLE_FMT.format(season=SEASON_NAME, lat=LAT_LINE, lo1=LON_MIN, lo2=LON_MAX)
#ax.set_title(title)
#ax.set_xlabel("Longitude (°E)")
#ax.set_ylabel("Temperature (°C)")
ax.grid(True, alpha=0.3)
ax.set_xlim(LON_PLOT_MIN, LON_PLOT_MAX)
ax.set_ylim(*TEMP_YLIM)

# ✅ tick 설정
ax.set_xticks(XTICKS)
ax.set_yticks(YTICKS)

# 컬러바 아래 (ML 고도)
#if lc_ml is not None:
#    cbar = fig.colorbar(lc_ml, ax=ax, orientation='horizontal', fraction=0.08, pad=0.18)
#    cbar.set_label("Altitude (m)")

# Legend는 일부러 OFF였다면 아래 주석 유지, 아니면 켜기
# ax.legend().set_visible(False)

# 저장
out_png = os.path.join(fig_dir, FNAME_FMT.format(lat=LAT_LINE, lo1=LON_MIN, lo2=LON_MAX))
plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.close()
print("✅ 완료:", out_png)
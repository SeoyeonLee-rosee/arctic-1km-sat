#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
GPP / FPAR monthly climatology at fixed latitude, colored by elevation

- 데이터:
  ../data/MOD/lat.npy, lon.npy
  ../data/MOD/GPP/01/ave.npy ... 12/ave.npy
  ../data/MOD/FPAR/01/ave.npy ... 12/ave.npy
- 고도: ETOPO 2022 15s 타일 (*.nc)

출력:
  ../FIG/timeseries/<그림들>.png

참고:
  월별(1~12) 시계열: 경도별 곡선(같은 위도 ±RADIUS_DEG 박스 평균), 색=고도
  계절 평균(Cold: Nov–Mar / Warm: Apr–Oct): 경도-평균값, 색=고도
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from netCDF4 import Dataset
from scipy.spatial import cKDTree

# ==============================
# 기본 설정
# ==============================
REGION_LIST = ['U', 'V']         # 그림을 만들 지역들
DATA_TYPE_LIST = ['GPP', 'FPAR'] # ML 변수
DATA_BASE_PATH = '../data/MOD'
FIG_BASE_PATH  = '../FIG/timeseries'
os.makedirs(FIG_BASE_PATH, exist_ok=True)

# ETOPO 2022 타일 디렉토리
ETOPO_DIR = '/data1/DATA_ARCHIVE/ETOPO2022/'

# 시각화/계산 파라미터
RADIUS_DEG = 0.005                # 위경도 박스 반경 (deg)
LINE_WIDTH = 1.2
ALPHA_LINE = 0.85

# 월 표기
MON_LABS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
MON_NUMS = np.arange(1, 13)

# Cold/Warm 시즌 인덱스(0-based)
COLD_MONTH_IDX = [10, 11, 0, 1, 2]      # Nov, Dec, Jan, Feb, Mar
WARM_MONTH_IDX = list(range(3, 10))     # Apr..Oct

# 지역 정의
REGION_CONFIG = {
    'U': {'lat': 65.0, 'lon_start': 59.75,  'lon_end': 60.25,  'title': 'Ural 65°N'},
    'V': {'lat': 67.0, 'lon_start': 132.75, 'lon_end': 133.25, 'title': 'Verkhoyansk 67°N'}
}

# y축 라벨(단위는 데이터 제작과정에 따름—확실치 않다면 단위 생략 권장)
YLABEL = {
    'GPP': 'GPP',
    'FPAR': 'FPAR'
}

# Altitude color scale for ML
ALT_CMAP = plt.get_cmap('viridis')
ALT_NORM = Normalize(vmin=0, vmax=1400)  # 0–1400 m

# ==============================
# 유틸: 데이터 로드
# ==============================
def load_lat_lon():
    lat_fp = os.path.join(DATA_BASE_PATH, 'lat.npy')
    lon_fp = os.path.join(DATA_BASE_PATH, 'lon.npy')
    if not (os.path.exists(lat_fp) and os.path.exists(lon_fp)):
        raise FileNotFoundError(f"lat.npy 또는 lon.npy 를 {DATA_BASE_PATH}에서 찾을 수 없습니다.")
    lat = np.load(lat_fp).astype(np.float32)
    lon = np.load(lon_fp).astype(np.float32)
    return lat, lon

def load_monthly_ave(data_type):
    """data_type(GPP|FPAR)의 1~12월 ave.npy를 리스트로 반환 (없으면 None)"""
    arrs = []
    base_dir = os.path.join(DATA_BASE_PATH, data_type)
    for m in range(1, 13):
        fp = os.path.join(base_dir, f"{m:02d}", "ave.npy")
        if not os.path.exists(fp):
            print(f"  ⚠️ {data_type} {m:02d}월 ave.npy 없음: {fp}")
            arrs.append(None)
            continue
        arr = np.load(fp)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        arrs.append(arr)
    return arrs

def radius_mean(arrs, lat_arr, lon_arr, lat0, lon0, radius_deg=RADIUS_DEG):
    """각 월(arrs[i])에서 lat/lon 박스 평균값 하나를 뽑아 12길이 벡터로 반환"""
    out = np.full(12, np.nan, dtype=np.float32)
    mask0 = (np.abs(lat_arr - lat0) <= radius_deg) & (np.abs(lon_arr - lon0) <= radius_deg)
    if not np.any(mask0):
        return out
    for i in range(12):
        a = arrs[i]
        if a is None:
            continue
        v = a[mask0]
        v = v[np.isfinite(v)]
        if v.size:
            out[i] = float(np.nanmean(v))
    return out

# ==============================
# 유틸: ETOPO 타일 처리(고도 샘플링)
# ==============================
def _lat_str(d): return f"N{abs(int(d))//15*15:02d}" if d >= 0 else f"S{abs(int(d))//15*15:02d}"
def _lon_str(d): return f"E{abs(int(d))//15*15:03d}" if d >= 0 else f"W{abs(int(d))//15*15:03d}"

def get_etopo_tile_names(lat_range, lon_range):
    """주어진 위경도 범위를 덮는 15° x 15° 타일 목록 생성 (lat_range, lon_range: (min, max))"""
    la0 = int(np.floor(min(lat_range))); la1 = int(np.ceil(max(lat_range)))
    lo0 = int(np.floor(min(lon_range))); lo1 = int(np.ceil(max(lon_range)))
    lat_vals = range(la0//15*15, la1 + 15, 15)
    lon_vals = range(lo0//15*15, lo1 + 15, 15)
    return [f"ETOPO_2022_v1_15s_{_lat_str(la)}{_lon_str(lo)}_surface.nc"
            for la in lat_vals for lo in lon_vals]

def load_etopo_tiles(tile_dir, names):
    """필요 타일을 로드해서 lat/lon/z를 리스트로 반환"""
    lat_all, lon_all, z_all = [], [], []
    for nm in names:
        p = os.path.join(tile_dir, nm)
        if not os.path.exists(p):
            print(f"  ⚠️ ETOPO tile 없음: {p}")
            continue
        with Dataset(p) as nc:
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
            z   = nc.variables['z'][:].astype(np.float32)
            z[z <= 0] = np.nan  # 해양/수면은 NaN
            lat_all.append(lat); lon_all.append(lon); z_all.append(z)
    if not lat_all:
        raise FileNotFoundError("요청 범위를 덮는 ETOPO 타일을 하나도 찾지 못했습니다.")
    return lat_all, lon_all, z_all

def build_kdtree(lat_arrs, lon_arrs, z_arrs):
    """여러 타일을 하나로 묶어 KDTree와 값 벡터를 만든다"""
    pts, vals = [], []
    for la, lo, zz in zip(lat_arrs, lon_arrs, z_arrs):
        LA, LO = np.meshgrid(la, lo, indexing='ij')  # (nlat, nlon)
        pts.append(np.stack([LA.ravel(), LO.ravel()], axis=1))
        vals.append(zz.ravel())
    P = np.concatenate(pts, axis=0)
    V = np.concatenate(vals, axis=0)
    return cKDTree(P), V

def sample_etopo(tree, vals, lat_vec, lon_vec):
    """KDTree로 ETOPO 고도를 샘플링 (m), NaN 가능"""
    xy = np.stack([lat_vec, lon_vec], axis=1)
    ok = np.all(np.isfinite(xy), axis=1)
    out = np.full(lat_vec.shape[0], np.nan, dtype=np.float32)
    if np.any(ok):
        _, idx = tree.query(xy[ok])
        out[ok] = vals[idx]
    return out

# ==============================
# 플로팅 함수
# ==============================
def plot_monthly_timeseries_colored(lons, monthly_data, altitudes, title, ylabel, outpath, y_limits=None):
    """
    lons: (n_lon,)
    monthly_data: (n_lon, 12)
    altitudes: (n_lon,) in meters (can include NaN)
    """
    fig, ax = plt.subplots(figsize=(10, 5.6))

    for i in range(len(lons)):
        y = monthly_data[i]
        if not np.any(np.isfinite(y)):
            continue
        alt_i = altitudes[i]
        color = ALT_CMAP(ALT_NORM(alt_i)) if np.isfinite(alt_i) else (0.5, 0.5, 0.5, 0.8)
        ax.plot(MON_NUMS, y, lw=LINE_WIDTH, alpha=ALPHA_LINE, color=color)

    ax.set_xlim(1, 12)
    ax.set_xticks(MON_NUMS)
    ax.set_xticklabels(MON_LABS)
    ax.set_xlabel('Month')
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_title(title)
    ax.grid(alpha=0.25)

    # Colorbar for elevation
    sm = ScalarMappable(cmap=ALT_CMAP, norm=ALT_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015, aspect=35)
    cbar.set_label('Elevation (m)')

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print('✅ Saved:', outpath)

def plot_seasonal_means_vs_lon(lons, cold_means, warm_means, altitudes, title, ylabel, outpath, y_limits=None):
    fig, ax = plt.subplots(figsize=(10, 5.6))

    # 선(시즌별 평균)
    ax.plot(lons, cold_means, label='Cold (Nov–Mar)', lw=1.6, color='tab:blue', alpha=0.9)
    ax.plot(lons, warm_means, label='Warm (Apr–Oct)', lw=1.6, color='tab:red',  alpha=0.9)

    # 고도 색 점
    sc1 = ax.scatter(lons, cold_means, s=14, c=altitudes, cmap=ALT_CMAP, norm=ALT_NORM,
                     alpha=0.95, edgecolors='none')
    sc2 = ax.scatter(lons, warm_means, s=14, c=altitudes, cmap=ALT_CMAP, norm=ALT_NORM,
                     alpha=0.95, edgecolors='none')

    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc='best')

    sm = ScalarMappable(cmap=ALT_CMAP, norm=ALT_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015, aspect=35)
    cbar.set_label('Elevation (m)')

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print('✅ Saved:', outpath)

# ==============================
# 메인
# ==============================
def main():
    lat_arr, lon_arr = load_lat_lon()

    for region_key in REGION_LIST:
        conf = REGION_CONFIG[region_key]
        LAT_FIXED  = conf['lat']
        LON_START  = conf['lon_start']
        LON_END    = conf['lon_end']
        TITLE_PREF = conf['title']
        LONS = np.round(np.arange(LON_START, LON_END + 1e-9, 0.01), 4)

        # ETOPO 고도 샘플링 (해당 위도선에서 경도별)
        print(f'• Sampling ETOPO altitude for {region_key} ({LAT_FIXED}°N, {LON_START}–{LON_END}°E) ...')
        tile_names = get_etopo_tile_names((LAT_FIXED, LAT_FIXED), (LON_START, LON_END))
        la_list, lo_list, z_list = load_etopo_tiles(ETOPO_DIR, tile_names)
        tree_z, zvals = build_kdtree(la_list, lo_list, z_list)
        alt_lons = sample_etopo(tree_z, zvals,
                                np.full_like(LONS, LAT_FIXED, dtype=float), LONS)  # meters

        for data_type in DATA_TYPE_LIST:
            print(f'=== Processing {data_type} for region {region_key} ===')
            arrs_mm = load_monthly_ave(data_type)

            # 경도별(0.01° 간격) 월별 값 벡터(길이 12) 계산
            monthly_data_all_lons = []
            for lo in LONS:
                monthly_data_all_lons.append(
                    radius_mean(arrs_mm, lat_arr, lon_arr, LAT_FIXED, lo, radius_deg=RADIUS_DEG)
                )
            monthly_data_all_lons = np.stack(monthly_data_all_lons, axis=0)  # (n_lon, 12)

            # y축 범위 설정: U 지역 & GPP일 때 0–0.4 고정
            y_limits = (0.0, 0.4) if (region_key == 'U' and data_type == 'GPP') else None

            # 월별 시계열(색=고도)
            outpath_monthly = os.path.join(
                FIG_BASE_PATH,
                f'ML_{data_type}_{region_key}_{LAT_FIXED:.1f}N_{LON_START}-{LON_END}E_monthly_colorelev.png'
            )
            plot_monthly_timeseries_colored(
                LONS, monthly_data_all_lons, alt_lons,
                title=f"{TITLE_PREF} — {LON_START}°E to {LON_END}°E\nMonthly mean across years",
                ylabel=YLABEL.get(data_type, data_type),
                outpath=outpath_monthly,
                y_limits=y_limits
            )

            # 시즌 평균(Cold / Warm)
            cold_mean = np.nanmean(monthly_data_all_lons[:, COLD_MONTH_IDX], axis=1)
            warm_mean = np.nanmean(monthly_data_all_lons[:, WARM_MONTH_IDX], axis=1)
            outpath_season = os.path.join(
                FIG_BASE_PATH,
                f'ML_{data_type}_{region_key}_{LAT_FIXED:.1f}N_{LON_START}-{LON_END}E_seasonal_mean_colorelev.png'
            )
            plot_seasonal_means_vs_lon(
                LONS, cold_mean, warm_mean, alt_lons,
                title=f"{TITLE_PREF} — {LON_START}°E to {LON_END}°E\nSeasonal mean across years",
                ylabel=YLABEL.get(data_type, data_type),
                outpath=outpath_season,
                y_limits=y_limits
            )

if __name__ == "__main__":
    main()

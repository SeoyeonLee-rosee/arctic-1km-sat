#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.spatial import cKDTree

# ------------------------------
# 경로 설정
# ------------------------------
ml_data_path   = '../data/MODIS/'
cru_nc_path    = '../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc'
cru_ave_path   = '../data/CRU/grid/ave_gr.npy'     # (12, lat, lon)
ml_ave0p5_path = '../data/MODIS/ave_0.5.npy'       # (12, lat, lon) 0.5°
etopo_path     = '/data1/DATA_ARCHIVE/ETOPO2022/'
fig_save_path  = '../FIG/scatter/'

os.makedirs(fig_save_path, exist_ok=True)

# ------------------------------
# ETOPO 유틸
# ------------------------------
def get_etopo_tile_names(lat_range, lon_range):
    def lat_str(deg): return f"N{abs(int(deg))//15*15:02d}" if deg >= 0 else f"S{abs(int(deg))//15*15:02d}"
    def lon_str(deg): return f"E{abs(int(deg))//15*15:03d}" if deg >= 0 else f"W{abs(int(deg))//15*15:03d}"
    lat_vals = range(int(lat_range[0])//15*15, int(lat_range[1]) + 15, 15)
    lon_vals = range(int(lon_range[0])//15*15, int(lon_range[1]) + 15, 15)
    return [f"ETOPO_2022_v1_15s_{lat_str(lat)}{lon_str(lon)}_surface.nc" for lat in lat_vals for lon in lon_vals]

def load_and_merge_etopo_tiles(tile_dir, tile_names):
    lat_all, lon_all, elev_all = [], [], []
    for name in tile_names:
        path = os.path.join(tile_dir, name)
        if not os.path.exists(path):
            print(f"❗ 누락: {path}")
            continue
        print(f"📥 로딩: {path}")
        with Dataset(path) as nc:
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
            if 'z' not in nc.variables:
                raise KeyError(f"❌ 'z' 변수 없음: {path}")
            z = nc.variables['z'][:].astype(np.float32)
            z[z <= 0] = np.nan  # 바다 마스크
            lat_all.append(lat)
            lon_all.append(lon)
            elev_all.append(z)
    return lat_all, lon_all, elev_all

def build_etopo_kdtree(lat_arrs, lon_arrs, elev_arrs):
    pts, vals = [], []
    for lat, lon, elev in zip(lat_arrs, lon_arrs, elev_arrs):
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
        pts.append(np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1))
        vals.append(elev.ravel())
    all_points = np.concatenate(pts, axis=0)
    all_values = np.concatenate(vals, axis=0)
    return cKDTree(all_points), all_values

def sample_altitude(tree, values, lat_grid, lon_grid):
    coords = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)
    mask = np.all(np.isfinite(coords), axis=1)
    out = np.full(coords.shape[0], np.nan, dtype=np.float32)
    if np.any(mask):
        _, idx = tree.query(coords[mask])
        out[mask] = values[idx]
    return out.reshape(lat_grid.shape)

# ------------------------------
# CRU 위경도 (0.5°) 읽기 & 위도마스크
# ------------------------------
with xr.open_dataset(cru_nc_path) as ds:
    cru_lat_all = ds['lat'].values
    cru_lon_all = ds['lon'].values
cru_lon_all = np.where(cru_lon_all > 180, cru_lon_all - 360, cru_lon_all)
lat_mask = cru_lat_all >= 60
lat_60 = cru_lat_all[lat_mask]
lon_all = cru_lon_all

print(f"✅ lat_60 shape: {lat_60.shape}, lon_all shape: {lon_all.shape}")

# ------------------------------
# 시즌 평균 유틸
# ------------------------------
COLD_IDX = [10, 11, 0, 1, 2]          
WARM_IDX = [3, 4, 5, 6, 7, 8, 9]       

def season_mean(arr12):
    cold = np.nanmean(arr12[COLD_IDX], axis=0)
    warm = np.nanmean(arr12[WARM_IDX], axis=0)
    return cold, warm

# ------------------------------
# CRU & ML 데이터 로드 (조건부 lat_mask)
# ------------------------------
cru_mon_full = np.load(cru_ave_path).astype(np.float32)
ml_mon_full  = np.load(ml_ave0p5_path).astype(np.float32)

if cru_mon_full.shape[1] == cru_lat_all.size:
    cru_mon = cru_mon_full[:, lat_mask, :]
    ml_mon  = ml_mon_full[:,  lat_mask, :]
else:
    cru_mon = cru_mon_full
    ml_mon  = ml_mon_full

# ------------------------------
# 시즌 평균 및 차이
# ------------------------------
cru_cold, cru_warm = season_mean(cru_mon)
ml_cold,  ml_warm  = season_mean(ml_mon)

diff_cold = ml_cold - cru_cold
diff_warm = ml_warm - cru_warm

print(f"✅ ML-CRU cold min/max: {np.nanmin(diff_cold)}, {np.nanmax(diff_cold)}")
print(f"✅ ML-CRU warm min/max: {np.nanmin(diff_warm)}, {np.nanmax(diff_warm)}")

# ------------------------------
# ETOPO → CRU 0.5° 격자 샘플링
# ------------------------------
lat_range = (float(lat_60.min()), float(lat_60.max()))
lon_range = (float(lon_all.min()), float(lon_all.max()))
tile_list = get_etopo_tile_names(lat_range, lon_range)

lat_arrs, lon_arrs, elev_arrs = load_and_merge_etopo_tiles(etopo_path, tile_list)
tree, elev_values = build_etopo_kdtree(lat_arrs, lon_arrs, elev_arrs)

lat2d, lon2d = np.meshgrid(lat_60, lon_all, indexing='ij')
altitude_0p5 = sample_altitude(tree, elev_values, lat2d, lon2d)

print(f"✅ Altitude(0.5°) min/max: {np.nanmin(altitude_0p5)}, {np.nanmax(altitude_0p5)}")

# ------------------------------
# 산점도: x=ML-CRU, y=고도, 색=위도 band
# ------------------------------
titles = ['Cold Season (Nov–Mar)', 'Warm Season (Apr–Oct)']
diff_fields = [diff_cold, diff_warm]

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
scatter_objs = []
latbands = list(range(60, 90, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(latbands)))

for ax, df, ttl in zip(axs, diff_fields, titles):
    for band in latbands:
        band_mask = (lat2d >= band) & (lat2d < band + 5)
        mask = band_mask & np.isfinite(df) & np.isfinite(altitude_0p5)
        if not np.any(mask):
            continue
        sc = ax.scatter(df[mask], altitude_0p5[mask],
                        s=3, alpha=0.5, color=colors[(band - 60)//5],
                        label=f"{band}°N–{band+5}°N")
        if ttl.startswith("Cold"):
            scatter_objs.append(sc)
    ax.set_title(ttl)
    ax.set_xlabel("ML − CRU (°C)")
    ax.grid(True)

axs[0].set_ylabel("Altitude (m)")

if scatter_objs:
    axs[0].legend(handles=scatter_objs, loc='upper left',
                  title='Latitude Bands', fontsize='small',
                  title_fontsize='medium', markerscale=5)

plt.tight_layout()
out_png = os.path.join(fig_save_path, 'scatter_ml_minus_cru_vs_altitude.png')
plt.savefig(out_png, dpi=300)
plt.close()

print(f"✅ 완료: {out_png}")

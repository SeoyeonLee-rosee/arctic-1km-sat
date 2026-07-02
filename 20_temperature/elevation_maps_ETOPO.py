#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""Arctic windows **ALTITUDE** maps (ETOPO2022 elevation)

- Reads ETOPO elevation data from multiple 15s tiles
- Masks sea (z ≤ 0) as white
- Seasonal means not needed (static topography)
- Visualizes 2 Arctic windows with 0–1400 m range
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from matplotlib.colors import Normalize
from netCDF4 import Dataset

# ------------------------------
# 사용자 설정
# ------------------------------
etopo_dir = '/data1/DATA_ARCHIVE/ETOPO2022/'
outdir = "../FIG/ALT_ETOPO/"
os.makedirs(outdir, exist_ok=True)

# 시각화 설정
alt_norm  = Normalize(vmin=0, vmax=1400)
alt_ticks = np.arange(0, 1500, 200)
alt_cmap  = plt.get_cmap("viridis")
alt_cmap.set_bad(color='white')  # 바다 마스킹

windows = [
    {"key": "E130_135_N65_70", "name": "130–135°E\n65–70°N", "lon": (130, 135),  "lat": (65, 70)},
    {"key": "E58_62_N63_67",   "name": " 58– 62°E\n63–67°N", "lon": (58, 62),    "lat": (63, 67)},
]

# ------------------------------
# 유틸: ETOPO 타일 이름 생성
# ------------------------------
def _lat_str(d): return f"N{abs(int(d))//15*15:02d}" if d >= 0 else f"S{abs(int(d))//15*15:02d}"
def _lon_str(d): return f"E{abs(int(d))//15*15:03d}" if d >= 0 else f"W{abs(int(d))//15*15:03d}"

def get_etopo_tile_names(lat_range, lon_range):
    lat_vals = range(int(lat_range[0])//15*15, int(lat_range[1])+15, 15)
    lon_vals = range(int(lon_range[0])//15*15, int(lon_range[1])+15, 15)
    return [f"ETOPO_2022_v1_15s_{_lat_str(la)}{_lon_str(lo)}_surface.nc"
            for la in lat_vals for lo in lon_vals]

# ------------------------------
# 유틸: 여러 타일 병합
# ------------------------------
def load_merged_etopo(tile_dir, names):
    lat_all, lon_all, z_all = [], [], []
    for nm in names:
        p = os.path.join(tile_dir, nm)
        if not os.path.exists(p):
            print(f"❗ 누락: {p}")
            continue
        with Dataset(p) as nc:
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
            z = nc.variables['z'][:].astype(np.float32)
            z[z <= 0] = np.nan
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            lat_all.append(LA.ravel())
            lon_all.append(LO.ravel())
            z_all.append(z.ravel())
    if not lat_all:
        return None, None, None
    return np.concatenate(lat_all), np.concatenate(lon_all), np.concatenate(z_all)

# ------------------------------
# 2. 스테이션 파일 (옵션)
# ------------------------------
NAME = pd.read_csv("../data/CRU/station/NAME.csv", encoding="cp949")
ALL = NAME[NAME.W >= 1]
WO  = NAME[NAME.W == 0]

# ------------------------------
# 3. 시각화 루프
# ------------------------------
for win in windows:
    lat_min, lat_max = win['lat']
    lon_min, lon_max = win['lon']

    tiles = get_etopo_tile_names(win['lat'], win['lon'])
    la, lo, zz = load_merged_etopo(etopo_dir, tiles)

    if la is None:
        print(f"❌ No grid points in window: {win['key']}")
        continue

    msk = (la >= lat_min) & (la <= lat_max) & \
          (lo >= lon_min) & (lo <= lon_max)

    if np.sum(msk) == 0:
        print(f"❌ No points inside window mask: {win['key']}")
        continue

    fig = plt.figure(figsize=(6, 4))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min+0.01, lon_max-0.01, lat_min+0.01, lat_max-0.01], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2)

    ax.set_xticks(np.arange(lon_min, lon_max+0.1, 2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(lat_min, lat_max+0.1, 2), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(labelsize=8, direction='out')

    sc = ax.scatter(lo[msk], la[msk], c=zz[msk], cmap=alt_cmap,
                    norm=alt_norm, s=1, transform=ccrs.PlateCarree())

    sel = lambda df: df.LON.between(lon_min, lon_max) & df.LAT.between(lat_min, lat_max)
    ax.scatter(WO.loc[sel(WO), 'LON'], WO.loc[sel(WO), 'LAT'], marker='o', c='k', s=4,
               transform=ccrs.PlateCarree())
    ax.scatter(ALL.loc[sel(ALL), 'LON'], ALL.loc[sel(ALL), 'LAT'], marker='o', c='k', s=4,
               transform=ccrs.PlateCarree())

    # ✅ 컬러바: 수직
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical',
                        fraction=0.046, pad=0.04, ticks=alt_ticks)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(f"ETOPO altitude – {win['name'].replace(chr(10), ' – ')}", fontsize=10, pad=6)

    out_png = os.path.join(outdir, f"{win['key']}_ETOPO_ALT.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved:", out_png)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""LNT, LDT, LNA, LDA 22-yr mean (34 patches) → 3 Arctic windows + station scatter."""

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
import cartopy.mpl.ticker as cticker


# ───────── 1. 데이터 읽기 및 초기 설정 ─────────────────────────────────
lat = np.load("../RAW/lat.npy")          # (34,1200,1200)
lon = np.load("../RAW/lon.npy")

variables = ['LNt', 'LDt', 'LNa', 'LDa']
months = [f"{m:02d}" for m in range(1, 13)]

# ───────── 2. 관측소 CSV ───────────────────────────────────────────────
NAME = pd.read_csv("../data/CRU/station/NAME.csv", encoding="cp949")
WO   = NAME[NAME["W"] == 0]
ALL  = NAME[NAME["W"] >= 1]

# ───────── 3. 창(window) 정의 ──────────────────────────────────────────
windows = [
    {"key":"E120_150_N60_70","name":"120–150°E\n60–70°N","lon":(120,150),"lat":(60,70)},
    {"key":"E55_65_N60_70",  "name":" 55– 65°E\n60–70°N","lon":(55,65), "lat":(60,70)},
    {"key":"E20_30_N60_65",  "name":" 20– 30°E\n60–65°N","lon":(20,30), "lat":(60,65)},
    {"key":"E90_120_N60_70","name":"90–120°E\n60–70°N","lon":(90,120),"lat":(60,70)},
]

# ───────── 4. 색 스케일 ───────────────────────────────────────────────
norm_main  = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
ticks_main = [-10,-5,0,5,10]
#norm_main  = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)
#ticks_main = [-20,-10,0,10,20]

outdir = "../FIG_new/"; os.makedirs(outdir, exist_ok=True)
#outdir = "../FIG/"; os.makedirs(outdir, exist_ok=True)

# ───────── 5. 이중루프(변수, 월) ────────────────────────────────────────────
for var in variables:
    for mm in months:
        filepath = f"../RAW/{var}_{mm}.npy"
        data = np.load(filepath)

        # 빈 데이터 예외처리 추가
        if data.size == 0 or np.all(np.isnan(data)):
            print(f"Empty or all-NaN data skipped: {filepath}")
            continue

        data_mean = np.nanmean(data, axis=0)

        for win in windows:
            fig = plt.figure(figsize=(6,4))
            ax  = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([*win["lon"], *win["lat"]], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
            ax.add_feature(cfeature.BORDERS, linewidth=0.2)

            ax.set_xticks(np.arange(win["lon"][0], win["lon"][1]+1, 5), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(win["lat"][0], win["lat"][1]+1, 2), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
            ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
            ax.tick_params(labelsize=8, direction='out')

            for p in range(34):
                mask = (lon[p] >= win["lon"][0]) & (lon[p] <= win["lon"][1]) & \
                       (lat[p] >= win["lat"][0]) & (lat[p] <= win["lat"][1])
                if not mask.any(): continue

                rows, cols = np.where(mask)
                r0,r1 = rows.min(), rows.max()+1
                c0,c1 = cols.min(), cols.max()+1

                lon_sub = lon[p][r0:r1, c0:c1]
                lat_sub = lat[p][r0:r1, c0:c1]
                fld_sub = data_mean[p][r0:r1, c0:c1]

                ax.pcolormesh(
                    lon_sub, lat_sub, fld_sub,
                    cmap="coolwarm", norm=norm_main,
                    shading="auto", transform=ccrs.PlateCarree()
                )

            sel = lambda df: df["LON"].between(*win["lon"]) & df["LAT"].between(*win["lat"])
            ax.scatter(WO.loc[sel(WO),"LON"],  WO.loc[sel(WO),"LAT"],
                       transform=ccrs.PlateCarree(), marker="o", c="k", s=4)
            ax.scatter(ALL.loc[sel(ALL),"LON"], ALL.loc[sel(ALL),"LAT"],
                       transform=ccrs.PlateCarree(), marker="o", c="k", s=4)

            ax.set_title(f"{win['name'].replace(chr(10),' – ')} ({var}, {mm})", fontsize=10, pad=6)

            out = os.path.join(outdir, f"{win['key']}_{var}_mean_{mm}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("Saved:", out)
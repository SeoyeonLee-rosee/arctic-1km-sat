#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
CRU vs MODIS – winter & summer
· 3 개의 북극 소영역에 대해 영역별 PNG 저장
  행 = 겨울·여름,  열 = CRU | MODIS | MODIS-CRU
· 컬러바 : 패널 아래 가로, 0 °C 중심 diverging
    └ CRU·MODIS  → −45 … +20 °C  (ticks −45 −30 −15 0 15 20)
    └ MOD-CRU    → −15 … +15 °C  (ticks −15 −10 −5 0 5 10 15)
· MOD-CRU 패널에 관측소 산점도 추가
    – 검은 ○ : W = 0,   빨강 ● : W ≥ 1
"""

import os, numpy as np, pandas as pd, xarray as xr, matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm

# ────────────────────────── 1. 좌표 ──────────────────────────────────────
nc = "../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc"
with xr.open_dataset(nc) as ds:
    lat_all = ds["lat"].values            # 360
    lon_all = ds["lon"].values            # 720
lon_all = np.where(lon_all > 180, lon_all - 360, lon_all)
lat_60  = lat_all[lat_all >= 60]          # 60줄 (60–89.5 °N)

# ────────────────────────── 2. 관측소 CSV ────────────────────────────────
name_csv = "../data/CRU/station/NAME.csv"
NAME = pd.read_csv(name_csv, encoding="cp949")
WO   = NAME[NAME["W"] == 0]    # black
ALL  = NAME[NAME["W"] >= 1]    # red

# ────────────────────────── 3. 자료 로드 & 시즌 평균 ─────────────────────
def load_season(fp):
    arr = np.load(fp).astype(np.float32)         # (12, lat, lon)
    if arr.shape[1] == lat_all.size:             # 360줄 → 60줄 슬라이스
        arr = arr[:, lat_all >= 60, :]
    idx = {"winter": [10, 11, 0, 1, 2],
           "summer": list(range(3, 10))}
    return {s: np.nanmean(arr[i], axis=0) for s, i in idx.items()}

cru  = load_season("../data/CRU/grid/ave_gr.npy")
mod  = load_season("../data/MODIS/ave_0.5.npy")
diff = {s: mod[s] - cru[s] for s in cru}

# ────────────────────────── 4. 컬러 스케일 ───────────────────────────────
norm_main  = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
ticks_main = [-40, -20, 0, 20, 40]

norm_diff  = TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)
ticks_diff = [-15, -10, -5, 0, 5, 10, 15]

# ────────────────────────── 5. 영역 설정 ────────────────────────────────
windows = [
    {"key": "E120_150_N60_70", "name": "120–150°E\n60–70°N",
     "lon": (120, 150), "lat": (60, 70)},
    {"key": "E55_65_N60_70",   "name": " 55– 65°E\n60–70°N",
     "lon": (55, 65),   "lat": (60, 70)},
    {"key": "E20_30_N60_65",   "name": " 20– 30°E\n60–65°N",
     "lon": (20, 30),   "lat": (60, 65)},
]

outdir = "../FIG/"; os.makedirs(outdir, exist_ok=True)

# ────────────────────────── 6. 그리기 루프 ───────────────────────────────
for win in windows:
    lat_m = (lat_60 >= win["lat"][0]) & (lat_60 <= win["lat"][1])
    lon_m = (lon_all >= win["lon"][0]) & (lon_all <= win["lon"][1])

    fig, axes = plt.subplots(
        2, 3, figsize=(10, 6),
        subplot_kw=dict(projection=ccrs.PlateCarree()),
        constrained_layout=True
    )

    for r, season in enumerate(["winter", "summer"]):
        dataset = {
            "CRU":        cru [season][np.ix_(lat_m, lon_m)],
            "MODIS":      mod [season][np.ix_(lat_m, lon_m)],
            "MOD − CRU":  diff[season][np.ix_(lat_m, lon_m)],
        }

        for c, (ttl, fld) in enumerate(dataset.items()):
            ax = axes[r, c]
            ax.set_extent([*win["lon"], *win["lat"]], ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
            ax.add_feature(cfeature.BORDERS,   linewidth=0.2)

            if np.isnan(fld).all():
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="red")
                continue

            is_diff = ("MOD" in ttl and "CRU" in ttl)
            norm  = norm_diff  if is_diff else norm_main
            ticks = ticks_diff if is_diff else ticks_main

            im = ax.pcolormesh(
                lon_all[lon_m], lat_60[lat_m], fld,
                cmap="coolwarm", norm=norm,
                shading="auto", transform=ccrs.PlateCarree()
            )

            # ── 관측소 산점도 (MOD-CRU 패널만) ─────────────────
            if is_diff:
                filt = lambda df: (
                    df["LON"].between(*win["lon"]) & df["LAT"].between(*win["lat"])
                )
                ax.scatter(WO.loc[filt(WO), "LON"], WO.loc[filt(WO), "LAT"],
                           transform=ccrs.PlateCarree(),
                           marker="o", c="k", s=4, zorder=4)
                ax.scatter(ALL.loc[filt(ALL), "LON"], ALL.loc[filt(ALL), "LAT"],
                           transform=ccrs.PlateCarree(),
                           marker="o", c="k", s=4, zorder=4)
            # ────────────────────────────────────────────────

            if r == 0:
                ax.set_title(ttl, fontsize=10, pad=4)

#            if c == 0:
#                ax.text(0.02, 0.98, f"{season.upper()}\n{win['name']}",
#                        transform=ax.transAxes, va="top", ha="left",
#                        fontsize=8, bbox=dict(fc="w", alpha=0.65, lw=0))

            gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                              color='gray', alpha=0.5, linestyle="--")
            gl.top_labels = gl.right_labels = False
            gl.xlabel_style = gl.ylabel_style = {"fontsize": 6}

            plt.colorbar(im, ax=ax, orientation="horizontal",
                         fraction=0.046, pad=0.06, ticks=ticks
                         ).ax.tick_params(labelsize=6)

    fig.suptitle(f"{win['name'].replace(chr(10),' ')}  -  CRU vs MODIS", fontsize=12)
    outfile = os.path.join(outdir, f"{win['key']}_CRU_MODIS_ws.png")
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outfile)


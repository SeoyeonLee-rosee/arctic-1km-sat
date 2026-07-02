#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Monthly climatology of CRU temperature – 3 Arctic windows
Generates monthly mean temperature for CRU TS dataset for each window.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
import cartopy.mpl.ticker as cticker

# ────────────────────────── 1. Load CRU dataset ────────────────────────────
nc = "../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc"
with xr.open_dataset(nc) as ds:
    # adjust longitudes to [-180,180]
    lon_all = ds.lon.values
    lon_all = np.where(lon_all > 180, lon_all - 360, lon_all)
    lat_all = ds.lat.values
    tmp = ds.tmp.values        # (time, lat, lon)
    months_all = ds['time.month'].values  # (time,)

# select Arctic (>60°N)
lat_mask = lat_all >= 60
lat_arctic = lat_all[lat_mask]

# ────────────────────────── 2. Compute monthly climatology ─────────────────
# monthly: list of 12 arrays (lat_arctic, lon)
monthly = []
for m in range(1, 13):
    sel_time = months_all == m
    if not np.any(sel_time):
        # no data for this month
        monthly.append(np.full((lat_arctic.size, lon_all.size), np.nan))
    else:
        data_m = tmp[sel_time, :, :]               # (n_time, lat, lon)
        data_m = data_m[:, lat_mask, :]            # select arctic lat
        monthly.append(np.nanmean(data_m, axis=0)) # (lat_arctic, lon)
monthly = np.stack(monthly, axis=0)               # shape (12, lat_arctic, lon_all)

# ────────────────────────── 3. Station data ─────────────────────────────────
name_csv = "../data/CRU/station/NAME.csv"
NAME = pd.read_csv(name_csv, encoding="cp949")
WO  = NAME[NAME["W"] == 0]
ALL = NAME[NAME["W"] >= 1]

# ────────────────────────── 4. Windows ─────────────────────────────────────
windows = [
    {"key":"E120_150_N60_70","name":"120–150°E\n60–70°N","lon":(120,150),"lat":(60,70)},
    {"key":"E55_65_N60_70",  "name":" 55– 65°E\n60–70°N","lon":(55,65), "lat":(60,70)},
    {"key":"E20_30_N60_65",  "name":" 20– 30°E\n60–65°N","lon":(20,30), "lat":(60,65)},
    {"key":"E90_120_N60_70","name":"90–120°E\n60–70°N","lon":(90,120),"lat":(60,70)},
]

# ────────────────────────── 5. Color scale ─────────────────────────────────
norm_main = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
ticks_main = [-10, -5, 0, 5, 10]
#norm_main = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)
#ticks_main = [-20, -10, 0, 10, 20]

# ────────────────────────── 6. Plot loop ───────────────────────────────────
outdir = "../FIG_new/CRU_MONTHLY/"
#outdir = "../FIG/CRU_MONTHLY/"

os.makedirs(outdir, exist_ok=True)

for win in windows:
    # masks for window
    lon_mask = (lon_all >= win["lon"][0]) & (lon_all <= win["lon"][1])
    lat_mask_win = (lat_arctic >= win["lat"][0]) & (lat_arctic <= win["lat"][1])

    for month_idx in range(1, 13):
        data = monthly[month_idx-1]                        # (lat_arctic, lon)
        sub = data[np.ix_(lat_mask_win, lon_mask)]         # window subset
        lon_sub = lon_all[lon_mask]
        lat_sub = lat_arctic[lat_mask_win]

        fig = plt.figure(figsize=(6, 4))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([*win["lon"], *win["lat"]], ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.2)

        ax.set_xticks(np.arange(win["lon"][0], win["lon"][1] + 1, 5), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(win["lat"][0], win["lat"][1] + 1, 2), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        ax.tick_params(labelsize=8, direction='out')

        im = ax.pcolormesh(
            lon_sub, lat_sub, sub,
            cmap="coolwarm", norm=norm_main,
            shading="auto", transform=ccrs.PlateCarree()
        )

        sel = lambda df: df["LON"].between(*win["lon"]) & df["LAT"].between(*win["lat"])
        ax.scatter(WO.loc[sel(WO), "LON"],  WO.loc[sel(WO), "LAT"],
                   transform=ccrs.PlateCarree(), marker="o", c="k", s=4)
        ax.scatter(ALL.loc[sel(ALL), "LON"], ALL.loc[sel(ALL), "LAT"],
                   transform=ccrs.PlateCarree(), marker="o", c="k", s=4)

        ax.set_title(f"{win['name'].replace(chr(10),' – ')} (CRU, Month {month_idx:02d})", fontsize=10, pad=6)
        cb = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.1, ticks=ticks_main)
        cb.ax.tick_params(labelsize=6)

        outfile = os.path.join(outdir, f"{win['key']}_CRU_month_{month_idx:02d}.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", outfile)
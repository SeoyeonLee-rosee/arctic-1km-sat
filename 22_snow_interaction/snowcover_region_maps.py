#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
SCE_3.py

SCE_2.py 에서 생성한

  - SCE_ave.npy
  - SCE_trend_per_decade.npy
  - (optionally SCS_ave.npy, SCS_trend_per_decade.npy)

와 lat/lon (34,1200,1200)를 이용해서

  - Ural
  - Verkhoyansk
  - Alaska

세 영역에 대해 SCE/SCS의 ave (DOY)와 trend (days/decade) 지도를
Cartopy PlateCarree 위에 scatter 형태로 플롯하여 PNG로 저장한다.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

# ----------------- Paths & config -----------------
BASE = "../data/MOD/SCE_SCS"
LAT_FILE = os.path.join(BASE, "lat.npy")
LON_FILE = os.path.join(BASE, "lon.npy")

OUT_DIR = "./FIG_SCE_SCS_REGIONS_LL"
os.makedirs(OUT_DIR, exist_ok=True)

TILES = 34
H = 1200
W = 1200

# 세 영역 정의 (lat_min, lat_max, lon_min, lon_max)
REGIONS = [
    dict(name="Ural",        lat_min=62.0,  lat_max=67.0,  lon_min=58.0,   lon_max=63.0),
    dict(name="Verkhoyansk", lat_min=65.0,  lat_max=70.0,  lon_min=129.0,  lon_max=134.0),
    dict(name="Alaska",      lat_min=63.5,  lat_max=68.5,  lon_min=-155.0, lon_max=-150.0),
]


# =================================================
# I/O 유틸: npy + raw memmap fallback
# =================================================
def _raw_guess_memmap_3d(path, tiles=TILES, H=H, W=W, dtype=np.float32):
    """
    np.save로 저장되지 않은 raw memmap 파일을 (tiles, H, W) float32로 복구 시도.
    """
    fsize = os.path.getsize(path)
    bsz = np.dtype(dtype).itemsize
    denom = tiles * H * W * bsz
    if denom == 0 or fsize % denom != 0:
        raise RuntimeError(f"raw memmap guess 실패: {path}, size={fsize}, denom={denom}")
    Ny = fsize // denom
    if Ny != 1:
        print(f"  ⚠ raw memmap guess: Ny={Ny}, "
              f"(tiles,H,W)=({tiles},{H},{W}) → shape=(Ny,tiles,H,W)?")
    print(f"  ⚠ raw memmap fallback for {os.path.basename(path)} "
          f"(dtype={dtype}, shape=({tiles},{H},{W}))")
    return np.memmap(path, mode="r", dtype=dtype, shape=(tiles, H, W))


def safe_load_3d(path):
    """
    (34,1200,1200) 배열을 로드. np.save 포맷이면 np.load,
    아니면 raw memmap fallback.
    """
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 3:
            raise ValueError(f"Unexpected ndim={arr.ndim} in {path}")
        if arr.shape != (TILES, H, W):
            raise ValueError(f"Unexpected shape {arr.shape} in {path}, "
                             f"expected ({TILES},{H},{W})")
        return arr
    except Exception as e:
        print(f"  ⚠ {os.path.basename(path)}: np.load 실패 ({e}), raw memmap 시도")
        return _raw_guess_memmap_3d(path, tiles=TILES, H=H, W=W, dtype=np.float32)


# =================================================
# Norm helpers (윈도우 기반 percentiles)
# =================================================
def seq_norm_window(arr, lat, lon, lat_min, lat_max, lon_min, lon_max,
                    p_lo=2, p_hi=98):
    """
    윈도우(lat/lon 범위) 안에서의 값 분포 기반 seq(단조) Normalize.
    ave(DOY)용.
    """
    m = (np.isfinite(arr) & np.isfinite(lat) & np.isfinite(lon) &
         (lat >= lat_min) & (lat <= lat_max) &
         (lon >= lon_min) & (lon <= lon_max))
    vv = arr[m]
    if vv.size < 10:
        return None
    vmin, vmax = np.percentile(vv, [p_lo, p_hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return None
    return Normalize(vmin=float(vmin), vmax=float(vmax))


def div0_norm_window(arr, lat, lon, lat_min, lat_max, lon_min, lon_max,
                     p_lo=2, p_hi=98):
    """
    트렌드용: 윈도우 안에서 p_lo~p_hi percentile 추출 후,
    max(|lo|,|hi|)을 사용해 [-m,+m] 대칭 Normalize.
    """
    m = (np.isfinite(arr) & np.isfinite(lat) & np.isfinite(lon) &
         (lat >= lat_min) & (lat <= lat_max) &
         (lon >= lon_min) & (lon <= lon_max))
    vv = arr[m]
    if vv.size < 10:
        return None
    lo, hi = np.percentile(vv, [p_lo, p_hi])
    mabs = max(abs(lo), abs(hi))
    if not np.isfinite(mabs) or mabs <= 0:
        mabs = 1.0
    return Normalize(vmin=-float(mabs), vmax=float(mabs))


# =================================================
# Cartopy axes 스타일
# =================================================
def style_geo_axes(ax, lon_min, lon_max, lat_min, lat_max):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2)


def set_window_ticks(ax, lon_min, lon_max, lat_min, lat_max):
    # 1° or 2° 간격
    dx = 1.0 if (lon_max - lon_min) <= 10 else 2.0
    dy = 1.0 if (lat_max - lat_min) <= 10 else 2.0
    ax.set_xticks(np.arange(lon_min, lon_max + 0.0001, dx),
                  crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(lat_min, lat_max + 0.0001, dy),
                  crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(labelsize=8, direction="out")


# =================================================
# Scatter plot on lat/lon using tile grid
# =================================================
def plot_region_scatter(lat_all, lon_all, field_all,
                        region_name, lat_min, lat_max, lon_min, lon_max,
                        which="ave", varname="SCE",
                        out_dir=OUT_DIR):
    """
    lat_all, lon_all, field_all: (34,1200,1200)

    which:
      - "ave"   : DOY
      - "trend" : days/decade
    """
    print(f"\n▶ Region: {region_name}, VAR={varname}, WHICH={which}")

    # norm 계산
    if which == "ave":
        norm = seq_norm_window(field_all, lat_all, lon_all,
                               lat_min, lat_max, lon_min, lon_max,
                               p_lo=2, p_hi=98)
        cmap = "viridis"
        cbar_label = "DOY"
    else:
        norm = div0_norm_window(field_all, lat_all, lon_all,
                                lat_min, lat_max, lon_min, lon_max,
                                p_lo=2, p_hi=98)
        cmap = "coolwarm"
        cbar_label = "days / decade"

    if norm is None:
        print("  ⚠ norm 계산 실패 (데이터 부족?) → 스킵")
        return

    fig = plt.figure(figsize=(5.0, 4.2))
    ax = plt.axes(projection=ccrs.PlateCarree())
    style_geo_axes(ax, lon_min, lon_max, lat_min, lat_max)
    set_window_ticks(ax, lon_min, lon_max, lat_min, lat_max)

    total_points = 0
    for tk in range(TILES):
        la = lat_all[tk]
        lo = lon_all[tk]
        vv = field_all[tk]

        m = (np.isfinite(la) & np.isfinite(lo) & np.isfinite(vv) &
             (la >= lat_min) & (la <= lat_max) &
             (lo >= lon_min) & (lo <= lon_max))
        if not np.any(m):
            continue

        total_points += int(m.sum())
        ax.scatter(lo[m], la[m], c=vv[m],
                   s=2.0, cmap=cmap, norm=norm,
                   transform=ccrs.PlateCarree(), linewidths=0)

    print(f"  - plotted points: {total_points}")
    if total_points == 0:
        plt.close(fig)
        print("  ⚠ region 내 유효 포인트 없음 → 스킵")
        return

    cbar = plt.colorbar(ax.collections[-1], ax=ax,
                        fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.set_label(cbar_label, fontsize=9)

    ax.set_title(f"{varname} {which} — {region_name}",
                 fontsize=10, pad=6)

    fname = f"{varname}_{which}_{region_name}.png"
    out_png = os.path.join(out_dir, fname)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ saved: {out_png}")


# =================================================
# main
# =================================================
def main():
    print("====================================")
    print(" SCE/SCS region plots on lat/lon")
    print(" BASE:", BASE)
    print(" OUT :", OUT_DIR)
    print("====================================")

    # lat/lon 로드
    print("* Loading lat/lon ...")
    lat_all = np.load(LAT_FILE)   # (34,1200,1200)
    lon_all = np.load(LON_FILE)
    if lat_all.shape != (TILES, H, W) or lon_all.shape != (TILES, H, W):
        raise ValueError(f"lat/lon shape mismatch: lat={lat_all.shape}, lon={lon_all.shape}")

    for varname in ["SCE", "SCS"]:
        ave_path = os.path.join(BASE, f"{varname}_ave.npy")
        trend_path = os.path.join(BASE, f"{varname}_trend_per_decade.npy")

        print(f"\n=== VAR={varname} ===")
        if not os.path.exists(ave_path) or not os.path.exists(trend_path):
            print(f"  ⚠ {varname} ave/trend 파일이 없음 → 스킵")
            continue

        ave_all = safe_load_3d(ave_path)
        trend_all = safe_load_3d(trend_path)

        for R in REGIONS:
            name = R["name"]
            lat_min = R["lat_min"]; lat_max = R["lat_max"]
            lon_min = R["lon_min"]; lon_max = R["lon_max"]

            # ave
            plot_region_scatter(lat_all, lon_all, ave_all,
                                region_name=name,
                                lat_min=lat_min, lat_max=lat_max,
                                lon_min=lon_min, lon_max=lon_max,
                                which="ave", varname=varname,
                                out_dir=OUT_DIR)

            # trend
            plot_region_scatter(lat_all, lon_all, trend_all,
                                region_name=name,
                                lat_min=lat_min, lat_max=lat_max,
                                lon_min=lon_min, lon_max=lon_max,
                                which="trend", varname=varname,
                                out_dir=OUT_DIR)

    print("\n====================================")
    print(" All region plots done. 🎉")
    print(" Output dir:", OUT_DIR)
    print("====================================")


if __name__ == "__main__":
    main()


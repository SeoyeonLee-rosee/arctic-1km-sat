#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
gpp_max_3.py
=========================================
GPP MAX — PART 3: MAPS & REGIONS
=========================================

입력 (from gpp_max_1.py & gpp_max_2.py)
---------------------------------------
기본 경로: ../FIGs/GPP_MAX/

  - gpp_years.npy
      (Ny,)            # 예: 2000–2022

  - gpp_max_yearly.npy
      (tiles, Ny, H, W)   # 연도별 grid GPP MAX  (여기서 이미 GPP<1 필터 적용됨)

  - gpp_max_month.npy
      (tiles, Ny, H, W)   # 해당 연도 MAX가 나타난 month (1~12, 0=없음)

  - gpp_max_trend_decade.npy
      (tiles, H, W)       # trend (per decade)

  - gpp_max_trend_ci_low.npy
      (tiles, H, W)       # trend 95% CI 하한 (per decade)

  - gpp_max_trend_ci_high.npy
      (tiles, H, W)       # trend 95% CI 상한 (per decade)

  - ../data/MOD/lat.npy, lon.npy
      (tiles, H, W)

출력 (../FIGs/GPP_MAX/)
-----------------------
  - GPP_MAX_mean_window_*.png       # window별 GPP MAX (time-mean) 지도
  - GPP_MAX_trend_window_*.png      # window별 trend(per decade) 지도 (CI 점 없음)
  - GPP_MAX_monthmode_window_*.png  # window별 month-of-MAX mode 지도

  - GPP_MAX_violin_mean_by_region.png    # 세 window별 GPP MAX 분포(violin)
  - GPP_MAX_violin_trend_by_region.png   # 세 window별 trend 분포(violin)
  - GPP_MAX_hist_monthmode_by_region.png # 세 window별 month-of-MAX mode 히스토그램
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

# ---------------------------------------
# Paths
# ---------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/ 기준

DATA_MOD = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "MOD"))
IN_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "FIGs", "GPP_MAX"))
OUT_DIR  = IN_DIR

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------
# Windows / Regions
# ---------------------------------------
WINDOWS = [
    {
        "key": "Ural",
        "name": "Ural (58–63°E, 62–67°N)",
        "lon": (58.0, 63.0),
        "lat": (62.0, 67.0),
    },
    {
        "key": "Verkhoyansk",
        "name": "Verkhoyansk (129–134°E, 65–70°N)",
        "lon": (129.0, 134.0),
        "lat": (65.0, 70.0),
    },
    {
        "key": "Alaska",
        "name": "Alaska (150–155°W, 63.5–68.5°N)",
        "lon": (-155.0, -150.0),
        "lat": (63.5, 68.5),
    },
]

# ---------------------------------------
# Helpers: lat/lon & core arrays
# ---------------------------------------
def load_latlon():
    lat_p = os.path.join(DATA_MOD, "lat.npy")
    lon_p = os.path.join(DATA_MOD, "lon.npy")
    if not (os.path.exists(lat_p) and os.path.exists(lon_p)):
        raise FileNotFoundError(f"lat.npy / lon.npy not found under {DATA_MOD}")
    lat = np.load(lat_p).astype(np.float32)
    lon = np.load(lon_p).astype(np.float32)
    if np.nanmax(lon) > 180:
        lon = ((lon + 180.0) % 360.0) - 180.0
    print(f"* lat/lon loaded: shape={lat.shape}")
    return lat, lon

def load_core_arrays():
    print("* Loading core GPP MAX arrays ...")

    years_p   = os.path.join(IN_DIR, "gpp_years.npy")
    max_p     = os.path.join(IN_DIR, "gpp_max_yearly.npy")
    maxm_p    = os.path.join(IN_DIR, "gpp_max_month.npy")
    trend_p   = os.path.join(IN_DIR, "gpp_max_trend_decade.npy")
    ci_lo_p   = os.path.join(IN_DIR, "gpp_max_trend_ci_low.npy")
    ci_hi_p   = os.path.join(IN_DIR, "gpp_max_trend_ci_high.npy")

    if not os.path.exists(years_p):
        raise FileNotFoundError(f"gpp_years.npy not found at {years_p}")
    years = np.load(years_p).astype(np.int16)
    print(f"  - years: {years[0]}–{years[-1]} (Ny={years.size})")

    # memmap 로딩
    gpp_max        = np.load(max_p,  mmap_mode="r")     # (tiles,Ny,H,W)
    gpp_max_month  = np.load(maxm_p, mmap_mode="r")     # (tiles,Ny,H,W)
    gpp_trend      = np.load(trend_p, mmap_mode="r")    # (tiles,H,W)
    gpp_ci_low     = np.load(ci_lo_p, mmap_mode="r")    # (tiles,H,W)
    gpp_ci_high    = np.load(ci_hi_p, mmap_mode="r")    # (tiles,H,W)

    print(f"  - gpp_max_yearly shape       = {gpp_max.shape}")
    print(f"  - gpp_max_month shape        = {gpp_max_month.shape}")
    print(f"  - gpp_max_trend_decade shape = {gpp_trend.shape}")

    lat, lon = load_latlon()

    tiles, Ny, H, W = gpp_max.shape
    if lat.shape != (tiles, H, W) or lon.shape != (tiles, H, W):
        raise ValueError(f"lat/lon shape {lat.shape} not matching gpp_max {gpp_max.shape}")

    return lat, lon, years, gpp_max, gpp_max_month, gpp_trend, gpp_ci_low, gpp_ci_high

# ---------------------------------------
# Compute mean MAX & mode(month-of-MAX)
# ---------------------------------------
def compute_mean_max_and_mode_month(gpp_max, gpp_max_month):
    tiles, Ny, H, W = gpp_max.shape
    print("\n[STEP] Compute time-mean MAX & mode(month-of-MAX)")

    mean_max_tiles   = np.full((tiles, H, W), np.nan, dtype=np.float32)
    mode_month_tiles = np.zeros((tiles, H, W), dtype=np.uint8)

    for ti in range(tiles):
        print(f"  - tile {ti+1}/{tiles} ...", flush=True)

        block_max  = np.array(gpp_max[ti],       dtype=np.float32)  # (Ny,H,W)
        block_mmon = np.array(gpp_max_month[ti], dtype=np.uint8)    # (Ny,H,W)

        # mean MAX
        mean_max_tiles[ti] = np.nanmean(block_max, axis=0).astype(np.float32)

        # mode of month-of-MAX
        mode_tile = np.zeros((H, W), dtype=np.uint8)
        for j in range(H):
            row_m = block_mmon[:, j, :]  # (Ny, W)
            for k in range(W):
                series = row_m[:, k].astype(np.int16)
                series = series[(series >= 1) & (series <= 12)]
                if series.size == 0:
                    mode_tile[j, k] = 0
                else:
                    counts = np.bincount(series, minlength=13)
                    counts[0] = 0
                    mode_tile[j, k] = np.uint8(np.argmax(counts))
        mode_month_tiles[ti] = mode_tile

    return mean_max_tiles, mode_month_tiles

# ---------------------------------------
# Window mask & flatten helpers
# ---------------------------------------
def window_mask(lat, lon, win):
    lat_min, lat_max = win["lat"]
    lon_min, lon_max = win["lon"]
    m = (
        np.isfinite(lat) & np.isfinite(lon) &
        (lat >= lat_min) & (lat <= lat_max) &
        (lon >= lon_min) & (lon <= lon_max)
    )
    return m

def flatten_tiles(arr_tiles):
    return arr_tiles.reshape(-1)

def norm_sequential(vals, mask, p_lo=2, p_hi=98):
    v = vals[mask]
    v = v[np.isfinite(v)]
    if v.size < 10:
        return Normalize(vmin=np.nanmin(v) if v.size > 0 else 0.0,
                         vmax=np.nanmax(v) if v.size > 0 else 1.0)
    lo, hi = np.nanpercentile(v, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = np.nanmin(v), np.nanmax(v)
    return Normalize(vmin=float(lo), vmax=float(hi))

def norm_diverging(vals, mask, p_lo=2, p_hi=98):
    v = vals[mask]
    v = v[np.isfinite(v)]
    if v.size < 10:
        mx = np.nanmax(np.abs(v)) if v.size > 0 else 1.0
        if not np.isfinite(mx) or mx == 0:
            mx = 1.0
        return Normalize(vmin=-mx, vmax=mx)
    lo, hi = np.nanpercentile(v, [p_lo, p_hi])
    m = max(abs(lo), abs(hi))
    if not np.isfinite(m) or m == 0:
        m = 1.0
    return Normalize(vmin=-m, vmax=m)

# ---------------------------------------
# Cartopy helpers
# ---------------------------------------
def style_geo_axes(ax, win):
    lon_min, lon_max = win["lon"]
    lat_min, lat_max = win["lat"]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    ax.set_xticks(
        np.arange(lon_min, lon_max + 0.001, 2.0),
        crs=ccrs.PlateCarree(),
    )
    ax.set_yticks(
        np.arange(lat_min, lat_max + 0.001, 2.0),
        crs=ccrs.PlateCarree(),
    )
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(labelsize=8, direction="out")

# ---------------------------------------
# Mapping functions
# ---------------------------------------
def plot_window_map_scalar(lat, lon, field_tiles, win, title, out_png,
                           cmap="rainbow", diverging=False, extra_mask=None):
    print(f"  - plotting {title}  → {out_png}")

    la_flat = flatten_tiles(lat)
    lo_flat = flatten_tiles(lon)
    val_flat = flatten_tiles(field_tiles)

    lat_min, lat_max = win["lat"]
    lon_min, lon_max = win["lon"]

    m = (
        np.isfinite(la_flat) & np.isfinite(lo_flat) &
        np.isfinite(val_flat) &
        (la_flat >= lat_min) & (la_flat <= lat_max) &
        (lo_flat >= lon_min) & (lo_flat <= lon_max)
    )
    if extra_mask is not None:
        extra_flat = flatten_tiles(extra_mask.astype(bool))
        m = m & extra_flat

    if not np.any(m):
        print("    · no valid points in this window.")
        fig = plt.figure(figsize=(5, 4))
        ax = plt.axes(projection=ccrs.PlateCarree())
        style_geo_axes(ax, win)
        ax.set_title(title, fontsize=10)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="0.3")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    if diverging:
        norm = norm_diverging(val_flat, m, p_lo=2, p_hi=98)
    else:
        norm = norm_sequential(val_flat, m, p_lo=2, p_hi=98)

    fig = plt.figure(figsize=(6, 4.2))
    ax = plt.axes(projection=ccrs.PlateCarree())
    style_geo_axes(ax, win)
    ax.set_title(title, fontsize=10)

    sc = ax.scatter(lo_flat[m], la_flat[m], c=val_flat[m], s=2.0,
                    cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✔ saved {out_png} (n={int(m.sum())})")

def plot_window_map_trend(lat, lon, trend_tiles,
                          win, title, out_png, cmap="coolwarm"):
    """
    trend_tiles : (tiles,H,W) per decade
    CI는 로딩은 하지만, 그림에는 **점 표기하지 않음**.
    """
    print(f"  - plotting trend  → {out_png}")

    la_flat = flatten_tiles(lat)
    lo_flat = flatten_tiles(lon)
    tr_flat = flatten_tiles(trend_tiles)

    lat_min, lat_max = win["lat"]
    lon_min, lon_max = win["lon"]

    m = (
        np.isfinite(la_flat) & np.isfinite(lo_flat) &
        np.isfinite(tr_flat) &
        (la_flat >= lat_min) & (la_flat <= lat_max) &
        (lo_flat >= lon_min) & (lo_flat <= lon_max)
    )

    if not np.any(m):
        print("    · no valid trend in this window.")
        fig = plt.figure(figsize=(5, 4))
        ax = plt.axes(projection=ccrs.PlateCarree())
        style_geo_axes(ax, win)
        ax.set_title(title, fontsize=10)
        ax.text(0.5, 0.5, "No trend data", transform=ax.transAxes,
                ha="center", va="center", color="0.3")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    norm = norm_diverging(tr_flat, m, p_lo=2, p_hi=98)

    fig = plt.figure(figsize=(6, 4.2))
    ax = plt.axes(projection=ccrs.PlateCarree())
    style_geo_axes(ax, win)
    ax.set_title(title + " (per decade)", fontsize=10)

    sc = ax.scatter(lo_flat[m], la_flat[m], c=tr_flat[m], s=2.0,
                    cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Trend (units / decade)", fontsize=9)

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✔ saved {out_png} (n={int(m.sum())})")

def plot_window_map_month_mode(lat, lon, mode_tiles, win, title, out_png):
    print(f"  - plotting month-of-MAX mode  → {out_png}")

    la_flat = flatten_tiles(lat)
    lo_flat = flatten_tiles(lon)
    mode_flat = flatten_tiles(mode_tiles).astype(np.int16)

    lat_min, lat_max = win["lat"]
    lon_min, lon_max = win["lon"]

    m = (
        np.isfinite(la_flat) & np.isfinite(lo_flat) &
        (mode_flat >= 1) & (mode_flat <= 12) &
        (la_flat >= lat_min) & (la_flat <= lat_max) &
        (lo_flat >= lon_min) & (lo_flat <= lon_max)
    )

    if not np.any(m):
        print("    · no month-mode data in this window.")
        fig = plt.figure(figsize=(5, 4))
        ax = plt.axes(projection=ccrs.PlateCarree())
        style_geo_axes(ax, win)
        ax.set_title(title, fontsize=10)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="0.3")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:12]
    cmap = ListedColormap(colors)
    boundaries = np.arange(0.5, 12.5 + 1.0, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)

    fig = plt.figure(figsize=(6, 4.2))
    ax = plt.axes(projection=ccrs.PlateCarree())
    style_geo_axes(ax, win)
    ax.set_title(title + " (mode month of MAX)", fontsize=10)

    sc = ax.scatter(lo_flat[m], la_flat[m], c=mode_flat[m], s=3.0,
                    cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.04, ticks=range(1, 13))
    cbar.ax.set_yticklabels([str(i) for i in range(1, 13)])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Month of MAX (mode)", fontsize=9)

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✔ saved {out_png} (n={int(m.sum())})")

# ---------------------------------------
# Region-wise distributions
# ---------------------------------------
def collect_region_values(lat, lon, field_tiles, mode_month_tiles=None):
    region_vals = {}
    for win in WINDOWS:
        key = win["key"]
        m = window_mask(lat, lon, win)
        vals_max = field_tiles[m]
        vals_max = vals_max[np.isfinite(vals_max)]
        region_vals[key] = {"max": vals_max}
        if mode_month_tiles is not None:
            mm = mode_month_tiles[m].astype(np.int16)
            mm = mm[(mm >= 1) & (mm <= 12)]
            region_vals[key]["month"] = mm
    return region_vals

def collect_region_trend(lat, lon, trend_tiles):
    region_tr = {}
    for win in WINDOWS:
        key = win["key"]
        m = window_mask(lat, lon, win)
        vals = trend_tiles[m]
        vals = vals[np.isfinite(vals)]
        region_tr[key] = vals
    return region_tr

def plot_violin_by_region(data_dict, ylabel, out_png, title):
    print(f"\n[STEP] Violin plot  → {out_png}")
    regions = [w["key"] for w in WINDOWS]
    pos = np.arange(1, len(regions) + 1)

    vals_list = []
    labels = []
    for r in regions:
        arr = data_dict.get(r, np.array([]))
        arr = arr[np.isfinite(arr)]
        vals_list.append(arr)
        labels.append(r)

    fig, ax = plt.subplots(figsize=(7, 4))
    vp = ax.violinplot(
        vals_list,
        positions=pos,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    for b in vp.get("bodies", []):
        b.set_alpha(0.8)
        b.set_edgecolor("black")

    if "cmedians" in vp:
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ saved {out_png}")

def plot_month_hist_by_region(region_vals, out_png, title):
    print(f"\n[STEP] Month-of-MAX hist by region  → {out_png}")

    regions = [w["key"] for w in WINDOWS]
    fig, axes = plt.subplots(1, len(regions), figsize=(4*len(regions), 4), sharey=True)
    if len(regions) == 1:
        axes = [axes]

    bins = np.arange(0.5, 12.5 + 1.0, 1.0)
    for ax, rk in zip(axes, regions):
        mm = region_vals.get(rk, {}).get("month", np.array([], dtype=int))
        mm = mm[(mm >= 1) & (mm <= 12)]
        if mm.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="0.4")
            ax.set_xticks(range(1,13))
            ax.set_xlabel("Month")
            ax.set_title(rk)
            continue
        ax.hist(mm, bins=bins, edgecolor="black", alpha=0.7)
        ax.set_xticks(range(1,13))
        ax.set_xlabel("Month")
        ax.set_title(rk)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Count")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ saved {out_png}")

# ---------------------------------------
# main
# ---------------------------------------
def main():
    print("=================================================")
    print("      GPP MAX — PART 3: MAPS & REGIONS")
    print("=================================================\n")

    lat, lon, years, gpp_max, gpp_max_month, gpp_trend, gpp_ci_low, gpp_ci_high = load_core_arrays()

    mean_max_tiles, mode_month_tiles = compute_mean_max_and_mode_month(gpp_max, gpp_max_month)

    print("\n[STEP] Window maps")
    for win in WINDOWS:
        key = win["key"]
        long_name = win["name"]

        out1 = os.path.join(OUT_DIR, f"GPP_MAX_mean_window_{key}.png")
        plot_window_map_scalar(
            lat, lon, mean_max_tiles, win,
            title=f"GPP time-mean MAX — {long_name}",
            out_png=out1,
            cmap="viridis",
            diverging=False,
        )

        out2 = os.path.join(OUT_DIR, f"GPP_MAX_trend_window_{key}.png")
        plot_window_map_trend(
            lat, lon, gpp_trend,
            win,
            title=f"GPP MAX trend — {long_name}",
            out_png=out2,
            cmap="coolwarm",
        )

        out3 = os.path.join(OUT_DIR, f"GPP_MAX_monthmode_window_{key}.png")
        plot_window_map_month_mode(
            lat, lon, mode_month_tiles,
            win,
            title=f"GPP MAX month (mode) — {long_name}",
            out_png=out3,
        )

    print("\n[STEP] Region-wise distributions")

    region_vals = collect_region_values(lat, lon, mean_max_tiles, mode_month_tiles)
    out_v1 = os.path.join(OUT_DIR, "GPP_MAX_violin_mean_by_region.png")
    plot_violin_by_region(
        {k: v["max"] for k, v in region_vals.items()},
        ylabel="GPP time-mean MAX (native units, GPP<1)",
        out_png=out_v1,
        title="GPP time-mean MAX by region (GPP < 1.0)",
    )

    region_tr = collect_region_trend(lat, lon, gpp_trend)
    out_v2 = os.path.join(OUT_DIR, "GPP_MAX_violin_trend_by_region.png")
    plot_violin_by_region(
        region_tr,
        ylabel="GPP MAX trend (units / decade)",
        out_png=out_v2,
        title="GPP MAX trend per decade by region (GPP < 1.0)",
    )

    out_h1 = os.path.join(OUT_DIR, "GPP_MAX_hist_monthmode_by_region.png")
    plot_month_hist_by_region(
        region_vals,
        out_png=out_h1,
        title="GPP MAX month-of-MAX (mode) by region (GPP < 1.0)",
    )

    nan_ratio = np.count_nonzero(~np.isfinite(gpp_trend)) / gpp_trend.size
    print(f"\n* trend NaN ratio = {nan_ratio:.3f}")
    if nan_ratio >= 0.5:
        print("  ⚠ trend NaN 비율 ≥ 50% (이 경우 해석 시 주의 / 스킵 허용)")

    print("\n🎉 DONE gpp_max_3.py — maps & region plots saved to ../FIGs/GPP_MAX/")

if __name__ == "__main__":
    main()

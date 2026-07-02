#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
SCE_GPP_trend_scatter_altbins.py
-------------------------------------------------------------
목표:
 - SCE_trend (DOY / decade) vs GPP_trend (units / decade)
   → 지역(Ural / Verkhoyansk / Alaska)별 & 고도 bin(4구간)별 trend–trend scatter
 - 고도 bin별 평균 trend profile (altitude-gradient) 작성

입력:
  - ../data/MODIS/lat.npy, lon.npy          (tiles,H,W)
  - ../data/MOD/SCE_SCS/SCE_trend_per_decade.npy  (tiles,H,W)
  - ../FIGs/GPP_MAX/gpp_max_trend_decade.npy      (tiles,H,W)

출력 (../FIGs/ALT_INTERACT/<Region>/):
  - <Region>_SCEtrend_vs_GPPtrend_4bins.png   (2×2 panel, alt bins)
  - <Region>_trend_alt_profile.png           (altitude-gradient profile)
  - <Region>_trend_corr_table.csv            (bin별 corr)
  - <Region>_trend_alt_profile_stats.csv     (bin별 mean/std/N)
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from dataclasses import dataclass
from netCDF4 import Dataset
from scipy.spatial import cKDTree
import csv

# ---------------- Paths ----------------
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/ 기준

PATHS = {
    "ml_data_path": os.path.join(SCRIPT_DIR, "..", "data", "MODIS"),
    "sce_dir":      os.path.join(SCRIPT_DIR, "..", "data", "MOD", "SCE_SCS"),
    "gpp_dir":      os.path.join(SCRIPT_DIR, "..", "FIGs", "GPP_MAX"),
    "etopo_dir":    "/data1/DATA_ARCHIVE/ETOPO2022/",
    "outdir":       os.path.join(SCRIPT_DIR, "..", "FIGs", "ALT_INTERACT"),
}
os.makedirs(PATHS["outdir"], exist_ok=True)

# ---------------- Grid info ----------------
TILES = 34
H = 1200
W = 1200

# ---------------- Altitude bins ----------------
ALT_MIN = 0
ALT_MAX = 2000
EDGES   = np.array([0, 500, 1000, 1500, 2000], dtype=float)
BIN_LABELS = ["0–500", "500–1000", "1000–1500", "1500–2000"]

# ---------------- Correlation settings ----------------
MIN_SAMPLES = 20  # corr 계산 최소 샘플 수

# ---------------- Plot style ----------------
mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.spines.right": False,
    "axes.spines.top": False,
})

# ---------------- Regions ----------------
@dataclass
class RegionBox:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

REGIONS = [
    RegionBox("Ural",        62.0, 67.0, 58.0,   63.0),
    RegionBox("Verkhoyansk", 65.0, 70.0, 129.0,  134.0),
    RegionBox("Alaska",      63.5, 68.5, -155.0, -150.0),
]

# ============================================================
# Utilities: lon, ETOPO
# ============================================================
def _to_180(lon):
    lon = np.asarray(lon)
    return np.where(lon > 180, lon - 360, lon)

def _lat_str(d):
    return f"N{abs(int(d))//15*15:02d}" if d >= 0 else f"S{abs(int(d))//15*15:02d}"

def _lon_str(d):
    return f"E{abs(int(d))//15*15:03d}" if d >= 0 else f"W{abs(int(d))//15*15:03d}"

def get_etopo_tile_names(lat_range, lon_range):
    lat_vals = range(int(lat_range[0])//15*15, int(lat_range[1])+15, 15)
    lon_vals = range(int(lon_range[0])//15*15, int(lon_range[1])+15, 15)
    names = []
    for la in lat_vals:
        for lo in lon_vals:
            names.append(f"ETOPO_2022_v1_15s_{_lat_str(la)}{_lon_str(lo)}_surface.nc")
    return names

def load_etopo_tiles(tile_dir, names):
    lat_all, lon_all, z_all = [], [], []
    for nm in names:
        p = os.path.join(tile_dir, nm)
        if not os.path.exists(p):
            print(f"  ❗ Missing ETOPO tile: {p}")
            continue
        with Dataset(p) as nc:
            lat = nc.variables["lat"][:]
            lon = nc.variables["lon"][:]
            z   = nc.variables["z"][:].astype(np.float32)
            z[z < 0] = np.nan  # 해수면 아래 제외
            lat_all.append(lat); lon_all.append(lon); z_all.append(z)
    return lat_all, lon_all, z_all

def build_kdtree(lat_arrs, lon_arrs, z_arrs):
    pts, vals = [], []
    for la, lo, zz in zip(lat_arrs, lon_arrs, z_arrs):
        LA, LO = np.meshgrid(la, lo, indexing="ij")
        pts.append(np.stack([LA.ravel(), LO.ravel()], axis=1))
        vals.append(zz.ravel())
    P = np.concatenate(pts, axis=0)
    V = np.concatenate(vals, axis=0)
    return cKDTree(P), V

def sample_etopo_vec(tree, vals, lat_vec, lon_vec):
    xy = np.stack([lat_vec, lon_vec], axis=1)
    ok = np.all(np.isfinite(xy), axis=1)
    out = np.full(lat_vec.shape[0], np.nan, dtype=np.float32)
    if np.any(ok):
        _, idx = tree.query(xy[ok])
        out[ok] = vals[idx]
    return out

# ============================================================
# Load lat/lon (MODIS grid)
# ============================================================
def load_latlon_modis():
    lat_p = os.path.join(PATHS["ml_data_path"], "lat.npy")
    lon_p = os.path.join(PATHS["ml_data_path"], "lon.npy")
    if not (os.path.exists(lat_p) and os.path.exists(lon_p)):
        raise FileNotFoundError("MODIS lat.npy / lon.npy not found")
    lat = np.load(lat_p).astype(np.float32)
    lon = np.load(lon_p).astype(np.float32)
    lon = _to_180(lon)
    print(f"* MODIS lat/lon loaded: {lat.shape}")
    return lat, lon

# ============================================================
# Load SCE trend (tiles,H,W) – raw memmap fallback
# ============================================================
def safe_load_sce_trend(lat_shape):
    """
    SCE_trend_per_decade.npy 로더
      1) np.load(mmap_mode='r') 시도
      2) 실패하면: lat_shape 기반 float32 raw memmap으로 로딩
         (파일 크기 = tiles*H*W*4 bytes 인지 확인)
    """
    trend_p = os.path.join(PATHS["sce_dir"], "SCE_trend_per_decade.npy")
    if not os.path.exists(trend_p):
        raise FileNotFoundError(f"SCE_trend_per_decade.npy not found: {trend_p}")

    # 1) 정상 npy로 시도
    try:
        arr = np.load(trend_p, mmap_mode="r")
        if arr.ndim != 3:
            raise ValueError(f"SCE_trend ndim must be 3, got {arr.ndim}")
        print(f"* SCE_trend_per_decade loaded via np.load: shape={arr.shape}")
        return arr.astype(np.float32)
    except ValueError as e:
        # 여기로 오면 pickled data 에러일 가능성이 큼 → raw binary 가정
        print(f"  ⚠ np.load(mmap_mode='r') 실패 ({e}), "
              f"lat.shape 기반 raw memmap(float32)으로 재시도: {trend_p}")

        tiles, H_, W_ = lat_shape
        fsize = os.path.getsize(trend_p)
        itemsize = np.dtype(np.float32).itemsize
        expect = tiles * H_ * W_ * itemsize

        if fsize != expect:
            raise RuntimeError(
                f"SCE_trend_per_decade.npy raw-load size mismatch: "
                f"file={fsize} bytes, expected={expect} (= {tiles}*{H_}*{W_}*4)"
            )

        arr = np.memmap(
            trend_p,
            dtype=np.float32,
            mode="r",
            shape=(tiles, H_, W_),
        )
        print(f"  ✔ raw memmap load 성공: shape={arr.shape}, dtype=float32")
        return np.array(arr, dtype=np.float32)

    except Exception as e:
        raise RuntimeError(f"SCE_trend_per_decade.npy load 완전 실패: {e}")

# ============================================================
# Load GPP trend (tiles,H,W)
# ============================================================
def safe_load_gpp_trend():
    trend_p = os.path.join(PATHS["gpp_dir"], "gpp_max_trend_decade.npy")
    if not os.path.exists(trend_p):
        raise FileNotFoundError(f"gpp_max_trend_decade.npy not found: {trend_p}")
    arr = np.load(trend_p, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"GPP_trend ndim must be 3, got {arr.ndim}")
    print(f"* gpp_max_trend_decade loaded via np.load: shape={arr.shape}")
    return arr.astype(np.float32)

# ============================================================
# Region-wise flatten + alt
# ============================================================
def build_region_trend_data(lat, lon, sce_trend, gpp_trend, box: RegionBox,
                            etopo_tree, etopo_vals):
    """
    lat, lon     : (tiles,H,W)
    sce_trend    : (tiles,H,W)  (DOY / decade)
    gpp_trend    : (tiles,H,W)  (units / decade)

    반환:
      dict with keys:
        'alt'        : (N,) altitude
        'SCE_trend'  : (N,)
        'GPP_trend'  : (N,)
    """
    lat_flat = lat.ravel()
    lon_flat = lon.ravel()

    region_mask = (
        np.isfinite(lat_flat)
        & np.isfinite(lon_flat)
        & (lat_flat >= box.lat_min) & (lat_flat <= box.lat_max)
        & (lon_flat >= box.lon_min) & (lon_flat <= box.lon_max)
    )
    if not np.any(region_mask):
        print(f"  [{box.name}] no grid points in region.")
        return None

    lats = lat_flat[region_mask]
    lons = lon_flat[region_mask]

    # altitude from ETOPO
    alts = sample_etopo_vec(etopo_tree, etopo_vals, lats, lons)
    ok_alt = np.isfinite(alts)
    if not np.any(ok_alt):
        print(f"  [{box.name}] no valid altitude from ETOPO.")
        return None

    idx_all = np.where(region_mask)[0]
    idx_use = idx_all[ok_alt]
    alts_use = alts[ok_alt]

    sce_flat = sce_trend.ravel()[idx_use]
    gpp_flat = gpp_trend.ravel()[idx_use]

    out = {
        "alt": alts_use,
        "SCE_trend": sce_flat,
        "GPP_trend": gpp_flat,
    }
    print(f"  [{box.name}] region trend data: N={alts_use.size}")
    return out

# ============================================================
# Correlation by altitude bin
# ============================================================
def compute_trend_corr_by_bin(region_data):
    """
    region_data: dict('alt','SCE_trend','GPP_trend')
    반환:
      corr_trend[4], N_trend[4]
    """
    alt = region_data["alt"]
    sce = region_data["SCE_trend"]
    gpp = region_data["GPP_trend"]

    Nbin = len(EDGES) - 1
    corr_trend = np.full(Nbin, np.nan, dtype=np.float32)
    N_trend    = np.zeros(Nbin, dtype=int)

    for bi in range(Nbin):
        a0, a1 = EDGES[bi], EDGES[bi+1]
        bm = (alt >= a0) & (alt < a1) & np.isfinite(alt)
        if not np.any(bm):
            continue

        sce_vec = sce[bm]
        gpp_vec = gpp[bm]
        m = np.isfinite(sce_vec) & np.isfinite(gpp_vec)
        if np.count_nonzero(m) >= MIN_SAMPLES:
            X = sce_vec[m].astype(np.float64)
            Y = gpp_vec[m].astype(np.float64)
            r = np.corrcoef(X, Y)[0, 1]
            corr_trend[bi] = r
            N_trend[bi] = X.size

    return corr_trend, N_trend

# ============================================================
# Altitude-gradient profile (bin 평균)
# ============================================================
def compute_trend_alt_profile(region_data):
    """
    반환:
      alt_centers[4],
      sce_mean[4], sce_std[4], sce_N[4]
      gpp_mean[4], gpp_std[4], gpp_N[4]
    """
    alt = region_data["alt"]
    sce = region_data["SCE_trend"]
    gpp = region_data["GPP_trend"]

    Nbin = len(EDGES) - 1
    centers = (EDGES[:-1] + EDGES[1:]) / 2.0

    sce_mean = np.full(Nbin, np.nan, dtype=np.float32)
    sce_std  = np.full(Nbin, np.nan, dtype=np.float32)
    sce_N    = np.zeros(Nbin, dtype=int)
    gpp_mean = np.full(Nbin, np.nan, dtype=np.float32)
    gpp_std  = np.full(Nbin, np.nan, dtype=np.float32)
    gpp_N    = np.zeros(Nbin, dtype=int)

    for bi in range(Nbin):
        a0, a1 = EDGES[bi], EDGES[bi+1]
        bm = (alt >= a0) & (alt < a1) & np.isfinite(alt)
        if not np.any(bm):
            continue

        sce_vec = sce[bm]
        gpp_vec = gpp[bm]

        # SCE trend
        ms = np.isfinite(sce_vec)
        if np.count_nonzero(ms) > 0:
            vals = sce_vec[ms]
            sce_mean[bi] = np.nanmean(vals)
            sce_std[bi]  = np.nanstd(vals)
            sce_N[bi]    = vals.size

        # GPP trend
        mg = np.isfinite(gpp_vec)
        if np.count_nonzero(mg) > 0:
            vals = gpp_vec[mg]
            gpp_mean[bi] = np.nanmean(vals)
            gpp_std[bi]  = np.nanstd(vals)
            gpp_N[bi]    = vals.size

    return centers, sce_mean, sce_std, sce_N, gpp_mean, gpp_std, gpp_N

# ============================================================
# Plot: trend–trend scatter (2×2 panel)
# ============================================================
def plot_trend_scatter_panel(region_data, corr_trend, N_trend, region_name):
    alt = region_data["alt"]
    sce = region_data["SCE_trend"]
    gpp = region_data["GPP_trend"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)
    axes = axes.ravel()

    for bi in range(len(EDGES) - 1):
        a0, a1 = EDGES[bi], EDGES[bi+1]
        ax = axes[bi]

        bm = (alt >= a0) & (alt < a1) & np.isfinite(alt)
        if not np.any(bm):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")
            ax.set_title(f"{BIN_LABELS[bi]} m")
            continue

        sce_vec = sce[bm]
        gpp_vec = gpp[bm]
        m = np.isfinite(sce_vec) & np.isfinite(gpp_vec)
        if np.count_nonzero(m) < MIN_SAMPLES:
            ax.text(0.5, 0.5, "N<MIN", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")
            ax.set_title(f"{BIN_LABELS[bi]} m")
            continue

        X = sce_vec[m]
        Y = gpp_vec[m]

        ax.scatter(X, Y, s=6, alpha=0.5, edgecolor="none")

        # 회귀선 + corr
        r = np.corrcoef(X.astype(np.float64), Y.astype(np.float64))[0, 1]
        try:
            p = np.polyfit(X, Y, 1)
            x_line = np.linspace(np.nanmin(X), np.nanmax(X), 50)
            y_line = np.polyval(p, x_line)
            ax.plot(x_line, y_line, "r-", lw=1.2)
        except Exception:
            pass

        N_here = np.count_nonzero(m)
        ax.set_title(f"{BIN_LABELS[bi]} m  (N={N_here}, r={r:.2f})")
        ax.grid(True, alpha=0.3)
        if bi in (0, 2):
            ax.set_ylabel("GPP trend (units / decade)")
        if bi in (2, 3):
            ax.set_xlabel("SCE trend (DOY / decade)")

    fig.suptitle(f"{region_name} • SCE trend vs GPP trend by altitude bins", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    subdir = os.path.join(PATHS["outdir"], region_name)
    os.makedirs(subdir, exist_ok=True)
    fname = f"{region_name}_SCEtrend_vs_GPPtrend_4bins.png"
    out_png = os.path.join(subdir, fname)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved trend scatter panel: {out_png}")

# ============================================================
# Plot: altitude-gradient profile
# ============================================================
def plot_trend_alt_profile(centers, sce_mean, sce_std, sce_N,
                           gpp_mean, gpp_std, gpp_N,
                           region_name):
    fig, ax = plt.subplots(figsize=(7, 5))

    # SCE
    mask_s = np.isfinite(sce_mean)
    if np.any(mask_s):
        ax.errorbar(centers[mask_s], sce_mean[mask_s],
                    yerr=sce_std[mask_s],
                    fmt="o-", color="C0", label="SCE trend (DOY / decade)")

    # GPP
    mask_g = np.isfinite(gpp_mean)
    if np.any(mask_g):
        ax.errorbar(centers[mask_g], gpp_mean[mask_g],
                    yerr=gpp_std[mask_g],
                    fmt="s--", color="C1", label="GPP trend (units / decade)")

    ax.set_xlim(ALT_MIN, ALT_MAX)
    ax.set_xticks(centers)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("Altitude bin (m)")
    ax.set_ylabel("Trend (per decade)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{region_name} • SCE & GPP trend altitude-gradient")

    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    subdir = os.path.join(PATHS["outdir"], region_name)
    os.makedirs(subdir, exist_ok=True)
    out_png = os.path.join(subdir, f"{region_name}_trend_alt_profile.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved trend altitude profile: {out_png}")

    # CSV로 요약 저장
    out_csv = os.path.join(subdir, f"{region_name}_trend_alt_profile_stats.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Region", "BinIndex", "AltBin(m)",
                    "SCE_mean", "SCE_std", "SCE_N",
                    "GPP_mean", "GPP_std", "GPP_N"])
        for bi in range(len(EDGES)-1):
            a0, a1 = EDGES[bi], EDGES[bi+1]
            w.writerow([
                region_name,
                bi,
                f"{int(a0)}-{int(a1)}",
                sce_mean[bi] if np.isfinite(sce_mean[bi]) else "",
                sce_std[bi]  if np.isfinite(sce_std[bi]) else "",
                int(sce_N[bi]),
                gpp_mean[bi] if np.isfinite(gpp_mean[bi]) else "",
                gpp_std[bi]  if np.isfinite(gpp_std[bi]) else "",
                int(gpp_N[bi]),
            ])
    print(f"  saved trend altitude profile stats: {out_csv}")

# ============================================================
# Main per-region workflow
# ============================================================
def process_region_trend(box: RegionBox, lat, lon, sce_trend, gpp_trend):
    print(f"\n==============================")
    print(f" Region (trend): {box.name}")
    print(f"==============================")

    # ETOPO tree for region
    names = get_etopo_tile_names((box.lat_min, box.lat_max),
                                 (box.lon_min, box.lon_max))
    la, lo, zz = load_etopo_tiles(PATHS["etopo_dir"], names)
    if len(la) == 0:
        print(f"  No ETOPO tiles for {box.name}, skip.")
        return
    tree_z, zvals = build_kdtree(la, lo, zz)

    # region data
    region_data = build_region_trend_data(lat, lon, sce_trend, gpp_trend,
                                          box, tree_z, zvals)
    if region_data is None:
        return

    # corr by bin
    corr_trend, N_trend = compute_trend_corr_by_bin(region_data)
    print(f"  corr(SCE_trend, GPP_trend) by bin:")
    for bi in range(len(EDGES)-1):
        a0, a1 = EDGES[bi], EDGES[bi+1]
        r = corr_trend[bi]
        n = N_trend[bi]
        if n > 0 and np.isfinite(r):
            print(f"    bin {bi} ({int(a0)}-{int(a1)} m): N={n}, r={r:.2f}")
        else:
            print(f"    bin {bi} ({int(a0)}-{int(a1)} m): N={n}, r=NaN")

    # corr table CSV
    subdir = os.path.join(PATHS["outdir"], box.name)
    os.makedirs(subdir, exist_ok=True)
    out_csv = os.path.join(subdir, f"{box.name}_trend_corr_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Region", "BinIndex", "AltBin(m)", "N", "corr_SCEtrend_vs_GPPtrend"])
        for bi in range(len(EDGES)-1):
            a0, a1 = EDGES[bi], EDGES[bi+1]
            r = corr_trend[bi]
            n = N_trend[bi]
            w.writerow([
                box.name,
                bi,
                f"{int(a0)}-{int(a1)}",
                int(n),
                f"{r:.4f}" if n > 0 and np.isfinite(r) else ""
            ])
    print(f"  saved trend corr table: {out_csv}")

    # scatter panel
    plot_trend_scatter_panel(region_data, corr_trend, N_trend, box.name)

    # altitude-gradient profile
    centers, sce_mean, sce_std, sce_N, gpp_mean, gpp_std, gpp_N = \
        compute_trend_alt_profile(region_data)
    plot_trend_alt_profile(centers, sce_mean, sce_std, sce_N,
                           gpp_mean, gpp_std, gpp_N,
                           box.name)

# ============================================================
# main
# ============================================================
def main():
    print("=== SCE–GPP trend–trend altitude-bin scatter & gradient ===")
    # lat/lon
    lat, lon = load_latlon_modis()

    # SCE trend (lat.shape 기반 raw memmap fallback)
    sce_trend = safe_load_sce_trend(lat.shape)
    if sce_trend.shape != lat.shape:
        raise ValueError(f"SCE_trend shape {sce_trend.shape} vs lat {lat.shape}")

    # GPP trend
    gpp_trend = safe_load_gpp_trend()
    if gpp_trend.shape != lat.shape:
        raise ValueError(f"GPP_trend shape {gpp_trend.shape} vs lat {lat.shape}")

    for box in REGIONS:
        process_region_trend(box, lat, lon, sce_trend, gpp_trend)

    print("\n=== DONE SCE_GPP_trend_scatter_altbins.py ===")

if __name__ == "__main__":
    main()


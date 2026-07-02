#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
SAT_SCE_GPP_scatter_altbins.py
-------------------------------------------------------------
목표:
 - ML SAT 월별(mean) vs SCE_mean vs GPP_MAX_mean
 - 지역(Ural / Verkhoyansk / Alaska)별 & 고도 bin(4구간)별 상관/산점도 분석

기능:
 1) ML SAT: 1~12월 월별 mean (arr[:,5]) 로딩
 2) SCE mean (DOY): ../data/MOD/SCE_SCS/SCE_ave.npy
       - np.load 실패 시 lat.shape 기반 float32 raw memmap 로딩
 3) GPP MAX time-mean: ../FIGs/GPP_MAX/gpp_max_yearly.npy → 연도 평균
 4) ETOPO로 각 grid의 고도 샘플링 후, 4개 고도 bin:
       0–500, 500–1000, 1000–1500, 1500–2000 m
 5) 각 지역 × 고도 bin × 월에 대해:
       - corr(SAT_mm, SCE_mean)
       - corr(SAT_mm, GPP_mean)
 6) |corr| 기준 상위 3개월 자동 선택
 7) 선택된 month마다, 한 지역당 2×2 panel (4 bins) scatter plot 작성
 8) corr 값을 CSV로 저장

출력:
  - ../FIGs/ALT_INTERACT/<Region>/
       <Region>_Mxx_SAT_vs_SCE_4bins.png
       <Region>_Mxx_SAT_vs_GPP_4bins.png
       <Region>_corr_table.csv
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

# ---------------- Grid info (기본값; 실제는 lat.shape에서 확인) ----------------
TILES = 34
H = 1200
W = 1200

# ---------------- Altitude bins ----------------
ALT_MIN = 0
ALT_MAX = 2000
EDGES   = np.array([0, 500, 1000, 1500, 2000], dtype=float)
BIN_LABELS = ["0–500", "500–1000", "1000–1500", "1500–2000"]

# ---------------- Correlation settings ----------------
TOP_K_MONTHS = 3      # |corr| 기준 상위 K개월
MIN_SAMPLES  = 20     # corr 계산에 필요한 최소 sample 수

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
# Utilities: longitudes, ETOPO
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
    lon_vals = range(int(lon_range[0])//15*15, int(l_range_end := int(lon_range[1])+15), 15)
    # 위 한 줄은 가독성을 위한 치환일 뿐, 실질적 역할은 lon_range[1]+15까지 15도 step
    # Python <3.8에서는 walrus 연산자 안 쓸 거면 단순하게 다시 써도 됨

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
# Load ML monthly SAT
# ============================================================
def load_ml_monthly_sat():
    lat_p = os.path.join(PATHS["ml_data_path"], "lat.npy")
    lon_p = os.path.join(PATHS["ml_data_path"], "lon.npy")
    if not (os.path.exists(lat_p) and os.path.exists(lon_p)):
        raise FileNotFoundError("ML lat.npy / lon.npy not found")

    lat = np.load(lat_p).astype(np.float32)   # (tiles,H,W)
    lon = np.load(lon_p).astype(np.float32)
    lon = _to_180(lon)

    sat_mon = np.full((12,) + lat.shape, np.nan, dtype=np.float32)  # (12,tiles,H,W)
    found = 0
    for mm in range(1, 13):
        fp = os.path.join(PATHS["ml_data_path"], f"{mm:02d}", "ave_gr.npy")
        if not os.path.exists(fp):
            print(f"  ❗ ML missing: {fp}")
            continue
        arr = np.load(fp)  # (tiles, 6, H, W)
        if arr.shape[1] <= 5:
            raise ValueError(f"ave_gr.npy shape unexpected (no index 5): {arr.shape}")
        sat_mon[mm-1] = arr[:, 5].astype(np.float32)
        found += 1

    if found == 0:
        raise FileNotFoundError("No ML ave_gr.npy monthly files found for SAT.")

    print(f"* ML SAT monthly loaded: shape={sat_mon.shape}")
    return lat, lon, sat_mon   # (tiles,H,W), (tiles,H,W), (12,tiles,H,W)

# ============================================================
# Load SCE mean (lat.shape 기반 raw memmap fallback)
# ============================================================
def safe_load_sce_mean(lat_shape):
    """
    SCE_ave.npy 로더
      1) np.load(mmap_mode='r') 시도
      2) 실패하면: lat_shape에 맞춰 float32 raw memmap으로 로딩
         (파일 크기 = tiles*H*W*4 bytes 인지 확인)
    """
    sce_p = os.path.join(PATHS["sce_dir"], "SCE_ave.npy")
    if not os.path.exists(sce_p):
        raise FileNotFoundError(f"SCE_ave.npy not found: {sce_p}")

    # 1) 우선 정상 npy로 시도
    try:
        arr = np.load(sce_p, mmap_mode="r")
        if arr.ndim != 3:
            raise ValueError(f"SCE_ave ndim must be 3, got {arr.ndim}")
        print(f"* SCE_ave loaded via np.load: shape={arr.shape}")
        return arr.astype(np.float32)
    except ValueError as e:
        # 여기로 온다는 건 "pickled data" 에러일 가능성이 큼 → raw binary로 가정
        print(f"  ⚠ np.load(mmap_mode='r') 실패 ({e}), "
              f"lat.shape 기반 raw memmap(float32)으로 재시도: {sce_p}")

        tiles, H_, W_ = lat_shape
        fsize = os.path.getsize(sce_p)
        itemsize = np.dtype(np.float32).itemsize
        expect = tiles * H_ * W_ * itemsize

        if fsize != expect:
            raise RuntimeError(
                f"SCE_ave.npy raw-load size mismatch: file={fsize} bytes, "
                f"expected={expect} (= {tiles}*{H_}*{W_}*4)"
            )

        arr = np.memmap(
            sce_p,
            dtype=np.float32,
            mode="r",
            shape=(tiles, H_, W_),
        )
        print(f"  ✔ raw memmap load 성공: shape={arr.shape}, dtype=float32")
        return np.array(arr, dtype=np.float32)  # 일반 ndarray로 변환해서 반환

    except Exception as e:
        # 그 외 예외는 그대로 throw
        raise RuntimeError(f"SCE_ave.npy load 완전 실패: {e}")

# ============================================================
# Load GPP MAX yearly → time-mean
# ============================================================
def raw_load_4d(path, tiles=TILES, H=H, W=W):
    fsize = os.path.getsize(path)
    for dt in (np.float32, np.float64):
        b = np.dtype(dt).itemsize
        denom = tiles * H * W * b
        if denom <= 0:
            continue
        if fsize % denom != 0:
            continue
        Ny = fsize // denom
        print(f"  ⚠ RAW fallback for {os.path.basename(path)}: "
              f"dtype={dt}, shape=({tiles},{Ny},{H},{W})")
        return np.memmap(path, dtype=dt, mode="r",
                         shape=(tiles, Ny, H, W))
    raise RuntimeError(f"Cannot raw-load {path} (size={fsize})")

def load_gpp_max_time_mean():
    yearly_p = os.path.join(PATHS["gpp_dir"], "gpp_max_yearly.npy")
    if not os.path.exists(yearly_p):
        raise FileNotFoundError(f"gpp_max_yearly.npy not found: {yearly_p}")
    try:
        arr = np.load(yearly_p, mmap_mode="r")
        if arr.ndim != 4:
            raise ValueError(f"gpp_max_yearly ndim must be 4, got {arr.ndim}")
        print(f"* gpp_max_yearly loaded via np.load: shape={arr.shape}")
        yearly = arr.astype(np.float32)
    except Exception as e:
        print(f"  ⚠ np.load failed for gpp_max_yearly: {e}")
        yearly = raw_load_4d(yearly_p).astype(np.float32)

    # time-mean over year axis (axis=1)
    gpp_mean = np.nanmean(yearly, axis=1)  # (tiles,H,W)
    print(f"* GPP MAX time-mean computed: shape={gpp_mean.shape}")
    return gpp_mean

# ============================================================
# Region-wise flatten + alt
# ============================================================
def build_region_data(lat, lon, sat_mon, sce_mean, gpp_mean, box: RegionBox,
                      etopo_tree, etopo_vals):
    """
    lat, lon  : (tiles,H,W)
    sat_mon   : (12,tiles,H,W)
    sce_mean  : (tiles,H,W)
    gpp_mean  : (tiles,H,W)

    반환:
      dict with keys:
        'alt'       : (N,) altitude
        'SCE_mean'  : (N,)
        'GPP_mean'  : (N,)
        'SAT_mon'   : (12,N)
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

    # index within flattened grid
    idx_all = np.where(region_mask)[0]
    idx_use = idx_all[ok_alt]
    alts_use = alts[ok_alt]

    # variables
    sce_flat  = sce_mean.ravel()[idx_use]
    gpp_flat  = gpp_mean.ravel()[idx_use]

    # SAT monthly
    Npts = idx_use.size
    SAT_mon_mat = np.full((12, Npts), np.nan, dtype=np.float32)
    for mm in range(12):
        sat_flat = sat_mon[mm].ravel()[idx_use]
        SAT_mon_mat[mm] = sat_flat.astype(np.float32)

    out = {
        "alt": alts_use,
        "SCE_mean": sce_flat,
        "GPP_mean": gpp_flat,
        "SAT_mon": SAT_mon_mat,  # (12,N)
    }
    print(f"  [{box.name}] region data: N={Npts}")
    return out

# ============================================================
# Correlation + top-month selection
# ============================================================
def compute_corr_by_bin_region(region_data):
    """
    region_data: dict('alt', 'SCE_mean', 'GPP_mean', 'SAT_mon')
    반환:
      corr_sat_sce[12,4], N_sat_sce[12,4]
      corr_sat_gpp[12,4], N_sat_gpp[12,4]
    """
    alt = region_data["alt"]
    sce = region_data["SCE_mean"]
    gpp = region_data["GPP_mean"]
    sat_mon = region_data["SAT_mon"]  # (12,N)
    NMONTH = 12
    Nbin = len(EDGES) - 1

    corr_sat_sce = np.full((NMONTH, Nbin), np.nan, dtype=np.float32)
    corr_sat_gpp = np.full((NMONTH, Nbin), np.nan, dtype=np.float32)
    N_sat_sce    = np.zeros((NMONTH, Nbin), dtype=int)
    N_sat_gpp    = np.zeros((NMONTH, Nbin), dtype=int)

    for bi in range(Nbin):
        a0, a1 = EDGES[bi], EDGES[bi+1]
        bm = (alt >= a0) & (alt < a1) & np.isfinite(alt)
        if not np.any(bm):
            continue

        for mm in range(NMONTH):
            sat_vec = sat_mon[mm, bm]
            sce_vec = sce[bm]
            gpp_vec = gpp[bm]

            # SAT vs SCE
            m1 = np.isfinite(sat_vec) & np.isfinite(sce_vec)
            if np.count_nonzero(m1) >= MIN_SAMPLES:
                X = sat_vec[m1].astype(np.float64)
                Y = sce_vec[m1].astype(np.float64)
                r = np.corrcoef(X, Y)[0, 1]
                corr_sat_sce[mm, bi] = r
                N_sat_sce[mm, bi] = X.size

            # SAT vs GPP
            m2 = np.isfinite(sat_vec) & np.isfinite(gpp_vec)
            if np.count_nonzero(m2) >= MIN_SAMPLES:
                X = sat_vec[m2].astype(np.float64)
                Y = gpp_vec[m2].astype(np.float64)
                r = np.corrcoef(X, Y)[0, 1]
                corr_sat_gpp[mm, bi] = r
                N_sat_gpp[mm, bi] = X.size

    return corr_sat_sce, N_sat_sce, corr_sat_gpp, N_sat_gpp

def select_top_months(corr_mat):
    """
    corr_mat: (12,4)  (month, bin)
    반환: top_indices (길이 ≤12, 보통 TOP_K_MONTHS개)
    기준: 각 month에 대해 |corr|의 bin-average → 그 중 큰 순서 TOP_K_MONTHS
    """
    # bin-average of |corr| (NaN 제외)
    abs_corr = np.abs(corr_mat)
    avg_abs = np.nanmean(abs_corr, axis=1)  # (12,)

    valid_idx = np.where(np.isfinite(avg_abs))[0]
    if valid_idx.size == 0:
        return []

    # 내림차순 정렬
    sorted_idx = valid_idx[np.argsort(avg_abs[valid_idx])[::-1]]
    top = sorted_idx[:TOP_K_MONTHS]
    return list(top)

# ============================================================
# Scatter panel plotting
# ============================================================
def scatter_panel_by_altbins(region_data, pair_key, region_name, top_months):
    """
    pair_key: 'SAT_vs_SCE' 또는 'SAT_vs_GPP'
    top_months: [mm0, mm1, ...] (0-based)
    """
    alt = region_data["alt"]
    sat_mon = region_data["SAT_mon"]
    sce = region_data["SCE_mean"]
    gpp = region_data["GPP_mean"]

    pair_label = {"SAT_vs_SCE": "SAT (°C) vs SCE (DOY)",
                  "SAT_vs_GPP": "SAT (°C) vs GPP MAX (units)"}[pair_key]

    for mm in top_months:
        month_label = f"M{mm+1:02d}"
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

            sat_vec = sat_mon[mm, bm]
            if pair_key == "SAT_vs_SCE":
                y_vec = sce[bm]
                y_label = "SCE (DOY, mean)"
            else:
                y_vec = gpp[bm]
                y_label = "GPP MAX (time-mean, units)"

            valid = np.isfinite(sat_vec) & np.isfinite(y_vec)
            if np.count_nonzero(valid) < MIN_SAMPLES:
                ax.text(0.5, 0.5, "N<MIN", transform=ax.transAxes,
                        ha="center", va="center", color="0.5")
                ax.set_title(f"{BIN_LABELS[bi]} m")
                continue

            X = sat_vec[valid]
            Y = y_vec[valid]

            ax.scatter(X, Y, s=6, alpha=0.5, edgecolor="none")

            # corr + 회귀선
            r = np.corrcoef(X.astype(np.float64), Y.astype(np.float64))[0, 1]
            try:
                p = np.polyfit(X, Y, 1)
                x_line = np.linspace(np.nanmin(X), np.nanmax(X), 50)
                y_line = np.polyval(p, x_line)
                ax.plot(x_line, y_line, "r-", lw=1.2)
            except Exception:
                pass

            ax.set_title(f"{BIN_LABELS[bi]} m  (N={X.size}, r={r:.2f})")
            ax.grid(True, alpha=0.3)
            if bi in (0, 2):
                ax.set_ylabel(y_label)
            if bi in (2, 3):
                ax.set_xlabel("SAT (°C, monthly mean)")

        fig.suptitle(f"{region_name} • {pair_label} • {month_label}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        # save
        subdir = os.path.join(PATHS["outdir"], region_name)
        os.makedirs(subdir, exist_ok=True)
        fname = f"{region_name}_{month_label}_{pair_key}_4bins.png"
        out_png = os.path.join(subdir, fname)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved scatter panel: {out_png}")

# ============================================================
# Main per-region workflow
# ============================================================
def process_region(box: RegionBox, lat, lon, sat_mon, sce_mean, gpp_mean):
    print(f"\n==============================")
    print(f" Region: {box.name}")
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
    region_data = build_region_data(lat, lon, sat_mon, sce_mean, gpp_mean,
                                    box, tree_z, zvals)
    if region_data is None:
        return

    # corr 계산
    corr_sat_sce, N_sat_sce, corr_sat_gpp, N_sat_gpp = compute_corr_by_bin_region(region_data)

    # top months 선택
    top_sce = select_top_months(corr_sat_sce)
    top_gpp = select_top_months(corr_sat_gpp)

    print(f"  Top months (SAT vs SCE): {[m+1 for m in top_sce]}")
    print(f"  Top months (SAT vs GPP): {[m+1 for m in top_gpp]}")

    # CSV로 corr 테이블 저장
    subdir = os.path.join(PATHS["outdir"], box.name)
    os.makedirs(subdir, exist_ok=True)
    out_csv = os.path.join(subdir, f"{box.name}_corr_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Region", "Pair", "Month", "BinIndex", "AltBin(m)",
                    "N", "corr"])
        for mm in range(12):
            for bi in range(len(EDGES)-1):
                a0, a1 = EDGES[bi], EDGES[bi+1]
                # SAT vs SCE
                n1 = N_sat_sce[mm, bi]
                r1 = corr_sat_sce[mm, bi]
                if n1 > 0 and np.isfinite(r1):
                    w.writerow([box.name, "SAT_vs_SCE", mm+1, bi,
                                f"{int(a0)}-{int(a1)}", n1, f"{r1:.4f}"])
                # SAT vs GPP
                n2 = N_sat_gpp[mm, bi]
                r2 = corr_sat_gpp[mm, bi]
                if n2 > 0 and np.isfinite(r2):
                    w.writerow([box.name, "SAT_vs_GPP", mm+1, bi,
                                f"{int(a0)}-{int(a1)}", n2, f"{r2:.4f}"])
    print(f"  corr table saved: {out_csv}")

    # 선택된 month들에 대해 scatter panel 그림
    if len(top_sce) > 0:
        scatter_panel_by_altbins(region_data,
                                 pair_key="SAT_vs_SCE",
                                 region_name=box.name,
                                 top_months=top_sce)
    if len(top_gpp) > 0:
        scatter_panel_by_altbins(region_data,
                                 pair_key="SAT_vs_GPP",
                                 region_name=box.name,
                                 top_months=top_gpp)

# ============================================================
# main
# ============================================================
def main():
    print("=== SAT–SCE–GPP altitude-bin scatter/correlation ===")
    # 1) ML SAT monthly
    lat, lon, sat_mon = load_ml_monthly_sat()  # (tiles,H,W), (tiles,H,W), (12,tiles,H,W)

    # 2) SCE mean  (lat.shape 기반 raw memmap fallback)
    sce_mean = safe_load_sce_mean(lat.shape)
    if sce_mean.shape != lat.shape:
        raise ValueError(f"SCE_ave shape {sce_mean.shape} vs ML lat {lat.shape}")

    # 3) GPP mean
    gpp_mean = load_gpp_max_time_mean()
    if gpp_mean.shape != lat.shape:
        raise ValueError(f"GPP mean shape {gpp_mean.shape} vs ML lat {lat.shape}")

    # 각 region 처리
    for box in REGIONS:
        process_region(box, lat, lon, sat_mon, sce_mean, gpp_mean)

    print("\n=== DONE SAT_SCE_GPP_scatter_altbins.py ===")

if __name__ == "__main__":
    main()


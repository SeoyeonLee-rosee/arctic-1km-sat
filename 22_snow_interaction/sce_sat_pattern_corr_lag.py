#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
FIG7.py (SC_SAT_pattern_corr_ts_both_seasons)

목표:
- SAT (../RAW/sat_01.npy ~ sat_12.npy, shape=(34, Ny, 1200, 1200))
- SCE_yearly.npy, SCS_yearly.npy (shape=(34, Ny, 1200, 1200))
- lat.npy, lon.npy (shape=(34, 1200, 1200))

각 Region(Ural/Verkhoyansk/Alaska)에 대해
각 시즌(season = warm/cold)에 대해:

1) 연도별 공간 패턴 상관 (pattern correlation)
   - r_y( SCS(y, i,j), SAT_season(y, i,j) )
   - r_y( SCE(y, i,j), SAT_season(y, i,j) )

2) Region 평균 time series 상관 (1D 상관)
   - SAT_season_mean(y) = region 평균 SAT (해당 시즌)
   - SCE_mean(y), SCS_mean(y)

   0-lag:
     - r0(SCS_mean, SAT_mean)
     - r0(SCE_mean, SAT_mean)
     - r0(SCE_mean, SCS_mean)

   1-year lag:
     - r1(SAT_mean(y), SCS_mean(y+1))
     - r1(SAT_mean(y), SCE_mean(y+1))

시즌 정의:
- warm: 같은 해 4–10월 평균
- cold (DJF): (Y-1년 11,12 + Y년 1,2,3월) → 겨울 연도 Y

출력:
- FIG_SC_SAT_PATTERN_CORR/
    <Region>_<season>_SCE_SCS_vs_SAT_pattern_corr.png
    <Region>_<season>_TS_mean_SCE_SCS_SAT_corr.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

# ------------------- 경로 & 설정 -------------------
BASE_SCE     = "../data/MOD/SCE_SCS"
RAW_SAT_DIR  = "../RAW"           # sat_01.npy ~ sat_12.npy
FIG_DIR      = "./FIG_SC_SAT_PATTERN_CORR"
os.makedirs(FIG_DIR, exist_ok=True)

# 연도축 (SCE/SCS/SAT의 2번째 축과 맞춰야 함)
YEARS = np.arange(2000, 2022, dtype=int)  # 필요시 수정

TILES, H, W = 34, 1200, 1200

# 시즌 정의 (months는 메타 정보용)
Season = namedtuple("Season", ["key", "label", "months"])

SEASONS = [
    Season("warm", "Warm (Apr–Oct)",  [4, 5, 6, 7, 8, 9, 10]),   # 같은 연도 4~10월
    Season("cold", "Cold (Nov–Mar)", [11, 12, 1, 2, 3]),         # DJF: (Y-1년 11,12 + Y년 1,2,3)
]

# Region 정의 (lat_min, lat_max, lon_min, lon_max)
RegionBox = namedtuple("RegionBox", ["name", "lat_min", "lat_max", "lon_min", "lon_max"])

REGIONS = [
    RegionBox("Ural",        62.0, 67.0,   58.0,   63.0),
    RegionBox("Verkhoyansk", 65.0, 70.0,  129.0,  134.0),
    RegionBox("Alaska",      63.5, 68.5, -155.0, -150.0),
]

# y축 span 최소
MIN_Y_SPAN = 0.2


# ------------------- I/O helper: raw memmap 복구 -------------------
def _raw_guess_memmap(path, tiles=TILES, H=H, W=W,
                      dtypes=(np.float32, np.uint16, np.int16)):
    """
    확장자는 .npy지만 실제로는 np.save가 아니라
    np.memmap(..., dtype=dt, shape=(tiles, Ny, H, W)) 으로 만들어진
    'raw binary'일 수 있어서, 파일 크기 기반으로 Ny, dtype 추정해서 memmap 복구.
    """
    fsize = os.path.getsize(path)
    for dt in dtypes:
        bsz  = np.dtype(dt).itemsize
        denom = tiles * H * W * bsz
        if denom == 0:
            continue
        if fsize % denom != 0:
            continue
        Ny = fsize // denom
        print(f"  ? raw memmap fallback for {os.path.basename(path)} "
              f"(dtype={dt}, shape=({tiles},{Ny},{H},{W}))")
        return np.memmap(path, mode="r", dtype=dt, shape=(tiles, Ny, H, W))
    raise RuntimeError(f"Raw recovery failed for {path}: size {fsize} not compatible with {dtypes}")


def safe_open_stack(path_expected):
    """
    1) np.load(..., mmap_mode='r') 시도
    2) 실패하면 allow_pickle=True
    3) 그래도 안 되면 raw memmap 복구 (_raw_guess_memmap)
    """
    try:
        arr = np.load(path_expected, mmap_mode="r")
        if arr.ndim != 4:
            raise ValueError(f"Unexpected ndim={arr.ndim} in {path_expected}")
        return arr
    except Exception as e:
        print(f"  ? {os.path.basename(path_expected)}: np.load 실패 ({e}), raw memmap 시도")

    try:
        arr = np.load(path_expected, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.ndim == 4:
            print(f"  ? {os.path.basename(path_expected)}: allow_pickle=True로 로드")
            return arr
    except Exception:
        pass

    # raw memmap
    return _raw_guess_memmap(path_expected)


# ------------------- 유틸 함수들 -------------------
def pearsonr_nan(x, y):
    """NaN 제거 후 Pearson r 계산."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xm = x[mask] - x[mask].mean()
    ym = y[mask] - y[mask].mean()
    den = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    if den <= 0:
        return np.nan
    return float((xm * ym).sum() / den)


def nice_ylim(y, min_span=MIN_Y_SPAN):
    """상관계수 시계열용 y-lim."""
    yy = y[np.isfinite(y)]
    if yy.size == 0:
        return -1.0, 1.0
    lo = float(yy.min())
    hi = float(yy.max())
    if hi - lo < min_span:
        mid = 0.5 * (lo + hi)
        lo = mid - min_span / 2.0
        hi = mid + min_span / 2.0
    lo = max(-1.0, lo - 0.05)
    hi = min( 1.0, hi + 0.05)
    return lo, hi


def standardize_ts(ts):
    """TS를 z-score로 표준화 (mean=0, std=1)."""
    ts = np.asarray(ts, dtype=float)
    mask = np.isfinite(ts)
    out = np.full_like(ts, np.nan, dtype=float)
    if mask.sum() < 2:
        return out
    m = ts[mask].mean()
    s = ts[mask].std()
    if s == 0:
        out[mask] = 0.0
    else:
        out[mask] = (ts[mask] - m) / s
    return out


# ------------------- 1. SCE/SCS & lat/lon 로딩 -------------------
print("* Load SCE/SCS yearly & lat/lon")

sce_yearly_path = os.path.join(BASE_SCE, "SCE_yearly.npy")
scs_yearly_path = os.path.join(BASE_SCE, "SCS_yearly.npy")
lat_path        = os.path.join(BASE_SCE, "lat.npy")
lon_path        = os.path.join(BASE_SCE, "lon.npy")

# ✅ 여기서 safe_open_stack 사용
sce_yearly = safe_open_stack(sce_yearly_path)   # (34, Ny, 1200, 1200)
scs_yearly = safe_open_stack(scs_yearly_path)

lat = np.load(lat_path)  # (34,1200,1200) : 이건 np.save로 만들어졌다고 가정
lon = np.load(lon_path)

if sce_yearly.shape != scs_yearly.shape:
    raise ValueError(f"SCE_yearly {sce_yearly.shape} vs SCS_yearly {scs_yearly.shape} 불일치")

if lat.shape != (TILES, H, W) or lon.shape != (TILES, H, W):
    print(f"⚠ lat/lon shape={lat.shape}, {lon.shape}, expected {(TILES, H, W)}")

Ny_file = sce_yearly.shape[1]
if Ny_file != YEARS.size:
    print(f"⚠ YEARS.size={YEARS.size} vs file Ny={Ny_file} 불일치. 작은 쪽에 맞춰서 사용합니다.")
Ny = Ny_file
years = YEARS[:Ny]

# 경도를 -180~180으로 정규화
lon_norm = ((lon + 180.0) % 360.0) - 180.0


# ------------------- 2. Region 별 픽셀 인덱스 준비 -------------------
print("* Build region pixel index lists")

region_pixel_info = {}  # name -> dict(tile_ids, row_ids, col_ids, npoints)

for reg in REGIONS:
    tile_list = []
    row_list  = []
    col_list  = []

    for t in range(TILES):
        mask_t = (
            (lat[t] >= reg.lat_min) & (lat[t] <= reg.lat_max) &
            (lon_norm[t] >= reg.lon_min) & (lon_norm[t] <= reg.lon_max)
        )
        if not np.any(mask_t):
            continue
        iy, ix = np.where(mask_t)
        tile_list.append(np.full(iy.size, t, dtype=np.int16))
        row_list.append(iy.astype(np.int16))
        col_list.append(ix.astype(np.int16))

    if len(tile_list) == 0:
        print(f"  ⚠ {reg.name}: region 내 픽셀이 없음")
        region_pixel_info[reg.name] = None
        continue

    tile_ids = np.concatenate(tile_list)
    row_ids  = np.concatenate(row_list)
    col_ids  = np.concatenate(col_list)
    npoints  = tile_ids.size

    print(f"  - {reg.name}: Npoints = {npoints}")

    region_pixel_info[reg.name] = dict(
        tile_ids=tile_ids,
        row_ids=row_ids,
        col_ids=col_ids,
        npoints=npoints,
    )


# ------------------- 3. Region별 SCE/SCS matrix -------------------
def build_sce_scs_matrix_for_region(reg_name, Ny):
    """
    region reg_name에 대해
      SCE_mat[year, pixel], SCS_mat[year, pixel]
    (shape=(Ny, Npoints_region)) 생성.
    """
    info = region_pixel_info[reg_name]
    if info is None:
        return None, None

    tile_ids = info["tile_ids"]
    row_ids  = info["row_ids"]
    col_ids  = info["col_ids"]
    P        = info["npoints"]

    tiles_in_reg = np.unique(tile_ids)
    tile_to_idx = {t: np.where(tile_ids == t)[0] for t in tiles_in_reg}

    SCE_mat = np.full((Ny, P), np.nan, dtype=np.float32)
    SCS_mat = np.full((Ny, P), np.nan, dtype=np.float32)

    print(f"  * Build SCE/SCS matrix for region={reg_name}")

    for t in tiles_in_reg:
        idxs = tile_to_idx[t]
        rr   = row_ids[idxs]
        cc   = col_ids[idxs]

        sce_tile = sce_yearly[t]  # (Ny, 1200, 1200)
        scs_tile = scs_yearly[t]
        Ny_eff   = min(Ny, sce_tile.shape[0], scs_tile.shape[0])

        for yi in range(Ny_eff):
            SCE_mat[yi, idxs] = sce_tile[yi, rr, cc].astype(np.float32)
            SCS_mat[yi, idxs] = scs_tile[yi, rr, cc].astype(np.float32)

    return SCE_mat, SCS_mat


# ------------------- 4. Region별 SAT seasonal matrix (warm/cold) -------------------
def build_sat_seasonal_matrix_for_region(reg_name, season_key, Ny, years):
    """
    region reg_name에 대해
    SAT_season[year, pixel] (shape=(Ny, Npoints_region)) 생성.

    season_key:
      - "warm": 같은 연도 4~10월 평균
      - "cold": winter Y = (Y-1년 11,12 + Y년 1,2,3) 평균  (DJF 스타일)

    SAT 월파일은 memmap 으로 하나씩 열고, region 픽셀만 가져와서 누적.
    """

    info = region_pixel_info[reg_name]
    if info is None:
        return None

    tile_ids = info["tile_ids"]
    row_ids  = info["row_ids"]
    col_ids  = info["col_ids"]
    P        = info["npoints"]

    tiles_in_reg = np.unique(tile_ids)
    tile_to_idx  = {t: np.where(tile_ids == t)[0] for t in tiles_in_reg}

    SAT_season = np.zeros((Ny, P), dtype=np.float32)
    month_count = np.zeros(Ny, dtype=np.float32)  # 연도별 유효 월수

    # 어떤 월을 쓸지 정의
    if season_key == "warm":
        month_list = [4, 5, 6, 7, 8, 9, 10]
    elif season_key == "cold":
        # DJF: (Y-1년 11,12 + Y년 1,2,3) → winter Y
        month_list = [11, 12, 1, 2, 3]
    else:
        raise ValueError(f"Unknown season_key={season_key}")

    for mm in month_list:
        if not (1 <= mm <= 12):
            continue

        fname = os.path.join(RAW_SAT_DIR, f"sat_{mm:02d}.npy")
        if not os.path.exists(fname):
            print(f"    ❗ SAT 파일 없음 (skip): {fname}")
            continue

        print(f"    · load SAT month={mm:02d} (memmap)")
        sat_mm = np.load(fname, mmap_mode="r")  # (34, Ny_sat, 1200,1200)
        if sat_mm.shape[0] != TILES or sat_mm.shape[2:] != (H, W):
            raise ValueError(f"SAT {fname} shape 이상: {sat_mm.shape}")

        Ny_sat = sat_mm.shape[1]

        # 각 겨울연도 index(yi)에 대해, 이 월이 참조할 연도 index j를 결정
        for yi in range(Ny):
            if season_key == "warm":
                # warm: 같은 해 월 → index j = yi
                j = yi
            else:  # cold (DJF)
                if mm in (1, 2, 3):
                    j = yi          # Y년 1,2,3 → 겨울연도 Y
                else:  # 11, 12
                    j = yi - 1      # (Y-1)년 11,12 → 겨울연도 Y

            if j < 0 or j >= Ny_sat:
                continue

            for t in tiles_in_reg:
                idxs = tile_to_idx[t]
                rr   = row_ids[idxs]
                cc   = col_ids[idxs]

                tile_data = sat_mm[t]  # (Ny_sat, 1200,1200)
                vals = tile_data[j, rr, cc].astype(np.float32)  # (len(idxs),)
                SAT_season[yi, idxs] += vals

            month_count[yi] += 1.0

    # 월수로 나누기 (연도별 다를 수 있음)
    for yi in range(Ny):
        if month_count[yi] > 0:
            SAT_season[yi, :] /= month_count[yi]
        else:
            SAT_season[yi, :] = np.nan

    return SAT_season


# ------------------- 5. 메인 Region × Season 루프 -------------------
print(f"* Ny (from SCE_yearly) = {Ny}, years={years[0]}–{years[-1]}")

for reg in REGIONS:
    info = region_pixel_info.get(reg.name, None)
    if info is None or info["npoints"] == 0:
        print(f"\n=== {reg.name}: region 픽셀 없음, skip ===")
        continue

    print(f"\n=== Region: {reg.name} ===")

    # Region SCE/SCS matrix는 시즌과 무관하니 한 번만 계산
    SCE_mat, SCS_mat = build_sce_scs_matrix_for_region(reg.name, Ny)

    for season in SEASONS:
        print(f"\n  >>> Season: {season.key} ({season.label})")

        # SAT seasonal matrix
        print(f"  * Build SAT {season.key}-season matrix")
        SAT_season_mat = build_sat_seasonal_matrix_for_region(reg.name, season.key, Ny, years)
        if SAT_season_mat is None:
            print(f"  ⚠ {reg.name}, {season.key}: SAT_season_mat 없음, skip")
            continue

        # 1) 연도별 pattern correlation
        r_scs_sat = np.full(Ny, np.nan, dtype=np.float32)
        r_sce_sat = np.full(Ny, np.nan, dtype=np.float32)

        for yi in range(Ny):
            scs_vec = SCS_mat[yi]
            sce_vec = SCE_mat[yi]
            sat_vec = SAT_season_mat[yi]

            r_scs_sat[yi] = pearsonr_nan(scs_vec, sat_vec)
            r_sce_sat[yi] = pearsonr_nan(sce_vec, sat_vec)

        print(f"    - mean r(SCS, SAT_{season.key}) = {np.nanmean(r_scs_sat):.3f}")
        print(f"    - mean r(SCE, SAT_{season.key}) = {np.nanmean(r_sce_sat):.3f}")

        # Pattern correlation time series plot
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

        # (1) SCS–SAT
        ax = axes[0]
        ax.plot(years, r_scs_sat, marker="o", lw=1.5, label=f"corr(SCS, SAT_{season.key})")
        ax.axhline(0.0, color="0.4", lw=1.0, ls="--")
        ylo, yhi = nice_ylim(r_scs_sat)
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel(f"r(SCS, SAT_{season.key})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.set_title(f"{reg.name} | SAT {season.label} pattern corr")

        # (2) SCE–SAT
        ax = axes[1]
        ax.plot(years, r_sce_sat, marker="o", lw=1.5, color="C1",
                label=f"corr(SCE, SAT_{season.key})")
        ax.axhline(0.0, color="0.4", lw=1.0, ls="--")
        ylo, yhi = nice_ylim(r_sce_sat)
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel(f"r(SCE, SAT_{season.key})")
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

        fig.tight_layout()
        out_png = os.path.join(
            FIG_DIR,
            f"{reg.name}_{season.key}_SCE_SCS_vs_SAT_pattern_corr.png"
        )
        fig.savefig(out_png, dpi=250)
        plt.close(fig)
        print(f"    ✅ saved pattern-corr figure: {out_png}")

        # 2) Region 평균 time series + 0-lag / lag1 상관
        SCE_mean = np.nanmean(SCE_mat, axis=1)        # (Ny,)
        SCS_mean = np.nanmean(SCS_mat, axis=1)
        SAT_mean = np.nanmean(SAT_season_mat, axis=1)

        # z-score
        SCE_z = standardize_ts(SCE_mean)
        SCS_z = standardize_ts(SCS_mean)
        SAT_z = standardize_ts(SAT_mean)

        # 0-lag
        r0_SCS_SAT = pearsonr_nan(SCS_mean, SAT_mean)
        r0_SCE_SAT = pearsonr_nan(SCE_mean, SAT_mean)
        r0_SCE_SCS = pearsonr_nan(SCE_mean, SCS_mean)

        # 1-year lag: SAT(y) vs SCE/SCS(y+1)
        SAT0 = SAT_mean[:-1]
        SCS1 = SCS_mean[1:]
        SCE1 = SCE_mean[1:]

        r1_SCS_SAT = pearsonr_nan(SAT0, SCS1)
        r1_SCE_SAT = pearsonr_nan(SAT0, SCE1)

        print(f"    - 0-lag r(SCS_mean, SAT_{season.key}_mean) = {r0_SCS_SAT:.3f}")
        print(f"    - 0-lag r(SCE_mean, SAT_{season.key}_mean) = {r0_SCE_SAT:.3f}")
        print(f"    - 0-lag r(SCE_mean, SCS_mean)              = {r0_SCE_SCS:.3f}")
        print(f"    - lag1 r(SAT_{season.key}_mean(y), SCS_mean(y+1)) = {r1_SCS_SAT:.3f}")
        print(f"    - lag1 r(SAT_{season.key}_mean(y), SCE_mean(y+1)) = {r1_SCE_SAT:.3f}")

        # Time series plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

        ax.plot(years, SAT_z, marker="o", lw=1.5, label=f"SAT_{season.key} (z)")
        ax.plot(years, SCE_z, marker="s", lw=1.5, label="SCE_mean (z)")
        ax.plot(years, SCS_z, marker="^", lw=1.5, label="SCS_mean (z)")

        ax.axhline(0.0, color="0.4", lw=1.0, ls="--")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Year")
        ax.set_ylabel("Standardized anomaly (z-score)")
        ax.set_title(f"{reg.name} | Region-mean TS ({season.label})")
        ax.legend(loc="upper left", fontsize=8)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

        # 상관계수 텍스트 박스
        txt = (
            f"{season.key} season\n"
            f"0-lag:\n"
            f"  r(SCS, SAT) = {r0_SCS_SAT:.2f}\n"
            f"  r(SCE, SAT) = {r0_SCE_SAT:.2f}\n"
            f"  r(SCE, SCS) = {r0_SCE_SCS:.2f}\n"
            f"\nlag1 (SAT_y vs ·(y+1)):\n"
            f"  r(SAT, SCS(+1)) = {r1_SCS_SAT:.2f}\n"
            f"  r(SAT, SCE(+1)) = {r1_SCE_SAT:.2f}"
        )
        ax.text(
            0.01, 0.99, txt,
            transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
        )

        fig.tight_layout()
        out_png2 = os.path.join(
            FIG_DIR,
            f"{reg.name}_{season.key}_TS_mean_SCE_SCS_SAT_corr.png"
        )
        fig.savefig(out_png2, dpi=250)
        plt.close(fig)
        print(f"    ✅ saved TS-mean-corr figure: {out_png2}")

print("\n✅ 모든 region에 대해 warm(4–10) / cold(DJF 11–3) 패턴 + TS + lag 상관 계산 및 그림 저장 완료.")

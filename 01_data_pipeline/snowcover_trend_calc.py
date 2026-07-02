#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
SCE_2.py

SCE_1.py에서 생성한

  - SCE_yearly.npy
  - SCS_yearly.npy

(형태: (34, Ny, 1200, 1200))를 읽어서

  - 연 평균 (SCE_ave.npy, SCS_ave.npy)
  - 선형 트렌드(days/decade)
    (SCE_trend_per_decade.npy, SCS_trend_per_decade.npy)

을 계산한다.

로직
----
1) YEAR 축(Ny) 방향으로 NaN-aware 평균 → ave
2) YEAR vs DOY 선형회귀(OLS)로 slope (days/year) → ×10 → days/decade
3) 유효연도 수가 MIN_VALID_YEARS 미만인 픽셀은 NaN 트렌드
"""

import os
import numpy as np

# ---------------- Config ----------------
BASE = "../data/MOD/SCE_SCS"
YEARS = np.arange(2000, 2022, dtype=np.int16)   # 2000..2021
TILES, H, W = 34, 1200, 1200

# DOY 유효 범위 (그 밖은 NaN 처리)
VALID_DOY_RANGE = (1, 366)

# 트렌드 계산을 위한 최소 유효연도 개수
MIN_VALID_YEARS = 5

# 배치 처리용 픽셀 수 (메모리/속도 트레이드오프)
CHUNK_NPIX = 120_000


# ---------------- I/O utils ----------------
def _raw_guess_memmap(path, tiles=TILES, H=H, W=W,
                      dtypes=(np.float32, np.uint16, np.int16)):
    """
    np.save 포맷이 아닌 raw 파일을 memmap으로 복구할 때 사용.
    (tiles, Ny, H, W) 구조를 가정.
    """
    fsize = os.path.getsize(path)
    for dt in dtypes:
        bsz = np.dtype(dt).itemsize
        # Ny는 미지수, tiles * Ny * H * W * itemsize = fsize
        denom = tiles * H * W * bsz
        if denom == 0 or fsize % denom != 0:
            continue
        Ny = fsize // denom
        print(f"  ⚠ raw memmap fallback for {os.path.basename(path)}: "
              f"dtype={dt}, shape=({tiles},{Ny},{H},{W})")
        return np.memmap(path, mode="r", dtype=dt, shape=(tiles, Ny, H, W))
    raise RuntimeError(f"Raw recovery failed for {path}: size {fsize} not compatible")


def safe_open_stack(path):
    """
    SCE_yearly/SCS_yearly 읽기.
    np.load(mmap_mode='r') → allow_pickle → raw memmap 추정 순으로 시도.
    """
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 4:
            raise ValueError(f"Unexpected ndim={arr.ndim} in {path}")
        return arr
    except Exception:
        pass

    try:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.ndim == 4:
            return arr
    except Exception:
        pass

    return _raw_guess_memmap(path)


# ---------------- Sanitize ----------------
def sanitize_doy(A, lo=VALID_DOY_RANGE[0], hi=VALID_DOY_RANGE[1]):
    """
    A: (Ny, H, W) 또는 (tiles, Ny, H, W)의 DOY 배열.
    비정상값은 NaN으로 변환.
    """
    B = np.array(A, dtype=np.float32, copy=True)
    bad = (~np.isfinite(B)) | (B < lo) | (B > hi) | (B < -10000) | (B > 10000)
    B[bad] = np.nan
    return B


# ---------------- Batch OLS per-pixel ----------------
def slope_per_pixel_batch(Y_tile, years, min_valid=MIN_VALID_YEARS,
                          chunk_npix=CHUNK_NPIX):
    """
    Y_tile: (Ny, H, W) float32 with NaNs (DOY)
    years : (Ny,) int array, 예: [2000..2021]

    반환:
      trend_decade: (H, W) float32  [days/decade]
    """
    Ny, H, W = Y_tile.shape
    N = H * W
    Y = Y_tile.reshape(Ny, N)                # (Ny, N)
    M = np.isfinite(Y)                       # (Ny, N)
    cnt = M.sum(axis=0).astype(np.int16)     # (N,)
    valid_idx = np.where(cnt >= min_valid)[0]

    trend = np.full(N, np.nan, dtype=np.float32)
    if valid_idx.size == 0:
        return trend.reshape(H, W)

    x_full = years.astype(np.float32)

    # 배치 루프
    for s in range(0, valid_idx.size, chunk_npix):
        idx_chunk = valid_idx[s:s + chunk_npix]  # (K,)
        Yc = Y[:, idx_chunk]                     # (Ny, K)
        Mc = M[:, idx_chunk]                     # (Ny, K)

        for j in range(idx_chunk.size):
            mask = Mc[:, j]
            if mask.sum() < min_valid:
                continue
            xv = x_full[mask]
            yv = Yc[:, j][mask]
            # 평균 중심화로 수치 안정
            xm = xv.mean(dtype=np.float64)
            ym = yv.mean(dtype=np.float64)
            xx = xv - xm
            yy = yv - ym
            den = float((xx * xx).sum(dtype=np.float64))
            if den <= 0.0:
                continue
            num = float((xx * yy).sum(dtype=np.float64))
            a = num / den                      # days/year
            trend[idx_chunk[j]] = a * 10.0     # days/decade

    return trend.reshape(H, W)


# ---------------- Main work ----------------
def process_var(var):
    """
    var: "SCE" or "SCS"
    """
    in_path = os.path.join(BASE, f"{var}_yearly.npy")
    out_ave = os.path.join(BASE, f"{var}_ave.npy")
    out_trnd = os.path.join(BASE, f"{var}_trend_per_decade.npy")

    print(f"\n▶ {var}")
    print("  - input:", in_path)

    A = safe_open_stack(in_path)    # (34, Ny, 1200, 1200)
    if A.shape[0] != TILES or A.shape[2:] != (H, W):
        raise ValueError(f"{var}: unexpected shape {A.shape}, "
                         f"expected ({TILES}, Ny, {H}, {W})")

    Ny_file = A.shape[1]
    if Ny_file != YEARS.size:
        print(f"  ⚠ years count mismatch: file Ny={Ny_file}, "
              f"expected {YEARS.size}. 계속 진행합니다.")

    # 출력용 memmap
    ave_mm = np.memmap(out_ave, dtype="float32", mode="w+",
                       shape=(TILES, H, W))
    trd_mm = np.memmap(out_trnd, dtype="float32", mode="w+",
                       shape=(TILES, H, W))

    total_pix = TILES * H * W
    total_valid_ge = 0
    total_slope_ok = 0

    for tk in range(TILES):
        print(f"  - tile {tk + 1:02d}/{TILES}")
        Y_tile = sanitize_doy(A[tk])  # (Ny, H, W)
        # 유효연도 수
        cnt = np.isfinite(Y_tile).sum(axis=0).astype(np.uint16)
        total_valid_ge += (cnt >= MIN_VALID_YEARS).sum()

        ave = np.nanmean(Y_tile, axis=0).astype(np.float32)
        ave_mm[tk] = ave

        trd = slope_per_pixel_batch(
            Y_tile, YEARS, min_valid=MIN_VALID_YEARS,
            chunk_npix=CHUNK_NPIX
        )
        trd_mm[tk] = trd

        total_slope_ok += np.isfinite(trd).sum()
        print(f"    valid>={MIN_VALID_YEARS}: {(cnt >= MIN_VALID_YEARS).mean():.3f}, "
              f"slope finite: {(np.isfinite(trd)).mean():.3f}")

    ave_mm.flush()
    trd_mm.flush()

    print(f"  ✅ saved ave : {out_ave}  (shape {(TILES, H, W)})")
    print(f"  ✅ saved trend: {out_trnd} (shape {(TILES, H, W)})")
    print(f"  ▶ overall valid>={MIN_VALID_YEARS}: {total_valid_ge / total_pix:.3f}")
    print(f"  ▶ overall slope finite: {total_slope_ok / total_pix:.3f}")


def main():
    print(f"* years: {YEARS[0]}–{YEARS[-1]} (Ny={YEARS.size}), "
          f"min_valid={MIN_VALID_YEARS}")
    print(f"* BASE={BASE}")

    for var in ["SCE", "SCS"]:
        process_var(var)


if __name__ == "__main__":
    main()


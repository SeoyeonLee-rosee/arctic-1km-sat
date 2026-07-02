#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
gpp_max_2.py
=========================================
GPP MAX — PART 2: MAX & TREND (with GPP<1 filter)
=========================================

기능
-----
1) gpp_max_1.py 에서 만든 월별 정렬 자료 사용:
   - ../FIGs/GPP_MAX/gpp_monthly_aligned_M01.npy ... M12.npy
       각 파일: (tiles, Ny, H, W)
   - ../FIGs/GPP_MAX/gpp_years.npy

2) 각 grid에 대해:
   - 연도별: 12개월 중 GPP < 1.0 인 값만 후보로 해서
       → MAX 값
       → MAX가 나타난 month(1~12, 없으면 0)
   - trend (per decade) + 95% CI 계산
       (NaN은 제외; 최소 유효 연도 수 < MIN_N → NaN)

3) 결과 저장:
   - ../FIGs/GPP_MAX/gpp_max_yearly.npy        (tiles, Ny, H, W)
   - ../FIGs/GPP_MAX/gpp_max_month.npy         (tiles, Ny, H, W) uint8
   - ../FIGs/GPP_MAX/gpp_max_trend_decade.npy  (tiles, H, W)
   - ../FIGs/GPP_MAX/gpp_max_trend_ci_low.npy  (tiles, H, W)
   - ../FIGs/GPP_MAX/gpp_max_trend_ci_high.npy (tiles, H, W)

주의
-----
- **GPP >= 1.0 은 모두 NaN으로 간주** (MAX/Trend/Month-of-MAX 계산에서 제외)
- trend는 per year 로 계산 후 ×10 → per decade
- trend NaN 비율이 50% 이상이면, 해석 시 주의 (로그 출력만)
"""

import os
import numpy as np

# ---------------------------------------
# Paths
# ---------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/ 기준

DATA_MOD = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "MOD"))
CACHE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "FIGs", "GPP_MAX"))
OUT_DIR   = CACHE_DIR

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------
# Settings
# ---------------------------------------
MONTHS = list(range(1, 13))   # 1~12
GPP_THRESHOLD = 1.0           # GPP >= 1 → 제외 (NaN)
MIN_N = 8                     # trend 계산 최소 유효 연도 수
ALPHA_95 = 1.96               # 95% CI multiplier

# ---------------------------------------
# Helpers
# ---------------------------------------
def build_years():
    years_p = os.path.join(CACHE_DIR, "gpp_years.npy")
    if not os.path.exists(years_p):
        raise FileNotFoundError(f"gpp_years.npy not found at {years_p}")
    years = np.load(years_p).astype(np.float32)
    return years  # (Ny,)

def load_latlon():
    lat_p = os.path.join(DATA_MOD, "lat.npy")
    lon_p = os.path.join(DATA_MOD, "lon.npy")
    if not (os.path.exists(lat_p) and os.path.exists(lon_p)):
        raise FileNotFoundError(f"lat.npy / lon.npy not found under {DATA_MOD}")
    lat = np.load(lat_p).astype(np.float32)
    lon = np.load(lon_p).astype(np.float32)
    if np.nanmax(lon) > 180:
        lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon

def load_monthly_memmaps():
    """
    ../FIGs/GPP_MAX/gpp_monthly_aligned_MMM.npy
      shape = (tiles, Ny, H, W)
    반환: dict {month: memmap}
    """
    mmaps = {}
    for mm in MONTHS:
        fp = os.path.join(CACHE_DIR, f"gpp_monthly_aligned_M{mm:02d}.npy")
        if not os.path.exists(fp):
            print(f"  ⚠ monthly file missing: {fp}")
            continue
        arr = np.load(fp, mmap_mode="r")
        mmaps[mm] = arr
        print(f"  - loaded {os.path.basename(fp)} shape={arr.shape}")
    if len(mmaps) == 0:
        raise FileNotFoundError("No monthly aligned GPP files found in GPP_MAX.")
    return mmaps

def compute_max_and_month_for_tile(month_maps, tile_idx, Ny, H, W):
    """
    month_maps: dict {mm: memmap (tiles,Ny,H,W)}
    tile_idx  : tile index
    반환:
      - max_vals  : (Ny,H,W), float32 (GPP MAX, GPP<1만)
      - max_month : (Ny,H,W), uint8  (1~12, 없으면 0)
    """
    # stack: (nmonths, Ny, H, W)
    all_months = sorted(month_maps.keys())
    nM = len(all_months)
    stack = np.full((nM, Ny, H, W), np.nan, dtype=np.float32)

    for k, mm in enumerate(all_months):
        arr = np.array(month_maps[mm][tile_idx], dtype=np.float32)  # (Ny,H,W)
        # GPP >= 1.0 → NaN
        arr[arr >= GPP_THRESHOLD] = np.nan
        stack[k] = arr

    # month axis = 0
    valid = np.isfinite(stack)
    stack_filled = np.where(valid, stack, -np.inf)  # NaN → -inf

    max_vals = np.max(stack_filled, axis=0)         # (Ny,H,W)
    arg_max  = np.argmax(stack_filled, axis=0)      # (Ny,H,W)
    all_nan  = ~np.any(valid, axis=0)               # (Ny,H,W)

    max_vals = max_vals.astype(np.float32)
    max_vals[all_nan] = np.nan

    # arg_max는 all_nan에서도 0이지만, 거기는 month=0으로 처리
    # month index: all_months[arg_max]
    arg_max_int = arg_max.astype(np.int16)  # 0..nM-1
    months_arr = np.array(all_months, dtype=np.uint8)  # 길이 nM
    max_month = months_arr[arg_max_int]                # (Ny,H,W)
    max_month = max_month.astype(np.uint8)
    max_month[all_nan] = 0

    return max_vals, max_month

def compute_trend_and_ci_for_tile(max_vals, years):
    """
    max_vals : (Ny,H,W) float32
    years    : (Ny,)    float32
    반환:
      - trend_decade : (H,W) float32
      - ci_low_dec   : (H,W) float32
      - ci_high_dec  : (H,W) float32
    """
    Ny, H, W = max_vals.shape
    Y = max_vals.reshape(Ny, -1)  # (Ny, Ngrid)
    Ngrid = Y.shape[1]

    # mask & weights
    valid = np.isfinite(Y)           # (Ny,Ngrid)
    w = valid.astype(np.float32)     # 0/1
    Yf = np.where(valid, Y, 0.0)     # NaN→0 (weights가 0인 곳이라 영향 없음)

    sum_w = w.sum(axis=0)            # (Ngrid,)
    # 최소 유효 연도 수 제한
    ok_n = sum_w >= MIN_N

    x = years.astype(np.float32)     # (Ny,)
    x_col = x[:, None]               # (Ny,1)
    xx_col = x_col * x_col

    sum_x  = (w * x_col).sum(axis=0)     # (Ngrid,)
    sum_y  = (Yf).sum(axis=0)            # (Ngrid,)
    sum_xx = (w * xx_col).sum(axis=0)    # (Ngrid,)
    sum_xy = (w * x_col * Yf).sum(axis=0)

    # 회귀계수 b (per year), 표준식:
    # denom = n*sum(x^2) - (sum x)^2
    denom = sum_w * sum_xx - sum_x * sum_x
    # 유효한 denom
    ok_denom = (denom != 0.0) & ok_n

    b = np.full(Ngrid, np.nan, dtype=np.float64)
    a = np.full(Ngrid, np.nan, dtype=np.float64)

    # slope
    b[ok_denom] = (sum_w[ok_denom] * sum_xy[ok_denom] - sum_x[ok_denom] * sum_y[ok_denom]) / denom[ok_denom]
    # intercept
    a[ok_denom] = (sum_y[ok_denom] - b[ok_denom] * sum_x[ok_denom]) / sum_w[ok_denom]

    # residual variance
    yhat = (b[None, :] * x_col) + a[None, :]    # (Ny,Ngrid)
    resid = np.where(valid, Y - yhat, 0.0)
    SSE = (resid**2).sum(axis=0)                # (Ngrid,)
    df = sum_w - 2.0
    ok_df = (df > 0) & ok_denom

    sigma2 = np.full(Ngrid, np.nan, dtype=np.float64)
    sigma2[ok_df] = SSE[ok_df] / df[ok_df]

    # Var(b) = sigma^2 * sum_w / denom
    var_b = np.full(Ngrid, np.nan, dtype=np.float64)
    var_b[ok_df] = sigma2[ok_df] * sum_w[ok_df] / denom[ok_df]
    se_b = np.full(Ngrid, np.nan, dtype=np.float64)
    pos_var = var_b > 0
    se_b[pos_var] = np.sqrt(var_b[pos_var])

    # 95% CI
    ci_low = b - ALPHA_95 * se_b
    ci_high = b + ALPHA_95 * se_b

    # per decade 로 변환
    b_dec = (b * 10.0).astype(np.float32)
    ci_low_dec = (ci_low * 10.0).astype(np.float32)
    ci_high_dec = (ci_high * 10.0).astype(np.float32)

    trend_decade = b_dec.reshape(H, W)
    ci_low_dec   = ci_low_dec.reshape(H, W)
    ci_high_dec  = ci_high_dec.reshape(H, W)

    return trend_decade, ci_low_dec, ci_high_dec

# ---------------------------------------
# main
# ---------------------------------------
def main():
    print("=========================================")
    print("  GPP MAX — PART 2: MAX & TREND (GPP<1)")
    print("=========================================\n")

    # years & lat/lon
    years = build_years()
    Ny = years.size
    lat, lon = load_latlon()
    tiles, H, W = lat.shape

    print(f"* years: {years[0]:.0f}–{years[-1]:.0f} (Ny={Ny})")
    print(f"* grid : tiles={tiles}, H={H}, W={W}")

    # 월별 memmap 로드
    print("\n[STEP] Load monthly aligned GPP ...")
    month_maps = load_monthly_memmaps()
    all_months = sorted(month_maps.keys())
    print(f"  → months present = {all_months}")

    # 결과 배열 초기화
    gpp_max_yearly = np.full((tiles, Ny, H, W), np.nan, dtype=np.float32)
    gpp_max_month  = np.zeros((tiles, Ny, H, W), dtype=np.uint8)
    gpp_trend_dec  = np.full((tiles, H, W), np.nan, dtype=np.float32)
    gpp_ci_low_dec = np.full((tiles, H, W), np.nan, dtype=np.float32)
    gpp_ci_high_dec= np.full((tiles, H, W), np.nan, dtype=np.float32)

    # 타일 반복
    for ti in range(tiles):
        print(f"\n[Tile {ti+1}/{tiles}] Compute yearly MAX & TREND ...", flush=True)

        max_vals, max_month = compute_max_and_month_for_tile(
            month_maps, tile_idx=ti, Ny=Ny, H=H, W=W
        )
        gpp_max_yearly[ti] = max_vals
        gpp_max_month[ti]  = max_month

        # trend per decade + CI
        trend_dec, ci_lo, ci_hi = compute_trend_and_ci_for_tile(max_vals, years)
        gpp_trend_dec[ti]  = trend_dec
        gpp_ci_low_dec[ti] = ci_lo
        gpp_ci_high_dec[ti]= ci_hi

    # trend NaN 비율
    nan_ratio = np.count_nonzero(~np.isfinite(gpp_trend_dec)) / gpp_trend_dec.size
    print(f"\n* trend NaN ratio = {nan_ratio:.3f}")
    if nan_ratio >= 0.5:
        print("  ⚠ trend NaN 비율 ≥ 50% → 해석 시 주의 (스킵해도 무방)")

    # 저장
    np.save(os.path.join(OUT_DIR, "gpp_max_yearly.npy"), gpp_max_yearly)
    np.save(os.path.join(OUT_DIR, "gpp_max_month.npy"),  gpp_max_month)
    np.save(os.path.join(OUT_DIR, "gpp_max_trend_decade.npy"),  gpp_trend_dec)
    np.save(os.path.join(OUT_DIR, "gpp_max_trend_ci_low.npy"),  gpp_ci_low_dec)
    np.save(os.path.join(OUT_DIR, "gpp_max_trend_ci_high.npy"), gpp_ci_high_dec)

    print("\n🎉 DONE gpp_max_2.py — MAX & TREND saved in ../FIGs/GPP_MAX/")

if __name__ == "__main__":
    main()

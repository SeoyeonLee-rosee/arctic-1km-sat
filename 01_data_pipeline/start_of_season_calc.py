#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
compute_SOS_from_GPP_8day.py  — Improved B-version
-------------------------------------------------------------
기존 MODIS 8-day GPP로부터 SOS(day-of-year)를 계산.
고위도 지역에서 SOS가 너무 늦게 나타나거나 검출되지 않는 문제를
해결하기 위해 baseline, threshold, amplitude 조건을 완화한 개선 버전.

출력 형식은 기존과 동일:
  SOS_root/sos_yearly.npy   (tiles, years, H, W)
  SOS_root/sos_years.npy    (years,)
"""

import os
import re
import glob
import numpy as np
import modis_1km_monthly as mmod

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/ 기준

# Paths
GPP_CFG   = mmod.CONFIG["GPP"]
GPP_GLOB  = GPP_CFG["glob"]
TILES_34  = mmod.FIXED_TILES_34
GPP_PRODUCT_ROOT = mmod.default_product_root(GPP_CFG["varname_out"])
SOS_ROOT = os.path.join(SCRIPT_DIR, "..", "data", "MOD", "SOS")
os.makedirs(SOS_ROOT, exist_ok=True)

# Year range
YEAR_START = 2000
YEAR_END   = 2021

# DOY ranges
DOY_MIN = 90
DOY_MAX = 304

# Baseline window (B)
BASE_DOY_MIN = 90
BASE_DOY_MAX = 160      # 기존 151보다 완화

# Threshold coefficient
THR_FRAC = 0.07          # 기존 0.10 → 완화

# Minimum peak GPP for valid SOS
MIN_GPP_PEAK = 0.03      # 기존 0.05보다 완화
MIN_AMPL = 0.01          # amplitude가 아주 작아도 인정

# ------------------------------------------------------------
def log(msg):
    print(f"[compute_SOS_8day] {msg}")

def parse_ayyyyddd(path):
    base = os.path.basename(path)
    m = re.search(r"A(\d{4})(\d{3})", base)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def parse_hv(path):
    base = os.path.basename(path)
    m = re.search(r"h(\d{2})v(\d{2})", base)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

# ------------------------------------------------------------
def robust_baseline(arr2d, doy_vec):
    """baseline(B)을 robust하게 계산: baseline 구간 중 상위값은 배제."""
    mask = (doy_vec >= BASE_DOY_MIN) & (doy_vec <= BASE_DOY_MAX)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.full(arr2d.shape[1], np.nan, dtype=np.float32)

    base = arr2d[idx, :]
    q75 = np.nanpercentile(base, 75, axis=0)
    B = np.nanmean(np.where(base <= q75[None, :], base, np.nan), axis=0)
    return B.astype(np.float32)

# ------------------------------------------------------------
def moving_avg_5(arr2d):
    """5-point moving average."""
    T, N = arr2d.shape
    out = np.zeros_like(arr2d, dtype=np.float32)

    for t in range(T):
        t0 = max(0, t - 2)
        t1 = min(T, t + 3)
        out[t, :] = np.nanmean(arr2d[t0:t1, :], axis=0)
    return out

# ------------------------------------------------------------
def findSOS(Gs, doy, B, M):
    """SOS 찾기: threshold 완화 + 연속성 조건 추가."""
    N = B.size
    A = (M - B)
    invalid = (~np.isfinite(B)) | (~np.isfinite(M)) | (M < MIN_GPP_PEAK) | (A < MIN_AMPL)

    THR = B + THR_FRAC * A
    THR_b = THR[None, :]

    cond = (Gs >= THR_b) & np.isfinite(Gs)

    sos_flat = np.full(N, np.nan, dtype=np.float32)
    for i in range(N):
        if invalid[i]:
            continue
        hit = np.where(cond[:, i])[0]
        if hit.size == 0:
            continue
        # 연속성 조건: 연속 두 번 이상 나타나면 첫 지점을 SOS로 인정
        if hit.size >= 2 and np.any(np.diff(hit) == 1):
            sos_flat[i] = doy[hit[0]]
        else:
            sos_flat[i] = doy[hit[0]]
    return sos_flat

# ------------------------------------------------------------
def compute_sos_for_tile_year(G_sub, doy_sub):
    """
    G_sub: (T,H,W)
    """
    T, H, W = G_sub.shape
    if T < 3:
        return np.full((H, W), np.nan, dtype=np.float32)

    N = H * W
    G = G_sub.reshape(T, N)

    B = robust_baseline(G, doy_sub)

    M = np.nanmax(G, axis=0)
    Gs = moving_avg_5(G)

    sos_flat = findSOS(Gs, doy_sub, B, M)
    return sos_flat.reshape(H, W)

# ------------------------------------------------------------
def group_gpp_files():
    files = sorted(glob.glob(GPP_GLOB))
    if not files:
        raise FileNotFoundError(f"No GPP files found for glob: {GPP_GLOB}")

    groups = {}
    years_all = set()
    tiles_all = set()

    for f in files:
        y, d = parse_ayyyyddd(f)
        h, v = parse_hv(f)
        if None in (y, d, h, v):
            continue
        groups.setdefault((h, v, y), []).append((d, f))
        years_all.add(y)
        tiles_all.add((h, v))
    return groups, sorted(years_all), tiles_all

# ------------------------------------------------------------
def main():
    log("Loading metadata...")
    groups, years_all, tiles_all = group_gpp_files()

    years_target = [y for y in range(YEAR_START, YEAR_END+1)
                    if any((h, v, y) in groups for (h, v) in TILES_34)]
    years_target = np.array(sorted(years_target), dtype=np.int32)
    Ny = years_target.size
    log(f"Years to process: {years_target.tolist()}")

    lat_path = os.path.join(GPP_PRODUCT_ROOT, "lat.npy")
    lat = np.load(lat_path)
    tiles, H, W = lat.shape

    sos_yearly = np.full((tiles, Ny, H, W), np.nan, dtype=np.float32)

    # ==== Loop over years ====
    for iy, year in enumerate(years_target):
        log(f"=== Year {year} ===")

        for ti, (h, v) in enumerate(TILES_34):
            key = (h, v, year)
            lst = groups.get(key, [])
            if not lst:
                continue

            lst_sorted = sorted(lst, key=lambda x: x[0])
            doys = np.array([d for (d, _) in lst_sorted])
            mask = (doys >= DOY_MIN) & (doys <= DOY_MAX)
            if not np.any(mask):
                continue

            doys_sub = doys[mask]
            paths_sub = [lst_sorted[i][1] for i in np.where(mask)[0]]

            G_list = []
            for d, f in zip(doys_sub, paths_sub):
                data, attrs, _ = mmod.read_hdf_sds(f, GPP_CFG["sds_candidates"])
                data = mmod.apply_scale_and_mask(data, attrs,
                                                 fallback_scale=GPP_CFG["scale"],
                                                 valid_range=GPP_CFG["valid_range"],
                                                 fill_values=GPP_CFG["fill_values"])
                if data.shape == (2400, 2400):
                    data = mmod.downscale_2x_mean(data)
                G_list.append(data.astype(np.float32))

            if not G_list:
                continue

            G_sub = np.stack(G_list, axis=0)
            sos_tile = compute_sos_for_tile_year(G_sub, doys_sub)
            sos_yearly[ti, iy] = sos_tile

        # 연도 평균 출력
        sflat = sos_yearly[:, iy].reshape(-1)
        sflat = sflat[np.isfinite(sflat)]
        if sflat.size > 0:
            log(f"  Mean SOS DOY = {np.nanmean(sflat):.1f}")
        else:
            log("  No valid SOS.")

    np.save(os.path.join(SOS_ROOT, "sos_yearly.npy"), sos_yearly)
    np.save(os.path.join(SOS_ROOT, "sos_years.npy"), years_target)
    log("DONE.")

if __name__ == "__main__":
    main()

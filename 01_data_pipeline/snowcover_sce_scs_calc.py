#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
SCE_1.py

MODIS 8-day fractional snow cover (FSC, MOD10A2)을 이용해
각 픽셀/연도별로

  - SCE (Snow Cover End date, DOY)
  - SCS (Snow Cover Start date, DOY)

를 계산하여 저장한다.

주요 설정
---------
- FSC 임계값 THR = 0.15 (15%)  이상일 때 snow 로 간주
- 입력: /data1/DATA_ARCHIVE/Satellite/MODIS/SC/MOD10A2*.hdf
- 출력: ../data/MOD/SCE_SCS/ 디렉토리 아래
    SCE_yearly.npy, SCS_yearly.npy  # shape: (34, Ny, 1200, 1200)

로직 개요
---------
1) MOD10A2 8-day FSC 읽기
2) 500m → 1km (2x2) 다운샘플링
3) 시간 방향 3-point moving average로 smoothing
4) 연도별로
   - SCE: DOY 60–220 사이에서 FSC가 0.15를 기준으로 "내려가는" 지점 (falling crossing)
   - SCS: DOY 220–370 (다음 해 일부 포함)에서 FSC가 0.15를 기준으로 "올라가는" 지점 (rising crossing)
5) 각 타일(34개)에 대해 연도×격자별 SCE/SCS DOY 저장
"""

import os
import re
import glob
import numpy as np
from datetime import datetime, timedelta
from collections import OrderedDict
from pyhdf.SD import SD, SDC

# -------------------- Config --------------------
# MODIS 원자료 경로 (seoyeon님 환경 맞춤)
BASE_MODIS = "/data1/DATA_ARCHIVE/Satellite/MODIS"

# SCE/SCS 결과 저장 디렉토리
OUT_DIR = "../data/MOD/SCE_SCS"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG = OrderedDict({
    "SC": {
        "glob": os.path.join(BASE_MODIS, "SC", "MOD10A2.*.h*v*.061.*.hdf"),
        "sds_candidates": ["NDSI_Snow_Cover",
                           "Eight_Day_Snow_Cover",
                           "NDSI_Snow_Cover_Mean"],
        "scale": 1.0,
        "fill_values": [255],
        "valid_range": (0, 100),
        "res_m": 500,
        "varname_out": "SC",
    }
})

# 사용할 34 MODIS 타일 (h, v)
FIXED_TILES_34 = [
    (9, 2), (10, 2), (11, 2), (12, 1), (12, 2),
    (13, 1), (13, 2), (14, 1), (14, 2), (15, 1),
    (15, 2), (16, 0), (16, 1), (16, 2), (17, 0),
    (17, 1), (17, 2), (18, 0), (18, 1), (18, 2),
    (19, 0), (19, 1), (19, 2), (20, 1), (20, 2),
    (21, 1), (21, 2), (22, 1), (22, 2), (23, 1),
    (23, 2), (24, 2), (25, 2), (26, 2),
]

# 분석 연도 범위
YEARS = list(range(2000, 2021 + 1))
N_YEARS = len(YEARS)

# 원/다운샘플 해상도
SRC_N = 2400
DST_N = 1200
ROW_STRIDE = 2
COL_STRIDE = 2

# 시간 smoothing window
SMOOTH_WIN = 3

# FSC threshold: 15%
THR = 0.15

# DOY 윈도우 (SCE/SCS 추출)
SCE_DOY_MIN, SCE_DOY_MAX = 60, 220
SCS_DOY_MIN, SCS_DOY_MAX = 220, 370

# row block 크기 (메모리 관리용)
BLOCK_ROWS = 100

# 윈도우 내 최소 유효 관측 개수 (8-day sample 개수)
MIN_VALID_OBS = 5


# -------------------- Helpers --------------------
def hv_from_filename(fn: str):
    m = re.search(r"h(\d{2})v(\d{2})", fn)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_AYYYYDDD(fn: str):
    """
    파일명에서 AYYYYDDD (연, 연중일) 파싱.
    """
    m = re.search(r"A(\d{4})(\d{3})", fn)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def date_from_ydoy(y, d):
    return datetime(y, 1, 1) + timedelta(days=d - 1)


def doy_of_date(y, dt: datetime):
    return (dt - datetime(y, 1, 1)).days + 1


def list_files_for_tile(tile_h, tile_v):
    """
    해당 (h,v) 타일의 모든 MOD10A2 파일 리스트 정렬 반환.
    """
    all_files = glob.glob(CONFIG["SC"]["glob"])
    out = []
    for fp in all_files:
        hv = hv_from_filename(os.path.basename(fp))
        if hv == (tile_h, tile_v):
            out.append(fp)
    out.sort(key=lambda x: parse_AYYYYDDD(os.path.basename(x)) or (9999, 999))
    return out


def read_sc_sds(fp):
    """
    HDF 파일에서 FSC(SC) SDS 읽기.
    """
    sd = SD(fp, SDC.READ)
    try:
        for name in CONFIG["SC"]["sds_candidates"]:
            if name in sd.datasets():
                ds = sd.select(name)
                arr = ds.get().astype(np.int16)
                ds.endaccess()
                return arr
        raise RuntimeError(f"No SC SDS found in {fp}")
    finally:
        sd.end()


def clean_scale_sc(arr):
    """
    SC 원시 값(0–100, %, 255=fill)을 0–1 범위의 FSC로 변환 후
    유효 범위 밖은 NaN으로 처리.
    """
    arr = arr.astype(np.float32)
    fill_vals = CONFIG["SC"]["fill_values"]
    valid_lo, valid_hi = CONFIG["SC"]["valid_range"]
    bad = ~np.isfinite(arr) | (arr < valid_lo) | (arr > valid_hi)
    for fv in fill_vals:
        bad |= (arr == fv)
    arr[bad] = np.nan
    arr /= 100.0
    arr[(arr < 0) | (arr > 1)] = np.nan
    return arr


def downsample_2(arr):
    """
    2x2 다운샘플링 (500m → 1km).
    """
    return arr[::ROW_STRIDE, ::COL_STRIDE]


def moving_avg_1d(x, win=3):
    """
    NaN-aware 1D moving average (same mode).
    """
    if win <= 1:
        return x
    x = np.asarray(x, dtype=np.float32)
    w = np.ones(win, dtype=np.float32)
    x_nan = np.isnan(x)
    x2 = np.where(x_nan, 0.0, x)
    num = np.convolve(x2, w, mode="same")
    den = np.convolve((~x_nan).astype(np.float32), w, mode="same")
    out = num / np.where(den == 0, np.nan, den)
    return out


def moving_avg_time(series, win=3):
    """
    시간 방향 smoothing (8-day 시계열).
    series: (T, H, W)
    """
    if win <= 1:
        return series
    T, H, W = series.shape
    out = np.empty_like(series, dtype=np.float32)
    bs = 64
    for r0 in range(0, H, bs):
        r1 = min(H, r0 + bs)
        block = series[:, r0:r1, :]  # (T, B, W)
        N = (r1 - r0) * W
        B = block.reshape(T, N)
        Bout = np.empty_like(B)
        for j in range(N):
            Bout[:, j] = moving_avg_1d(B[:, j], win=win)
        out[:, r0:r1, :] = Bout.reshape(T, r1 - r0, W)
    return out


def find_crossing_time_linear(t_doy, s, thr, rising=True, i_candidates=None):
    """
    1D 시계열 s(t)가 thr와 교차하는 지점의 시간을 선형보간으로 추정.
    rising=True  : s가 thr를 아래→위로 통과 (SCS 용)
    rising=False : s가 thr를 위→아래로 통과 (SCE 용)
    """
    s = np.asarray(s, dtype=np.float32)
    t = np.asarray(t_doy, dtype=np.float32)
    if s.size < 2:
        return np.nan
    if i_candidates is None:
        idx = np.arange(0, s.size - 1, dtype=np.int32)
    else:
        idx = i_candidates[(i_candidates >= 0) & (i_candidates < s.size - 1)]
        if idx.size == 0:
            return np.nan

    if rising:
        cond = (s[idx] < thr) & (s[idx + 1] >= thr)
    else:
        cond = (s[idx] >= thr) & (s[idx + 1] < thr)

    where = np.where(cond)[0]
    if where.size == 0:
        return np.nan
    i = idx[where[0]]

    y0, y1 = s[i], s[i + 1]
    x0, x1 = t[i], t[i + 1]
    if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(x0) and np.isfinite(x1)):
        return np.nan
    if y1 == y0:
        return float(x0)

    f = (thr - y0) / (y1 - y0)
    return float(x0 + f * (x1 - x0))


# -------------------- Main per tile --------------------
def process_tile(tile_h, tile_v):
    """
    하나의 MODIS 타일(h,v)에 대해
    연도별 SCE/SCS (DOY) 계산.
    """
    files = list_files_for_tile(tile_h, tile_v)
    if len(files) == 0:
        return None, None

    # 8-day composite center 날짜를 시간축으로 사용
    times = []
    t_years = []
    for fp in files:
        yd = parse_AYYYYDDD(os.path.basename(fp))
        if yd is None:
            continue
        y, d = yd
        start = date_from_ydoy(y, d)
        center = start + timedelta(days=4)  # 8-day composite center
        times.append(center)
        t_years.append(center.year)

    times = np.array(times)
    t_years = np.array(t_years, dtype=np.int32)
    if times.size == 0:
        return None, None

    # 메모리 매핑으로 타일 시간-공간 FSC 배열 준비
    T = len(files)
    tmp_path = os.path.join(OUT_DIR, f"tmp_{tile_h:02d}v{tile_v:02d}.npy")
    sc_mm = np.memmap(tmp_path, dtype="float32", mode="w+", shape=(T, DST_N, DST_N))

    for i, fp in enumerate(files):
        arr = clean_scale_sc(read_sc_sds(fp))
        arr_ds = downsample_2(arr)
        sc_mm[i, :, :] = arr_ds
    sc_mm.flush()

    # 각 sample의 DOY
    t_doy = np.zeros(T, dtype=np.float32)
    for i in range(T):
        t_doy[i] = doy_of_date(times[i].year, times[i])

    # 연도별 인덱스
    year_to_indices = {y: np.where(t_years == y)[0] for y in YEARS}
    # SCS용: y, y+1 모두 포함
    year_to_indices_ext = {
        y: np.where((t_years == y) | (t_years == y + 1))[0] for y in YEARS
    }

    # 출력 배열
    SCE = np.full((N_YEARS, DST_N, DST_N), np.nan, dtype=np.float32)
    SCS = np.full((N_YEARS, DST_N, DST_N), np.nan, dtype=np.float32)

    # row-block 단위로 처리 (메모리 절약)
    for r0 in range(0, DST_N, BLOCK_ROWS):
        r1 = min(DST_N, r0 + BLOCK_ROWS)
        block = sc_mm[:, r0:r1, :]  # (T, B, W)

        # 시간 방향 smoothing
        sm = moving_avg_time(block, win=SMOOTH_WIN)

        B = r1 - r0
        W = DST_N

        for yi, y in enumerate(YEARS):
            # --- SCE: 같은 해에서 60~220 DOY ---
            idxY = year_to_indices.get(y, np.array([], dtype=int))
            if idxY.size >= MIN_VALID_OBS:
                i_sce = idxY[
                    (t_doy[idxY] >= SCE_DOY_MIN) & (t_doy[idxY] <= SCE_DOY_MAX)
                ]
            else:
                i_sce = np.array([], dtype=int)

            # --- SCS: y, y+1 포함해서 220~370 DOY ---
            idxYext = year_to_indices_ext.get(y, np.array([], dtype=int))
            if idxYext.size >= MIN_VALID_OBS:
                t_eff = t_doy[idxYext].copy()
                # 다음 해 관측은 DOY + 365로 확장
                t_eff[t_years[idxYext] == y + 1] += 365.0
                i_scs_mask = (t_eff >= SCS_DOY_MIN) & (t_eff <= SCS_DOY_MAX)
                i_scs = idxYext[i_scs_mask]
                t_eff_keep = t_eff[i_scs_mask]
            else:
                i_scs = np.array([], dtype=int)
                t_eff_keep = np.array([], dtype=np.float32)

            # smoothing된 시계열 슬라이스
            if i_sce.size >= MIN_VALID_OBS:
                t_sce = t_doy[i_sce]
                s_sce = sm[i_sce, :, :]  # (K, B, W)
            else:
                t_sce = None
                s_sce = None

            if i_scs.size >= MIN_VALID_OBS:
                t_scs_eff = t_eff_keep
                s_scs = sm[i_scs, :, :]
            else:
                t_scs_eff = None
                s_scs = None

            # 블록 결과
            sce_blk = np.full((B, W), np.nan, dtype=np.float32)
            scs_blk = np.full((B, W), np.nan, dtype=np.float32)

            # --- SCE: falling crossing ---
            if s_sce is not None:
                for rr in range(B):
                    s_row = s_sce[:, rr, :]
                    for cc in range(W):
                        s_vec = s_row[:, cc]
                        if np.isfinite(s_vec).sum() < MIN_VALID_OBS:
                            continue
                        if np.all(np.isnan(s_vec)):
                            continue
                        sce_doy = find_crossing_time_linear(
                            t_sce, s_vec, THR, rising=False
                        )
                        if np.isfinite(sce_doy):
                            sce_blk[rr, cc] = sce_doy

            # --- SCS: rising crossing ---
            if s_scs is not None:
                for rr in range(B):
                    s_row = s_scs[:, rr, :]
                    for cc in range(W):
                        s_vec = s_row[:, cc]
                        if np.isfinite(s_vec).sum() < MIN_VALID_OBS:
                            continue
                        if np.all(np.isnan(s_vec)):
                            continue
                        scs_doy_eff = find_crossing_time_linear(
                            t_scs_eff, s_vec, THR, rising=True
                        )
                        if np.isfinite(scs_doy_eff):
                            # DOY 범위가 0~366 넘으면 365 빼서 원연도로 환산
                            if scs_doy_eff > 366:
                                scs_doy_eff -= 365.0
                            scs_blk[rr, cc] = scs_doy_eff

            SCE[yi, r0:r1, :] = sce_blk
            SCS[yi, r0:r1, :] = scs_blk

    # temp 파일 정리
    try:
        del sc_mm
        os.remove(tmp_path)
    except Exception:
        pass

    return SCE, SCS


# -------------------- Orchestrate all tiles --------------------
def main():
    tile_list = [f"h{h:02d}v{v:02d}" for (h, v) in FIXED_TILES_34]
    print("* Using fixed 34 tiles:", tile_list)
    print(f"* THR(FSC threshold)={THR}, MIN_VALID_OBS={MIN_VALID_OBS}")
    print(f"* YEARS: {YEARS[0]}–{YEARS[-1]} (N={N_YEARS})")
    print(f"* OUT_DIR={OUT_DIR}")

    # yearly stacks를 위한 memmap
    sce_mm = np.memmap(
        os.path.join(OUT_DIR, "SCE_yearly.npy"),
        dtype="float32",
        mode="w+",
        shape=(len(FIXED_TILES_34), N_YEARS, DST_N, DST_N),
    )
    scs_mm = np.memmap(
        os.path.join(OUT_DIR, "SCS_yearly.npy"),
        dtype="float32",
        mode="w+",
        shape=(len(FIXED_TILES_34), N_YEARS, DST_N, DST_N),
    )

    for ti, (h, v) in enumerate(FIXED_TILES_34):
        files = list_files_for_tile(h, v)
        print(f"▶ Processing h{h:02d}v{v:02d} ({ti + 1}/{len(FIXED_TILES_34)}), files={len(files)}")

        if len(files) == 0:
            sce_mm[ti, :, :, :] = np.nan
            scs_mm[ti, :, :, :] = np.nan
            continue

        SCE, SCS = process_tile(h, v)
        if SCE is None:
            sce_mm[ti, :, :, :] = np.nan
            scs_mm[ti, :, :, :] = np.nan
            continue

        sce_mm[ti, :, :, :] = SCE
        scs_mm[ti, :, :, :] = SCS
        sce_mm.flush()
        scs_mm.flush()

    print("✅ Done. Yearly stacks saved:")
    print("  -", os.path.join(OUT_DIR, "SCE_yearly.npy"))
    print("  -", os.path.join(OUT_DIR, "SCS_yearly.npy"))


if __name__ == "__main__":
    main()


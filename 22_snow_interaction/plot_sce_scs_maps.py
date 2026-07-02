#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Plot AVE & TREND (per decade) for SCE/SCS with spatial ROI

입력:
  BASE/{SCE,SCS}_ave.npy                # (34, 1200, 1200), DOY
  BASE/{SCE,SCS}_trend_per_decade.npy   # (34, 1200, 1200), days/decade

[선택] 위경도 파일이 있을 경우:
  BASE/lat_tiles.npy, BASE/lon_tiles.npy   # 각 (34, 1200, 1200)

기능:
- 타일 인덱스 + 행/열 인덱스로 공간 영역(ROI) 제한해서 그리기
- (옵션) 위·경도 박스(BBOX_LAT, BBOX_LON)로 영역 추가 제한
- 변수(SCE/SCS), AVE/TREND 선택
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---------------- Config ----------------
BASE = "../data/MOD/SCE_SCS"

# 그릴 변수 / 필드
VAR = "SCE"               # "SCE" 또는 "SCS"
WHICH = "trend"           # "ave" 또는 "trend"

# 사용할 타일 인덱스 (0~33). 여러 개 주면 여러 장 반복 출력
TILE_IDX_LIST = [10, 11, 12]   # 예: [0], [10,11,12], 또는 list(range(34))

# 행/열 영역 제한 (0~1199), Python 슬라이스 규칙
ROW_MIN, ROW_MAX = 200, 800    # [ROW_MIN:ROW_MAX]
COL_MIN, COL_MAX = 300, 900    # [COL_MIN:COL_MAX]

# (선택) 위경도 기반 박스 (lat, lon)
USE_LATLON_BBOX = False        # True로 바꾸면 위경도 박스 사용
LAT_FILE = os.path.join(BASE, "lat_tiles.npy")
LON_FILE = os.path.join(BASE, "lon_tiles.npy")
BBOX_LAT = (30.0, 55.0)        # (lat_min, lat_max)
BBOX_LON = (90.0, 150.0)       # (lon_min, lon_max)

# 시각화 옵션
CMAP_AVE = "viridis"
CMAP_TREND = "coolwarm"
SAVE_DIR = "./FIG_SCE_SCS"     # None이면 저장 안 함
os.makedirs(SAVE_DIR, exist_ok=True)

TILES, H, W = 34, 1200, 1200


# ---------------- IO ----------------
def _raw_guess_memmap_3d(path, tiles=TILES, H=H, W=W,
                         dtypes=(np.float32, np.float64, np.int16, np.int32, np.uint16)):
    """
    .npy 헤더가 없는 raw 바이너리일 때: 파일 크기만 보고 (tiles,H,W) 3D memmap으로 복구.
    트렌드/평균 파일처럼 memmap으로 직접 만든 경우를 처리하기 위한 함수.
    """
    fsize = os.path.getsize(path)
    for dt in dtypes:
        bsz = np.dtype(dt).itemsize
        expected = tiles * H * W * bsz
        if expected == 0:
            continue
        if fsize == expected:
            print(f"  ⚠ raw memmap fallback for {os.path.basename(path)} "
                  f"(dtype={dt}, shape=({tiles},{H},{W}))")
            return np.memmap(path, mode="r", dtype=dt, shape=(tiles, H, W))
    raise RuntimeError(f"raw memmap guess failed for {path} (size={fsize})")


def safe_load_3d(path_expected):
    """
    AVE/TREND용 3D 배열 로더.
    1) np.load(mmap_mode='r') 시도
    2) 실패하면 raw memmap (3D)로 복구 시도
    """
    try:
        arr = np.load(path_expected, mmap_mode="r")
        return arr
    except Exception as e:
        print(f"  ⚠ {os.path.basename(path_expected)}: np.load 실패 ({e}), raw memmap 시도")
        arr = _raw_guess_memmap_3d(path_expected)
        return arr


def load_field(base, var, which):
    if which == "ave":
        path = os.path.join(base, f"{var}_ave.npy")
    elif which == "trend":
        path = os.path.join(base, f"{var}_trend_per_decade.npy")
    else:
        raise ValueError("WHICH must be 'ave' or 'trend'")

    arr = safe_load_3d(path)   # (34,1200,1200)가 이상적

    # 만약 4D로 로드되면 (34, Ny, 1200, 1200) → 연도 축 평균으로 3D로 변환
    if arr.ndim == 4:
        print(f"  ⚠ {os.path.basename(path)}: ndim=4 → 연도축 평균으로 3D 변환")
        arr = np.nanmean(arr, axis=1)

    if arr.ndim != 3:
        raise ValueError(f"{os.path.basename(path)}: unexpected ndim={arr.ndim}, "
                         f"expected 3 or 4")

    if arr.shape[0] != TILES or arr.shape[1:] != (H, W):
        print(f"  ⚠ {os.path.basename(path)}: shape={arr.shape}, expected (34,1200,1200)")

    return arr


def load_latlon():
    lat = safe_load_3d(LAT_FILE)
    lon = safe_load_3d(LON_FILE)
    if lat.ndim == 4:
        lat = lat[:, 0, :, :]
    if lon.ndim == 4:
        lon = lon[:, 0, :, :]
    return lat, lon


# ---------------- Helpers ----------------
def auto_limits(data, pmin=2, pmax=98):
    v = data[np.isfinite(data)]
    if v.size == 0:
        return None, None
    lo = np.nanpercentile(v, pmin)
    hi = np.nanpercentile(v, pmax)
    if lo == hi:
        hi = lo + 1e-3
    return float(lo), float(hi)


def apply_rowcol_roi(tile2d, rmin, rmax, cmin, cmax):
    return tile2d[rmin:rmax, cmin:cmax]


def apply_latlon_bbox(tile2d, lat2d, lon2d, bbox_lat, bbox_lon):
    """
    위경도 박스 밖은 NaN으로 마스킹 (shape 유지)
    """
    lat_min, lat_max = bbox_lat
    lon_min, lon_max = bbox_lon
    mask = (
        (lat2d < lat_min) | (lat2d > lat_max) |
        (lon2d < lon_min) | (lon2d > lon_max)
    )
    out = tile2d.copy()
    out[mask] = np.nan
    return out


def plot_tile(tile2d, var, which, title="", save_path=None):
    plt.figure(figsize=(6, 5))

    # 유효값 체크
    finite_mask = np.isfinite(tile2d)
    if not np.any(finite_mask):
        print("  ⚠ ROI 안에 유효한 값이 없습니다 (all NaN). 더미 스케일로 플롯합니다.")
        # 그냥 NaN 배열 imshow (matplotlib가 알아서 빈 그림처럼 그려줌)
        im = plt.imshow(tile2d, origin="upper", cmap=CMAP_TREND if which == "trend" else CMAP_AVE)
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        cb.set_label("Trend (days / decade)" if which == "trend" else "DOY")
        plt.title(title + " (no valid data)")
        plt.axis("off")
        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"  - saved: {save_path}")
        plt.show()
        return

    vmin, vmax = auto_limits(tile2d)
    if which == "trend":
        cmap = CMAP_TREND
        # zero-centered
        vabs = max(abs(vmin or 0), abs(vmax or 0))
        if vabs == 0:   # 값이 전부 0이거나 상수인 경우
            vabs = 1.0
        vmin, vmax = -vabs, vabs
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        im = plt.imshow(tile2d, origin="upper", cmap=cmap, norm=norm)
        cblabel = "Trend (days / decade)"
    else:
        cmap = CMAP_AVE
        im = plt.imshow(tile2d, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        cblabel = "DOY"

    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label(cblabel)
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  - saved: {save_path}")
    plt.show()


# ---------------- Main ----------------
def main():
    print(f"* BASE={BASE}")
    print(f"* VAR={VAR}, WHICH={WHICH}")
    print(f"* TILE_IDX_LIST={TILE_IDX_LIST}")
    print(f"* ROW[{ROW_MIN}:{ROW_MAX}], COL[{COL_MIN}:{COL_MAX}]")
    if USE_LATLON_BBOX:
        print(f"* LATLON BBOX lat={BBOX_LAT}, lon={BBOX_LON}")

    arr = load_field(BASE, VAR, WHICH)  # (34, 1200, 1200) 기대

    if USE_LATLON_BBOX:
        lat, lon = load_latlon()
    else:
        lat = lon = None

    for k in TILE_IDX_LIST:
        tile = np.array(arr[k], dtype=np.float32)  # copy

        # 1단계: 행/열로 잘라내기
        tile_roi = apply_rowcol_roi(tile, ROW_MIN, ROW_MAX, COL_MIN, COL_MAX)

        # 2단계: (선택) 위경도 박스 마스킹
        if USE_LATLON_BBOX and (lat is not None) and (lon is not None):
            lat_tile = lat[k][ROW_MIN:ROW_MAX, COL_MIN:COL_MAX]
            lon_tile = lon[k][ROW_MIN:ROW_MAX, COL_MIN:COL_MAX]
            tile_roi = apply_latlon_bbox(tile_roi, lat_tile, lon_tile,
                                         BBOX_LAT, BBOX_LON)

        title = f"{VAR} {WHICH.upper()} | tile {k} ROI"
        save_path = None
        if SAVE_DIR is not None:
            fname = (
                f"{VAR}_{WHICH}_tile{k:02d}"
                f"_r{ROW_MIN}-{ROW_MAX}_c{COL_MIN}-{COL_MAX}.png"
            )
            save_path = os.path.join(SAVE_DIR, fname)

        plot_tile(tile_roi, VAR, WHICH, title=title, save_path=save_path)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Longitudinal transect of ML LST-derived fields (LDt, LNt, LDa, LNa)
along 65°N or 67°N, using 3D (tile, H, W) grids for lat/lon and data.

- ERA/CRU 제거.
- 모든 필드는 동일한 최근접 규칙으로 샘플링:
  • 대상 경도: LON_MIN..LON_MAX, step ML_LON_STEP
  • 대상 위도: 고정 LAT_LINE
  • 후보군: (|lat - LAT_LINE| <= LAT_TOL) & (lon within [LON_MIN-LON_PAD, LON_MAX+LON_PAD])
  • KDTree 로 최근접 후보를 찾아 각 변수 값 추출
- 계절 선택: DO_WINTER=True → *_winter_ave.npy, 아니면 *_summer_ave.npy

입출력
- 입력 경로:
  ml_data_path: lat.npy, lon.npy (모두 (M,H,W))
  ml_ave_dir:   LDt_*.npy, LNt_*.npy, LDa_*.npy, LNa_*.npy (모두 (M,H,W))
- 출력: ../FIG/scatter/<Region>_<season>_lst.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset  # (여기선 사용 안하지만 환경 확인용)
from scipy.spatial import cKDTree

# ───────────────────────────────────────────────
# 사용자 설정
# ───────────────────────────────────────────────
region_key = 'U'   # 'V' = Verkhoyansk, 'U' = Ural
DO_WINTER  = False  # True = winter, False = summer

# 경로
ml_data_path = '../RAW/'        # lat.npy, lon.npy
ml_ave_dir   = '../RAW_AVE/'    # LDt_*, LNt_*, LDa_*, LNa_* (3D, (M,H,W))
fig_dir      = '../FIG/scatter/'
os.makedirs(fig_dir, exist_ok=True)

# 샘플링 파라미터
ML_LON_STEP = 0.001
LAT_TOL     = 0.10
LON_PAD     = 0.50

# 스타일
COLORS = {
    'LDt': '#d62728',  # red
    'LNt': '#1f77b4',  # blue
    'LDa': '#2ca02c',  # green
    'LNa': '#9467bd',  # purple
}
LW = 1.8
ALPHA = 0.95

# ───────────────────────────────────────────────
# 지역 설정
# ───────────────────────────────────────────────
if region_key == 'V':
    LAT_LINE     = 67.0
    LON_MIN      = 129.5
    LON_MAX      = 135.5
    LON_PLOT_MIN = 130.0
    LON_PLOT_MAX = 135.0
    TEMP_YLIM    = (-42, -27) if DO_WINTER else (-9, 12)
    XTICKS       = np.arange(130, 136, 1)
    YTICKS       = np.arange(-42, -26, 3) if DO_WINTER else np.arange(-9, 13, 3)
    TITLE_FMT    = "Verkhoyansk [{lat:.0f}°N]"
    FNAME_FMT    = "Verkhoyansk_{season}_lst.png"
elif region_key == 'U':
    LAT_LINE     = 65.0
    LON_MIN      = 57.5
    LON_MAX      = 62.5
    LON_PLOT_MIN = 58.0
    LON_PLOT_MAX = 62.0
    TEMP_YLIM    = (-26, -15)  if DO_WINTER else (-6, 11)
    XTICKS       = np.arange(58, 63, 1)
    YTICKS       = np.arange(-26, -14, 3)  if DO_WINTER else np.arange(-6, 12, 3)
    TITLE_FMT    = "Ural [{lat:.0f}°N]"
    FNAME_FMT    = "Ural_{season}_lst.png"
else:
    raise ValueError(f"Unknown region_key: {region_key}")

SEASON_TAG  = 'winter' if DO_WINTER else 'summer'
SEASON_NAME = "Cold (Nov–Mar)" if DO_WINTER else "Warm (Apr–Oct)"

# ───────────────────────────────────────────────
# 유틸
# ───────────────────────────────────────────────
def load_grid3d(path):
    lat = np.load(os.path.join(path, 'lat.npy'))  # (M,H,W)
    lon = np.load(os.path.join(path, 'lon.npy'))  # (M,H,W)
    if lat.shape != lon.shape or lat.ndim != 3:
        raise ValueError(f"lat/lon must be (M,H,W), got lat {lat.shape}, lon {lon.shape}")
    return lat.astype(np.float32), lon.astype(np.float32)

def load_field3d(name, season_tag, expect_shape):
    fp = os.path.join(ml_ave_dir, f"{name}_{season_tag}_ave.npy")
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    arr = np.load(fp)
    if arr.shape != expect_shape:
        raise ValueError(f"{name} shape {arr.shape} != lat/lon shape {expect_shape}")
    return arr.astype(np.float32)

def sample_lines_by_kdtree(lat3d, lon3d, values3d, lat_line, lon_min, lon_max,
                           step=0.001, lat_tol=0.1, lon_pad=0.5):
    """
    lat3d/lon3d/values3d: (M,H,W)
    반환: (target_lons, line_vals) where line_vals.shape == (n_points,)
    """
    target_lons = np.arange(lon_min, lon_max + step/2, step, dtype=np.float64)
    target_lats = np.full_like(target_lons, lat_line, dtype=np.float64)

    mask = (np.abs(lat3d - lat_line) <= lat_tol) & \
           (lon3d >= (lon_min - lon_pad)) & (lon3d <= (lon_max + lon_pad))
    if not np.any(mask):
        raise RuntimeError("No candidate pixels; widen LAT_TOL/LON_PAD.")

    lat_cand = lat3d[mask].ravel()
    lon_cand = lon3d[mask].ravel()
    val_cand = values3d[mask].ravel()

    # 최근접 매핑
    tree = cKDTree(np.column_stack([lat_cand, lon_cand]))
    _, idx = tree.query(np.column_stack([target_lats, target_lons]))
    line_vals = val_cand[idx].astype(np.float32)
    return target_lons.astype(np.float32), line_vals

# ───────────────────────────────────────────────
# 데이터 로드
# ───────────────────────────────────────────────
print("• Loading 3D grid ...")
lat_ml, lon_ml = load_grid3d(ml_data_path)   # (M,H,W)
MHW = lat_ml.shape

print("• Loading 3D fields (LDt/LNt/LDa/LNa) ...")
LDt = load_field3d('LDt', SEASON_TAG, MHW)
LNt = load_field3d('LNt', SEASON_TAG, MHW)
LDa = load_field3d('LDa', SEASON_TAG, MHW)
LNa = load_field3d('LNa', SEASON_TAG, MHW)

# ───────────────────────────────────────────────
# 라인 샘플링 (네 필드 동일 최근접 규칙)
# ───────────────────────────────────────────────
print("• Sampling longitudinal lines via KDTree ...")
target_lons, LDt_line = sample_lines_by_kdtree(
    lat_ml, lon_ml, LDt, LAT_LINE, LON_MIN, LON_MAX,
    step=ML_LON_STEP, lat_tol=LAT_TOL, lon_pad=LON_PAD
)
_, LNt_line = sample_lines_by_kdtree(
    lat_ml, lon_ml, LNt, LAT_LINE, LON_MIN, LON_MAX,
    step=ML_LON_STEP, lat_tol=LAT_TOL, lon_pad=LON_PAD
)
_, LDa_line = sample_lines_by_kdtree(
    lat_ml, lon_ml, LDa, LAT_LINE, LON_MIN, LON_MAX,
    step=ML_LON_STEP, lat_tol=LAT_TOL, lon_pad=LON_PAD
)
_, LNa_line = sample_lines_by_kdtree(
    lat_ml, lon_ml, LNa, LAT_LINE, LON_MIN, LON_MAX,
    step=ML_LON_STEP, lat_tol=LAT_TOL, lon_pad=LON_PAD
)

# ───────────────────────────────────────────────
# 플롯
# ───────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.6))

ax.plot(target_lons, LDt_line, color=COLORS['LDt'], lw=LW, alpha=ALPHA, label='LDt')
ax.plot(target_lons, LNt_line, color=COLORS['LNt'], lw=LW, alpha=ALPHA, label='LNt')
ax.plot(target_lons, LDa_line, color=COLORS['LDa'], lw=LW, alpha=ALPHA, label='LDa')
ax.plot(target_lons, LNa_line, color=COLORS['LNa'], lw=LW, alpha=ALPHA, label='LNa')

title = TITLE_FMT.format(lat=LAT_LINE)
ax.set_title(title)
ax.set_xlim(LON_PLOT_MIN, LON_PLOT_MAX)
ax.set_ylim(*TEMP_YLIM)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Value")
ax.grid(True, alpha=0.30)
ax.set_xticks(XTICKS)
ax.set_yticks(YTICKS)
ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.20))

plt.tight_layout()
out_png = os.path.join(fig_dir, FNAME_FMT.format(season=('winter' if DO_WINTER else 'summer')))
plt.savefig(out_png, dpi=300)
plt.close()
print("✅ 완료:", out_png)

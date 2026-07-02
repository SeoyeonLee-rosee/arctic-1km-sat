# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.path as mpath
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata

# 월 설정
stmm, enmm = 5, 6  # 1~12월 처리 가능

# 변수 설정
variable_indices = ["LDt", "NDt", "NDa", "LNa", "LDa", "LNt"]
file_types = ["ave", "trend"]

# 저장 경로 설정
input_dir = "./"
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

# ✅ lat, lon 데이터 로드
lat = np.load("./raw/lat.npy")  # (34, 1200, 1200)
lon = np.load("./raw/lon.npy")  # (34, 1200, 1200)

# 목표 격자 생성 (위도: 60-90N, 경도: -180~180)
lat_target = np.arange(60, 90, 0.25)  # 60~90N, 0.25도 간격
lon_target = np.arange(-180, 180, 0.25)  # -180~180, 0.24도 간격
lon_target, lat_target = np.meshgrid(lon_target, lat_target)

# 마스킹 데이터 로드
mask_data_1 = np.load("./raw/sat_0.25.npy")  # 마스킹 데이터 (120, 1440)
mask_data=mask_data_1[0,0,:,:]
# lat, lon 크기 확인
if lat.shape[0] != 34 or lon.shape[0] != 34:
    raise ValueError("lat 및 lon의 첫 번째 차원 크기가 34가 아닙니다. 데이터 형식을 확인하세요.")

# NaN 또는 Inf 값 제거 및 범위 조정
lat = np.nan_to_num(lat, nan=0.0, posinf=90.0, neginf=-90.0)
lon = np.nan_to_num(lon, nan=0.0, posinf=180.0, neginf=-180.0)

# ✅ 변수 그룹별 min/max 스케일 계산 (스케일 통일)
data_ranges = {"ND": None, "L": None}

# 먼저 min/max 값 찾기
for var in variable_indices:
    var_type = "ND" if "ND" in var else "L"
    all_data = []
    
    for mm in range(stmm, enmm + 1):
        for ftype in file_types:
            file_path = os.path.join(input_dir, f"{var}_{ftype}_{mm:02d}.npy")
            if os.path.exists(file_path):
                data = np.load(file_path)
                all_data.append(data)
    
    if all_data:
        stacked_data = np.concatenate([d.flatten() for d in all_data])
        data_ranges[var_type] = (np.nanmin(stacked_data), np.nanmax(stacked_data))  # NaN 무시

print(f"데이터 범위: {data_ranges}")

# ✅ 변수별, 월별 데이터 리그리딩 및 마스킹
for mm in range(stmm, enmm + 1):
    for var in variable_indices:
        for ftype in file_types:
            file_path = os.path.join(input_dir, f"{var}_{ftype}_{mm:02d}.npy")
            
            if not os.path.exists(file_path):
                print(f"파일 없음: {file_path}, 건너뜀")
                continue

            # 데이터 로드
            data = np.load(file_path)  # (34, 1200, 1200)
            print(f"Processing: {file_path}, Shape: {data.shape}")
            
            # 34개 패치를 하나의 데이터로 합치기
            lat_combined = lat.reshape(-1)
            lon_combined = lon.reshape(-1)
            data_combined = data.reshape(-1)

            # 유효한 데이터만 선택 (NaN 제거 및 경도 범위 제한)
            valid_mask = (~np.isnan(data_combined)) & (~np.isnan(lon_combined)) & (lon_combined >= -180) & (lon_combined <= 180)
            lat_valid = lat_combined[valid_mask]
            lon_valid = lon_combined[valid_mask]
            data_valid = data_combined[valid_mask]

            # 목표 격자로 보간 (0.25도 간격), 메모리 절약을 위해 meshgrid 사용 안 함
            points_target = np.column_stack((lon_target.ravel(), lat_target.ravel()))
            grid_data = griddata((lon_valid, lat_valid), data_valid, points_target, method="linear")
            grid_data = grid_data.reshape((120, 1440))  # 강제 변환

            # 마스킹 적용 (mask_data 사용)
            if mask_data.shape == grid_data.shape:
                grid_data[np.isnan(mask_data)] = np.nan
            else:
                print(f"⚠️ Warning: Mask shape {mask_data.shape} does not match grid shape {grid_data.shape}. Skipping masking.")

            # 저장
            output_file = os.path.join(output_dir, f"{var}_{ftype}_{mm:02d}_regridded.npy")
            np.save(output_file, grid_data)
            print(f"Saved: {output_file}, Shape: {grid_data.shape}")

print("모든 데이터 리그리딩 및 마스킹 완료!")

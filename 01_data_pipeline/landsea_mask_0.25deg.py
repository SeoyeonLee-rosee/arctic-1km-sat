# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import xarray as xr
import os
import numpy as np

# 파일 경로 설정
file_path = "./raw/land_sea_mask_0.01.nc"

# Load the NetCDF file using xarray
dataset = xr.open_dataset(file_path)

# Extract dataset structure information
dataset_info = {
    "dimensions": dict(dataset.dims),
    "variables": list(dataset.variables.keys()),
}

# Extract the land-sea mask variable again
lsm_data = dataset["lsm"].values
print(f"Original mask shape: {lsm_data.shape}")

# Check unique values in the mask
unique_values = np.unique(lsm_data)

# Shape and unique values of the land-sea mask
lsm_info = {
    "shape": lsm_data.shape,
    "unique_values": unique_values.tolist(),
}

# Get min and max values to determine threshold for land vs. sea
min_value = np.min(lsm_data)
max_value = np.max(lsm_data)

# Assuming values close to 0 represent the ocean, let's check a reasonable threshold
threshold = 0.5  # Often, a value < 0.5 means ocean

mask_threshold_info = {
    "min_value": min_value,
    "max_value": max_value,
    "proposed_threshold": threshold
}

# Create a binary ocean mask where ocean areas are NaN
ocean_mask_1 = np.where(lsm_data < 0.5, np.nan, 1)  # Ocean = NaN, Land = 1

# Extract the required shape (Ensure the correct slicing)
ocean_mask = ocean_mask_1[0, :120, :]
ocean_mask = ocean_mask[::-1,:]
print(f"Final mask shape (after slicing): {ocean_mask.shape}")

# Save the ocean mask for reference
ocean_mask_path = "./raw/ocean_mask.npy"
os.makedirs("./raw", exist_ok=True)
np.save(ocean_mask_path, ocean_mask)

# 저장 경로 설정
input_dir = "./"
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

# ✅ 마스킹 데이터 로드
ocean_mask = np.load("./raw/ocean_mask.npy")  # 생성된 해양 마스크

# ✅ 변수 및 월 설정
stmm, enmm = 1, 12  # 1~12월 처리 가능
variable_indices = ["LDt", "NDt", "NDa", "LNa", "LDa", "LNt"]
file_types = ["ave", "trend"]

# ✅ 저장된 파일에 대해 새로운 마스킹 적용
for mm in range(stmm, enmm + 1):
    for var in variable_indices:
        for ftype in file_types:
            file_path = os.path.join(output_dir, f"{var}_{ftype}_{mm:02d}_regridded.npy")
            
            if not os.path.exists(file_path):
                print(f"파일 없음: {file_path}, 건너뜀")
                continue

            # 데이터 로드
            grid_data = np.load(file_path)
            print(f"Processing: {file_path}, Shape: {grid_data.shape}")

            # 해양 마스킹 적용 (해양은 NaN 처리)
            if ocean_mask.shape == grid_data.shape:
                grid_data[np.isnan(ocean_mask)] = np.nan  # 해양 지역을 NaN으로 설정
            else:
                print(f"⚠️ Warning: Mask shape {ocean_mask.shape} does not match grid shape {grid_data.shape}. Skipping masking.")

            # 새로운 마스킹 적용 후 저장
            output_file = os.path.join(output_dir, f"{var}_{ftype}_{mm:02d}_masked.npy")
            np.save(output_file, grid_data)
            print(f"Saved: {output_file}, Shape: {grid_data.shape}")

print("모든 데이터에 새로운 해양 마스킹 적용 완료!")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""LNT, LDT, LNA, LDA 계절별(겨울/여름) 평균 계산 및 저장"""

import os, numpy as np

# ───────── 1. 데이터 읽기 & 초기 설정 ─────────────────────────────────
variables = ['LNt', 'LDt', 'LNa', 'LDa']

winter_months = ['11', '12', '01', '02', '03']
summer_months = ['04', '05', '06', '07', '08', '09', '10']

# ───────── 2. 계절별 평균 계산 함수 ──────────────────────────────────
def seasonal_mean(var, months):
    count = 0
    sum_data = None
    for yy in range(2000, 2023):
        for mm in months:
            if (yy == 2000 and mm in ['01', '02']) or (yy == 2022 and mm not in ['01', '02', '03', '04']):
                continue

            filepath = f"../RAW/{var}_{mm}.npy"
            data_month = np.load(filepath)
            year_index = yy - 2000
            if year_index >= data_month.shape[0]: continue
            data = data_month[year_index]

            if sum_data is None:
                sum_data = np.zeros_like(data)

            np.nan_to_num(data, copy=False, nan=0.0)
            sum_data += data
            count += 1

    mean_data = sum_data / count
    mean_data[mean_data == 0] = np.nan
    return mean_data

# ───────── 3. 평균 계산 후 저장 ──────────────────────────────────────
outdir = "../RAW_AVE/"; os.makedirs(outdir, exist_ok=True)

for var in variables:
    winter_mean = seasonal_mean(var, winter_months)
    summer_mean = seasonal_mean(var, summer_months)

    np.save(os.path.join(outdir, f"{var}_winter_ave.npy"), winter_mean)
    np.save(os.path.join(outdir, f"{var}_summer_ave.npy"), summer_mean)

    print(f"Saved {var}_winter_ave.npy and {var}_summer_ave.npy")


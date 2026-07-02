# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
import pandas as pd
import xarray as xr

# --- 설정 및 상수 ---
stnpath      = '../data/CRU/'
MODpath      = '../RAW/'
station_file = os.path.join(stnpath, 'converted_sat.csv')
tile_file    = os.path.join(stnpath, 'sn_bound.txt')
cru_file     = os.path.join(stnpath, 'grid', 'cru_ts4.09.1901.2024.tmp.dat.nc')
output_file  = os.path.join(MODpath, 'sat_station_data.csv')

# 좌표 변환용 상수
rr = 6371007.181
xmin   = -20015109.0
tt     = 1111950.0
npixel = 1200
res    = tt / npixel  # 해상도
# ML 허용 오차 단계적 확대 리스트
degree_list = [0.02, 0.05, 0.10, 0.25]

def geomapping(hh, vv, npix):
    """격자 타일 (hh, vv) → 위경도 매핑"""
    mlat = np.zeros((npix, npix))
    mlon = np.zeros((npix, npix))
    for jj in range(npix):
        yy = (9 - vv) * tt - (jj + 0.5) * res
        lat = yy * 180.0 / (rr * np.pi)
        mlat[jj, :] = lat
        for ii in range(npix):
            xx = (ii + 0.5) * res + hh * tt + xmin
            lon = xx * 180.0 / (rr * np.cos(lat * np.pi / 180.0)) / np.pi
            mlon[jj, ii] = lon
    return mlat, mlon

# --- 1) 데이터 로드 ---
ds = xr.open_dataset(cru_file)
# 시간 범위 2000-03 ~ 2022-04
ds2 = ds.sel(time=slice('2000-03','2022-04'))
# 중간일 → 월초로 변환
time_mid   = pd.to_datetime(ds2.time.values)
time_idx   = time_mid.to_period('M').to_timestamp()
lat_vals   = ds2.lat.values
lon_vals   = ds2.lon.values

df_stn = pd.read_csv(station_file)

# --- 2) 각 스테이션별 ML·CRU 계산 ---
results = []
years   = np.arange(2000, 2023)
n_years = len(years)
# 타일 메타데이터 로드 (외부 루프 전에 한 번만)
tile = np.genfromtxt(tile_file, skip_header=7)

for stn, group in df_stn.groupby('STN'):
    lat     = group['LAT'].iloc[0]
    lon     = group['LON'].iloc[0]
    alt     = group['ALT'].iloc[0]
    wght    = group['W'].iloc[0]
    lon_adj = lon + 360 if lon < 0 else lon

    # 1) 스테이션이 속한 타일 찾기
    mask = (
        (tile[:,4] <= lat) & (lat <= tile[:,5]) &
        (tile[:,2] <= lon) & (lon <= tile[:,3])
    )
    idx_tiles = np.where(mask)[0]
    if idx_tiles.size == 0:
        raise ValueError(f"스테이션 {stn}의 타일을 찾을 수 없습니다.")
    j   = idx_tiles[0]
    vv  = int(tile[j,0]); hh = int(tile[j,1]); hv = int(tile[j,6])

    # 2) 타일 내 위경도 매핑
    mlat, mlon = geomapping(hh, vv, npixel)

    # 3) 관심 픽셀 인덱스 찾기 (허용 오차 단계적 확대)
    for deg in degree_list:
        idx_pts = np.where(
            (mlat >  lat - deg) & (mlat <  lat + deg) &
            (mlon > lon_adj - deg) & (mlon < lon_adj + deg)
        )
        if idx_pts[0].size > 0:
            break
    else:
        # 전부 NaN일 경우 가장 가까운 픽셀 하나 선택
        dist2   = (mlat - lat)**2 + (mlon - lon_adj)**2
        flat    = np.nanargmin(dist2)
        idx_pts = np.unravel_index(flat, mlat.shape)

    a, b = idx_pts
    n_pts = a.size if hasattr(a, '__len__') else 1

    # 4) ML 시계열 배열 초기화 (12개월 × 년도 × 포인트)
    res_ml = np.full((12, n_years, n_pts), np.nan)
    for m in range(1, 13):
        im    = m - 1
        arr_p = np.load(os.path.join(MODpath, f"sat_{m:02}.npy"), mmap_mode='r')
        n_loc = arr_p.shape[1]
        start = 1 if m <= 2 else 0
        # 월별 타임스텝 × 픽셀맵
        tmp   = arr_p[hv, :n_loc, :, :]
        # 관심 픽셀 값 추출 → (n_loc,) 또는 (n_loc, n_pts)
        sel   = tmp[:, a, b]
        # 2D로 reshape: (n_loc, n_pts)
        sel   = sel.reshape(sel.shape[0], -1)
        # 결과 배열에 할당
        res_ml[im, start:start+n_loc, :] = sel

    # 5) CRU 시계열 추출 (가장 가까운 격자)
    dist2      = (lat_vals[:,None] - lat)**2 + (lon_vals[None,:] - lon_adj)**2
    flat_idx   = np.nanargmin(dist2)
    li, lj     = np.unravel_index(flat_idx, dist2.shape)
    cru_arr    = ds2['tmp'].isel(lat=li, lon=lj).values
    cru_series = pd.Series(cru_arr, index=time_idx)

    # 6) SAT·ML·CRU 결합 및 저장
    for _, row in group.iterrows():
        yy, mm = int(row['YEAR']), int(row['MONTH'])
        sat    = float(row['SAT'])

        # ML 값 계산
        ml_val = np.nan
        if 2000 <= yy <= 2022:
            iy = yy - 2000
            if (mm <= 2 and yy >= 2001) or (mm >= 3):
                vals = res_ml[mm-1, iy, :]
                if vals.size and not np.all(np.isnan(vals)):
                    ml_val = np.nanmean(vals)

        # CRU 값 추출
        ts      = pd.Timestamp(f"{yy}-{mm:02d}")
        cru_val = cru_series.get(ts, np.nan)

        results.append({
            'STN': stn, 'YEAR': yy, 'MONTH': mm,
            'LAT': lat, 'LON': lon, 'ALT': alt, 'W': wght,
            'SAT': sat, 'ML': ml_val, 'CRU': cru_val
        })

# --- 3) 결과 저장 ---
df_out = pd.DataFrame(results)
df_out.to_csv(output_file, index=False)
print(f"완료: {output_file}에 저장되었습니다.")

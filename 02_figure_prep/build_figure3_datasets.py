#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/ 기준

RAW_DIR = os.path.join(SCRIPT_DIR, "..", "RAW")
MODIS_DIR = os.path.join(SCRIPT_DIR, "..", "data", "MODIS")
GPP_ROOT = os.path.join(SCRIPT_DIR, "..", "data", "MOD", "GPP")
SCE_DIR = os.path.join(SCRIPT_DIR, "..", "data", "MOD", "SCE")
CRU_NC = os.path.join(SCRIPT_DIR, "..", "data", "CRU", "grid", "cru_ts4.06.1901.2021.tmp.dat.nc")

OUT_ROOT = os.path.join(SCRIPT_DIR, "..", "data", "figure3")
OUT_GRID = os.path.join(OUT_ROOT, "grid05")
OUT_REG = os.path.join(OUT_ROOT, "regions")
os.makedirs(OUT_GRID, exist_ok=True)
os.makedirs(OUT_REG, exist_ok=True)

YEARS_TARGET = np.arange(2000, 2022, dtype=np.int32)
MONTHS = np.arange(1, 13, dtype=np.int32)
WARM_MONTH_IDXS = [3,4,5,6,7,8,9]
EXPECTED_TILES = 34
DST_N = 1200

REGIONS = [
    dict(key="UR", rowname="Ural",             lon=(58.0, 63.0),      lat=(62.0, 67.0)),
    dict(key="CS", rowname="Central Siberia",  lon=(89.0, 94.0),      lat=(65.0, 70.0)),
    dict(key="ES", rowname="Eastern Siberia",  lon=(125, 130),    lat=(65.0, 70.0)),
    dict(key="AK", rowname="Northern Alaska",  lon=(-155.0, -150.0),  lat=(63.5, 68.5)),
]

PAD_LON = 3.0
PAD_LAT = 3.0

def log(msg):
    print(f"[build_figure3_inputs_v7] {msg}")

def np_load_robust(path, mmap=False):
    if mmap:
        try:
            return np.load(path, mmap_mode="r")
        except Exception:
            return np.load(path, allow_pickle=True, mmap_mode="r")
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return np.load(path, allow_pickle=True)

def monthly_years_from_convention(mm):
    if mm <= 2:
        styy, enyy = 2001, 2022
    elif mm in (3, 4):
        styy, enyy = 2000, 2022
    else:
        styy, enyy = 2000, 2021
    return np.arange(styy, enyy + 1, dtype=np.int32)

def normalize_time_first_tile_stack(arr):
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array, got {arr.shape}")
    if arr.shape[0] == EXPECTED_TILES:
        return np.moveaxis(arr, 1, 0).astype(np.float32)
    if arr.shape[1] == EXPECTED_TILES:
        return arr.astype(np.float32)
    raise ValueError(f"Could not infer time/tile axis: {arr.shape}")

def hydro_to_calendar_doy(arr):
    arr = np.asarray(arr, dtype=np.float32)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    m = np.isfinite(arr)
    a = arr[m]
    out[m] = np.where(a >= 154.0, a - 153.0, a + 212.0)
    return out

def load_modis_latlon():
    lat = np_load_robust(os.path.join(MODIS_DIR, "lat.npy")).astype(np.float32)
    lon = np_load_robust(os.path.join(MODIS_DIR, "lon.npy")).astype(np.float32)
    lon = np.where(lon > 180.0, lon - 360.0, lon).astype(np.float32)
    return lat, lon

def choose_sat_file(mm):
    p_land = os.path.join(RAW_DIR, f"sat_{mm:02d}_land.npy")
    p_raw = os.path.join(RAW_DIR, f"sat_{mm:02d}.npy")
    return p_land if os.path.exists(p_land) else p_raw

def load_cru_grid_and_cube():
    ds = xr.open_dataset(CRU_NC)
    lat = ds["lat"].values.astype(np.float32)
    lon = ds["lon"].values.astype(np.float32)
    lon = np.where(lon > 180.0, lon - 360.0, lon).astype(np.float32)

    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    lat_mask = (lat >= 60.0) & (lat <= 90.0)
    lat = lat[lat_mask]

    da = ds["tmp"]
    da = da.sel(time=da["time"].dt.year.isin(YEARS_TARGET))
    da = da.sel(time=da["time"].dt.month.isin(MONTHS))
    da = da.where(ds["lat"] >= 60.0, drop=True)
    da = da.where(ds["lat"] <= 90.0, drop=True)

    years = da["time"].dt.year.values.astype(np.int32)
    months = da["time"].dt.month.values.astype(np.int32)

    cube = np.full((len(YEARS_TARGET), 12, len(lat), len(lon)), np.nan, dtype=np.float32)
    for i in range(da.sizes["time"]):
        yy = int(years[i]); mm = int(months[i])
        iy = np.where(YEARS_TARGET == yy)[0][0]
        cube[iy, mm - 1] = da.isel(time=i).values.astype(np.float32)[:, lon_order]

    land_mask = np.any(np.isfinite(cube), axis=(0,1))
    ds.close()
    return lat, lon, cube, land_mask

def build_cellmean_index(lat_native, lon_native, target_lat, target_lon):
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    dlat = np.median(np.diff(target_lat))
    dlon = np.median(np.diff(target_lon))
    lat0 = target_lat[0]
    lon0 = target_lon[0]

    latf = lat_native.ravel().astype(np.float64)
    lonf = lon_native.ravel().astype(np.float64)
    valid = np.isfinite(latf) & np.isfinite(lonf)
    valid_idx_full = np.where(valid)[0]

    latf = latf[valid]
    lonf = lonf[valid]

    ilat = np.floor((latf - (lat0 - dlat/2.0)) / dlat).astype(int)
    ilon = np.floor((lonf - (lon0 - dlon/2.0)) / dlon).astype(int)

    good = (ilat >= 0) & (ilat < target_lat.size) & (ilon >= 0) & (ilon < target_lon.size)
    valid_idx = valid_idx_full[good]
    ilat = ilat[good]
    ilon = ilon[good]
    weights = np.cos(np.deg2rad(latf[good])).astype(np.float64)

    base_count = np.zeros((target_lat.size, target_lon.size), dtype=np.int32)
    np.add.at(base_count, (ilat, ilon), 1)
    return valid_idx, ilat, ilon, weights, base_count

def regrid_field_cellmean(field_native, regrid_index, nlat, nlon):
    valid_idx, ilat, ilon, weights, _ = regrid_index
    vf = field_native.ravel().astype(np.float64)[valid_idx]
    m = np.isfinite(vf)

    out_sum = np.zeros((nlat, nlon), dtype=np.float64)
    out_wgt = np.zeros((nlat, nlon), dtype=np.float64)
    out_cnt = np.zeros((nlat, nlon), dtype=np.int16)

    np.add.at(out_sum, (ilat[m], ilon[m]), vf[m] * weights[m])
    np.add.at(out_wgt, (ilat[m], ilon[m]), weights[m])
    np.add.at(out_cnt, (ilat[m], ilon[m]), 1)

    out = np.full((nlat, nlon), np.nan, dtype=np.float32)
    ok = out_wgt > 0
    out[ok] = (out_sum[ok] / out_wgt[ok]).astype(np.float32)
    return out, out_cnt

def build_ml_05(regrid_index, nlat, nlon):
    cube = np.full((len(YEARS_TARGET), 12, nlat, nlon), np.nan, dtype=np.float32)
    cnts = np.zeros((len(YEARS_TARGET), 12, nlat, nlon), dtype=np.int16)
    for mm in MONTHS:
        path = choose_sat_file(mm)
        arr = normalize_time_first_tile_stack(np_load_robust(path, mmap=True))
        years = monthly_years_from_convention(mm)
        for i, yy in enumerate(years):
            if yy < YEARS_TARGET[0] or yy > YEARS_TARGET[-1]:
                continue
            iy = np.where(YEARS_TARGET == yy)[0][0]
            field, cnt = regrid_field_cellmean(arr[i], regrid_index, nlat, nlon)
            cube[iy, mm - 1] = field
            cnts[iy, mm - 1] = cnt
        log(f"ML month {mm:02d} done from {os.path.basename(path)}")
    cube[0, 0:2] = np.nan
    cnts[0, 0:2] = 0
    return cube, cnts

def build_gpp_05(regrid_index, nlat, nlon):
    cube = np.full((len(YEARS_TARGET), 12, nlat, nlon), np.nan, dtype=np.float32)
    cnts = np.zeros((len(YEARS_TARGET), 12, nlat, nlon), dtype=np.int16)
    for mm in MONTHS:
        root = os.path.join(GPP_ROOT, f"{mm:02d}")
        data = normalize_time_first_tile_stack(np_load_robust(os.path.join(root, "GPP_tiles.npy"), mmap=True))
        years = np.load(os.path.join(root, "years.npy")).astype(np.int32)
        data = data.astype(np.float32)
        data[data > 0.5] = np.nan
        data *= 1000.0
        for i, yy in enumerate(years):
            if yy < YEARS_TARGET[0] or yy > YEARS_TARGET[-1]:
                continue
            iy = np.where(YEARS_TARGET == yy)[0][0]
            field, cnt = regrid_field_cellmean(data[i], regrid_index, nlat, nlon)
            cube[iy, mm - 1] = field
            cnts[iy, mm - 1] = cnt
        log(f"GPP month {mm:02d} done")
    cube[0, 0:2] = np.nan
    cnts[0, 0:2] = 0
    return cube, cnts

def load_sce_native():
    years = np.load(os.path.join(SCE_DIR, "hydro_years.npy")).astype(np.int32)
    path = os.path.join(SCE_DIR, "SCE_yearly.npy")
    try:
        arr = np_load_robust(path, mmap=True)
        if arr.shape[0] == EXPECTED_TILES:
            arr = np.moveaxis(arr, 1, 0).astype(np.float32)
        elif arr.shape[1] == EXPECTED_TILES:
            arr = arr.astype(np.float32)
        else:
            raise ValueError(f"Unexpected SCE shape: {arr.shape}")
    except Exception:
        log("SCE_yearly.npy appears to be raw float32 memmap; opening as raw stack.")
        nyr = len(years)
        arr = np.memmap(path, dtype="float32", mode="r", shape=(EXPECTED_TILES, nyr, DST_N, DST_N))
        arr = np.moveaxis(arr, 1, 0).astype(np.float32)

    vmax = np.nanmax(arr)
    if np.isfinite(vmax) and vmax > 220:
        arr = hydro_to_calendar_doy(arr)
    return years, arr

def build_sce_05(regrid_index, nlat, nlon):
    sce_years, sce_native = load_sce_native()
    cube = np.full((len(YEARS_TARGET), nlat, nlon), np.nan, dtype=np.float32)
    cnts = np.zeros((len(YEARS_TARGET), nlat, nlon), dtype=np.int16)
    for i, yy in enumerate(sce_years):
        if yy < YEARS_TARGET[0] or yy > YEARS_TARGET[-1]:
            continue
        iy = np.where(YEARS_TARGET == yy)[0][0]
        field, cnt = regrid_field_cellmean(sce_native[i], regrid_index, nlat, nlon)
        cube[iy] = field
        cnts[iy] = cnt
    return cube, cnts, sce_years, sce_native

def fill_nan_nearest_2d(arr):
    arr = np.asarray(arr, dtype=np.float32)
    out = arr.copy()
    good = np.isfinite(out)
    if np.all(good) or (not np.any(good)):
        return out
    inds = distance_transform_edt(~good, return_distances=False, return_indices=True)
    out[~good] = out[tuple(inds[:, ~good])]
    return out

def region_mask_native(lat2d, lon2d, reg):
    return (
        np.isfinite(lat2d) & np.isfinite(lon2d) &
        (lat2d >= reg["lat"][0]) & (lat2d <= reg["lat"][1]) &
        (lon2d >= reg["lon"][0]) & (lon2d <= reg["lon"][1])
    )

def subset_cru_source(lat1d, lon1d, reg):
    lat_min = reg["lat"][0] - PAD_LAT
    lat_max = reg["lat"][1] + PAD_LAT
    lon_min = reg["lon"][0] - PAD_LON
    lon_max = reg["lon"][1] + PAD_LON
    iy = np.where((lat1d >= lat_min) & (lat1d <= lat_max))[0]
    ix = np.where((lon1d >= lon_min) & (lon1d <= lon_max))[0]
    if len(iy) < 2 or len(ix) < 2:
        raise RuntimeError(f"CRU source subset too small for {reg['key']}")
    return iy, ix

def build_ml_warm_native():
    month_data = {}; month_years = {}
    for mm in range(4, 11):
        path = choose_sat_file(mm)
        arr = normalize_time_first_tile_stack(np_load_robust(path, mmap=True))
        years = monthly_years_from_convention(mm)
        keep = np.isin(years, YEARS_TARGET)
        month_data[mm] = arr[keep].astype(np.float32)
        month_years[mm] = years[keep]
    out = []
    for yy in YEARS_TARGET:
        fields = []
        for mm in range(4, 11):
            idx = np.where(month_years[mm] == yy)[0]
            if len(idx) == 0:
                fields = []; break
            fields.append(month_data[mm][idx[0]])
        out.append(np.nanmean(np.stack(fields, axis=0), axis=0).astype(np.float32) if fields else np.full((EXPECTED_TILES,1200,1200), np.nan, dtype=np.float32))
    return np.stack(out, axis=0)

def build_gpp_warm_native():
    month_data = {}; month_years = {}
    for mm in range(4, 11):
        root = os.path.join(GPP_ROOT, f"{mm:02d}")
        data = normalize_time_first_tile_stack(np_load_robust(os.path.join(root, "GPP_tiles.npy"), mmap=True))
        years = np.load(os.path.join(root, "years.npy")).astype(np.int32)
        data = data.astype(np.float32); data[data > 0.5] = np.nan; data *= 1000.0
        keep = np.isin(years, YEARS_TARGET)
        month_data[mm] = data[keep]; month_years[mm] = years[keep]
    out = []
    for yy in YEARS_TARGET:
        fields = []
        for mm in range(4, 11):
            idx = np.where(month_years[mm] == yy)[0]
            if len(idx) == 0:
                fields = []; break
            fields.append(month_data[mm][idx[0]])
        out.append(np.nanmean(np.stack(fields, axis=0), axis=0).astype(np.float32) if fields else np.full((EXPECTED_TILES,1200,1200), np.nan, dtype=np.float32))
    return np.stack(out, axis=0)

def save_npz(path, **kwargs):
    np.savez_compressed(path, **kwargs)
    print(f"[saved] {path}")

def main():
    lat_native, lon_native = load_modis_latlon()
    cru_lat, cru_lon, cru_cube, cru_land_mask = load_cru_grid_and_cube()
    nlat, nlon = len(cru_lat), len(cru_lon)
    regrid_index = build_cellmean_index(lat_native, lon_native, cru_lat, cru_lon)
    _, _, _, _, base_count = regrid_index

    log("Build grid05 products ...")
    ml05, mlcnt = build_ml_05(regrid_index, nlat, nlon)
    gpp05, gppcnt = build_gpp_05(regrid_index, nlat, nlon)
    sce05, scecnt, sce_native_years, sce_native = build_sce_05(regrid_index, nlat, nlon)

    cru_cube[:, :, ~cru_land_mask] = np.nan
    ml05[:, :, ~cru_land_mask] = np.nan
    gpp05[:, :, ~cru_land_mask] = np.nan
    sce05[:, ~cru_land_mask] = np.nan

    LAT05, LON05 = np.meshgrid(cru_lat, cru_lon, indexing="ij")
    np.save(os.path.join(OUT_GRID, "ML.npy"), ml05.astype(np.float32))
    np.save(os.path.join(OUT_GRID, "GPP.npy"), gpp05.astype(np.float32))
    np.save(os.path.join(OUT_GRID, "CRU.npy"), cru_cube.astype(np.float32))
    np.save(os.path.join(OUT_GRID, "SCE_yearly.npy"), sce05.astype(np.float32))
    np.save(os.path.join(OUT_GRID, "LAT.npy"), LAT05.astype(np.float32))
    np.save(os.path.join(OUT_GRID, "LON.npy"), LON05.astype(np.float32))
    np.save(os.path.join(OUT_GRID, "lat1d.npy"), cru_lat.astype(np.float32))
    np.save(os.path.join(OUT_GRID, "lon1d.npy"), cru_lon.astype(np.float32))
    np.save(os.path.join(OUT_GRID, "YEARS.npy"), YEARS_TARGET)
    np.save(os.path.join(OUT_GRID, "MONTHS.npy"), MONTHS)
    np.save(os.path.join(OUT_GRID, "ML_count.npy"), mlcnt.astype(np.int16))
    np.save(os.path.join(OUT_GRID, "GPP_count.npy"), gppcnt.astype(np.int16))
    np.save(os.path.join(OUT_GRID, "SCE_count.npy"), scecnt.astype(np.int16))
    np.save(os.path.join(OUT_GRID, "base_count.npy"), base_count.astype(np.int32))

    log("Build native warm-yearly ML/GPP ...")
    mln = build_ml_warm_native()
    gppn = build_gpp_warm_native()

    scen = np.full((len(YEARS_TARGET), EXPECTED_TILES, 1200, 1200), np.nan, dtype=np.float32)
    for i, yy in enumerate(sce_native_years):
        if yy < YEARS_TARGET[0] or yy > YEARS_TARGET[-1]:
            continue
        iy = np.where(YEARS_TARGET == yy)[0][0]
        scen[iy] = sce_native[i]

    log("Save region products ...")
    for reg in REGIONS:
        mask = region_mask_native(lat_native, lon_native, reg)
        pix_lat = lat_native[mask].astype(np.float32)
        pix_lon = lon_native[mask].astype(np.float32)
        points = np.column_stack([pix_lat.astype(np.float64), pix_lon.astype(np.float64)])

        save_npz(os.path.join(OUT_REG, f"ML_warm_yearly_{reg['key']}.npz"), years=YEARS_TARGET, lat=pix_lat, lon=pix_lon, data=mln[:, mask].astype(np.float32))
        save_npz(os.path.join(OUT_REG, f"GPP_warm_yearly_{reg['key']}.npz"), years=YEARS_TARGET, lat=pix_lat, lon=pix_lon, data=gppn[:, mask].astype(np.float32))
        save_npz(os.path.join(OUT_REG, f"SCE_yearly_{reg['key']}.npz"), years=YEARS_TARGET, lat=pix_lat, lon=pix_lon, data=scen[:, mask].astype(np.float32))

        iy, ix = subset_cru_source(cru_lat, cru_lon, reg)
        src_lat = cru_lat[iy]; src_lon = cru_lon[ix]
        cru_monthly_reg = np.full((len(YEARS_TARGET), 12, points.shape[0]), np.nan, dtype=np.float32)
        for iyear in range(len(YEARS_TARGET)):
            for imonth in range(12):
                src = cru_cube[iyear, imonth][np.ix_(iy, ix)].astype(np.float32)
                src_fill = fill_nan_nearest_2d(src)
                interp = RegularGridInterpolator((src_lat.astype(np.float64), src_lon.astype(np.float64)), src_fill.astype(np.float64), method="linear", bounds_error=False, fill_value=np.nan)
                cru_monthly_reg[iyear, imonth] = interp(points).astype(np.float32)
        cru_warm_reg = np.nanmean(cru_monthly_reg[:, WARM_MONTH_IDXS, :], axis=1).astype(np.float32)
        save_npz(os.path.join(OUT_REG, f"CRU_warm_yearly_{reg['key']}.npz"), years=YEARS_TARGET, lat=pix_lat, lon=pix_lon, data=cru_warm_reg)

    log("Done.")

if __name__ == "__main__":
    main()

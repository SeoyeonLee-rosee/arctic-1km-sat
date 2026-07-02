#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
data_make.py
-------------------------------------------------------------
Rebuild mean.npy, trend.npy, lat.npy, lon.npy for:

  - ERA5      : t2m.nc
  - CPC       : (tmax.yyyy.nc + tmin.yyyy.nc)/2
  - GHLCN_CAMS: air.mon.mean.nc
  - MERRA2    : only lat.npy lon.npy from a sample file

Period:
  - 2000-03 to 2021-12

Domain:
  - latitude 60N to 90N
  - all longitudes available in file

Outputs per dataset:
  mean.npy  : (12, nlat, nlon) climatological monthly mean (°C)
  trend.npy : (12, nlat, nlon) monthly OLS trend (°C/year)
  lat.npy   : (nlat, nlon) 2D meshgrid
  lon.npy   : (nlat, nlon) 2D meshgrid

Trend computed for each calendar month using yearly means (2000–2021):
  for each grid cell:
    y(year) = monthly_mean_of_that_year
    slope = OLS slope (°C/year)

If input is Kelvin (max > 150), convert to Celsius.
Any extra dims beyond (time, lat, lon) are averaged out automatically.
"""

import os
import glob
import numpy as np
import xarray as xr

# ============================================================
# Config
# ============================================================
OUTROOT = "../data/revise/data"

ERA5_NC   = os.path.join(OUTROOT, "era5", "t2m.nc")
GHLCN_NC  = os.path.join(OUTROOT, "GHLCN_CAMS", "air.mon.mean.nc")
CPC_DIR   = os.path.join(OUTROOT, "cpc")
MERRA_SMP = os.path.join(OUTROOT, "merra", "MERRA2_200.instM_2d_asm_Nx.200001.nc4")

LAT_MIN, LAT_MAX = 60.0, 90.0
START_YEAR, END_YEAR = 2000, 2021
START_DATE = "2000-03-01"
END_DATE   = "2021-12-31"

# ============================================================
# Utilities
# ============================================================
def log(msg):
    print(f"[MAKE_MEAN_TREND] {msg}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def is_kelvin(da: xr.DataArray) -> bool:
    vmax = float(da.max(skipna=True))
    return vmax > 150.0

def to_celsius(da: xr.DataArray) -> xr.DataArray:
    return da - 273.15

def get_lat_lon_names(ds):
    lat_candidates = ["lat", "latitude", "y", "LAT"]
    lon_candidates = ["lon", "longitude", "x", "LON"]
    lat_name = next((n for n in lat_candidates if n in ds.coords or n in ds.variables), None)
    lon_name = next((n for n in lon_candidates if n in ds.coords or n in ds.variables), None)
    if lat_name is None or lon_name is None:
        raise KeyError("Cannot find lat/lon in dataset.")
    return lat_name, lon_name

def get_time_name(ds):
    for cand in ["time", "valid_time", "date"]:
        if cand in ds.coords or cand in ds.variables:
            return cand
    raise KeyError("Cannot find time coordinate in dataset.")

def pick_var(ds, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
        if v in ds.variables and v not in ds.coords:
            return v
    raise KeyError(f"Cannot find variable among: {candidates}")

def subset_arctic_time(ds, da, lat_name, lon_name, time_name):
    # latitude subset
    lat = ds[lat_name]
    if lat[0] > lat[-1]:
        da_sub = da.sel({lat_name: slice(LAT_MAX, LAT_MIN)})
    else:
        da_sub = da.sel({lat_name: slice(LAT_MIN, LAT_MAX)})

    # time subset
    da_sub = da_sub.sel({time_name: slice(START_DATE, END_DATE)})
    return da_sub

def drop_extra_dims(da: xr.DataArray, time_name: str, lat_name: str, lon_name: str, label=""):
    """
    Ensure da dims are exactly (time, lat, lon) by averaging any extra dims.
    """
    keep = {time_name, lat_name, lon_name}
    extra_dims = [d for d in da.dims if d not in keep]
    if len(extra_dims) > 0:
        log(f"   {label} extra dims detected {extra_dims} -> averaging them out")
        da = da.mean(dim=extra_dims, skipna=True)
    return da

def build_latlon_2d_from_1d(lat1d, lon1d):
    lon2d, lat2d = np.meshgrid(lon1d, lat1d, indexing="xy")
    return lat2d.astype(np.float32), lon2d.astype(np.float32)

# ============================================================
# OLS slope per grid (vectorized)
# ============================================================
def ols_slope_per_year(Y: np.ndarray, years: np.ndarray):
    """
    Y: (Ny, Ngrid) with NaNs
    years: (Ny,)
    return slope (Ngrid,) in unit of Y per year
    """
    y = Y.astype(np.float32, copy=False)
    x = years.astype(np.float32)
    x = x - np.nanmean(x)
    X = x[:, None]

    M = np.isfinite(y).astype(np.float32)
    n = M.sum(axis=0)

    xw_mean = (M * X).sum(axis=0) / np.maximum(n, 1)
    yw_mean = np.nansum(y, axis=0) / np.maximum(n, 1)

    dx = X - xw_mean[None, :]
    dy = y - yw_mean[None, :]

    num = np.nansum(M * dx * dy, axis=0)
    den = np.nansum(M * dx * dx, axis=0)

    slope = np.full_like(den, np.nan, dtype=np.float32)
    good = (den > 0) & (n >= 2)
    slope[good] = num[good] / den[good]
    return slope

# ============================================================
# Core: monthly climatology mean + monthly trend
# ============================================================
def compute_monthly_mean_and_trend(da: xr.DataArray, time_name: str):
    """
    da: (time, lat, lon) in °C
    returns:
      mean12:  (12, lat, lon) monthly climatology mean
      trend12: (12, lat, lon) monthly OLS trend per year
    """
    years = np.arange(START_YEAR, END_YEAR + 1, dtype=np.float32)
    nY = years.size

    latN = da.sizes[da.dims[-2]]
    lonN = da.sizes[da.dims[-1]]

    mean12  = np.full((12, latN, lonN), np.nan, dtype=np.float32)
    trend12 = np.full((12, latN, lonN), np.nan, dtype=np.float32)

    for m in range(1, 13):
        da_m = da.where(da[time_name].dt.month == m, drop=True)
        if da_m.sizes[time_name] == 0:
            log(f"   month {m:02d}: no data -> skip")
            continue

        # climatological mean over all years
        mean_m = da_m.mean(time_name, skipna=True).values.astype(np.float32)
        mean12[m-1] = mean_m

        # yearly means stack for trend
        yearly_stack = []
        for y in years:
            da_y = da_m.where(da_m[time_name].dt.year == int(y), drop=True)
            if da_y.sizes[time_name] == 0:
                yearly_stack.append(np.full((latN, lonN), np.nan, dtype=np.float32))
            else:
                yearly_stack.append(da_y.mean(time_name, skipna=True).values.astype(np.float32))
        yearly_stack = np.stack(yearly_stack, axis=0)  # (Ny, lat, lon)

        Y2 = yearly_stack.reshape(nY, -1)
        slope = ols_slope_per_year(Y2, years)
        trend12[m-1] = slope.reshape(latN, lonN).astype(np.float32)

        log(f"   month {m:02d} mean finite={np.count_nonzero(np.isfinite(mean12[m-1]))} "
            f"trend finite={np.count_nonzero(np.isfinite(trend12[m-1]))}")

    return mean12, trend12

# ============================================================
# ERA5
# ============================================================
def process_era5():
    outdir = os.path.join(OUTROOT, "era5")
    ensure_dir(outdir)
    log(f"[ERA5] open {ERA5_NC}")

    ds = xr.open_dataset(ERA5_NC)
    time_name = get_time_name(ds)
    lat_name, lon_name = get_lat_lon_names(ds)
    vname = pick_var(ds, ["t2m", "2t", "tas"])

    da = ds[vname]
    da = subset_arctic_time(ds, da, lat_name, lon_name, time_name)
    da = drop_extra_dims(da, time_name, lat_name, lon_name, label="ERA5")

    if is_kelvin(da):
        log("   ERA5 detected Kelvin -> convert to Celsius")
        da = to_celsius(da)

    lat1d = da[lat_name].values.astype(np.float32)
    lon1d = da[lon_name].values.astype(np.float32)
    lat2d, lon2d = build_latlon_2d_from_1d(lat1d, lon1d)

    np.save(os.path.join(outdir, "lat.npy"), lat2d)
    np.save(os.path.join(outdir, "lon.npy"), lon2d)

    log("   compute monthly mean/trend ...")
    mean12, trend12 = compute_monthly_mean_and_trend(da, time_name)

    np.save(os.path.join(outdir, "mean.npy"), mean12)
    np.save(os.path.join(outdir, "trend.npy"), trend12)

    ds.close()
    log(f"[ERA5] saved: mean.npy trend.npy lat.npy lon.npy")

# ============================================================
# GHLCN_CAMS
# ============================================================
def process_ghlcn():
    outdir = os.path.join(OUTROOT, "GHLCN_CAMS")
    ensure_dir(outdir)
    log(f"[GHLCN_CAMS] open {GHLCN_NC}")

    ds = xr.open_dataset(GHLCN_NC)
    time_name = get_time_name(ds)
    lat_name, lon_name = get_lat_lon_names(ds)
    vname = pick_var(ds, ["air", "tas", "t2m", "temperature"])

    da = ds[vname]
    da = subset_arctic_time(ds, da, lat_name, lon_name, time_name)
    da = drop_extra_dims(da, time_name, lat_name, lon_name, label="GHLCN_CAMS")

    if is_kelvin(da):
        log("   GHLCN_CAMS detected Kelvin -> convert to Celsius")
        da = to_celsius(da)

    lat1d = da[lat_name].values.astype(np.float32)
    lon1d = da[lon_name].values.astype(np.float32)
    lat2d, lon2d = build_latlon_2d_from_1d(lat1d, lon1d)

    np.save(os.path.join(outdir, "lat.npy"), lat2d)
    np.save(os.path.join(outdir, "lon.npy"), lon2d)

    log("   compute monthly mean/trend ...")
    mean12, trend12 = compute_monthly_mean_and_trend(da, time_name)

    np.save(os.path.join(outdir, "mean.npy"), mean12)
    np.save(os.path.join(outdir, "trend.npy"), trend12)

    ds.close()
    log(f"[GHLCN_CAMS] saved: mean.npy trend.npy lat.npy lon.npy")

# ============================================================
# CPC: (tmax + tmin)/2
# ============================================================
def process_cpc():
    outdir = os.path.join(OUTROOT, "cpc")
    ensure_dir(outdir)

    tmax_files = sorted(glob.glob(os.path.join(CPC_DIR, "tmax.*.nc")))
    tmin_files = sorted(glob.glob(os.path.join(CPC_DIR, "tmin.*.nc")))

    if len(tmax_files) == 0 or len(tmin_files) == 0:
        raise FileNotFoundError("CPC requires tmax.yyyy.nc and tmin.yyyy.nc files.")

    def year_from_name(fp):
        base = os.path.basename(fp)
        # tmax.2000.nc -> 2000
        return int(base.split(".")[1])

    tmax_files = [f for f in tmax_files if START_YEAR <= year_from_name(f) <= END_YEAR]
    tmin_files = [f for f in tmin_files if START_YEAR <= year_from_name(f) <= END_YEAR]

    if len(tmax_files) != len(tmin_files):
        raise RuntimeError("CPC tmax and tmin file counts differ after filtering years.")

    log(f"[CPC] found {len(tmax_files)} yearly pairs")

    da_list = []
    lat2d = lon2d = None
    time_name = lat_name = lon_name = None

    for fmax, fmin in zip(tmax_files, tmin_files):
        y = year_from_name(fmax)
        log(f"   open {os.path.basename(fmax)} + {os.path.basename(fmin)}")

        dsmax = xr.open_dataset(fmax)
        dsmin = xr.open_dataset(fmin)

        time_name = get_time_name(dsmax)
        lat_name, lon_name = get_lat_lon_names(dsmax)

        vmax = pick_var(dsmax, ["tmax", "tasmax", "air", "tmp", "temperature"])
        vmin = pick_var(dsmin, ["tmin", "tasmin", "air", "tmp", "temperature"])

        da_max = dsmax[vmax]
        da_min = dsmin[vmin]

        da_max = subset_arctic_time(dsmax, da_max, lat_name, lon_name, time_name)
        da_min = subset_arctic_time(dsmin, da_min, lat_name, lon_name, time_name)

        da = (da_max + da_min) / 2.0
        da = drop_extra_dims(da, time_name, lat_name, lon_name, label=f"CPC({y})")

        if is_kelvin(da):
            log("     CPC detected Kelvin -> convert to Celsius")
            da = to_celsius(da)

        da_list.append(da)

        if lat2d is None:
            lat1d = da[lat_name].values.astype(np.float32)
            lon1d = da[lon_name].values.astype(np.float32)
            lat2d, lon2d = build_latlon_2d_from_1d(lat1d, lon1d)

        dsmax.close()
        dsmin.close()

    da_all = xr.concat(da_list, dim=time_name)

    np.save(os.path.join(outdir, "lat.npy"), lat2d)
    np.save(os.path.join(outdir, "lon.npy"), lon2d)

    log("   compute monthly mean/trend ...")
    mean12, trend12 = compute_monthly_mean_and_trend(da_all, time_name)

    np.save(os.path.join(outdir, "mean.npy"), mean12)
    np.save(os.path.join(outdir, "trend.npy"), trend12)

    log(f"[CPC] saved: mean.npy trend.npy lat.npy lon.npy")

# ============================================================
# MERRA2: lat/lon only (60–90N)
# ============================================================
def process_merra2_latlon_only():
    outdir = os.path.join(OUTROOT, "merra")
    ensure_dir(outdir)
    log(f"[MERRA2] lat/lon only from {MERRA_SMP}")

    if not os.path.exists(MERRA_SMP):
        raise FileNotFoundError(MERRA_SMP)

    ds = xr.open_dataset(MERRA_SMP)
    lat_name, lon_name = get_lat_lon_names(ds)

    lat1d = ds[lat_name].values.astype(np.float32)
    lon1d = ds[lon_name].values.astype(np.float32)

    # subset lat 60-90
    if lat1d[0] > lat1d[-1]:
        lat_use = lat1d[(lat1d <= LAT_MAX) & (lat1d >= LAT_MIN)]
    else:
        lat_use = lat1d[(lat1d >= LAT_MIN) & (lat1d <= LAT_MAX)]

    lon_use = lon1d
    lat2d, lon2d = build_latlon_2d_from_1d(lat_use, lon_use)

    np.save(os.path.join(outdir, "lat.npy"), lat2d)
    np.save(os.path.join(outdir, "lon.npy"), lon2d)

    ds.close()
    log("[MERRA2] saved: lat.npy lon.npy")

# ============================================================
# main
# ============================================================
def main():
    log("=== START rebuilding mean/trend/lat/lon ===")
    process_era5()
    process_cpc()
    process_ghlcn()
    process_merra2_latlon_only()
    log("=== DONE ===")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Arctic windows — ML(Scatter) + ERA/CRU(Contour) seasonal AVE & TREND maps

Per window, outputs (각 데이터셋별로 동일 항목 생성):
  1) AVE_warm_minus_cold (Apr–Oct minus Nov–Mar)   -> sequential cmap, 5–95%
  2) AVE_cold            (Nov–Mar)                  -> sequential cmap, 2–98%
  3) AVE_warm            (Apr–Oct)                  -> sequential cmap, 2–98%
  4) TREND_cold          (mean monthly slopes, °C/decade)  -> diverging (0-centered), ±max(|p2|,|p98|)
  5) TREND_warm          (mean monthly slopes, °C/decade)  -> diverging (0-centered), ±max(|p2|,|p98|)

• ML  : ../data/MODIS/{mm}/ave_gr.npy (level=5), trend_gr.npy (level=5)
        lat.npy, lon.npy (same shape) – ML은 scatter(1.6pt)로 그림
• ERA : ../data/ERA5_t2m_mon_1940-2025.nc  (t2m[K] → °C), contourf로 그림
• CRU : ../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc (tmp[°C]), contourf로 그림
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from netCDF4 import Dataset

# ------------ User paths ------------
ml_dir     = "../data/MODIS/"
era_nc     = "../data/ERA5_t2m_mon_1940-2025.nc"
cru_nc     = "../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc"
etopo_dir  = "/data1/DATA_ARCHIVE/ETOPO2022/"
outdir     = "../FIG/ML_WINDOWS/"
os.makedirs(outdir, exist_ok=True)

# ------------ Period ------------
TIME_START = "2000-03-01"
TIME_END   = "2022-04-30"

# ------------ Seasons ------------
WARM = [4, 5, 6, 7, 8, 9, 10]     # Apr–Oct
COLD = [11, 12, 1, 2, 3]          # Nov–Mar

# ------------ Windows ------------
windows = [
    {"key": "E130_135_N65_70", "name": "130–135°E\n65–70°N", "lon": (130, 135), "lat": (65, 70)},
    {"key": "E58_62_N63_67",   "name": " 58– 62°E\n63–67°N", "lon": (58, 62),   "lat": (63, 67)},
]

# ------------ Altitude (optional background; fixed 0–1400 m) ------------
alt_norm  = Normalize(vmin=0, vmax=1400)
alt_cmap  = plt.get_cmap("viridis")
alt_cmap.set_bad(color="white")

def _lat_str(d): return f"N{abs(int(d))//15*15:02d}" if d >= 0 else f"S{abs(int(d))//15*15:02d}"
def _lon_str(d): return f"E{abs(int(d))//15*15:03d}" if d >= 0 else f"W{abs(int(d))//15*15:03d}"

def get_etopo_tile_names(lat_range, lon_range):
    lat_vals = range(int(lat_range[0])//15*15, int(lat_range[1])+15, 15)
    lon_vals = range(int(lon_range[0])//15*15, int(lon_range[1])+15, 15)
    return [f"ETOPO_2022_v1_15s_{_lat_str(la)}{_lon_str(lo)}_surface.nc"
            for la in lat_vals for lo in lon_vals]

def load_merged_etopo(tile_dir, names):
    lat_all, lon_all, z_all = [], [], []
    for nm in names:
        p = os.path.join(tile_dir, nm)
        if not os.path.exists(p):
            print(f"❗ ETOPO tile missing: {p}")
            continue
        with Dataset(p) as nc:
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
            z   = nc.variables['z'][:].astype(np.float32)
            z[z <= 0] = np.nan  # mask sea
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            lat_all.append(LA.ravel()); lon_all.append(LO.ravel()); z_all.append(z.ravel())
    if not lat_all:
        return None, None, None
    return np.concatenate(lat_all), np.concatenate(lon_all), np.concatenate(z_all)

# ------------ ML loaders ------------
LEVEL_IDX = 5  # use the 6th level (index 5)

def _extract_level(arr, level_idx=LEVEL_IDX):
    if arr.ndim == 4:  # (tile, level, y, x)
        return arr[:, level_idx, :, :]
    if arr.ndim == 3:
        if arr.shape[0] > 20:  # (tile,y,x)
            return arr
        return arr[level_idx, :, :]   # (y,x)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected arr shape {arr.shape}")

print("• Loading ML lat/lon ...")
lat_ml = np.load(os.path.join(ml_dir, "lat.npy"))
lon_ml = np.load(os.path.join(ml_dir, "lon.npy"))
if lat_ml.shape != lon_ml.shape:
    raise ValueError("lat.npy and lon.npy shapes differ")
grid_shape = lat_ml.shape

def load_monthly_field(ml_root, month, fname):
    fp = os.path.join(ml_root, f"{month:02d}", fname)
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    arr = np.load(fp)
    out = _extract_level(arr, LEVEL_IDX)
    if out.shape != grid_shape:
        raise ValueError(f"{fname} {month:02d} shape {out.shape} != {grid_shape}")
    return out.astype(np.float32)

def _nanmean_stack(lst):
    if len(lst) == 0:
        return np.full(grid_shape, np.nan, dtype=np.float32)
    return np.nanmean(np.stack(lst, axis=0), axis=0).astype(np.float32)

# ------------ ML seasonal composites ------------
print("• Building ML seasonal composites (AVE & TREND) ...")
ave_warm_list, ave_cold_list = [], []
trend_warm_list, trend_cold_list = [], []
for m in range(1, 13):
    try:  ave_m   = load_monthly_field(ml_dir, m, "ave_gr.npy")
    except FileNotFoundError: ave_m = None
    try:  slope_m = load_monthly_field(ml_dir, m, "trend_gr.npy")   # °C/year
    except FileNotFoundError: slope_m = None
    if m in WARM:
        if ave_m   is not None: ave_warm_list.append(ave_m)
        if slope_m is not None: trend_warm_list.append(slope_m)
    if m in COLD:
        if ave_m   is not None: ave_cold_list.append(ave_m)
        if slope_m is not None: trend_cold_list.append(slope_m)

ML_AVE_warm  = _nanmean_stack(ave_warm_list)
ML_AVE_cold  = _nanmean_stack(ave_cold_list)
ML_AVE_diff  = (ML_AVE_warm - ML_AVE_cold).astype(np.float32)
ML_TR_warm   = (10.0 * _nanmean_stack(trend_warm_list)).astype(np.float32)  # °C/decade
ML_TR_cold   = (10.0 * _nanmean_stack(trend_cold_list)).astype(np.float32)

# ------------ ERA/CRU helpers ------------
def to_years(tcoord):
    t = xr.DataArray(tcoord).dt
    return (t.year + (t.dayofyear - 1)/365.25).values.astype(float)

def seasonal_means_and_trends(da, warm_months=WARM, cold_months=COLD):
    """da: (time, lat, lon) in °C. Return AVE_warm, AVE_cold, AVE_diff, TR_warm_dec, TR_cold_dec."""
    da = da.sortby('time')
    years = to_years(da['time'])

    def month_mean(m):
        sub = da.sel(time=da['time'].dt.month == m)
        return sub.mean('time', skipna=True)

    def month_trend(m):
        sub = da.sel(time=da['time'].dt.month == m)
        if sub.sizes.get('time', 0) < 3:
            return xr.full_like(sub.isel(time=0), np.nan)
        y = sub.values  # (T, lat, lon)
        t = years[da['time'].dt.month.values == m]  # (T,)
        # OLS slope along time, ignore NaNs
        t0 = t - t.mean()
        Sxx = np.nansum(t0**2)
        if not np.isfinite(Sxx) or Sxx <= 0:
            return xr.full_like(sub.isel(time=0), np.nan)
        # mean along time (nanmean)
        ymean = np.nanmean(y, axis=0)
        num = np.nansum((t0[:, None, None]) * (y - ymean[None, :, :]), axis=0)
        slope = num / Sxx  # °C/year
        out = xr.DataArray(slope, coords=sub.isel(time=0).drop_vars('time').coords, dims=sub.isel(time=0).dims)
        return out

    # seasonal AVE: avg of "monthly means"
    warm_means = [month_mean(m) for m in warm_months]
    cold_means = [month_mean(m) for m in cold_months]
    AVE_warm = xr.concat(warm_means, 'm').mean('m', skipna=True)
    AVE_cold = xr.concat(cold_means, 'm').mean('m', skipna=True)
    AVE_diff = AVE_warm - AVE_cold

    # seasonal TREND: avg of "monthly slopes"
    warm_trs = [month_trend(m) for m in warm_months]
    cold_trs = [month_trend(m) for m in cold_months]
    TR_warm_dec = xr.concat(warm_trs, 'm').mean('m', skipna=True) * 10.0
    TR_cold_dec = xr.concat(cold_trs, 'm').mean('m', skipna=True) * 10.0

    return AVE_warm, AVE_cold, AVE_diff, TR_warm_dec, TR_cold_dec

print("• Loading ERA ...")
with xr.open_dataset(era_nc) as ds:
    tname = 'valid_time' if 'valid_time' in ds.variables else 'time'
    latn  = 'latitude'   if 'latitude'   in ds.coords    else 'lat'
    lonn  = 'longitude'  if 'longitude'  in ds.coords    else 'lon'
    da = (ds['t2m'] - 273.15).sel({tname: slice(TIME_START, TIME_END)})
    da = da.rename({tname: 'time', latn: 'lat', lonn: 'lon'})
    # convert lon to -180..180 if needed
    if float(da['lon'].max()) > 180:
        da = da.assign_coords(lon=((da['lon'] + 180) % 360) - 180).sortby('lon')
    ERA = da  # (time, lat, lon) in °C

print("• Loading CRU ...")
with xr.open_dataset(cru_nc) as ds:
    var_c = [v for v in ds.data_vars if v not in ('lat','lon')][0]  # e.g., tmp
    da = ds[var_c].sel(time=slice(TIME_START, TIME_END))
    da = da.rename({'lat':'lat','lon':'lon'})
    if float(da['lon'].max()) > 180:
        da = da.assign_coords(lon=((da['lon'] + 180) % 360) - 180).sortby('lon')
    CRU = da  # (time, lat, lon) in °C

# ------------ Norm helpers (window-specific) ------------
def get_window_mask(lat_grid, lon_grid, lon_min, lon_max, lat_min, lat_max):
    la = lat_grid
    lo = lon_grid
    return np.isfinite(la) & np.isfinite(lo) & \
           (la >= lat_min) & (la <= lat_max) & (lo >= lon_min) & (lo <= lon_max)

def seq_norm_window_np(arr, mask, p_lo=2, p_hi=98, min_span=1e-6):
    vv = arr[mask]; vv = vv[np.isfinite(vv)]
    if vv.size < 10: return None
    vmin, vmax = np.percentile(vv, [p_lo, p_hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax: return None
    if (vmax - vmin) < min_span:
        pad = max(abs(vmin), 1.0) * 0.05
        vmin, vmax = vmin - pad, vmax + pad
    return Normalize(vmin=float(vmin), vmax=float(vmax))

def div0_norm_window_np(arr, mask, p_lo=2, p_hi=98, min_span=1e-6):
    vv = arr[mask]; vv = vv[np.isfinite(vv)]
    if vv.size < 10: return None
    lo, hi = np.percentile(vv, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi): return None
    m = max(abs(lo), abs(hi))
    if m < min_span: m = 1.0
    return Normalize(vmin=-float(m), vmax=float(m))

def seq_norm_window_xr(da2d, lon_min, lon_max, lat_min, lat_max, p_lo=2, p_hi=98):
    sub = da2d.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    vv = sub.values
    vv = vv[np.isfinite(vv)]
    if vv.size < 10: return None
    vmin, vmax = np.percentile(vv, [p_lo, p_hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax: return None
    return Normalize(vmin=float(vmin), vmax=float(vmax))

def div0_norm_window_xr(da2d, lon_min, lon_max, lat_min, lat_max, p_lo=2, p_hi=98):
    sub = da2d.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    vv = sub.values
    vv = vv[np.isfinite(vv)]
    if vv.size < 10: return None
    lo, hi = np.percentile(vv, [p_lo, p_hi])
    m = max(abs(lo), abs(hi))
    if m < 1e-6: m = 1.0
    return Normalize(vmin=-float(m), vmax=float(m))

# ------------ Plot helpers (with 1-decimal colorbar) ------------
def style_geo_axes(ax, lon_min, lon_max, lat_min, lat_max):
    ax.set_extent([lon_min+0.01, lon_max-0.01, lat_min+0.01, lat_max-0.01], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2)
    ax.set_xticks(np.arange(lon_min, lon_max+0.1, 2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(lat_min, lat_max+0.1, 2), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(labelsize=8, direction='out')

def plot_window_scatter(lon_min, lon_max, lat_min, lat_max,
                        lon_grid, lat_grid, value_grid,
                        title, out_png, cmap="plasma", norm=None, add_etopo=False):
    fig = plt.figure(figsize=(6.2, 4.2))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    style_geo_axes(ax, lon_min, lon_max, lat_min, lat_max)

    # optional altitude
    if add_etopo:
        tiles = get_etopo_tile_names((lat_min, lat_max), (lon_min, lon_max))
        la_bg, lo_bg, zz_bg = load_merged_etopo(etopo_dir, tiles)
        if la_bg is not None:
            m_bg = (la_bg >= lat_min) & (la_bg <= lat_max) & (lo_bg >= lon_min) & (lo_bg <= lon_max)
            ax.scatter(lo_bg[m_bg], la_bg[m_bg], c=zz_bg[m_bg], s=1, cmap=alt_cmap, norm=alt_norm,
                       alpha=0.35, transform=ccrs.PlateCarree())

    la = lat_grid.ravel(); lo = lon_grid.ravel(); vv = value_grid.ravel()
    m = np.isfinite(la) & np.isfinite(lo) & np.isfinite(vv) & \
        (la >= lat_min) & (la <= lat_max) & (lo >= lon_min) & (lo <= lon_max)
    sc = ax.scatter(lo[m], la[m], c=vv[m], s=1.6, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # ← 1 decimal

    ax.set_title(title, fontsize=10, pad=6)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved:", out_png)

def plot_window_contour(da2d, lon_min, lon_max, lat_min, lat_max, title, out_png,
                        cmap="plasma", norm=None, levels=12):
    """da2d: xarray DataArray (lat, lon)"""
    sub = da2d.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    LON, LAT = np.meshgrid(sub['lon'].values, sub['lat'].values)

    if norm is None:
        vmin = float(np.nanpercentile(sub.values, 2))
        vmax = float(np.nanpercentile(sub.values, 98))
        norm = Normalize(vmin=vmin, vmax=vmax)

    if isinstance(levels, int):
        levs = np.linspace(norm.vmin, norm.vmax, levels)
    else:
        levs = levels

    fig = plt.figure(figsize=(6.2, 4.2))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    style_geo_axes(ax, lon_min, lon_max, lat_min, lat_max)

    cs = ax.contourf(LON, LAT, sub.values, levels=levs, cmap=cmap, norm=norm, extend='both', transform=ccrs.PlateCarree())

    cbar = plt.colorbar(cs, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # ← 1 decimal

    ax.set_title(title, fontsize=10, pad=6)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved:", out_png)

# ------------ Make plots ------------
for w in windows:
    lon_min, lon_max = w["lon"]; lat_min, lat_max = w["lat"]
    key  = w["key"];            name = w["name"].replace("\n", " – ")

    # ===== ML norms (window-specific) =====
    mask_win = get_window_mask(lat_ml, lon_ml, lon_min, lon_max, lat_min, lat_max)
    norm_diff_seq = seq_norm_window_np(ML_AVE_diff, mask_win, p_lo=5, p_hi=95)
    norm_ave_cold = seq_norm_window_np(ML_AVE_cold, mask_win, p_lo=2, p_hi=98)
    norm_ave_warm = seq_norm_window_np(ML_AVE_warm, mask_win, p_lo=2, p_hi=98)
    norm_tr_cold  = div0_norm_window_np(ML_TR_cold, mask_win, p_lo=2, p_hi=98)
    norm_tr_warm  = div0_norm_window_np(ML_TR_warm, mask_win, p_lo=2, p_hi=98)

    # ===== ERA seasonal fields (subset to window first for speed) =====
    ERA_sub = ERA.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    ERA_AVE_warm, ERA_AVE_cold, ERA_AVE_diff, ERA_TR_warm, ERA_TR_cold = seasonal_means_and_trends(ERA_sub)

    # ERA norms (창 내부)
    norm_e_diff = seq_norm_window_xr(ERA_AVE_diff, lon_min, lon_max, lat_min, lat_max, p_lo=5, p_hi=95)
    norm_e_cold = seq_norm_window_xr(ERA_AVE_cold, lon_min, lon_max, lat_min, lat_max, p_lo=2, p_hi=98)
    norm_e_warm = seq_norm_window_xr(ERA_AVE_warm, lon_min, lon_max, lat_min, lat_max, p_lo=2, p_hi=98)
    norm_e_tr_c = div0_norm_window_xr(ERA_TR_cold, lon_min, lon_max, lat_min, lat_max, p_lo=2, p_hi=98)
    norm_e_tr_w = div0_norm_window_xr(ERA_TR_warm, lon_min, lon_max, lat_min, lat_max, p_lo=2, p_hi=98)

    # ===== CRU seasonal fields =====
    CRU_sub = CRU.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    CRU_AVE_warm, CRU_AVE_cold, CRU_AVE_diff, CRU_TR_warm, CRU_TR_cold = seasonal_means_and_trends(CRU_sub)

    # CRU norms (창 내부)
    norm_c_diff = seq_norm_window_xr(CRU_AVE_diff, lon_min, lon_max, lat_min, lat_max, p_lo=5, p_hi=95)
    norm_c_cold = seq_norm_window_xr(CRU_AVE_cold, lon_min, lon_max, lat_min, lat_max, p_lo=2, p_hi=98)
    norm_c_warm = seq_norm_window_xr(CRU_AVE_warm, lon_min, lon_max, lat_min, lat_max, p_lo=2, p_hi=98)
    norm_c_tr_c = div0_norm_window_xr(CRU_TR_cold, lon_min, lon_max, lat_min, lat_max, p_lo=2, p_hi=98)
    norm_c_tr_w = div0_norm_window_xr(CRU_TR_warm, lon_min, lon_max, lat_min, lat_max, p_lo=2, p_hi=98)

    # ===== ML (scatter) =====
    plot_window_scatter(lon_min, lon_max, lat_min, lat_max,
                        lon_ml, lat_ml, ML_AVE_diff,
                        title=f"[ML] AVE (warm − cold) — {name}",
                        out_png=os.path.join(outdir, f"{key}_ML_AVE_warm_minus_cold.png"),
                        cmap="plasma", norm=norm_diff_seq, add_etopo=False)

    plot_window_scatter(lon_min, lon_max, lat_min, lat_max,
                        lon_ml, lat_ml, ML_AVE_cold,
                        title=f"[ML] AVE (cold) — {name}",
                        out_png=os.path.join(outdir, f"{key}_ML_AVE_cold.png"),
                        cmap="plasma", norm=norm_ave_cold, add_etopo=False)

    plot_window_scatter(lon_min, lon_max, lat_min, lat_max,
                        lon_ml, lat_ml, ML_AVE_warm,
                        title=f"[ML] AVE (warm) — {name}",
                        out_png=os.path.join(outdir, f"{key}_ML_AVE_warm.png"),
                        cmap="plasma", norm=norm_ave_warm, add_etopo=False)

    plot_window_scatter(lon_min, lon_max, lat_min, lat_max,
                        lon_ml, lat_ml, ML_TR_cold,
                        title=f"[ML] TREND (cold, °C/decade) — {name}",
                        out_png=os.path.join(outdir, f"{key}_ML_TREND_cold_decade.png"),
                        cmap="coolwarm", norm=norm_tr_cold, add_etopo=False)

    plot_window_scatter(lon_min, lon_max, lat_min, lat_max,
                        lon_ml, lat_ml, ML_TR_warm,
                        title=f"[ML] TREND (warm, °C/decade) — {name}",
                        out_png=os.path.join(outdir, f"{key}_ML_TREND_warm_decade.png"),
                        cmap="coolwarm", norm=norm_tr_warm, add_etopo=False)

    # ===== ERA (contour) =====
    plot_window_contour(ERA_AVE_diff, lon_min, lon_max, lat_min, lat_max,
                        title=f"[ERA5] AVE (warm − cold) — {name}",
                        out_png=os.path.join(outdir, f"{key}_ERA_AVE_warm_minus_cold.png"),
                        cmap="plasma", norm=norm_e_diff, levels=12)

    plot_window_contour(ERA_AVE_cold, lon_min, lon_max, lat_min, lat_max,
                        title=f"[ERA5] AVE (cold) — {name}",
                        out_png=os.path.join(outdir, f"{key}_ERA_AVE_cold.png"),
                        cmap="plasma", norm=norm_e_cold, levels=12)

    plot_window_contour(ERA_AVE_warm, lon_min, lon_max, lat_min, lat_max,
                        title=f"[ERA5] AVE (warm) — {name}",
                        out_png=os.path.join(outdir, f"{key}_ERA_AVE_warm.png"),
                        cmap="plasma", norm=norm_e_warm, levels=12)

    plot_window_contour(ERA_TR_cold, lon_min, lon_max, lat_min, lat_max,
                        title=f"[ERA5] TREND (cold, °C/decade) — {name}",
                        out_png=os.path.join(outdir, f"{key}_ERA_TREND_cold_decade.png"),
                        cmap="coolwarm", norm=norm_e_tr_c, levels=12)

    plot_window_contour(ERA_TR_warm, lon_min, lon_max, lat_min, lat_max,
                        title=f"[ERA5] TREND (warm, °C/decade) — {name}",
                        out_png=os.path.join(outdir, f"{key}_ERA_TREND_warm_decade.png"),
                        cmap="coolwarm", norm=norm_e_tr_w, levels=12)

    # ===== CRU (contour) =====
    plot_window_contour(CRU_AVE_diff, lon_min, lon_max, lat_min, lat_max,
                        title=f"[CRU] AVE (warm − cold) — {name}",
                        out_png=os.path.join(outdir, f"{key}_CRU_AVE_warm_minus_cold.png"),
                        cmap="plasma", norm=norm_c_diff, levels=12)

    plot_window_contour(CRU_AVE_cold, lon_min, lon_max, lat_min, lat_max,
                        title=f"[CRU] AVE (cold) — {name}",
                        out_png=os.path.join(outdir, f"{key}_CRU_AVE_cold.png"),
                        cmap="plasma", norm=norm_c_cold, levels=12)

    plot_window_contour(CRU_AVE_warm, lon_min, lon_max, lat_min, lat_max,
                        title=f"[CRU] AVE (warm) — {name}",
                        out_png=os.path.join(outdir, f"{key}_CRU_AVE_warm.png"),
                        cmap="plasma", norm=norm_c_warm, levels=12)

    plot_window_contour(CRU_TR_cold, lon_min, lon_max, lat_min, lat_max,
                        title=f"[CRU] TREND (cold, °C/decade) — {name}",
                        out_png=os.path.join(outdir, f"{key}_CRU_TREND_cold_decade.png"),
                        cmap="coolwarm", norm=norm_c_tr_c, levels=12)

    plot_window_contour(CRU_TR_warm, lon_min, lon_max, lat_min, lat_max,
                        title=f"[CRU] TREND (warm, °C/decade) — {name}",
                        out_png=os.path.join(outdir, f"{key}_CRU_TREND_warm_decade.png"),
                        cmap="coolwarm", norm=norm_c_tr_w, levels=12)


# Close datasets
ds_e.close()
ds_c.close()

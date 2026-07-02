#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Monthly climatology (mean only, 2000-03 to 2022-04) at 65°N / 67°N
Overlay many longitudes (Δ=0.01°):
  - ML  : line color by altitude (ETOPO), value from ave_gr.npy (level 5),
          spatial mean within ±0.005° box
  - ERA : gray solid (nearest grid) monthly means across years
  - CRU : gray dashed (nearest grid) monthly means across years
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

# ==============================
# User switches
# ==============================
REGION_KEY = 'U'        # 'U' or 'V'
TIME_START = '2000-03-01'
TIME_END   = '2022-04-30'

# Paths
ml_data_path = '../data/MODIS/'
cru_nc_path  = '../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc'
era_nc_path  = '../data/ERA5_t2m_mon_1940-2025.nc'
etopo_dir    = '/data1/DATA_ARCHIVE/ETOPO2022/'
fig_dir      = '../FIG/timeseries/'
os.makedirs(fig_dir, exist_ok=True)

# Plot styling
COL_ERA = '0.25'   # dark gray
COL_CRU = '0.45'   # mid gray
ERA_LW = 1.0
CRU_LW = 1.0
ML_LW  = 1.1
ALPHA  = 0.75
CMAP   = get_cmap('viridis')
ALT_NORM = Normalize(vmin=0, vmax=1400)  # altitude color scale

MON_LABS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
MON_NUMS = np.arange(1,13)

# ==============================
# Region config
# ==============================
if REGION_KEY.upper() == 'U':
    LAT_FIXED = 65.0
    LON_START, LON_END, LON_STEP = 59.75, 60.25, 0.01
    title_prefix = 'Ural 65°N'
    out_png = os.path.join(fig_dir, 'Ural_65N_59.75-60.25E_monthly_mean.png')
    ylab = 'Temperature (°C)'
elif REGION_KEY.upper() == 'V':
    LAT_FIXED = 67.0
    LON_START, LON_END, LON_STEP = 132.75, 133.25, 0.01
    title_prefix = 'Verkhoyansk 67°N'
    out_png = os.path.join(fig_dir, 'Verkhoyansk_67N_132.75-133.25E_monthly_mean.png')
    ylab = 'Temperature (°C)'
else:
    raise ValueError("REGION_KEY must be 'U' or 'V'")

LONS = np.round(np.arange(LON_START, LON_END + 0.0001, LON_STEP), 4)

# ==============================
# Helpers
# ==============================
LEVEL_IDX = 5  # ML level index to use (6th)

def nearest_index(coord_vals, target):
    coord_vals = np.asarray(coord_vals)
    return int(np.nanargmin(np.abs(coord_vals - target)))

def lon_to_0360_if_needed(lon_vals, query_lon):
    if np.nanmax(lon_vals) > 180:
        return query_lon % 360
    return query_lon

def _extract_level(arr, level_idx):
    """
    Return an array with the SAME spatial shape as lat/lon (tile,y,x) or (y,x),
    selecting level_idx when present.
    """
    if arr.ndim == 4:  # (tile, level, y, x)
        if level_idx >= arr.shape[1]:
            raise ValueError(f"LEVEL_IDX {level_idx} out of range for {arr.shape}")
        return arr[:, level_idx, :, :]  # (tile,y,x)
    elif arr.ndim == 3:
        # Heuristic: if first dim small (<=12), treat as (level,y,x); else (tile,y,x)
        if arr.shape[0] <= 12:
            if level_idx >= arr.shape[0]:
                raise ValueError(f"LEVEL_IDX {level_idx} out of range for {arr.shape}")
            return arr[level_idx, :, :]  # (y,x)
        else:
            return arr  # (tile,y,x)
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"Unexpected array shape: {arr.shape}")

# ---------- ETOPO helpers ----------
def _lat_str(d): return f"N{abs(int(d))//15*15:02d}" if d >= 0 else f"S{abs(int(d))//15*15:02d}"
def _lon_str(d): return f"E{abs(int(d))//15*15:03d}" if d >= 0 else f"W{abs(int(d))//15*15:03d}"

def get_etopo_tile_names(lat_range, lon_range):
    lat_vals = range(int(lat_range[0])//15*15, int(lat_range[1])+15, 15)
    lon_vals = range(int(lon_range[0])//15*15, int(lon_range[1])+15, 15)
    return [f"ETOPO_2022_v1_15s_{_lat_str(la)}{_lon_str(lo)}_surface.nc"
            for la in lat_vals for lo in lon_vals]

def load_etopo_tiles(tile_dir, names):
    lat_all, lon_all, z_all = [], [], []
    for nm in names:
        p = os.path.join(tile_dir, nm)
        if not os.path.exists(p):
            print(f"❗ Missing ETOPO tile: {p}")
            continue
        with Dataset(p) as nc:
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
            z   = nc.variables['z'][:].astype(np.float32)
            z[z <= 0] = np.nan  # mask sea
            lat_all.append(lat); lon_all.append(lon); z_all.append(z)
    return lat_all, lon_all, z_all

def build_kdtree(lat_arrs, lon_arrs, z_arrs):
    pts, vals = [], []
    for la, lo, zz in zip(lat_arrs, lon_arrs, z_arrs):
        LA, LO = np.meshgrid(la, lo, indexing='ij')
        pts.append(np.stack([LA.ravel(), LO.ravel()], axis=1))
        vals.append(zz.ravel())
    P = np.concatenate(pts, axis=0)
    V = np.concatenate(vals, axis=0)
    return cKDTree(P), V

def sample_etopo(tree, vals, lat_vec, lon_vec):
    xy = np.stack([lat_vec, lon_vec], axis=1)
    ok = np.all(np.isfinite(xy), axis=1)
    out = np.full(lat_vec.shape[0], np.nan, dtype=np.float32)
    if np.any(ok):
        _, idx = tree.query(xy[ok])
        out[ok] = vals[idx]
    return out

# ==============================
# 1) ML monthly means at ±0.005° box
# ==============================
print('• Loading ML monthly mean (ave_gr.npy, level 5) ...')
lat_ml = np.load(os.path.join(ml_data_path, 'lat.npy'))   # (tile,y,x) or (y,x)
lon_ml = np.load(os.path.join(ml_data_path, 'lon.npy'))
if lat_ml.shape != lon_ml.shape:
    raise ValueError('ML lat.npy and lon.npy shapes differ')

ml_mean_mon = [None]*12
for m in range(1, 13):
    subdir = os.path.join(ml_data_path, f"{m:02d}")
    ave_fp = os.path.join(subdir, 'ave_gr.npy')
    if not os.path.exists(ave_fp):
        print(f"  ⚠️ Missing ML ave file for month {m}: {ave_fp}")
        ml_mean_mon[m-1] = None
        continue
    arr = np.load(ave_fp)
    arr2 = _extract_level(arr, LEVEL_IDX)
    if arr2.shape != lat_ml.shape:
        raise ValueError(f"ML month {m} shape mismatch: {arr2.shape} vs {lat_ml.shape}")
    ml_mean_mon[m-1] = arr2.astype(np.float32)

latv = lat_ml.ravel()
lonv = lon_ml.ravel()

def ml_radius_mean(lat0, lon0, radius_deg=0.005):
    mm = np.full(12, np.nan, dtype=np.float32)
    mask = (np.abs(latv - lat0) <= radius_deg) & (np.abs(lonv - lon0) <= radius_deg)
    if not np.any(mask): return mm
    idx = np.where(mask)[0]
    for m in range(12):
        f = ml_mean_mon[m]
        if f is None: continue
        vals = f.ravel()[idx]
        if vals.size > 0:
            mm[m] = float(np.nanmean(vals))
    return mm

# ==============================
# 2) CRU monthly means @ nearest grid
# ==============================
print('• Loading CRU ...')
with xr.open_dataset(cru_nc_path) as ds_c:
    varname = [v for v in ds_c.data_vars if v not in ('lat','lon')][0]  # probably 'tmp'
    da_c = ds_c[varname]  # (time, lat, lon)
    if 'time' not in da_c.coords:
        raise ValueError('CRU file has no time coordinate')
    sel_c = da_c.sel(time=slice(TIME_START, TIME_END))
    lat_c_vals = ds_c['lat'].values
    lon_c_vals = ds_c['lon'].values

def cru_nearest_mean(lat0, lon0):
    with xr.open_dataset(cru_nc_path) as ds_c:
        varname = [v for v in ds_c.data_vars if v not in ('lat','lon')][0]
        da = ds_c[varname].sel(time=slice(TIME_START, TIME_END))
        lat_vals = ds_c['lat'].values
        lon_vals = ds_c['lon'].values
        lonq = lon_to_0360_if_needed(lon_vals, lon0)
        iy = nearest_index(lat_vals, lat0)
        ix = nearest_index(lon_vals, lonq)
        out = np.full(12, np.nan, dtype=np.float32)
        for im, m in enumerate(MON_NUMS):
            sel_m = da.sel(time=da['time'].dt.month==m)[:, iy, ix].values
            if sel_m.size > 0:
                out[im] = float(np.nanmean(sel_m))
        return out

# ==============================
# 3) ERA monthly means @ nearest grid
# ==============================
print('• Loading ERA ...')
with xr.open_dataset(era_nc_path) as ds_e:
    time_name = 'valid_time' if 'valid_time' in ds_e.variables else 'time'
    lat_name  = 'latitude'   if 'latitude'   in ds_e.coords    else 'lat'
    lon_name  = 'longitude'  if 'longitude'  in ds_e.coords    else 'lon'
    da_e = (ds_e['t2m'] - 273.15).sel({time_name: slice(TIME_START, TIME_END)})
    lat_e_vals = ds_e[lat_name].values
    lon_e_vals = ds_e[lon_name].values

def era_nearest_mean(lat0, lon0):
    with xr.open_dataset(era_nc_path) as ds_e:
        time_name = 'valid_time' if 'valid_time' in ds_e.variables else 'time'
        lat_name  = 'latitude'   if 'latitude'   in ds_e.coords    else 'lat'
        lon_name  = 'longitude'  if 'longitude'  in ds_e.coords    else 'lon'
        da = (ds_e['t2m'] - 273.15).sel({time_name: slice(TIME_START, TIME_END)})
        lat_vals = ds_e[lat_name].values
        lon_vals = ds_e[lon_name].values
        lonq = lon_to_0360_if_needed(lon_vals, lon0)
        iy = nearest_index(lat_vals, lat0)
        ix = nearest_index(lon_vals, lonq)
        out = np.full(12, np.nan, dtype=np.float32)
        tcoord = da[time_name]
        for im, m in enumerate(MON_NUMS):
            sel_m = da.sel({time_name: tcoord.dt.month==m})[:, iy, ix].values
            if sel_m.size > 0:
                out[im] = float(np.nanmean(sel_m))
        return out

# ==============================
# 4) ETOPO altitude for color per longitude (at fixed LAT)
# ==============================
print('• Sampling ETOPO altitude for each longitude ...')
tiles = get_etopo_tile_names((LAT_FIXED, LAT_FIXED), (LON_START, LON_END))
la, lo, zz = load_etopo_tiles(etopo_dir, tiles)
tree_z, zvals = build_kdtree(la, lo, zz)
altitudes = sample_etopo(tree_z, zvals,
                         lat_vec=np.full(LONS.shape, LAT_FIXED),
                         lon_vec=LONS)
# fallback if all NaN
if not np.any(np.isfinite(altitudes)):
    altitudes = np.zeros_like(LONS)

# ==============================
# 5) Compute series for all longitudes & plot
# ==============================
print(f'• Computing monthly MEANS at {LAT_FIXED}N, {LON_START}–{LON_END}E ...')

ml_all_mean  = []
era_all_mean = []
cru_all_mean = []
for lo in LONS:
    ml_all_mean.append( ml_radius_mean(LAT_FIXED, lo, radius_deg=0.005) )
    era_all_mean.append( era_nearest_mean(LAT_FIXED, lo) )
    cru_all_mean.append( cru_nearest_mean(LAT_FIXED, lo) )

ml_all_mean  = np.stack(ml_all_mean,  axis=0)  # (nLON, 12)
era_all_mean = np.stack(era_all_mean, axis=0)
cru_all_mean = np.stack(cru_all_mean, axis=0)

fig, ax = plt.subplots(1,1, figsize=(9.6, 5.2))

# ----- draw ML (color by altitude per longitude) -----
for i, lo in enumerate(LONS):
    col = CMAP(ALT_NORM(altitudes[i] if np.isfinite(altitudes[i]) else 0.0))
    ax.plot(MON_NUMS, ml_all_mean[i], color=col, lw=ML_LW, alpha=ALPHA)

# ----- draw ERA (gray solid) & CRU (gray dashed) -----
for i in range(len(LONS)):
    ax.plot(MON_NUMS, era_all_mean[i], color=COL_ERA, lw=ERA_LW, alpha=0.35)
    ax.plot(MON_NUMS, cru_all_mean[i], color=COL_CRU, lw=CRU_LW, alpha=0.35, linestyle=(0,(6,3)))

# Cosmetics
ax.set_xlim(1,12)
ax.set_xticks(MON_NUMS)
ax.set_xticklabels(MON_LABS)
ax.set_xlabel('Month')
ax.set_ylabel(ylab)
ax.grid(True, alpha=0.25)
ax.set_title(f"{title_prefix} — {LON_START}°E to {LON_END}°E (Δ=0.01°)\n"
             f"Monthly mean across years: {TIME_START[:7]} to {TIME_END[:7]}")

# altitude colorbar (optional; comment out if not needed)
from matplotlib.colorbar import ColorbarBase
cax = fig.add_axes([0.92, 0.18, 0.02, 0.6])
ColorbarBase(cax, cmap=CMAP, norm=ALT_NORM, orientation='vertical')
cax.set_ylabel('Altitude (m)')

plt.tight_layout(rect=[0,0,0.9,1])  # leave space for colorbar
plt.savefig(out_png, dpi=300)
plt.close()
print('✅ Saved:', out_png)

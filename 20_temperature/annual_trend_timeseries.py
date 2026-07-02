#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Monthly linear trend (°C/decade) at 65°N / 67°N for ML, ERA5, CRU.
One combined figure per region with many longitudes (0.01° spacing).

Style:
  - ML  : line color = ETOPO altitude (viridis, 0–1400m)
  - ERA : gray solid line
  - CRU : gray dashed line
  - No ±1σ band, no thick center line
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from matplotlib.colors import Normalize
import matplotlib as mpl

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
etopo_dir    = '/data1/DATA_ARCHIVE/ETOPO2022/'   # ETOPO 2022 tiles directory
fig_dir      = '../FIG/timeseries/'
os.makedirs(fig_dir, exist_ok=True)

# Plot styling
COL_ERA = '0.25'   # gray solid
COL_CRU = '0.40'   # gray dashed
ALPHA_ALL = 0.55
LW_ALL    = 1.2

# Month axis
MON_LABS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
MON_NUMS = np.arange(1,13)

# Altitude color scale for ML
ALT_CMAP = plt.get_cmap('viridis')
ALT_NORM = Normalize(vmin=0, vmax=1400)  # 0–1400 m

# ML vertical level index to use
LEVEL_IDX = 5  # 6th level

# ==============================
# Region config
# ==============================
if REGION_KEY.upper() == 'U':
    LAT_FIXED = 65.0
    LON_START, LON_END, LON_STEP = 59.75, 60.25, 0.01
    title_prefix = 'Ural 65°N'
    out_png = os.path.join(fig_dir, 'Ural_65N_59.75-60.25E_monthly_trend_colorelev.png')
elif REGION_KEY.upper() == 'V':
    LAT_FIXED = 67.0
    LON_START, LON_END, LON_STEP = 132.75, 133.25, 0.01
    title_prefix = 'Verkhoyansk 67°N'
    out_png = os.path.join(fig_dir, 'Verkhoyansk_67N_132.75-133.25E_monthly_trend_colorelev.png')
else:
    raise ValueError("REGION_KEY must be 'U' or 'V'")

LONS = np.round(np.arange(LON_START, LON_END + 1e-6, LON_STEP), 4)
LON_CENTER = np.round(0.5*(LON_START+LON_END), 4)

# ==============================
# Small helpers
# ==============================
def nearest_index(coord_vals, target):
    coord_vals = np.asarray(coord_vals)
    return int(np.nanargmin(np.abs(coord_vals - target)))

def lon_to_0360_if_needed(lon_vals, query_lon):
    return query_lon % 360 if np.nanmax(lon_vals) > 180 else query_lon

def _extract_level(arr, level_idx):
    """Extract 2D field for ML given possible shapes."""
    if arr.ndim == 4:  # (tile, level, y, x)
        if level_idx >= arr.shape[1]:
            raise ValueError(f"LEVEL_IDX {level_idx} out of range for {arr.shape}")
        return arr[:, level_idx, :, :]
    elif arr.ndim == 3:
        # (level,y,x) or (tile,y,x)
        if arr.shape[0] > 20:     # likely (tile,y,x)
            return arr
        if level_idx >= arr.shape[0]:
            raise ValueError(f"LEVEL_IDX {level_idx} out of range for {arr.shape}")
        return arr[level_idx, :, :]
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"Unexpected ML array shape: {arr.shape}")

def _to_years(tcoord):
    t = xr.DataArray(tcoord).dt
    return (t.year + (t.dayofyear - 1)/365.25).values.astype(float)

def lintrend_and_stderr(y, x_years):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x_years, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    xx = x[m]; yy = y[m]
    xmean = xx.mean(); ymean = yy.mean()
    Sxx = np.sum((xx - xmean)**2)
    if Sxx <= 0:
        return np.nan, np.nan
    slope = np.sum((xx - xmean)*(yy - ymean)) / Sxx
    resid = yy - (slope*(xx - xmean) + ymean)
    dof = max(len(yy) - 2, 1)
    s2 = np.sum(resid**2) / dof
    stderr = np.sqrt(s2 / Sxx)
    return slope, stderr

# ==============================
# ETOPO utilities (altitude sampling at given lat/lon)
# ==============================
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
            print(f"  ⚠️ missing ETOPO tile: {p}")
            continue
        with Dataset(p) as nc:
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
            z   = nc.variables['z'][:].astype(np.float32)
            z[z <= 0] = np.nan  # ocean = NaN
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

# Load ETOPO altitude for our longitude list
print('• Sampling ETOPO altitude for ML coloring ...')
tile_names = get_etopo_tile_names((LAT_FIXED, LAT_FIXED), (LON_START, LON_END))
la, lo, zz = load_etopo_tiles(etopo_dir, tile_names)
tree_z, zvals = build_kdtree(la, lo, zz)
alt_lons = sample_etopo(tree_z, zvals, np.full_like(LONS, LAT_FIXED, dtype=float), LONS)  # meters

# ==============================
# 1) ML trend: slope (°C/year) within ±0.005° box
# ==============================
print('• Loading ML monthly TREND (slope only) ...')
lat_ml = np.load(os.path.join(ml_data_path, 'lat.npy'))
lon_ml = np.load(os.path.join(ml_data_path, 'lon.npy'))
if lat_ml.shape != lon_ml.shape:
    raise ValueError('ML lat.npy and lon.npy shapes differ')

ml_slope_mon = [None]*12
for m in range(1, 13):
    subdir = os.path.join(ml_data_path, f"{m:02d}")
    slope_fp = os.path.join(subdir, 'trend_gr.npy')
    if not os.path.exists(slope_fp):
        print(f"  ⚠️ missing ML slope file for month {m}: {slope_fp}")
        ml_slope_mon[m-1] = None
        continue
    slope_arr = np.load(slope_fp)
    slope2d = _extract_level(slope_arr, LEVEL_IDX)
    if slope2d.shape != lat_ml.shape:
        raise ValueError(f"ML trend shape mismatch for month {m}: {slope2d.shape} vs {lat_ml.shape}")
    ml_slope_mon[m-1] = slope2d.astype(np.float32)  # °C/year

def ml_radius_trend(lat0, lon0, radius_deg=0.005):
    """Return 12-month slope (°C/year) averaged in ±radius box."""
    mm = np.full(12, np.nan, dtype=np.float32)
    latv = lat_ml.ravel()
    lonv = lon_ml.ravel()
    mask = (np.abs(latv - lat0) <= radius_deg) & (np.abs(lonv - lon0) <= radius_deg)
    if not np.any(mask):
        return mm
    for m in range(12):
        fs = ml_slope_mon[m]
        if fs is None:
            continue
        vals = fs.ravel()[mask]
        if vals.size > 0:
            mm[m] = float(np.nanmean(vals))
    return mm

# ==============================
# 2) CRU: monthly trend at nearest grid
# ==============================
print('• Loading CRU ...')
ds_c = xr.open_dataset(cru_nc_path)
var_c = [v for v in ds_c.data_vars if v not in ('lat','lon')][0]  # e.g., 'tmp'
da_c = ds_c[var_c]
if 'time' not in da_c.coords:
    raise ValueError('CRU file has no time coordinate')
sel_c = da_c.sel(time=slice(TIME_START, TIME_END)).sortby('time')
t_years_c = _to_years(sel_c['time'])
lat_c_vals = ds_c['lat'].values
lon_c_vals = ds_c['lon'].values

def cru_nearest_trend(lat0, lon0):
    lonq = lon_to_0360_if_needed(lon_c_vals, lon0)
    iy = nearest_index(lat_c_vals, lat0)
    ix = nearest_index(lon_c_vals, lonq)
    mm = np.full(12, np.nan, dtype=np.float32)
    for im, m in enumerate(MON_NUMS):
        month_mask = (sel_c['time'].dt.month == m).values
        series = sel_c.values[month_mask, iy, ix]
        slope, _ = lintrend_and_stderr(series, t_years_c[month_mask])
        mm[im] = slope
    return mm

# ==============================
# 3) ERA5: monthly trend at nearest grid
# ==============================
print('• Loading ERA ...')
ds_e = xr.open_dataset(era_nc_path)
time_name = 'valid_time' if 'valid_time' in ds_e.variables else 'time'
lat_name  = 'latitude'   if 'latitude'   in ds_e.coords    else 'lat'
lon_name  = 'longitude'  if 'longitude'  in ds_e.coords    else 'lon'

da_e = (ds_e['t2m'] - 273.15).sel({time_name: slice(TIME_START, TIME_END)}).sortby(time_name)
t_years_e = _to_years(da_e[time_name])
lat_e_vals = ds_e[lat_name].values
lon_e_vals = ds_e[lon_name].values

def era_nearest_trend(lat0, lon0):
    lonq = lon_to_0360_if_needed(lon_e_vals, lon0)
    iy = nearest_index(lat_e_vals, lat0)
    ix = nearest_index(lon_e_vals, lonq)
    mm = np.full(12, np.nan, dtype=np.float32)
    tcoord = da_e[time_name]
    for im, m in enumerate(MON_NUMS):
        month_mask = (tcoord.dt.month == m).values
        series = da_e.values[month_mask, iy, ix]
        slope, _ = lintrend_and_stderr(series, t_years_e[month_mask])
        mm[im] = slope
    return mm

# ==============================
# 4) Compute series for all longitudes & plot
# ==============================
print(f'• Computing monthly trends at {LAT_FIXED}N, {LON_START}–{LON_END}E ...')

ml_all_slope = []
era_all_slope = []
cru_all_slope = []

for lo in LONS:
    ms_ml  = ml_radius_trend(LAT_FIXED, lo, radius_deg=0.005)  # °C/yr
    ms_era = era_nearest_trend(LAT_FIXED, lo)                  # °C/yr
    ms_cru = cru_nearest_trend(LAT_FIXED, lo)                  # °C/yr
    ml_all_slope.append(ms_ml)
    era_all_slope.append(ms_era)
    cru_all_slope.append(ms_cru)

ml_all_slope  = np.stack(ml_all_slope,  axis=0)  # (nLON, 12)
era_all_slope = np.stack(era_all_slope, axis=0)
cru_all_slope = np.stack(cru_all_slope, axis=0)

# Convert to °C/decade
scale = 10.0
ml_plot  = ml_all_slope  * scale
era_plot = era_all_slope * scale
cru_plot = cru_all_slope * scale

fig, ax = plt.subplots(1,1, figsize=(9.5, 5.2))

# ML: color by altitude
for i, lo in enumerate(LONS):
    alt_m = alt_lons[i]
    color = ALT_CMAP(ALT_NORM(alt_m)) if np.isfinite(alt_m) else ALT_CMAP(0.0)
    ax.plot(MON_NUMS, ml_plot[i], color=color, alpha=ALPHA_ALL, lw=LW_ALL)

# ERA: gray solid
for i in range(len(LONS)):
    ax.plot(MON_NUMS, era_plot[i], color=COL_ERA, alpha=ALPHA_ALL, lw=LW_ALL, linestyle='-')

# CRU: gray dashed
for i in range(len(LONS)):
    ax.plot(MON_NUMS, cru_plot[i], color=COL_CRU, alpha=ALPHA_ALL, lw=LW_ALL, linestyle=(0,(5,3)))

# Axes, labels, title
ax.set_xlim(1,12)
ax.set_xticks(MON_NUMS)
ax.set_xticklabels(MON_LABS)
ax.set_xlabel('Month')
ax.set_ylabel('Trend (°C / decade)')
ax.grid(True, alpha=0.25)
ax.set_title(f"{title_prefix} — {LON_START}°E to {LON_END}°E (Δ=0.01°)\n"
             f"Monthly linear trend, {TIME_START[:7]} to {TIME_END[:7]}")

# Colorbar for altitude (applies to ML lines)
sm = mpl.cm.ScalarMappable(norm=ALT_NORM, cmap=ALT_CMAP)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.06, pad=0.04)
cbar.set_label('Altitude (m) — ML line color')

plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.close()
print('✅ Saved:', out_png)

# Close datasets
ds_c.close()
ds_e.close()

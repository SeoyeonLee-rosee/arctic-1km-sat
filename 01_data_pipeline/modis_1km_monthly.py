#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Process additional MODIS products consistent with the existing pipeline.

Targets under /data1/DATA_ARCHIVE/Satellite/MODIS :
  - water_reservoir_monthly/MOD28C3.*.hdf                   (monthly, global grid)
  - SC/MOD10A2.*.h*v*.061.*.hdf                             (8-day Snow Cover, 500 m, sinusoidal tiles)
  - GPP/yyyy/MOD17A2HGF.*.h*v*.061.*.hdf                    (8-day GPP gap-filled, 500 m, sinusoidal tiles)
  - FPAR/yyyy/MOD15A2H.*.h*v*.061.*.hdf                     (8-day FPAR/LAI, 500 m, sinusoidal tiles)

Outputs (per product):
  - Per month:
      * Integrated tile stack: (34, n_years, NY, NX)  → <var>/MM/<var>_tiles.npy  (+ years.npy)
      * Per-tile stacks: (n_years, NY, NX)           → <var>/MM/hXXvYY/<var>.npy
      * Tile means CSV
      * Summaries (34, NY, NX): ave.npy, std.npy, trend.npy (per decade), trend_var.npy (per decade^2)
  - Lat/Lon stacks for the 34 tiles once per product/resolution: lat_tiles_*.npy / lon_tiles_*.npy
  - Default target resolution for tile products is 1 km (1200x1200) when --to_1km

Notes:
  * 500 m tile products (SC, GPP, FPAR/LAI) can be downscaled to 1 km via 2x2 NaN-aware averaging.
  * 8-day products aggregated to monthly: mean (SC/FPAR/LAI) or sum (GPP).
  * MOD28C3 is native global grid; with --reproject_mod28 also produces tile stacks/statistics.
  * Tiles are locked to a fixed set of 34 tiles.

Examples:
  python modis_1km_monthly.py --products GPP,FPAR,SC --to_1km --start_ym 2000-03 --end_ym 2022-04
  python modis_1km_monthly.py --products MOD28C3 --reproject_mod28 --to_1km --start_ym 2000-03 --end_ym 2022-04
"""

import os
import re
import glob
import math
import calendar
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd

# Try to import pyhdf.SD for HDF4-EOS
try:
    from pyhdf.SD import SD, SDC
except Exception:
    SD = None
    SDC = None

# Optional: SciPy KDTree for MOD28C3 reprojection
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

# Optional: GDAL for HDF4 fallback reading (try both import styles)
gdal = None
try:
    from osgeo import gdal as _gdal
    gdal = _gdal
except Exception:
    try:
        import gdal as _gdal
        gdal = _gdal
    except Exception:
        gdal = None

# -------------------------
# Configuration
# -------------------------
BASE = "/data1/DATA_ARCHIVE/Satellite/MODIS"
CONFIG = OrderedDict({
    "SC": {
        "glob": os.path.join(BASE, "SC", "MOD10A2.*.h*v*.061.*.hdf"),
        "sds_candidates": ["NDSI_Snow_Cover", "Eight_Day_Snow_Cover", "NDSI_Snow_Cover_Mean"],
        "scale": 1.0,                      # percentage already scaled
        "fill_values": [255],
        "valid_range": (0, 100),
        "agg": "mean",                     # monthly mean
        "res_m": 500,
        "varname_out": "SC",
    },
    "GPP": {
        "glob": os.path.join(BASE, "GPP", "*", "MOD17A2HGF.*.h*v*.061.*.hdf"),
        "sds_candidates": ["Gpp_500m", "Gpp", "GPP"],
        "scale": 1e-4,                     # typical MOD17 scale (kg C m-2 per 8-day)
        "fill_values": [65535, 32767],
        "valid_range": (0, 65000),
        "agg": "sum",                      # monthly sum of 8-day periods
        "res_m": 500,
        "varname_out": "GPP",              # units after scale: kg C m-2 per month
    },
    "FPAR": {
        "glob": os.path.join(BASE, "FPAR", "*", "MOD15A2H.*.h*v*.061.*.hdf"),
        "sds_candidates": ["Fpar_500m", "Fpar"],
        "scale": 0.01,                     # 0..1 after scale
        "fill_values": [255],
        "valid_range": (0, 100),
        "agg": "mean",                     # monthly mean
        "res_m": 500,
        "varname_out": "FPAR",
    },
    "LAI": {
        "glob": os.path.join(BASE, "FPAR", "*", "MOD15A2H.*.h*v*.061.*.hdf"),
        "sds_candidates": ["Lai_500m", "Lai"],
        "scale": 0.1,                      # LAI 0..10 after scale
        "fill_values": [255],
        "valid_range": (0, 100),
        "agg": "mean",                     # monthly mean
        "res_m": 500,
        "varname_out": "LAI",
    },
    "MOD28C3": {
        "glob": os.path.join(BASE, "water_reservoir_monthly", "MOD28C3.*.hdf"),
        "sds_candidates": ["Water_Reservoir", "Reservoir", "SST", "temperature", "Variable"],
        "scale": None,                     # attempt to read from attributes if present
        "fill_values": None,               # attempt to read from attributes if present
        "valid_range": None,               # attempt to read from attributes if present
        "agg": "native",                   # already monthly
        "res_m": None,
        "varname_out": "MOD28C3",
    },
    # Optional alias: WR (same input as MOD28C3) – outputs under ../data/MOD/WR/
    "WR": {
        "glob": os.path.join(BASE, "water_reservoir_monthly", "MOD28C3.*.hdf"),
        "sds_candidates": ["Water_Reservoir", "Reservoir", "SST", "temperature", "Variable"],
        "scale": None,
        "fill_values": None,
        "valid_range": None,
        "agg": "native",
        "res_m": None,
        "varname_out": "WR",
    }
})

# --- Fixed 34 tiles ---
FIXED_TILES_34 = [
    (9, 2), (10, 2), (11, 2), (12, 1), (12, 2),
    (13, 1), (13, 2), (14, 1), (14, 2), (15, 1),
    (15, 2), (16, 0), (16, 1), (16, 2), (17, 0),
    (17, 1), (17, 2), (18, 0), (18, 1), (18, 2),
    (19, 0), (19, 1), (19, 2), (20, 1), (20, 2),
    (21, 1), (21, 2), (22, 1), (22, 2), (23, 1),
    (23, 2), (24, 2), (25, 2), (26, 2),
]
DEFAULT_MONTHS = list(range(1, 13))

# Default product root helper: ../data/MOD/<varname>
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/ 기준
def default_product_root(varname: str) -> str:
    return os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "MOD", varname))

# -------------------------
# Utility functions
# -------------------------
def log(msg: str):
    print(f"[process_modis_extras] {msg}")

def parse_ayyyyddd(path):
    """Extract (year, doy) from filename pattern 'AYYYYDDD'. Return (year, month)."""
    m = re.search(r"A(\d{4})(\d{3})", os.path.basename(path))
    if not m:
        return None, None
    y = int(m.group(1)); doy = int(m.group(2))
    dt = datetime(y, 1, 1) + timedelta(doy - 1)
    return y, dt.month

def parse_hv(path):
    m = re.search(r"h(\d{2})v(\d{2})", os.path.basename(path))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def month_name(mm):
    return f"{mm:02d}_{calendar.month_abbr[mm]}"

def _parse_ym(s):
    m = re.match(r"^(\d{4})-(\d{2})$", s or "")
    if not m:
        raise ValueError(f"YM format must be YYYY-MM, got: {s}")
    return int(m.group(1)), int(m.group(2))

def _in_window(y, m, start_ym=None, end_ym=None):
    if start_ym and (y, m) < start_ym: return False
    if end_ym   and (y, m) > end_ym:   return False
    return True

# --- MODIS sinusoidal tile mapping to lat/lon ---
def geomapping(h: int, v: int, npix: int = 1200):
    R = 6371007.181
    TILE_SIZE_M = 1111950.519765  # ~10 degrees at equator in sinusoidal meters
    pix_size = TILE_SIZE_M / npix  # if npix=1200 => 1 km; if 2400 => 500 m
    x0 = - R * math.pi + h * TILE_SIZE_M
    y0 =  R * math.pi - v * TILE_SIZE_M
    i = np.arange(npix); j = np.arange(npix)
    xx = x0 + (i + 0.5) * pix_size
    yy = y0 - (j + 0.5) * pix_size
    X, Y = np.meshgrid(xx, yy)
    lon = np.degrees(X / (R * np.cos(Y / R)))
    lat = np.degrees(Y / R)
    lon = np.clip(lon, -180, 180)
    lat = np.clip(lat, -90, 90)
    return lat.astype(np.float32), lon.astype(np.float32)

# --- HDF reading helpers ---
def _attr_number(attrs, keys, default=None):
    for k in keys:
        if k in attrs:
            try:
                return float(attrs[k])
            except Exception:
                try:
                    return float(attrs[k][0])
                except Exception:
                    pass
    return default

def read_hdf_sds(hdf_path, candidates=None):
    """
    Return (array, attrs, sds_name).
    1) pyhdf 우선: 후보 이름으로 시도 → 실패시 첫 2D numeric SDS
    2) GDAL 폴백: subdataset 중 후보/첫 번째 선택
    """
    # --- pyhdf path ---
    if SD is not None:
        try:
            sd = SD(hdf_path, SDC.READ)
        except Exception:
            sd = None
        else:
            datasets = sd.datasets()
            if candidates:
                for name in candidates:
                    try:
                        if name in datasets:
                            sds = sd.select(name)
                            arr = sds[:]
                            if getattr(arr, "ndim", 0) >= 2 and np.issubdtype(arr.dtype, np.number):
                                attrs = sds.attributes()
                                sd.end()
                                return arr.astype(np.float32), attrs, name
                    except Exception:
                        pass
            for name in datasets.keys():
                try:
                    sds = sd.select(name); arr = sds[:]
                    if getattr(arr, "ndim", 0) == 2 and np.issubdtype(arr.dtype, np.number):
                        attrs = sds.attributes(); sd.end()
                        return arr.astype(np.float32), attrs, name
                except Exception:
                    continue
            names_preview = list(datasets.keys())[:8]; sd.end()
            raise RuntimeError("No suitable 2D SDS found in %s. Datasets: %s..." % (os.path.basename(hdf_path), names_preview))

    # --- GDAL fallback path ---
    if gdal is not None:
        ds = gdal.Open(hdf_path)
        if ds is None:
            raise RuntimeError("GDAL could not open %s" % hdf_path)
        subs = ds.GetSubDatasets() or []
        pick = None
        if candidates:
            cand_lower = [c.lower() for c in candidates]
            for name, desc in subs:
                if any(c in desc.lower() for c in cand_lower):
                    pick = name; break
        if pick is None and subs:
            pick = subs[0][0]
        if not pick:
            raise RuntimeError("No subdatasets found in %s and pyhdf unavailable." % os.path.basename(hdf_path))
        sds = gdal.Open(pick)
        arr = sds.ReadAsArray().astype(np.float32)
        meta = sds.GetMetadata() or {}
        attrs = dict(meta)
        for k in ("scale_factor", "Scale", "SCALE", "scaleFactor"):
            if k in meta:
                try: attrs["scale_factor"] = float(meta[k]); break
                except Exception: pass
        for k in ("add_offset", "Offset", "offset"):
            if k in meta:
                try: attrs["add_offset"] = float(meta[k]); break
                except Exception: pass
        return arr, attrs, os.path.basename(pick)

    raise RuntimeError(
        "Neither 'pyhdf' nor 'GDAL' is available. Install one of them:\n"
        " - conda: conda install -c conda-forge pyhdf\n"
        " - or (fallback) conda install -c conda-forge gdal"
    )

def apply_scale_and_mask(arr, attrs, fallback_scale=None, valid_range=None, fill_values=None):
    if fill_values is None:
        fill_values = []
        for k in ["_FillValue", "fillvalue", "MissingValue", "missing_value", "_MissingValue"]:
            if k in attrs:
                v = attrs[k]
                if isinstance(v, (list, tuple, np.ndarray)): fill_values.extend(list(v))
                else: fill_values.append(v)
        fill_values = [float(v) for v in fill_values]
    else:
        fill_values = [float(v) for v in fill_values]

    if valid_range is None:
        if "valid_range" in attrs:
            vr = attrs["valid_range"]
            try: valid_range = (float(vr[0]), float(vr[1]))
            except Exception: valid_range = None
        else:
            valid_range = None

    scale = _attr_number(attrs, ["scale_factor", "Scale", "Slope"], default=fallback_scale if fallback_scale is not None else 1.0)
    add_offset = _attr_number(attrs, ["add_offset", "Intercept"], default=0.0)

    out = arr.astype(np.float32)
    if fill_values:
        mask_fill = np.isin(out, np.array(fill_values, dtype=out.dtype))
        out[mask_fill] = np.nan
    if valid_range is not None:
        lo, hi = valid_range
        mask_vr = (out < lo) | (out > hi)
        out[mask_vr] = np.nan
    out = out * (scale if scale is not None else 1.0) + (add_offset if add_offset is not None else 0.0)
    return out

# --- Aggregation helpers ---
def nanmean(arrs):
    if not arrs: return None
    stack = np.stack(arrs, axis=0)
    return np.nanmean(stack, axis=0)

def nansum(arrs):
    if not arrs: return None
    stack = np.stack(arrs, axis=0)
    return np.nansum(stack, axis=0)

def downscale_2x_mean(arr):
    if arr is None: return None
    ny, nx = arr.shape
    if ny % 2 != 0 or nx % 2 != 0:
        raise ValueError(f"Array shape {arr.shape} is not even; cannot 2x downscale.")
    a = arr.reshape(ny//2, 2, nx//2, 2)
    with np.errstate(invalid='ignore'):
        out = np.nanmean(np.nanmean(a, axis=3), axis=1)
    return out

# --- Stats helpers ---
def _ols_slope_and_var(y_TN: np.ndarray, years_vec: np.ndarray):
    """
    y_TN: (T, N) float32 with NaNs; years_vec: (T,)
    returns slope_per_year(N,), var_slope_per_year(N,)
    """
    y = y_TN.astype(np.float32, copy=False)
    x = years_vec.astype(np.float32)
    x = x - np.nanmean(x)  # center
    X = x[:, None]
    M = np.isfinite(y).astype(np.float32)
    n = M.sum(axis=0)

    xw_mean = (M * X).sum(axis=0) / np.maximum(n, 1)
    yw_mean = (np.nansum(y, axis=0)) / np.maximum(n, 1)

    dx = X - xw_mean[None, :]
    dy = y - yw_mean[None, :]

    num = np.nansum(M * dx * dy, axis=0)
    den = np.nansum(M * dx * dx, axis=0)
    slope = np.full_like(den, np.nan, dtype=np.float32)
    good = (den > 0) & (n >= 2)
    slope[good] = num[good] / den[good]

    var_slope = np.full_like(den, np.nan, dtype=np.float32)
    if np.any(good):
        intercept = yw_mean[good] - slope[good] * xw_mean[good]
        yhat = X * slope[good][None, :] + intercept[None, :]
        resid = (y[:, good] - yhat) * M[:, good]
        rss = np.nansum(resid * resid, axis=0)
        dof = np.clip(n[good] - 2, 1, None)
        s2 = rss / dof
        var_slope[good] = s2 / den[good]
    return slope, var_slope

def compute_monthly_stats(product_root: str, varname: str, months):
    """
    For each month in months, if <var>_tiles.npy exists:
      - load years.npy (or 0..T-1)
      - compute ave, std, trend (per decade), trend_var (per decade^2)
      - save to month dir
    """
    for mm in months:
        month_dir = os.path.join(product_root, f"{mm:02d}")
        stack_path = os.path.join(month_dir, f"{varname}_tiles.npy")
        if not os.path.exists(stack_path):
            log(f"[stats] Skip {varname} {mm:02d}: no integrated stack found ({stack_path}).")
            continue

        tiles_stack = np.load(stack_path, mmap_mode="r")   # (34, T, H, W)
        if tiles_stack.ndim != 4 or tiles_stack.shape[0] != 34:
            log(f"[stats] Unexpected shape for {stack_path}: {tiles_stack.shape}")
            continue

        T, H, W = tiles_stack.shape[1], tiles_stack.shape[2], tiles_stack.shape[3]
        years_path = os.path.join(month_dir, "years.npy")
        years = np.load(years_path).astype(np.float32) if os.path.exists(years_path) else np.arange(T, dtype=np.float32)
        if years.shape[0] != T:
            log(f"[stats] years.npy length {years.shape[0]} != T {T}; falling back to 0..T-1")
            years = np.arange(T, dtype=np.float32)

        ave_all   = np.full((34, H, W), np.nan, dtype=np.float32)
        std_all   = np.full((34, H, W), np.nan, dtype=np.float32)
        trend_all = np.full((34, H, W), np.nan, dtype=np.float32)  # per decade
        tvar_all  = np.full((34, H, W), np.nan, dtype=np.float32)  # per decade^2

        for ti in range(34):
            Y = np.array(tiles_stack[ti], dtype=np.float32, copy=False)  # (T, H, W)
            ave_all[ti] = np.nanmean(Y, axis=0)
            std_all[ti] = np.nanstd(Y, axis=0)

            Y2 = Y.reshape(T, -1)
            slope_per_year, var_slope_per_year = _ols_slope_and_var(Y2, years)
            # Convert to per decade:
            slope_per_decade = slope_per_year * 10.0
            var_slope_per_decade = var_slope_per_year * 100.0

            trend_all[ti] = slope_per_decade.reshape(H, W)
            tvar_all[ti]  = var_slope_per_decade.reshape(H, W)

        np.save(os.path.join(month_dir, "ave.npy"),       ave_all)
        np.save(os.path.join(month_dir, "std.npy"),       std_all)
        np.save(os.path.join(month_dir, "trend.npy"),     trend_all)     # per decade
        np.save(os.path.join(month_dir, "trend_var.npy"), tvar_all)      # per decade^2
        log(f"[stats] Saved summaries (per decade) for {varname} {mm:02d} in {month_dir}.")

# --- Grouping files by (h,v,year,month)
def group_tile_products(files):
    groups = defaultdict(list)
    years = set()
    tiles = set()
    for f in files:
        y, m = parse_ayyyyddd(f)
        h, v = parse_hv(f)
        if y is None or m is None or h is None or v is None:
            continue
        years.add(y); tiles.add((h, v))
        groups[(h, v, y, m)].append(f)
    return groups, sorted(list(years)), sorted(list(tiles))

# --- Main processors ---
def process_tile_product(cfg, months=DEFAULT_MONTHS, years=None, tiles=None,
                         to_1km=True, out_root=None, start_ym=None, end_ym=None):
    varname = cfg["varname_out"]
    files = sorted(glob.glob(cfg["glob"]))
    if not files:
        log(f"No files matched: {cfg['glob']}")
        return

    log(f"Found {len(files)} files for {varname}")
    groups, years_found, tiles_found = group_tile_products(files)

    if years is None:
        years = years_found
    else:
        years = [y for y in years if y in years_found]

    tiles_use = FIXED_TILES_34  # locked to 34 tiles

    product_root = default_product_root(varname) if out_root is None else out_root
    os.makedirs(product_root, exist_ok=True)

    npix_native = 2400 if cfg["res_m"] == 500 else 1200
    npix_target = 1200 if to_1km else npix_native

    # Save lat/lon stacks once per product and resolution
    lat_tiles_path = os.path.join(product_root, f"lat_tiles_{npix_target}.npy")
    lon_tiles_path = os.path.join(product_root, f"lon_tiles_{npix_target}.npy")
    if not (os.path.exists(lat_tiles_path) and os.path.exists(lon_tiles_path)):
        lat_tiles = np.empty((len(tiles_use), npix_target, npix_target), dtype=np.float32)
        lon_tiles = np.empty((len(tiles_use), npix_target, npix_target), dtype=np.float32)
        for ti, (h, v) in enumerate(tiles_use):
            lat, lon = geomapping(h, v, npix=npix_target)
            lat_tiles[ti] = lat; lon_tiles[ti] = lon
        np.save(lat_tiles_path, lat_tiles); np.save(lon_tiles_path, lon_tiles)
        with open(os.path.join(product_root, "tiles_34.csv"), "w") as fw:
            fw.write("index,h,v,tile\n")
            for i, (h, v) in enumerate(tiles_use):
                fw.write(f"{i},{h},{v},h{h:02d}v{v:02d}\n")

    # -------- Monthly processing --------
    for mm in months:
        years_valid = [y for y in years if _in_window(y, mm, start_ym, end_ym)]
        if not years_valid:
            continue

        mm_name = month_name(mm)
        nyears = len(years_valid)
        all_tiles_stack = np.full(
            (len(tiles_use), nyears, npix_target, npix_target),
            np.nan, dtype=np.float32
        )
        tile_means_rows = []

        for ti, (h, v) in enumerate(tiles_use):
            yearly_grids = []
            for y in years_valid:
                f_list = groups.get((h, v, y, mm), [])
                if not f_list:
                    yearly_grids.append(np.full((npix_target, npix_target), np.nan, dtype=np.float32))
                    continue
                arrs = []
                for f in sorted(f_list):
                    data, attrs, _ = read_hdf_sds(f, cfg["sds_candidates"])
                    data = apply_scale_and_mask(
                        data, attrs,
                        fallback_scale=cfg.get("scale"),
                        valid_range=cfg.get("valid_range"),
                        fill_values=cfg.get("fill_values")
                    )
                    if to_1km and data.shape == (2400, 2400):
                        data = downscale_2x_mean(data)
                    arrs.append(data)
                monthly = nansum(arrs) if cfg["agg"] == "sum" else nanmean(arrs)
                yearly_grids.append(monthly.astype(np.float32))

            stack = np.stack(yearly_grids, axis=0)  # (T, H, W)
            all_tiles_stack[ti] = stack

            out_dir_tile = os.path.join(product_root, f"{mm:02d}", f"h{h:02d}v{v:02d}")
            os.makedirs(out_dir_tile, exist_ok=True)
            np.save(os.path.join(out_dir_tile, f"{varname}.npy"), stack)

            tile_mean = np.nanmean(stack.reshape(stack.shape[0], -1), axis=1)
            tile_means_rows.append([f"h{h:02d}v{v:02d}"] + list(tile_mean))

        # Save integrated stack & years
        out_dir_month = os.path.join(product_root, f"{mm:02d}")
        os.makedirs(out_dir_month, exist_ok=True)
        packed_path = os.path.join(out_dir_month, f"{varname}_tiles.npy")
        np.save(packed_path, all_tiles_stack)
        np.save(os.path.join(out_dir_month, "years.npy"), np.array(years_valid, dtype=np.int32))
        log(f"Saved {packed_path} (shape={all_tiles_stack.shape})")

        if tile_means_rows:
            cols = ["tile"] + [str(y) for y in years_valid]
            df = pd.DataFrame(tile_means_rows, columns=cols)
            csv_path = os.path.join(out_dir_month, f"{varname}_{mm_name}_tile_means.csv")
            df.to_csv(csv_path, index=False)
            log(f"Saved {csv_path}")

    # After building stacks: compute monthly stats (per decade trend)
    compute_monthly_stats(product_root, varname, months)

# --- MOD28C3 / WR handler ---
def process_mod28c3_like(cfg, months=DEFAULT_MONTHS, out_root=None,
                         reproject_to_tiles=False, to_1km=True,
                         start_ym=None, end_ym=None):
    varname = cfg["varname_out"]
    files = sorted(glob.glob(cfg["glob"]))
    if not files:
        log(f"No files matched: {cfg['glob']}")
        return

    product_root = default_product_root(varname) if out_root is None else out_root
    os.makedirs(product_root, exist_ok=True)

    # group by (year, month)
    groups = defaultdict(list)
    years = set()
    for f in files:
        y, m = parse_ayyyyddd(f)
        if y is None or m is None:
            continue
        years.add(y)
        if m in months:
            groups[(y, m)].append(f)
    years = sorted(list(years))

    # detect sample grid & optional lat/lon
    sample_file = files[0]
    sample_arr, sample_attrs, sample_sds = read_hdf_sds(sample_file, cfg["sds_candidates"])
    lat, lon = None, None
    try:
        sd = SD(sample_file, SDC.READ)
        for key in ["latitude", "lat", "Latitude", "YDim"]:
            if key in sd.datasets():
                lat = sd.select(key)[:].astype(np.float32); break
        for key in ["longitude", "lon", "Longitude", "XDim"]:
            if key in sd.datasets():
                lon = sd.select(key)[:].astype(np.float32); break
        sd.end()
    except Exception:
        pass

    # Output native grid stacks per month (T, ny, nx)
    for mm in months:
        stacks = []; years_in_month = []
        for y in years:
            if not _in_window(y, mm, start_ym, end_ym):
                continue
            f_list = groups.get((y, mm), [])
            if not f_list:
                continue
            arrs = []
            for f in sorted(f_list):
                data, attrs, _ = read_hdf_sds(f, cfg["sds_candidates"])
                data = apply_scale_and_mask(
                    data, attrs,
                    fallback_scale=cfg.get("scale"),
                    valid_range=cfg.get("valid_range"),
                    fill_values=cfg.get("fill_values")
                )
                arrs.append(data)
            monthly = nanmean(arrs)
            stacks.append(monthly.astype(np.float32)); years_in_month.append(y)

        if not stacks:
            continue

        stack = np.stack(stacks, axis=0)  # (T, ny, nx)
        out_dir = os.path.join(product_root, f"{mm:02d}")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{varname}.npy"), stack)
        if lat is not None and lon is not None:
            np.save(os.path.join(out_dir, "lat_native.npy"), lat)
            np.save(os.path.join(out_dir, "lon_native.npy"), lon)

        gmean = np.nanmean(stack.reshape(stack.shape[0], -1), axis=1)
        df = pd.DataFrame({"year": years_in_month, f"{varname}_global_mean": gmean})
        df.to_csv(os.path.join(out_dir, f"{varname}_{month_name(mm)}_global_means.csv"), index=False)
        log(f"Saved {varname} native stacks for month {mm:02d}")

        # Optional: Reproject to MODIS sinusoidal tiles (nearest-neighbor) and integrated save + stats
        if reproject_to_tiles:
            if KDTree is None:
                log("SciPy not available; skipping reprojection to tiles.")
            else:
                # Build KDTree from native lat/lon (if missing, approximate)
                lat0, lon0 = lat, lon
                if lat0 is None or lon0 is None:
                    ny, nx = stack.shape[1:]; jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
                    lat0 = (jj / (ny - 1)) * 180.0 - 90.0
                    lon0 = (ii / (nx - 1)) * 360.0 - 180.0
                pts = np.vstack([lat0.ravel(), lon0.ravel()]).T
                kdt = KDTree(pts)

                tiles_use = FIXED_TILES_34
                npix = 1200 if to_1km else 2400
                sinu_root = os.path.join(os.path.dirname(product_root), os.path.basename(product_root) + "_sinu")
                os.makedirs(sinu_root, exist_ok=True)

                # Save lat/lon stacks once for the reprojected grid
                lat_tiles_path = os.path.join(sinu_root, f"lat_tiles_{npix}.npy")
                lon_tiles_path = os.path.join(sinu_root, f"lon_tiles_{npix}.npy")
                if not (os.path.exists(lat_tiles_path) and os.path.exists(lon_tiles_path)):
                    lat_tiles = np.empty((len(tiles_use), npix, npix), dtype=np.float32)
                    lon_tiles = np.empty((len(tiles_use), npix, npix), dtype=np.float32)
                    for ti, (h, v) in enumerate(tiles_use):
                        lat_t, lon_t = geomapping(h, v, npix=npix)
                        lat_tiles[ti] = lat_t; lon_tiles[ti] = lon_t
                    np.save(lat_tiles_path, lat_tiles); np.save(lon_tiles_path, lon_tiles)
                    with open(os.path.join(sinu_root, "tiles_34.csv"), "w") as fw:
                        fw.write("index,h,v,tile\n")
                        for i, (h, v) in enumerate(tiles_use):
                            fw.write(f"{i},{h},{v},h{h:02d}v{v:02d}\n")

                # integrated stack for this month
                all_tiles_stack = np.full((len(tiles_use), stack.shape[0], npix, npix), np.nan, dtype=np.float32)

                for ti, (h, v) in enumerate(tiles_use):
                    lat_t, lon_t = geomapping(h, v, npix=npix)
                    qry = np.vstack([lat_t.ravel(), lon_t.ravel()]).T
                    _, idx = kdt.query(qry, k=1)

                    tile_stack = []
                    for k in range(stack.shape[0]):
                        vals = stack[k].ravel()[idx]
                        tile_stack.append(vals.reshape(npix, npix))
                    tile_stack = np.stack(tile_stack, axis=0).astype(np.float32)
                    all_tiles_stack[ti] = tile_stack

                    out_dir_tile = os.path.join(sinu_root, f"{mm:02d}", f"h{h:02d}v{v:02d}")
                    os.makedirs(out_dir_tile, exist_ok=True)
                    np.save(os.path.join(out_dir_tile, f"{varname}.npy"), tile_stack)
                    np.save(os.path.join(out_dir_tile, "lat.npy"), lat_t)
                    np.save(os.path.join(out_dir_tile, "lon.npy"), lon_t)

                out_dir_month = os.path.join(sinu_root, f"{mm:02d}")
                os.makedirs(out_dir_month, exist_ok=True)
                packed_path = os.path.join(out_dir_month, f"{varname}_tiles.npy")
                np.save(packed_path, all_tiles_stack)
                np.save(os.path.join(out_dir_month, "years.npy"), np.array(years_in_month, dtype=np.int32))
                log(f"Saved {packed_path} (shape={all_tiles_stack.shape})")

                # stats for reprojected (tile) stacks
                compute_monthly_stats(sinu_root, varname, [mm])

# -------------------------
# CLI
# -------------------------
def parse_range(spec, cast=int, all_keyword="all", full=None):
    if spec is None:
        return full
    if isinstance(spec, (list, tuple)):
        return list(spec)
    s = str(spec).strip()
    if s.lower() == all_keyword:
        return full
    if "," in s:
        return [cast(x) for x in s.split(",")]
    if "-" in s:
        a, b = s.split("-")
        a = cast(a); b = cast(b)
        return list(range(a, b + 1))
    return [cast(s)]

def main():
    import argparse
    p = argparse.ArgumentParser(description="Process MODIS tile/global products into monthly stacks with summaries.")
    p.add_argument("--products", type=str, default="SC,GPP,FPAR,LAI,MOD28C3",
                   help="Comma-separated list of products to process. Also accepts 'WR' as alias of MOD28C3.")
    p.add_argument("--months", type=str, default="all", help="Months to process: all | 6-9 | 1,2,3 ...")
    p.add_argument("--years", type=str, default=None, help="Years to restrict (e.g., 2001-2024 or 2001,2002,2005)")
    p.add_argument("--tiles", type=str, default="fixed34", help="(ignored; always fixed 34 tiles)")
    p.add_argument("--out_root", type=str, default=None, help="Product root directory. Default: ../data/MOD/<varname>/")
    p.add_argument("--to_1km", action="store_true", help="Downscale 500 m tile products to 1 km (1200x1200).")
    p.add_argument("--keep_native", action="store_true", help="Keep native resolution (e.g., 2400x2400). Overrides --to_1km.")
    p.add_argument("--reproject_mod28", action="store_true", help="Reproject MOD28C3/WR to MODIS sinusoidal tiles (needs SciPy).")
    p.add_argument("--start_ym", type=str, default=None, help="Start YYYY-MM inclusive, e.g., 2000-03")
    p.add_argument("--end_ym",   type=str, default=None, help="End   YYYY-MM inclusive, e.g., 2022-04")

    args = p.parse_args()

    months = parse_range(args.months, cast=int, all_keyword="all", full=DEFAULT_MONTHS)
    years = parse_range(args.years, cast=int, all_keyword="all", full=None) if args.years else None
    to_1km = bool(args.to_1km and not args.keep_native)

    start_ym = _parse_ym(args.start_ym) if args.start_ym else None
    end_ym   = _parse_ym(args.end_ym)   if args.end_ym   else None

    # Normalize product names & handle alias WR→MOD28C3 processing path but with varname_out='WR'
    products = [s.strip() for s in args.products.split(",") if s.strip()]
    for prod in products:
        if prod not in CONFIG:
            log(f"Unknown product '{prod}'. Skipping.")
            continue
        cfg = CONFIG[prod]
        if prod in ("MOD28C3", "WR"):
            process_mod28c3_like(cfg, months=months, out_root=args.out_root,
                                 reproject_to_tiles=args.reproject_mod28, to_1km=to_1km,
                                 start_ym=start_ym, end_ym=end_ym)
        else:
            process_tile_product(cfg, months=months, years=years, tiles=None,
                                 to_1km=to_1km, out_root=args.out_root,
                                 start_ym=start_ym, end_ym=end_ym)

if __name__ == "__main__":
    main()

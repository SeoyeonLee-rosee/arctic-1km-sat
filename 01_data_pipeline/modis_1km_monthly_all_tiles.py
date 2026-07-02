#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Process additional MODIS products in a way consistent with the existing MODIS pipeline.

Targets under /data1/DATA_ARCHIVE/Satellite/MODIS :
  - water_reservoir_monthly/MOD28C3.*.hdf                   (monthly, global grid)
  - SC/MOD10A2.*.h*v*.061.*.hdf                             (8-day Snow Cover, 500 m, sinusoidal tiles)
  - GPP/yyyy/MOD17A2HGF.*.h*v*.061.*.hdf                    (8-day GPP gap-filled, 500 m, sinusoidal tiles)
  - FPAR/yyyy/MOD15A2H.*.h*v*.061.*.hdf                     (8-day FPAR/LAI, 500 m, sinusoidal tiles)

Outputs (by default):
  - Per product & month: per-tile stacks with shape (nyears, NY, NX) saved as .npy
  - Per product & month: tile means (one value per tile per year) saved as CSV
  - Lat/Lon grids per tile (saved once) using MODIS sinusoidal grid, defaulting to 1 km (1200x1200)

Notes:
  * 500 m tile products (SC, GPP, FPAR/LAI) are optionally downscaled to 1 km (1200x1200) via 2x2 NaN-aware averaging
    for consistency with prior 1 km products (NDVI/LST). Disable with --keep_native to keep 2400x2400.
  * 8-day products are aggregated to monthly by mean (state/index variables) or sum (flux/accumulation, e.g. GPP).
  * MOD28C3 is handled in its native (global) grid. If you need it reprojected to MODIS sinusoidal tiles,
    pass --reproject_mod28 to enable a simple nearest-neighbor reprojection (requires SciPy for KDTree).

Usage examples:
  python process_modis_extras.py --months 6-9 --out_root /data1/DATA_ARCHIVE/Satellite/MODIS/processed
  python process_modis_extras.py --months all --products SC,GPP,FPAR,LAI --to_1km --years 2001-2024
  python process_modis_extras.py --reproject_mod28 --tiles all
"""

import os
import re
import glob
import math
import json
import calendar
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd

# Try to import pyhdf.SD for HDF4-EOS
try:
    from pyhdf.SD import SD, SDC
except Exception as e:
    SD = None
    SDC = None

# Optional: SciPy KDTree for MOD28C3 reprojection
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

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
        "agg": "mean",                    # monthly mean
        "res_m": 500,
        "varname_out": "SC",
    },
    "GPP": {
        "glob": os.path.join(BASE, "GPP", "*", "MOD17A2HGF.*.h*v*.061.*.hdf"),
        "sds_candidates": ["Gpp_500m", "Gpp", "GPP"],
        "scale": 1e-4,                    # typical MOD17 scale (kg C m-2 per 8-day)
        "fill_values": [65535, 32767],
        "valid_range": (0, 65000),
        "agg": "sum",                     # monthly sum of 8-day periods
        "res_m": 500,
        "varname_out": "GPP",            # units after scale: kg C m-2 per month
    },
    "FPAR": {
        "glob": os.path.join(BASE, "FPAR", "*", "MOD15A2H.*.h*v*.061.*.hdf"),
        "sds_candidates": ["Fpar_500m", "Fpar"],
        "scale": 0.01,                    # 0..1 after scale
        "fill_values": [255],
        "valid_range": (0, 100),
        "agg": "mean",                    # monthly mean
        "res_m": 500,
        "varname_out": "FPAR",
    },
    "LAI": {
        "glob": os.path.join(BASE, "FPAR", "*", "MOD15A2H.*.h*v*.061.*.hdf"),
        "sds_candidates": ["Lai_500m", "Lai"],
        "scale": 0.1,                     # LAI 0..10 after scale
        "fill_values": [255],
        "valid_range": (0, 100),
        "agg": "mean",                    # monthly mean
        "res_m": 500,
        "varname_out": "LAI",
    },
    "MOD28C3": {
        "glob": os.path.join(BASE, "water_reservoir_monthly", "MOD28C3.*.hdf"),
        "sds_candidates": ["Water_Reservoir", "Reservoir", "SST", "temperature", "Variable"],
        "scale": None,                    # attempt to read from attributes if present
        "fill_values": None,              # attempt to read from attributes if present
        "valid_range": None,              # attempt to read from attributes if present
        "agg": "native",                 # already monthly
        "res_m": None,
        "varname_out": "MOD28C3",
    }
})

# Default months / years / tiles
DEFAULT_MONTHS = list(range(1, 13))
DEFAULT_TILES = [(h, v) for h in range(0, 36) for v in range(0, 18)]  # superset; we'll filter to what's actually found

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


# --- MODIS sinusoidal tile mapping to lat/lon ---
# Following standard MODIS Sinusoidal grid (WGS84 sphere, R=6371007.181 m)
# Each tile is 10x10 degrees (approx) in sinusoidal; 1km tile size = 1200x1200; 500m tile size = 2400x2400.
# This function returns lat/lon arrays for the center of each pixel in the requested tile at npix resolution.

def geomapping(h: int, v: int, npix: int = 1200):
    R = 6371007.181
    TILES_H = 36
    TILES_V = 18
    TILE_SIZE_M = 1111950.519765  # ~10 degrees at equator in sinusoidal meters
    PIX_M_1KM = 926.6254330558339 # approx 1 km in sinusoidal meters

    # pixel size based on requested npix: if npix=1200 => 1 km; if 2400 => 500 m
    pix_size = TILE_SIZE_M / npix

    # Tile origin (upper-left of tile) in sinusoidal meters
    x0 = - R * math.pi + h * TILE_SIZE_M
    y0 =  R * math.pi - v * TILE_SIZE_M

    # pixel centers
    i = np.arange(npix)
    j = np.arange(npix)
    xx = x0 + (i + 0.5) * pix_size
    yy = y0 - (j + 0.5) * pix_size
    X, Y = np.meshgrid(xx, yy)

    # Inverse sinusoidal projection
    lon = np.degrees(X / (R * np.cos(Y / R)))
    lat = np.degrees(Y / R)

    # Handle numerical issues near poles
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
    """Return (array, attrs, sds_name). Tries candidates in order; if none given, pick first 2D numeric dataset.
       Requires pyhdf.SD.
    """
    if SD is None:
        raise RuntimeError("pyhdf is not available; please install 'pyhdf' (HDF4) to read MODIS HDF.")

    sd = SD(hdf_path, SDC.READ)
    datasets = sd.datasets()
    # Find 2D numeric datasets
    def is_numeric_2d(info):
        rank = info[0]
        dtype = info[3]
        return rank == 2 and ("int" in str(dtype).lower() or "float" in str(dtype).lower())

    choice = None
    if candidates:
        for name in candidates:
            if name in datasets and is_numeric_2d(datasets[name]):
                choice = name
                break
    if choice is None:
        # pick the first 2D numeric dataset
        for name, info in datasets.items():
            if is_numeric_2d(info):
                choice = name
                break
    if choice is None:
        raise RuntimeError(f"No 2D numeric SDS found in {os.path.basename(hdf_path)}. Datasets: {list(datasets.keys())[:8]}...")

    sds = sd.select(choice)
    arr = sds[:].astype(np.float32)
    attrs = sds.attributes()
    sd.end()
    return arr, attrs, choice


def apply_scale_and_mask(arr, attrs, fallback_scale=None, valid_range=None, fill_values=None):
    # Infer fill/missing
    if fill_values is None:
        fill_values = []
        for k in ["_FillValue", "fillvalue", "MissingValue", "missing_value", "_MissingValue"]:
            if k in attrs:
                v = attrs[k]
                if isinstance(v, (list, tuple, np.ndarray)):
                    fill_values.extend(list(v))
                else:
                    fill_values.append(v)
        fill_values = [float(v) for v in fill_values]
    else:
        fill_values = [float(v) for v in fill_values]

    # Infer valid range
    if valid_range is None:
        if "valid_range" in attrs:
            vr = attrs["valid_range"]
            try:
                valid_range = (float(vr[0]), float(vr[1]))
            except Exception:
                valid_range = None
        else:
            valid_range = None

    # Infer scale
    scale = _attr_number(attrs, ["scale_factor", "Scale", "Slope"], default=fallback_scale if fallback_scale is not None else 1.0)
    add_offset = _attr_number(attrs, ["add_offset", "Intercept"], default=0.0)

    out = arr.astype(np.float32)

    # Mask fills
    if fill_values:
        mask_fill = np.isin(out, np.array(fill_values, dtype=out.dtype))
        out[mask_fill] = np.nan

    # Mask valid range
    if valid_range is not None:
        lo, hi = valid_range
        mask_vr = (out < lo) | (out > hi)
        out[mask_vr] = np.nan

    # Scale & offset
    out = out * (scale if scale is not None else 1.0) + (add_offset if add_offset is not None else 0.0)

    return out


# --- Aggregation helpers ---

def nanmean(arrs):
    if not arrs:
        return None
    stack = np.stack(arrs, axis=0)
    return np.nanmean(stack, axis=0)


def nansum(arrs):
    if not arrs:
        return None
    stack = np.stack(arrs, axis=0)
    return np.nansum(stack, axis=0)


def downscale_2x_mean(arr):
    """Downscale by 2x via NaN-aware averaging of 2x2 blocks. Requires even dims."""
    if arr is None:
        return None
    ny, nx = arr.shape
    if ny % 2 != 0 or nx % 2 != 0:
        raise ValueError(f"Array shape {arr.shape} is not even; cannot 2x downscale.")
    a = arr.reshape(ny//2, 2, nx//2, 2)
    with np.errstate(invalid='ignore'):
        out = np.nanmean(np.nanmean(a, axis=3), axis=1)
    return out


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

def process_tile_product(cfg, months=DEFAULT_MONTHS, years=None, tiles=None, to_1km=True, out_root=None):
    varname = cfg["varname_out"]
    glob_pattern = cfg["glob"]
    files = sorted(glob.glob(glob_pattern))
    if not files:
        log(f"No files matched: {glob_pattern}")
        return

    log(f"Found {len(files)} files for {varname}")
    groups, years_found, tiles_found = group_tile_products(files)

    if years is None:
        years = years_found
    else:
        years = [y for y in years if y in years_found]
    if tiles is None or tiles == 'all':
        tiles = tiles_found
    else:
        # filter to those actually present
        tiles = [t for t in tiles if t in tiles_found]

    if out_root is None:
        out_root = os.path.join(BASE, "processed")

    # Determine target npix
    npix_native = 2400 if cfg["res_m"] == 500 else 1200
    npix_target = 1200 if to_1km else npix_native

    # Prepare per-month outputs
    for mm in months:
        mm_name = month_name(mm)
        # Collect tile means per year
        tile_means = []  # list of [tile_label, y1, y2, ...]
        for (h, v) in tiles:
            yearly_grids = []
            for y in years:
                f_list = groups.get((h, v, y, mm), [])
                if not f_list:
                    yearly_grids.append(np.full((npix_target, npix_target), np.nan, dtype=np.float32))
                    continue
                arrs = []
                for f in sorted(f_list):
                    data, attrs, sds_name = read_hdf_sds(f, cfg["sds_candidates"])
                    data = apply_scale_and_mask(
                        data, attrs,
                        fallback_scale=cfg.get("scale"),
                        valid_range=cfg.get("valid_range"),
                        fill_values=cfg.get("fill_values")
                    )
                    # Downscale if needed
                    if to_1km and data.shape[0] == 2400 and data.shape[1] == 2400:
                        data = downscale_2x_mean(data)
                    arrs.append(data)
                # aggregate across 8-day periods in the month
                if cfg["agg"] == "sum":
                    monthly = nansum(arrs)
                else:
                    monthly = nanmean(arrs)
                yearly_grids.append(monthly.astype(np.float32))

            # Stack years -> (nyears, npix, npix)
            stack = np.stack(yearly_grids, axis=0)

            # Output directories
            out_dir_tile = os.path.join(out_root, varname, f"{mm:02d}", f"h{h:02d}v{v:02d}")
            os.makedirs(out_dir_tile, exist_ok=True)

            # Save lat/lon once per tile at target resolution
            lat_path = os.path.join(out_dir_tile, "lat.npy")
            lon_path = os.path.join(out_dir_tile, "lon.npy")
            if not (os.path.exists(lat_path) and os.path.exists(lon_path)):
                lat, lon = geomapping(h, v, npix=npix_target)
                np.save(lat_path, lat)
                np.save(lon_path, lon)

            # Save data
            data_path = os.path.join(out_dir_tile, f"{varname}.npy")
            np.save(data_path, stack)

            # Compute tile means per year
            tile_mean = np.nanmean(stack.reshape(stack.shape[0], -1), axis=1)
            tile_means.append([f"h{h:02d}v{v:02d}"] + list(tile_mean))

        # Save CSV of tile means for this month
        if tile_means:
            cols = ["tile"] + [str(y) for y in years]
            df = pd.DataFrame(tile_means, columns=cols)
            out_dir_month = os.path.join(out_root, varname, f"{mm:02d}")
            os.makedirs(out_dir_month, exist_ok=True)
            csv_path = os.path.join(out_dir_month, f"{varname}_{mm_name}_tile_means.csv")
            df.to_csv(csv_path, index=False)
            log(f"Saved {csv_path}")


# --- MOD28C3 handler ---

def process_mod28c3(cfg, months=DEFAULT_MONTHS, out_root=None, reproject_to_tiles=False, tiles=None, to_1km=True):
    varname = cfg["varname_out"]
    files = sorted(glob.glob(cfg["glob"]))
    if not files:
        log(f"No files matched: {cfg['glob']}")
        return
    if out_root is None:
        out_root = os.path.join(BASE, "processed")

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

    # detect SDS and grid from the first file
    sample_file = files[0]
    sample_arr, sample_attrs, sample_sds = read_hdf_sds(sample_file, cfg["sds_candidates"])
    # try to detect lat/lon datasets
    lat, lon = None, None
    try:
        sd = SD(sample_file, SDC.READ)
        for key in ["latitude", "lat", "Latitude", "YDim"]:
            if key in sd.datasets():
                lat = sd.select(key)[:].astype(np.float32)
                break
        for key in ["longitude", "lon", "Longitude", "XDim"]:
            if key in sd.datasets():
                lon = sd.select(key)[:].astype(np.float32)
                break
        sd.end()
    except Exception:
        pass

    # Output native grid stacks per month (nyears, ny, nx)
    for mm in months:
        stacks = []
        years_in_month = []
        for y in years:
            f_list = groups.get((y, mm), [])
            if not f_list:
                continue
            arrs = []
            for f in sorted(f_list):
                data, attrs, sds_name = read_hdf_sds(f, cfg["sds_candidates"])
                data = apply_scale_and_mask(
                    data, attrs,
                    fallback_scale=cfg.get("scale"),
                    valid_range=cfg.get("valid_range"),
                    fill_values=cfg.get("fill_values")
                )
                arrs.append(data)
            # if multiple files per month exist, average
            monthly = nanmean(arrs)
            stacks.append(monthly.astype(np.float32))
            years_in_month.append(y)

        if not stacks:
            continue

        stack = np.stack(stacks, axis=0)
        out_dir = os.path.join(out_root, varname, f"{mm:02d}")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{varname}.npy"), stack)
        # Save available lat/lon if found (native grid)
        if lat is not None and lon is not None:
            np.save(os.path.join(out_dir, "lat_native.npy"), lat)
            np.save(os.path.join(out_dir, "lon_native.npy"), lon)

        # Save global mean per year
        gmean = np.nanmean(stack.reshape(stack.shape[0], -1), axis=1)
        df = pd.DataFrame({"year": years_in_month, f"{varname}_global_mean": gmean})
        df.to_csv(os.path.join(out_dir, f"{varname}_{month_name(mm)}_global_means.csv"), index=False)
        log(f"Saved MOD28C3 native stacks for month {mm:02d}")

        # Optional: Reproject to MODIS sinusoidal tiles (nearest-neighbor)
        if reproject_to_tiles:
            if KDTree is None:
                log("SciPy not available; skipping MOD28C3 reprojection to tiles.")
            else:
                # Build KDTree from native lat/lon (if lat/lon not provided, approximate from array indices)
                if lat is None or lon is None:
                    ny, nx = stack.shape[1:]
                    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
                    # crude lat/lon assuming regular lat-lon grid
                    lat = (jj / (ny - 1)) * 180.0 - 90.0
                    lon = (ii / (nx - 1)) * 360.0 - 180.0
                pts = np.vstack([lat.ravel(), lon.ravel()]).T
                kdt = KDTree(pts)

                if tiles is None or tiles == 'all':
                    tiles = DEFAULT_TILES

                for (h, v) in tiles:
                    # default to 1 km unless keep_native requested
                    npix = 1200 if to_1km else 2400
                    lat_t, lon_t = geomapping(h, v, npix=npix)
                    qry = np.vstack([lat_t.ravel(), lon_t.ravel()]).T
                    _, idx = kdt.query(qry, k=1)
                    # gather per-year tiles
                    tile_stack = []
                    for k in range(stack.shape[0]):
                        vals = stack[k].ravel()[idx]
                        tile_stack.append(vals.reshape(npix, npix))
                    tile_stack = np.stack(tile_stack, axis=0).astype(np.float32)

                    out_dir_tile = os.path.join(out_root, varname + "_sinu", f"{mm:02d}", f"h{h:02d}v{v:02d}")
                    os.makedirs(out_dir_tile, exist_ok=True)
                    np.save(os.path.join(out_dir_tile, f"{varname}.npy"), tile_stack)
                    np.save(os.path.join(out_dir_tile, "lat.npy"), lat_t)
                    np.save(os.path.join(out_dir_tile, "lon.npy"), lon_t)
                    # Tile mean
                    tile_mean = np.nanmean(tile_stack.reshape(tile_stack.shape[0], -1), axis=1)
                    df_t = pd.DataFrame({"year": years_in_month, f"{varname}_mean": tile_mean})
                    df_t.to_csv(os.path.join(out_dir_tile, f"{varname}_{month_name(mm)}_tile_mean.csv"), index=False)


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
    p = argparse.ArgumentParser(description="Process MODIS tile/global products into monthly stacks consistent with existing pipeline.")
    p.add_argument("--products", type=str, default="SC,GPP,FPAR,LAI,MOD28C3", help="Comma-separated list of products to process.")
    p.add_argument("--months", type=str, default="all", help="Months to process: all | 6-9 | 1,2,3 ...")
    p.add_argument("--years", type=str, default=None, help="Years to restrict to (e.g., 2001-2024 or 2001,2002,2005)")
    p.add_argument("--tiles", type=str, default="all", help="Tiles to restrict, e.g., 'h34v08' or 'h34v08,h35v08' or 'all'")
    p.add_argument("--out_root", type=str, default=None, help="Output root directory. Defaults to <BASE>/processed")
    p.add_argument("--to_1km", action="store_true", help="Downscale 500 m tile products to 1 km (1200x1200). Default if set.")
    p.add_argument("--keep_native", action="store_true", help="Keep native resolution for tile products (e.g., 2400x2400). Overrides --to_1km.")
    p.add_argument("--reproject_mod28", action="store_true", help="Reproject MOD28C3 to MODIS sinusoidal tiles (nearest-neighbor; needs SciPy).")

    args = p.parse_args()

    months = parse_range(args.months, cast=int, all_keyword="all", full=DEFAULT_MONTHS)

    years = None
    if args.years:
        years = parse_range(args.years, cast=int, all_keyword="all", full=None)

    tiles = None
    tiles_arg = args.tiles
    if tiles_arg and tiles_arg.lower() != 'all':
        tiles = []
        for tok in tiles_arg.split(','):
            m = re.match(r"h(\d{2})v(\d{2})", tok.strip())
            if m:
                tiles.append((int(m.group(1)), int(m.group(2))))
        if not tiles:
            tiles = None

    out_root = args.out_root
    to_1km = bool(args.to_1km and not args.keep_native)

    products = [s.strip() for s in args.products.split(',') if s.strip()]

    for prod in products:
        if prod not in CONFIG:
            log(f"Unknown product '{prod}'. Skipping.")
            continue
        cfg = CONFIG[prod]
        if prod == "MOD28C3":
            process_mod28c3(cfg, months=months, out_root=out_root, reproject_to_tiles=args.reproject_mod28, tiles=tiles, to_1km=to_1km)
        else:
            process_tile_product(cfg, months=months, years=years, tiles=tiles, to_1km=to_1km, out_root=out_root)


if __name__ == "__main__":
    main()

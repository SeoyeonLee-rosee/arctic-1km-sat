#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
gpp_max_1.py
------------------
Load monthly GPP (GPP_01.npy ... GPP_12.npy)
→ Fix year-axis alignment differences
→ Save as 12 separate month files:
     gpp_monthly_aligned_M01.npy ... M12.npy
Also save:
     gpp_years.npy
"""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/ 기준
GPP_DIR     = os.path.join(SCRIPT_DIR, "..", "data", "MOD", "GPP")
OUT_DIR     = os.path.join(SCRIPT_DIR, "..", "FIGs", "GPP_MAX")
os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------
# RAW loader with fallback
# ------------------------------------------
def raw_load_4d(path, tiles=34, H=1200, W=1200):
    """Load raw binary .npy via memmap guess."""
    fsize = os.path.getsize(path)
    for dt in (np.float32, np.float64):
        b = np.dtype(dt).itemsize
        denom = tiles * H * W * b
        if denom <= 0: 
            continue
        if fsize % denom != 0:
            continue
        Ny = fsize // denom
        print(f"[RAW fallback] {os.path.basename(path)} → dtype={dt}, shape=({tiles},{Ny},{H},{W})")
        return np.memmap(path, dtype=dt, mode="r", shape=(tiles, Ny, H, W))
    raise RuntimeError(f"Cannot raw-load {path} (size={fsize})")


def safe_load_gpp(path):
    """Try np.load → allow_pickle → raw fallback."""
    try:
        return np.load(path, mmap_mode="r")
    except Exception:
        pass
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        pass
    return raw_load_4d(path)


# ------------------------------------------
# Load single month file
# ------------------------------------------
def load_gpp_month(mm):
    fpath = os.path.join(GPP_DIR, f"GPP_{mm:02d}.npy")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"GPP file missing: {fpath}")
    arr = safe_load_gpp(fpath)

    # infer year indexing
    Ny = arr.shape[1]
    # GPP_03: 2000–2022 → Ny=23
    # GPP_01: 2001–2022 → Ny=22
    # GPP_05: 2000–2021 → Ny=22
    # Make explicit year vector:
    if mm == 3:
        years = np.arange(2000, 2000+Ny)
    elif mm == 1:
        years = np.arange(2001, 2001+Ny)
    elif mm == 2:
        years = np.arange(2001, 2001+Ny)
    elif mm == 4:
        years = np.arange(2000, 2000+Ny)
    else:
        years = np.arange(2000, 2000+Ny)

    return arr.astype(np.float32), years


# ------------------------------------------
# Compute global aligned year axis
# ------------------------------------------
def build_global_year_axis():
    year_sets = []
    for mm in range(1, 13):
        arr, yrs = load_gpp_month(mm)
        year_sets.append(set(yrs))

    all_years = sorted(set.union(*year_sets))
    print(f"🔹 Global GPP year axis = {all_years[0]} ~ {all_years[-1]}  (Ny={len(all_years)})")
    return np.array(all_years, dtype=np.int16)


# ------------------------------------------
# Align each month to global year index
# ------------------------------------------
def align_month(arr, yrs, global_years):
    tiles, Ny_local, H, W = arr.shape
    Ny_global = len(global_years)

    out = np.full((tiles, Ny_global, H, W), np.nan, dtype=np.float32)

    # map local years → index
    year_to_idx = {y: i for i, y in enumerate(yrs)}

    for j, gy in enumerate(global_years):
        if gy in year_to_idx:
            out[:, j] = arr[:, year_to_idx[gy]]
    return out


# ------------------------------------------
# MAIN
# ------------------------------------------
def main():
    print("================================================")
    print("       GPP MAX — PART 1: MONTHLY ALIGNMENT")
    print("================================================")

    print("Building global year index...")
    global_years = build_global_year_axis()
    Ny_global = len(global_years)

    # load lat/lon (unchanged)
    lat = np.load(os.path.join(GPP_DIR, "lat.npy"))
    lon = np.load(os.path.join(GPP_DIR, "lon.npy"))
    np.save(os.path.join(OUT_DIR, "lat.npy"), lat)
    np.save(os.path.join(OUT_DIR, "lon.npy"), lon)
    print("✔ Saved lat/lon")

    # Process each month separately
    print("Aligning each month and saving separately...")
    for mm in range(1, 13):
        arr, yrs = load_gpp_month(mm)
        print(f"GPP_{mm:02d}: shape_in={arr.shape}, yrs={yrs[0]}~{yrs[-1]}")
        aligned = align_month(arr, yrs, global_years)
        out_path = os.path.join(OUT_DIR, f"gpp_monthly_aligned_M{mm:02d}.npy")
        np.save(out_path, aligned)
        print(f"  ✔ Saved: {out_path} (shape={aligned.shape})")

    np.save(os.path.join(OUT_DIR, "gpp_years.npy"), global_years)
    print("✔ Saved gpp_years.npy")
    print("🎉 DONE gpp_max_1.py")


if __name__ == "__main__":
    main()

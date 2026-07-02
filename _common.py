#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
_common.py — code/ 전체에서 공유하는 경로·상수·헬퍼

여러 스크립트에 복사돼 있던 공통 코드를 한 곳으로 모은 모듈.
하위 폴더(code/<sub>/)의 스크립트에서 다음처럼 import 한다:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import _common as C
    # 예) C.MODIS, C.ETOPO_DIR, C.REGIONS, C.EDGES, C.build_etopo_kdtree_for_region(...)

주의:
- 여기 정의된 REGIONS / EDGES 는 **최종 논문(scripts_fin) 기준**이다.
  초기 분석(20_temperature, 21_lst 등)은 다른 지역 박스를 쓰므로 이 값을 쓰지 말 것.
"""
import os
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# =========================================================
# 경로 (ver1/ 기준, __file__ 로 자동 해석 → 위치 독립적)
# =========================================================
HERE = Path(__file__).resolve().parent      # .../ver1/code
VER1 = HERE.parent                           # .../ver1

DATA    = VER1 / "data"
RAW     = VER1 / "RAW"
RAW_AVE = VER1 / "RAW_AVE"
FIG     = VER1 / "FIG"
FIGS    = VER1 / "FIGs"
FIG_FIN = VER1 / "FIG_fin"

MODIS   = DATA / "MODIS"
MOD     = DATA / "MOD"
CRU_DIR = DATA / "CRU"
CRU_NC  = CRU_DIR / "grid" / "cru_ts4.06.1901.2021.tmp.dat.nc"
REVISE  = DATA / "revise" / "data"           # ERA5/CPC/MERRA2/GHCN-CAMS mean/trend

# 외부 아카이브 (ver1 밖)
ETOPO_DIR = "/data1/DATA_ARCHIVE/ETOPO2022/"

# =========================================================
# 연구 지역 (최종 논문 4개 영역)
# =========================================================
@dataclass
class RegionBox:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


REGIONS = [
    RegionBox("Ural",             62.0, 67.0,   58.0,   63.0),
    RegionBox("Central Siberia",  65.0, 70.0,   89.0,   94.0),
    RegionBox("Eastern Siberia",  65.0, 70.0,  125.0,  130.0),
    RegionBox("Northern Alaska",  63.5, 68.5, -155.0, -150.0),
]

# dict 표현이 필요한 스크립트용(build_figure3_datasets 등)
REGIONS_DICT = [
    dict(key="UR", rowname="Ural",            lon=(58.0, 63.0),     lat=(62.0, 67.0)),
    dict(key="CS", rowname="Central Siberia", lon=(89.0, 94.0),     lat=(65.0, 70.0)),
    dict(key="ES", rowname="Eastern Siberia", lon=(125.0, 130.0),   lat=(65.0, 70.0)),
    dict(key="AK", rowname="Northern Alaska", lon=(-155.0, -150.0), lat=(63.5, 68.5)),
]

# =========================================================
# 고도 구간
# =========================================================
EDGES = np.array([0, 500, 1000, 1500, 2000], dtype=float)
BIN_LABELS = ["0–500", "500–1000", "1000–1500", "≥1500"]

# =========================================================
# 작은 유틸
# =========================================================
def to_180(lon):
    """경도를 [-180,180] 범위로."""
    lon = np.asarray(lon)
    return np.where(lon > 180, lon - 360, lon)

def np_load_robust(path):
    """allow_pickle 여부에 견고한 np.load."""
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return np.load(path, allow_pickle=True)

def clip_negative_to_nan(z):
    """음수(해양/무효)를 NaN 처리한 float32 배열."""
    z = np.asarray(z).astype(np.float32, copy=False)
    z[z < 0] = np.nan
    return z

# =========================================================
# 고도 binning
# =========================================================
def collect_values_per_bin(vals, alts, edges=EDGES):
    bins = []
    nbin = len(edges) - 1
    for i in range(nbin):
        a0, a1 = edges[i], edges[i + 1]
        if i < nbin - 1:
            m = (alts >= a0) & (alts < a1) & np.isfinite(vals)
        else:
            m = (alts >= a0) & np.isfinite(vals)
        bins.append(vals[m])
    return bins

def clip_bin_p10_p90(arr):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr
    q10, q90 = np.nanpercentile(arr, [10, 90])
    if not np.isfinite(q10) or not np.isfinite(q90) or q10 >= q90:
        return arr
    return arr[(arr >= q10) & (arr <= q90)]

# =========================================================
# ETOPO 2022 고도 (KDTree 최근접 샘플링)
# =========================================================
def list_etopo_surface_tiles(etopo_dir=ETOPO_DIR):
    pats = [
        os.path.join(etopo_dir, "**", "ETOPO_2022_v1_15s_*_surface.nc"),
        os.path.join(etopo_dir, "**", "*surface.nc"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(files))

def tile_overlaps_region(tile_path, box):
    from netCDF4 import Dataset
    try:
        with Dataset(tile_path) as nc:
            lat = nc.variables["lat"][:]
            lon = nc.variables["lon"][:]
        lon = to_180(lon)
        lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
        lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    except Exception:
        return False
    return not (
        lat_max < box.lat_min or lat_min > box.lat_max or
        lon_max < box.lon_min or lon_min > box.lon_max
    )

def build_etopo_kdtree_for_region(box, etopo_dir=ETOPO_DIR):
    """지역 box 와 겹치는 ETOPO 타일로 KDTree(위·경도)와 고도값 V 반환."""
    from netCDF4 import Dataset
    from scipy.spatial import cKDTree
    all_tiles = list_etopo_surface_tiles(etopo_dir)
    if not all_tiles:
        raise RuntimeError(f"No ETOPO surface tiles under: {etopo_dir}")
    picked = [f for f in all_tiles if tile_overlaps_region(f, box)]
    if not picked:
        raise RuntimeError(f"ETOPO tiles not overlapping region {box.name}.")

    pts_list, z_list = [], []
    for f in picked:
        with Dataset(f) as nc:
            lat = nc.variables["lat"][:]
            lon = nc.variables["lon"][:]
            z = nc.variables["z"][:].astype(np.float32)
        lon = to_180(lon)
        z = clip_negative_to_nan(z)
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        pts_list.append(np.stack([LA.ravel(), LO.ravel()], axis=1))
        z_list.append(z.ravel())

    P = np.concatenate(pts_list, axis=0)
    V = np.concatenate(z_list, axis=0)
    return cKDTree(P), V

def sample_altitude(tree, vals, lat_vec, lon_vec):
    xy = np.stack([lat_vec, lon_vec], axis=1)
    ok = np.all(np.isfinite(xy), axis=1)
    out = np.full(lat_vec.shape[0], np.nan, dtype=np.float32)
    if np.any(ok):
        _, idx = tree.query(xy[ok])
        out[ok] = vals[idx]
    return out

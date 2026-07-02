#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt

ROOT_NATIVE = "../data/warm_yearly_native"
ROOT_REGION = "../data/regions_native_cru"
MODIS_DIR = "../data/MODIS"
OUT_DIR = "./check_regions_out"
os.makedirs(OUT_DIR, exist_ok=True)

REGIONS = [
    dict(key="UR", rowname="Ural",             lon=(58.0, 63.0),      lat=(62.0, 67.0)),
    dict(key="CS", rowname="Central Siberia",  lon=(89.0, 94.0),      lat=(65.0, 70.0)),
    dict(key="ES", rowname="Eastern Siberia",  lon=(125, 130),    lat=(65.0, 70.0)),
    dict(key="AK", rowname="Northern Alaska",  lon=(-155.0, -150.0),  lat=(63.5, 68.5)),
]

def safe_load_npz(path):
    try:
        return np.load(path)
    except zipfile.BadZipFile:
        print(f"[BAD NPZ] {path}")
        return None
    except Exception as e:
        print(f"[LOAD FAIL] {path}: {e}")
        return None

ML = np.load(os.path.join(ROOT_NATIVE, "ML.npy")).astype(np.float32)
GPP = np.load(os.path.join(ROOT_NATIVE, "GPP.npy")).astype(np.float32)
SCE = np.load(os.path.join(ROOT_NATIVE, "SCE_yearly.npy")).astype(np.float32)
YEARS = np.load(os.path.join(ROOT_NATIVE, "YEARS.npy")).astype(np.int32)

LAT = np.load(os.path.join(MODIS_DIR, "lat.npy")).astype(np.float32)
LON = np.load(os.path.join(MODIS_DIR, "lon.npy")).astype(np.float32)
LON = np.where(LON > 180.0, LON - 360.0, LON)

print("ML shape :", ML.shape)
print("GPP shape:", GPP.shape)
print("SCE shape:", SCE.shape)

for reg in REGIONS:
    key = reg["key"]
    print(f"\n===== {key} ({reg['rowname']}) =====")

    mask = (
        np.isfinite(LAT) & np.isfinite(LON) &
        (LAT >= reg["lat"][0]) & (LAT <= reg["lat"][1]) &
        (LON >= reg["lon"][0]) & (LON <= reg["lon"][1])
    )

    npix = np.count_nonzero(mask)
    print("native pixel count:", npix)

    ml_ts = np.nanmean(ML[:, mask], axis=1)
    gpp_ts = np.nanmean(GPP[:, mask], axis=1)
    sce_ts = np.nanmean(SCE[:, mask], axis=1)

    print("ML  mean range :", np.nanmin(ml_ts), np.nanmax(ml_ts))
    print("GPP mean range :", np.nanmin(gpp_ts), np.nanmax(gpp_ts))
    print("SCE mean range :", np.nanmin(sce_ts), np.nanmax(sce_ts))

    plt.figure(figsize=(7, 4))
    plt.plot(YEARS, ml_ts, label="ML")
    plt.plot(YEARS, gpp_ts, label="GPP")
    plt.plot(YEARS, sce_ts, label="SCE")
    plt.legend()
    plt.title(f"{key} native regional mean")
    plt.xlabel("Year")
    plt.savefig(os.path.join(OUT_DIR, f"check_{key}_native_timeseries.png"), dpi=200, bbox_inches="tight")
    plt.close()

    year_indices = [0, len(YEARS)//2, len(YEARS)-1]
    for iy in year_indices:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        fields = [
            np.where(mask, ML[iy], np.nan),
            np.where(mask, GPP[iy], np.nan),
            np.where(mask, SCE[iy], np.nan),
        ]
        titles = [
            f"{key} ML {YEARS[iy]}",
            f"{key} GPP {YEARS[iy]}",
            f"{key} SCE {YEARS[iy]}",
        ]

        for ax, field, title in zip(axes, fields, titles):
            sc = ax.scatter(
                LON[mask], LAT[mask],
                c=field[mask],
                s=0.2, cmap="coolwarm", linewidths=0
            )
            ax.set_xlim(reg["lon"])
            ax.set_ylim(reg["lat"])
            ax.set_title(title)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"check_{key}_native_map_{YEARS[iy]}.png"), dpi=200, bbox_inches="tight")
        plt.close()

    path_cru_m = os.path.join(ROOT_REGION, f"CRU_monthly_{key}.npz")
    path_cru_w = os.path.join(ROOT_REGION, f"CRU_warm_yearly_{key}.npz")
    path_sce_r = os.path.join(ROOT_REGION, f"SCE_yearly_{key}.npz")

    cru_m = safe_load_npz(path_cru_m)
    cru_w = safe_load_npz(path_cru_w)
    sce_r = safe_load_npz(path_sce_r)

    if cru_m is None or cru_w is None or sce_r is None:
        print(f"[SKIP REGION NPZ CHECK] {key}")
        continue

    print("CRU_monthly data shape     :", cru_m["data"].shape)
    print("CRU_warm_yearly data shape :", cru_w["data"].shape)
    print("SCE_yearly data shape      :", sce_r["data"].shape)

    sce_native_reg = SCE[:, mask]
    if sce_native_reg.shape == sce_r["data"].shape:
        diff = np.nanmean(np.abs(sce_native_reg - sce_r["data"]))
        print("mean abs diff native SCE vs saved regional SCE:", diff)
    else:
        print("shape mismatch native SCE vs saved regional SCE:",
              sce_native_reg.shape, sce_r["data"].shape)

    cru_ts = np.nanmean(cru_w["data"], axis=1)

    plt.figure(figsize=(7, 4))
    plt.plot(YEARS, ml_ts, label="ML")
    plt.plot(YEARS, cru_ts, label="CRU")
    plt.plot(YEARS, gpp_ts, label="GPP")
    plt.plot(YEARS, sce_ts, label="SCE")
    plt.legend()
    plt.title(f"{key} regional mean comparison")
    plt.xlabel("Year")
    plt.savefig(os.path.join(OUT_DIR, f"check_{key}_regional_compare.png"), dpi=200, bbox_inches="tight")
    plt.close()

    years_r = cru_w["years"]
    lat_r = cru_w["lat"]
    lon_r = cru_w["lon"]
    cru_warm = cru_w["data"]
    sce_reg = sce_r["data"]

    plt.figure(figsize=(6, 4))
    plt.plot(years_r, np.nanmean(cru_warm, axis=1), label="CRU warm")
    plt.plot(years_r, np.nanmean(sce_reg, axis=1), label="SCE yearly")
    plt.legend()
    plt.title(f"{key} saved region mean")
    plt.xlabel("Year")
    plt.savefig(os.path.join(OUT_DIR, f"check_{key}_region_ts.png"), dpi=200, bbox_inches="tight")
    plt.close()

    for iy in [0, len(years_r)//2, len(years_r)-1]:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        fields = [cru_warm[iy], sce_reg[iy]]
        titles = [f"{key} CRU warm {years_r[iy]}", f"{key} SCE {years_r[iy]}"]

        for ax, field, title in zip(axes, fields, titles):
            sc = ax.scatter(
                lon_r, lat_r,
                c=field,
                s=3, cmap="coolwarm", linewidths=0
            )
            ax.set_title(title)
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"check_{key}_region_map_{years_r[iy]}.png"), dpi=200, bbox_inches="tight")
        plt.close()

print(f"\nSaved outputs in: {OUT_DIR}")

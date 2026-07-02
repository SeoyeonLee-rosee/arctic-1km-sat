#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
figure3_new_vertical.py

Panels
------
a : Regional domains on the native ML 1 km grid, with NH shown for reference
    on the CRU-aligned coarse grid.
b : Same as (a), but restricted to pixels with elevation >= 500 m
    for the regional domains.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from scipy.spatial import cKDTree

# =========================
# Paths
# =========================
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/ 기준

ROOT = os.path.join(SCRIPT_DIR, "..", "data", "figure3")
GRID = os.path.join(ROOT, "grid05")
REG = os.path.join(ROOT, "regions")

ETOPO_DIR = "/data1/DATA_ARCHIVE/ETOPO2022/"

OUT_DIR = os.path.join(SCRIPT_DIR, "FIG_fin")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "FIG3_patterncorr_vertical_elev500m.png")
OUT_PDF = os.path.join(OUT_DIR, "FIG3_patterncorr_vertical_elev500m.pdf")

# =========================
# Style
# =========================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
})

# =========================
# Settings
# =========================
SAT_ADD_CONST = 100.0
ELEV_THRESHOLD = 500.0
ETOPO_PAD_DEG = 0.25

XMIN, XMAX = -0.85, -0.30
YMIN, YMAX = 0.60, 0.95

XTICKS = np.arange(-0.8, -0.29, 0.1)
YTICKS = np.arange(0.60, 0.951, 0.1)

YEAR_ALPHA = 0.20
YEAR_SIZE = 28
MEAN_SIZE = 95

LABEL_FS = 11
TICK_FS = 11
LEGEND_FS = 11
PANEL_FS = 18

GRID_COLOR = "0.85"
GRID_LW = 0.8

BOX_ASPECT = (YMAX - YMIN) / (XMAX - XMIN)

LABEL_OFFSET_BROAD = {
    "NH": (0.000, -0.012),
}

LABEL_OFFSET_REG = {
    "UR": (0.000, -0.012),
    "CS": (0.000, -0.012),
    "ES": (0.000, -0.012),
    "AK": (0.000, -0.012),
}

BROAD_DOMAIN = dict(
    key="NH",
    label="NH",
    lat=(60.0, 70.0),
    lon=(-180.0, 180.0),
    color="0.35"
)

REGIONS = [
    dict(key="UR", label="UR", color="#2b7bba", lon=(58.0, 63.0),      lat=(62.0, 67.0)),
    dict(key="CS", label="CS", color="#33a02c", lon=(89.0, 94.0),      lat=(65.0, 70.0)),
    dict(key="ES", label="ES", color="#ff7f00", lon=(125.0, 130.0),    lat=(65.0, 70.0)),
    dict(key="AK", label="AK", color="#de2d26", lon=(-155.0, -150.0),  lat=(63.5, 68.5)),
]

# =========================
# Basic helpers
# =========================
def corr_1d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(m) < 3:
        return np.nan

    aa = a[m].copy()
    bb = b[m].copy()

    aa -= aa.mean()
    bb -= bb.mean()

    da = np.sqrt(np.sum(aa * aa))
    db = np.sqrt(np.sum(bb * bb))
    if da == 0.0 or db == 0.0:
        return np.nan

    return float(np.sum(aa * bb) / (da * db))


def safe_nanmean(arr, axis=0):
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=axis)


def broad_mask_05(lat2d, lon2d, dom):
    return (
        np.isfinite(lat2d) & np.isfinite(lon2d) &
        (lat2d >= dom["lat"][0]) & (lat2d <= dom["lat"][1]) &
        (lon2d >= dom["lon"][0]) & (lon2d <= dom["lon"][1])
    )


def remove_greenland(lat2d, lon2d):
    return ~(
        (lat2d >= 60.0) & (lat2d <= 85.0) &
        (lon2d >= -75.0) & (lon2d <= -10.0)
    )


def calc_yearly_corrs_against_clim(sat_yearly, gpp_yearly, sce_yearly):
    sce_clim = safe_nanmean(sce_yearly, axis=0)
    gpp_clim = safe_nanmean(gpp_yearly, axis=0)

    ny = sat_yearly.shape[0]
    xs = np.full(ny, np.nan, dtype=np.float32)
    ys = np.full(ny, np.nan, dtype=np.float32)

    for iy in range(ny):
        xs[iy] = corr_1d(sat_yearly[iy] + SAT_ADD_CONST, sce_clim)
        ys[iy] = corr_1d(sat_yearly[iy] + SAT_ADD_CONST, gpp_clim)

    return xs, ys


def calc_meanfield_corr(sat_yearly, gpp_yearly, sce_yearly):
    sat_clim = safe_nanmean(sat_yearly, axis=0)
    gpp_clim = safe_nanmean(gpp_yearly, axis=0)
    sce_clim = safe_nanmean(sce_yearly, axis=0)

    x = corr_1d(sat_clim + SAT_ADD_CONST, sce_clim)
    y = corr_1d(sat_clim + SAT_ADD_CONST, gpp_clim)
    return x, y

# =========================
# ETOPO helpers
# =========================
def _lat_str(d):
    return f"N{abs(int(d)) // 15 * 15:02d}" if d >= 0 else f"S{abs(int(d)) // 15 * 15:02d}"

def _lon_str(d):
    return f"E{abs(int(d)) // 15 * 15:03d}" if d >= 0 else f"W{abs(int(d)) // 15 * 15:03d}"

def get_etopo_tile_names(lat_range, lon_range):
    lat_vals = range(int(np.floor(lat_range[0] / 15)) * 15,
                     int(np.ceil(lat_range[1] / 15)) * 15 + 15, 15)
    lon_vals = range(int(np.floor(lon_range[0] / 15)) * 15,
                     int(np.ceil(lon_range[1] / 15)) * 15 + 15, 15)
    return [
        f"ETOPO_2022_v1_15s_{_lat_str(la)}{_lon_str(lo)}_surface.nc"
        for la in lat_vals for lo in lon_vals
    ]

def load_etopo_tiles(tile_dir, names):
    lat_all, lon_all, z_all = [], [], []
    for nm in names:
        p = os.path.join(tile_dir, nm)
        if not os.path.exists(p):
            continue
        with Dataset(p) as nc:
            lat = nc.variables["lat"][:].astype(np.float64)
            lon = nc.variables["lon"][:].astype(np.float64)
            lon = np.where(lon > 180, lon - 360, lon)
            z = nc.variables["z"][:].astype(np.float32)
            z[z < 0] = np.nan
            lat_all.append(lat)
            lon_all.append(lon)
            z_all.append(z)
    return lat_all, lon_all, z_all

def build_kdtree(lat_arrs, lon_arrs, z_arrs):
    pts, vals = [], []
    for la, lo, zz in zip(lat_arrs, lon_arrs, z_arrs):
        LA, LO = np.meshgrid(la, lo, indexing="ij")
        pts.append(np.stack([LA.ravel(), LO.ravel()], axis=1))
        vals.append(zz.ravel())
    P = np.concatenate(pts, axis=0)
    V = np.concatenate(vals, axis=0)
    return cKDTree(P), V

ETOPO_CACHE = {}

def get_etopo_tree(lat_range, lon_range):
    key = (tuple(np.round(lat_range, 3)), tuple(np.round(lon_range, 3)))
    if key in ETOPO_CACHE:
        return ETOPO_CACHE[key]

    names = get_etopo_tile_names(lat_range, lon_range)
    la_e, lo_e, zz_e = load_etopo_tiles(ETOPO_DIR, names)
    if len(la_e) == 0:
        ETOPO_CACHE[key] = (None, None)
        return None, None

    tree, zvals = build_kdtree(la_e, lo_e, zz_e)
    ETOPO_CACHE[key] = (tree, zvals)
    return tree, zvals

def etopo_sample_points(latp, lonp, lat_range, lon_range):
    tree, zvals = get_etopo_tree(lat_range, lon_range)
    if tree is None:
        return np.full(latp.shape[0], np.nan, dtype=np.float32)

    xy = np.stack([latp.astype(np.float64), lonp.astype(np.float64)], axis=1)
    ok = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    out = np.full(latp.shape[0], np.nan, dtype=np.float32)
    if np.any(ok):
        _, idx = tree.query(xy[ok], k=1)
        out[ok] = zvals[idx].astype(np.float32)
    return out

# =========================
# Plot helpers
# =========================
def plot_years_and_mean(ax, xs, ys, xm, ym, color, label="", filled=True, dx=0.0, dy=-0.012):
    good = np.isfinite(xs) & np.isfinite(ys)

    ax.scatter(
        xs[good], ys[good],
        s=YEAR_SIZE,
        facecolor=color if filled else "white",
        edgecolor=color,
        linewidth=0.8,
        alpha=YEAR_ALPHA,
        zorder=1
    )

    if np.isfinite(xm) and np.isfinite(ym):
        ax.scatter(
            [xm], [ym],
            s=MEAN_SIZE,
            facecolor=color if filled else "white",
            edgecolor=color,
            linewidth=1.5,
            zorder=4
        )

        if label:
            ax.text(
                xm + dx, ym + dy, label,
                color=color,
                ha="center", va="top",
                fontsize=LABEL_FS,
                fontweight="bold",
                zorder=5
            )

def style_axis(ax, hide_top_xlabels=False):
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LW)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=TICK_FS)
    if hide_top_xlabels:
        ax.tick_params(axis="x", labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_box_aspect(BOX_ASPECT)

# =========================
# Regional panel worker
# =========================
def load_regional_inputs(reg_key):
    ml = np.load(os.path.join(REG, f"ML_warm_yearly_{reg_key}.npz"))
    gpp = np.load(os.path.join(REG, f"GPP_warm_yearly_{reg_key}.npz"))
    sce = np.load(os.path.join(REG, f"SCE_yearly_{reg_key}.npz"))
    cru = np.load(os.path.join(REG, f"CRU_warm_yearly_{reg_key}.npz"))

    MLr = ml["data"].astype(np.float32)
    GPPr = gpp["data"].astype(np.float32)
    SCEr = sce["data"].astype(np.float32)
    CRUr = cru["data"].astype(np.float32)

    lat = ml["lat"].astype(np.float32)
    lon = ml["lon"].astype(np.float32)

    return MLr, GPPr, SCEr, CRUr, lat, lon

def build_high_elev_mask(lat, lon, reg):
    lat_range = (reg["lat"][0] - ETOPO_PAD_DEG, reg["lat"][1] + ETOPO_PAD_DEG)
    lon_range = (reg["lon"][0] - ETOPO_PAD_DEG, reg["lon"][1] + ETOPO_PAD_DEG)

    elev = etopo_sample_points(lat, lon, lat_range, lon_range)
    mask_hi = np.isfinite(elev) & (elev >= ELEV_THRESHOLD)
    return mask_hi

def add_nh_to_panel(ax, ML_W, GPP_W, CRU_W, SCE, LAT, LON):
    dom = BROAD_DOMAIN
    m = broad_mask_05(LAT, LON, dom)
    m &= remove_greenland(LAT, LON)

    sat_ml = ML_W[:, m]
    sat_cr = CRU_W[:, m]
    gpp = GPP_W[:, m]
    sce = SCE[:, m]

    x_ml_year, y_ml_year = calc_yearly_corrs_against_clim(sat_ml, gpp, sce)
    x_cr_year, y_cr_year = calc_yearly_corrs_against_clim(sat_cr, gpp, sce)

    x_ml_mean, y_ml_mean = calc_meanfield_corr(sat_ml, gpp, sce)
    x_cr_mean, y_cr_mean = calc_meanfield_corr(sat_cr, gpp, sce)

    dx, dy = LABEL_OFFSET_BROAD["NH"]

    plot_years_and_mean(
        ax, x_ml_year, y_ml_year, x_ml_mean, y_ml_mean,
        dom["color"], dom["label"], filled=True, dx=dx, dy=dy
    )
    plot_years_and_mean(
        ax, x_cr_year, y_cr_year, x_cr_mean, y_cr_mean,
        dom["color"], "", filled=False
    )

def plot_regional_panel(ax, ML_W, GPP_W, CRU_W, SCE, LAT, LON,
                        include_nh=False, high_elev_only=False, hide_top_xlabels=False):
    if include_nh:
        add_nh_to_panel(ax, ML_W, GPP_W, CRU_W, SCE, LAT, LON)

    for reg in REGIONS:
        MLr, GPPr, SCEr, CRUr, lat, lon = load_regional_inputs(reg["key"])

        if high_elev_only:
            mask_hi = build_high_elev_mask(lat, lon, reg)
            MLr = MLr[:, mask_hi]
            GPPr = GPPr[:, mask_hi]
            SCEr = SCEr[:, mask_hi]
            CRUr = CRUr[:, mask_hi]

        x_ml_year, y_ml_year = calc_yearly_corrs_against_clim(MLr, GPPr, SCEr)
        x_cr_year, y_cr_year = calc_yearly_corrs_against_clim(CRUr, GPPr, SCEr)

        x_ml_mean, y_ml_mean = calc_meanfield_corr(MLr, GPPr, SCEr)
        x_cr_mean, y_cr_mean = calc_meanfield_corr(CRUr, GPPr, SCEr)

        dx, dy = LABEL_OFFSET_REG[reg["key"]]

        plot_years_and_mean(
            ax, x_ml_year, y_ml_year, x_ml_mean, y_ml_mean,
            reg["color"], reg["label"], filled=True, dx=dx, dy=dy
        )
        plot_years_and_mean(
            ax, x_cr_year, y_cr_year, x_cr_mean, y_cr_mean,
            reg["color"], "", filled=False
        )

    style_axis(ax, hide_top_xlabels=hide_top_xlabels)

# =========================
# Main
# =========================
def main():
    print("SCRIPT_DIR =", SCRIPT_DIR)
    print("GRID =", GRID)
    print("REG =", REG)
    print("ETOPO_DIR =", ETOPO_DIR)

    ML = np.load(os.path.join(GRID, "ML.npy")).astype(np.float32)
    GPP = np.load(os.path.join(GRID, "GPP.npy")).astype(np.float32)
    CRU = np.load(os.path.join(GRID, "CRU.npy")).astype(np.float32)
    SCE = np.load(os.path.join(GRID, "SCE_yearly.npy")).astype(np.float32)
    LAT = np.load(os.path.join(GRID, "LAT.npy")).astype(np.float32)
    LON = np.load(os.path.join(GRID, "LON.npy")).astype(np.float32)

    ML_W = safe_nanmean(ML[:, 3:10], axis=1)
    GPP_W = safe_nanmean(GPP[:, 3:10], axis=1)
    CRU_W = safe_nanmean(CRU[:, 3:10], axis=1)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(7.6, 9.6),
        sharex=True,
        gridspec_kw={"hspace": 0.06}
    )
    ax_top, ax_bot = axes

    plot_regional_panel(
        ax_top,
        ML_W=ML_W, GPP_W=GPP_W, CRU_W=CRU_W, SCE=SCE, LAT=LAT, LON=LON,
        include_nh=True,
        high_elev_only=False,
        hide_top_xlabels=True
    )

    plot_regional_panel(
        ax_bot,
        ML_W=ML_W, GPP_W=GPP_W, CRU_W=CRU_W, SCE=SCE, LAT=LAT, LON=LON,
        include_nh=False,
        high_elev_only=True,
        hide_top_xlabels=False
    )

    # panel labels
    ax_top.text(0.02, 0.98, "a", transform=ax_top.transAxes, ha="left", va="top",
                fontsize=PANEL_FS, fontweight="bold")
    ax_bot.text(0.02, 0.98, "b", transform=ax_bot.transAxes, ha="left", va="top",
                fontsize=PANEL_FS, fontweight="bold")

    # b panel annotation
    ax_bot.text(
        0.03, 0.06, ">= 500 m",
        transform=ax_bot.transAxes,
        ha="left", va="bottom",
        fontsize=11
    )

    # y-axis title only once, centered between a and b
    fig.supylabel("SAT–GPP pattern correlation", fontsize=15, x=0.04)

    # x-axis title only for b
    ax_bot.set_xlabel("SAT–SCE pattern correlation", fontsize=15)

    # legend in top panel
    ax_top.scatter([], [], s=YEAR_SIZE, facecolor="0.45", edgecolor="0.45",
                   alpha=YEAR_ALPHA, label="Individual years")
    ax_top.scatter([], [], s=MEAN_SIZE, facecolor="0.20", edgecolor="0.20",
                   linewidth=1.5, label="ML")
    ax_top.scatter([], [], s=MEAN_SIZE, facecolor="white", edgecolor="0.20",
                   linewidth=1.5, label="CRU")
    ax_top.legend(frameon=False, loc="upper right", fontsize=LEGEND_FS)

    plt.tight_layout(rect=[0.06, 0.04, 1.0, 1.0])

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:")
    print(OUT_PNG)
    print(OUT_PDF)

if __name__ == "__main__":
    main()
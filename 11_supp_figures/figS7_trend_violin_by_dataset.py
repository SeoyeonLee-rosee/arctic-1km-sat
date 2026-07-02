#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Supplementary Fig. 7
Altitudinal distributions of warm-season SAT trends across multiple gridded datasets

Panels:
    a) Ural
    b) Central Siberia
    c) Eastern Siberia
    d) Northern Alaska

Warm season:
    Apr-Oct (4-10)

Datasets:
    ERA5, MERRA2, CPC, GHCN_CAMS

Input files expected in each dataset directory:
    lat.npy
    lon.npy
    trend.npy   or trend_gr.npy

Monthly trend array shape:
    (12, lat, lon)

Trend unit:
    assumed to be °C/year -> converted to °C/decade
"""

import os
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from dataclasses import dataclass
from netCDF4 import Dataset
from scipy.spatial import cKDTree

# =========================================================
# Paths
# =========================================================
ROOT = "../data/revise/data"
ETOPO_DIR = "/data1/DATA_ARCHIVE/ETOPO2022/"
OUTDIR = "../FIG_fin"
os.makedirs(OUTDIR, exist_ok=True)

OUT_PNG = os.path.join(OUTDIR, "FigS7_warm_trend_altitude_violin_4regions_refined.png")
OUT_PDF = os.path.join(OUTDIR, "FigS7_warm_trend_altitude_violin_4regions_refined.pdf")
OUT_CSV = os.path.join(OUTDIR, "FigS7_warm_trend_altitude_violin_4regions_refined_stats.csv")

# =========================================================
# Dataset settings
# =========================================================
DATASETS = [
    ("ERA5", "era5", "#d67fc4"),
    ("MERRA2", "merra", "#a9a9a9"),
    ("CPC", "cpc", "#b9b51b"),
    ("GHCN_CAMS", "GHLCN_CAMS", "#56a7b1"),
]

WARM = [4, 5, 6, 7, 8, 9, 10]
TREND_IS_PER_YEAR = True

# =========================================================
# Altitude bins
# =========================================================
EDGES = np.array([0, 500, 1000, 1500, 2000], dtype=float)
BIN_LABELS = ["0–500", "500–1000", "1000–1500", "≥1500"]
BIN_CENTERS = np.arange(len(BIN_LABELS), dtype=float)

# =========================================================
# Regions
# =========================================================
@dataclass
class RegionBox:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

REGIONS = [
    RegionBox("Ural", 62.0, 67.0, 58.0, 63.0),
    RegionBox("Central Siberia", 65.0, 70.0, 89.0, 94.0),
    RegionBox("Eastern Siberia", 65.0, 70.0, 125, 130),
    RegionBox("Northern Alaska", 63.5, 68.5, -155.0, -150.0),
]

# =========================================================
# Style
# =========================================================
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

VIOLIN_ALPHA = 0.90
VIOLIN_EDGE = "0.20"
VIOLIN_LW = 1.0
MEDIAN_LW = 1.9
IQR_LW = 1.2
COUNT_FS = 8.5

# violin width controls
GROUP_WIDTH = 0.78
WIDTH_MIN_FRAC = 0.25
WIDTH_MAX_FRAC = 0.92

# =========================================================
# General helpers
# =========================================================
def log(msg):
    print(f"[FigS7] {msg}")

def _to_180(lon):
    lon = np.asarray(lon)
    return np.where(lon > 180, lon - 360, lon)

def np_load_robust(path):
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return np.load(path, allow_pickle=True)

# =========================================================
# Dataset loader
# =========================================================
def load_dataset_fields(ds_folder):
    base = os.path.join(ROOT, ds_folder)

    lat_fp = os.path.join(base, "lat.npy")
    lon_fp = os.path.join(base, "lon.npy")
    if not os.path.exists(lat_fp) or not os.path.exists(lon_fp):
        raise FileNotFoundError(f"Missing lat/lon in {base}")

    lat2d = np_load_robust(lat_fp).astype(np.float32)
    lon2d = np_load_robust(lon_fp).astype(np.float32)
    lon2d = _to_180(lon2d)

    trend_fp = os.path.join(base, "trend.npy")
    trend_gr_fp = os.path.join(base, "trend_gr.npy")

    if os.path.exists(trend_fp):
        trend12 = np_load_robust(trend_fp).astype(np.float32)
    elif os.path.exists(trend_gr_fp):
        trend12 = np_load_robust(trend_gr_fp).astype(np.float32)
    else:
        raise FileNotFoundError(f"No trend.npy or trend_gr.npy found in {base}")

    return lat2d, lon2d, trend12

# =========================================================
# Seasonal trend
# =========================================================
def season_trend_from_12(trend12, months):
    idx = [m - 1 for m in months]
    out = np.nanmean(trend12[idx, :, :], axis=0).astype(np.float32)
    if TREND_IS_PER_YEAR:
        out = out * 10.0
    return out

# =========================================================
# ETOPO helpers
# =========================================================
def _lat_str(d):
    return f"N{abs(int(d))//15*15:02d}" if d >= 0 else f"S{abs(int(d))//15*15:02d}"

def _lon_str(d):
    return f"E{abs(int(d))//15*15:03d}" if d >= 0 else f"W{abs(int(d))//15*15:03d}"

def get_etopo_tile_names(lat_range, lon_range):
    lat_vals = range(int(lat_range[0]) // 15 * 15, int(lat_range[1]) + 15, 15)
    lon_vals = range(int(lon_range[0]) // 15 * 15, int(lon_range[1]) + 15, 15)
    out = []
    for la in lat_vals:
        for lo in lon_vals:
            out.append(f"ETOPO_2022_v1_15s_{_lat_str(la)}{_lon_str(lo)}_surface.nc")
    return out

def load_etopo_tiles(tile_dir, names):
    lat_all, lon_all, z_all = [], [], []
    for nm in names:
        fp = os.path.join(tile_dir, nm)
        if not os.path.exists(fp):
            log(f"Missing ETOPO tile: {fp}")
            continue
        with Dataset(fp) as nc:
            lat = nc.variables["lat"][:]
            lon = nc.variables["lon"][:]
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

def sample_etopo_vec(tree, vals, lat_vec, lon_vec):
    xy = np.stack([lat_vec, lon_vec], axis=1)
    ok = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    out = np.full(lat_vec.shape[0], np.nan, dtype=np.float32)
    if np.any(ok):
        _, idx = tree.query(xy[ok])
        out[ok] = vals[idx]
    return out

# =========================================================
# Region subset + altitude binning
# =========================================================
def subset_region_values(lat2d, lon2d, val2d, box, etopo_tree, etopo_vals):
    mask = (
        np.isfinite(val2d) &
        (lat2d >= box.lat_min) & (lat2d <= box.lat_max) &
        (lon2d >= box.lon_min) & (lon2d <= box.lon_max)
    )
    if not np.any(mask):
        return None, None

    lats = lat2d[mask].ravel()
    lons = lon2d[mask].ravel()
    vals = val2d[mask].ravel()
    alts = sample_etopo_vec(etopo_tree, etopo_vals, lats, lons)

    ok = np.isfinite(vals) & np.isfinite(alts)
    if not np.any(ok):
        return None, None

    return vals[ok], alts[ok]

def collect_values_per_bin(vals, alts, edges):
    bins = []
    nb = len(edges) - 1
    for i in range(nb):
        a0, a1 = edges[i], edges[i + 1]
        if i < nb - 1:
            m = (alts >= a0) & (alts < a1) & np.isfinite(vals)
        else:
            m = (alts >= a0) & np.isfinite(vals)
        bins.append(vals[m])
    return bins

def compute_panel_ylim(all_bins_by_dataset):
    vals = []
    for bins in all_bins_by_dataset:
        if bins is None:
            continue
        for arr in bins:
            if arr is None:
                continue
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                vals.append(arr)

    if not vals:
        return -0.5, 0.5

    cat = np.concatenate(vals)
    vmin = float(np.nanmin(cat))
    vmax = float(np.nanmax(cat))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return vmin - 0.1, vmax + 0.1

    span = vmax - vmin
    pad = max(0.06 * span, 0.025)
    return vmin - pad, vmax + pad

# =========================================================
# Violin plotting
# =========================================================
def get_global_max_count(all_bins_by_dataset):
    max_n = 1
    for bins_by_dataset in all_bins_by_dataset:
        if bins_by_dataset is None:
            continue
        for bins in bins_by_dataset:
            if bins is None:
                continue
            for arr in bins:
                if arr is None:
                    continue
                arr = arr[np.isfinite(arr)]
                if arr.size > max_n:
                    max_n = arr.size
    return max_n

def draw_grouped_violins(ax, bins_by_dataset, colors, global_max_n):
    nds = len(bins_by_dataset)
    nbin = len(BIN_LABELS)
    one_width = GROUP_WIDTH / nds

    for ib in range(nbin):
        for j in range(nds):
            arr = None if bins_by_dataset[j] is None else bins_by_dataset[j][ib]
            if arr is None or arr.size == 0:
                continue

            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue

            xpos = BIN_CENTERS[ib] - GROUP_WIDTH / 2 + (j + 0.5) * one_width

            width_frac = WIDTH_MIN_FRAC + (WIDTH_MAX_FRAC - WIDTH_MIN_FRAC) * (arr.size / global_max_n)
            width_frac = min(max(width_frac, WIDTH_MIN_FRAC), WIDTH_MAX_FRAC)
            violin_width = one_width * width_frac

            vp = ax.violinplot(
                [arr],
                positions=[xpos],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            for body in vp["bodies"]:
                body.set_facecolor(colors[j])
                body.set_edgecolor(VIOLIN_EDGE)
                body.set_alpha(VIOLIN_ALPHA)
                body.set_linewidth(VIOLIN_LW)

            q25, q50, q75 = np.nanpercentile(arr, [25, 50, 75])
            x0 = xpos - violin_width * 0.28
            x1 = xpos + violin_width * 0.28

            ax.hlines(q25, x0, x1, color="black", lw=IQR_LW)
            ax.hlines(q50, x0, x1, color="black", lw=MEDIAN_LW)
            ax.hlines(q75, x0, x1, color="black", lw=IQR_LW)

            ax.text(
                xpos, q50, f"{arr.size}",
                ha="center", va="center",
                fontsize=COUNT_FS, color="white",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="black", alpha=0.85)]
            )

def make_stats_rows(region_name, bins_by_dataset, dataset_labels):
    rows = []
    for ib, alt_label in enumerate(BIN_LABELS):
        for ds_name, bins in zip(dataset_labels, bins_by_dataset):
            arr = None if bins is None else bins[ib]
            if arr is None or arr.size == 0:
                n = 0
                q25 = q50 = q75 = np.nan
            else:
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    n = 0
                    q25 = q50 = q75 = np.nan
                else:
                    n = int(arr.size)
                    q25, q50, q75 = np.nanpercentile(arr, [25, 50, 75])

            rows.append([
                region_name,
                alt_label,
                ds_name,
                n,
                q25,
                q50,
                q75,
            ])
    return rows

# =========================================================
# Main
# =========================================================
def main():
    log("Loading datasets ...")

    dataset_labels = []
    dataset_colors = []
    seasonal_trend = {}
    data_lat = {}
    data_lon = {}

    for label, folder, color in DATASETS:
        lat2d, lon2d, trend12 = load_dataset_fields(folder)
        tr = season_trend_from_12(trend12, WARM)

        dataset_labels.append(label)
        dataset_colors.append(color)
        seasonal_trend[label] = tr
        data_lat[label] = lat2d
        data_lon[label] = lon2d

        log(f"{label:10s} loaded: lat/lon={lat2d.shape}, trend12={trend12.shape}")

    fig, axes = plt.subplots(
        nrows=4, ncols=1,
        figsize=(8.9, 13.4),
        sharex=True
    )

    panel_tags = ["a", "b", "c", "d"]
    all_stats = []
    all_bins_for_width = []

    # first pass to compute bins and max N
    region_bins_list = []
    for region in REGIONS:
        log(f"Processing region: {region.name}")

        etopo_names = get_etopo_tile_names(
            (region.lat_min, region.lat_max),
            (region.lon_min, region.lon_max)
        )
        la_e, lo_e, zz_e = load_etopo_tiles(ETOPO_DIR, etopo_names)
        if not la_e:
            raise RuntimeError(f"No ETOPO tiles found for region {region.name}")

        tree_z, zvals = build_kdtree(la_e, lo_e, zz_e)

        bins_by_dataset = []
        for ds_name in dataset_labels:
            vals, alts = subset_region_values(
                data_lat[ds_name],
                data_lon[ds_name],
                seasonal_trend[ds_name],
                region,
                tree_z,
                zvals
            )
            if vals is None:
                bins_by_dataset.append(None)
            else:
                bins_by_dataset.append(collect_values_per_bin(vals, alts, EDGES))

        region_bins_list.append(bins_by_dataset)
        all_bins_for_width.append(bins_by_dataset)

    global_max_n = get_global_max_count(all_bins_for_width)

    for ax, region, ptag, bins_by_dataset in zip(axes, REGIONS, panel_tags, region_bins_list):
        ymin, ymax = compute_panel_ylim(bins_by_dataset)

        ax.set_xlim(-0.5, len(BIN_LABELS) - 0.5)
        ax.set_ylim(ymin, ymax)
        ax.grid(False)

        draw_grouped_violins(ax, bins_by_dataset, dataset_colors, global_max_n)

        ax.text(
            -0.070, 0.50, region.name,
            transform=ax.transAxes,
            rotation=90,
            ha="center", va="center",
            fontsize=16, fontweight="bold"
        )

        ax.text(
            0.010, 0.020, ptag,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=17, fontweight="bold"
        )

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="y", labelsize=10.5)

        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.9)

        all_stats.extend(make_stats_rows(region.name, bins_by_dataset, dataset_labels))

    axes[-1].set_xticks(BIN_CENTERS)
    axes[-1].set_xticklabels(BIN_LABELS, fontsize=15)
    axes[-1].set_xlabel("Altitude [m]", fontsize=15, fontweight="normal")

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    axes[0].text(
        -0.005, 1.015, "[°C 10-year$^{-1}$]",
        transform=axes[0].transAxes,
        ha="left", va="bottom",
        fontsize=11.5, fontweight="normal"
    )

    legend_handles = [
        Line2D([0], [0], color=c, lw=8, label=lab)
        for lab, c in zip(dataset_labels, dataset_colors)
    ]

    # legend inside panel a, above ≥1500 bin, low-y region
    axes[0].legend(
        handles=legend_handles,
        title="SAT (Apr–Oct)",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.86, 0.28),
        fontsize=10.5,
        title_fontsize=11.5,
        handlelength=1.8,
        handletextpad=0.6,
        borderaxespad=0.0
    )

    fig.subplots_adjust(
        left=0.12, right=0.88, top=0.965, bottom=0.08, hspace=0.08
    )

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF, dpi=300)
    plt.close(fig)

    log(f"Saved figure: {OUT_PNG}")
    log(f"Saved figure: {OUT_PDF}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Region", "AltitudeBin", "Dataset", "N", "Q25", "Median", "Q75"])
        for row in all_stats:
            out = row[:4] + [
                "" if not np.isfinite(row[4]) else f"{row[4]:.4f}",
                "" if not np.isfinite(row[5]) else f"{row[5]:.4f}",
                "" if not np.isfinite(row[6]) else f"{row[6]:.4f}",
            ]
            w.writerow(out)

    log(f"Saved stats: {OUT_CSV}")
    log("Done.")

if __name__ == "__main__":
    main()
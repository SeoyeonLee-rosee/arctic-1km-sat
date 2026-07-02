#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
Supplementary figure: mean difference map + vertical half-violin
Layout: 4 rows × 4 columns

For each dataset row:
    warm map | warm half-violin | cold map | cold half-violin

Rows:
    ERA5, MERRA2, CPC, GHCN-CAMS

Difference:
    ML - dataset  [°C]

Map color scale:
    discrete bounds from BD_BOUNDS
    colorbar tick labels from CB_TICKS

Violin:
    warm: -5 to 5 °C
    cold: -10 to 10 °C
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from netCDF4 import Dataset

# ================= PATHS =================
BASE = "../data/revise/data"
DATASETS = ["era5", "merra", "cpc", "GHLCN_CAMS"]
DS_LABEL = {
    "era5": "ERA5",
    "merra": "MERRA2",
    "cpc": "CPC",
    "GHLCN_CAMS": "GHCN-CAMS",
}

ML_AVE = "../data/MODIS/ave_0.5.npy"
CRU_NC = "../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc"

OUTDIR = "../FIG_fin"
os.makedirs(OUTDIR, exist_ok=True)
OUT_PNG = os.path.join(OUTDIR, "Figs3_sup_mean_map_halfviolin_4x4.png")
OUT_PDF = os.path.join(OUTDIR, "Figs3_sup_mean_map_halfviolin_4x4.pdf")

# ================= SETTINGS =================
WARM = [4, 5, 6, 7, 8, 9, 10]
COLD = [11, 12, 1, 2, 3]

CMAP_DIFF = "bwr"

# map color bins
BD_BOUNDS = np.array([
    -10.0, -7.0,
    -5.0, -4.0, -3.0, -2.0, -1.0,
     0.0,
     1.0,  2.0,  3.0,  4.0,  5.0,
     7.0, 10.0
], dtype=float)

# colorbar labels only
CB_TICKS = np.array([-10, -7, -5, -3, -1, 1, 3, 5, 7, 10], dtype=float)

# violin axis ranges
VIOLIN_RANGE_WARM = (-5.0, 5.0)
VIOLIN_RANGE_COLD = (-10.0, 10.0)

FIG_DPI = 300

# vertical half-violin style
VIOLIN_FACE = "0.55"
VIOLIN_ALPHA = 0.22
VIOLIN_LINE = "0.35"
DOT_COLOR = "black"
MEAN_TEXT_COLOR = "black"

# ================= HELPERS =================
def log(msg):
    print(f"[fig_sup_map_halfviolin] {msg}")

def month_to_index(mm):
    return int(mm) - 1

def season_mean(arr12, months):
    idx = [month_to_index(m) for m in months]
    return np.nanmean(arr12[idx, ...], axis=0).astype(np.float32)

def centers_to_edges_1d(c1d):
    c = np.asarray(c1d, dtype=float)
    if c.size < 2:
        raise ValueError("Need >=2 centers to build edges.")
    e = np.empty(c.size + 1, dtype=float)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - 0.5 * (c[1] - c[0])
    e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return e

def load_latlon_from_cru(lat_min=60.0):
    with Dataset(CRU_NC) as nc:
        lat_all = nc.variables["lat"][:].astype(float)
        lon_all = nc.variables["lon"][:].astype(float)
    lon_all = np.where(lon_all > 180, lon_all - 360, lon_all)
    latmask = lat_all >= lat_min
    lat60 = lat_all[latmask]
    return lat60.astype(np.float32), lon_all.astype(np.float32), latmask

def ensure_lat_subset(arr12, latmask, nlat_expected):
    if arr12.shape[1] == nlat_expected:
        return arr12
    if arr12.shape[1] == latmask.size:
        return arr12[:, latmask, :]
    raise ValueError(f"Unexpected lat dim: {arr12.shape[1]}")

def finite_1d(x):
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]

def quantiles(x, qs=(0.05, 0.25, 0.75, 0.95)):
    x = finite_1d(x)
    if x.size == 0:
        return {q: np.nan for q in qs}
    return {q: float(np.quantile(x, q)) for q in qs}

def gaussian_kernel_1d(sigma_bins, radius):
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x**2) / (2 * sigma_bins**2))
    k /= k.sum()
    return k

def smooth_hist_density(x, xgrid, bins=320, sigma_bins=3.0):
    x = finite_1d(x)
    if x.size == 0:
        return np.zeros_like(xgrid)

    xmin, xmax = float(xgrid.min()), float(xgrid.max())
    hist, edges = np.histogram(x, bins=bins, range=(xmin, xmax), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    radius = int(max(3, np.ceil(3 * sigma_bins)))
    k = gaussian_kernel_1d(sigma_bins=sigma_bins, radius=radius)
    hist_s = np.convolve(hist, k, mode="same")

    return np.interp(xgrid, centers, hist_s, left=0.0, right=0.0)

def polar_ax(ax):
    ax.set_extent([0, 359, 60, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, lw=0.4)

    theta = np.linspace(0, 2*np.pi, 256)
    circle = mpath.Path(np.vstack([np.cos(theta), np.sin(theta)]).T * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")

def shade_greenland(ax, color="0.95", zorder=1):
    shp = shpreader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries"
    )
    for rec in shpreader.Reader(shp).records():
        name = rec.attributes.get("NAME_LONG") or rec.attributes.get("NAME")
        if name == "Greenland":
            ax.add_geometries(
                [rec.geometry],
                crs=ccrs.PlateCarree(),
                facecolor=color,
                edgecolor="none",
                zorder=zorder
            )

def add_polar_labels_outside(ax):
    lat_color = "0.6"
    fs_lat = 7.5
    ax.text(0.50, 0.02, "60°N", color=lat_color, fontsize=fs_lat,
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.50, 0.19, "70°N", color=lat_color, fontsize=fs_lat,
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.50, 0.36, "80°N", color=lat_color, fontsize=fs_lat,
            ha="center", va="center", transform=ax.transAxes)

    lon_color = "black"
    fs_lon = 9.0
    ax.text(0.50, 1.01, "180°", color=lon_color, fontsize=fs_lon,
            ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.50, -0.01, "0°", color=lon_color, fontsize=fs_lon,
            ha="center", va="top", transform=ax.transAxes)

    y120 = 0.78
    y60 = 0.23
    ax.text(-0.05, y120, "120°W", color=lon_color, fontsize=fs_lon,
            ha="left", va="center", transform=ax.transAxes)
    ax.text(-0.04, y60, "60°W", color=lon_color, fontsize=fs_lon,
            ha="left", va="center", transform=ax.transAxes)
    ax.text(1.05, y120, "120°E", color=lon_color, fontsize=fs_lon,
            ha="right", va="center", transform=ax.transAxes)
    ax.text(1.03, y60, "60°E", color=lon_color, fontsize=fs_lon,
            ha="right", va="center", transform=ax.transAxes)

def add_panel_tag(ax, tag):
    ax.text(
        0.02, 0.98, tag,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=15, fontweight="bold"
    )

def pcolormesh_latlon(ax, lon1d, lat1d, fld2d, cmap, norm):
    lon_e = centers_to_edges_1d(lon1d)
    lat_e = centers_to_edges_1d(lat1d)
    pm = ax.pcolormesh(
        lon_e, lat_e, fld2d,
        cmap=cmap, norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        rasterized=True
    )
    return pm

def add_colorbar_under_axis(fig, ax_ref, norm, cmap, ticks, yshift=-0.10):
    cax = ax_ref.inset_axes([0.06, yshift, 0.88, 0.045])

    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])

    cb = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        boundaries=BD_BOUNDS,
        ticks=ticks,
        spacing="proportional"
    )
    cb.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    cb.ax.tick_params(labelsize=9)
    return cb

def draw_half_violin_vertical(ax, data, ylim, yticks):
    d = finite_1d(data)
    q = quantiles(d)

    ygrid = np.linspace(ylim[0], ylim[1], 900)
    dens = smooth_hist_density(d, ygrid, bins=320, sigma_bins=3.0)

    x0 = 0.0
    max_halfwidth = 0.42

    if np.nanmax(dens) > 0:
        w = dens / np.nanmax(dens) * max_halfwidth
        ax.fill_betweenx(
            ygrid, x0, x0 + w,
            color=VIOLIN_FACE, alpha=VIOLIN_ALPHA, linewidth=0
        )
        ax.plot(x0 + w, ygrid, color=VIOLIN_LINE, lw=1.0)

    # 5–95%
    ax.plot([x0, x0], [q[0.05], q[0.95]], color="0.25", lw=1.2, zorder=5)
    # IQR
    ax.plot([x0, x0], [q[0.25], q[0.75]], color="0.10", lw=4.5,
            solid_capstyle="round", zorder=6)

    mu = float(np.nanmean(d)) if d.size else np.nan
    if np.isfinite(mu):
        ax.plot([x0], [mu], marker="o", ms=6.5, color=DOT_COLOR, zorder=7)
        ax.text(
            x0 + 0.08, mu, f"{mu:+.2f}",
            ha="left", va="center", fontsize=8.5, color=MEAN_TEXT_COLOR
        )

    ax.axhline(0.0, color="0.4", lw=1.0, ls="--", zorder=1)

    ax.set_xlim(-0.03, 0.47)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.grid(True, axis="y", alpha=0.25, ls="--", lw=0.5)
    ax.tick_params(axis="y", labelsize=10, length=3)

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

# ================= CORE =================
def draw_mean_map_halfviolin_4x4():
    lat60, lon, latmask = load_latlon_from_cru(lat_min=60.0)
    nlat_expected = lat60.size

    ML12 = np.load(ML_AVE).astype(np.float32)
    ML12 = ensure_lat_subset(ML12, latmask, nlat_expected)

    ML_w = season_mean(ML12, WARM)
    ML_c = season_mean(ML12, COLD)

    rng_w = VIOLIN_RANGE_WARM
    rng_c = VIOLIN_RANGE_COLD

    yticks_w = [-5, -3, -1, 0, 1, 3, 5]
    yticks_c = [-10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10]

    norm_map = BoundaryNorm(BD_BOUNDS, ncolors=plt.get_cmap(CMAP_DIFF).N, clip=True)
    ticks_map = CB_TICKS

    proj = ccrs.NorthPolarStereo(central_longitude=0)

    fig = plt.figure(figsize=(13.8, 15.2))
    gs = fig.add_gridspec(
        4, 4,
        width_ratios=[1.72, 0.16, 1.72, 0.16],
        wspace=0.005, hspace=0.12,
        left=0.035, right=0.985, bottom=0.07, top=0.98
    )

    axes = []
    for r in range(4):
        row_axes = []
        for c in range(4):
            if c in [0, 2]:
                ax = fig.add_subplot(gs[r, c], projection=proj)
            else:
                ax = fig.add_subplot(gs[r, c])
            row_axes.append(ax)
        axes.append(row_axes)

    map_tags = ["a", "b", "c", "d", "e", "f", "g", "h"]
    map_tag_idx = 0

    for row, ds in enumerate(DATASETS):
        fp = f"{BASE}/{ds}/ave_gr_0.5.npy"
        DS12 = np.load(fp).astype(np.float32)
        DS12 = ensure_lat_subset(DS12, latmask, nlat_expected)

        DS_w = season_mean(DS12, WARM)
        DS_c = season_mean(DS12, COLD)

        diff_w = (ML_w - DS_w).astype(np.float32)
        diff_c = (ML_c - DS_c).astype(np.float32)

        # warm map
        ax = axes[row][0]
        polar_ax(ax)
        shade_greenland(ax, color="0.95")
        add_polar_labels_outside(ax)
        pcolormesh_latlon(ax, lon, lat60, diff_w, cmap=CMAP_DIFF, norm=norm_map)
        add_panel_tag(ax, map_tags[map_tag_idx])
        map_tag_idx += 1
        ax.text(
            -0.10, 0.50, DS_LABEL[ds],
            transform=ax.transAxes,
            rotation=90,
            ha="center", va="center",
            fontsize=14
        )

        # warm vertical half-violin
        ax = axes[row][1]
        draw_half_violin_vertical(ax, diff_w, ylim=rng_w, yticks=yticks_w)

        # cold map
        ax = axes[row][2]
        polar_ax(ax)
        shade_greenland(ax, color="0.95")
        add_polar_labels_outside(ax)
        pcolormesh_latlon(ax, lon, lat60, diff_c, cmap=CMAP_DIFF, norm=norm_map)
        add_panel_tag(ax, map_tags[map_tag_idx])
        map_tag_idx += 1

        # cold vertical half-violin
        ax = axes[row][3]
        draw_half_violin_vertical(ax, diff_c, ylim=rng_c, yticks=yticks_c)

    add_colorbar_under_axis(
        fig, axes[3][0],
        norm=norm_map, cmap=CMAP_DIFF, ticks=ticks_map, yshift=-0.10
    )
    add_colorbar_under_axis(
        fig, axes[3][2],
        norm=norm_map, cmap=CMAP_DIFF, ticks=ticks_map, yshift=-0.10
    )

    fig.savefig(OUT_PNG, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    log(f"Saved: {OUT_PNG}")
    log(f"Saved: {OUT_PDF}")

def main():
    draw_mean_map_halfviolin_4x4()

if __name__ == "__main__":
    main()
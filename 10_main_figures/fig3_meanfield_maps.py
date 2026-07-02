#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
NC-ready 4x3 mean maps (scatter on MODIS 34-tile 1 km grid)

Rows:
  Ural / Central Siberia / Eastern Siberia / Northern Alaska

Cols:
  SAT (Apr–Oct) / GPP (Apr–Oct) / SCE

Updates applied:
- wider left margin so row labels do not overlap y ticks
- narrower gaps between columns
- slightly larger gaps between rows so titles/ticks do not collide
- column titles moved closer to panels
- horizontal colorbar unit labels aligned at the upper-right end of each bar
- no y tick labels on columns 2 and 3
- one shared horizontal colorbar per column
- forced horizontal compression between columns using manual axis shifts

Additional update:
- if SCE_ave.npy does not exist, build it automatically from SCE_yearly.npy
- robust fallback for both standard .npy and raw float32 yearly stacks
- infer year count from file size when needed
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker


# =========================
# Paths
# =========================
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/ 기준

ML_DIR   = os.path.join(SCRIPT_DIR, "..", "data", "MODIS")
GPP_ROOT = os.path.join(SCRIPT_DIR, "..", "data", "MOD", "GPP")
SCE_ROOT = os.path.join(SCRIPT_DIR, "..", "data", "MOD", "SCE")

OUTDIR = os.path.join(SCRIPT_DIR, "..", "FIG_fin")
os.makedirs(OUTDIR, exist_ok=True)

OUT_PDF = os.path.join(OUTDIR, "FIG3_new_SAT_GPP_SCE_MEAN_4x3_NC_sharedcbar_forcedclose.pdf")
OUT_PNG = os.path.join(OUTDIR, "FIG3_new_SAT_GPP_SCE_MEAN_4x3_NC_sharedcbar_forcedclose.png")


# =========================
# Style
# =========================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,

    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# =========================
# Settings
# =========================
WARM = [4, 5, 6, 7, 8, 9, 10]
LEVEL_IDX = 5

COL_TITLES = ["SAT (Apr–Oct)", "GPP (Apr–Oct)", "SCE"]
UNITS_ABC  = ["[°C]", r"[g C m$^{-2}$ month$^{-1}$]", "[DOY]"]

LETTERS = list("abcdefghijkl")
PANEL_LABEL_FS = 9

FIG_W_INCH = 7.5
FIG_H_INCH = 8.7

LEFT   = 0.120
RIGHT  = 0.975
BOTTOM = 0.120
TOP    = 0.928

# gridspec spacing
WSPACE = 0.06
HSPACE = 0.115

# extra manual compression after axes creation
SHIFT_COL2 = 0.050   # move 2nd column left
SHIFT_COL3 = 0.100   # move 3rd column left

COLTITLE_Y  = 0.933
COLTITLE_FS = 8

ROWLABEL_FS  = 8
ROWLABEL_PAD = 0.055

CB_PAD_Y = 0.052
CB_H     = 0.015

COAST_LW  = 0.35
BORDER_LW = 0.25

SCATTER_S = 1.0
RASTERIZE_SCATTER = True

CMAPS = ["RdYlBu_r", "YlGn", "Blues"]


# =========================
# Regions
# =========================
REGIONS = [
    dict(key="U",  rowname="Ural",             lon=(58.0, 63.0),      lat=(62.0, 67.0)),
    dict(key="CS", rowname="Central Siberia",  lon=(89.0, 94.0),      lat=(65.0, 70.0)),
    dict(key="ES", rowname="Eastern Siberia",  lon=(125, 130),    lat=(65.0, 70.0)),
    dict(key="NA", rowname="Northern Alaska",  lon=(-155.0, -150.0),  lat=(63.5, 68.5)),
]


# =========================
# Utilities
# =========================
def np_load_robust(path):
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return np.load(path, allow_pickle=True)

def _extract_level(arr, level_idx=LEVEL_IDX):
    if arr.ndim == 4:
        return arr[:, level_idx, :, :]
    if arr.ndim == 3:
        if arr.shape[0] > 20:
            return arr
        return arr[level_idx, :, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected ML array shape: {arr.shape}")

def load_ml_latlon():
    lat = np_load_robust(os.path.join(ML_DIR, "lat.npy")).astype(np.float32)
    lon = np_load_robust(os.path.join(ML_DIR, "lon.npy")).astype(np.float32)
    if lat.shape != lon.shape:
        raise ValueError(f"ML lat/lon shape mismatch: {lat.shape} vs {lon.shape}")
    return lat, lon

def load_ml_month(month, fname, ref_shape):
    fp = os.path.join(ML_DIR, f"{month:02d}", fname)
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    arr = np_load_robust(fp)
    out = _extract_level(arr, LEVEL_IDX).astype(np.float32)
    if out.shape != ref_shape:
        raise ValueError(f"ML {fname} {month:02d} shape {out.shape} != {ref_shape}")
    return out

def build_sat_warm_mean(lat_ref):
    ref_shape = lat_ref.shape
    lst = [load_ml_month(m, "ave_gr.npy", ref_shape) for m in WARM]
    arr = np.stack(lst, axis=0)

    if np.isfinite(arr).sum() == 0:
        raise RuntimeError("SAT warm-month arrays contain no finite values.")

    with np.errstate(invalid="ignore"):
        out = np.nanmean(arr, axis=0).astype(np.float32)
    return out

def load_gpp_latlon():
    lat = np_load_robust(os.path.join(GPP_ROOT, "lat.npy")).astype(np.float32)
    lon = np_load_robust(os.path.join(GPP_ROOT, "lon.npy")).astype(np.float32)
    if lat.shape != lon.shape:
        raise ValueError(f"GPP lat/lon shape mismatch: {lat.shape} vs {lon.shape}")
    return lat, lon

def build_gpp_warm_mean(lat_ref):
    lst = []
    for m in WARM:
        fp = os.path.join(GPP_ROOT, f"{m:02d}", "ave.npy")
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        arr = np_load_robust(fp).astype(np.float32)
        if arr.shape != lat_ref.shape:
            raise ValueError(f"GPP ave {m:02d} shape {arr.shape} != {lat_ref.shape}")
        lst.append(arr)

    arr = np.stack(lst, axis=0)
    if np.isfinite(arr).sum() == 0:
        raise RuntimeError("GPP warm-month arrays contain no finite values.")

    with np.errstate(invalid="ignore"):
        gpp = np.nanmean(arr, axis=0).astype(np.float32)

    mask_keep = (gpp <= 0.5)
    gpp[~mask_keep] = np.nan
    gpp = gpp * 1000.0
    return gpp

def load_sce_mean(lat_ref):
    fp_ave = os.path.join(SCE_ROOT, "SCE_ave.npy")
    fp_year = os.path.join(SCE_ROOT, "SCE_yearly.npy")
    fp_years = os.path.join(SCE_ROOT, "years.npy")
    fp_hyears = os.path.join(SCE_ROOT, "hydro_years.npy")

    # ---------------------------
    # case 1: SCE_ave.npy exists
    # ---------------------------
    if os.path.exists(fp_ave):
        try:
            sce = np_load_robust(fp_ave).astype(np.float32)
        except Exception:
            sce = np.fromfile(fp_ave, dtype=np.float32).reshape(lat_ref.shape)

        if sce.shape != lat_ref.shape:
            raise ValueError(f"SCE_ave shape {sce.shape} != {lat_ref.shape}")

        # convert only if still in hydrological DOY
        if np.nanmax(sce) > 220:
            sce = hydro_to_calendar_doy(sce)

        return sce

    # ---------------------------
    # case 2: build from SCE_yearly.npy
    # ---------------------------
    if not os.path.exists(fp_year):
        raise FileNotFoundError(fp_year)

    print("  → SCE_ave.npy not found. Building from SCE_yearly.npy ...")

    if lat_ref.ndim != 3:
        raise ValueError(f"Expected lat_ref to be 3D (tile, y, x), got {lat_ref.shape}")

    ntile, ny, nx = lat_ref.shape
    grid_size = ntile * ny * nx

    years = None
    if os.path.exists(fp_years):
        years = np_load_robust(fp_years).astype(np.int32)
    elif os.path.exists(fp_hyears):
        years = np_load_robust(fp_hyears).astype(np.int32)

    sce_ave = None

    try:
        arr = np.load(fp_year, mmap_mode="r")
        if arr.ndim != 4:
            raise ValueError(f"SCE_yearly npy ndim={arr.ndim}, expected 4")

        if arr.shape[0] != ntile or arr.shape[2] != ny or arr.shape[3] != nx:
            raise ValueError(f"SCE_yearly npy shape {arr.shape} incompatible with {lat_ref.shape}")

        with np.errstate(invalid="ignore"):
            sce_ave = np.nanmean(arr, axis=1).astype(np.float32)

    except Exception:
        pass

    if sce_ave is None:
        nfloat = os.path.getsize(fp_year) // np.dtype("float32").itemsize

        if nfloat % grid_size != 0:
            raise ValueError(
                f"SCE_yearly raw file size incompatible with lat_ref shape: "
                f"nfloat={nfloat}, grid_size={grid_size}"
            )

        nyr_from_file = nfloat // grid_size

        if years is not None and len(years) != nyr_from_file:
            print(
                f"  → warning: years length ({len(years)}) != inferred year count ({nyr_from_file}). "
                f"Using inferred count."
            )

        print(f"  → inferred SCE_yearly shape = ({ntile}, {nyr_from_file}, {ny}, {nx})")

        arr = np.memmap(
            fp_year,
            dtype="float32",
            mode="r",
            shape=(ntile, nyr_from_file, ny, nx)
        )

        with np.errstate(invalid="ignore"):
            sce_ave = np.nanmean(arr, axis=1).astype(np.float32)

    if sce_ave.shape != lat_ref.shape:
        raise ValueError(f"Built SCE_ave shape {sce_ave.shape} != {lat_ref.shape}")

    sce_ave = hydro_to_calendar_doy(sce_ave)

    np.save(fp_ave, sce_ave)
    print("  → saved:", fp_ave)

    return sce_ave

def style_geo(ax, lon_min, lon_max, lat_min, lat_max):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=COAST_LW)
    ax.add_feature(cfeature.BORDERS, linewidth=BORDER_LW)

def set_ticks(ax, lon_min, lon_max, lat_min, lat_max, win_key=None, show_ylabels=True):
    xt0 = int(np.ceil(lon_min / 2) * 2)
    xt1 = int(np.floor(lon_max / 2) * 2)
    xt = np.arange(xt0, xt1 + 1, 2, dtype=int)
    ax.set_xticks(xt, crs=ccrs.PlateCarree())

    if win_key == "NA":
        yt = np.array([64, 66, 68], dtype=int)
    else:
        yt0 = int(np.ceil(lat_min / 2) * 2)
        yt1 = int(np.floor(lat_max / 2) * 2)
        yt = np.arange(yt0, yt1 + 1, 2, dtype=int)
    ax.set_yticks(yt, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter(number_format=".0f"))
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter(number_format=".0f"))
    ax.tick_params(direction="out", labelsize=7, pad=2)

    if not show_ylabels:
        ax.tick_params(axis="y", labelleft=False)
        ax.set_yticklabels([])

def region_mask(lat, lon, lon_min, lon_max, lat_min, lat_max, arr):
    return (
        np.isfinite(arr) &
        np.isfinite(lat) &
        np.isfinite(lon) &
        (lat >= lat_min) & (lat <= lat_max) &
        (lon >= lon_min) & (lon <= lon_max)
    )

def shared_norm_by_variable(arr, lat, lon, regions, p_lo=2, p_hi=98):
    vals = []
    for reg in regions:
        m = region_mask(
            lat, lon,
            reg["lon"][0], reg["lon"][1],
            reg["lat"][0], reg["lat"][1],
            arr
        )
        vv = arr[m]
        vv = vv[np.isfinite(vv)]
        if vv.size > 0:
            vals.append(vv)

    if len(vals) == 0:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = 0.0, 1.0
        return Normalize(vmin=vmin, vmax=vmax)

    allv = np.concatenate(vals)
    if allv.size < 10:
        vmin = float(np.nanmin(allv))
        vmax = float(np.nanmax(allv))
    else:
        vmin, vmax = np.nanpercentile(allv, [p_lo, p_hi])

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(allv))
        vmax = float(np.nanmax(allv))

    return Normalize(vmin=float(vmin), vmax=float(vmax))

def scatter_panel(ax, lat, lon, val, lon_min, lon_max, lat_min, lat_max, norm, cmap, letter):
    style_geo(ax, lon_min, lon_max, lat_min, lat_max)

    la = lat.ravel()
    lo = lon.ravel()
    vv = val.ravel()
    m = (
        np.isfinite(la) & np.isfinite(lo) & np.isfinite(vv) &
        (la >= lat_min) & (la <= lat_max) &
        (lo >= lon_min) & (lo <= lon_max)
    )

    ax.scatter(
        lo[m], la[m], c=vv[m],
        s=SCATTER_S, cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        linewidths=0,
        rasterized=RASTERIZE_SCATTER
    )

    ax.text(
        0.035, 0.03, letter,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=PANEL_LABEL_FS, fontweight="bold", color="black"
    )

def add_shared_bottom_cbar(fig, ref_ax, norm, cmap, unit_text):
    pos = ref_ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - CB_PAD_Y, pos.width, CB_H])

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.locator = MaxNLocator(nbins=5, min_n_ticks=4)
    cb.formatter = FormatStrFormatter("%.0f")
    cb.update_ticks()
    cb.ax.tick_params(labelsize=7, width=0.6, length=3, pad=1.5)
    cb.outline.set_linewidth(0.6)

    cb.ax.text(
        1.0, 1.18, unit_text,
        transform=cb.ax.transAxes,
        ha="right", va="bottom",
        fontsize=6
    )
    return cb

def shift_axis_left(ax, dx):
    pos = ax.get_position()
    ax.set_position([pos.x0 - dx, pos.y0, pos.width, pos.height])

def hydro_to_calendar_doy(arr):
    """
    Convert hydrological DOY (Aug 1 = 1) to calendar DOY (Jan 1 = 1).

    Mapping:
      hydro 1..153   -> cal 213..365   (Aug-Dec)
      hydro 154..366 -> cal   1..213   (Jan-Jul; leap-safe enough for plotting)

    For SCED in this study, most valid values should fall in Feb-Jul,
    so values are typically converted by: cal = hydro - 153.
    """
    arr = np.asarray(arr, dtype=np.float32)
    out = np.full(arr.shape, np.nan, dtype=np.float32)

    m = np.isfinite(arr)
    a = arr[m]

    out_m = np.where(a >= 154.0, a - 153.0, a + 212.0)
    out[m] = out_m
    return out

# =========================
# Main
# =========================
def main():
    lat_ml, lon_ml = load_ml_latlon()
    lat_gpp, lon_gpp = load_gpp_latlon()
    if lat_ml.shape != lat_gpp.shape:
        raise ValueError(f"Grid mismatch: ML {lat_ml.shape} vs GPP {lat_gpp.shape}")

    lat = lat_ml
    lon = lon_ml

    print("• Build SAT warm mean ...")
    SAT = build_sat_warm_mean(lat)

    print("• Build GPP warm mean (×1000) ...")
    GPP = build_gpp_warm_mean(lat)

    print("• Load SCE mean ...")
    SCE = load_sce_mean(lat)

    DATA = [SAT, GPP, SCE]

    NORMS = [
        shared_norm_by_variable(SAT, lat, lon, REGIONS, p_lo=2, p_hi=98),
        shared_norm_by_variable(GPP, lat, lon, REGIONS, p_lo=2, p_hi=98),
        shared_norm_by_variable(SCE, lat, lon, REGIONS, p_lo=2, p_hi=98),
    ]

    fig = plt.figure(figsize=(FIG_W_INCH, FIG_H_INCH))
    gs = fig.add_gridspec(
        4, 3,
        left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP,
        wspace=WSPACE, hspace=HSPACE
    )

    axes = [[None] * 3 for _ in range(len(REGIONS))]
    last_row_axes = [None, None, None]
    li = 0

    for irow, reg in enumerate(REGIONS):
        lon_min, lon_max = reg["lon"]
        lat_min, lat_max = reg["lat"]
        key = reg["key"]

        for jcol in range(3):
            ax = fig.add_subplot(gs[irow, jcol], projection=ccrs.PlateCarree())
            axes[irow][jcol] = ax

            if jcol == 1:
                shift_axis_left(ax, SHIFT_COL2)
            elif jcol == 2:
                shift_axis_left(ax, SHIFT_COL3)

            show_ylabels = (jcol == 0)
            set_ticks(ax, lon_min, lon_max, lat_min, lat_max, win_key=key, show_ylabels=show_ylabels)

            arr = DATA[jcol]
            norm = NORMS[jcol]

            scatter_panel(
                ax, lat, lon, arr,
                lon_min, lon_max, lat_min, lat_max,
                norm, CMAPS[jcol], LETTERS[li]
            )
            li += 1

            if irow == len(REGIONS) - 1:
                last_row_axes[jcol] = ax

    for jcol, title in enumerate(COL_TITLES):
        pos = axes[0][jcol].get_position()
        x_center = 0.5 * (pos.x0 + pos.x1)
        fig.text(
            x_center, COLTITLE_Y, title,
            ha="center", va="bottom",
            fontsize=COLTITLE_FS, fontweight="bold"
        )

    for irow, reg in enumerate(REGIONS):
        pos = axes[irow][0].get_position()
        y_center = 0.5 * (pos.y0 + pos.y1)
        x_left = pos.x0
        fig.text(
            x_left - ROWLABEL_PAD, y_center, reg["rowname"],
            rotation=90, ha="center", va="center",
            fontsize=ROWLABEL_FS, fontweight="bold"
        )

    for jcol in range(3):
        add_shared_bottom_cbar(
            fig=fig,
            ref_ax=last_row_axes[jcol],
            norm=NORMS[jcol],
            cmap=CMAPS[jcol],
            unit_text=UNITS_ABC[jcol]
        )

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=450, bbox_inches="tight")
    plt.close(fig)

    print("✅ Saved:", OUT_PDF)
    print("✅ Saved:", OUT_PNG)


if __name__ == "__main__":
    main()
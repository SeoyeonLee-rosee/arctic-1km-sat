#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import glob
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from dataclasses import dataclass
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


# =========================
# PATHS (EDIT IF NEEDED)
# =========================
PATHS = {
    "ml_root": "../data/MODIS/",
    "cru_nc": "../data/CRU/grid/cru_ts4.06.1901.2021.tmp.dat.nc",
    "gpp_root": "../data/MOD/GPP/",
    "sce_trend": "../data/MOD/SCE/SCE_trend_per_decade.npy",
    "etopo_dir": "/data1/DATA_ARCHIVE/ETOPO2022/",
    "outdir": "../FIG_fin/",
}
os.makedirs(PATHS["outdir"], exist_ok=True)

OUT_PNG = os.path.join(PATHS["outdir"], "Fig4_new_violin_SAT_GPP_SCE_warmseason.png")
OUT_PDF = os.path.join(PATHS["outdir"], "Fig4_new_violin_SAT_GPP_SCE_warmseason.pdf")


# =========================
# SETTINGS
# =========================
WARM_MONTHS = list(range(4, 11))  # Apr–Oct
SAT_LEVEL_IDX = 5
TREND_START_YEAR = 2000
TREND_END_YEAR = 2021

# altitude bins
EDGES = np.array([0, 500, 1000, 1500, 2000], dtype=float)
BIN_LABELS = ["0–500", "500–1000", "1000–1500", "≥1500"]

# violin uses clipped values
CLIP_P10_P90 = True

# GPP scaling and filter
GPP_SCALE = 1000.0
GPP_MAX_KEEP = 12.0

# aesthetics
VIOLIN_ALPHA = 0.90
EDGE_COLOR = "black"
MED_LW = 2.2
IQR_LW = 1.4
SHOW_N = True
N_FONTSIZE = 9

# width scaling by sample size
WIDTH_MIN_FRAC = 0.20
WIDTH_EXP = 0.60
BASE_W_SINGLE = 0.55
BASE_W_SAT_ML = 0.34
BASE_W_SAT_CRU = 0.34

# gridlines
GRID_COLOR = "0.90"
GRID_LW = 0.6
GRID_ALPHA = 0.45

# font / style
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# colors
C_SAT_ML = "C0"
C_SAT_CRU = "C1"
C_GPP = "C5"
C_SCE = "C4"


# =========================
# REGIONS
# =========================
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
    RegionBox("Eastern Siberia",  65.0, 70.0,  125,  130),
    RegionBox("Northern Alaska",  63.5, 68.5, -155.0, -150.0),
]


# =========================
# HELPERS
# =========================
def log(msg: str):
    print(f"[FIG4] {msg}")

def np_load_robust(path):
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return np.load(path, allow_pickle=True)

def _to_180(lon):
    lon = np.asarray(lon)
    return np.where(lon > 180, lon - 360, lon)

def clip_negative_to_nan(z):
    z = z.astype(np.float32, copy=False)
    z[z < 0] = np.nan
    return z

def extract_ml_sat_level(arr, level_idx=SAT_LEVEL_IDX):
    if arr.ndim == 4:
        return arr[:, level_idx, :, :]
    if arr.ndim == 3:
        if arr.shape[0] > 20:
            return arr
        return arr[level_idx, :, :][None, ...]
    if arr.ndim == 2:
        return arr[None, ...]
    raise ValueError(f"Unexpected ML array shape: {arr.shape}")

def collect_values_per_bin(vals, alts, edges):
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

def safe_percentile(arr, ps):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [np.nan for _ in ps]
    return np.nanpercentile(arr, ps)

def per_bin_n(bins):
    return np.array([int(np.isfinite(b).sum()) if b is not None else 0 for b in bins], dtype=int)

def width_from_n(n, nmax, base_width):
    if nmax <= 0 or n <= 0:
        return base_width * WIDTH_MIN_FRAC
    frac = min(n / float(nmax), 1.0)
    scale = WIDTH_MIN_FRAC + (1.0 - WIDTH_MIN_FRAC) * (frac ** WIDTH_EXP)
    return base_width * scale


# =========================
# LOADERS
# =========================
def load_ml_warmseason_trend_per_decade(ml_root, months):
    lat = np_load_robust(os.path.join(ml_root, "lat.npy")).astype(np.float32)
    lon = np_load_robust(os.path.join(ml_root, "lon.npy")).astype(np.float32)
    if lat.shape != lon.shape:
        raise ValueError("ML lat/lon mismatch")

    tstack = []
    for m in months:
        fp = os.path.join(ml_root, f"{m:02d}", "trend_gr.npy")
        if not os.path.exists(fp):
            log(f"  missing ML trend: {fp}")
            continue
        arr = np_load_robust(fp)
        sat = extract_ml_sat_level(arr, SAT_LEVEL_IDX).astype(np.float32)
        tstack.append(sat)

    if len(tstack) == 0:
        trend_dec = np.full_like(lat, np.nan, dtype=np.float32)
    else:
        with np.errstate(invalid="ignore"):
            trend_year = np.nanmean(np.stack(tstack, axis=0), axis=0).astype(np.float32)
        trend_dec = trend_year * 10.0

    return lat, lon, trend_dec

def compute_sat_cru_trend_apr_oct(cru_nc_path, months, y0, y1):
    ds = xr.open_dataset(cru_nc_path)

    cand = ["tmp", "tas", "t2m", "temperature"]
    vname = None
    for c in cand:
        if c in ds.variables:
            vname = c
            break
    if vname is None:
        ds.close()
        raise KeyError("CRU netCDF: cannot find tmp/tas/t2m/temperature variable.")

    T = ds[vname]
    time = T["time"]
    years = time.dt.year.values
    mons = time.dt.month.values

    mask = (years >= y0) & (years <= y1) & np.isin(mons, months)
    if not np.any(mask):
        ds.close()
        raise RuntimeError("CRU: no data found for requested period/months.")

    Tsel = T.sel(time=mask)
    years_sel = Tsel["time"].dt.year.values
    uniq_years = np.unique(years_sel)

    stacks = []
    year_vec = []
    for y in uniq_years:
        my = (years_sel == y)
        if not np.any(my):
            continue
        Ty = Tsel.isel(time=my).mean("time")
        stacks.append(Ty.values.astype(np.float32))
        year_vec.append(float(y))

    stacks = np.stack(stacks, axis=0)
    Ny, nlat, nlon = stacks.shape
    Y = stacks.reshape(Ny, -1)

    x = np.array(year_vec, dtype=np.float32)
    x = x - np.nanmean(x)
    X = x[:, None]
    M = np.isfinite(Y).astype(np.float32)
    n = M.sum(axis=0)

    xw = (M * X).sum(axis=0) / np.maximum(n, 1)
    yw = np.nansum(Y, axis=0) / np.maximum(n, 1)

    dx = X - xw[None, :]
    dy = Y - yw[None, :]

    num = np.nansum(M * dx * dy, axis=0)
    den = np.nansum(M * dx * dx, axis=0)

    slope_y = np.full_like(den, np.nan, dtype=np.float32)
    good = (den > 0) & (n >= 2)
    slope_y[good] = num[good] / den[good]

    trend_dec = (slope_y * 10.0).reshape(nlat, nlon).astype(np.float32)

    lat = ds["lat"].values.astype(np.float32)
    lon = _to_180(ds["lon"].values.astype(np.float32))
    ds.close()

    Lon2d, Lat2d = np.meshgrid(lon, lat, indexing="xy")
    return Lat2d, Lon2d, trend_dec

def load_gpp_warmseason_trend(gpp_root, months):
    lat = np_load_robust(os.path.join(gpp_root, "lat.npy")).astype(np.float32)
    lon = np_load_robust(os.path.join(gpp_root, "lon.npy")).astype(np.float32)
    if lat.shape != lon.shape:
        raise ValueError("GPP lat/lon mismatch")

    tstack = []
    for m in months:
        fp = os.path.join(gpp_root, f"{m:02d}", "trend.npy")
        if not os.path.exists(fp):
            log(f"  missing GPP trend: {fp}")
            continue
        t = np_load_robust(fp).astype(np.float32)
        if t.shape != lat.shape:
            raise ValueError(f"GPP trend shape {t.shape} != lat shape {lat.shape}")
        tstack.append(t)

    if len(tstack) == 0:
        tr = np.full_like(lat, np.nan, dtype=np.float32)
    else:
        with np.errstate(invalid="ignore"):
            tr = np.nanmean(np.stack(tstack, axis=0), axis=0).astype(np.float32)

    tr = tr * GPP_SCALE
    tr = tr.astype(np.float32, copy=False)
    tr[tr > GPP_MAX_KEEP] = np.nan
    return lat, lon, tr

def _ols_slope_per_year(yvec, xvec):
    m = np.isfinite(yvec) & np.isfinite(xvec)
    if m.sum() < 2:
        return np.nan

    x = xvec[m].astype(np.float64)
    y = yvec[m].astype(np.float64)

    x_mean = x.mean()
    y_mean = y.mean()

    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return np.nan

    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    return np.float32(slope)  # days per year

def load_sce_trend_robust(sce_path, expected_shape):
    """
    Load SCE trend [days/decade].
    If SCE_trend_per_decade.npy does not exist, build it from SCE_yearly.npy.

    Supports:
    - standard .npy yearly stack
    - raw float32 memmap yearly stack
    """

    # -------------------------------------------------
    # case 1: trend file already exists
    # -------------------------------------------------
    if os.path.exists(sce_path):
        try:
            arr = np.load(sce_path, allow_pickle=False)
            if arr.shape != expected_shape:
                raise ValueError(f"SCE npy shape {arr.shape} != expected {expected_shape}")
            return arr.astype(np.float32)
        except Exception:
            log("  SCE np.load failed. Trying raw float32 fallback...")
            data = np.fromfile(sce_path, dtype=np.float32)
            need = int(np.prod(expected_shape))
            if data.size != need:
                raise RuntimeError(
                    f"SCE raw size mismatch: got {data.size}, expected {need}. File={sce_path}"
                )
            return data.reshape(expected_shape).astype(np.float32)

    # -------------------------------------------------
    # case 2: trend file missing -> build from yearly
    # -------------------------------------------------
    sce_root = os.path.dirname(sce_path)
    sce_yearly_path = os.path.join(sce_root, "SCE_yearly.npy")
    years_npy_path = os.path.join(sce_root, "years.npy")
    hyears_npy_path = os.path.join(sce_root, "hydro_years.npy")

    if not os.path.exists(sce_yearly_path):
        raise FileNotFoundError(
            f"{sce_path} not found, and source yearly stack also missing: {sce_yearly_path}"
        )

    log("  SCE_trend_per_decade.npy not found. Building from SCE_yearly.npy ...")

    ntile, ny, nx = expected_shape
    grid_size = ntile * ny * nx

    years = None
    if os.path.exists(years_npy_path):
        years = np.load(years_npy_path).astype(np.float32)
        log(f"  using years.npy: {int(years[0])}-{int(years[-1])}")
    elif os.path.exists(hyears_npy_path):
        years = np.load(hyears_npy_path).astype(np.float32)
        log(f"  using hydro_years.npy: {int(years[0])}-{int(years[-1])}")

    sce_year = None
    nyr = None

    # -------------------------------------------------
    # try 1: standard npy yearly stack
    # -------------------------------------------------
    try:
        arr = np.load(sce_yearly_path, mmap_mode="r")
        if arr.ndim != 4:
            raise ValueError(f"SCE_yearly npy ndim={arr.ndim}, expected 4")
        if arr.shape[0] != ntile or arr.shape[2] != ny or arr.shape[3] != nx:
            raise ValueError(f"SCE_yearly npy shape {arr.shape} incompatible with {expected_shape}")

        sce_year = arr
        nyr = arr.shape[1]

        if years is None:
            years = np.arange(TREND_START_YEAR, TREND_START_YEAR + nyr, dtype=np.float32)
            log(f"  years metadata missing. Using inferred years: {int(years[0])}-{int(years[-1])}")
        elif len(years) != nyr:
            log(f"  warning: metadata year count ({len(years)}) != file year count ({nyr}). Using file count.")
            years = np.arange(years[0], years[0] + nyr, dtype=np.float32)

    except Exception:
        sce_year = None

    # -------------------------------------------------
    # try 2: raw float32 yearly stack
    # -------------------------------------------------
    if sce_year is None:
        nfloat = os.path.getsize(sce_yearly_path) // np.dtype("float32").itemsize

        if nfloat % grid_size != 0:
            raise ValueError(
                f"SCE_yearly raw file size incompatible with expected shape: "
                f"nfloat={nfloat}, grid_size={grid_size}"
            )

        nyr = nfloat // grid_size

        if years is None:
            years = np.arange(TREND_START_YEAR, TREND_START_YEAR + nyr, dtype=np.float32)
            log(f"  years metadata missing. Using inferred years: {int(years[0])}-{int(years[-1])}")
        elif len(years) != nyr:
            log(f"  warning: metadata year count ({len(years)}) != inferred year count ({nyr}). Using inferred count.")
            years = np.arange(years[0], years[0] + nyr, dtype=np.float32)

        log(f"  inferred SCE_yearly shape = ({ntile}, {nyr}, {ny}, {nx})")

        sce_year = np.memmap(
            sce_yearly_path,
            dtype="float32",
            mode="r",
            shape=(ntile, nyr, ny, nx)
        )

    # -------------------------------------------------
    # build trend
    # -------------------------------------------------
    out = np.full((ntile, ny, nx), np.nan, dtype=np.float32)

    block_rows = 50
    for ti in range(ntile):
        log(f"  building SCE trend: tile {ti + 1}/{ntile}")
        for r0 in range(0, ny, block_rows):
            r1 = min(ny, r0 + block_rows)

            block = np.asarray(sce_year[ti, :, r0:r1, :], dtype=np.float32)  # (year, rows, nx)
            b_rows = r1 - r0
            flat = block.reshape(nyr, b_rows * nx)

            trend_flat = np.full(b_rows * nx, np.nan, dtype=np.float32)
            for j in range(b_rows * nx):
                slope_per_year = _ols_slope_per_year(flat[:, j], years)
                if np.isfinite(slope_per_year):
                    trend_flat[j] = slope_per_year * 10.0  # days per decade

            out[ti, r0:r1, :] = trend_flat.reshape(b_rows, nx)

    np.save(sce_path, out)
    log(f"  saved: {sce_path}")

    return out


# =========================
# ETOPO KDTree
# =========================
def list_etopo_surface_tiles(etopo_dir):
    pats = [
        os.path.join(etopo_dir, "**", "ETOPO_2022_v1_15s_*_surface.nc"),
        os.path.join(etopo_dir, "**", "*surface.nc"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    return sorted(list(set(files)))

def tile_overlaps_region(tile_path, box: RegionBox):
    try:
        with Dataset(tile_path) as nc:
            lat = nc.variables["lat"][:]
            lon = nc.variables["lon"][:]
        lon = _to_180(lon)
        lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
        lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    except Exception:
        return False
    return not (
        lat_max < box.lat_min or lat_min > box.lat_max or
        lon_max < box.lon_min or lon_min > box.lon_max
    )

def build_etopo_kdtree_for_region(etopo_dir, box: RegionBox):
    all_tiles = list_etopo_surface_tiles(etopo_dir)
    if len(all_tiles) == 0:
        raise RuntimeError(f"No ETOPO surface tiles found under: {etopo_dir}")
    picked = [f for f in all_tiles if tile_overlaps_region(f, box)]
    if len(picked) == 0:
        raise RuntimeError(f"ETOPO tiles not found/overlapping for region {box.name}. Check etopo_dir or coverage.")

    pts_list, z_list = [], []
    for f in picked:
        with Dataset(f) as nc:
            lat = nc.variables["lat"][:]
            lon = nc.variables["lon"][:]
            z = nc.variables["z"][:].astype(np.float32)
        lon = _to_180(lon)
        z = clip_negative_to_nan(z)
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        pts_list.append(np.stack([LA.ravel(), LO.ravel()], axis=1))
        z_list.append(z.ravel())

    P = np.concatenate(pts_list, axis=0)
    V = np.concatenate(z_list, axis=0)
    tree = cKDTree(P)
    return tree, V

def sample_altitude(tree, vals, lat_vec, lon_vec):
    xy = np.stack([lat_vec, lon_vec], axis=1)
    ok = np.all(np.isfinite(xy), axis=1)
    out = np.full(lat_vec.shape[0], np.nan, dtype=np.float32)
    if np.any(ok):
        _, idx = tree.query(xy[ok])
        out[ok] = vals[idx]
    return out

def build_bins_from_field(lat, lon, field, box: RegionBox, etopo_tree, etopo_vals):
    la = np.asarray(lat).ravel()
    lo = np.asarray(lon).ravel()
    vv = np.asarray(field).ravel()

    m = (
        np.isfinite(la) & np.isfinite(lo) & np.isfinite(vv) &
        (la >= box.lat_min) & (la <= box.lat_max) &
        (lo >= box.lon_min) & (lo <= box.lon_max)
    )
    if not np.any(m):
        return [np.array([], dtype=np.float32) for _ in range(len(EDGES) - 1)]

    la = la[m]
    lo = lo[m]
    vv = vv[m]
    alt = sample_altitude(etopo_tree, etopo_vals, la, lo)

    ok = np.isfinite(alt) & np.isfinite(vv)
    vv = vv[ok]
    alt = alt[ok]

    bins = collect_values_per_bin(vv, alt, EDGES)
    if CLIP_P10_P90:
        bins = [clip_bin_p10_p90(b) for b in bins]
    return bins


# =========================
# PLOTTING
# =========================
def set_panel_tag(ax, tag):
    ax.text(
        0.02, 0.98, tag,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13, fontweight="bold",
        path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.9)]
    )

def annotate_n(ax, x, y, n):
    ax.text(
        x, y, f"{n}",
        ha="center", va="center",
        fontsize=N_FONTSIZE, color="white",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="black", alpha=0.95)]
    )

def draw_grouped_violin_sat(ax, bins_ml, bins_cru, show_legend=False):
    nbin = len(EDGES) - 1
    ax.set_xlim(0.5, nbin + 0.5)
    ax.set_xticks(range(1, nbin + 1))
    ax.set_xticklabels(BIN_LABELS)

    n_ml = per_bin_n(bins_ml)
    n_cru = per_bin_n(bins_cru)

    ml_max = int(np.max(n_ml)) if np.any(n_ml > 0) else 0
    cru_max = int(np.max(n_cru)) if np.any(n_cru > 0) else 0
    eps = 1e-12

    for i in range(nbin):
        xc = i + 1
        xL = xc - 0.18
        xR = xc + 0.18

        w_ml = width_from_n(int(n_ml[i]), ml_max, BASE_W_SAT_ML)
        w_cru = width_from_n(int(n_cru[i]), cru_max, BASE_W_SAT_CRU)

        if n_ml[i] > 0 and n_cru[i] > 0:
            ratio = min(n_cru[i] / (n_ml[i] + eps), 1.0)
            w_cru *= (WIDTH_MIN_FRAC + (1 - WIDTH_MIN_FRAC) * (ratio ** WIDTH_EXP))

        for x, arr, col, w in [
            (xL, bins_ml[i], C_SAT_ML, w_ml),
            (xR, bins_cru[i], C_SAT_CRU, w_cru),
        ]:
            arr = np.asarray(arr)
            arr = arr[np.isfinite(arr)]
            if arr.size < 3:
                continue

            vp = ax.violinplot([arr], positions=[x], widths=w,
                               showmeans=False, showmedians=False, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col)
                b.set_edgecolor(EDGE_COLOR)
                b.set_alpha(VIOLIN_ALPHA)

            q25, q50, q75 = safe_percentile(arr, [25, 50, 75])
            ax.hlines(q50, x - 0.14, x + 0.14, colors="k", linewidth=MED_LW)
            ax.hlines([q25, q75], x - 0.12, x + 0.12, colors="k", linewidth=IQR_LW)

            if SHOW_N and np.isfinite(q50):
                annotate_n(ax, x, q50, int(arr.size))

    if show_legend:
        ax.plot([], [], color=C_SAT_ML, lw=8, alpha=0.9, label="ML")
        ax.plot([], [], color=C_SAT_CRU, lw=8, alpha=0.9, label="CRU")
        ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98),
                  fontsize=10, handlelength=1.6, borderaxespad=0.2)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=GRID_LW, alpha=GRID_ALPHA)

def draw_single_violin_scaled(ax, bins, color):
    nbin = len(EDGES) - 1
    ax.set_xlim(0.5, nbin + 0.5)
    ax.set_xticks(range(1, nbin + 1))
    ax.set_xticklabels(BIN_LABELS)

    n = per_bin_n(bins)
    nmax = int(np.max(n)) if np.any(n > 0) else 0

    for i in range(nbin):
        x = i + 1
        arr = np.asarray(bins[i])
        arr = arr[np.isfinite(arr)]
        if arr.size < 3:
            continue

        w = width_from_n(int(arr.size), nmax, BASE_W_SINGLE)

        vp = ax.violinplot([arr], positions=[x], widths=w,
                           showmeans=False, showmedians=False, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(color)
            b.set_edgecolor(EDGE_COLOR)
            b.set_alpha(VIOLIN_ALPHA)

        q25, q50, q75 = safe_percentile(arr, [25, 50, 75])
        ax.hlines(q50, x - 0.20, x + 0.20, colors="k", linewidth=MED_LW)
        ax.hlines([q25, q75], x - 0.16, x + 0.16, colors="k", linewidth=IQR_LW)

        if SHOW_N and np.isfinite(q50):
            annotate_n(ax, x, q50, int(arr.size))

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=GRID_LW, alpha=GRID_ALPHA)


# =========================
# MAIN
# =========================
def main():
    log("Loading ML SAT warm-season trend (Apr–Oct) ...")
    lat_ml, lon_ml, sat_ml_tr = load_ml_warmseason_trend_per_decade(PATHS["ml_root"], WARM_MONTHS)

    log("Loading CRU SAT warm-season trend (Apr–Oct) ...")
    lat_cru, lon_cru, sat_cru_tr = compute_sat_cru_trend_apr_oct(
        PATHS["cru_nc"], WARM_MONTHS, TREND_START_YEAR, TREND_END_YEAR
    )

    log("Loading MODIS lat/lon from GPP ...")
    lat_gpp = np_load_robust(os.path.join(PATHS["gpp_root"], "lat.npy")).astype(np.float32)
    lon_gpp = np_load_robust(os.path.join(PATHS["gpp_root"], "lon.npy")).astype(np.float32)

    log("Loading GPP warm-season trend (Apr–Oct; ×1000) ...")
    _, _, gpp_tr = load_gpp_warmseason_trend(PATHS["gpp_root"], WARM_MONTHS)

    log("Loading SCE trend (per decade) ...")
    sce_tr = load_sce_trend_robust(PATHS["sce_trend"], expected_shape=lat_gpp.shape)

    region_bins = {}
    for box in REGIONS:
        log(f"Building ETOPO KDTree for {box.name} ...")
        tree, zvals = build_etopo_kdtree_for_region(PATHS["etopo_dir"], box)

        sat_ml_bins = build_bins_from_field(lat_ml, lon_ml, sat_ml_tr, box, tree, zvals)
        sat_cru_bins = build_bins_from_field(lat_cru, lon_cru, sat_cru_tr, box, tree, zvals)
        gpp_bins = build_bins_from_field(lat_gpp, lon_gpp, gpp_tr, box, tree, zvals)
        sce_bins = build_bins_from_field(lat_gpp, lon_gpp, sce_tr, box, tree, zvals)

        region_bins[box.name] = {
            "sat_ml": sat_ml_bins,
            "sat_cru": sat_cru_bins,
            "gpp": gpp_bins,
            "sce": sce_bins,
        }

    fig, axes = plt.subplots(4, 3, figsize=(13.5, 12.0))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.085,
                        wspace=0.14, hspace=0.18)

    axes[0, 0].set_title("SAT (Apr–Oct)", fontweight="bold", pad=16)
    axes[0, 1].set_title("GPP (Apr–Oct)", fontweight="bold", pad=16)
    axes[0, 2].set_title("SCE", fontweight="bold", pad=16)

    unit_fs = 8
    axes[0, 0].text(-0.10, 1.01, "[°C 10-year⁻¹]", transform=axes[0, 0].transAxes,
                    ha="left", va="bottom", fontsize=unit_fs, color="0")
    axes[0, 1].text(-0.10, 1.01, "[g C m⁻² month⁻¹ 10-year⁻¹]", transform=axes[0, 1].transAxes,
                    ha="left", va="bottom", fontsize=unit_fs, color="0")
    axes[0, 2].text(-0.10, 1.01, "[day 10-year⁻¹]", transform=axes[0, 2].transAxes,
                    ha="left", va="bottom", fontsize=unit_fs, color="0")

    tags = list("abcdefghijkl")
    k = 0

    for i, box in enumerate(REGIONS):
        name = box.name
        bins = region_bins[name]

        ax = axes[i, 0]
        draw_grouped_violin_sat(ax, bins["sat_ml"], bins["sat_cru"], show_legend=(i == 0))
        ax.text(-0.18, 0.50, name, rotation=90,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, fontweight="bold")
        set_panel_tag(ax, tags[k]); k += 1

        ax = axes[i, 1]
        draw_single_violin_scaled(ax, bins["gpp"], C_GPP)
        set_panel_tag(ax, tags[k]); k += 1

        ax = axes[i, 2]
        draw_single_violin_scaled(ax, bins["sce"], C_SCE)
        set_panel_tag(ax, tags[k]); k += 1

    for i in range(len(REGIONS)):
        for j in range(3):
            axes[i, j].set_ylabel("")

    for i in range(len(REGIONS) - 1):
        for j in range(3):
            axes[i, j].set_xlabel("")
            axes[i, j].set_xticklabels([])

    for j in range(3):
        axes[len(REGIONS) - 1, j].set_xlabel("")

    for i in range(len(REGIONS)):
        axes[i, 1].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        axes[i, 2].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    fig.text(0.5, 0.045, "Altitude [m]", ha="center", va="center",
             fontsize=13, fontweight="bold")

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    log(f"Saved: {OUT_PNG}")
    log(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
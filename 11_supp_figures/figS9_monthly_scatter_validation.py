#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

# =========================================================
# paths
# =========================================================
CSV_FILE = "test_comp.csv"
OUTDIR = "../FIG_fin"
os.makedirs(OUTDIR, exist_ok=True)

OUT_PNG = os.path.join(OUTDIR, "FigS9_test_comp_monthly_scatter.png")
OUT_PDF = os.path.join(OUTDIR, "FigS9_test_comp_monthly_scatter.pdf")

# =========================================================
# style
# =========================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

COL_ML = "#4C72B0"
COL_CRU = "#DD8452"
COL_ONE2ONE = "0.75"
COL_ZERO = "0.55"

MONTH_NAMES = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

# =========================================================
# helpers
# =========================================================
def nice_limits(vals, step=None, pad_frac=0.05):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1, 1

    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    if vmin == vmax:
        return vmin - 1, vmax + 1

    span = vmax - vmin
    pad = span * pad_frac
    lo = vmin - pad
    hi = vmax + pad

    if step is not None:
        lo = np.floor(lo / step) * step
        hi = np.ceil(hi / step) * step

    return lo, hi

def set_clean_axes(ax, xlim, ylim, show_left=False, show_bottom=False, show_right=False):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # light grid
    ax.grid(True, alpha=0.18, linewidth=0.7, zorder=0)

    # darker zero lines
    if xlim[0] < 0 < xlim[1]:
        ax.axvline(0, color=COL_ZERO, lw=1.1, zorder=1)
    if ylim[0] < 0 < ylim[1]:
        ax.axhline(0, color=COL_ZERO, lw=1.1, zorder=1)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.tick_params(length=0)

    if not show_left:
        ax.tick_params(axis="y", labelleft=False)
    if show_right:
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelright=True)

    if not show_bottom:
        ax.tick_params(axis="x", labelbottom=False)

def draw_one_to_one(ax, xlim, ylim):
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], color=COL_ONE2ONE, lw=1.0, zorder=1)

# =========================================================
# read data
# =========================================================
df = pd.read_csv(CSV_FILE)

required_cols = [
    "MONTH", "SAT", "SAT_estimate", "CRU",
    "SAT_t", "SAT_estimate_t", "CRU_t"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# mean temperature limits
mean_vals = np.concatenate([
    df["SAT"].to_numpy(dtype=float),
    df["SAT_estimate"].to_numpy(dtype=float),
    df["CRU"].to_numpy(dtype=float),
])
mean_xlim = nice_limits(mean_vals, step=20, pad_frac=0.03)
mean_ylim = mean_xlim

# trend limits
trend_vals = np.concatenate([
    df["SAT_t"].to_numpy(dtype=float),
    df["SAT_estimate_t"].to_numpy(dtype=float),
    df["CRU_t"].to_numpy(dtype=float),
])
trend_xlim = nice_limits(trend_vals, step=1, pad_frac=0.08)
trend_ylim = trend_xlim

# fixed limits to match example style
mean_xlim = (-55, 20)
mean_ylim = (-55, 30)
trend_xlim = (-5, 5)
trend_ylim = (-5, 5)

# =========================================================
# figure
# =========================================================
fig = plt.figure(figsize=(10.2, 16.2))
outer = fig.add_gridspec(
    nrows=2, ncols=1,
    left=0.08, right=0.94, bottom=0.06, top=0.96,
    hspace=0.12
)

gs_top = outer[0].subgridspec(3, 4, wspace=0.05, hspace=0.12)
gs_bot = outer[1].subgridspec(3, 4, wspace=0.05, hspace=0.12)

axes_top = []
axes_bot = []

panel_tags = list("abcdefghijkl")

# =========================================================
# a) Mean temperature
# =========================================================
for i, month in enumerate(range(1, 13)):
    r, c = divmod(i, 4)
    ax = fig.add_subplot(gs_top[r, c])
    axes_top.append(ax)

    sub = df[df["MONTH"] == month].copy()

    x = sub["SAT"].to_numpy(dtype=float)
    y_ml = sub["SAT_estimate"].to_numpy(dtype=float)
    y_cru = sub["CRU"].to_numpy(dtype=float)

    draw_one_to_one(ax, mean_xlim, mean_ylim)

    ax.scatter(
        x, y_ml,
        s=10, color=COL_ML, alpha=0.85,
        edgecolors="none", label="ML", zorder=3
    )

    ax.scatter(
        x, y_cru,
        s=10, color=COL_CRU, alpha=0.80,
        edgecolors="none", label="CRU", zorder=3
    )

    set_clean_axes(
        ax, mean_xlim, mean_ylim,
        show_left=(c == 0),
        show_bottom=(r == 2),
        show_right=(c == 3)
    )

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    ax.text(
        0.03, 0.97, f"{panel_tags[i]}. {MONTH_NAMES[i]}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10, fontweight="bold"
    )

# =========================================================
# b) Rate of change
# =========================================================
for i, month in enumerate(range(1, 13)):
    r, c = divmod(i, 4)
    ax = fig.add_subplot(gs_bot[r, c])
    axes_bot.append(ax)

    sub = df[df["MONTH"] == month].copy()

    x = sub["SAT_t"].to_numpy(dtype=float)
    y_ml = sub["SAT_estimate_t"].to_numpy(dtype=float)
    y_cru = sub["CRU_t"].to_numpy(dtype=float)

    draw_one_to_one(ax, trend_xlim, trend_ylim)

    ax.scatter(
        x, y_ml,
        s=10, color=COL_ML, alpha=0.85,
        edgecolors="none", label="ML", zorder=3
    )

    ax.scatter(
        x, y_cru,
        s=10, color=COL_CRU, alpha=0.80,
        edgecolors="none", label="CRU", zorder=3
    )

    set_clean_axes(
        ax, trend_xlim, trend_ylim,
        show_left=(c == 0),
        show_bottom=(r == 2),
        show_right=(c == 3)
    )

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    ax.text(
        0.03, 0.97, f"{panel_tags[i]}. {MONTH_NAMES[i]}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10, fontweight="bold"
    )

# =========================================================
# section titles
# =========================================================
top_pos = axes_top[0].get_position()
bot_pos = axes_bot[0].get_position()

fig.text(
    0.03, top_pos.y1 + 0.01,
    "a) Mean temperature",
    ha="left", va="bottom",
    fontsize=18, fontweight="bold"
)

fig.text(
    0.03, bot_pos.y1 + 0.01,
    "b) Rate of change",
    ha="left", va="bottom",
    fontsize=18, fontweight="bold"
)

# =========================================================
# common axis labels
# =========================================================
top_bottom = min(ax.get_position().y0 for ax in axes_top)
top_top = max(ax.get_position().y1 for ax in axes_top)
top_center_y = 0.5 * (top_bottom + top_top)

bot_bottom = min(ax.get_position().y0 for ax in axes_bot)
bot_top = max(ax.get_position().y1 for ax in axes_bot)
bot_center_y = 0.5 * (bot_bottom + bot_top)

fig.text(
    0.015, top_center_y,
    "Estimated SAT [°C]",
    rotation=90, ha="center", va="center",
    fontsize=15
)

fig.text(
    0.015, bot_center_y,
    "Estimated SAT [°C 10-yr$^{-1}$]",
    rotation=90, ha="center", va="center",
    fontsize=15
)

fig.text(
    0.50, top_bottom - 0.02,
    "SAT [°C]",
    ha="center", va="center",
    fontsize=15
)

fig.text(
    0.50, bot_bottom - 0.025,
    "SAT [°C 10-yr$^{-1}$]",
    ha="center", va="center",
    fontsize=15
)

# =========================================================
# legend
# =========================================================
legend_handles = [
    Line2D([0], [0], marker="o", linestyle="None", markersize=6,
           markerfacecolor=COL_ML, markeredgecolor="none", label="ML"),
    Line2D([0], [0], marker="o", linestyle="None", markersize=6,
           markerfacecolor=COL_CRU, markeredgecolor="none", label="CRU"),
    Line2D([0], [0], color=COL_ONE2ONE, lw=1.5, label="y=x"),
]

fig.legend(
    handles=legend_handles,
    loc="upper right",
    bbox_to_anchor=(0.95, 0.975),
    ncol=3,
    frameon=False,
    handletextpad=0.3,
    columnspacing=0.8
)

# =========================================================
# save
# =========================================================
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT_PNG)
print("Saved:", OUT_PDF)
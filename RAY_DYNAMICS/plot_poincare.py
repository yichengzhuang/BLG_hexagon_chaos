#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a Poincare density plot from poincare_points_*.csv."""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = os.environ.get("BLG_RESULTS_DIR", str(_HERE / "result_example"))

# ---------------- Plot style ----------------
plt.rcParams["figure.dpi"] = 160
plt.rcParams["figure.constrained_layout.use"] = True

label_fontsize = 44
tick_fontsize  = 44

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": label_fontsize,
    "axes.linewidth": 1.5,
    "xtick.labelsize": tick_fontsize,
    "ytick.labelsize": tick_fontsize,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"

# ---------------- Inputs / outputs ----------------
CSV_PATH = os.environ.get(
    "BLG_POINCARE_CSV",
    os.path.join(RESULTS_DIR, "poincare", "poincare_points_fourband.csv"),
)
OUT_DIR  = os.environ.get("BLG_FIGURES_DIR", os.path.join(RESULTS_DIR, "figures"))
OUT_NAME = "FIG5(a).pdf"

USE_LOG_BINS = True
GRIDSIZE     = 320

# ---------------- Load and plot ----------------
edge, s01, phi_in_deg, phi_out_deg, alpha_e_deg = np.loadtxt(
    CSV_PATH, delimiter=",", skiprows=1, unpack=True
)
edge = edge.astype(int)

theta_rad = np.deg2rad(phi_out_deg)
p = np.sin(theta_rad)

unique_edges = np.unique(edge)
if len(unique_edges) == 1:
    x = s01
    x_min, x_max = 0.0, 1.0
else:
    x = edge + s01
    x_min, x_max = 0.0, float(np.max(edge) + 1)

fig, ax = plt.subplots(1, 1, figsize=(8, 7))

ax.hexbin(
    x, p,
    gridsize=GRIDSIZE,
    mincnt=1,
    bins="log" if USE_LOG_BINS else None,
    linewidths=0
)

ax.set_xlabel(r"$s$", fontsize=label_fontsize)
ax.set_ylabel(r"$p = \sin\theta$", fontsize=label_fontsize)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1.0, 1.0)
ax.grid(False)

# Shift the -1 tick slightly upward so it doesn't sit on the axis line.
ymin, ymax = ax.get_ylim()
dy = 0.03 * (ymax - ymin)
ax.set_yticks([-1.0 + dy, 0.0, 1.0])
ax.set_yticklabels([r"$-1.0$", r"$0$", r"$1.0$"])

xmin, xmax = ax.get_xlim()
xticks = [xmin]
xlabels = [rf"${xmin:.1f}$"]
if xmin <= 0.0 <= xmax:
    xticks.append(0.0)
    xlabels.append(r"$0$")
xticks.append(xmax)
xlabels.append(rf"${xmax:.1f}$")
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, OUT_NAME)
fig.savefig(out_path, format="pdf", bbox_inches="tight")
plt.close(fig)

print("saved:", out_path)
print("Total Poincare points:", len(x))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treat the reflection map f: phi_in -> phi_out as a 1D circle map.
Plot the local magnification |df/dphi|, locate fixed points (1-cycles)
and 2-cycles, and report their stability multipliers.
"""

import os, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = os.environ.get("BLG_RESULTS_DIR", str(_HERE / "result_example"))

# ---------------- Plot style ----------------
plt.rcParams["figure.dpi"] = 160
plt.rcParams["figure.constrained_layout.use"] = True

label_fontsize  = 44
tick_fontsize   = 44
legend_fontsize = 24

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
NPZ_PATH = os.environ.get(
    "BLG_NPZ_PATH",
    os.path.join(
        RESULTS_DIR, "fourband",
        "E0.2_ang15_ntFS200000_4band_numericFS_360",
        "reflect_table.npz",
    ),
)
OUT_DIR  = os.environ.get("BLG_MAG_DIR", os.path.join(RESULTS_DIR, "calc_mag"))
tag = os.environ.get(
    "BLG_MAG_TAG",
    os.path.basename(os.path.dirname(NPZ_PATH)) or "out",
)

N_SAMPLES = 10000
PLOT_LOG_MAG = False
DO_FIND_2CYCLE = True

# ---------------- Helpers ----------------
def wrap_pi(a):
    a = (np.asarray(a) + np.pi) % (2*np.pi) - np.pi
    if np.ndim(a) == 0:
        a = float(a)
        if a <= -math.pi:
            a += 2*math.pi
        return a
    return np.where(a <= -np.pi, a + 2*np.pi, a)

def _ensure_radians(x):
    """If |x| values look like degrees (>~ pi), convert to radians."""
    x = np.asarray(x, float)
    if not np.isfinite(x).any():
        return x, "rad"
    xmax = float(np.nanmax(np.abs(x)))
    if xmax > (np.pi + 0.05):
        return np.deg2rad(x), "deg->rad"
    return x, "rad"

def load_table(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    meta = z["meta"].item() if ("meta" in z.files) else {}
    if ("phi_in_360" not in z.files) or ("phi_out_360" not in z.files):
        raise RuntimeError("NPZ missing required keys: phi_in_360 / phi_out_360")

    x = np.asarray(z["phi_in_360"], float)
    y = np.asarray(z["phi_out_360"], float)

    x, unit_x = _ensure_radians(x)
    y, unit_y = _ensure_radians(y)
    if unit_x != "rad" or unit_y != "rad":
        print(f"[info] degree-like inputs detected; converted to radians: x({unit_x}), y({unit_y})")

    order = np.argsort(x)
    x = x[order]; y = y[order]
    if len(x) >= 2:
        dx = np.diff(x)
        keep = np.ones_like(x, dtype=bool)
        keep[1:] = np.abs(dx) > 1e-12
        x = x[keep]; y = y[keep]

    return x, y, meta

def build_periodic_linear_interpolator(x, y):
    """Periodic piecewise-linear map f and its slope f'. Uses shortest-arc dy."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("invalid table: size mismatch or too few points")

    order = np.argsort(x); x = x[order]; y = y[order]
    x_all = np.concatenate([x - 2*np.pi, x, x + 2*np.pi])
    y_all = np.concatenate([y,           y, y          ])

    def _segment_index(q):
        j = int(np.searchsorted(x_all, q, side="right"))
        return max(1, min(j, len(x_all) - 1))

    def f(q):
        q = wrap_pi(q)
        q = np.asarray(q, float)
        out = np.empty_like(q, dtype=float)
        q_flat = q.ravel()
        for idx, qq in enumerate(q_flat):
            j0 = _segment_index(qq)
            x1, x2 = x_all[j0-1], x_all[j0]
            y1, y2 = y_all[j0-1], y_all[j0]
            if x2 == x1:
                val = float(wrap_pi(y1))
            else:
                w = (qq - x1) / (x2 - x1)
                dy = wrap_pi(y2 - y1)
                val = wrap_pi(y1 + w*dy)
            out.flat[idx] = val
        return out

    def fprime(q):
        q = wrap_pi(q)
        q = np.asarray(q, float)
        out = np.empty_like(q, dtype=float)
        q_flat = q.ravel()
        for idx, qq in enumerate(q_flat):
            j0 = _segment_index(qq)
            x1, x2 = x_all[j0-1], x_all[j0]
            y1, y2 = y_all[j0-1], y_all[j0]
            if x2 == x1:
                slope = 0.0
            else:
                dy = wrap_pi(y2 - y1)
                slope = float(dy / (x2 - x1))
            out.flat[idx] = slope
        return out

    return f, fprime

def wrapped_diff(a):
    return ((a + np.pi) % (2*np.pi)) - np.pi

def find_fixed_points(f, ngrid=72000, tol=1e-10):
    grid = np.linspace(-np.pi, np.pi, ngrid, endpoint=False)
    h = wrapped_diff(f(grid) - grid)
    roots = []
    for i in range(len(grid)):
        a = grid[i]; b = grid[(i+1) % len(grid)]
        ha = h[i];   hb = h[(i+1) % len(grid)]
        if abs(ha) > np.pi*0.9 or abs(hb) > np.pi*0.9:
            continue
        if ha == 0.0:
            roots.append(a); continue
        if ha*hb > 0:
            continue

        lo, hi = a, b
        for _ in range(60):
            mid = 0.5*(lo+hi)
            hm  = wrapped_diff(f(mid) - mid)
            if abs(hm) < tol or abs(hi-lo) < 1e-12:
                lo = hi = mid; break
            if ha*hm <= 0:
                hi = mid; hb = hm
            else:
                lo = mid; ha = hm
        roots.append(0.5*(lo+hi))

    roots = np.array(sorted(roots))
    keep = []
    for i, r in enumerate(roots):
        if i == 0:
            keep.append(True); continue
        d = abs(wrapped_diff(r - roots[i-1]))
        keep.append(d > math.radians(0.1))
    return roots[np.array(keep)]

def find_2cycle_points(f, ngrid=72000, tol=1e-10, exclude_1cycles=None):
    grid = np.linspace(-np.pi, np.pi, ngrid, endpoint=False)
    h = wrapped_diff(f(f(grid)) - grid)

    roots = []
    for i in range(len(grid)):
        a = grid[i]; b = grid[(i+1) % len(grid)]
        ha = h[i];   hb = h[(i+1) % len(grid)]
        if abs(ha) > np.pi*0.9 or abs(hb) > np.pi*0.9:
            continue
        if ha == 0.0:
            roots.append(a); continue
        if ha*hb > 0:
            continue

        lo, hi = a, b
        for _ in range(60):
            mid = 0.5*(lo+hi)
            hm  = wrapped_diff(f(f(mid)) - mid)
            if abs(hm) < tol or abs(hi-lo) < 1e-12:
                lo = hi = mid; break
            if ha*hm <= 0:
                hi = mid; hb = hm
            else:
                lo = mid; ha = hm
        roots.append(0.5*(lo+hi))

    roots = np.array(sorted(roots))

    if exclude_1cycles is not None and len(exclude_1cycles):
        mask = []
        for r in roots:
            if np.min(np.abs(wrapped_diff(r - exclude_1cycles))) < math.radians(0.2):
                mask.append(False)
            else:
                mask.append(True)
        roots = roots[np.array(mask)]

    keep = []
    for i, r in enumerate(roots):
        if i == 0:
            keep.append(True); continue
        d = abs(wrapped_diff(r - roots[i-1]))
        keep.append(d > math.radians(0.1))
    return roots[np.array(keep)]

# ---------------- Main ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    x, y, meta = load_table(NPZ_PATH)
    f, fprime = build_periodic_linear_interpolator(x, y)

    xs = np.linspace(-np.pi, np.pi, int(N_SAMPLES), endpoint=False)
    ys = f(xs)
    mag = np.abs(fprime(xs))

    eps = 1e-15
    logmag = np.log(mag + eps)
    print("[stats] |f'|: min={:.4g}, max={:.4g}, median={:.4g}, mean={:.4g}".format(
        float(np.min(mag)), float(np.max(mag)),
        float(np.median(mag)), float(np.mean(mag))))
    print("[stats] mean(log|f'|) = {:.4g}".format(float(np.mean(logmag))))

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    phi_in_deg = np.degrees(xs)

    ax.plot(phi_in_deg, mag, "-", lw=3, color="#398BFD",
            label="BLG anisotropic reflection")
    ax.axhline(1.0, color="red", lw=1.2, ls="--", label="Specular mirror")

    ax.legend(loc="upper right", fontsize=legend_fontsize, frameon=False)
    ax.set_xlabel(r"$\phi_{\mathrm{in}}\ (\mathrm{deg})$", fontsize=label_fontsize)
    ax.set_ylabel(r"$\left|d\phi_{\mathrm{out}}/d\phi_{\mathrm{in}}\right|$", fontsize=label_fontsize)
    ax.set_xlim(-180, 180)

    if PLOT_LOG_MAG:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 10)

    fixed = find_fixed_points(f)
    print("\n[fixed points]")
    for phi in fixed:
        slope = float(fprime(phi))
        print("  phi* = {:+.8f} deg,  f'(phi*) = {:+.4f}  -> {}".format(
            math.degrees(phi), slope, "stable" if abs(slope) < 1 else "unstable"
        ))

    if DO_FIND_2CYCLE:
        cyc2 = find_2cycle_points(f, exclude_1cycles=fixed)
        print("\n[2-cycles] (one representative per pair)")
        for phi1 in cyc2:
            phi2 = f(phi1)
            mult = abs(float(fprime(phi1)) * float(fprime(phi2)))
            print("  (phi1, phi2) = ({:+.3f} deg, {:+.3f} deg),  multiplier = {:.4f}  -> {}".format(
                math.degrees(phi1), math.degrees(phi2), mult,
                "stable" if mult < 1 else "unstable"
            ))

    fig.tight_layout()
    pdf_path = os.path.join(OUT_DIR, f"reflect_magnification_{tag}.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", pdf_path)

    npz_out = os.path.join(OUT_DIR, f"reflect_magnification_samples_{tag}.npz")
    np.savez_compressed(
        npz_out,
        phi_in=xs, phi_out=ys, mag=np.asarray(mag, float),
        meta=meta
    )
    print("saved", npz_out)

if __name__ == "__main__":
    main()

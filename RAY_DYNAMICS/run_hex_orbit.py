#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace ray orbits in a hexagonal cavity using a precomputed reflection table.
Outputs a Poincare point cloud and the raw hit sequence (CSV + NPZ).
"""

import os, math, csv
from pathlib import Path
import numpy as np, numpy.linalg as LA
import matplotlib.pyplot as plt
from collections import deque
from typing import List

from orbit_panels import (
    PoincarePanel, FermiSurfacePanel, VKMapper,
    wrap_pi, deg2rad
)

# ---------------- Paths ----------------
_HERE = Path(__file__).resolve().parent
RESULTS_DIR = os.environ.get("BLG_RESULTS_DIR", str(_HERE / "result_example"))

TABLE_ROOT = os.environ.get("BLG_TABLE_ROOT", RESULTS_DIR)
TABLE_TAG  = os.environ.get(
    "BLG_TABLE_TAG",
    "fourband/E0.2_ang15_ntFS200000_4band_numericFS_360",
)
NPZ_NAME   = os.environ.get("BLG_NPZ_NAME",   "reflect_table_sym_ynegx.npz")
NPZ_PATH   = os.path.join(TABLE_ROOT, TABLE_TAG, NPZ_NAME)

# ---------------- Geometry and initial state ----------------
R_HEX   = 2.0
ROT_DEG = 15.0
START_XY = (1.082467979, 0.797380676)
START_ABS_ANGLE_DEG = 33.239138123

# ---------------- Simulation controls ----------------
MAX_BOUNCES       = 100000
STEPS_PER_UNIT    = 5.0
FPS               = 1e10
SHOW_EDGE_LABELS  = True
LOG_FIRST_N       = 500

# Per-edge sign convention for the directed tangent used to define phi_in/phi_out.
EDGE_ALIGN_CHOICE: List[str] = ["flip", "keep", "flip", "keep", "flip", "keep"]
TABLE_IN_SIGN       = +1.0
TABLE_OUT_SIGN      = +1.0
TABLE_IN_SHIFT_DEG  = 0.0
TABLE_OUT_SHIFT_DEG = 0.0

ANGLE_UNIT = "deg"            # "rad" | "deg"
RUN_MODE   = "poincare_only"  # "live" | "poincare_only" | "fast"
LAYOUT     = "two_figure"

# ---------------- Poincare output ----------------
EDGE_TO_PLOT        = 0
ONLY_POINCARE_FIG   = True
POINCARE_SVG_NAME   = f"poincare_edge{EDGE_TO_PLOT}.svg"

POINCARE_SAVE_DIR = os.environ.get("BLG_POINCARE_DIR", os.path.join(RESULTS_DIR, f"poincare_{ROT_DEG}"))
POINCARE_SAVE_TAG = "poincare_points_fourband"
POINCARE_USE_PHI_OUT = True

SHOW_NORMAL_AT_HIT = True
NORMAL_SEG_LEN     = 0.25
TRAJ_MAX_BOUNCES_TO_SHOW = 100

SAVE_HITS = True
HITS_CSV_NAME = "hits_abs_xy.csv"
HITS_NPZ_NAME = "hits_abs_xy.npz"

# ---------------- Geometry helpers ----------------
def regular_hexagon_vertices(R=1.0, rot=0.0):
    """Regular hexagon vertices (counterclockwise), rotated by 'rot' (radians)."""
    angs = rot + np.deg2rad(np.arange(0, 360, 60))
    return np.stack([R*np.cos(angs), R*np.sin(angs)], 1)

#def regular_hexagon_vertices(R=1.0, rot=0.0):
#    """Irregular hexagon vertices (used in the paper for the chaotic case)."""
#    angs = [rot + np.deg2rad([0,  60, (math.sqrt(5)-1)/2 * 180, 180, (math.sqrt(3)+1)/2 * 180, 300])[i] for i in range(6)]
#    return np.stack([R*np.cos(angs), R*np.sin(angs)], 1)

def edges_from_vertices(V):
    return [(V[i], V[(i+1) % len(V)], i) for i in range(len(V))]

def edge_tangent(a, b):
    e = b - a
    L = LA.norm(e)
    return e / (L + 1e-30)

def ray_segment_intersection(p, v, a, b, eps=1e-12):
    """Ray p + t*v vs. segment [a, b]. Returns t if it intersects in front of p, else None."""
    e = b - a
    M = np.array([v, -e]).T
    rhs = a - p
    det = np.linalg.det(M)
    if abs(det) < eps:
        return None
    t, u = np.linalg.solve(M, rhs)
    if t > eps and -eps <= u <= 1 + eps:
        return t
    return None

def next_collision(p, v, edges, last_edge_idx=None):
    """Return (q, edge_idx, t_to_hit) for the nearest forward collision, or None."""
    best = None
    hit = None
    for (a, b, ei) in edges:
        t = ray_segment_intersection(p, v, a, b)
        if t is None:
            continue
        # Skip the just-hit edge to avoid floating-point re-intersection
        if last_edge_idx is not None and ei == last_edge_idx and t < 1e-9:
            continue
        if best is None or t < best:
            best = t
            hit = (a, b, ei)
    if hit is None:
        return None
    a, b, ei = hit
    q = p + best * v
    return q, ei, best

def wrap_ang(v):
    return math.atan2(float(v[1]), float(v[0]))

# ---------------- Reflection-table lookup ----------------
class Reflector:
    """
    Loads a 360-degree single-valued reflection table from .npz and maps phi_in -> phi_out.
    If the file is missing, falls back to specular reflection phi_out = -phi_in.
    """
    def __init__(self, npz_path: str):
        self.edge_angle = 0.0
        self.x = None
        self.y = None
        self.mode = "mirror"
        self.npz_obj = None

        if not os.path.exists(npz_path):
            return

        z = np.load(npz_path, allow_pickle=True)
        self.npz_obj = z

        if "meta" in z.files:
            m = z["meta"].item()
            if "edge_angle" in m:
                self.edge_angle = float(m["edge_angle"])

        if ("phi_in_360" in z.files) and ("phi_out_360" in z.files):
            x = np.asarray(z["phi_in_360"], float)
            y = np.asarray(z["phi_out_360"], float)
            o = np.argsort(x)
            self.x = x[o]
            self.y = y[o]
            self.mode = "360"

    def has_table(self):
        return self.mode == "360"

    def map_linear_periodic(self, q: float) -> float:
        """Periodic linear interpolation on (-pi, pi] with shortest-arc dy."""
        x, y = self.x, self.y
        if x is None or y is None or len(x) < 2:
            return float("nan")

        x_all = np.concatenate([x - 2*math.pi, x, x + 2*math.pi])
        y_all = np.concatenate([y,           y, y          ])

        j = int(np.searchsorted(x_all, q, side="right"))
        j = max(1, min(j, len(x_all) - 1))
        x1, x2 = x_all[j-1], x_all[j]
        y1, y2 = y_all[j-1], y_all[j]

        if x2 == x1:
            return float(y1)

        w = (q - x1) / (x2 - x1)
        dy = ((y2 - y1 + math.pi) % (2*math.pi)) - math.pi
        return float(y1 + w * dy)

def edge_alpha(a, b, choice, alpha_ref):
    """Directed-tangent angle for edge (a, b); 'choice' may flip the direction."""
    t = edge_tangent(a, b)
    choice = (choice or "keep").lower()
    base = math.atan2(float(t[1]), float(t[0]))

    if choice == "flip":
        return wrap_pi(base + math.pi)
    if choice == "auto":
        t_ref = np.array([math.cos(alpha_ref), math.sin(alpha_ref)])
        return wrap_pi(base + (math.pi if np.dot(t, t_ref) < 0 else 0.0))
    return base

def reflect_once(v_in, a, b, ei, Rf: Reflector):
    """One reflection: phi_in (in local edge frame) -> phi_out via the table."""
    alpha_e = edge_alpha(
        a, b,
        EDGE_ALIGN_CHOICE[ei] if ei < len(EDGE_ALIGN_CHOICE) else "keep",
        Rf.edge_angle
    )

    psi_in = wrap_ang(v_in)
    phi_in_rel = wrap_pi(psi_in - alpha_e)

    if Rf.has_table():
        phi_q = TABLE_IN_SIGN * (phi_in_rel + math.radians(TABLE_IN_SHIFT_DEG))
        phi_out_rel = Rf.map_linear_periodic(phi_q)
        if not np.isfinite(phi_out_rel):
            phi_out_rel = -phi_in_rel
        else:
            phi_out_rel = TABLE_OUT_SIGN * phi_out_rel + math.radians(TABLE_OUT_SHIFT_DEG)
    else:
        phi_out_rel = -phi_in_rel

    psi_out = wrap_pi(phi_out_rel + alpha_e)
    v_out = np.array([math.cos(psi_out), math.sin(psi_out)], float)
    v_out /= (LA.norm(v_out) + 1e-30)
    return v_out, phi_in_rel, phi_out_rel, alpha_e, psi_out

# ---------------- Main simulation ----------------
def run():
    V = regular_hexagon_vertices(R=R_HEX, rot=np.deg2rad(ROT_DEG))
    edges = edges_from_vertices(V)

    Rf = Reflector(NPZ_PATH)
    vk = VKMapper(Rf.npz_obj)

    p = np.array(START_XY, float)
    psi0_v = deg2rad(START_ABS_ANGLE_DEG)
    v = np.array([math.cos(psi0_v), math.sin(psi0_v)], float)

    # Optional Fermi-surface points for the side panel
    fs_xy = None
    if Rf.npz_obj is not None:
        z = Rf.npz_obj
        if ("fs_x" in z.files and "fs_y" in z.files):
            fs_xy = np.stack([z["fs_x"].astype(float), z["fs_y"].astype(float)], 1)
        elif ("kx_360" in z.files and "ky_360" in z.files):
            fs_xy = np.stack([z["kx_360"].astype(float), z["ky_360"].astype(float)], 1)

    # 3 unique edge tangents (mod pi) for FS panel overlays
    uniq_edge_angles = []
    for i in range(6):
        a, b, _ = edges[i]
        ang_i = math.atan2(*((b - a)[::-1]))
        base = (ang_i + math.pi) % math.pi
        if not any(abs(((base - x + math.pi) % math.pi) - math.pi) < 1e-6 for x in uniq_edge_angles):
            uniq_edge_angles.append(base)
        if len(uniq_edge_angles) == 3:
            break

    fs_panel = None
    geom_enabled = True

    if RUN_MODE == "poincare_only" and ONLY_POINCARE_FIG:
        fig, ax_p = plt.subplots(1, 1, figsize=(5, 4))
        pc_panel = PoincarePanel(
            ax_p, n_edges=6, use_phi_out=POINCARE_USE_PHI_OUT,
            angle_unit=ANGLE_UNIT,
            title=f"Poincare (edge {EDGE_TO_PLOT})",
            angle_relative="normal",
            edge_focus=EDGE_TO_PLOT
        )
        ax = None
        geom_enabled = False
    else:
        if LAYOUT == "one_figure":
            fig, (ax_p, ax_fs, ax_geom) = plt.subplots(1, 3, figsize=(15, 5))
            pc_panel = PoincarePanel(
                ax_p, n_edges=6, use_phi_out=POINCARE_USE_PHI_OUT,
                angle_unit=ANGLE_UNIT,
                title="Poincare: x=edge+s, y=" + ("phi_out" if POINCARE_USE_PHI_OUT else "phi_in")
            )
            fs_panel = FermiSurfacePanel(
                ax_fs, fs_points=fs_xy, edge_angles=uniq_edge_angles,
                angle_unit=ANGLE_UNIT, title="Fermi surface (k)"
            )
            ax = ax_geom
        else:
            fig = plt.figure(figsize=(10, 5))
            ax_p = fig.add_subplot(1, 2, 1)
            ax_fs = fig.add_subplot(1, 2, 2)
            pc_panel = PoincarePanel(
                ax_p, n_edges=6, use_phi_out=POINCARE_USE_PHI_OUT,
                angle_unit=ANGLE_UNIT
            )
            fs_panel = FermiSurfacePanel(
                ax_fs, fs_points=fs_xy, edge_angles=uniq_edge_angles,
                angle_unit=ANGLE_UNIT
            )
            fig2, ax = plt.subplots(figsize=(7, 7))

    if geom_enabled and ax is not None:
        poly = np.vstack([V, V[0]])
        ax.plot(poly[:, 0], poly[:, 1], "k-", lw=2)
        if SHOW_EDGE_LABELS:
            for i in range(6):
                a, b, _ = edges[i]
                m = 0.5 * (a + b)
                ax.text(m[0], m[1], str(i), ha="center", va="center")
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])

        ln, = ax.plot([], [], "-", lw=1.5, color="tab:blue")
        head, = ax.plot([p[0]], [p[1]], "o", ms=4, color="tab:blue")

        segments = deque(maxlen=TRAJ_MAX_BOUNCES_TO_SHOW)

        def rebuild(cur_xs=None, cur_ys=None):
            xs, ys = [], []
            for a1, a2 in segments:
                xs += a1
                ys += a2
                xs.append(np.nan)
                ys.append(np.nan)
            if cur_xs is not None:
                xs += cur_xs
                ys += cur_ys
            ln.set_data(xs, ys)
    else:
        def rebuild(cur_xs=None, cur_ys=None):
            return

    if fs_panel is not None:
        fs_panel.update_k(vk.v_to_k(psi0_v))

    if RUN_MODE == "live":
        plt.ion()

    hits = []
    print(f"[start] x={p[0]:+.9f}  y={p[1]:+.9f}  psi0={START_ABS_ANGLE_DEG:+.6f} deg")
    if SAVE_HITS:
        hits.append((
            -1, -1, float(p[0]), float(p[1]),
            float(psi0_v), float("nan"),
            float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan")
        ))

    last_edge = None
    for bounce in range(MAX_BOUNCES):
        hit = next_collision(p, v, edges, last_edge_idx=last_edge)
        if hit is None:
            break
        q, ei, dist = hit

        if geom_enabled and RUN_MODE in ("live", "fast"):
            nframes = max(1, int(dist * STEPS_PER_UNIT))
            cur_xs = [float(p[0])]
            cur_ys = [float(p[1])]
            for t in (np.linspace(0, 1, nframes, True) if RUN_MODE == "live" else [1.0]):
                r = p + t * (q - p)
                cur_xs.append(float(r[0]))
                cur_ys.append(float(r[1]))
                rebuild(cur_xs, cur_ys)
                head.set_data([r[0]], [r[1]])
                if RUN_MODE == "live":
                    plt.pause(1.0 / max(1.0, FPS))
            segments.append((cur_xs, cur_ys))

        a, b, _ = edges[ei]
        v_out, phi_in_rel, phi_out_rel, alpha_e, psi_out = reflect_once(v, a, b, ei, Rf)

        if LOG_FIRST_N and bounce < LOG_FIRST_N:
            print(f"[{bounce:04d}] edge={ei}  alpha_e={alpha_e/math.pi*180:+.3f} deg  "
                  f"phi_in={phi_in_rel/math.pi*180:+.3f} deg -> phi_out={phi_out_rel/math.pi*180:+.3f} deg")

        e_vec = b - a
        L_e = LA.norm(e_vec)
        t_hat = e_vec / (L_e + 1e-30)
        s = float(np.dot(q - a, t_hat))
        s01 = max(0.0, min(1.0, s / (L_e + 1e-30)))

        psi_in = wrap_ang(v)
        if ANGLE_UNIT == "deg":
            print(f"[hit {bounce:06d}] edge={ei}  x={q[0]:+.9f}  y={q[1]:+.9f}  "
                  f"psi_in={math.degrees(psi_in):+9.4f} deg  psi_out={math.degrees(psi_out):+9.4f} deg  "
                  f"s={s:.9f}  s01={s01:.9f}")
        else:
            print(f"[hit {bounce:06d}] edge={ei}  x={q[0]:+.9f}  y={q[1]:+.9f}  "
                  f"psi_in={psi_in:+.6f}  psi_out={psi_out:+.6f}  s={s:.9f}  s01={s01:.9f}")

        if SAVE_HITS:
            hits.append((
                int(bounce), int(ei),
                float(q[0]), float(q[1]),
                float(psi_in), float(psi_out),
                float(phi_in_rel), float(phi_out_rel),
                float(alpha_e), float(s), float(s01), float(dist)
            ))

        pc_panel.add_hit(edge_idx=ei, s01=s01,
                         phi_in=phi_in_rel, phi_out=phi_out_rel, alpha_e=alpha_e)

        if fs_panel is not None:
            fs_panel.update_k(vk.v_to_k(psi_out))

        p = q
        v = v_out
        last_edge = ei

        if RUN_MODE == "poincare_only":
            continue

    csv_path, npz_path = pc_panel.save(POINCARE_SAVE_DIR, tag=POINCARE_SAVE_TAG)
    print(f"saved Poincare:\n  CSV: {csv_path}\n  NPZ: {npz_path}")

    if SAVE_HITS and len(hits) > 0:
        out_dir = POINCARE_SAVE_DIR
        os.makedirs(out_dir, exist_ok=True)
        csv_out = os.path.join(out_dir, HITS_CSV_NAME)
        npz_out = os.path.join(out_dir, HITS_NPZ_NAME)

        with open(csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "bounce", "edge", "x", "y", "psi_in_deg", "psi_out_deg",
                "phi_in_rel_deg", "phi_out_rel_deg", "alpha_e_deg", "s", "s01", "dist"
            ])
            for (b, ei, xq, yq, psi_in, psi_out, ph_in, ph_out, a_e, s, s01, dist) in hits:
                w.writerow([
                    b, ei, f"{xq:.9f}", f"{yq:.9f}",
                    f"{math.degrees(psi_in):.6f}" if np.isfinite(psi_in) else "",
                    f"{math.degrees(psi_out):.6f}" if np.isfinite(psi_out) else "",
                    f"{math.degrees(ph_in):.6f}" if np.isfinite(ph_in) else "",
                    f"{math.degrees(ph_out):.6f}" if np.isfinite(ph_out) else "",
                    f"{math.degrees(a_e):.6f}" if np.isfinite(a_e) else "",
                    f"{s:.9f}", f"{s01:.9f}",
                    f"{dist:.9e}" if np.isfinite(dist) else ""
                ])

        arr = np.array(hits, float)
        np.savez_compressed(
            npz_out,
            cols=np.array(
                ["bounce", "edge", "x", "y", "psi_in", "psi_out", "phi_in_rel",
                 "phi_out_rel", "alpha_e", "s", "s01", "dist"],
                dtype=object
            ),
            data=arr
        )
        print(f"saved hits:\n  CSV: {csv_out}\n  NPZ: {npz_out}")

    if RUN_MODE == "poincare_only":
        os.makedirs(POINCARE_SAVE_DIR, exist_ok=True)
        svg_path = os.path.join(POINCARE_SAVE_DIR, POINCARE_SVG_NAME)
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"saved Poincare SVG: {svg_path}")

    if RUN_MODE == "live":
        plt.ioff()

    svg_path = os.path.join(POINCARE_SAVE_DIR, "poincare_edge0.svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    print("saved:", svg_path)
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(NPZ_PATH):
        print("WARNING: reflection table not found; falling back to specular mirror:", NPZ_PATH)
    run()

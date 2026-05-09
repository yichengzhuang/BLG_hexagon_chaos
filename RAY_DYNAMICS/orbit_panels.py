#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualization helpers used by run_hex_orbit.py: Poincare panel, Fermi-surface panel, v->k mapper."""

from __future__ import annotations
import math, os
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from typing import Optional, List

def wrap_pi(a: float) -> float:
    a = (a + math.pi) % (2*math.pi) - math.pi
    if a <= -math.pi: a += 2*math.pi
    return a

def deg2rad(x): return x * math.pi / 180.0
def rad2deg(x): return x * 180.0 / math.pi

# ---------------- v-direction -> k-direction mapping ----------------
class VKMapper:
    """Look up the k-space angle that corresponds to a given v-space angle."""
    def __init__(self, npz: Optional[np.lib.npyio.NpzFile]):
        self.have_map = False; self.v=None; self.k=None
        if npz is None: return
        for a,b in [('theta_v_360','theta_k_360'),('v_ang_360','k_ang_360')]:
            if a in npz.files and b in npz.files:
                v = np.asarray(npz[a], float); k = np.asarray(npz[b], float)
                if v.ndim==1 and k.ndim==1 and len(v)==len(k)>=2:
                    o = np.argsort(v); self.v=v[o]; self.k=k[o]; self.have_map=True; break

    def v_to_k(self, psi_v: float) -> float:
        if not self.have_map: return wrap_pi(psi_v)
        v,k = self.v, self.k
        v_all = np.concatenate([v-2*math.pi, v, v+2*math.pi])
        k_all = np.concatenate([k,            k, k           ])
        j = int(np.searchsorted(v_all, psi_v, side='right'))
        j = max(1, min(j, len(v_all)-1))
        v1,v2 = v_all[j-1], v_all[j]; k1,k2 = k_all[j-1], k_all[j]
        if v2==v1: return wrap_pi(float(k1))
        w=(psi_v-v1)/(v2-v1); dk=((k2-k1+math.pi)% (2*math.pi))-math.pi
        return wrap_pi(k1+w*dk)

# ---------------- Fermi-surface panel ----------------
class FermiSurfacePanel:
    def __init__(self, ax: plt.Axes,
                 fs_points: Optional[np.ndarray]=None,
                 edge_angles: Optional[List[float]]=None,
                 angle_unit: str="rad",
                 title="Fermi surface (k)"):
        self.ax = ax; self.angle_unit = angle_unit
        self.ax.set_aspect('equal','box'); self.ax.set_title(title)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        if fs_points is None:
            th = np.linspace(0,2*math.pi,720)
            fs_points = np.stack([np.cos(th), np.sin(th)],1)
        self.fs = fs_points
        self.ax.plot(self.fs[:,0], self.fs[:,1], lw=1.8, color='tab:gray')

        if edge_angles:
            for ang in edge_angles[:3]:
                d = np.array([math.cos(ang), math.sin(ang)])
                L = 1.2*np.max(LA.norm(self.fs, axis=1))
                self.ax.plot([-L*d[0], L*d[0]], [-L*d[1], L*d[1]],
                             ls='--', lw=1.0, color='tab:blue', alpha=0.6)
        self.k_point, = self.ax.plot([0],[0],'o',ms=6,color='tab:red')
        self.k_arrow=None
        m = 1.2*float(np.max(np.abs(self.fs)))
        self.ax.set_xlim(-m,m); self.ax.set_ylim(-m,m)

    def _closest_on_fs(self, theta_k: float) -> np.ndarray:
        th = np.arctan2(self.fs[:,1], self.fs[:,0])
        d = np.abs(np.angle(np.exp(1j*(th-theta_k))))
        return self.fs[int(np.argmin(d))]

    def update_k(self, theta_k: float):
        pt = self._closest_on_fs(theta_k)
        self.k_point.set_data([pt[0]], [pt[1]])
        if self.k_arrow is not None:
            self.k_arrow.remove(); self.k_arrow=None
        L = 0.6*max(self.ax.get_xlim()[1], self.ax.get_ylim()[1])
        head = np.array([math.cos(theta_k), math.sin(theta_k)])*L
        self.k_arrow = self.ax.arrow(0,0, head[0], head[1],
                                     width=0.0, head_width=L*0.05, head_length=L*0.08,
                                     length_includes_head=True, color='tab:red', alpha=0.85)


# ---------------- Poincare panel ----------------
class PoincarePanel:
    def __init__(self, ax: plt.Axes, n_edges=6, use_phi_out=True,
                 angle_unit="rad", title="Boundary Poincare",
                 angle_relative: str = "tangent",   # "tangent" | "normal"
                 edge_focus: Optional[int] = None   # restrict to one edge; None = all
                 ):
        self.ax = ax
        self.n_edges = n_edges
        self.use_phi_out = use_phi_out
        self.angle_unit = angle_unit
        self.angle_relative = angle_relative
        self.edge_focus = edge_focus

        if self.edge_focus is None:
            self.ax.set_xlabel("s",fontsize=28)
            self.ax.set_xlim(0, n_edges)
        else:
            self.ax.set_xlabel("s",fontsize=28)
            self.ax.set_xlim(0.0, 1.0)

        if self.angle_relative == "normal":
            ymax = 90.0 if angle_unit == "deg" else math.pi/2
        else:
            ymax = 180.0 if angle_unit == "deg" else math.pi

        self.ax.set_ylabel(r"$\theta$", fontsize=28)
        self.ax.grid(True, alpha=0.25)
        self.ax.set_ylim(-ymax, ymax)

        self.data = []
        self._sc = None

    @staticmethod
    def _angle_rel_normal(phi: float, alpha_e: float) -> float:
        """Convert phi (rel. to tangent) to the angle relative to the inward normal."""
        psi = phi + alpha_e
        n_in = alpha_e - math.pi/2
        theta = wrap_pi(psi - n_in)
        if theta > math.pi/2:
            theta = math.pi - theta
        elif theta < -math.pi/2:
            theta = -math.pi - theta
        return theta

    def add_hit(self, edge_idx: int, s01: float,
                phi_in: float, phi_out: float, alpha_e: float):
        if (self.edge_focus is not None) and (edge_idx != self.edge_focus):
            return

        if self.angle_relative == "normal":
            phi_in = self._angle_rel_normal(phi_in, alpha_e)
            phi_out = self._angle_rel_normal(phi_out, alpha_e)

        if self.angle_unit == "deg":
            phi_in = rad2deg(phi_in)
            phi_out = rad2deg(phi_out)
            alpha_e = rad2deg(alpha_e)

        self.data.append((edge_idx, s01, phi_in, phi_out, alpha_e))

        if self.edge_focus is None:
            x = np.array([d[0] + d[1] for d in self.data])
        else:
            x = np.array([d[1] for d in self.data])
        y = np.array([d[3] if self.use_phi_out else d[2] for d in self.data])

        offs = np.column_stack([x, y])
        if self._sc is None:
            self._sc = self.ax.scatter(
                x, y, s=3, alpha=0.7, marker=",", linewidths=0, color="C0"
            )
        else:
            self._sc.set_offsets(offs)
        self.ax.figure.canvas.draw_idle()

    def save(self, out_dir: str, tag: str = "poincare_points"):
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{tag}.csv")
        npz_path = os.path.join(out_dir, f"{tag}.npz")
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            cols = [
                "edge", "s01",
                "phi_in({})".format(self.angle_unit),
                "phi_out({})".format(self.angle_unit),
                "alpha_e({})".format(self.angle_unit),
            ]
            w.writerow(cols)
            for r in self.data:
                w.writerow(r)
        arr = np.array(self.data, dtype=float)
        meta = {
            "columns": cols,
            "angle_unit": self.angle_unit,
            "angle_relative": self.angle_relative,
            "edge_focus": self.edge_focus,
        }
        np.savez(npz_path, data=arr, meta=meta)
        return csv_path, npz_path

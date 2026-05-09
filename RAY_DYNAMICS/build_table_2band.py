#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLG two-band (with trigonal warping v3) reflection table on a single Fermi
contour, covering the full 360 degrees in phi_in.

Conventions
-----------
- phi is measured w.r.t. the same directed edge tangent t:
    phi = wrap(arg(v) - alpha_edge) in (-pi, pi].
- Signed parallel momentum k_parallel = k . t is conserved across reflection.
- The 360 degrees are obtained by solving both incoming halves (n and -n);
  no symmetry copy is applied here.
"""

from __future__ import annotations
import os, math, csv
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Tuple, List
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = os.environ.get("BLG_RESULTS_DIR", str(_HERE / "result_example"))

# ---------------- Parameters ----------------
EF: float = 20.0
N_THETA: int = 300000
EDGE_ANGLES_DEG: List[float] = list(np.linspace(3, 4, 2))

BAND: int = +1
MSTAR: float = 1.0
DELTA: float = 0.0
XI: int = -1
V3: float = 0.5
THETA_WARP_DEG: float = 0.0

OUT_ROOT: str = os.environ.get("BLG_OUT_ROOT_2BAND", os.path.join(RESULTS_DIR, "twoband"))
DO_PLOT: bool = True

# --------------- Utilities ---------------
def wrap_pi(a):
    """Wrap angle(s) into (-pi, pi]."""
    a = (np.asarray(a) + np.pi) % (2*np.pi) - np.pi
    if np.ndim(a) == 0:
        a = float(a)
        if a <= -math.pi:
            a += 2*math.pi
        return a
    return np.where(a <= -np.pi, a + 2*np.pi, a)

def ang(v: np.ndarray) -> float:
    return math.atan2(float(v[1]), float(v[0]))

def rot2d(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s],[s, c]], float)

# --------------- Model ---------------
@dataclass
class ModelOps:
    energy: Callable[[float,float], float]
    velocity: Callable[[float,float], Tuple[float,float]]

def make_blg_v3(mstar: float, delta: float, xi: int, v3: float,
                theta_warp: float, band: int) -> ModelOps:
    """Two-band BLG with trigonal warping. Returns energy(kx,ky) and v_g(kx,ky)."""
    sgn = +1 if band>=0 else -1
    xi  = +1 if xi>=0 else -1
    R = rot2d(theta_warp); RT = R.T

    def energy(kx, ky):
        kx_p, ky_p = (RT @ np.array([kx,ky], float)).tolist()
        kp = kx_p + 1j * xi * ky_p
        km = kx_p - 1j * xi * ky_p
        f  = -(km**2)/(2.0*mstar) + xi * v3 * kp
        Eabs = math.sqrt((0.5*delta)**2 + (f.real*f.real + f.imag*f.imag))
        return sgn * Eabs

    def velocity(kx, ky):
        kx_p, ky_p = (RT @ np.array([kx,ky], float)).tolist()
        kp = kx_p + 1j * xi * ky_p
        km = kx_p - 1j * xi * ky_p
        f  = -(km**2)/(2.0*mstar) + xi * v3 * kp

        df_dkx = -(1.0/mstar)*km + xi*v3
        df_dky = (1j*xi/mstar)*km + 1j*v3

        Eabs = math.sqrt((0.5*delta)**2 + (f.real*f.real + f.imag*f.imag))
        if Eabs == 0.0:
            v_prime = np.array([0.0,0.0], float)
        else:
            re_dkx = (np.conjugate(f)*df_dkx).real
            re_dky = (np.conjugate(f)*df_dky).real
            v_prime = np.array([re_dkx/Eabs, re_dky/Eabs], float) * (1.0 if band>=0 else -1.0)

        return tuple((rot2d(theta_warp) @ v_prime).tolist())

    return ModelOps(energy=energy, velocity=velocity)

# --------------- Fermi surface sampling ---------------
def _bracket(f, a, b, tol=1e-12, it=100):
    fa, fb = f(a), f(b)
    if fa == 0: return a
    if fb == 0: return b
    if fa*fb > 0: raise ValueError("no bracket")
    for _ in range(it):
        m = 0.5*(a+b); fm = f(m)
        if abs(fm) < tol or abs(b-a) < tol: return m
        if fa*fm <= 0: b, fb = m, fm
        else: a, fa = m, fm
    return 0.5*(a+b)

def sample_fermi(ops: ModelOps, EF: float, n_theta=4096,
                 rmin=1e-6, rmax=10.0):
    """Radially sample the Fermi contour. Returns (thetas, ks, vs); NaN where unsolved."""
    thetas = np.linspace(-math.pi, math.pi, n_theta, endpoint=False)
    ks = np.full((n_theta,2), np.nan); vs = np.full((n_theta,2), np.nan)
    for i, th in enumerate(thetas):
        ux, uy = math.cos(th), math.sin(th)
        last=None; lastr=None; ok=False
        for r in np.geomspace(rmin, rmax, 80):
            val = ops.energy(r*ux, r*uy) - EF
            if last is not None and val*last <= 0:
                rr = _bracket(lambda q: ops.energy(q*ux,q*uy)-EF, lastr, r)
                kx, ky = rr*ux, rr*uy
                vx, vy = ops.velocity(kx, ky)
                ks[i]=(kx,ky); vs[i]=(vx,vy); ok=True; break
            last, lastr = val, r
        if not ok: pass
    return thetas, ks, vs

# --------------- 360-degree single-valued table ---------------
@dataclass
class Table360:
    meta: dict
    phi_in_360: np.ndarray
    phi_out_360: np.ndarray

def _phi2theta_pm90(x):
    """Map phi in (-pi, pi] to theta in (-pi/2, pi/2] by adding/subtracting pi."""
    a = (x + np.pi) % (2*np.pi) - np.pi
    a = np.where(a >  np.pi/2, a - np.pi, a)
    a = np.where(a <= -np.pi/2, a + np.pi, a)
    return a

def _build_one_side(ks, vs, t, n, in_is_neg=True):
    """
    Build (phi_in, phi_out) on the half selected by sign(v.n).
    in_is_neg=True : incoming v.n < 0, outgoing v.n > 0 (and vice versa).
    """
    ok = np.isfinite(ks[:,0]) & np.isfinite(vs[:,0])
    vin = np.einsum('ij,j->i', vs, n)
    kpar = np.einsum('ij,j->i', ks, t)

    if in_is_neg:
        mask_in = ok & (vin < 0)
        out_sign = +1
    else:
        mask_in = ok & (vin > 0)
        out_sign = -1

    phi_in_list, phi_out_list = [], []
    N = len(ks)
    for i in np.where(mask_in)[0]:
        target = kpar[i]
        phi_in = wrap_pi(ang(vs[i]) - math.atan2(t[1], t[0]))

        cand = []
        for j in range(N):
            j2 = (j+1) % N
            if not (ok[j] and ok[j2]): continue
            f1 = kpar[j]  - target
            f2 = kpar[j2] - target

            if f1 == 0.0:
                v = vs[j]
                if (np.dot(v, n) * out_sign) > 0:
                    cand.append(v)
                continue

            if f1 * f2 > 0:
                continue

            w = abs(f1) / (abs(f1)+abs(f2)+1e-30)
            v = (1-w)*vs[j] + w*vs[j2]
            if (np.dot(v, n) * out_sign) > 0:
                cand.append(v)

        if len(cand) == 0:
            continue

        # Single-valued selection: minimize |theta_out + theta_in|
        alpha = math.atan2(t[1], t[0])
        th_in = float(_phi2theta_pm90(np.array([phi_in]))[0])

        def cost(vout):
            phi_o = wrap_pi(ang(vout) - alpha)
            th_out = float(_phi2theta_pm90(np.array([phi_o]))[0])
            return abs(th_out + th_in)

        v_out = min(cand, key=cost)
        phi_out = wrap_pi(ang(v_out) - alpha)

        phi_in_list.append(phi_in)
        phi_out_list.append(phi_out)

    return np.array(phi_in_list, float), np.array(phi_out_list, float)

def build_table_360(ops: ModelOps, EF: float, edge_angle: float, n_theta: int) -> Table360:
    """Build the 360-degree single-valued reflection table for one edge angle."""
    ths, ks, vs = sample_fermi(ops, EF, n_theta)
    t = np.array([math.cos(edge_angle), math.sin(edge_angle)], float)
    n = np.array([-t[1], t[0]], float)

    pin_A, pout_A = _build_one_side(ks, vs, t, n,  in_is_neg=True)
    pin_B, pout_B = _build_one_side(ks, vs, t, n,  in_is_neg=False)

    phi_in_360  = np.concatenate([pin_A,  pin_B])
    phi_out_360 = np.concatenate([pout_A, pout_B])

    order = np.argsort(phi_in_360)
    x = phi_in_360[order]; y = phi_out_360[order]
    dx = np.diff(x); keep = np.ones_like(x, dtype=bool)
    tol = math.radians(1e-4)
    keep[1:] = np.abs(dx) > tol
    x = x[keep]; y = y[keep]

    meta = dict(
        model='BLG_v3', EF=float(EF), edge_angle=float(edge_angle), n_theta=int(n_theta),
        band=int(BAND), mstar=float(MSTAR), delta=float(DELTA), xi=int(XI), v3=float(V3),
        theta_warp=float(math.radians(THETA_WARP_DEG))
    )
    return Table360(meta=meta, phi_in_360=x, phi_out_360=y)

# --------------- Save / Plot ---------------
def save_table(rt: Table360, out_dir: str) -> Tuple[str, str]:
    """Save reflection table as .npz (radians) and .csv (degrees)."""
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, "reflect_table.npz")
    np.savez_compressed(
        npz,
        meta=rt.meta,
        phi_in_360=rt.phi_in_360, phi_out_360=rt.phi_out_360,
    )
    csvp = os.path.join(out_dir, "phi_in_out_360.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["phi_in_deg","phi_out_deg"])
        for a,b in zip(np.degrees(rt.phi_in_360), np.degrees(rt.phi_out_360)):
            w.writerow([f"{a:.8f}", f"{b:.8f}"])
    return npz, csvp

def plot_phi(rt: Table360, out_dir: str) -> str:
    x = np.degrees(rt.phi_in_360); y = np.degrees(rt.phi_out_360)
    plt.figure(figsize=(6.6,5.2))
    plt.plot(x, y, '.', ms=1.4, alpha=0.95, label="360 deg (single-valued)")
    xx = np.linspace(-180,180,721)
    plt.plot(xx,  xx,  ls='--', lw=1.0, alpha=0.5, label='y=x')
    plt.plot(xx, -xx,  ls='--', lw=1.0, alpha=0.6, label='y=-x')
    ttl = f"phi_in - phi_out  E_F={rt.meta['EF']:.4g}, edge={math.degrees(rt.meta['edge_angle']):.1f} deg, Ntheta={rt.meta['n_theta']}"
    plt.title(ttl); plt.xlabel("phi_in (deg, v vs. t)"); plt.ylabel("phi_out (deg)")
    plt.xlim(-180,180); plt.ylim(-180,180)
    plt.grid(True, alpha=0.3); plt.legend()
    out = os.path.join(out_dir, "phi_in_out_360.png")
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
    return out

if __name__ == "__main__":
    ops = make_blg_v3(MSTAR, DELTA, XI, V3, math.radians(THETA_WARP_DEG), BAND)

    for EDGE_ANGLE_DEG in EDGE_ANGLES_DEG:
        EDGE = math.radians(float(EDGE_ANGLE_DEG))
        rt   = build_table_360(ops, EF=EF, edge_angle=EDGE, n_theta=N_THETA)
        tag  = f"E{EF:g}_ang{EDGE_ANGLE_DEG:g}_nt{N_THETA}_360"
        outd = os.path.join(OUT_ROOT, tag)

        npz, csvp = save_table(rt, outd)
        print(f"[edge={EDGE_ANGLE_DEG:.3f} deg] saved", npz)
        print(f"[edge={EDGE_ANGLE_DEG:.3f} deg] saved", csvp)
        if DO_PLOT:
            png = plot_phi(rt, outd)
            print(f"[edge={EDGE_ANGLE_DEG:.3f} deg] saved", png)

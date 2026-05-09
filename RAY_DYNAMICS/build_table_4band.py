#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLG four-band (AB stacking) reflection table on the numeric Fermi contour
of the low-energy conduction band, covering the full 360 degrees in phi_in.

For each edge angle, outputs reflect_table.npz, phi_in_out_360.csv, and
a quick-look phi_in_out_360.png under <RESULTS_DIR>/fourband/<tag>/.
"""

from __future__ import annotations
import os, math, csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import numpy.linalg as LA

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = os.environ.get("BLG_RESULTS_DIR", str(_HERE / "result_example"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Tight-binding parameters ----------------
A_LATTICE_NM: float = 0.24595   # lattice constant a [nm]
D_INTERLAYER_NM: float = 0.335  # interlayer distance d [nm] (stored in meta)

T0_EV: float = 3.16   # gamma0, intralayer NN
T1_EV: float = 0.381  # gamma1, vertical interlayer dimer
T3_EV: float = 0.38   # gamma3, skew interlayer
T4_EV: float = 0.14   # gamma4, skew interlayer

EF_EV: float = 0.2

EDGE_ANGLES_DEG: List[float] = [0.0, 15.0]

XI: int = +1                  # +1: K, -1: K'
THETA_WARP_DEG: float = 0.0   # rotate k by this angle before evaluating H

# Fermi contour resolution and bracketing
N_THETA_FS: int = 200000
R_TOL: float = 1e-12
R_INIT_SCALE: float = 3.0
R_EXPAND_FACTOR: float = 1.6
R_EXPAND_STEPS_MAX: int = 80

DROP_NAN: bool = True
DEDUP_TOL_DEG: float = 1e-4

OUT_ROOT: str = os.environ.get("BLG_OUT_ROOT_4BAND", os.path.join(RESULTS_DIR, "fourband"))
DO_PLOT: bool = True

# ---------------- Utilities ----------------
HBAR_EV_S: float = 6.582119569e-16
SQRT3: float = math.sqrt(3.0)

def wrap_pi(a):
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

def _phi2theta_pm90(x):
    a = (x + np.pi) % (2*np.pi) - np.pi
    a = np.where(a >  np.pi/2, a - np.pi, a)
    a = np.where(a <= -np.pi/2, a + np.pi, a)
    return a

# ---------------- 4-band model ----------------
@dataclass
class BLG4Band:
    a_m: float
    t0: float
    t1: float
    t3: float
    t4: float
    xi: int
    theta_warp: float  # radians

    v0: float = 0.0
    v3: float = 0.0
    v4: float = 0.0
    a0: float = 0.0
    a3: float = 0.0
    a4: float = 0.0

    def __post_init__(self):
        self.xi = +1 if self.xi >= 0 else -1
        # v_i = (sqrt(3)/2) * a * t_i / hbar  [m/s];  alpha_i = hbar v_i  [eV*m]
        pref = (SQRT3/2.0) * self.a_m / HBAR_EV_S
        self.v0 = pref * self.t0
        self.v3 = pref * self.t3
        self.v4 = pref * self.t4
        self.a0 = HBAR_EV_S * self.v0
        self.a3 = HBAR_EV_S * self.v3
        self.a4 = HBAR_EV_S * self.v4

    def _rotate_k_in(self, kx: float, ky: float) -> Tuple[float,float]:
        if abs(self.theta_warp) < 1e-15:
            return kx, ky
        R = rot2d(-self.theta_warp)
        k = R @ np.array([kx,ky], float)
        return float(k[0]), float(k[1])

    def H(self, kx: float, ky: float) -> np.ndarray:
        """4-band continuum H near valley xi in basis (A1, B1, A2, B2)."""
        kx, ky = self._rotate_k_in(kx, ky)
        xi = self.xi
        pi  = xi*kx + 1j*ky
        pid = xi*kx - 1j*ky

        a0, a3, a4 = self.a0, self.a3, self.a4
        g1 = self.t1

        H = np.zeros((4,4), dtype=complex)

        H[0,1] = a0*pid;   H[1,0] = a0*pi
        H[0,2] = -a4*pid;  H[2,0] = -a4*pi
        H[0,3] = a3*pi;    H[3,0] = a3*pid
        H[1,2] = g1;       H[2,1] = g1
        H[1,3] = -a4*pid;  H[3,1] = -a4*pi
        H[2,3] = a0*pid;   H[3,2] = a0*pi

        return H

    def dH_dk(self, kx: float, ky: float) -> Tuple[np.ndarray, np.ndarray]:
        """Derivatives (dH/dkx, dH/dky) in eV*m, evaluated in the same rotated frame as H."""
        kx, ky = self._rotate_k_in(kx, ky)
        xi = self.xi
        a0, a3, a4 = self.a0, self.a3, self.a4

        dpi_dkx  = xi
        dpid_dkx = xi
        dpi_dky  = 1j
        dpid_dky = -1j

        dHx = np.zeros((4,4), dtype=complex)
        dHy = np.zeros((4,4), dtype=complex)

        dHx[0,1] = a0*dpid_dkx;  dHx[1,0] = a0*dpi_dkx
        dHy[0,1] = a0*dpid_dky;  dHy[1,0] = a0*dpi_dky

        dHx[0,2] = -a4*dpid_dkx; dHx[2,0] = -a4*dpi_dkx
        dHy[0,2] = -a4*dpid_dky; dHy[2,0] = -a4*dpi_dky

        dHx[0,3] = a3*dpi_dkx;   dHx[3,0] = a3*dpid_dkx
        dHy[0,3] = a3*dpi_dky;   dHy[3,0] = a3*dpid_dky

        dHx[1,3] = -a4*dpid_dkx; dHx[3,1] = -a4*dpi_dkx
        dHy[1,3] = -a4*dpid_dky; dHy[3,1] = -a4*dpi_dky

        dHx[2,3] = a0*dpid_dkx;  dHx[3,2] = a0*dpi_dkx
        dHy[2,3] = a0*dpid_dky;  dHy[3,2] = a0*dpi_dky

        return dHx, dHy

    def eig(self, kx: float, ky: float) -> Tuple[np.ndarray, np.ndarray]:
        H = self.H(kx, ky)
        w, U = LA.eigh(H)  # ascending
        return w.real, U

    def E_low_conduction(self, kx: float, ky: float) -> float:
        w, _ = self.eig(kx, ky)
        return float(w[2])

    def gradE_low_conduction(self, kx: float, ky: float) -> np.ndarray:
        """Group-velocity direction via Hellmann-Feynman: <u| dH/dk |u>."""
        w, U = self.eig(kx, ky)
        u = U[:,2]
        dHx, dHy = self.dH_dk(kx, ky)
        dEdkx = np.vdot(u, dHx @ u).real
        dEdky = np.vdot(u, dHy @ u).real
        return np.array([dEdkx, dEdky], float)

# ---------------- Numeric Fermi contour ----------------
def bisect_root(f, a, b, tol=1e-12, it=200) -> Optional[float]:
    fa, fb = f(a), f(b)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return None
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa*fb > 0:
        return None
    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(it):
        mid = 0.5*(lo+hi)
        fmid = f(mid)
        if not np.isfinite(fmid):
            return None
        if abs(fmid) < tol or abs(hi-lo) < tol:
            return mid
        if flo*fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5*(lo+hi)

def tabulate_fermi_contour(model: BLG4Band, EF: float, n_theta: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample the low-conduction Fermi contour radially.

    Returns
    -------
    ks : (n, 2) k-points on E_low(k) = EF, units 1/m
    vs : (n, 2) gradE at those points (parallel to v_g), units eV*m
    """
    thetas = np.linspace(-math.pi, math.pi, n_theta, endpoint=False)
    k_est = abs(EF) / max(1e-30, model.a0)
    r0 = max(1e-12, R_INIT_SCALE * k_est)

    ks = np.full((n_theta, 2), np.nan, float)
    vs = np.full((n_theta, 2), np.nan, float)

    for i, th in enumerate(thetas):
        ux, uy = math.cos(th), math.sin(th)

        def f(r):
            return model.E_low_conduction(r*ux, r*uy) - EF

        f0 = f(0.0)
        if not np.isfinite(f0):
            continue

        # Expand r outward until sign of f flips
        r_hi = r0
        f_hi = f(r_hi)
        steps = 0
        while np.isfinite(f_hi) and f0*f_hi > 0 and steps < R_EXPAND_STEPS_MAX:
            r_hi *= R_EXPAND_FACTOR
            f_hi = f(r_hi)
            steps += 1

        if (not np.isfinite(f_hi)) or (f0*f_hi > 0):
            continue

        r_star = bisect_root(f, 0.0, r_hi, tol=R_TOL)
        if r_star is None:
            continue

        kx, ky = r_star*ux, r_star*uy
        g = model.gradE_low_conduction(kx, ky)

        ks[i] = (kx, ky)
        vs[i] = g

    if DROP_NAN:
        ok = np.isfinite(ks[:,0]) & np.isfinite(vs[:,0])
        ks = ks[ok]
        vs = vs[ok]

    return ks, vs

# ---------------- Reflection mapping on convex contour ----------------
@dataclass
class Table360:
    meta: dict
    phi_in_360: np.ndarray
    phi_out_360: np.ndarray

def _make_monotone_chains(ks: np.ndarray, kpar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a convex contour at the kpar extrema into two index chains with kpar increasing."""
    N = len(ks)
    imin = int(np.argmin(kpar))
    imax = int(np.argmax(kpar))

    def forward_chain(a, b):
        idx = [a]; j = a
        while j != b:
            j = (j + 1) % N
            idx.append(j)
        return np.array(idx, int)

    def backward_chain(a, b):
        idx = [a]; j = a
        while j != b:
            j = (j - 1) % N
            idx.append(j)
        return np.array(idx, int)

    c1 = forward_chain(imin, imax)
    c2 = backward_chain(imin, imax)

    if kpar[c1[0]] > kpar[c1[-1]]:
        c1 = c1[::-1]
    if kpar[c2[0]] > kpar[c2[-1]]:
        c2 = c2[::-1]

    return c1, c2

def _interp_on_chain(target: float,
                     chain_idx: np.ndarray,
                     kpar: np.ndarray,
                     ks: np.ndarray,
                     vs: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Linear interpolation along a kpar-monotone chain to solve k.t = target."""
    arr = kpar[chain_idx]
    if target < arr[0] or target > arr[-1]:
        return None

    j = int(np.searchsorted(arr, target, side="left"))
    if j == 0:
        idx0, idx1 = chain_idx[0], chain_idx[1]
        a0, a1 = arr[0], arr[1]
    elif j >= len(arr):
        idx0, idx1 = chain_idx[-2], chain_idx[-1]
        a0, a1 = arr[-2], arr[-1]
    else:
        idx0, idx1 = chain_idx[j-1], chain_idx[j]
        a0, a1 = arr[j-1], arr[j]

    denom = (a1 - a0)
    if abs(denom) < 1e-30:
        w = 0.0
    else:
        w = (target - a0) / denom
        w = max(0.0, min(1.0, w))

    k_inter = (1-w)*ks[idx0] + w*ks[idx1]
    v_inter = (1-w)*vs[idx0] + w*vs[idx1]
    return k_inter, v_inter

def _build_one_side_convex(ks: np.ndarray, vs: np.ndarray,
                           t: np.ndarray, n: np.ndarray,
                           in_is_neg: bool = True,
                           eps_k: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (phi_in, phi_out) for one half-plane.
    in_is_neg=True : incoming v.n < 0, outgoing v.n > 0 (and vice versa).
    """
    N = len(ks)
    if N < 8:
        return np.array([]), np.array([])

    vin = vs @ n
    kpar = ks @ t

    if in_is_neg:
        mask_in = vin < 0
        out_sign = +1
    else:
        mask_in = vin > 0
        out_sign = -1

    chain1, chain2 = _make_monotone_chains(ks, kpar)

    alpha = math.atan2(float(t[1]), float(t[0]))
    phi_in_list, phi_out_list = [], []

    for i in np.where(mask_in)[0]:
        target = float(kpar[i])
        phi_in = wrap_pi(ang(vs[i]) - alpha)

        cand = []
        r1 = _interp_on_chain(target, chain1, kpar, ks, vs)
        r2 = _interp_on_chain(target, chain2, kpar, ks, vs)

        for r in (r1, r2):
            if r is None:
                continue
            k_int, v_int = r
            if LA.norm(k_int - ks[i]) < eps_k:
                continue
            if (float(np.dot(v_int, n)) * out_sign) > 0:
                cand.append(v_int)

        if len(cand) == 0:
            continue

        if len(cand) == 1:
            v_out = cand[0]
        else:
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

def build_table_360_from_numeric_fs(model: BLG4Band, EF: float, edge_angle_rad: float,
                                   n_theta_fs: int) -> Table360:
    ks, vs = tabulate_fermi_contour(model, EF=EF, n_theta=n_theta_fs)

    t = np.array([math.cos(edge_angle_rad), math.sin(edge_angle_rad)], float)
    n = np.array([-t[1], t[0]], float)

    pin_A, pout_A = _build_one_side_convex(ks, vs, t, n, in_is_neg=True)
    pin_B, pout_B = _build_one_side_convex(ks, vs, t, n, in_is_neg=False)

    phi_in_360  = np.concatenate([pin_A, pin_B])
    phi_out_360 = np.concatenate([pout_A, pout_B])

    order = np.argsort(phi_in_360)
    x = phi_in_360[order]
    y = phi_out_360[order]

    tol = math.radians(DEDUP_TOL_DEG)
    keep = np.ones_like(x, dtype=bool)
    if len(x) >= 2:
        keep[1:] = np.abs(np.diff(x)) > tol
    x = x[keep]
    y = y[keep]

    meta = dict(
        model="BLG_4band_numericFS",
        EF=float(EF),
        edge_angle=float(edge_angle_rad),
        xi=int(model.xi),
        theta_warp=float(model.theta_warp),
        n_theta_fs=int(n_theta_fs),
        a_nm=float(A_LATTICE_NM),
        d_nm=float(D_INTERLAYER_NM),
        t0_ev=float(T0_EV), t1_ev=float(T1_EV), t3_ev=float(T3_EV), t4_ev=float(T4_EV),
        v0_mps=float(model.v0), v3_mps=float(model.v3), v4_mps=float(model.v4),
    )

    return Table360(meta=meta, phi_in_360=x, phi_out_360=y)

# ---------------- Save / plot ----------------
def save_table(rt: Table360, out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, "reflect_table.npz")
    np.savez_compressed(
        npz,
        meta=rt.meta,
        phi_in_360=rt.phi_in_360,
        phi_out_360=rt.phi_out_360,
    )
    csvp = os.path.join(out_dir, "phi_in_out_360.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phi_in_deg", "phi_out_deg"])
        for a, b in zip(np.degrees(rt.phi_in_360), np.degrees(rt.phi_out_360)):
            w.writerow([f"{a:.8f}", f"{b:.8f}"])
    return npz, csvp

def plot_phi(rt: Table360, out_dir: str) -> str:
    x = np.degrees(rt.phi_in_360)
    y = np.degrees(rt.phi_out_360)
    plt.figure(figsize=(6.6, 5.2))
    plt.plot(x, y, ".", ms=1.4, alpha=0.95, label="360 deg (single-valued)")
    xx = np.linspace(-180, 180, 721)
    plt.plot(xx,  xx, ls="--", lw=1.0, alpha=0.5, label="y=x")
    plt.plot(xx, -xx, ls="--", lw=1.0, alpha=0.6, label="y=-x")
    ttl = (f"phi_in - phi_out (4-band, numeric FS)  "
           f"E_F={rt.meta['EF']:.3f} eV, edge={math.degrees(rt.meta['edge_angle']):.3f} deg, "
           f"Ntheta_FS={rt.meta['n_theta_fs']}")
    plt.title(ttl)
    plt.xlabel("phi_in (deg, v vs. t)")
    plt.ylabel("phi_out (deg)")
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = os.path.join(out_dir, "phi_in_out_360.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out

if __name__ == "__main__":
    a_m = A_LATTICE_NM * 1e-9
    model = BLG4Band(
        a_m=a_m,
        t0=T0_EV, t1=T1_EV, t3=T3_EV, t4=T4_EV,
        xi=XI,
        theta_warp=math.radians(THETA_WARP_DEG),
    )

    for EDGE_ANGLE_DEG in EDGE_ANGLES_DEG:
        EDGE = math.radians(float(EDGE_ANGLE_DEG))
        rt = build_table_360_from_numeric_fs(model, EF=EF_EV, edge_angle_rad=EDGE, n_theta_fs=N_THETA_FS)

        tag = f"E{EF_EV:g}_ang{EDGE_ANGLE_DEG:g}_ntFS{N_THETA_FS}_4band_numericFS_360"
        outd = os.path.join(OUT_ROOT, tag)

        npz, csvp = save_table(rt, outd)
        print(f"[edge={EDGE_ANGLE_DEG:.6f} deg] saved", npz)
        print(f"[edge={EDGE_ANGLE_DEG:.6f} deg] saved", csvp)
        if DO_PLOT:
            png = plot_phi(rt, outd)
            print(f"[edge={EDGE_ANGLE_DEG:.6f} deg] saved", png)

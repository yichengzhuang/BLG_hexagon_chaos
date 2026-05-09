#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extend a (phi_in, phi_out) reflection table by its y = -x symmetric counterpart
and write the merged 360-degree table back to the same directory.
"""

import os, math, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = os.environ.get("BLG_RESULTS_DIR", str(_HERE / "result_example"))

# TABLE_TAG selects the sub-folder produced by build_table_*.py.
TABLE_ROOT = os.environ.get("BLG_TABLE_ROOT", RESULTS_DIR)
TABLE_TAG  = os.environ.get(
    "BLG_TABLE_TAG",
    "fourband/E0.2_ang15_ntFS200000_4band_numericFS_360",
)

IN_NPZ_NAME   = "reflect_table.npz"
OUT_NPZ_NAME  = "reflect_table_sym_ynegx.npz"
OUT_CSV_NAME  = "phi_in_out_360_sym_ynegx.csv"
OUT_PNG_NAME  = "phi_in_out_360_sym_ynegx.png"

CSV_PNG_UNIT  = "deg"   # "rad" | "deg"
TOL_RAD = math.radians(1e-8)

def wrap_pi(a):
    a = (np.asarray(a) + np.pi) % (2*np.pi) - np.pi
    if np.ndim(a) == 0:
        a = float(a)
        if a <= -math.pi:
            a += 2*math.pi
        return a
    return np.where(a <= -np.pi, a + 2*np.pi, a)

def angdiff(a, b):
    return wrap_pi(np.asarray(a) - np.asarray(b))

def load_npz_keep_all(path):
    z = np.load(path, allow_pickle=True)
    return z, list(z.files)

def main():
    out_dir = os.path.join(TABLE_ROOT, TABLE_TAG)
    in_npz  = os.path.join(out_dir, IN_NPZ_NAME)
    if not os.path.exists(in_npz):
        raise FileNotFoundError(in_npz)

    z, keys = load_npz_keep_all(in_npz)

    if ("phi_in_360" not in keys) or ("phi_out_360" not in keys):
        raise RuntimeError("need keys: phi_in_360 / phi_out_360")

    x = np.asarray(z["phi_in_360"],  float)  # rad
    y = np.asarray(z["phi_out_360"], float)  # rad

    # 1) y = -x symmetric points
    x_sym = wrap_pi(y)
    y_sym = wrap_pi(x)

    # 2) merge with the original
    x_all = np.concatenate([wrap_pi(x), x_sym])
    y_all = np.concatenate([wrap_pi(y), y_sym])

    # 3) sort by phi_in and remove near-duplicates
    order = np.argsort(x_all)
    x_sorted = x_all[order]
    y_sorted = y_all[order]

    keep = np.ones_like(x_sorted, dtype=bool)
    if x_sorted.size > 1:
        last_idx = 0
        for i in range(1, x_sorted.size):
            dx = abs(x_sorted[i] - x_sorted[last_idx])
            dy = abs(angdiff(y_sorted[i], y_sorted[last_idx]))
            if dx <= TOL_RAD and dy <= TOL_RAD:
                keep[i] = False
            else:
                last_idx = i
    x_final = x_sorted[keep]
    y_final = y_sorted[keep]

    # 4) write CSV / PNG
    unit = CSV_PNG_UNIT.lower()
    conv = (lambda a: a) if unit == "rad" else (lambda a: np.degrees(a))
    csv_path = os.path.join(out_dir, OUT_CSV_NAME)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"phi_in({unit})", f"phi_out({unit})"])
        for xi, yi in zip(conv(x_final), conv(y_final)):
            w.writerow([float(xi), float(yi)])

    png_path = os.path.join(out_dir, OUT_PNG_NAME)
    plt.figure(figsize=(6.6,5.2))
    plt.plot(conv(x_final), conv(y_final), '.', ms=1.4, alpha=0.95, label="360 deg + sym(y=-x)")
    xx = np.linspace(-180, 180, 721) if unit == "deg" else np.linspace(-math.pi, math.pi, 721)
    plt.plot(xx,  xx,  ls='--', lw=1.0, alpha=0.5, label='y=x')
    plt.plot(xx, -xx,  ls='--', lw=1.0, alpha=0.6, label='y=-x')
    meta_dict = {}
    try:
        meta_dict = z["meta"].item() if "meta" in keys else {}
    except Exception:
        meta_dict = {}
    ef   = meta_dict.get("EF", None)
    edge = meta_dict.get("edge_angle", None)
    ntheta = meta_dict.get("n_theta", None)
    if unit == "deg" and edge is not None:
        edge_disp = math.degrees(edge)
    else:
        edge_disp = edge
    ttl = "phi_in - phi_out (360 deg + sym y=-x)"
    if ef is not None or edge is not None or ntheta is not None:
        ttl += f"  E_F={ef if ef is not None else '?'}"
        if edge is not None:
            ttl += f", edge={edge_disp:.4g}{' deg' if unit=='deg' else ' rad'}"
        if ntheta is not None:
            ttl += f", Ntheta={ntheta}"
    plt.title(ttl)
    plt.xlabel(f"phi_in ({unit})"); plt.ylabel(f"phi_out ({unit})")
    lim = 180 if unit == "deg" else math.pi
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(png_path, dpi=200); plt.close()

    # 5) write the merged npz (preserve all original keys, overwrite the two arrays)
    out_npz = os.path.join(out_dir, OUT_NPZ_NAME)
    payload = {k: z[k] for k in keys}
    payload["phi_in_360"]  = x_final.astype(float)
    payload["phi_out_360"] = y_final.astype(float)
    np.savez_compressed(out_npz, **payload)

    print("Done")
    print("  in npz :", in_npz)
    print("  out npz:", out_npz)
    print("  out csv:", csv_path)
    print("  out png:", png_path)
    print("  points :", len(x_final))

if __name__ == "__main__":
    main()

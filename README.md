# BLG Hexagon Chaos — Code Repository

Code accompanying

> J. Lin†, Y. Zhuang†, A. M. Graf, J. Keski-Rahkonen, E. J. Heller.
> *Shaping chaos in bilayer graphene cavities.*
> [arXiv:2512.10914](https://arxiv.org/abs/2512.10914).
>
> † These authors contributed equally.

Code by Yicheng Zhuang and Jucheng Lin. Released under the [MIT License](LICENSE).

The paper studies the transition from nearly integrable to chaotic dynamics
in bilayer graphene (BLG) hexagonal cavities as the cavity boundary is
rotated with respect to the underlying lattice. This repository contains
three independent modules, one per methodological pillar of the paper:

| Module | Pillar | What it produces |
|---|---|---|
| `RAY_DYNAMICS/` | semiclassical ray dynamics | reflection-angle tables and Poincaré sections in a hexagonal cavity |
| `EIGENSTATES/` | quantum eigenstates | real- and momentum-space eigenstates of a BLG hexagonal quantum dot |
| `LEVEL_STATISTIC/` | level statistics | combined eigenenergies per irrep and r-value statistic across rotation angles |

Each module is self-contained and can be run independently.

---

## Requirements

```
python >= 3.9
numpy
scipy
matplotlib
pybinding   (only for EIGENSTATES/ and LEVEL_STATISTIC/)
```

`RAY_DYNAMICS/` has no `pybinding` dependency.

---

## A. `RAY_DYNAMICS/` — Semiclassical ray dynamics

Builds the anisotropic reflection map `phi_in -> phi_out` from the BLG
Fermi contour, then traces ray orbits inside a hexagonal cavity using the table.

### Files

| File | Role |
|---|---|
| `build_table_4band.py` | Build the reflection table from the 4-band BLG Hamiltonian (γ0–γ4). |
| `build_table_2band.py` | Same, for the simpler 2-band model with trigonal warping v3. |
| `extend_table.py` | Densify a reflection table using the y = -x time-reversal symmetry. |
| `run_hex_orbit.py` | Trace ray orbits in a hexagonal cavity using a precomputed table. |
| `orbit_panels.py` | Visualization helpers for `run_hex_orbit.py`. |
| `analyze_map.py` | Treat the reflection map as a 1D circle map; plot \|df/dφ\|. |
| `plot_poincare.py` | Render the Poincaré density figure. |

### Workflow

The four-step pipeline:

```
build_table_4band.py  →  extend_table.py  →  run_hex_orbit.py  →  plot_poincare.py
```

```bash
cd RAY_DYNAMICS

# 1) Build the reflection table on a numeric Fermi contour (0° and 15°).
python build_table_4band.py

# 2) Densify it by the y = -x symmetry.
python extend_table.py

# 3) Trace ray orbits in the hexagon → Poincaré CSV + SVG.
python run_hex_orbit.py

# 4) Render a Poincaré density plot from the CSV.
python plot_poincare.py
```

Optional analysis of the reflection map as a circle map:

```bash
python analyze_map.py
```

### `result_example/`

Contains a small set of reference outputs (figures only).

```
result_example/
├── fourband/E0.2_ang{0,15}_*/phi_in_out_360_sym_ynegx.png   # reflection table look-up
├── poincare_{0,15.0}/poincare_edge0.svg                     # Poincaré sections
└── calc_mag/reflect_magnification_E0.2_ang15_*.pdf          # |df/dφ| from analyze_map.py
```

The bulky binary tables (`reflect_table*.npz`) are **not** shipped. They
are produced on demand by step 1 above.

### Configuration

All scripts default to writing under `RAY_DYNAMICS/result_example/`.
To redirect output (or point at data on a cluster), set `BLG_RESULTS_DIR`:

```bash
BLG_RESULTS_DIR=/scratch/$USER/blg_runs python build_table_4band.py
```

The other env vars (`BLG_TABLE_TAG`, `BLG_NPZ_NAME`, `BLG_OUT_ROOT_4BAND`,
`BLG_POINCARE_DIR`, `BLG_FIGURES_DIR`, `BLG_NPZ_PATH`, `BLG_MAG_DIR`,
`BLG_MAG_TAG`) override individual paths within that root; defaults match
the layout the scripts produce, so no override is needed for the standard
workflow above.

---

## B. `EIGENSTATES/` — Quantum eigenstates of the BLG hexagonal dot

Solves the 4-band tight-binding Hamiltonian for a finite BLG hexagonal quantum dot (using `pybinding`) and computes a few hundred eigenstates
near a chosen energy.

The script below is the per-task building block of a cluster sweep used in
the paper; it is meant to run on a compute server and requires
environment-specific adjustments before use.

### Files

| File | Role |
|---|---|
| `Eigenstates.py` | Build the BLG dot, diagonalize with ARPACK, plot real-space probability densities and their Fourier transforms, and dump per-state `.npz` data. |


## C. `LEVEL_STATISTIC/` — Energy-level statistics

For each cavity rotation angle, this module diagonalizes the BLG hexagonal
dot, classifies the eigenstates by C3v irrep (or by C3 sector after
projecting out C2 in the tilted-hexagon case), and computes the spacing-ratio (r-value) statistic, which serves as the chaos diagnostic
in the paper.

The scripts below are per-task building blocks of a cluster sweep over
rotation angles used in the paper; they are meant to run on a compute
server and require environment-specific adjustments before use.

### Files

| File | Role |
|---|---|
| `Eigenenergy spectrum_unrotated.py` | Use the full C6v symmetry (degree = 0 or 30°). Eigenvalues are projected onto the six 1D irreps `A1g, A1u, A2g, A2u, Eg, Eu`, and r-values are computed per irrep. |
| `Eigenenergy spectrum_rotated.py` | For a rotated hexagon (generic `degree`). Only C3 remains; eigenvalues are split into `m=0` and `m=±1` C3 sectors and r-values are computed in each. |

import matplotlib
matplotlib.use('TkAgg')
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, cos, sin
from numpy import exp
import os
from matplotlib.ticker import MaxNLocator
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial import cKDTree, KDTree
from pybinding.constants import phi0

# =======================================
# === Parameters ========================
# =======================================
scale = 3
a = 0.24595 * scale
a_cc = a / sqrt(3)
radius = 100
r_buffer = 0 * scale
sigma = 0.15
k = 50
d = 0.348
gamma0 = -3.16 / scale
gamma1 = -0.381
gamma3 = -0.38 / scale
gamma4 = -0.14 / scale
t0 = 0
delta = 0
degree = 0
B = 0
U = 0
hardwall = False

deg_rad = degree * np.pi / 180
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(
    base_dir,
    "results_BLG_Hex_filterr",
    f"r{radius}_buffer{r_buffer}_U{U}_hardwall_{hardwall}_degree{degree}_scale{scale}_sigma{sigma}_k{k}_delta{delta}"
)
os.makedirs(output_dir, exist_ok=True)

# =======================================
# === Lattice definition =================
# =======================================
def bilayer_graphene():
    lat = pb.Lattice(a1=[a, 0, 0], a2=[a/2, sqrt(3)/2*a, 0])
    lat.add_sublattices(('B1', [0, 0, 0], 0),
                        ('A1', [0, -a_cc, 0], 0),
                        ('A2', [0, 0, d], 0),
                        ('B2', [0, a_cc, d], 0))
    lat.add_hoppings(([0, 0], 'B1', 'A1', gamma0),
                     ([0, 1], 'B1', 'A1', gamma0),
                     ([-1, 1], 'B1', 'A1', gamma0),
                     ([0, 0], 'B2', 'A2', gamma0),
                     ([0, 1], 'B2', 'A2', gamma0),
                     ([-1, 1], 'B2', 'A2', gamma0),
                     ([0, 0], 'B1', 'A2', gamma1),
                     ([0, -1], 'A1', 'B2', gamma3),
                     ([1, -1], 'A1', 'B2', gamma3),
                     ([1, -2], 'A1', 'B2', gamma3),
                     ([1, -1], 'B1', 'B2', gamma4),
                     ([0, 0], 'B1', 'B2', gamma4),
                     ([0, -1], 'B1', 'B2', gamma4),
                     ([1, -1], 'A1', 'A2', gamma4),
                     ([0, 0], 'A1', 'A2', gamma4),
                     ([0, -1], 'A1', 'A2', gamma4))
    return lat


def hexagon_points(R):
    return [[R*np.cos(pi/3*i + degree*pi/180), R*np.sin(pi/3*i + degree*pi/180)] for i in range(6)]


# =======================================
# === Onsite potential ===================
# =======================================
def onsite_total_hex(t0, delta, R_core, R_total, degree0, U, hardwall):
    @pb.onsite_energy_modifier
    def f(sub_id, x, y):
        result = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            xi0, yi0 = x[i], y[i]
            degree = degree0 * pi / 180
            xi = xi0*cos(degree) + yi0*sin(degree)
            yi = xi0*sin(degree) - yi0*cos(degree)
            if (abs(yi) <= R_core*sqrt(3)/2 and abs(yi) <= sqrt(3)*R_core - abs(xi)*sqrt(3) and abs(xi) <= R_core):
                result[i] = t0
                if sub_id in ['A1', 'B1']:
                    result[i] += U/2
                if sub_id in ['A2', 'B2']:
                    result[i] -= U/2
            elif (abs(yi) <= R_total*sqrt(3)/2 and abs(yi) <= sqrt(3)*R_total - abs(xi)*sqrt(3) and abs(xi) <= R_total):
                if not hardwall:
                    if sub_id in ['A1','B2']:
                        result[i] = t0 + delta/2
                    elif sub_id in ['B1','A2']:
                        result[i] = t0 - delta/2
                else:
                    result[i] = t0 + delta
        return result
    return f


# =======================================
# === Magnetic field (Peierls) ==========
# =======================================
def constant_magnetic_field(B):
    @pb.hopping_energy_modifier
    def func(energy, x1, y1, x2, y2):
        y = 0.5*(y1 + y2) * 1e-9
        A_x = B*y
        peierls = A_x*(x1-x2)*1e-9
        return energy * exp(1j*2*pi/phi0*peierls)
    return func


# =======================================
# === Build model and solve ============
# =======================================
lat = bilayer_graphene()
model = pb.Model(lat, pb.Polygon(hexagon_points(R=radius+r_buffer)),
                 onsite_total_hex(t0, delta, radius, radius+r_buffer,
                                  degree, U, hardwall),
                 constant_magnetic_field(B))

# Solve with arpack (shift-invert)
solver = pb.solver.arpack(model, k=k, sigma=sigma)
eigenvalues = solver.calc_eigenvalues()   # pybinding solver object
eigenvectors = solver.eigenvectors
# eigenvalues.values is array of eigenenergies
vals = eigenvalues.values.copy()
vecs = eigenvectors.copy()

# =======================================
# === Get positions and sublattice info ===
# =======================================
prob_map = solver.calc_probability(0)
x, y = prob_map.x, prob_map.y
positions = np.column_stack((x, y))
sub_id = np.array(prob_map.sub)
atom_type = sub_id.copy()
N_sites = len(positions)
N_states = vecs.shape[1]

# Precompute center
center = np.array([0.0, 0.0])
tol_match = 1e-4  # KDTree match tolerance (adjust if necessary)

# =======================================
# === Build C3 rotation operator =========
# =======================================
def build_rotation_operator(positions, center, angle=2*pi/3, tol=1e-4):
    tree = KDTree(positions)
    N = len(positions)
    R = lil_matrix((N, N), dtype=complex)
    Rmat = np.array([[cos(angle), -sin(angle)],
                     [sin(angle),  cos(angle)]])
    for i in range(N):
        p = positions[i]
        p_rot = center + Rmat @ (p - center)
        dists, js = tree.query(p_rot, k=4)  # try up to 4 neighbors
        if np.isscalar(dists):
            dists = np.array([dists]); js = np.array([js])
        # prefer same sublattice (atom_type)
        mask = atom_type[js] == atom_type[i]
        if not np.any(mask):
            # fallback: accept nearest even if different sublattice
            j = js[np.argmin(dists)]
            if np.min(dists) < tol:
                R[j, i] = 1.0
        else:
            j_choice = js[mask][np.argmin(dists[mask])]
            if np.min(dists[mask]) < tol:
                R[j_choice, i] = 1.0
    return R.tocsr()

R_op = build_rotation_operator(positions, center, angle=2*pi/3, tol=tol_match)

# =======================================
# === Identify m (rotation) eigenvalues ==
# =======================================
# For each eigenstate compute <psi|R|psi> and deduce which of {1, e^{i2pi/3}, e^{-i2pi/3}} it is closest to.
m_labels = np.full(N_states, None)
targets = np.array([1.0, np.exp(1j*2*pi/3), np.exp(-1j*2*pi/3)])
for idx in range(N_states):
    wf = vecs[:, idx]
    ov = np.vdot(wf, R_op.dot(wf))
    # normalize phase
    if np.abs(ov) < 1e-12:
        m_labels[idx] = 'unknown'
        continue
    # find which target is nearest in phase
    diffs = [abs(ov - t) for t in targets]
    t_idx = int(np.argmin(diffs))
    if t_idx == 0:
        m_labels[idx] = 'm0'   # rotation eigenvalue 1
    elif t_idx == 1:
        m_labels[idx] = 'm+1'  # e^{i2pi/3}
    else:
        m_labels[idx] = 'm-1'  # e^{-i2pi/3}

m0_indices = [i for i, lab in enumerate(m_labels) if lab == 'm0']
m_pm_indices = [i for i, lab in enumerate(m_labels) if lab in ('m+1','m-1')]
print(f"Found {len(m0_indices)} states with m=0, and {len(m_pm_indices)} with m=±1.")

# =======================================
# === Build inversion operator (S=i) ===
# =======================================
# Using sublattice mapping for inversion if necessary (you had S_map)
# we reuse that mapping if consistent with your lattice ordering.
S_map = {0:2, 1:3, 2:0, 3:1}  # adapt if your sublattice ids differ
trees = []
idx_arrays = []
for target_id in range(4):
    mask = (sub_id == target_id)
    trees.append(cKDTree(positions[mask]))
    idx_arrays.append(np.where(mask)[0])

rows, cols, data_vals = [], [], []
positions_inv = 2*center - positions
mask_int = np.array(sub_id, dtype=int)
for i in range(N_sites):
    si = mask_int[i]
    j_target = S_map[si]
    dist, idx_loc = trees[j_target].query(positions_inv[i])
    if dist < 1e-2:
        j = idx_arrays[j_target][idx_loc]
        rows.append(j)
        cols.append(i)
        data_vals.append(1.0)
    else:
        # unmatched point -> leave unmapped (sparse operator)
        pass
S_op = csr_matrix((data_vals, (rows, cols)), shape=(N_sites, N_sites), dtype=complex)

# =======================================
# === Build mirror operator σ (x->-x) ===
# =======================================
tree_global = cKDTree(positions)
rows, cols, data_vals = [], [], []
positions_sigma = positions.copy()
positions_sigma[:, 0] = 2*center[0] - positions_sigma[:, 0]
for i in range(N_sites):
    dist, j = tree_global.query(positions_sigma[i])
    if dist < 1e-2:
        rows.append(j)
        cols.append(i)
        data_vals.append(1.0)
    else:
        # warn for debugging
        # print(f"mirror: site {i} not matched (dist {dist})")
        pass
Sigma_op = csr_matrix((data_vals, (rows, cols)), shape=(N_sites, N_sites), dtype=complex)

# =======================================
# === Compute S and σ eigenvalues for states ===
# =======================================
S_vals = np.full(N_states, 0, dtype=int)    # +1 or -1 (or 0 if undefined)
Sigma_vals = np.full(N_states, 0, dtype=int)

for idx in range(N_states):
    wf = vecs[:, idx]
    # S
    vS = S_op.dot(wf)
    ovS = np.vdot(wf, vS)
    # ovS ideally real; check magnitude
    if np.isfinite(ovS) and np.abs(ovS) > 1e-12:
        S_vals[idx] = 1 if np.real(ovS) > 0 else -1
    else:
        S_vals[idx] = 0  # undefined / not matched
    # Sigma
    vSig = Sigma_op.dot(wf)
    ovSig = np.vdot(wf, vSig)
    if np.isfinite(ovSig) and np.abs(ovSig) > 1e-12:
        Sigma_vals[idx] = 1 if np.real(ovSig) > 0 else -1
    else:
        Sigma_vals[idx] = 0

# =======================================
# === Classify irreps ===================
# For m=0: A1g (S=+1, σ=+1), A2g (S=+1, σ=-1), A1u (S=-1, σ=+1), A2u (S=-1, σ=-1)
# For m=±1: Eg if S=+1 (even under inversion), Eu if S=-1
# =======================================
irrep_labels = np.array(['unknown'] * N_states, dtype=object)
for idx in range(N_states):
    mlab = m_labels[idx]
    s = S_vals[idx]
    sig = Sigma_vals[idx]
    if mlab == 'm0':
        if s == 1 and sig == 1:
            irrep_labels[idx] = 'A1g'
        elif s == 1 and sig == -1:
            irrep_labels[idx] = 'A2g'
        elif s == -1 and sig == 1:
            irrep_labels[idx] = 'A1u'
        elif s == -1 and sig == -1:
            irrep_labels[idx] = 'A2u'
        else:
            irrep_labels[idx] = 'm0_unknown'
    elif mlab in ('m+1','m-1'):
        if s == 1:
            irrep_labels[idx] = 'Eg'
        elif s == -1:
            irrep_labels[idx] = 'Eu'
        else:
            irrep_labels[idx] = 'm±1_unknown'
    else:
        irrep_labels[idx] = 'unknown'

# =======================================
# === Save eigenvalue arrays by label ===
# =======================================
np.save(os.path.join(output_dir, "all_eigenvalues.npy"), vals)
np.save(os.path.join(output_dir, "irrep_labels.npy"), irrep_labels)

# Save grouped energies
unique_irreps = np.unique(irrep_labels)
for rep in unique_irreps:
    mask = (irrep_labels == rep)
    if np.any(mask):
        fname = os.path.join(output_dir, f"eigs_{rep}.npy")
        np.save(fname, vals[mask])

# Save m0 and m0 S=+1 as before
np.save(os.path.join(output_dir, "m0_eigenvalues.npy"), vals[m0_indices])
m0_S1_indices = [i for i in m0_indices if S_vals[i] == 1]
np.save(os.path.join(output_dir, "m0_S1_eigenvalues.npy"), vals[m0_S1_indices])

print("Saved eigenvalue files to", output_dir)
print("Irrep counts:")
for rep in unique_irreps:
    print(f"  {rep}: {np.sum(irrep_labels==rep)}")

# =======================================
# === Plot eigenvalues colored by irrep ==
# =======================================
# Create color map for some common irreps
color_map = {
    'A1g': 'green',
    'A2g': 'lime',
    'A1u': 'blue',
    'A2u': 'cyan',
    'Eg' : 'red',
    'Eu' : 'magenta',
    'm0_unknown': 'black',
    'm±1_unknown': 'black',
    'unknown': 'gray'
}

plt.figure(figsize=(8,5))
x_all = np.arange(len(vals))
plt.scatter(x_all, vals, s=10, color='lightgray', label='all')

# overlay per irrep
for rep in unique_irreps:
    mask = (irrep_labels == rep)
    if not np.any(mask):
        continue
    c = color_map.get(rep, 'k')
    plt.scatter(x_all[mask], vals[mask], s=18, color=c, label=rep)

plt.xlabel("State index")
plt.ylabel("Energy (eV)")
plt.title("Eigenvalues labeled by D3d irreps (m & S & σ)")
plt.legend(markerscale=2, bbox_to_anchor=(1.02,1.0), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eigenvalues_irreps.png"), dpi=300)
plt.show()

# =======================================
# === Optional: print sample diagnostics ==
# =======================================
# Show a few examples for manual inspection
for i in range(min(20, N_states)):
    print(f"idx {i:4d} E={vals[i]: .6f}  m={m_labels[i]:5s}  S={S_vals[i]:2d}  σ={Sigma_vals[i]:2d}  irrep={irrep_labels[i]}")
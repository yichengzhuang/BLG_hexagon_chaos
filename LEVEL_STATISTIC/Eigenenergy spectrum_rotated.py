import matplotlib
matplotlib.use('TkAgg')
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, cos, sin
from numpy import exp
import os
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
degree=15
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

solver = pb.solver.arpack(model, k=k, sigma=sigma)
eigenvalues = solver.calc_eigenvalues()
vecs = solver.eigenvectors
vals = eigenvalues.values.copy()

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

center = np.array([0.0, 0.0])
tol_match = 1e-4

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
        dists, js = tree.query(p_rot, k=4)
        if np.isscalar(dists):
            dists = np.array([dists]); js = np.array([js])
        mask = atom_type[js] == atom_type[i]
        if not np.any(mask):
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
m_labels = np.full(N_states, None)
targets = np.array([1.0, np.exp(1j*2*pi/3), np.exp(-1j*2*pi/3)])
for idx in range(N_states):
    wf = vecs[:, idx]
    ov = np.vdot(wf, R_op.dot(wf))
    if np.abs(ov) < 1e-12:
        m_labels[idx] = 'unknown'
        continue
    diffs = [abs(ov - t) for t in targets]
    t_idx = int(np.argmin(diffs))
    if t_idx == 0:
        m_labels[idx] = 'm0'
    elif t_idx == 1:
        m_labels[idx] = 'm+1'
    else:
        m_labels[idx] = 'm-1'

# =======================================
# === Build inversion operator (S=i) ===
# =======================================
S_map = {0:2, 1:3, 2:0, 3:1}
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
S_op = csr_matrix((data_vals, (rows, cols)), shape=(N_sites, N_sites), dtype=complex)

S_vals = np.full(N_states, 0, dtype=int)
for idx in range(N_states):
    wf = vecs[:, idx]
    vS = S_op.dot(wf)
    ovS = np.vdot(wf, vS)
    if np.isfinite(ovS) and np.abs(ovS) > 1e-12:
        S_vals[idx] = 1 if np.real(ovS) > 0 else -1
    else:
        S_vals[idx] = 0

# =======================================
# === Simplified irrep classification: C3 + S only ===
# Ag: m=0, S=+1
# Au: m=0, S=-1
# Eg: m=±1, S=+1
# Eu: m=±1, S=-1
# =======================================
simple_labels = np.array(['unknown']*N_states, dtype=object)
for idx in range(N_states):
    mlab = m_labels[idx]
    s = S_vals[idx]
    if mlab == 'm0':
        if s == 1:
            simple_labels[idx] = 'Ag'
        elif s == -1:
            simple_labels[idx] = 'Au'
    elif mlab in ('m+1','m-1'):
        if s == 1:
            simple_labels[idx] = 'Eg'
        elif s == -1:
            simple_labels[idx] = 'Eu'

# =======================================
# === Save eigenvalues by simple label ===
# =======================================
np.save(os.path.join(output_dir, "simple_labels.npy"), simple_labels)
for rep in ['Ag','Au','Eg','Eu']:
    mask = simple_labels == rep
    if np.any(mask):
        np.save(os.path.join(output_dir, f"eigs_{rep}.npy"), vals[mask])

# =======================================
# === Plot eigenvalues colored by simple irrep ===
# =======================================
color_map_simple = {'Ag':'green','Au':'blue','Eg':'red','Eu':'magenta','unknown':'gray'}
plt.figure(figsize=(8,5))
x_all = np.arange(len(vals))
plt.scatter(x_all, vals, s=10, color='lightgray', label='all')
for rep in ['Ag','Au','Eg','Eu']:
    mask = simple_labels == rep
    if np.any(mask):
        plt.scatter(x_all[mask], vals[mask], s=18, color=color_map_simple[rep], label=rep)

plt.xlabel("State index")
plt.ylabel("Energy (eV)")
plt.title("Eigenvalues labeled by C3 + inversion (Ag, Au, Eg, Eu)")
plt.legend(markerscale=2, bbox_to_anchor=(1.02,1.0), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eigenvalues_simple_irreps.png"), dpi=300)
plt.show()

print("Saved simple irrep eigenvalues to", output_dir)
for rep in ['Ag','Au','Eg','Eu']:
    print(f"{rep}: {np.sum(simple_labels==rep)} states")
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for displaying plots
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, cos, sin
import matplotlib.colors as mcolors
import os
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata
from numpy.fft import fftshift, fft2
from pybinding.constants import phi0
from numpy import exp, sqrt,pi

# === Parameters ===
scale=3
a = 0.24595*scale  # [nm] unit cell length
a_cc = a / sqrt(3)  # [nm] carbon-carbon distance
radius = 80  # [nm] radius of quantum dot
r_buffer =0*scale # [nm] width of buffer region
sigma = 0.23  # [eV] center energy around which eigenstates are computed
k = 20  # number of eigenstates to calculate
a = 0.24595*scale  # [nm] unit cell length
a_cc = a / sqrt(3)  # [nm] carbon-carbon distance
d = 0.348  # [nm] distance between two layers
gamma0 = -3.16/scale  # [eV] hopping energy between A1 and B1
gamma1 = -0.381  # [eV] hopping energy between B1 and A2
gamma3 = -0.38/scale  # [eV] hopping energy between A1 and B2
gamma4 = -0.14/scale  # [eV] hopping energy between A1 and A2 & B1 and B2
t0 = 0  # [eV] onsite energy
delta = 0 # [eV] energy difference
degree=15
B = 0.00            # Magnetic field strength in Tesla
U = -0.0/scale
hardwall=False
dy=a_cc/2*0
dx=a_cc/2*0

# degree: rotation angle in degrees
deg_rad = degree * np.pi / 180

# Output folder for results
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(
    base_dir,
    "results_BLG_Hex_buffer",
    f"r{radius}_buffer{r_buffer}_dy{dy/a_cc}_U{U}_hardwall_{hardwall}_degree{degree}_scale{scale}_sigma{sigma}_k{k}_delta{delta}_B{B}"
)
os.makedirs(output_dir, exist_ok=True)

def bilayer_graphene():
    lat = pb.Lattice(
        a1=[a, 0, 0],
        a2=[a / 2, sqrt(3)/2 * a, 0]
    )

    lat.add_sublattices(
        ('B1', [0, 0+dy, 0], t0),
        ('A1', [0, -a_cc+dy, 0], t0),
        ('A2', [0, 0+dy, d], t0),
        ('B2', [0, a_cc+dy, d], t0)
    )

    lat.add_hoppings(
        ([0, 0], 'B1', 'A1', gamma0),
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
        ([0, -1], 'A1', 'A2', gamma4)
    )

    return lat

def hexagon_points(R):
    return [[R*np.cos(pi/3*i+degree*pi/180), R*np.sin(pi/3*i+degree*pi/180)] for i in range(6)]

# === Define buffer region ===
def onsite_total_hex(t0, delta, R_core, R_total, degree0, U, hardwall):
    @pb.onsite_energy_modifier
    def f(sub_id, x, y):
        result = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            xi0, yi0 = x[i], y[i]
            degree=degree0*pi/180
            xi=xi0*cos(degree)+yi0*sin(degree)
            yi=xi0*sin(degree)-yi0*cos(degree)

            # --- core: quadratic potential inside hexagon ---
            if (abs(yi) <= R_core*sqrt(3)/2 and abs(yi) <= sqrt(3)*R_core - abs(xi)*sqrt(3) and abs(xi) <= R_core):
                result[i] = t0
                if sub_id == 'A1' or sub_id == 'A2':
                    result[i] -= U/2
                if sub_id == 'B1' or sub_id == 'B2':
                    result[i] += U/2
            # --- buffer: sublattice potential inside larger hexagon ---
            elif (abs(yi) <= R_total*sqrt(3)/2 and abs(yi) <= sqrt(3)*R_total - abs(xi)*sqrt(3) and abs(xi) <= R_total):
                if not hardwall:
                    if sub_id == 'A1' or sub_id == 'A2':
                        result[i] = t0 + delta / 2
                    elif sub_id == 'B1' or sub_id == 'B2':
                        result[i] = t0 - delta / 2
                else:
                    result[i] = t0 + delta
        return result

    return f

# === Define uniform magnetic field using Peierls substitution ===
def constant_magnetic_field(B):
    """
    Returns a hopping energy modifier to include a uniform magnetic field via Peierls substitution.
    This example currently does not apply a phase (exp(0j)).
    """
    @pb.hopping_energy_modifier
    def function(energy, x1, y1, x2, y2):
        y = 0.5 * (y1 + y2) * 1e-9  # Midpoint y position in meters
        A_x = B * y                 # Vector potential in Landau gauge (A_x = B * y)
        peierls = A_x * (x1 - x2) * 1e-9  # Line integral of A * dl (meters)
        return energy * exp(1j * 2*pi/phi0 * peierls)     # Placeholder: no phase applied yet
    return function


# === Create the model ===
lat = bilayer_graphene()
model = pb.Model(lat, pb.Polygon(hexagon_points(R=radius+r_buffer)), constant_magnetic_field(B),onsite_total_hex(t0, delta, radius, radius+r_buffer, degree, U, hardwall))

# === Solve eigenvalues ===
solver = pb.solver.arpack(model, k=k, sigma=sigma)
eigenvalues = solver.calc_eigenvalues()

# === Save eigenvalue spectrum plot ===
plt.figure()
x = np.arange(len(eigenvalues.values))  # integer indices
y = eigenvalues.values
plt.plot(x, y, marker='o', linestyle='none')

plt.title("Eigenvalues")
plt.ylabel("Energy (eV)")
plt.xlabel("State Index")

# Force integer ticks on X-axis
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(f"{output_dir}/eigenvalues.png", dpi=500)
plt.close()

# === Parameters for subplot grids ===
states_per_fig = 4
cols = 2
rows = 2
grid_size = 256  # resolution for Fourier transform grid

for fig_start in range(0, len(eigenvalues.values), states_per_fig):
    # === Real-space figure ===
    fig_real, axes_real = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes_real = axes_real.flatten()

    # === Momentum-space figure ===
    fig_k, axes_k = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes_k = axes_k.flatten()

    for i in range(states_per_fig):
        state_index = fig_start + i
        if state_index >= len(eigenvalues.values):
            # remove unused axes
            for j in range(i, states_per_fig):
                fig_real.delaxes(axes_real[j])
                fig_k.delaxes(axes_k[j])
            break

        energy = eigenvalues.values[state_index]

        # === Real-space probability ===
        prob_map = solver.calc_probability(state_index)
        x, y, z = prob_map.x, prob_map.y, prob_map.data  # psi is complex
        wf = solver.eigenvectors[:, state_index]
        # If available, get sublattice info
        try:
            sub_ids = prob_map.sub  # array of 'A1', 'B1', 'A2', 'B2'
        except AttributeError:
            # If prob_map has no sublattice, generate dummy array
            sub_ids = np.array(['A1']*len(x))  # replace as needed
            print("No Sublattice")

        ax_r = axes_real[i]
        scatter = ax_r.scatter(x, y, c=z, cmap='turbo', s=2, edgecolors='none')
        ax_r.set_title(f"State {state_index}\n{energy:.6f} eV", fontsize=8)
        ax_r.set_xlabel("x (nm)", fontsize=6)
        ax_r.set_ylabel("y (nm)", fontsize=6)
        ax_r.tick_params(axis='both', which='both', labelsize=6)
        ax_r.set_aspect('equal')

        # === Fourier transform (momentum-space) ===
        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        X, Y = np.meshgrid(xi, yi)
        psi_grid = griddata((x, y), wf, (X, Y), method="cubic", fill_value=0)
        fft_psi = fftshift(fft2(psi_grid))
        fft_intensity = np.abs(fft_psi) ** 2

        ax_k = axes_k[i]
        im = ax_k.imshow(
            fft_intensity,
            extent=[-np.pi/a, np.pi/a, -np.pi/a, np.pi/a],
            cmap="inferno",
            origin="lower"
        )
        ax_k.set_title(f"FT of State {state_index}", fontsize=8)
        ax_k.set_xlabel("kx (a.u.)", fontsize=6)
        ax_k.set_ylabel("ky (a.u.)", fontsize=6)
        ax_k.tick_params(axis='both', which='both', labelsize=6)

        # === Rotate coordinates for core mask ===
        x_rot = x * np.cos(deg_rad) + y * np.sin(deg_rad)
        y_rot = -x * np.sin(deg_rad) + y * np.cos(deg_rad)

        # === Mask for core hexagon ===
        mask_core = (
            (np.abs(y_rot) <= radius * np.sqrt(3)/2) &
            (np.abs(y_rot) <= np.sqrt(3)*radius - np.abs(x_rot)*np.sqrt(3)) &
            (np.abs(x_rot) <= radius)
        )

        # === Filtered coordinates, wavefunction, and sublattice ===
        x_core = x[mask_core]
        y_core = y[mask_core]
        wf_core = wf[mask_core]
        prob_core = z[mask_core]
        sub_core = sub_ids[mask_core]

        # === Save data to .npz ===
        np.savez(
            os.path.join(output_dir, f"state_{state_index:03d}.npz"),
            x=x_core,
            y=y_core,
            psi=wf_core,
            prob=prob_core,
            sub_id=sub_core,
            energy=energy
        )

    # === Save real-space figure ===
    fig_real.suptitle(
        f"Real-space Eigenstates {fig_start}–{min(fig_start + states_per_fig - 1, len(eigenvalues.values) - 1)}",
        fontsize=14
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig_real.savefig(f"{output_dir}/states_real_{fig_start}_{fig_start + states_per_fig - 1}.png", dpi=500)
    plt.close(fig_real)

    # === Save momentum-space figure ===
    fig_k.suptitle(
        f"Momentum-space (FT) of Eigenstates {fig_start}–{min(fig_start + states_per_fig - 1, len(eigenvalues.values) - 1)}",
        fontsize=14
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig_k.savefig(f"{output_dir}/states_fft_{fig_start}_{fig_start + states_per_fig - 1}.png", dpi=500)
    plt.close(fig_k)

# === Save all eigenvalues and eigenvectors globally ===
np.save(os.path.join(output_dir, "eigenvalues.npy"), eigenvalues.values)
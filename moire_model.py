# moire_model.py # latex notes in colab

from math import erfc, sqrt, pi, exp
import numpy as np

# -------- user-adjustable constants --------
a_m   = 8.031          # nm   moiré lattice constant, supercell length, from paper (the lattice mismatch between WSe2 and WS2)
V0    = 15.0         # meV  from paper

eps_r = 5.0         # dielectric constant in paper
e2_4pieps0 = 14.399645  # meV·nm (|e|²/4πϵ0)
hbar2_over_2m = 108.857 # meV·nm² in paper
phi = np.pi/4   # phase of the moiré potential from paper
n_sup = 3            # 3×3 super-cell
N_e   = 6               # # electrons (= # occupied orbitals)
cut   = 2  # |G|≤cut·|g| # cutoff for the plane-wave basis (G vectors) in the Hamiltonian

# --- Ewald parameters (renamed) ---
ew_alpha = 0.35      # nm⁻²  (splitting)
r_cut    = 2.5       # real-space cutoff in L units
k_cut    = 5         # k-space cutoff in 2π/L units
L     = n_sup * a_m  # nm   PBC box length used in Ewald

def a_vectors(a_m): # real supercell lattice vectors
    """Generates the 3 shortest moiré reciprocal vectors G_1,2,3, 60° apart, six-fold symmetry"""
    a1 = a_m * np.array([1.0,                0.0           ])
    a2 = a_m * np.array([0.5,  np.sqrt(3.0) / 2.0          ])
    a3 = -(a1 + a2)        # optional third vector (120° w.r.t. a1)
    return [a1, a2, a3]    # same interface style as your b_vectors()

def b_vectors(a_m): # reciprocal supercell lattice vectors
    """from paper: g_j = (4*pi / sqrt(3) / a_m) * [cos(2*pi*j/3), sin(2*pi*j/3)], for j=1,2,3"""
    g_list = []
    prefac = 4 * np.pi / (np.sqrt(3) * a_m)
    for j in range(1, 4):  # j = 1, 2, 3
        angle = 2 * np.pi * j / 3
        g = prefac * np.array([np.cos(angle), np.sin(angle)])
        g_list.append(g)
    print("g list", g_list)
    return g_list  # returns [g1, g2, g3]

def moire_potential(r, a_m = a_m, V0 = V0, phi = phi):
    """ V(r) = -2*V0*sum_{j=1}^{3} cos(g_j · r + phi)where g_j are 3 reciprocal lattice vectors (from paper)."""
    G = np.array(b_vectors(a_m))  # Get the three reciprocal vectors, shape (3,2)
    phase = np.dot(r, G.T) + phi  # r @ G.T + phi
    one_electron_moire = -2 * V0 * np.sum(np.cos(phase), axis=-1)
    return np.sum(one_electron_moire)

# ---------- Interaction part ----------
def pairwise_real_space(R, alpha=ew_alpha, r_lim=r_cut, L=L):
    """ Short‐range (real‐space) Ewald sum (equation A6 from the paper):
    E_real = ½ ∑_{i≠j} ∑_L erfc(√α·r_{ij}^L) / r_{ij}^L., 
    where α = 1/(4η²), and r_{ij}^L = |r_i - r_j + L|"""
    N, E = len(R), 0.0 #N:number particles
    maxn = int(np.ceil(r_lim)) # summing over neighbor cells from -max_n to +max_n
    for i in range(N):
        for j in range(i+1,N): # loop over ½ ∑_{i≠j}
            for nx in range(-maxn,maxn+1): #loop over n_x n_y
                for ny in range(-maxn,maxn+1):
                    dr = R[i]-R[j]+np.array([nx,ny])*L # dr = r_i - r_j + n·L
                    r  = np.linalg.norm(dr) # r = |dr|
                    if r<1e-9 or r>r_lim*L: continue # Skip self‐interaction (r≈0) or beyond cutoff r_lim·L
                    E += erfc(sqrt(alpha)*r)/r # α = 1/(4η²) we choose
    return E

def structure_factor(R,k):    # Σ e^{ik·r}
    """Structure factor S(k) = Σ e^{ik·r} (sum over all particles)"""
    phase = R @ k
    return np.sum(np.cos(phase))+1j*np.sum(np.sin(phase))

def reciprocal_space(R, alpha=ew_alpha, k_lim=k_cut, L=L):
    """ Long‐range (reciprocal‐space) Ewald sum (equation A7 and A10 from the paper):
    E_recip = (π/V) ∑_{k≠0} [ e^{-k²/(4α)} / k² ] |S(k)|²
    with V = L² in 2D, fast convergence."""
    area, k0 = L*L, 2*pi/L
    E=0.0
    # 2) Sum over discrete wavevectors q = (m_x, m_y)·(2π/L), the paper has ∑_{q≠0};
    # here we loop m_x, m_y ∈ [−k_lim,…,+k_lim]
    for mx in range(-k_lim,k_lim+1):
        for my in range(-k_lim,k_lim+1):
            if mx==0 and my==0: continue # skip the q = 0 term
            k = np.array([mx,my])*k0 # q_vec = (m_x, m_y)·(2π/L)
            k2 = k@k # q² = |q_vec|² = q_x² + q_y²
            E += exp(-k2/(4*alpha))*abs(structure_factor(R,k))**2 / k2 # factor e^{–q²/(4α)} # e^{-q²/(4α)}/q² · |S(q)|²
    return (pi/area)*E

def self_energy(N, alpha=ew_alpha):
    """ (equation A12 from the paper, Madelung constant) Self‐interaction correction: E_self = - ∑_i (√α / √π) · q_i²
    Here q_i are unit charges, so E_self = -N·(√α/√π)."""
    return -sqrt(alpha/pi)*N

def madelung_offset(alpha=ew_alpha,                 # α = 1/(4η²)
                    r_lim=r_cut, k_lim=k_cut, L=L):
    """ Compute ξ_M in Eq. (A12) Returns a scalar (dimensionless).  Multiply by e²/4πϵ₀ϵ_r later."""
    eta   = 0.5 / np.sqrt(alpha)     # because α = 1/(4η²)
    area  = L * L
    k0    = 2.0 * np.pi / L
    # ---- Real-space images   Σ_{L≠0} erfc(|L|/2η)/|L|
    rsum = 0.0
    maxn = int(np.ceil(r_lim))
    for nx in range(-maxn, maxn + 1):
        for ny in range(-maxn, maxn + 1):
            if nx == 0 and ny == 0:
                continue
            Rvec = np.array([nx, ny]) * L
            R    = np.linalg.norm(Rvec)
            if R > r_lim * L:
                continue
            rsum += erfc(R / (2.0 * eta)) / R
    # ---- Reciprocal-space images   (2π/Area) Σ_{G≠0} e^{−η²G²}/G
    ksum = 0.0
    for mx in range(-k_lim, k_lim + 1):
        for my in range(-k_lim, k_lim + 1):
            if mx == 0 and my == 0:
                continue
            Gvec = np.array([mx, my]) * k0
            G    = np.linalg.norm(Gvec)
            ksum += np.exp(-(eta * G) ** 2) / G
    ksum *= 2.0 * np.pi / area
    xi0_L = 1.0 / (eta * np.sqrt(np.pi))     # ---- ξ^L_0 term   1 / (η √π)
    return rsum + ksum - xi0_L     # ---- ξ_M (Eq. A12)

ξ_M = madelung_offset() * e2_4pieps0 / eps_r # compute once and store — units:   meV

def coulomb_ewald_2D(R):
    """
    Full 2-D Ewald energy for the set of positions R (shape (N,2)).
    Now includes ½ Σ_b ξ_M  so the result matches Eq. (A11) exactly.
    E_total = (e²/(4πϵ₀ ε_r))·( E_real + E_recip + E_self )
    """
    N  = len(R)
    # position-dependent part (your original implementation)
    E_config = (pairwise_real_space(R) +
                reciprocal_space(R)   +
                self_energy(N)) * e2_4pieps0 / eps_r

    # constant Madelung shift
    return E_config + 0.5 * N * ξ_M


def energy_static(R):
    """V_ext + V_ee  (independent of Ψ)."""
    return moire_potential(R) + coulomb_ewald_2D(R)

energy_moire = energy_static






# # run below to test the code

# # # ------------------------------------------------------------------------------
# print("== Unit tests for energy_static =========================================")

# # 1. single electron at Γ (no e–e term)
# r0 = np.array([[0.0, 0.0]])
# print("1-electron Γ-point :",
#       energy_static(r0), "meV   (should equal V_ext at Γ)")

# # 2. two electrons opposite corners of the super-cell  (max. separation L√2/2)
# r2 = np.array([[0.0, 0.0],
#                [0.5*L, 0.5*L]])        # nm
# E2_ext = moire_potential(r2).sum()
# E2_ee  = coulomb_ewald_2D(r2)
# print("2-electron test     : ext =", E2_ext, "  Coul =", E2_ee,
#       "  total =", energy_static(r2), "meV")

# # 3. six electrons at random positions in the 3×3 box  (paper’s occupation)
# np.random.seed(0)
# r6 = np.random.rand(6, 2) * L
# print("6-electron random   :", energy_static(r6), "meV")
# print("coordinates (nm):\n", r6)

# # 4. translational invariance check (shift by L/3, L/7)
# shift = np.array([L/3, L/7])       # arbitrary lattice-vector shift
# print("translational check :", energy_static(r6),
#       " ≟ ", energy_static(r6+shift))

# # 5. translational invariance check (shift by a1, a2)
# a1, a2 = a_vectors(a_m)
# shift  = 2*a1 - 1*a2          # any integer combination works
# print("translational check2: ",energy_static(r6), energy_static(r6+shift))  # should match to 1e-8
# # Now the two numbers will coincide, confirming full PBC consistency.





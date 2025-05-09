from math import erfc, sqrt, pi, exp, cos, sin
import numpy as np

# tunable physical constants:
a_m      = 8.5     # nm  – moiré lattice constant  (paper Table I)
V0       = 15.0    # meV – moiré-potential depth   (paper Eq. 1)
e2_4pieps0 = 14.399645  # meV·nm
L       = 8.5     # nm
eps_r   = 10.0    # dielectric constant
alpha   = 0.35    # nm⁻² (Gaussian‐screening parameter)
r_cut   = 2.5     # in units of L, real‐space cutoff
k_cut   = 5       # in units of 2π/L, k‐space cutoff


# # reciprocal-lattice vectors (length |G|=4π/√3a)
# # Generates the 3 shortest moiré reciprocal vectors G_1,2,3, 60° apart, six-fold symmetry:
def reciprocal_vectors(a):
    g = 4.0 * np.pi / (np.sqrt(3.0) * a) # magnitude
    G1 = np.array([g, 0.0]) # first vector along x-axis
    angle60 = np.pi/3
    c, s = np.cos(angle60), np.sin(angle60) # precompute cosine and sine
    G2 = np.array([ c*G1[0] - s*G1[1], s*G1[0] + c*G1[1] ]) # rotate G1 by +60°, get G2
    G3 = np.array([ c*G1[0] + s*G1[1],-s*G1[0] + c*G1[1] ]) # rotate G1 by –60°, get G3
    return np.stack([G1, G2, G3])  # shape (3,2), rows are G₁, G₂, G₃
G = reciprocal_vectors(a_m)                  # cache for speed


# external moiré potential  V(r) = V0 Σ_i cos(G_i·r) (paper Eq. 1)
def moire_potential(r, V0=V0):
    return V0 * np.sum(np.cos(r @ G.T), axis=-1)


def pairwise_real_space(R, alpha=alpha, r_lim=r_cut, L=L):
    """
    Short‐range (real‐space) Ewald sum (equation A6 from the paper):
      E_real = ½ ∑_{i≠j} ∑_L erfc(√α·r_{ij}^L) / r_{ij}^L., where α = 1/(4η²), and r_{ij}^L = |r_i - r_j + L|
    """
    N = len(R)    #number particles
    E_real = 0.0
    max_n = int(np.ceil(r_lim)) # summing over neighbor cells from -max_n to +max_n
    for i in range(N):
        for j in range(i + 1, N): # loop over ½ ∑_{i≠j}
            for nx in range(-max_n, max_n + 1):
                for ny in range(-max_n, max_n + 1): #loop over n_x n_y
                    dr = R[i] - R[j] + np.array([nx, ny]) * L  # dr = r_i - r_j + n·L
                    r = np.linalg.norm(dr)    # r = |dr|
                    if r < 1e-9 or r > r_lim * L:  # Skip self‐interaction (r≈0) or beyond cutoff r_lim·L
                        continue
                    E_real += erfc(sqrt(alpha) * r) / r # α = 1/(4η²) we choose
    return E_real



def structure_factor(R, kvec):
    """
    Fourier component: S(k) = ∑_j e^{-i k · r_j} Computed via cos/sin: ∑ cos(k·r) + i ∑ sin(k·r)
    """
    phase = R @ kvec
    return np.sum(np.cos(phase)) + 1j * np.sum(np.sin(phase))



def reciprocal_space(R, alpha=alpha, k_lim=k_cut, L=L):
    """
    Long‐range (reciprocal‐space) Ewald sum (equation A7 and A10 from the paper):
      E_recip = (π/V) ∑_{k≠0} [ e^{-k²/(4α)} / k² ] |S(k)|²
    with V = L² in 2D, fast convergence.
    """
    area = L * L
    k0 = 2.0 * pi / L
    E_k = 0.0

    # 2) Sum over discrete wavevectors q = (m_x, m_y)·(2π/L), the paper has ∑_{q≠0};
    # here we loop m_x, m_y ∈ [−k_lim,…,+k_lim]
    for mx in range(-k_lim, k_lim + 1):
        for my in range(-k_lim, k_lim + 1):
            if mx == 0 and my == 0:
                continue # skip the q = 0 term
            kvec = np.array([mx, my]) * k0   # q_vec = (m_x, m_y)·(2π/L)
            k2 = np.dot(kvec, kvec)         # q² = |q_vec|² = q_x² + q_y²
            damp = exp(-k2 / (4.0 * alpha)) # factor e^{–q²/(4α)}
            Sq2 = structure_factor(R, kvec)
            E_k += (damp * np.abs(Sq2) ** 2) / k2 # e^{-q²/(4α)}/q² · |S(q)|²
    return (pi / area) * E_k


def self_energy(N, alpha=alpha):
    """ (equation A12 from the paper, Madelung constant) Self‐interaction correction: E_self = - ∑_i (√α / √π) · q_i²
    Here q_i are unit charges, so E_self = -N·(√α/√π).
    """
    return -sqrt(alpha / pi) * N


def coulomb_ewald_2D(R):
    """
    Full 2D Ewald Coulomb energy for N unit charges in a square PBC box:
      E_total = (e² / (4πϵ₀ ε_r)) · ( E_real + E_recip + E_self );  R : (N,2) array of positions [nm]; returns energy [meV].
    """
    N = len(R)
    E = pairwise_real_space(R) + reciprocal_space(R) + self_energy(N)
    return E * e2_4pieps0 / eps_r

def energy_moire(R):
    ext = np.sum(moire_potential(R))
    ee  = coulomb_ewald_2D(R)
    print("E_total_moire  =", energy_moire(R), "meV")
    print('where:')
    print("E_ee =", coulomb_ewald_2D(R), "meV")
    print("E_ext =", np.sum(moire_potential(R)), "meV")
    return ext + ee

# moke-test
# if __name__ == "__main__": # Only runs when executing the file directly
#     np.random.seed(0)                       # Ensures same random number
#     R = np.random.rand(6,2) * a_m           # 6 random electrons positions scaled by a_m
#     print("E_total_moire  =", energy_moire(R), "meV")
#     print('where:')
#     print("E_ee =", coulomb_ewald_2D(R), "meV")
#     print("E_ext =", np.sum(moire_potential(R)), "meV")

# we can:
# from moire_model import energy_moire
# E_loc = energy_moire(R_config)
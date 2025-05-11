# moire_model.py # latex notes in colab
# This code computes the total static energy (in meV) of a set of unit‐charges (e.g. electrons) 
# in a 2D moiré superlattice: the energy_static(R) function takes an array of particle positions 
# R (in nm) and returns the sum of the external moiré potential energy and the electron–electron 
# Coulomb repulsion (via a 2D Ewald summation under periodic boundary conditions).

from math import erfc, sqrt, pi, exp
import numpy as np

# -------- user-adjustable constants --------
a_m   = 8.5          # nm   moiré lattice constant
V0    = 15.0         # meV  potential depth
eps_r = 10.0         # dielectric constant
e2_4pieps0 = 14.399645  # meV·nm (|e|²/4πϵ0)

# --- Ewald parameters (renamed) ---
ew_alpha = 0.35      # nm⁻²  (splitting)
r_cut    = 2.5       # real-space cutoff in L units
k_cut    = 5         # k-space cutoff in 2π/L units
L        = a_m

# ---------- reciprocal lattice ----------
def reciprocal_vectors(a):
    """Generates the 3 shortest moiré reciprocal vectors G_1,2,3, 60° apart, six-fold symmetry"""
    g = 4*np.pi/(np.sqrt(3)*a) #(length |G|=4π/√3a)
    G1 = np.array([g,0.0]) # first vector along x-axis
    rot = lambda v,ang: np.array([np.cos(ang)*v[0]-np.sin(ang)*v[1], # rotate G1 by +60°, get G2
                                  np.sin(ang)*v[0]+np.cos(ang)*v[1]]) # rotate G1 by –60°, get G3
    return np.stack([G1, rot(G1, np.pi/3), rot(G1,-np.pi/3)])
G = reciprocal_vectors(a_m)

def moire_potential(r):
    """V_ext(r) = V0 * Σ cos(G·r)  (sum over G vectors)"""
    return V0 * np.sum(np.cos(r @ G.T), axis=-1)

# ---------- Ewald helpers (using ew_alpha) ----------
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

def coulomb_ewald_2D(R):
    """Full 2D Ewald Coulomb energy for N unit charges in a square PBC box:
    E_total = (e²/(4πϵ₀ ε_r))·( E_real + E_recip + E_self );  R : (N,2) array of positions [nm]; returns energy [meV]."""
    N=len(R)
    E = pairwise_real_space(R)+reciprocal_space(R)+self_energy(N)
    return E*e2_4pieps0/eps_r

# ---------- public function ----------
def energy_static(R):
    """V_ext + V_ee  (independent of Ψ)."""
    return np.sum(moire_potential(R)) + coulomb_ewald_2D(R)

# alias for backward compatibility
energy_moire = energy_static

# ---------- smoke-test ----------
if __name__ == "__main__":
    np.random.seed(0)
    R = np.random.rand(6,2)*a_m
    print("static part =", energy_static(R), "meV")
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
    return g_list  # returns [g1, g2, g3]

def moire_potential(r, a_m = a_m, V0 = V0, phi = phi):
    """ V(r) = -2*V0*sum_{j=1}^{3} cos(g_j · r + phi)where g_j are 3 reciprocal lattice vectors (from paper)."""
    G = np.array(b_vectors(a_m))  # Get the three reciprocal vectors, shape (3,2)
    phase = np.dot(r, G.T) + phi  # r @ G.T + phi
    one_electron_moire = -2 * V0 * np.sum(np.cos(phase), axis=-1)
    return np.sum(one_electron_moire)

# ---------- New Interaction part ----------

# new function to calculate Ewald‐summed electron–electron interaction energy
def calculate_H_ee(rs):
    """
    input:
    rs: positions of electrons in 2D (shape: (N, 2))
    a_m: moiré lattice constant
    n_sup: number of supercells
    ew_alpha: Ewald splitting parameter (α)
    r_cut: real‐space cutoff radius
    k_cut: reciprocal‐space cutoff (in units of 2π/a_m)
    output:
    H_ee: Ewald‐summed electron–electron interaction energy
    """
    """
    Compute the Ewald‐summed electron–electron interaction energy H_ee[R]:
    
    H_ee = 1/2 ∑_{a≠b} { ∑_L [1/|r_b–r_a–L| Erfc(|r_b–r_a–L|/(2η)) – (2π/A_uc)(2η/√π)]
                          + (2π/A_uc) ∑_{G≠0} e^{iG·(r_b–r_a)} Erfc(ηG)/G }
           + 1/2 N { ∑_{L≠0} 1/|L| Erfc(|L|/(2η)) – (2π/A_uc)(2η/√π)
                      + (2π/A_uc) ∑_{G≠0} Erfc(ηG)/G – 1/(η√π) }
    """
    # a_m = a_m
    # n_sup = n_sup
    # ew_alpha = ew_alpha
    # r_cut = r_cut
    # k_cut = k_cut
    # --- splitting parameter η from α (ew_alpha) via α = 1/(4η²) ---
    eta = 1.0 / (2.0 * sqrt(ew_alpha)) # from α = 1/(4η²) => η = 1/(2√α) and α is ew_alpha
    
    # --- unit‐cell area A_uc = |a1 × a2| ---
    a1, a2, _ = a_vectors(a_m)
    A_uc = abs(np.cross(a1, a2))
    
    # --- build real‐space translation vectors L within cutoff r_cut ---
    L_list = []
    # loop over integer multiples of primitive vectors
    for i in range(-int(r_cut), int(r_cut)+1):
        for j in range(-int(r_cut), int(r_cut)+1):
            # form the 2D translation L = i·a1 + j·a2
            L = i * a1 + j * a2
            if np.linalg.norm(L) <= r_cut * a_m: # ensure L is within cutoff
                L_list.append(L)
                
    # --- build reciprocal‐space vectors G within cutoff k_cut ---
    g1, g2, _ = b_vectors(a_m)
    G_list = []
    for i in range(-k_cut, k_cut+1):
        for j in range(-k_cut, k_cut+1):
            # form the 2D reciprocal vector G = i·g1 + j·g2
            G = i * g1 + j * g2
            if np.linalg.norm(G) <= k_cut * np.linalg.norm(g1): # ensure G is within cutoff
                # exclude G = 0 in reciprocal sums
                if not np.allclose(G, 0.0):
                    G_list.append(G)
    
    N = len(rs)
    H = 0.0
    
    # --- Part 1: 1/2 ∑_{a≠b} [ real‐space term + reciprocal‐space term ] ---
    for a in range(N): 
        for b in range(N):
            if a == b:
                continue
            dr = rs[b] - rs[a]
            
            # (1a) real‐space sum ∑_L [1/|dr–L| Erfc(|dr–L|/(2η)) – (2π/A_uc)(2η/√π)]
            sum_L = 0.0
            for L in L_list:
                rvec = dr - L
                r = np.linalg.norm(rvec)
                sum_L += (1.0/r) * erfc(r/(2.0*eta)) - (2.0*pi/A_uc)*(2.0*eta/sqrt(pi))
            
            # (1b) reciprocal‐space sum (2π/A_uc) ∑_{G≠0} e^{iG·dr} Erfc(ηG)/G
            sum_G = 0.0
            for G in G_list:
                Gnorm = np.linalg.norm(G)
                sum_G += np.exp(1j * np.dot(G, dr)) * erfc(eta * Gnorm) / Gnorm
            
            sum_G *= (2.0 * pi / A_uc)
            
            # add 1/2 [real + rec] contribution
            H += 0.5 * (sum_L + sum_G)
    
    # --- Part 2: self‐interaction term 1/2 N { … } ---
    # (2a) ∑_{L≠0} 1/|L| Erfc(|L|/(2η))
    sum_L0 = 0.0
    for L in L_list:
        if np.allclose(L, 0.0):
            continue
        Lnorm = np.linalg.norm(L)
        sum_L0 += (1.0/Lnorm) * erfc(Lnorm/(2.0*eta))
    
    # (2b) (–2π/A_uc)(2η/√π)
    const_L = - (2.0*pi / A_uc) * (2.0*eta / sqrt(pi))
    
    # (2c) (2π/A_uc) ∑_{G≠0} Erfc(ηG)/G
    sum_G0 = 0.0
    for G in G_list:
        Gnorm = np.linalg.norm(G)
        sum_G0 += erfc(eta * Gnorm) / Gnorm
    const_G = (2.0*pi / A_uc) * sum_G0
    
    # (2d) – 1/(η√π)
    const_self = - 1.0 / (eta * sqrt(pi))
    
    # assemble self‐interaction bracket and add
    self_bracket = sum_L0 + const_L + const_G + const_self
    H += 0.5 * N * self_bracket
    
    return H.real

def energy_static(R):
    """V_ext + V_ee  (independent of Ψ)."""
    return moire_potential(R) + calculate_H_ee(R)



# ---------- Old Interaction part ----------
# def pairwise_real_space(R, alpha=ew_alpha, r_lim=r_cut, L=L):
#     """ Short‐range (real‐space) Ewald sum (equation A6 from the paper):
#     E_real = ½ ∑_{i≠j} ∑_L erfc(√α·r_{ij}^L) / r_{ij}^L., 
#     where α = 1/(4η²), and r_{ij}^L = |r_i - r_j + L|"""
#     N, E = len(R), 0.0 #N:number particles
#     maxn = int(np.ceil(r_lim)) # summing over neighbor cells from -max_n to +max_n
#     for i in range(N):
#         for j in range(i+1,N): # loop over ½ ∑_{i≠j}
#             for nx in range(-maxn,maxn+1): #loop over n_x n_y
#                 for ny in range(-maxn,maxn+1):
#                     dr = R[i]-R[j]+np.array([nx,ny])*L # dr = r_i - r_j + n·L
#                     r  = np.linalg.norm(dr) # r = |dr|
#                     if r<1e-9 or r>r_lim*L: continue # Skip self‐interaction (r≈0) or beyond cutoff r_lim·L
#                     E += erfc(sqrt(alpha)*r)/r # α = 1/(4η²) we choose
#     return E

# def structure_factor(R,k):    # Σ e^{ik·r}
#     """Structure factor S(k) = Σ e^{ik·r} (sum over all particles)"""
#     phase = R @ k
#     return np.sum(np.cos(phase))+1j*np.sum(np.sin(phase))

# def reciprocal_space(R, alpha=ew_alpha, k_lim=k_cut, L=L):
#     """ Long‐range (reciprocal‐space) Ewald sum (equation A7 and A10 from the paper):
#     E_recip = (π/V) ∑_{k≠0} [ e^{-k²/(4α)} / k² ] |S(k)|²
#     with V = L² in 2D, fast convergence."""
#     area, k0 = L*L, 2*pi/L
#     E=0.0
#     # 2) Sum over discrete wavevectors q = (m_x, m_y)·(2π/L), the paper has ∑_{q≠0};
#     # here we loop m_x, m_y ∈ [−k_lim,…,+k_lim]
#     for mx in range(-k_lim,k_lim+1):
#         for my in range(-k_lim,k_lim+1):
#             if mx==0 and my==0: continue # skip the q = 0 term
#             k = np.array([mx,my])*k0 # q_vec = (m_x, m_y)·(2π/L)
#             k2 = k@k # q² = |q_vec|² = q_x² + q_y²
#             E += exp(-k2/(4*alpha))*abs(structure_factor(R,k))**2 / k2 # factor e^{–q²/(4α)} # e^{-q²/(4α)}/q² · |S(q)|²
#     return (pi/area)*E

# def self_energy(N, alpha=ew_alpha):
#     """ (equation A12 from the paper, Madelung constant) Self‐interaction correction: E_self = - ∑_i (√α / √π) · q_i²
#     Here q_i are unit charges, so E_self = -N·(√α/√π)."""
#     return -sqrt(alpha/pi)*N

# def madelung_offset(alpha=ew_alpha,                 # α = 1/(4η²)
#                     r_lim=r_cut, k_lim=k_cut, L=L):
#     """ Compute ξ_M in Eq. (A12) Returns a scalar (dimensionless).  Multiply by e²/4πϵ₀ϵ_r later."""
#     eta   = 0.5 / np.sqrt(alpha)     # because α = 1/(4η²)
#     area  = L * L
#     k0    = 2.0 * np.pi / L
#     # ---- Real-space images   Σ_{L≠0} erfc(|L|/2η)/|L|
#     rsum = 0.0
#     maxn = int(np.ceil(r_lim))
#     for nx in range(-maxn, maxn + 1):
#         for ny in range(-maxn, maxn + 1):
#             if nx == 0 and ny == 0:
#                 continue
#             Rvec = np.array([nx, ny]) * L
#             R    = np.linalg.norm(Rvec)
#             if R > r_lim * L:
#                 continue
#             rsum += erfc(R / (2.0 * eta)) / R
#     # ---- Reciprocal-space images   (2π/Area) Σ_{G≠0} e^{−η²G²}/G
#     ksum = 0.0
#     for mx in range(-k_lim, k_lim + 1):
#         for my in range(-k_lim, k_lim + 1):
#             if mx == 0 and my == 0:
#                 continue
#             Gvec = np.array([mx, my]) * k0
#             G    = np.linalg.norm(Gvec)
#             ksum += np.exp(-(eta * G) ** 2) / G
#     ksum *= 2.0 * np.pi / area
#     xi0_L = 1.0 / (eta * np.sqrt(np.pi))     # ---- ξ^L_0 term   1 / (η √π)
#     return rsum + ksum - xi0_L     # ---- ξ_M (Eq. A12)

# ξ_M = madelung_offset() * e2_4pieps0 / eps_r # compute once and store — units:   meV

# def coulomb_ewald_2D(R):
#     """
#     Full 2-D Ewald energy for the set of positions R (shape (N,2)).
#     Now includes ½ Σ_b ξ_M  so the result matches Eq. (A11) exactly.
#     E_total = (e²/(4πϵ₀ ε_r))·( E_real + E_recip + E_self )
#     """
#     N  = len(R)
#     # position-dependent part (your original implementation)
#     E_config = (pairwise_real_space(R) +
#                 reciprocal_space(R)   +
#                 self_energy(N)) * e2_4pieps0 / eps_r

#     # constant Madelung shift
#     return E_config + 0.5 * N * ξ_M

# def energy_static(R):
#     """V_ext + V_ee  (independent of Ψ)."""
#     return moire_potential(R) + coulomb_ewald_2D(R)






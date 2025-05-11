# moire_model.py  (patched)
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
    g = 4*np.pi/(np.sqrt(3)*a)
    G1 = np.array([g,0.0])
    rot = lambda v,ang: np.array([np.cos(ang)*v[0]-np.sin(ang)*v[1],
                                  np.sin(ang)*v[0]+np.cos(ang)*v[1]])
    return np.stack([G1, rot(G1, np.pi/3), rot(G1,-np.pi/3)])
G = reciprocal_vectors(a_m)

def moire_potential(r):
    return V0 * np.sum(np.cos(r @ G.T), axis=-1)

# ---------- Ewald helpers (using ew_alpha) ----------
def pairwise_real_space(R, alpha=ew_alpha, r_lim=r_cut, L=L):
    N, E = len(R), 0.0
    maxn = int(np.ceil(r_lim))
    for i in range(N):
        for j in range(i+1,N):
            for nx in range(-maxn,maxn+1):
                for ny in range(-maxn,maxn+1):
                    dr = R[i]-R[j]+np.array([nx,ny])*L
                    r  = np.linalg.norm(dr)
                    if r<1e-9 or r>r_lim*L: continue
                    E += erfc(sqrt(alpha)*r)/r
    return E

def structure_factor(R,k):    # Σ e^{ik·r}
    phase = R @ k
    return np.sum(np.cos(phase))+1j*np.sum(np.sin(phase))

def reciprocal_space(R, alpha=ew_alpha, k_lim=k_cut, L=L):
    area, k0 = L*L, 2*pi/L
    E=0.0
    for mx in range(-k_lim,k_lim+1):
        for my in range(-k_lim,k_lim+1):
            if mx==0 and my==0: continue
            k = np.array([mx,my])*k0
            k2 = k@k
            E += exp(-k2/(4*alpha))*abs(structure_factor(R,k))**2 / k2
    return (pi/area)*E

def self_energy(N, alpha=ew_alpha):
    return -sqrt(alpha/pi)*N

def coulomb_ewald_2D(R):
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
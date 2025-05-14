# plane_wave_moire.py - this is what we did in moire_band.ipynb
# This code computes the plane-wave Hamiltonian matrix H(k) for a 2D moiré superlattice.
# Diagonal elements are the kinetic energy of the plane-wave orbitals, and off-diagonal elements are the potential energy due to the moiré lattice.
import numpy as np
from moire_model import b_vectors
from k_selection import occupied_k_points

def build_G_basis(cut, a_m):
    """ Build the reciprocal lattice basis vectors G = n1b1 + n2b2 up to cutoff: |n1|,|n2| ≤ cut.
        Return (G_list, idx_map) where: 
            G_list = [(n1,n2,G_vec), ...]  for all integer pairs |n1|,|n2| ≤ cut
            idx_map[(n1,n2)] = row/col index in the Hamiltonian matrix"""

    G1, G2, _ = b_vectors(a_m)

    G_list, idx_map = [], {}
    for n1 in range(-cut, cut+1):
        for n2 in range(-cut, cut+1):
            G = n1*G1 + n2*G2
            idx = len(G_list)
            G_list.append((n1, n2, G))
            idx_map[(n1, n2)] = idx
    return G_list, idx_map #size of G_list is (2*cut+1)², and idx_map is a dict of size (2*cut+1)²


# form H matrix H_G,G'(k) = (1/2)|k+G'|^2 delta_{G,G'} + V_{G'-G} (in notes)
# The diagonal elements are the KE of the plane-wave orbitals, 
# and the off-diagonal elements are the V due to the moiré lattice.
def build_Hk(k, G_list, idx_map, a_m, V0, phi, hbar2_over_2m): 
    # phi is the phase of the potential
    """ Assemble the NG×NG plane-wave Hamiltonian H_G,G'(k).
    Only ±g_j Fourier components are non-zero (six shifts)."""
    NG = len(G_list)
    H  = np.zeros((NG, NG), dtype=complex)

    # kinetic
    for i,(_,_,G) in enumerate(G_list):
        H[i,i] = hbar2_over_2m * np.dot(k+G, k+G)

    # potential shifts (same six as previous codes)
    # V_{G'-G} = V0 * exp(iφ) for G' = G+g_j, and V_{G'-G} = V0 * exp(-iφ) for G' = G-g_j
    # where g_j are the six shifts in the moiré lattice.
    # The shifts are: ±b1, ±b2, ±(b1+b2) (in the notes)
    shifts = [
        ( 1,  0, -V0*np.exp( 1j*phi)), (-1, 0, -V0*np.exp(-1j*phi)),
        ( 0,  1, -V0*np.exp( 1j*phi)), ( 0,-1, -V0*np.exp(-1j*phi)),
        (-1, -1,-V0*np.exp( 1j*phi)),  ( 1, 1, -V0*np.exp(-1j*phi)),
    ]

    for i,(n1,n2,_) in enumerate(G_list):
        for dn1,dn2,coeff in shifts:
            j = idx_map.get((n1+dn1, n2+dn2))
            if j is not None:
                H[i,j] += coeff
    H = (H + H.conj().T) * 0.5 # make it Hermitian
    return H    # The Hamiltonian is a square matrix of size NG×NG, here NG = (2*cut+1)² = 25
# Note: The Hamiltonian is Hermitian, so we only need to fill the upper triangle.


# test the function
if __name__ == "__main__":
    cut = 2
    a_m = 8.5
    V0 = 15.0
    phi = 0.0
    hbar2_over_2m = 0.5

    G_list, idx_map = build_G_basis(cut, a_m)

    ks = list(occupied_k_points(6))
    ks[-1] *= -1        # makes the sum of k’s exactly zero # flip (0.246,0.426) → (−0.246,−0.426)
    for k in ks:           # six paper k-points
        Hk = build_Hk(k, G_list, idx_map,a_m, V0, phi, hbar2_over_2m)
        eigval = np.linalg.eigvalsh(Hk)[0]    # lowest band n=0
        # eigval = np.linalg.eigvalsh(Hk)[1]  # second band n=1
        # eigval = np.linalg.eigvalsh(Hk)[2]  # third band n=2
        print(f"k = {k};  eigenvalue ε₀(k) = {eigval:.4f} meV")

# results look correct, since:
# All five vectors have identical ∣k∣, so KE are equal, since hbar²∣k+G∣²/2m
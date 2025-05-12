# plane_wave_moire.py - this is what we did in moire_band.ipynb
# This code computes the plane-wave Hamiltonian matrix H(k) for a 2D moiré superlattice.
# Diagonal elements are the kinetic energy of the plane-wave orbitals, and off-diagonal elements are the potential energy due to the moiré lattice.
import numpy as np

def build_G_basis(cut, a_m):
    """ Build the reciprocal lattice basis vectors G = n1b1 + n2b2 up to cutoff: |n1|,|n2| ≤ cut.
        Return (G_list, idx_map) where: 
            G_list = [(n1,n2,G_vec), ...]  for all integer pairs |n1|,|n2| ≤ cut
            idx_map[(n1,n2)] = row/col index in the Hamiltonian matrix"""
    
    g_len = 4*np.pi / (np.sqrt(3)*a_m)
    b1 = g_len * np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3)])
    b2 = g_len * np.array([np.cos(4*np.pi/3), np.sin(4*np.pi/3)])

    G_list, idx_map = [], {}
    for n1 in range(-cut, cut+1):
        for n2 in range(-cut, cut+1):
            G = n1*b1 + n2*b2
            idx = len(G_list)
            G_list.append((n1, n2, G))
            idx_map[(n1, n2)] = idx
    return G_list, idx_map


# form H matrix H_G,G'(k) = (1/2)|k+G'|^2 delta_{G,G'} + V_{G'-G} (in notes)
# The diagonal elements are the KE of the plane-wave orbitals, 
# and the off-diagonal elements are the V due to the moiré lattice.
def build_Hk(k, G_list, idx_map, a_m, V0, phi=0.0, hbar2_over_2m=0.5):
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
    return H

# Note: The Hamiltonian is Hermitian, so we only need to fill the upper triangle.
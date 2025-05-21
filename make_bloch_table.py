# make_bloch_table.py - Compute and save Bloch coefficients once
# If we want spin-degeneracy or multiple bands just pick more columns of "eigvec"
import numpy as np
import pickle
from plane_wave_moire import build_Hk, build_G_basis
from k_selection import occupied_k_points
from moire_model import a_m, V0, hbar2_over_2m, phi, n_sup, N_e, cut
#run this to produce:

# kvecs = occupied_k_points(3, n_sup) #N_e = 3 electrons
# G_list, idx_map = build_G_basis(cut, a_m) # plane-wave basis (same as dispersion script)
# NG = len(G_list)    # Hamiltonian size: HG times N_e

# bloch = []   # list of dicts: {'k':k, 'coeff':c_G, 'k_plus_G':k+G}
# for k in kvecs: 
#     # for each k-point build the matrix H(k) and diagonalize it
#     # to get eignevalues E_n(k) and eigenvectors {c^(n)_G(k)}
#     H = build_Hk(k, G_list, idx_map, a_m, V0, phi, hbar2_over_2m = hbar2_over_2m)
#     eigval, eigvec = np.linalg.eigh(H)
#     coeff = eigvec[:,0]         
#     # pick the n = 0 column {c^(n)_G(k)} lowest band
#     # which defines the Bloch wavefunc ψ_k^(0)(r) = ∑_G c_G(k)*exp(i*(k+G)·r)

#     bloch.append({
#         "k"    : k, # Bloch momentum
#         "coeff": coeff, # c_G(k) for the Bloch wavefunction (recall: eigenvector of H(k)), we pick n=0 lowest band
#         "kG"   : np.array([k + G for *_ , G in G_list]), # k+G vectors for the Bloch wavefunction
#         "eps"  : eigval[0]                     # eigenvalue of the lowest band
#     })

# pickle.dump(bloch, open("bloch_coeffs_3electrons.pkl","wb"))
# print("Saved Bloch table for", len(bloch), "k-points")
# # Now we can load the Bloch table and use it in our VMC code


# or we can just use this in general directly:
def build_bloch_table(a_m, V0, phi, hbar2_over_2m, n_sup, N_e, cut):
    kvecs    = occupied_k_points(N_e, n_sup, a_m)
    G_list, idx_map = build_G_basis(cut, a_m)

    bloch = []
    for k in kvecs:
        H       = build_Hk(k, G_list, idx_map, V0, phi, hbar2_over_2m)
        evals, evecs = np.linalg.eigh(H)
        c0      = evecs[:,0]                # lowest‐band Fourier amplitudes
        bloch.append({
            "k":   k,
            "coeff": c0,
            "kG": np.array([k+G for *_,G in G_list]),
            "eps": evals[0]
        })
    return bloch
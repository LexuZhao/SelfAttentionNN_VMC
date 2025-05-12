# make_bloch_table.py - Compute and save Bloch coefficients once
# If we want spin-degeneracy or multiple bands just pick more columns of "eigvec"
import numpy as np
import pickle
from plane_wave_moire import build_Hk      # use Hk function
from plane_wave_moire import build_G_basis # use G basis
from k_selection import hexagonal_fermi_sea # use k-selection

a_m   = 8.5
V0    = 15.0
cut   = 2               # |G|≤cut·|g|
N_e   = 6               # # electrons (= # occupied orbitals)
kvecs = hexagonal_fermi_sea(N_e, a_m)
G_list, idx_map = build_G_basis(cut, a_m) # plane-wave basis (same as dispersion script)
NG = len(G_list)    # Hamiltonian size: HG times N_e

bloch = []   # list of dicts: {'k':k, 'coeff':c_G, 'k_plus_G':k+G}
for k in kvecs: 
    # for each k-point build the matrix H(k) and diagonalize it
    # to get eignevalues E_n(k) and eigenvectors {c^(n)_G(k)}
    H = build_Hk(k, G_list, idx_map, a_m, V0)
    eigval, eigvec = np.linalg.eigh(H)
    coeff = eigvec[:,0]         
    # pick the n = 0 column {c^(n)_G(k)} lowest band
    # which defines the Bloch wavefunc ψ_k^(0)(r) = ∑_G c_G(k)*exp(i*(k+G)·r)
    bloch.append(dict(k=k, coeff=coeff,
                      kG=np.array([k+G for *_,G in G_list])))

pickle.dump(bloch, open("bloch_coeffs.pkl","wb"))
print("Saved Bloch table for", len(bloch), "k-points")
# Now we can load the Bloch table and use it in our VMC code

# Block table contains, a list of Ne, each enry is a dictionary with 3 keys:
#   - k: the k-point (Bloch momentum)
#   - coeff: c_G(k) for the Bloch wavefunction (recall: eigenvector of H(k))
#   - kG: the k+G vectors for the Bloch wavefunction
# make_bloch_table.py - Compute and save Bloch coefficients once
# If you want spin-degeneracy or multiple bands just pick more columns of "eigvec"
import numpy as np
import pickle
from plane_wave_moire import build_Hk      # reuse your diagonaliser
from plane_wave_moire import build_G_basis # reuse your G basis
from k_selection import hexagonal_fermi_sea

a_m   = 8.5
V0    = 15.0
cut   = 2                # |G|≤cut·|g|
N_e   = 6                # # electrons (= # occupied orbitals)
kvecs = hexagonal_fermi_sea(N_e, a_m)

# plane-wave basis (same as dispersion script)
G_list, idx_map = build_G_basis(cut, a_m)     # you already have this
NG = len(G_list)

bloch = []   # list of dicts: {'k':k, 'coeff':c_G, 'k_plus_G':k+G}
for k in kvecs:
    H = build_Hk(k, G_list, idx_map, a_m, V0)
    eigval, eigvec = np.linalg.eigh(H)
    coeff = eigvec[:,0]         # pick the lowest band
    bloch.append(dict(k=k, coeff=coeff,
                      kG=np.array([k+G for *_,G in G_list])))

pickle.dump(bloch, open("bloch_coeffs.pkl","wb"))
print("Saved Bloch table for", len(bloch), "k-points")
# Now you can load the Bloch table and use it in your VMC code

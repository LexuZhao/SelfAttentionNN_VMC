# k_selection.py - Fill a hexagonal Fermi sea (free electrons)
import numpy as np
from moire_model import reciprocal_vectors

def hexagonal_fermi_sea(N_e, a_m):
    """ Input: N_e = number of electrons (or orbitals) to fill
    Output: k-vectors of the filled orbitals in a hexagonal Fermi sea.

    This code constructs all reciprocal-lattice momenta k = n1b1 + n2b2 up to some shell, sorts them by ∣k∣, 
    and picks the lowest- ∣k∣ of them—thereby selecting the plane-wave eigenstates that minimize the 
    non-interacting KE, ℏ^2∣k∣^2/2m, i.e. filling a perfect hexagonal Fermi sea."""
    
    # basis vectors for triangular reciprocal lattice.
    b1, b2, _ = reciprocal_vectors(a_m)

    # grow out shells until we have enough k-vectors
    shell = 0
    k_list = []
    while len(k_list) < N_e:
        # iterate over all integer pairs (n1,n2) with |n1|+|n2| ≤ shell
        for n1 in range(-shell, shell+1):
            for n2 in range(-shell, shell+1):
                if abs(n1)+abs(n2) > shell:   # stay on current shell
                    continue
                k = n1*b1 + n2*b2
                k_list.append(k)
        shell += 1

    # sort by |k| and pick the first N_e
    k_list.sort(key=lambda q: np.linalg.norm(q))
    return np.array(k_list[:N_e])         # shape (N_e,2)

# quick demo
if __name__ == "__main__":
    ks = hexagonal_fermi_sea(6, a_m=8.5)
    for v in ks: print(v)

# k_selection.py
import numpy as np
from moire_model import b_vectors

# def occupied_k_points(N_e, n_sup): # n_sup is the supercell size
#     """
#     3×3 super-cell ⇒ k = (n1/n_sup) b1 + (n2/n_sup) b2  with n1,n2 = 0,1,2.
#     Return the N_e lowest-|k| vectors (Γ + first hexagon).
#     """
#     b1, b2, _ = b_vectors(a_m)

#     full_mesh = [(n1/n_sup)*b1 + (n2/n_sup)*b2
#                  for n1 in range(n_sup) for n2 in range(n_sup)]   # 9 vectors
#     full_mesh = np.array(full_mesh)
#     # sort by |k| and pick the first N_e
#     idx = np.argsort(np.linalg.norm(full_mesh, axis=1))
#     return full_mesh[idx[:N_e]]

def occupied_k_points(N_e, n_sup, a_m):
    """k = (n1/n_sup)·g1 + (n2/n_sup)·g2, sort by |k|, return first N_e."""
    g1, g2, _ = b_vectors(a_m)
    mesh = [(n1/n_sup)*g1 + (n2/n_sup)*g2
            for n1 in range(n_sup) for n2 in range(n_sup)]
    mesh = np.array(mesh)
    idx  = np.argsort(np.linalg.norm(mesh, axis=1))
    print("occuplied k", mesh[idx[:N_e]])
    return mesh[idx[:N_e]]

# if __name__ == "__main__":
#     for k in occupied_k_points(6):
#         print(k)















# # Fill a hexagonal Fermi sea (free electrons)
# def hexagonal_fermi_sea(N_e, a_m):
#     """
#     Return N_e distinct crystal-momentum vectors that fill a hexagonal Fermi
#     sea on the triangular reciprocal lattice (spin-polarised case).

#     Each k is of the form k = n1*b1 + n2*b2 where (b1,b2) are the reciprocal
#     primitive vectors of the moiré super-lattice.  Vectors are generated in
#     concentric ‘hexagonal shells’ and sorted by |k| so the lowest-kinetic-
#     energy states are chosen first.
#     """
#     # primitive reciprocal vectors (2D)
#     b1, b2, _ = b_vectors(a_m)

#     k_list = []
#     seen    = set()          # holds tuples (n1, n2) already added
#     shell   = 0

#     while len(k_list) < N_e:
#         for n1 in range(-shell, shell + 1):
#             for n2 in range(-shell, shell + 1):
#                 if abs(n1) + abs(n2) > shell:      # stay on current shell
#                     continue
#                 if (n1, n2) in seen:               # skip duplicates (e.g. Γ)
#                     continue

#                 seen.add((n1, n2))
#                 k_vec = n1 * b1 + n2 * b2
#                 k_list.append(k_vec)

#         shell += 1

#     # sort by |k| and keep the first N_e
#     k_list.sort(key=lambda q: np.linalg.norm(q))
#     return np.array(k_list[:N_e])      # shape (N_e, 2)

# # to reproduce 3by3 moiré superlattice with 2/3 filling factor: N_e = 3*3*2/3 = 6
# # quick demo
# if __name__ == "__main__":
#     ks = hexagonal_fermi_sea(6, a_m=8.5)
#     for v in ks:
#         print(v)

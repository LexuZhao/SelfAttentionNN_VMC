# wf_bloch_slater.py ----------------------------------------------------
import numpy as np, pickle

class BlochSlaterWF:
    """
    Slater determinant made of pre-computed Bloch orbitals.
    The constructor expects the list produced by make_bloch_table.py.
    """

    def __init__(self, bloch_table, L):
        """
        bloch_table : list of dicts with keys
             'k', 'coeff', 'kG', 'eps'  (exactly what make_bloch_table stored)
        L           : box length (3 a_m) for wrapping Metropolis moves
        """
        self.orb   = bloch_table
        self.N_e   = len(bloch_table)     # number of occupied orbitals
        self.L     = L
        self.params = np.empty(0)         # no variational parameters here

    # ------------ single-orbital helpers --------------------------------
    @staticmethod
    def _exp_iqr(kG, r):
        "e^{i q·r} for a vector *array* kG (shape NG,2) and single r (2,)"
        phase = kG.dot(r)                 # shape (NG,)
        return np.exp(1j*phase)

    def _phi(self, orb, r):
        "orbital value  φ(r) = Σ_G c_G e^{i(k+G)·r}"
        return np.sum(orb['coeff'] * self._exp_iqr(orb['kG'], r))

    def _grad_phi(self, orb, r):
        "gradient  ∇φ = i Σ_G (k+G) c_G e^{i(k+G)·r}"
        eikr = self._exp_iqr(orb['kG'], r)[:,None]      # (NG,1)
        return 1j*np.sum( (orb['kG'] * orb['coeff'][:,None]) * eikr , axis=0)

    def _lap_phi(self, orb, r):
        "Laplacian  ∇²φ = -|k+G|² Σ_G c_G e^{i(k+G)·r}"
        k2 = (orb['kG']**2).sum(axis=1)                  # |k+G|²  (NG,)
        return -np.sum(k2 * orb['coeff'] * self._exp_iqr(orb['kG'], r))

    # ------------ Slater matrix & its inverse ---------------------------
    def _matrix(self, R):
        "Slater matrix M_{ij}=φ_j(r_i)  (shape N_e × N_e)"
        N = self.N_e
        M = np.empty((N, N), dtype=complex)
        for i, r in enumerate(R):
            for j, orb in enumerate(self.orb):
                M[i, j] = self._phi(orb, r)
        return M

    # ------------ interface for vmc_core --------------------------------
    def log_psi(self, R):
        sign, logdet = np.linalg.slogdet(self._matrix(R))
        return np.log(sign) + logdet                    # complex

    def grad_log_psi(self, R):
        """ gradient ∇ ln ψ = ∇ ln |ψ| + i ∇θ"""
        M    = self._matrix(R)               # (N_e, N_e) complex
        Minv = np.linalg.inv(M)           # (N_e, N_e) complex
        grad = np.zeros((self.N_e, 2), dtype=complex)   
        for i, r in enumerate(R):           # loop over electrons
            for j, orb in enumerate(self.orb): # loop over orbitals
                grad[i] += self._grad_phi(orb, r) * Minv[j, i] # (2,)
        return grad                                     # (N_e,2) complex

    def laplacian_log_psi(self, R):
        """ Laplacian ∇² ln ψ = ∇² ln |ψ| + i ∇²θ"""
        M    = self._matrix(R)
        Minv = np.linalg.inv(M)
        lap  = 0.0 + 0.0j

        for i, r in enumerate(R):
            for j, orb in enumerate(self.orb):
                lap += self._lap_phi(orb, r) * Minv[j, i]
        return lap                                      # complex scalar

    # no parameters → empty derivative matrix
    def param_derivatives(self, cfgs):
        return np.zeros((len(cfgs), 0))










# # wf_bloch_slater.py -  A Bloch-determinant wave-function compatible with VMC

# # important to note:
# # “Plain Bloch Slater” is not an exact interacting ground-state wf,
# # but it is a good approx for the ground-state of a non-interacting system.
# # It is a reference trial wf derived from a one-body ED.

# import numpy as np, pickle

# class BlochSlaterWF:
#     """
#     Slater determinant built from pre-computed Bloch orbitals.
#     Requires the pickled list produced by make_bloch_table.py.
#     """
#     def __init__(self, bloch_table, L):
#         self.orb = bloch_table          # length N_e
#         self.N_e = len(bloch_table)
#         self.L   = L
#         self.params = np.empty(0)       # no trainable params yet

#     # ---------- orbital helpers ----------------------------------------
#     def _phi(self, orb, r):
#         """ Σ c_G exp(i(K+G)·r) """
#         return np.sum(orb['coeff'] * np.exp(1j*orb['kG'].dot(r)))

#     def _grad_phi(self, orb, r):
#         """ i Σ (K+G) c_G exp(i(K+G)·r) """
#         phase = np.exp(1j*orb['kG'].dot(r))
#         return 1j*np.sum((orb['kG'].T*orb['coeff']).T * phase[:,None], axis=0)

#     def _lap_phi(self, orb, r):
#         """ -|K+G|² Σ c_G exp(i(K+G)·r) """
#         k2 = (orb['kG']**2).sum(axis=1)
#         return -np.sum(k2 * orb['coeff'] * np.exp(1j*orb['kG'].dot(r))) #check if its real or complex

#     # ---------- construct Slater matrix --------------------------------
#     def _matrix(self, R):
#         """ M_{ij} = φ_i(r_j) """
#         M = np.empty((self.N_e, self.N_e), dtype=complex) # N_e x N_e
#         for i,r in enumerate(R): # loop over electrons
#             for j,orb in enumerate(self.orb): # loop over orbitals
#                 M[i,j] = self._phi(orb,r) # φ_i(r_j)
#         return M

#     # ---------- API for vmc_core ---------------------------------------
#     def log_psi(self, R):
#         # recall: ln(z) = ln|z| + iθ
#         sign, logdet = np.linalg.slogdet(self._matrix(R))
#         return np.log(sign) + logdet  # now returns a complex log ψ


#     def grad_log_psi(self, R):
#         M    = self._matrix(R)
#         Minv = np.linalg.inv(M)
#         grad = np.zeros((self.N_e,2), dtype=complex)
#         for i,r in enumerate(R):
#             for j,orb in enumerate(self.orb):
#                 grad[i] += self._grad_phi(orb,r) * Minv[j,i]
#         return grad            # complexed

#     def laplacian_log_psi(self, R):
#         M    = self._matrix(R)
#         Minv = np.linalg.inv(M)
#         lap  = 0.0+0.0j
#         for i,r in enumerate(R):
#             for j,orb in enumerate(self.orb):
#                 lap += self._lap_phi(orb,r) * Minv[j,i]
#         return lap  # complexd

#     def param_derivatives(self, cfgs):
#         return np.zeros((len(cfgs),0))   # no params yet

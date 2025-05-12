# wf_bloch_slater.py -  A Bloch-determinant wave-function compatible with VMC
import numpy as np, pickle

class BlochSlaterWF:
    """
    Slater determinant built from pre-computed Bloch orbitals.
    Requires the pickled list produced by make_bloch_table.py.
    """
    def __init__(self, bloch_table, L):
        self.orb = bloch_table          # length N_e
        self.N_e = len(bloch_table)
        self.L   = L
        self.params = np.empty(0)       # no trainable params yet

    # ---------- orbital helpers ----------------------------------------
    def _phi(self, orb, r):
        """ Σ_G  c_G exp(i(K+G)·r) """
        return np.sum(orb['coeff'] * np.exp(1j*orb['kG'].dot(r)))

    def _grad_phi(self, orb, r):
        """ i Σ (K+G) c_G exp(i(K+G)·r) """
        phase = np.exp(1j*orb['kG'].dot(r))
        return 1j*np.sum((orb['kG'].T*orb['coeff']).T * phase[:,None], axis=0)

    def _lap_phi(self, orb, r):
        """ -|K+G|² Σ c_G exp(i(K+G)·r) """
        k2 = (orb['kG']**2).sum(axis=1)
        return -np.sum(k2 * orb['coeff'] * np.exp(1j*orb['kG'].dot(r)))

    # ---------- construct Slater matrix --------------------------------
    def _matrix(self, R):
        M = np.empty((self.N_e, self.N_e), dtype=complex)
        for i,r in enumerate(R):
            for j,orb in enumerate(self.orb):
                M[i,j] = self._phi(orb,r)
        return M

    # ---------- API for vmc_core ---------------------------------------
    def log_psi(self, R):
        sign, logdet = np.linalg.slogdet(self._matrix(R))
        return logdet.real          # sign phase dropped (OK for |Ψ|²)

    def grad_log_psi(self, R):
        M    = self._matrix(R)
        Minv = np.linalg.inv(M)
        grad = np.zeros((self.N_e,2), dtype=complex)
        for i,r in enumerate(R):
            for j,orb in enumerate(self.orb):
                grad[i] += self._grad_phi(orb,r) * Minv[j,i]
        return grad.real            # use real part for kinetic term

    def laplacian_log_psi(self, R):
        M    = self._matrix(R)
        Minv = np.linalg.inv(M)
        lap  = 0.0+0.0j
        for i,r in enumerate(R):
            for j,orb in enumerate(self.orb):
                lap += self._lap_phi(orb,r) * Minv[j,i]
        return lap.real

    def param_derivatives(self, cfgs):
        return np.zeros((len(cfgs),0))   # no params yet

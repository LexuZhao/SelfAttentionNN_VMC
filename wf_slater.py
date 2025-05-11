# wf_slater.py
import numpy as np

class SlaterWF:
    """
    Plain Slater determinant of *real* orbitals φ_k(r)=cos(k·r).

    Parameters
    ----------
    k_vectors : (N_e, 2) array
        List of wave-vectors that label the occupied orbitals.
    L : float
        Box length so electrons live in [0,L)×[0,L).
    """
    def __init__(self, k_vectors, L):
        self.kvecs = np.asarray(k_vectors)        # shape (N_e,2)
        self.N_e   = self.kvecs.shape[0]
        self.L     = L
        self.params = np.empty(0)                 # no trainable params

    # ----- single-orbital helpers ---------------------------------------
    def _phi(self, k, r):            #  φₖ(r) = cos(k · r)
        return np.cos(k.dot(r))     

    def _grad_phi(self, k, r):       # ∇ₙ φ = ∂/∂r [cos(k·r)] = -sin(k·r)·k
        return -np.sin(k.dot(r)) * k

    def _lap_phi(self, k, r):        # ∇²φ = -|k|² cos(k·r)
        return -np.dot(k,k) * np.cos(k.dot(r))

    # ----- Slater-matrix & its inverse ----------------------------------
    def _matrix(self, R): # R is an array of shape (N_e,2): positions of each electron
        M = np.empty((self.N_e, self.N_e))
        for i, r in enumerate(R):
            for j, k in enumerate(self.kvecs):
                M[i, j] = self._phi(k, r)
        return M

    # ----- API required by vmc_core.py ----------------------------------
    def log_psi(self, R): # ln|det| – we drop the overall sign; it cancels in |Ψ|²
        sign, logdet = np.linalg.slogdet(self._matrix(R))
        return logdet

    def grad_log_psi(self, R): # ∇_i lnΨ = Σ_j (∇_i φ_j) (M⁻¹)_{j,i} for each electron i
        M    = self._matrix(R)
        Minv = np.linalg.inv(M)
        grad = np.zeros((self.N_e, 2))
        for i, r in enumerate(R):
            for j, k in enumerate(self.kvecs):
                grad[i] += self._grad_phi(k, r) * Minv[j, i]
        return grad                      # shape (N_e,2)

    def laplacian_log_psi(self, R):
        # ∇²_i lnΨ  = Σ_j (∇² φ_j) (M⁻¹)_{j,i}  – no extra term needed
        # because vmc_core already adds |∇ lnΨ|² separately.
        M    = self._matrix(R)
        Minv = np.linalg.inv(M)
        lapl = 0.0
        for i, r in enumerate(R):
            for j, k in enumerate(self.kvecs):
                lapl += self._lap_phi(k, r) * Minv[j, i]
        return lapl                      # scalar

    def param_derivatives(self, cfgs): # no variational parameters → derivative = 0
        n_cfg = len(cfgs)
        return np.zeros((n_cfg, 0))

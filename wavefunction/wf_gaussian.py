# # wf_gaussian.py - minimal Gaussian wavefunction, supplies exactly the 4 methods the VMC asks for
# import numpy as np

# class GaussianWF:
#     def __init__(self, N_e, alpha0=0.05):
#         self.N_e     = N_e
#         self.params  = np.array([np.log(alpha0)])   # optimise logα

#     # ---- tiny helpers --------------------------------------------------
#     @property
#     def alpha(self):            # α = exp(logα)  >0 by construction
#         return np.exp(self.params[0])

#     def log_psi(self, R):
#         return -self.alpha * np.sum(R**2)

#     def grad_log_psi(self, R): # ∇_i lnΨ = -2α r_i
#         return -2.0 * self.alpha * R

#     def laplacian_log_psi(self, R): # ∇² lnΨ = -4α N_e
#         return -4.0 * self.alpha * self.N_e

#     # ---- derivative wrt *parameters* -----------------------------------
#     def param_derivatives(self, cfgs): # O(R) = ∂ lnΨ / ∂ logα = -α Σ r² / α = -Σ r²
#         return -np.array([cfg.sum() for cfg in (R**2 for R in cfgs)])[:,None]
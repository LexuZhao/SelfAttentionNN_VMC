# vmc_core.py - engine that knows nothing about the ansatz Ψ
# old version

import numpy as np
from moire_model import L, hbar2_over_2m # (box length) # L

# This code defines a generic VMC engine: it requires a Ψ that implements log_psi(R), grad_log_psi(R), 
# laplacian_log_psi(R) and param_derivatives(cfgs), plus a static energy function E0(R) (moire_energy we had). 
# It uses Metropolis–Hastings to sample electron config. R, in a 2D periodic box, computes the local E as the sum of KE 
# -hbar2_over_2m * (∇² lnΨ + |∇ lnΨ|²) and the moire_energy we had, and then updates the variational parameters via
# stochastic gradient descent on the estimated expectation value of the energy.


class VMC:
    """ Generic variational-Monte-Carlo engine.The wavefunction object must implement:
        • log_psi(R)           -> scalar
        • grad_log_psi(R)      -> (N,2) array  (∇_i lnΨ)
        • laplacian_log_psi(R) -> scalar       (Σ_i ∇²_i lnΨ)
    and expose a mutable .params attribute that the optimiser can update.
    """
    def __init__(self, wavefunction, energy_static, hbar2_over_2m = hbar2_over_2m, box_length= L):
        self.wf   = wavefunction
        self.E0   = energy_static   # external+Coulomb (R-only) its exactly the old "energy_moire(R)"
        self.h2m  = hbar2_over_2m
        self.L    = box_length

    # ---------- M-H sampler (doesn't know what Ψ is) ----------
    def _metro_step(self, R, delta):
        """using Metropolis–Hastings to draw random e config according to our wf's probability density,
        so that we can estimate energies and their derivatives by simple averages over those config."""
        logp0 = 2.0 * self.wf.log_psi(R).real    # ln |Ψ|²
        for i in range(len(R)):
            trial = R.copy()
            trial[i] = (R[i] + np.random.uniform(-delta, delta, 2)) % self.L
            logp1 = 2.0 * self.wf.log_psi(trial).real # real because we only need |Ψ|²
            if np.log(np.random.rand()) < (logp1 - logp0):
                R[i] = trial[i]; logp0 = logp1
        return R


    def sample(self, n_cfg=400, burn=200, delta=0.3):
        """Sample n_cfg configurations using Metropolis–Hastings."""
        R = np.random.rand(self.wf.N_e, 2) * self.L # Initialize R with N_e electrons at random positions in [0, L)×[0, L)
        cfgs=[]
        for t in range(n_cfg+burn): # Perform n_steps Metropolis–Hastings sweeps
            R = self._metro_step(R, delta)
            if t >= burn:
                cfgs.append(R.copy())
        return np.array(cfgs) # Return an array of shape (n_steps – burn, N_e, 2) returns the remaining sampled config as an array.

    # ---------- local-energy helper ----------
    def local_energy(self, R):
        """Compute the local energy at configuration R."""
        grad = self.wf.grad_log_psi(R)          # complex (N,2)
        lap  = self.wf.laplacian_log_psi(R)     # complex scalar
        # grad2 = (grad.conj()*grad).sum()        # |∇lnΨ|²  (real ≥0)
        grad2 = np.dot(grad, grad)         # no .conj()

        kinetic = -self.h2m * (lap + grad2)     # energy operator acting on the wavefunction
        return kinetic.real + self.E0(R)        # guarantee real energy
    # “H acting on the Slater determinant” is executed every time local_energy() is called
    # fix 

    # ---------- one optimisation epoch ----------
    def step(self, n_cfg=2000, lr=1e-5, clip=100.0):
        """Perform one optimisation step using stochastic gradient descent."""
        cfgs  = self.sample(n_cfg=n_cfg)
        E_loc = np.array([self.local_energy(R) for R in cfgs])
        deriv = self.wf.param_derivatives(cfgs)                     
        Obar  = deriv.mean(axis=0)                                                                    

        #   grad = 2 Re[ ⟨O* E_loc⟩  - ⟨O*⟩⟨E_loc⟩ ]
        cov   = (E_loc[:,None] * deriv.conj()).mean(axis=0) - E_loc.mean()*Obar.conj()
        grad  = 2.0 * cov.real                                      # real params
        grad  = np.clip(grad, -clip, clip)

        self.wf.params -= lr * grad
        if hasattr(self.wf, "sync_params_from_numpy"):
            self.wf.sync_params_from_numpy(self.wf.params)
        return E_loc.mean(), grad, self.wf.params.copy()

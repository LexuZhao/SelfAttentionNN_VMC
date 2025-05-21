# # wf_slaternet.py - old 

# import numpy as np, torch, torch.nn as nn
# from moire_model import reciprocal_vectors, a_m                # for G-vectors

# DTYPE = torch.double

# # ---------- helper to vectorise / devectorise torch params -------------
# def flat_params(param_iter):
#     return nn.utils.parameters_to_vector(param_iter).detach().cpu().numpy()

# def set_flat_params(vec, param_iter):
#     vec_t = torch.tensor(vec, dtype=DTYPE)
#     nn.utils.vector_to_parameters(vec_t, param_iter)

# # =======================================================================
# class SlaterNetWF:
#     """
#     Exact implementation of Sec. III A & B (“SlaterNet”) in the paper.

#     • Periodic feature f⁰(r)  =  [sin(G₁·r), cos(G₁·r), …, sin(G₃·r), cos(G₃·r)]
#     • Shared residual MLP     h^{l+1} = h^{l} + tanh(W^{l+1} h^{l} + b^{l+1})
#     • Projection heads        φ_j(r) = w_{2j}·h^L + i w_{2j+1}·h^L
#     • Slater determinant      det[φ_j(r_i)]
#     """

#     # ---------------- initialisation -----------------------------------
#     def __init__(self, N_e, n_hidden=32, n_layer=3, L=a_m, device="cpu"):
#         """
#         N_e       : # electrons  (= # orbitals)
#         n_hidden  : hidden width d_h in the paper
#         n_layer   : L residual layers
#         L         : box length (nm) for periodic wrap
#         """

#         self.N_e  = N_e
#         self.L    = L
#         self.dev  = torch.device(device)

#         # ---- periodic feature vectors  G₁,G₂,G₃  ----------------------
#         Gvec = reciprocal_vectors(L)           # (3,2) NumPy
#         self.G = torch.tensor(Gvec, dtype=DTYPE, device=self.dev)

#         # ---- feed-forward residual tower ------------------------------
#         d0 = 6                    # feature dimension (sin,cos × 3)
#         d  = n_hidden

#         layers = nn.ModuleList()
#         layers.append(nn.Linear(d0, d))
#         for _ in range(n_layer):
#             layers.append(nn.Linear(d, d))
#         self.layers = layers.to(self.dev)

#         # ---- projection vectors  (2N_e of length d) -------------------
#         self.W_proj = nn.Parameter(torch.randn(2*N_e, d, dtype=DTYPE))

#         # ---- expose parameters to NumPy world -------------------------
#         self.params = flat_params(self.parameters())

#     # ---------------- feature map f⁰(r) --------------------------------
#     def _feature(self, R):
#         """
#         R  : (N_e,2) tensor
#         return (N_e,6) tensor  [sin(G1·r),cos(G1·r), …, sin(G3·r),cos(G3·r)]
#         """
#         phase = R @ self.G.T                   # (N_e,3)
#         f = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
#         return f.reshape(-1, 6)

#     # ---------------- forward pass per electron ------------------------
#     def _hL(self, R_t):
#         """
#         R_t : (N_e,2) tensor with requires_grad
#         returns h^L  shape (N_e, d)
#         """
#         h = self.layers[0](self._feature(R_t))
#         for lin in self.layers[1:]:
#             h = h + torch.tanh(lin(h))
#         return h                               # (N_e,d)

#     # ---------------- Slater matrix Φ_{i,j} ----------------------------
#     def _slater_matrix(self, R_t):
#         hL = self._hL(R_t)                     # (N_e,d)
#         # real & imag heads
#         proj_real = torch.matmul(hL, self.W_proj[0::2 ].T)  # (N_e,N_e)
#         proj_imag = torch.matmul(hL, self.W_proj[1::2 ].T)
#         return proj_real + 1j*proj_imag

#     # ========== API required by vmc_core.py ============================
#     def log_psi(self, R):
#         R_t = torch.tensor(R, dtype=DTYPE, device=self.dev, requires_grad=False)
#         Φ   = self._slater_matrix(R_t).detach().cpu().numpy()
#         sign, logdet = np.linalg.slogdet(Φ)
#         return logdet                    # complex log(Ψ) = ln|Ψ| + iθ

#     def grad_log_psi(self, R):
#         R_t = torch.tensor(R, dtype=DTYPE, device=self.dev, requires_grad=True)
#         # logdet = torch.linalg.slogdet(self._slater_matrix(R_t))[1].real
#         logdet = torch.log(torch.linalg.det(self._slater_matrix(R_t))) # complex
#         grad,  = torch.autograd.grad(logdet, R_t, create_graph=False)
#         return grad.detach().cpu().numpy()

#     def laplacian_log_psi(self, R):
#         R_t = torch.tensor(R, dtype=DTYPE, device=self.dev, requires_grad=True)
#         # logdet = torch.linalg.slogdet(self._slater_matrix(R_t))[1].real
#         logdet = torch.log(torch.linalg.det(self._slater_matrix(R_t))) # complex
#         grad,  = torch.autograd.grad(logdet, R_t, create_graph=True)
#         lap = torch.zeros((), dtype=DTYPE, device=self.dev)
#         for i in range(self.N_e):
#             for mu in range(2):
#                 g = grad[i,mu]
#                 g2, = torch.autograd.grad(g, R_t, retain_graph=True)
#                 lap += g2[i,mu]
#         return lap.detach().cpu().item()

#     def param_derivatives(self, cfgs):
#         deriv = []
#         for R in cfgs:
#             R_t = torch.tensor(R, dtype=DTYPE, device=self.dev, requires_grad=False)
#             # logdet = torch.linalg.slogdet(self._slater_matrix(R_t))[1].real
#             logdet = torch.log(torch.linalg.det(self._slater_matrix(R_t))) # complex
#             g_par  = torch.autograd.grad(
#                          logdet, self.parameters(), retain_graph=False)
#             deriv.append(flat_params(g_par))
#         return np.stack(deriv, axis=0)

#     # ---------- keep NumPy <-> torch weights in sync -------------------
#     def sync_params_from_numpy(self, vec):
#         set_flat_params(vec, self.parameters())
#         self.params = vec                     # store pointer

#     # ---------- expose iterator for torch parameter utilities ----------
#     def parameters(self):
#         return list(self.layers.parameters()) + [self.W_proj]

# print("SlaterNetWF loaded")
import torch
import torch.nn as nn
import numpy as np

def a_vectors(a_m):
    """Generates the 3 shortest moiré reciprocal vectors G_1,2,3, 60° apart, six-fold symmetry"""
    a1 = a_m * np.array([1.0,                0.0           ])
    a2 = a_m * np.array([0.5,  np.sqrt(3.0) / 2.0          ])
    a3 = -(a1 + a2)        # optional third vector (120° w.r.t. a1)
    return [a1, a2, a3]    # same interface style as your b_vectors()


def b_vectors(a_m):
    """from paper: g_j = (4*pi / sqrt(3) / a_m) * [cos(2*pi*j/3), sin(2*pi*j/3)], for j=1,2,3"""
    g_list = []
    prefac = 4 * np.pi / (np.sqrt(3) * a_m)
    for j in range(1, 4):  # j = 1, 2, 3
        angle = 2 * np.pi * j / 3
        g = prefac * np.array([np.cos(angle), np.sin(angle)])
        g_list.append(g)
    print("g list", g_list)
    return g_list  # returns [g1, g2, g3]

def supercell_vectors(n, a_m):
    a1, a2, _ = a_vectors(a_m)
    return n * a1, n * a2


class FeedForwardLayer(nn.Module):
    """A single feed-forward layer with a tanh activation and residual skip."""
    def __init__(self, L: int) -> None:
        super().__init__()
        self.Wl_1p = nn.Linear(L, L)
        self.tanh = nn.Tanh()

    def forward(self, hl: torch.Tensor) -> torch.Tensor:
        return hl + self.tanh(self.Wl_1p(hl))


class SlaterNet(nn.Module):
    """
    Neural-network Slater-determinant ansatz:
    - embeds periodic features
    - passes through a residual MLP
    - projects to complex single-particle orbitals
    - builds N×N wavefunction matrix and takes det
    """
    def __init__(self, a: float, N: int, L: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        self.N = N
        self.L = L
        self.a = a
        self.num_layers = num_layers

        G_vectors = torch.from_numpy(np.array(b_vectors(a))).float()
        self.G1_T = G_vectors[0].unsqueeze(-1)  # (2,1)
        self.G2_T = G_vectors[1].unsqueeze(-1)  # (2,1)

        # input embedding: 4 periodic features → L
        self.W_0 = nn.Linear(4, L, bias=False)
        # residual MLP
        self.MLP_layers = nn.ModuleList([FeedForwardLayer(L) for _ in range(num_layers)])

        # complex projectors w = [w_real, w_imag], shape (L, N)
        self.complex_proj = nn.Parameter(
            torch.complex(real=torch.randn(L, N), imag=torch.randn(L, N))
        )

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        # R: (N,2) real positions
        G1_R = R @ self.G1_T  # (N,1)
        G2_R = R @ self.G2_T  # (N,1)
        features = torch.cat([
            torch.sin(G1_R), torch.sin(G2_R),
            torch.cos(G1_R), torch.cos(G2_R)
        ], dim=1)  # (N,4)

        # high-dimensional embedding
        h = self.W_0(features)  # (N,L)
        # residual MLP
        for layer in self.MLP_layers:
            h = layer(h)

        # build Slater matrix and determinant
        WF_mat = h.to(torch.complex64) @ self.complex_proj  # (N,N)
        det = torch.linalg.det(WF_mat)
        return det

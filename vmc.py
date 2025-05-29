import torch
import torch.nn as nn
import torch.optim as optim
from moire_model import energy_static

# assumes energy_static(R) returns V_ext + V_ee for R shape (N_e,2)
# assumes SlaterNet is as you defined, taking R (N_e,2) → complex Ψ

def laplacian(psi: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Compute ∇²_R ψ via two nested autograd passes.
    R requires_grad=True.
    """
    grads = torch.autograd.grad(psi, R, create_graph=True)[0]      # (N_e,2)
    lap = 0.0
    # sum over electrons i and dimensions d
    for i in range(R.shape[0]):
        for d in range(R.shape[1]):
            grad_i_d = grads[i, d]
            second = torch.autograd.grad(grad_i_d, R, retain_graph=True)[0][i, d]
            lap = lap + second
    return lap

class VMC:
    def __init__(
        self,
        slater: nn.Module,
        step_size: float = 0.1,
        device: str = "cpu",
        burn_in: int = 100,
    ):
        self.slater = slater.to(device)
        self.device = device
        self.step_size = step_size
        self.burn_in = burn_in

    def metropolis_step(self, R: torch.Tensor) -> torch.Tensor:
        """
        Propose R' = R + δ, accept with prob min(1, |Ψ(R')|^2/|Ψ(R)|^2).
        """
        with torch.no_grad():
            R_prop = R + self.step_size * torch.randn_like(R)
            psi_old = self.slater(R)
            psi_new = self.slater(R_prop)
            p = (psi_new.abs() ** 2) / (psi_old.abs() ** 2)
            if torch.rand(1, device=self.device) < p.clamp(max=1):
                return R_prop
        return R

    def sample(self, R0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Run burn-in + collect n_steps samples.
        Returns tensor of shape (n_steps, N_e, 2).
        """
        R = R0.clone().to(self.device)
        for _ in range(self.burn_in):
            R = self.metropolis_step(R)
        samples = []
        for _ in range(n_steps):
            R = self.metropolis_step(R)
            samples.append(R.clone())
        return torch.stack(samples)  # (n_steps, N_e, 2)

    def local_energy(self, R: torch.Tensor) -> torch.Tensor:
        """
        Compute E_loc = (-½ ∇² + V) Ψ / Ψ
        Returns a real scalar.
        """
        R = R.clone().detach().requires_grad_(True)
        psi = self.slater(R)
        lap = laplacian(psi, R)
        E_kin = -0.5 * lap / psi
        E_pot = energy_static(R)  # your external + e–e potential
        return (E_kin + E_pot).real

    def train(
        self,
        R0: torch.Tensor,
        mc_steps: int = 500,
        iters: int = 200,
        lr: float = 1e-3,
    ):
        """
        R0: initial configuration (N_e,2)
        """
        optimizer = optim.Adam(self.slater.parameters(), lr=lr)

        for it in range(1, iters + 1):
            # 1) sample
            samples = self.sample(R0, mc_steps)  # (mc_steps, N_e, 2)

            # 2) compute local energies
            E_loc = torch.stack([self.local_energy(R) for R in samples])  # (mc_steps,)
            E_mean = E_loc.mean()

            # 3) build REINFORCE loss:
            #    ∇θ ⟨E⟩ = E[(E_loc - ⟨E⟩) ∇θ log|Ψ|]
            log_psi = torch.stack([
                torch.log(torch.abs(self.slater(R)))
                for R in samples
            ])
            loss = torch.mean((E_loc.detach() - E_mean.detach()) * log_psi)

            # 4) gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % 10 == 0:
                print(f"Iter {it:4d}  ⟨E⟩ = {E_mean.item():.6f}")

        print("Training complete.")

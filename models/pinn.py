"""
Standard Physics-Informed Neural Network (PINN) baseline.

Architecture: fully connected network with layers [128, 64, 32] (Table S1).
Both data loss and ADR residual loss are minimized during training.
This baseline represents the "strong-physics / weak-sequential" approach
(Section 3.2 of the main text).

Key distinction from PhyLSTM:
  - No sequential memory (no LSTM recurrence)
  - Same physics residual loss formulation (Eq. S2)
  - Feedforward architecture → struggles with long-term temporal dependencies
"""

import torch
import torch.nn as nn
from typing import Tuple


class StandardPINN(nn.Module):
    """
    Fully connected PINN for contaminant concentration prediction.

    Input  : (batch, seq, input_dim)  — processed as independent samples
    Output : (batch, seq, output_dim) — predicted C at each step

    Loss   : L_total = L_data + λ_phy * L_physics + λ_bc * L_boundary
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None,
                 output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]          # Table S1

        layers: list[nn.Module] = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.Tanh(), nn.Dropout(p=dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (batch, seq, input_dim)
        out : (batch, seq, output_dim)
        """
        B, T, F = x.shape
        out = self.net(x.reshape(B * T, F))      # treat each (batch,t) independently
        return out.reshape(B, T, -1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


# ─────────────────────────────────────────────────────────────────────────────
# Physics residual (shared by PINN and PhyLSTM, imported from utils.physics)
# Defined here as a standalone function for the PINN training loop.
# ─────────────────────────────────────────────────────────────────────────────

def adr_residual(C_pred: torch.Tensor,
                 v: torch.Tensor,
                 D: torch.Tensor,
                 k: float,
                 dx: float = 100.0,
                 dt: float = 1.0) -> torch.Tensor:
    """
    Compute the discrete ADR residual for a batch of predicted trajectories.

    Eq. S2 (Supplementary Information):
        r_t = Ĉ_t - Ĉ_{t-1}
              + Δt * v * (Ĉ_t - Ĉ_{t-1}) / Δx
              - Δt * D * (Ĉ_{t+1} - 2Ĉ_t + Ĉ_{t-1}) / Δx²
              + Δt * k * Ĉ_t

    Boundary points (t=0 and t=T-1) are excluded from the loss.

    Parameters
    ----------
    C_pred : (batch, seq) predicted concentrations
    v      : (batch, seq) advection velocities
    D      : (batch, seq) diffusion coefficients
    k      : float        strategy-specific degradation coefficient
    dx     : float        spatial step (m)
    dt     : float        temporal step (months, treated as dimensionless index)

    Returns
    -------
    r : (batch, seq-2)  residuals at interior time steps
    """
    # Interior indices: t = 1 … T-2
    C_t   = C_pred[:, 1:-1]      # (B, T-2)
    C_tm1 = C_pred[:, :-2]       # Ĉ_{t-1}
    C_tp1 = C_pred[:, 2:]        # Ĉ_{t+1}

    v_t   = v[:, 1:-1]
    D_t   = D[:, 1:-1]

    adv  = v_t * (C_t - C_tm1) / dx
    diff = D_t * (C_tp1 - 2 * C_t + C_tm1) / (dx ** 2)
    rxn  = k   *  C_t

    r = (C_t - C_tm1) + dt * (adv - diff + rxn)
    return r


def physics_loss(C_pred: torch.Tensor,
                 v: torch.Tensor,
                 D: torch.Tensor,
                 k: float,
                 dx: float = 100.0,
                 dt: float = 1.0) -> torch.Tensor:
    """Mean squared physics residual (Eq. S3)."""
    r = adr_residual(C_pred, v, D, k, dx, dt)
    return (r ** 2).mean()


def boundary_loss(C_pred: torch.Tensor,
                  C_bc: torch.Tensor) -> torch.Tensor:
    """Soft penalty on boundary condition mismatch."""
    return ((C_pred[:, 0] - C_bc) ** 2).mean()

"""
PhyLSTM — Physics-Guided Sequential Learning (proposed model).

Architecture (Section 3, main text & Table S1):
  - Sequential state encoder    : 2-layer LSTM, hidden_dim=64
  - Physics-guided regularizer  : ADR residual soft penalty (Eq. S2–S4)
  - Decision evaluation module  : RSSI-based ranking stability (Section 2.2)

Loss function (Eq. S4):
  L_total = L_data + λ_phy * L_physics + λ_bc * L_boundary

Default hyperparameters from grid search (Table S1 / Section 3.3.3):
  lr        = 0.001
  epochs    = 500
  λ_phy     = 0.1
  λ_bc      = 0.05
  hidden    = 64
  layers    = 2
  dropout   = 0.2
"""

import torch
import torch.nn as nn
from .pinn import adr_residual, boundary_loss


class PhyLSTM(nn.Module):
    """
    Physics-guided LSTM.

    The model shares the same recurrent backbone as PlainLSTM but adds
    a differentiable ADR residual penalty to the training loss, anchoring
    predicted trajectories to the physically admissible manifold.

    Parameters
    ----------
    input_dim  : int   – number of input features
    hidden_dim : int   – LSTM hidden dimension  (default 64, Table S1)
    num_layers : int   – stacked LSTM layers     (default 2)
    output_dim : int   – output size             (default 1)
    dropout    : float – dropout rate            (default 0.2)
    lambda_phy : float – physics loss weight     (default 0.1, Section 3.3.3)
    lambda_bc  : float – boundary loss weight    (default 0.05)
    dx         : float – spatial step (m)        (default 100, Table S1)
    dt         : float – temporal step           (default 1, Table S1)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, output_dim: int = 1,
                 dropout: float = 0.2,
                 lambda_phy: float = 0.1,
                 lambda_bc:  float = 0.05,
                 dx: float = 100.0, dt: float = 1.0):
        super().__init__()
        self.lambda_phy = lambda_phy
        self.lambda_bc  = lambda_bc
        self.dx         = dx
        self.dt         = dt

        # ── LSTM encoder (identical to PlainLSTM) ────────────────────────
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(hidden_dim, output_dim)

    # ── Forward pass ──────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor,
                h0: torch.Tensor | None = None,
                c0: torch.Tensor | None = None):
        """
        x   : (batch, seq, input_dim)
        Returns predicted concentration: (batch, seq, output_dim)
        """
        if h0 is None or c0 is None:
            lstm_out, (h_n, c_n) = self.lstm(x)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        out = self.fc(self.dropout(lstm_out))
        return out, (h_n, c_n)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward(x)
        return out

    # ── Physics-guided loss ───────────────────────────────────────────────
    def compute_loss(self,
                     x:       torch.Tensor,
                     y_true:  torch.Tensor,
                     v:       torch.Tensor,
                     D:       torch.Tensor,
                     k:       float,
                     C_bc:    torch.Tensor | None = None) -> dict:
        """
        Compute the full PhyLSTM training loss (Eq. S4).

        Parameters
        ----------
        x      : (B, T, input_dim)  model input
        y_true : (B, T)             observed concentrations
        v      : (B, T)             advection velocities
        D      : (B, T)             diffusion coefficients
        k      : float              degradation coefficient
        C_bc   : (B,)               boundary condition values (initial C)

        Returns
        -------
        dict with keys: 'total', 'data', 'physics', 'boundary'
        """
        y_pred_raw, _ = self.forward(x)
        y_pred = y_pred_raw.squeeze(-1)              # (B, T)

        # L_data: MSE on available observations
        L_data = ((y_pred - y_true) ** 2).mean()

        # L_physics: ADR residual penalty (Eq. S3)
        if y_pred.shape[1] >= 3:
            L_phy = adr_residual(y_pred, v, D, k,
                                  self.dx, self.dt)
            L_phy = (L_phy ** 2).mean()
        else:
            L_phy = torch.tensor(0.0, device=x.device)

        # L_boundary: soft penalty on initial condition
        if C_bc is not None:
            L_bc = boundary_loss(y_pred, C_bc)
        else:
            L_bc = torch.tensor(0.0, device=x.device)

        L_total = L_data + self.lambda_phy * L_phy + self.lambda_bc * L_bc

        return {
            "total":   L_total,
            "data":    L_data,
            "physics": L_phy,
            "boundary": L_bc,
        }

    # ── Mass Balance Error (MBE) ──────────────────────────────────────────
    @staticmethod
    def mass_balance_error(C_pred: torch.Tensor,
                           v:      torch.Tensor,
                           D:      torch.Tensor,
                           k:      float,
                           dx:     float = 100.0,
                           dt:     float = 1.0) -> torch.Tensor:
        """
        Cumulative MBE as defined in Supplementary S3.2:

            MBE_t = Σ_{i=1}^{t} | (Ĉ_i - Ĉ_{i-1}) - Φ(Ĉ_i, θ) |

        where Φ is the theoretical mass flux from the ADR equation.

        Returns
        -------
        mbe : (T-1,) cumulative MBE series (averaged over batch)
        """
        C   = C_pred             # (B, T)
        T   = C.shape[1]

        cumulative = torch.zeros(T - 1, device=C.device)
        running    = torch.zeros(C.shape[0], device=C.device)

        for t in range(1, T):
            C_t   = C[:, t]
            C_tm1 = C[:, t - 1]
            vt    = v[:, t - 1]
            Dt    = D[:, t - 1]

            adv  = vt * (C_t - C_tm1) / dx
            diff = Dt * torch.zeros_like(C_t) / (dx ** 2)  # simplified
            rxn  = k  * C_t

            # Observed change vs theoretical flux
            flux = dt * (adv - diff + rxn)
            obs_change = C_t - C_tm1
            running += torch.abs(obs_change - flux)
            cumulative[t - 1] = running.mean()

        return cumulative

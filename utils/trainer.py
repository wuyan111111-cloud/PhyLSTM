"""
Unified training loop for all four model variants.

Supports early stopping (patience from Table S1), Adam optimizer,
and the composite validation metric used for hyperparameter search:
    val_metric = 0.3 * RMSE + 0.7 * (1 - Kendall's τ)   (Section S2.2)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Any

from models import PlainLSTM, FeatureAugLSTM, StandardPINN, PhyLSTM
from utils.metrics import rmse, compute_rssi, generate_perturbations, aggregate_rssi
from utils.strategies import BASE_STRATEGIES
from utils.metrics import compute_decision_scores


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype).to(DEVICE)


def make_dataset(X: np.ndarray, y: np.ndarray,
                 v: np.ndarray, D: np.ndarray) -> TensorDataset:
    """Pack arrays into a TensorDataset."""
    return TensorDataset(
        to_tensor(X),
        to_tensor(y),
        to_tensor(v),
        to_tensor(D),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation metric
# ─────────────────────────────────────────────────────────────────────────────

def validation_metric(y_true: np.ndarray, y_pred: np.ndarray,
                      strategy_C_true: np.ndarray) -> float:
    """
    val_metric = 0.3 * RMSE + 0.7 * (1 - mean_RSSI)   (Section S2.2)

    Lower is better.
    """
    r = rmse(y_true, y_pred)

    costs = [s.cost_dict() for s in BASE_STRATEGIES]
    D_nom = compute_decision_scores(strategy_C_true, costs)
    perturbed = generate_perturbations(strategy_C_true, n_pert=20,
                                       noise_pct=0.05)
    D_pert = np.stack([
        compute_decision_scores(perturbed[k], costs)
        for k in range(perturbed.shape[0])
    ])
    rssi_arr = compute_rssi(D_nom, D_pert)
    tau_mean = aggregate_rssi(rssi_arr)

    return 0.3 * r + 0.7 * (1.0 - tau_mean)


# ─────────────────────────────────────────────────────────────────────────────
# Core trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Unified trainer for PlainLSTM, FeatureAugLSTM, StandardPINN, PhyLSTM.

    Parameters
    ----------
    model      : nn.Module
    lr         : float   learning rate (default 0.001, Table S1)
    max_epochs : int     maximum training epochs (default 500)
    patience   : int     early stopping patience (default 50)
    batch_size : int     mini-batch size (default 32)
    k_decay    : float   strategy-specific degradation coefficient
                         (used only for PhyLSTM / PINN physics loss)
    """

    def __init__(self, model: nn.Module,
                 lr:         float = 0.001,
                 max_epochs: int   = 500,
                 patience:   int   = 50,
                 batch_size: int   = 32,
                 k_decay:    float = 0.10,
                 l2_coef:    float = 1e-4):
        self.model      = model.to(DEVICE)
        self.lr         = lr
        self.max_epochs = max_epochs
        self.patience   = patience
        self.batch_size = batch_size
        self.k_decay    = k_decay

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=l2_coef)

    # ── Generic MSE loss for non-PhyLSTM models ──────────────────────────
    @staticmethod
    def _mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return ((y_pred.squeeze(-1) - y_true) ** 2).mean()

    # ── PhyLSTM loss ─────────────────────────────────────────────────────
    def _phylstm_loss(self, batch) -> torch.Tensor:
        X, y, v, D = batch
        loss_dict = self.model.compute_loss(
            X, y, v, D, self.k_decay, C_bc=y[:, 0])
        return loss_dict["total"]

    # ── PINN loss ─────────────────────────────────────────────────────────
    def _pinn_loss(self, batch) -> torch.Tensor:
        from models.pinn import physics_loss, boundary_loss
        X, y, v, D = batch
        y_pred = self.model(X).squeeze(-1)
        L_data = self._mse(y_pred.unsqueeze(-1), y)
        L_phy  = physics_loss(y_pred, v, D, self.k_decay)
        L_bc   = boundary_loss(y_pred, y[:, 0])
        return L_data + 1.0 * L_phy + 0.5 * L_bc

    # ── Main training loop ────────────────────────────────────────────────
    def fit(self, train_ds: TensorDataset,
            val_ds:   TensorDataset) -> dict:
        """
        Train the model with early stopping.

        Returns history dict with keys 'train_loss' and 'val_loss'.
        """
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=len(val_ds))

        best_val    = float("inf")
        patience_ct = 0
        history     = {"train_loss": [], "val_loss": []}

        is_phylstm  = isinstance(self.model, PhyLSTM)
        is_pinn     = isinstance(self.model, StandardPINN)

        for epoch in range(self.max_epochs):
            # ── Training ─────────────────────────────────────────────────
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                self.optimizer.zero_grad()
                if is_phylstm:
                    loss = self._phylstm_loss(batch)
                elif is_pinn:
                    loss = self._pinn_loss(batch)
                else:
                    X, y, _, _ = batch
                    pred = self.model.predict(X)
                    loss = self._mse(pred, y)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            history["train_loss"].append(epoch_loss)

            # ── Validation ────────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_batch  = next(iter(val_loader))
                X_v, y_v, v_v, D_v = val_batch

                if is_phylstm:
                    vl = self._phylstm_loss(val_batch).item()
                elif is_pinn:
                    vl = self._pinn_loss(val_batch).item()
                else:
                    vl = self._mse(self.model.predict(X_v), y_v).item()

            history["val_loss"].append(vl)

            # ── Early stopping ─────────────────────────────────────────────
            if vl < best_val - 1e-6:
                best_val    = vl
                patience_ct = 0
                best_weights = {k: v.clone()
                                for k, v in self.model.state_dict().items()}
            else:
                patience_ct += 1

            if patience_ct >= self.patience:
                self.model.load_state_dict(best_weights)
                break

        return history

    # ── Inference ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions as NumPy array, shape (n_sections, T)."""
        self.model.eval()
        Xt  = to_tensor(X)
        out = self.model.predict(Xt)
        return out.squeeze(-1).cpu().numpy()

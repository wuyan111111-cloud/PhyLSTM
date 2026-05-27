"""
Feature-Augmented LSTM.

Same LSTM backbone as PlainLSTM but the input feature vector is enriched
with physics-derived quantities (velocity v, diffusion coefficient D).
This baseline tests whether adding physical features as inputs (rather than
as structural constraints) can replicate the decision-stability gains of
PhyLSTM (Section 3.2, main text).
"""

import torch
import torch.nn as nn
from .lstm import PlainLSTM


class FeatureAugLSTM(nn.Module):
    """
    LSTM that receives the standard input features PLUS precomputed
    physics-derived quantities as additional channels.

    The augmentation concatenates [v, D] (or any subset) to x before
    feeding into the LSTM.  In the paper's experimental setup the full
    feature vector already includes v and D, so this model is identical
    to PlainLSTM in terms of architecture — the distinction is conceptual
    (the paper explicitly discusses that feature engineering ≠ structural
    regularization, Section 3.2).

    We therefore expose a `extra_dim` parameter so callers can inject
    additional derived features (e.g., Reynolds number, Péclet number).

    Parameters
    ----------
    input_dim  : int  – number of raw input features
    extra_dim  : int  – number of additional physics-derived features
    hidden_dim : int  – LSTM hidden dimension
    num_layers : int  – stacked LSTM layers
    output_dim : int  – output size
    dropout    : float
    """

    def __init__(self, input_dim: int, extra_dim: int = 0,
                 hidden_dim: int = 64, num_layers: int = 2,
                 output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.extra_dim = extra_dim
        total_dim = input_dim + extra_dim
        self.core = PlainLSTM(
            input_dim  = total_dim,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            output_dim = output_dim,
            dropout    = dropout,
        )

    def forward(self, x: torch.Tensor,
                extra: torch.Tensor | None = None):
        """
        Parameters
        ----------
        x     : (batch, seq, input_dim)
        extra : (batch, seq, extra_dim)  optional extra physics features

        Returns
        -------
        out   : (batch, seq, output_dim)
        states: (h_n, c_n)
        """
        if extra is not None and self.extra_dim > 0:
            x_aug = torch.cat([x, extra], dim=-1)
        else:
            x_aug = x
        return self.core(x_aug)

    def predict(self, x: torch.Tensor,
                extra: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.forward(x, extra)
        return out

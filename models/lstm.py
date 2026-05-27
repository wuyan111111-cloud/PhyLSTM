"""
Plain LSTM baseline model (no physics constraints).

Architecture from Table S1:
  - Hidden dim: 64
  - LSTM layers: 2
  - Dropout: 0.2
  - FC output layer
"""

import torch
import torch.nn as nn


class PlainLSTM(nn.Module):
    """
    Standard 2-layer LSTM with a fully connected output head.

    Parameters
    ----------
    input_dim  : int  – number of input features
    hidden_dim : int  – LSTM hidden dimension (default 64, Table S1)
    num_layers : int  – stacked LSTM layers (default 2, Table S1)
    output_dim : int  – output size (default 1, predicting C)
    dropout    : float – dropout probability (default 0.2, Table S1)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, output_dim: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size   = input_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(hidden_dim, output_dim)

    # ── Forward ────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor,
                h0: torch.Tensor | None = None,
                c0: torch.Tensor | None = None):
        """
        Parameters
        ----------
        x  : (batch, seq_len, input_dim)
        h0 : (num_layers, batch, hidden_dim)  optional initial hidden state
        c0 : (num_layers, batch, hidden_dim)  optional initial cell  state

        Returns
        -------
        out   : (batch, seq_len, output_dim)
        (h_n, c_n) : final hidden / cell states
        """
        if h0 is None or c0 is None:
            lstm_out, (h_n, c_n) = self.lstm(x)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        lstm_out = self.dropout(lstm_out)
        out      = self.fc(lstm_out)          # (batch, seq, output_dim)
        return out, (h_n, c_n)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions only (no state), shape (batch, seq, 1)."""
        out, _ = self.forward(x)
        return out

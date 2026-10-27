"""
Long Short-Term Memory (LSTM) Recurrent Neural Network for Ocean Wave Forecasting.

Architecture Concept:
- Explicitly models sequential dependencies over the historical input sequence.
- Maintains recurrent cell memory states (c_t) and hidden states (h_t) across time steps t = 1 ... L.
- Internal forget, input, and output gates regulate information retention and mitigate vanishing gradients.
- Uses the final hidden state h_L as a dynamic temporal summary vector for predicting the future state.
"""

from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from .base import BaseWaveModel


class WaveLSTM(BaseWaveModel):
    """
    LSTM Recurrent Neural Network for sequential wave height forecasting.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    seq_len : int
        Number of historical time steps in input window (L).
    hidden_dim : int
        Number of hidden units in LSTM memory cells (default: 64).
    num_layers : int
        Number of stacked LSTM layers (default: 2).
    output_dim : int
        Forecast target dimension (default: 1 for single-horizon swh).
    dropout : float
        Dropout probability applied between stacked LSTM layers and dense head (default: 0.2).
    bidirectional : bool
        Whether to use a bidirectional LSTM (default: False for causal forecasting).
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 24,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        lstm_feature_dim = hidden_dim * direction_factor

        self.fc = nn.Sequential(
            nn.Linear(lstm_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: Tensor of shape (batch_size, seq_len, input_dim).
        Returns: Tensor of shape (batch_size, output_dim).
        """
        lstm_out, _ = self.lstm(x)
        # Extract representation at the last historical time step (t = L)
        last_step = lstm_out[:, -1, :]
        return self.fc(last_step)

    def get_config(self) -> Dict[str, Any]:
        """Return model hyperparameter configuration."""
        return {
            "model_type": "LSTM",
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
        }

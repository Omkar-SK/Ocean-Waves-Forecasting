"""
Artificial Neural Network (Multilayer Perceptron) for Ocean Wave Forecasting.

Architecture Concept:
- Receives the historical window of shape (batch_size, seq_len, num_features).
- Flattens the temporal dimension into a static 1D feature vector of length (seq_len * num_features).
- Passes the flattened representation through dense feed-forward layers with non-linear activations.
- Note: The ANN treats each time step as an independent static dimension and does not possess
  explicit temporal memory or temporal translation equivariance.
"""

from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from .base import BaseWaveModel


class WaveANN(BaseWaveModel):
    """
    Multilayer Perceptron (ANN) baseline for time-series wave forecasting.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step (or total flattened input dimension if seq_len is None).
    seq_len : Optional[int]
        Number of historical time steps in input window (L). If provided, flattened_dim = seq_len * input_dim.
    hidden_dims : Optional[List[int]]
        Hidden layer sizes (default: [128, 64, 32]).
    output_dim : int
        Forecast target dimension (default: 1 for single-horizon swh).
    dropout : float
        Dropout probability for regularization (default: 0.2).
    use_batch_norm : bool
        Whether to apply Batch Normalization after linear transformations (default: True).
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64, 32]
        self.output_dim = output_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Compute flattened dimension
        if seq_len is not None:
            self.flattened_dim = seq_len * input_dim
        else:
            self.flattened_dim = input_dim

        layers = []
        prev_dim = self.flattened_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final projection layer to prediction target
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: Tensor of shape (batch_size, seq_len, num_features) or (batch_size, flattened_dim).
        Returns: Tensor of shape (batch_size, output_dim).
        """
        if x.dim() == 3:
            # Flatten historical window along time axis into 1D feature vector
            x = x.reshape(x.size(0), -1)
        return self.network(x)

    def get_config(self) -> Dict[str, Any]:
        """Return model hyperparameter configuration."""
        return {
            "model_type": "ANN",
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "flattened_dim": self.flattened_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
        }

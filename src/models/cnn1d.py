"""
1D Temporal Convolutional Neural Network (1D-CNN) for Ocean Wave Forecasting.

Architecture Concept:
- Operates strictly along the 1D historical time axis (t - L + 1, ..., t).
- Channels correspond to the physical wave and atmospheric features at each time step.
- Slides 1D temporal kernels over past hours to extract localized temporal dynamics, such as
  rapid wind speed acceleration, wave growth trends, or swell periodicity.
- CRITICAL ARCHITECTURAL DISTINCTION: This model is a 1D TEMPORAL CNN applied to historical
  sequences at a single offshore point; it does NOT perform 2D spatial convolutions across geographic maps.
"""

from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from .base import BaseWaveModel


class WaveCNN1D(BaseWaveModel):
    """
    1D Temporal CNN for feature extraction along the time axis.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step (acts as input channels for Conv1D).
    seq_len : int
        Number of historical time steps in input window (acts as length of temporal sequence).
    num_filters : Optional[List[int]]
        Filter channels in stacked temporal convolution layers (default: [64, 128]).
    kernel_size : int
        Temporal receptive field kernel size in hours (default: 3).
    output_dim : int
        Forecast target dimension (default: 1 for single-horizon swh).
    dropout : float
        Dropout probability (default: 0.2).
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 24,
        num_filters: Optional[List[int]] = None,
        kernel_size: int = 3,
        output_dim: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_filters = num_filters if num_filters is not None else [64, 128]
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.dropout = dropout

        layers = []
        in_channels = input_dim

        for out_channels in self.num_filters:
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.temporal_conv = nn.Sequential(*layers)

        # Compute flattened dimension dynamically using a dummy temporal tensor
        dummy_input = torch.zeros(1, input_dim, seq_len)
        with torch.no_grad():
            dummy_conv_out = self.temporal_conv(dummy_input)
            self.flattened_dim = dummy_conv_out.reshape(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: Tensor of shape (batch_size, seq_len, num_features).
        Transposes to (batch_size, num_features, seq_len) for 1D temporal convolution.
        Returns: Tensor of shape (batch_size, output_dim).
        """
        # Permute from (batch, time, features) to (batch, features, time)
        x_permuted = x.permute(0, 2, 1)
        conv_feats = self.temporal_conv(x_permuted)
        flattened = conv_feats.reshape(conv_feats.size(0), -1)
        return self.fc(flattened)

    def get_config(self) -> Dict[str, Any]:
        """Return model hyperparameter configuration."""
        return {
            "model_type": "1D-CNN",
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "flattened_dim": self.flattened_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
        }

"""
Base Model Class and Architectural Utilities for Ocean Wave Forecasting.
Provides common interfaces for parameter counting, configuration export, and model summaries.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class BaseWaveModel(nn.Module):
    """
    Abstract base class for all wave forecasting deep learning architectures (ANN, LSTM, 1D-CNN).
    Enforces consistent input/output interfaces and metadata logging.
    """

    def __init__(self):
        super().__init__()
        self.model_name = self.__class__.__name__

    def count_parameters(self) -> Dict[str, int]:
        """Compute trainable, non-trainable, and total parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + non_trainable
        return {
            "trainable_params": trainable,
            "non_trainable_params": non_trainable,
            "total_params": total,
        }

    def get_model_summary(self, sample_input_shape: Tuple[int, ...] = (1, 24, 13)) -> str:
        """
        Generate a human-readable architecture summary with parameter counts and forward verification.
        """
        counts = self.count_parameters()
        lines = [
            "=" * 70,
            f"MODEL ARCHITECTURE: {self.model_name}",
            "=" * 70,
        ]

        # Check forward pass
        dummy_x = torch.zeros(*sample_input_shape)
        self.eval()
        with torch.no_grad():
            try:
                out = self.forward(dummy_x)
                out_shape = list(out.shape)
            except Exception as e:
                out_shape = f"Forward error: {e}"

        lines.extend([
            f"Input Tensor Shape  : {list(sample_input_shape)} (batch_size, seq_len, num_features)",
            f"Output Tensor Shape : {out_shape} (batch_size, forecast_horizon)",
            "-" * 70,
            f"Trainable Parameters    : {counts['trainable_params']:,}",
            f"Non-Trainable Parameters: {counts['non_trainable_params']:,}",
            f"Total Parameters        : {counts['total_params']:,}",
            "-" * 70,
            "Layer Hierarchy:",
        ])
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters())
            lines.append(f"  • {name} ({module.__class__.__name__}): {n_params:,} parameters")

        lines.append("=" * 70)
        return "\n".join(lines)

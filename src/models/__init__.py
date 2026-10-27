"""
Deep Learning Model Architectures for Wave Forecasting:
- BaseWaveModel: Common interface, parameter counting, and summaries
- WaveANN: Artificial Neural Network (Multilayer Perceptron baseline)
- WaveLSTM: Long Short-Term Memory Network (Recurrent sequential model)
- WaveCNN1D: 1D Temporal Convolutional Neural Network (Time-axis feature extractor)
"""

from typing import Union
from .base import BaseWaveModel
from .ann import WaveANN
from .lstm import WaveLSTM
from .cnn1d import WaveCNN1D


def build_wave_model(
    model_type: str,
    input_dim: int,
    seq_len: int = 24,
    output_dim: int = 1,
    **kwargs,
) -> BaseWaveModel:
    """
    Factory function to instantiate wave forecasting architectures.

    Parameters
    ----------
    model_type : str
        One of 'ann', 'lstm', 'cnn1d' (case-insensitive).
    input_dim : int
        Number of input features per time step.
    seq_len : int
        Historical window length in hours (default 24).
    output_dim : int
        Forecast output dimension (default 1).
    **kwargs : dict
        Architecture-specific hyperparameters (e.g. hidden_dim, num_layers, num_filters, dropout).

    Returns
    -------
    BaseWaveModel
        Configured PyTorch model instance.
    """
    m_type = model_type.strip().lower()
    if m_type in ["ann", "mlp"]:
        return WaveANN(input_dim=input_dim, seq_len=seq_len, output_dim=output_dim, **kwargs)
    elif m_type in ["lstm", "rnn"]:
        return WaveLSTM(input_dim=input_dim, seq_len=seq_len, output_dim=output_dim, **kwargs)
    elif m_type in ["cnn1d", "1d-cnn", "cnn"]:
        return WaveCNN1D(input_dim=input_dim, seq_len=seq_len, output_dim=output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: 'ann', 'lstm', 'cnn1d'.")


__all__ = ["BaseWaveModel", "WaveANN", "WaveLSTM", "WaveCNN1D", "build_wave_model"]

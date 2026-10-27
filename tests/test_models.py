"""
Unit tests for PyTorch models: ANN, LSTM, and 1D-CNN.
"""

import pytest
import torch
from src.models.ann import WaveANN
from src.models.lstm import WaveLSTM
from src.models.cnn1d import WaveCNN1D


def test_ann_forward_pass():
    batch_size = 16
    seq_len = 24
    num_features = 6
    x = torch.randn(batch_size, seq_len, num_features)

    # Flattened input dimension: seq_len * num_features
    model = WaveANN(input_dim=seq_len * num_features, output_dim=1)
    out = model(x)
    assert out.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {out.shape}"


def test_lstm_forward_pass():
    batch_size = 16
    seq_len = 24
    num_features = 6
    x = torch.randn(batch_size, seq_len, num_features)

    model = WaveLSTM(input_dim=num_features, hidden_dim=32, num_layers=2, output_dim=1)
    out = model(x)
    assert out.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {out.shape}"


def test_cnn1d_forward_pass():
    batch_size = 16
    seq_len = 24
    num_features = 6
    x = torch.randn(batch_size, seq_len, num_features)

    model = WaveCNN1D(input_dim=num_features, seq_len=seq_len, num_filters=[32, 64], output_dim=1)
    out = model(x)
    assert out.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {out.shape}"

"""
Unit tests for data validation and evaluation metrics.
"""

import numpy as np
import pytest
from src.evaluation.metrics import (
    calculate_metrics,
    root_mean_squared_error,
    mean_absolute_error,
    scatter_index,
)


def test_metrics_calculation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    metrics = calculate_metrics(y_true, y_pred)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "si" in metrics
    assert "bias" in metrics
    assert metrics["rmse"] > 0
    assert metrics["r2"] > 0.95

"""
Evaluation metrics and model benchmarking suite for ocean wave forecasting.
"""

from .metrics import (
    calculate_comprehensive_metrics,
    calculate_metrics,
    mean_absolute_percentage_error_safe,
    measure_inference_latency,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score_metric,
    scatter_index,
    mean_bias_error,
)
from .evaluator import WaveModelEvaluator

__all__ = [
    "calculate_comprehensive_metrics",
    "mean_absolute_percentage_error_safe",
    "measure_inference_latency",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score_metric",
    "scatter_index",
    "mean_bias_error",
    "WaveModelEvaluator",
]

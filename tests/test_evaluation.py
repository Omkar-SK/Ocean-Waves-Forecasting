"""
Unit Tests for Model Evaluation, Metric Computations, and Data Alignment.
"""

import numpy as np
import pytest
import torch

from src.evaluation.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score_metric,
    scatter_index,
    mean_bias_error,
    mean_absolute_percentage_error_safe,
    calculate_comprehensive_metrics,
    measure_inference_latency,
)
from src.evaluation.evaluator import WaveModelEvaluator
from src.models.ann import WaveANN


def test_standard_regression_metrics():
    """Verify RMSE, MAE, R², Scatter Index, and Bias on known analytical vectors."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.1, 4.2, 4.8])

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score_metric(y_true, y_pred)
    si = scatter_index(y_true, y_pred)
    bias = mean_bias_error(y_true, y_pred)

    assert np.isclose(mae, 0.14, atol=1e-3)
    assert rmse > 0.0
    assert r2 > 0.98
    # Scatter index = RMSE / mean(y_true) = RMSE / 3.0
    assert np.isclose(si, rmse / 3.0, atol=1e-5)
    assert np.isclose(bias, 0.02, atol=1e-3)


def test_mape_near_zero_safety_and_limitations():
    """Verify that MAPE handles near-zero wave heights without raising ZeroDivisionError."""
    # Near-zero values simulating calm sea states
    y_true = np.array([0.0, 0.05, 0.1, 1.5, 2.0])
    y_pred = np.array([0.02, 0.08, 0.12, 1.45, 2.05])

    res = mean_absolute_percentage_error_safe(y_true, y_pred, min_threshold=0.2)

    # Must return structured dictionary with filtered MAPE and explanatory notes
    assert "mape_filtered_pct" in res
    assert "wape_pct" in res
    assert "mape_limitation_note" in res
    assert res["wape_pct"] > 0
    # Filtered samples should only evaluate >= 0.2m (2 samples: 1.5 and 2.0)
    assert res["mape_filtered_samples_evaluated"] == 2
    assert "calm sea states" in res["mape_limitation_note"].lower()


def test_evaluation_alignment_and_evaluator_ranking():
    """Verify that WaveModelEvaluator correctly aligns predictions and ranks models by RMSE."""
    n_samples = 50
    np.random.seed(42)
    y_true = np.random.uniform(0.5, 3.5, size=n_samples)

    # Model A: smaller noise (RMSE ~ 0.05)
    pred_a = y_true + np.random.normal(0, 0.05, size=n_samples)
    # Model B: larger noise (RMSE ~ 0.25)
    pred_b = y_true + np.random.normal(0, 0.25, size=n_samples)

    evaluator = WaveModelEvaluator(y_true=y_true, target_units="meters")
    evaluator.add_model_result("ModelA", pred_a)
    evaluator.add_model_result("ModelB", pred_b)

    best_name, best_metrics, df_ranks = evaluator.rank_models(criterion="rmse", ascending=True)

    # Model A should rank #1
    assert best_name == "ModelA"
    assert df_ranks.iloc[0]["rmse"] < df_ranks.iloc[1]["rmse"]
    assert len(df_ranks) == 2


def test_evaluator_length_mismatch_detection():
    """Verify that evaluator raises an error if prediction length does not match ground truth."""
    y_true = np.ones(20)
    y_pred_mismatch = np.ones(19)

    evaluator = WaveModelEvaluator(y_true=y_true)
    with pytest.raises(ValueError, match="does not match ground truth length"):
        evaluator.add_model_result("BadModel", y_pred_mismatch)


def test_inference_latency_measurement():
    """Verify that inference latency benchmarking executes properly and returns positive statistics."""
    model = WaveANN(input_dim=13, seq_len=24, output_dim=1)
    dummy_input = torch.randn(16, 24, 13)

    lat = measure_inference_latency(model, dummy_input, num_iterations=10, warmup=2, device="cpu")

    assert "latency_sample_ms" in lat
    assert "latency_batch_ms" in lat
    assert "throughput_samples_per_sec" in lat
    assert lat["latency_sample_ms"] > 0.0
    assert lat["throughput_samples_per_sec"] > 0.0
    assert "measurement_protocol" in lat

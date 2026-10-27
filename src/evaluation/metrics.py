"""
Forecasting evaluation metrics and inference latency benchmarks for wave height prediction.
Includes:
- RMSE (Root Mean Squared Error, meters)
- MAE (Mean Absolute Error, meters)
- R² (Coefficient of Determination)
- Scatter Index (SI, dimensionless oceanographic metric)
- Mean Bias Error (MBE, meters)
- MAPE (with numerical safety and documentation of near-zero wave height limitations)
- Inference latency & throughput benchmarking
"""

import time
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error as mae, r2_score
import torch


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE) in meters."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE) in meters."""
    return float(mae(y_true, y_pred))


def r2_score_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Coefficient of Determination (R² score)."""
    return float(r2_score(y_true, y_pred))


def scatter_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Scatter Index (SI) = RMSE / mean(y_true).
    Standard dimensionless oceanographic metric for wave model performance.
    """
    mean_val = float(np.mean(y_true))
    if mean_val == 0.0:
        return float("nan")
    return float(root_mean_squared_error(y_true, y_pred) / mean_val)


def mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Bias Error (MBE) = mean(y_pred - y_true) in meters."""
    return float(np.mean(y_pred - y_true))


def mean_absolute_percentage_error_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Compute MAPE safely, documenting limitations for near-zero physical values.

    Mathematical Limitation:
    Significant wave height can approach near-zero values during calm seas (e.g. 0.2m - 0.4m).
    Dividing by near-zero denominators causes MAPE to explode exponentially, even when
    the absolute physical error is small (e.g., error of 0.2m on 0.2m wave yields 100% error).
    Therefore, we report:
    1. 'mape_filtered': MAPE calculated only where y_true >= min_threshold (default 0.2m).
    2. 'wape': Weighted Absolute Percentage Error = sum(|y - y_hat|) / sum(y), which avoids division by zero.
    3. 'mape_unfiltered': Raw MAPE (may be skewed by calm sea states).
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    # Raw unfiltered MAPE (ignoring exact zeros)
    nonzero_mask = y_true > 1e-4
    if np.any(nonzero_mask):
        raw_mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)
    else:
        raw_mape = float("nan")

    # Threshold-filtered MAPE
    thresh_mask = y_true >= min_threshold
    if np.any(thresh_mask):
        filtered_mape = float(np.mean(np.abs((y_true[thresh_mask] - y_pred[thresh_mask]) / y_true[thresh_mask])) * 100.0)
        filtered_count = int(np.sum(thresh_mask))
    else:
        filtered_mape = float("nan")
        filtered_count = 0

    # Weighted Absolute Percentage Error (WAPE)
    sum_true = float(np.sum(np.abs(y_true)))
    if sum_true > 0:
        wape = float(np.sum(np.abs(y_true - y_pred)) / sum_true * 100.0)
    else:
        wape = float("nan")

    return {
        "mape_filtered_pct": filtered_mape,
        "mape_filter_threshold_m": min_threshold,
        "mape_filtered_samples_evaluated": filtered_count,
        "wape_pct": wape,
        "mape_unfiltered_pct": raw_mape,
        "mape_limitation_note": (
            "Standard MAPE is numerically sensitive to calm sea states with small wave heights (< 0.2m), "
            "where minor absolute errors yield disproportionately large percentage spikes. "
            "Scatter Index (SI) and WAPE provide more robust relative metrics for marine regression."
        ),
    }


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute comprehensive regression metrics in physical units (meters).
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    rmse = root_mean_squared_error(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    r2 = r2_score_metric(y_true, y_pred)
    si = scatter_index(y_true, y_pred)
    bias = mean_bias_error(y_true, y_pred)
    mape_info = mean_absolute_percentage_error_safe(y_true, y_pred)

    return {
        "model": model_name,
        "rmse": rmse,
        "mae": mae_val,
        "r2": r2,
        "scatter_index": si,
        "si": si,
        "bias": bias,
        "wape_pct": mape_info["wape_pct"],
        "mape_filtered_pct": mape_info["mape_filtered_pct"],
        "mape_unfiltered_pct": mape_info["mape_unfiltered_pct"],
        "mape_notes": mape_info["mape_limitation_note"],
        "sample_count": len(y_true),
        "y_true_mean": float(np.mean(y_true)),
        "y_true_min": float(np.min(y_true)),
        "y_true_max": float(np.max(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_pred_min": float(np.min(y_pred)),
        "y_pred_max": float(np.max(y_pred)),
    }


# Backward-compatible alias
calculate_metrics = calculate_comprehensive_metrics


def measure_inference_latency(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    num_iterations: int = 100,
    warmup: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measure inference latency per sample and batch under controlled benchmark conditions.

    Protocol:
    1. Model placed in eval() mode.
    2. Inputs pre-loaded onto target device.
    3. Warmup runs executed to eliminate cold-start, JIT, and memory allocation overhead.
    4. Exact execution time measured over N iterations using time.perf_counter().
    5. Statistics computed: latency per batch, latency per sample, and throughput.
    """
    model = model.to(device)
    model.eval()
    sample_input = sample_input.to(device)
    batch_size = sample_input.size(0)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)

    # Synchronize if CUDA
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark loop
    timings = []
    with torch.no_grad():
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = model(sample_input)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append(t1 - t0)

    timings_arr = np.array(timings)
    mean_batch_sec = float(np.mean(timings_arr))
    std_batch_sec = float(np.std(timings_arr))

    latency_batch_ms = mean_batch_sec * 1000.0
    latency_sample_ms = (mean_batch_sec / batch_size) * 1000.0
    throughput = float(batch_size / mean_batch_sec)

    return {
        "batch_size": batch_size,
        "latency_batch_ms": latency_batch_ms,
        "latency_batch_std_ms": std_batch_sec * 1000.0,
        "latency_sample_ms": latency_sample_ms,
        "throughput_samples_per_sec": throughput,
        "iterations_evaluated": num_iterations,
        "device": device,
        "measurement_protocol": (
            f"Measured over {num_iterations} iterations with {warmup} warmup steps using "
            f"torch.no_grad() and high-precision time.perf_counter() on {device.upper()}."
        ),
    }

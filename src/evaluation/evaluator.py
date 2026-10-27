"""
Evaluation Benchmark Runner for comparing ANN, LSTM, and 1D-CNN forecasting models.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from .metrics import calculate_comprehensive_metrics


class WaveModelEvaluator:
    """
    Evaluator that aggregates metrics and latency benchmarks for candidate models,
    ranks models based on a specified criterion, and formats comparison tables.
    """

    def __init__(self, y_true: np.ndarray, target_units: str = "meters"):
        self.y_true = np.asarray(y_true, dtype=float).flatten()
        self.target_units = target_units
        self.predictions: Dict[str, np.ndarray] = {}
        self.latencies: Dict[str, Dict[str, Any]] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}

    def add_model_result(
        self,
        model_name: str,
        y_pred: np.ndarray,
        latency_info: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ):
        """Register a model's test predictions and benchmark results."""
        pred_arr = np.asarray(y_pred, dtype=float).flatten()
        if len(pred_arr) != len(self.y_true):
            raise ValueError(
                f"Prediction length {len(pred_arr)} for {model_name} does not match ground truth length {len(self.y_true)}"
            )
        self.predictions[model_name] = pred_arr
        if latency_info is not None:
            self.latencies[model_name] = latency_info
        if extra_meta is not None:
            self.model_metadata[model_name] = extra_meta

    def evaluate_all(self) -> pd.DataFrame:
        """Compute metrics for all registered models and return as DataFrame."""
        rows = []
        for name, pred in self.predictions.items():
            metrics = calculate_comprehensive_metrics(self.y_true, pred, model_name=name)
            if name in self.latencies:
                lat = self.latencies[name]
                metrics["latency_sample_ms"] = lat.get("latency_sample_ms", None)
                metrics["latency_batch_ms"] = lat.get("latency_batch_ms", None)
                metrics["throughput_fps"] = lat.get("throughput_samples_per_sec", None)
            rows.append(metrics)

        df = pd.DataFrame(rows)
        # Reorder key columns
        primary_cols = [
            "model",
            "rmse",
            "mae",
            "r2",
            "scatter_index",
            "bias",
            "wape_pct",
            "mape_filtered_pct",
            "latency_sample_ms",
            "throughput_fps",
        ]
        existing = [c for c in primary_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in existing]
        return df[existing + other_cols]

    def rank_models(
        self, criterion: str = "rmse", ascending: bool = True
    ) -> Tuple[str, Dict[str, Any], pd.DataFrame]:
        """
        Rank models according to a specified primary criterion (e.g. 'rmse' or 'mae').
        Returns (best_model_name, best_model_metrics, ranked_dataframe).
        """
        df = self.evaluate_all()
        if criterion not in df.columns:
            raise ValueError(f"Criterion '{criterion}' not found in metrics columns: {list(df.columns)}")

        df_sorted = df.sort_values(by=criterion, ascending=ascending).reset_index(drop=True)
        best_name = str(df_sorted.iloc[0]["model"])
        best_metrics = df_sorted.iloc[0].to_dict()
        return best_name, best_metrics, df_sorted

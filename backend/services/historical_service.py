"""
Historical predictions service for time-series queries, scatter plots, error distributions, and exports.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.config import PREDICTIONS_DIR, SUPPORTED_HORIZONS
from backend.schemas.historical import (
    HistoricalPredictionsResponse,
    HistoricalSummaryResponse,
    HistoricalTimeSeriesPoint,
    ResidualHistogramBin,
    ScatterPoint,
)


class HistoricalService:
    """Manages cached access to test split predictions artifacts."""

    def __init__(self):
        self._cache: Dict[int, pd.DataFrame] = {}

    def _load_df(self, horizon: int) -> pd.DataFrame:
        """Loads and caches test predictions CSV for the requested horizon."""
        if horizon not in SUPPORTED_HORIZONS:
            raise ValueError(f"Unsupported horizon {horizon}h. Choose from {SUPPORTED_HORIZONS}.")

        if horizon in self._cache:
            return self._cache[horizon]

        filename = "test_predictions.csv" if horizon == 1 else f"test_predictions_h{horizon}.csv"
        csv_path = PREDICTIONS_DIR / filename
        if not csv_path.is_file():
            raise FileNotFoundError(f"Predictions artifact '{filename}' not found at {PREDICTIONS_DIR}")

        df = pd.read_csv(csv_path)
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df = df.sort_values("valid_time").reset_index(drop=True)
        self._cache[horizon] = df
        return df

    def get_summary(self, horizon: int = 1) -> HistoricalSummaryResponse:
        """Computes summary statistics across the test split predictions."""
        df = self._load_df(horizon)
        actuals = df["actual_swh_meters"]

        return HistoricalSummaryResponse(
            total_test_samples=len(df),
            date_range_start=str(df["valid_time"].min()),
            date_range_end=str(df["valid_time"].max()),
            overall_min_swh=round(float(actuals.min()), 3),
            overall_max_swh=round(float(actuals.max()), 3),
            overall_mean_swh=round(float(actuals.mean()), 3),
            high_wave_events_count=int((actuals >= 2.5).sum()),
            available_horizons=SUPPORTED_HORIZONS,
            total_samples=len(df),
            start_date=str(df["valid_time"].min()),
            end_date=str(df["valid_time"].max()),
        )

    def get_time_series(
        self,
        horizon: int = 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        downsample: int = 500,
    ) -> HistoricalPredictionsResponse:
        """Returns filtered and evenly downsampled actual vs predicted points."""
        df = self._load_df(horizon)

        filtered = df
        if start_date:
            s_dt = pd.to_datetime(start_date)
            filtered = filtered[filtered["valid_time"] >= s_dt]
        if end_date:
            e_dt = pd.to_datetime(end_date)
            filtered = filtered[filtered["valid_time"] <= e_dt]

        total_in_range = len(filtered)
        if total_in_range == 0:
            return HistoricalPredictionsResponse(
                horizon_hours=horizon,
                total_samples=0,
                returned_samples=0,
                start_time=start_date or "",
                end_time=end_date or "",
                points=[],
                data=[],
            )

        # Downsample evenly if sequence is large for instant browser rendering
        if total_in_range > downsample:
            step = max(1, total_in_range // downsample)
            sampled = filtered.iloc[::step].copy()
        else:
            sampled = filtered.copy()

        points: List[HistoricalTimeSeriesPoint] = []
        for _, r in sampled.iterrows():
            t_str = r["valid_time"].strftime("%Y-%m-%d %H:%M")
            act_val = round(float(r["actual_swh_meters"]), 3)
            ann_p = round(float(r["ann_pred_meters"]), 3) if "ann_pred_meters" in r else None
            lstm_p = round(float(r["lstm_pred_meters"]), 3) if "lstm_pred_meters" in r else None
            cnn_p = round(float(r["cnn1d_pred_meters"]), 3) if "cnn1d_pred_meters" in r else None
            ann_e = round(float(r["ann_error_meters"]), 4) if "ann_error_meters" in r else None
            lstm_e = round(float(r["lstm_error_meters"]), 4) if "lstm_error_meters" in r else None
            cnn_e = round(float(r["cnn1d_error_meters"]), 4) if "cnn1d_error_meters" in r else None

            points.append(
                HistoricalTimeSeriesPoint(
                    valid_time=t_str,
                    actual_swh=act_val,
                    ann_pred=ann_p,
                    lstm_pred=lstm_p,
                    cnn1d_pred=cnn_p,
                    ann_error=ann_e,
                    lstm_error=lstm_e,
                    cnn1d_error=cnn_e,
                    timestamp=t_str,
                    actual=act_val,
                    ann=ann_p,
                    lstm=lstm_p,
                    cnn1d=cnn_p,
                    predicted=lstm_p,
                    residual=lstm_e,
                    error_abs=abs(lstm_e) if lstm_e is not None else None,
                )
            )

        return HistoricalPredictionsResponse(
            horizon_hours=horizon,
            total_samples=total_in_range,
            returned_samples=len(points),
            start_time=str(filtered["valid_time"].min()),
            end_time=str(filtered["valid_time"].max()),
            points=points,
            data=points,
        )

    def get_scatter(
        self,
        horizon: int = 1,
        model: str = "lstm",
        sample_size: int = 800,
    ) -> List[ScatterPoint]:
        """Evenly samples actual vs predicted pairs for correlation scatter chart."""
        df = self._load_df(horizon)
        m_code = model.lower().replace("wave", "").strip()
        pred_col = f"{m_code}_pred_meters"
        err_col = f"{m_code}_error_meters"

        if pred_col not in df.columns:
            m_code = "lstm"
            pred_col = "lstm_pred_meters"
            err_col = "lstm_error_meters"

        # Downsample evenly across range
        step = max(1, len(df) // sample_size)
        sampled = df.iloc[::step]

        scatter_points: List[ScatterPoint] = []
        for _, r in sampled.iterrows():
            scatter_points.append(
                ScatterPoint(
                    actual=round(float(r["actual_swh_meters"]), 3),
                    predicted=round(float(r[pred_col]), 3),
                    error=round(float(r[err_col]), 4),
                    timestamp=r["valid_time"].strftime("%Y-%m-%d %H:%M"),
                )
            )
        return scatter_points

    def get_residuals(
        self,
        horizon: int = 1,
        model: str = "lstm",
        num_bins: int = 30,
    ) -> List[ResidualHistogramBin]:
        """Computes histogram bins for prediction error residuals."""
        df = self._load_df(horizon)
        m_code = model.lower().replace("wave", "").strip()
        err_col = f"{m_code}_error_meters"
        if err_col not in df.columns:
            err_col = "lstm_error_meters"

        errors = df[err_col].dropna().values
        counts, bin_edges = np.histogram(errors, bins=num_bins)

        bins: List[ResidualHistogramBin] = []
        for i in range(len(counts)):
            b_min = float(bin_edges[i])
            b_max = float(bin_edges[i + 1])
            b_center = (b_min + b_max) / 2.0
            bins.append(
                ResidualHistogramBin(
                    bin_center=round(b_center, 4),
                    bin_min=round(b_min, 4),
                    bin_max=round(b_max, 4),
                    count=int(counts[i]),
                )
            )
        return bins

    def get_csv_filepath(self, horizon: int) -> Path:
        """Returns the file path to test predictions CSV for download."""
        if horizon not in SUPPORTED_HORIZONS:
            horizon = 1
        filename = "test_predictions.csv" if horizon == 1 else f"test_predictions_h{horizon}.csv"
        path = PREDICTIONS_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Export file {filename} not found.")
        return path


historical_service = HistoricalService()

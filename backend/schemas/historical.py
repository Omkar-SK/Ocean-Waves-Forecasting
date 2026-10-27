"""
Historical analysis Pydantic schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HistoricalTimeSeriesPoint(BaseModel):
    valid_time: str
    actual_swh: float
    ann_pred: Optional[float] = None
    lstm_pred: Optional[float] = None
    cnn1d_pred: Optional[float] = None
    ann_error: Optional[float] = None
    lstm_error: Optional[float] = None
    cnn1d_error: Optional[float] = None
    # Compatibility aliases for frontend
    timestamp: Optional[str] = None
    actual: Optional[float] = None
    ann: Optional[float] = None
    lstm: Optional[float] = None
    cnn1d: Optional[float] = None
    predicted: Optional[float] = None
    residual: Optional[float] = None
    error_abs: Optional[float] = None


class ScatterPoint(BaseModel):
    actual: float
    predicted: float
    error: float
    timestamp: str


class ResidualHistogramBin(BaseModel):
    bin_center: float
    bin_min: float
    bin_max: float
    count: int


class HistoricalPredictionsResponse(BaseModel):
    horizon_hours: int
    total_samples: int
    returned_samples: int
    start_time: str
    end_time: str
    points: List[HistoricalTimeSeriesPoint]
    data: List[HistoricalTimeSeriesPoint] = Field(default_factory=list)


class HistoricalSummaryResponse(BaseModel):
    total_test_samples: int
    date_range_start: str
    date_range_end: str
    overall_min_swh: float
    overall_max_swh: float
    overall_mean_swh: float
    high_wave_events_count: int
    available_horizons: List[int]
    # Compatibility aliases
    total_samples: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

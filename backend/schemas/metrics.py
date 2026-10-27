"""
Evaluation metrics Pydantic schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MetricEntry(BaseModel):
    horizon_hours: int
    model: str
    model_type: str
    status: str
    test_sample_count: int
    target_start_time: str
    target_end_time: str
    rmse_meters: float
    mae_meters: float
    r2_score: float
    scatter_index_pct: float
    wape_pct: float
    bias_meters: float
    rmse_skill_pct: float
    mae_skill_pct: float
    notes: str
    # Frontend aliases
    horizon: Optional[int] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    is_best: Optional[bool] = False


class BestModelSummary(BaseModel):
    horizon_hours: int
    best_on_rmse: Dict[str, Any]
    best_on_mae: Dict[str, Any]
    best_neural_net: Dict[str, Any]
    key_takeaway: str


class MultiHorizonMetricsResponse(BaseModel):
    items: List[MetricEntry]
    metrics: List[MetricEntry] = Field(default_factory=list)
    horizons: List[int] = Field(default_factory=lambda: [1, 6, 24])
    best_by_horizon: Dict[int, BestModelSummary]
    summary_analysis: str


class DetailedModelMetric(BaseModel):
    model: str
    rmse: float
    mae: float
    r2: float
    scatter_index: float
    bias: float
    wape_pct: float
    mape_filtered_pct: float
    latency_sample_ms: float
    throughput_fps: float
    sample_count: int


class DetailedHorizonResponse(BaseModel):
    horizon_hours: int
    input_window_hours: int
    best_model: str
    selection_criterion: str
    models: List[DetailedModelMetric] = Field(default_factory=list)
    metrics: List[DetailedModelMetric] = Field(default_factory=list)
    evaluation_split: str = "Test Set (2024-07-31 to 2026-10-17)"
    target_date_range: Dict[str, str] = Field(
        default_factory=lambda: {"start": "2024-07-31 00:00", "end": "2026-10-17 23:00"}
    )


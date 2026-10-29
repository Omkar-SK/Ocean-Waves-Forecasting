"""
Pydantic schemas for coastal risk heatmap, telemetry nodes, and geographic metrics.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NodeMapSummaryItem(BaseModel):
    location_id: str
    name: str
    latitude: float
    longitude: float
    coordinate_label: str
    region: str
    is_reference: bool
    validation_status: str
    data_status: str
    data_source: str
    retrieval_time_utc: Optional[str] = None
    forecast_issue_time_utc: Optional[str] = None
    latest_obs_swh_m: Optional[float] = None
    latest_obs_time_utc: Optional[str] = None
    forecast_swh_m: Optional[float] = None
    forecast_horizon_hours: int
    forecast_valid_time_utc: Optional[str] = None
    max_window_swh_m: Optional[float] = None
    current_risk_level: str
    highest_risk_level: str
    marker_hex_color: str
    threshold_crossed_m: Optional[float] = None
    active_alert_count: int
    active_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    is_degraded: bool = False
    trigger_explanation: str = ""
    limitations: List[str] = Field(default_factory=list)
    multi_horizon_predictions: Dict[int, float] = Field(default_factory=dict)
    external_nwp_forecasts: Dict[int, float] = Field(default_factory=dict)


class CoastalMetricsResponse(BaseModel):
    total_nodes: int
    reference_nodes: int
    active_alerts: int
    high_risk_nodes: int
    degraded_feeds: int
    timestamp_utc: str

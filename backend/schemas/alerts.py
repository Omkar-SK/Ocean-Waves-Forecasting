"""
Pydantic schemas for coastal risk alerts and threshold configurations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AlertItem(BaseModel):
    alert_id: str
    fingerprint: str
    location_id: str
    location_name: str
    event_type: str
    severity: str
    status: str
    trigger_reason: str
    predicted_swh_m: float
    threshold_m: float
    forecast_horizon_hours: int
    expected_event_time_utc: str
    anchor_time_utc: str
    forecast_issue_time_utc: Optional[str] = None
    assessment_time_utc: str
    source_name: str
    data_status: str
    model_name: Optional[str] = None
    validation_status: str
    is_reference: bool
    created_at_utc: str
    updated_at_utc: str
    acknowledged_at_utc: Optional[str] = None
    resolved_at_utc: Optional[str] = None
    expired_at_utc: Optional[str] = None
    operator_notes: Optional[str] = None


class AlertAcknowledgeRequest(BaseModel):
    operator_notes: Optional[str] = Field(None, description="Optional notes logged by operator")


class AlertSummaryResponse(BaseModel):
    total_active: int
    critical_red: int
    high_orange: int
    caution_yellow: int
    resolved_count: int
    status_counts: Dict[str, int]
    last_assessment_utc: Optional[str] = None


class AlertHistoryItem(BaseModel):
    history_id: int
    from_status: Optional[str]
    to_status: str
    reason: str
    changed_at_utc: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ThresholdConfigResponse(BaseModel):
    caution_m: float
    high_risk_m: float
    critical_m: float
    hysteresis_buffer_m: float
    units: str
    version: str
    is_demonstration: bool
    guidance_notice: str
    location_id: Optional[str] = None

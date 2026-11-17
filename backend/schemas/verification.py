"""
Pydantic schemas for Continuous Real-Time Forecast Verification and Model Performance Monitoring.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VerificationRecordStatus(str, Enum):
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    VERIFIED = "VERIFIED"
    OBSERVATION_UNAVAILABLE = "OBSERVATION_UNAVAILABLE"
    INVALID_OBSERVATION = "INVALID_OBSERVATION"


class VerificationWindow(str, Enum):
    LAST_24H = "last_24h"
    LAST_7D = "last_7d"
    LAST_30D = "last_30d"
    ALL_TIME = "all_time"


class ForecastArchivalRequest(BaseModel):
    forecast_id: Optional[str] = Field(None, description="Unique forecast ID. Auto-generated if omitted.")
    location_id: str = Field(..., description="Registered coastal node identifier (e.g., mumbai_offshore)")
    forecast_origin_time: datetime = Field(..., description="Timestamp when the forecast was generated (UTC)")
    target_timestamp: datetime = Field(..., description="Target validity timestamp for the prediction (UTC)")
    forecast_horizon: int = Field(..., description="Forecast horizon in hours (1, 6, or 24)")
    current_swh: float = Field(..., ge=0.0, le=25.0, description="Significant Wave Height at forecast origin (m)")
    predicted_swh: float = Field(..., ge=0.0, le=25.0, description="Predicted Significant Wave Height (m)")
    predicted_residual: Optional[float] = Field(None, description="Predicted wave height change ΔSWH (m)")
    model_architecture: str = Field(..., description="Architecture name (e.g., Residual WaveLSTM, Residual WaveANN)")
    model_version: str = Field("2.0.0-residual", description="Model version tag")
    input_data_timestamp: datetime = Field(..., description="Timestamp of the latest input observation in sequence")
    data_source: str = Field("OpenMeteoProvider", description="Data ingestion provider name")
    data_freshness_minutes: Optional[int] = Field(None, ge=0, description="Telemetry age in minutes at forecast issuance")
    forecast_validation_status: str = Field("reference_validated", description="Geographic validation status")
    is_reference: bool = Field(True, description="Whether location is the reference node")
    external_nwp_swh: Optional[float] = Field(None, ge=0.0, le=25.0, description="Optional external NWP benchmark SWH (m)")


class ForecastArchivalResponse(BaseModel):
    success: bool
    forecast_id: str
    is_duplicate: bool
    status: VerificationRecordStatus
    message: str


class ObservationIngestRequest(BaseModel):
    location_id: str = Field(..., description="Registered coastal node identifier")
    observation_time: datetime = Field(..., description="Timestamp of observation measurement (UTC)")
    observed_swh: float = Field(..., description="Observed Significant Wave Height (m)")
    station_id: Optional[str] = Field(None, description="Optional buoy or sensor station code")
    source_name: str = Field("INCOIS_BUOY_NETWORK", description="Reporting agency or network")
    source_type: str = Field("in_situ_buoy", description="in_situ_buoy, coastal_sensor, satellite_altimetry, external_nwp")
    quality_status: str = Field("valid", description="valid, suspect, rejected")
    notes: Optional[str] = Field(None, description="Diagnostic observation metadata")


class ObservationIngestResponse(BaseModel):
    success: bool
    observation_id: str
    is_duplicate: bool
    matched_forecasts_count: int
    message: str


class VerificationRecordSchema(BaseModel):
    forecast_id: str
    location_id: str
    forecast_origin_time: datetime
    target_timestamp: datetime
    forecast_horizon: int
    model_architecture: str
    model_version: str
    current_swh: float
    predicted_swh: float
    predicted_residual: Optional[float] = None
    observed_swh: Optional[float] = None
    observation_time: Optional[datetime] = None
    forecast_error: Optional[float] = None  # predicted - observed
    absolute_error: Optional[float] = None  # abs(predicted - observed)
    persistence_error: Optional[float] = None  # current_swh - observed
    status: VerificationRecordStatus
    status_reason: Optional[str] = None
    is_prospective: bool
    is_severe_actual: bool
    is_severe_forecast: bool
    data_source: str
    validation_status: str
    created_at_utc: datetime
    verified_at_utc: Optional[datetime] = None


class AbsoluteErrorDistribution(BaseModel):
    p25: Optional[float] = None
    median: Optional[float] = None
    p75: Optional[float] = None
    p90: Optional[float] = None
    p95: Optional[float] = None
    max_error: Optional[float] = None


class ContinuousAccuracyMetrics(BaseModel):
    horizon_hours: int
    model_architecture: str
    window: str
    sample_count: int
    total_issued_count: int
    coverage_pct: float
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    bias: Optional[float] = None
    persistence_mae: Optional[float] = None
    persistence_rmse: Optional[float] = None
    rmse_skill_pct: Optional[float] = None
    mae_skill_pct: Optional[float] = None
    error_distribution: AbsoluteErrorDistribution
    is_statistically_reliable: bool
    min_samples_required: int
    reliability_notice: str


class SevereEventTrackingMetrics(BaseModel):
    actual_severe_events_count: int = Field(..., description="Count of verified observations where SWH >= 4.0m")
    forecasted_severe_events_count: int = Field(..., description="Count of forecasts where predicted SWH >= 4.0m")
    underprediction_frequency_pct: Optional[float] = Field(None, description="% of severe events where predicted < observed")
    missed_severe_events_count: int = Field(..., description="Count where observed >= 4.0m but predicted < 4.0m")
    false_alarms_count: int = Field(..., description="Count where predicted >= 4.0m but observed < 4.0m")
    average_detection_lead_time_hours: Optional[float] = Field(None, description="Mean lead time of correctly detected severe events")
    historical_validation_sample_size: int = Field(22, description="Reference sample count of severe events in ERA5 test split")
    historical_severe_rmse_skill: Dict[str, float] = Field(
        default_factory=lambda: {"1h": 11.91, "6h": -5.58, "24h": -35.61},
        description="Historical benchmark skill in severe sea states (SWH >= 4.0m)",
    )
    operational_severe_sample_size: int = Field(..., description="Real-time operational severe events observed")
    operational_severe_status: str = Field(..., description="INSUFFICIENT_OPERATIONAL_SAMPLE or OPERATIONAL_EVIDENCE_ACCUMULATING")
    scientific_notice: str = Field(
        ...,
        description="Scientific caveat distinguishing historical validation from real-time operational evidence",
    )


class ModelHealthMonitoringSummary(BaseModel):
    prediction_drift_status: str  # NORMAL, WATCH, DRIFT_DETECTED
    input_feature_drift_status: str  # NORMAL, WATCH, DRIFT_DETECTED
    missing_observations_count: int
    missing_observations_rate_pct: float
    stale_data_alerts_count: int
    increasing_forecast_errors_warning: bool
    error_degradation_ratio: Optional[float] = None
    extreme_underprediction_warning: bool
    model_inference_failures_count: int
    active_warnings: List[str]


class VerificationDashboardResponse(BaseModel):
    evaluation_time_utc: datetime
    location_id: Optional[str] = None
    total_forecasts_issued: int
    total_verified_observations: int
    total_pending_verifications: int
    total_unavailable_observations: int
    total_invalid_observations: int
    horizon_metrics: Dict[str, Dict[str, ContinuousAccuracyMetrics]]
    severe_events: SevereEventTrackingMetrics
    monitoring_health: ModelHealthMonitoringSummary
    recent_verifications: List[VerificationRecordSchema]
    historical_vs_operational_summary: Dict[str, Any]

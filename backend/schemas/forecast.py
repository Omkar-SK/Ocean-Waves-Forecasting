"""
Forecast request and response Pydantic schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    horizon: int = Field(1, description="Forecast horizon in hours: 1, 6, or 24")
    model: str = Field("all", description="'ann', 'lstm', 'cnn1d', 'candidate', or 'all'")
    source: str = Field("live", description="'live' (Open-Meteo REST) or 'demo' (ERA5 historical window)")
    latitude: Optional[float] = Field(18.50, description="Target offshore latitude in °N")
    longitude: Optional[float] = Field(72.50, description="Target offshore longitude in °E")
    location_id: Optional[str] = Field(None, description="Optional registered coastal node ID (e.g. 'mumbai_offshore', 'goa_mormugao')")
    forecast_mode: str = Field("residual", description="'residual' (validated ΔSWH forecasters) or 'absolute' (original checkpoints)")


class SingleModelForecast(BaseModel):
    model_name: str
    model_type: str
    checkpoint_file: str
    forecast_swh_m: float
    delta_vs_persistence_m: float
    delta_vs_nwp_m: Optional[float] = None
    sea_state_desc: str
    sea_state_color: str
    # Phase 5: Residual learning integration fields
    current_swh_m: Optional[float] = None
    predicted_residual_m: Optional[float] = None
    forecast_mode: str = "residual"
    model_version: str = "2.0.0-residual"
    validation_status: str = "reference_validated"
    rmse_skill_score: Optional[float] = None
    is_candidate_model: bool = False
    reconstruction_formula: Optional[str] = None
    # Explicit schema attributes
    predicted_swh: Optional[float] = None
    current_swh: Optional[float] = None
    predicted_residual: Optional[float] = None
    selected_model_architecture: Optional[str] = None
    model_validation_status: Optional[str] = None


class ObservationWindowStep(BaseModel):
    valid_time: str
    swh: float
    mwp: float
    mwd: Optional[float] = None
    wind_speed: float
    wind_dir: Optional[float] = None
    u10: Optional[float] = None
    v10: Optional[float] = None


class CoastalNodeSchema(BaseModel):
    location_id: str
    name: str
    latitude: float
    longitude: float
    coordinate_label: str
    region: str
    timezone: str
    enabled: bool
    is_reference: bool
    validation_status: str
    description: str
    notes: List[str]


class ForecastResponse(BaseModel):
    anchor_time_utc: datetime
    target_time_utc: datetime
    forecast_horizon_hours: int
    persistence_swh_m: float
    external_nwp_swh_m: Optional[float] = None
    predictions: Dict[str, SingleModelForecast]
    observation_window: List[Dict[str, Any]]
    source_name: str
    attribution: str
    is_demo: bool
    notes: List[str]
    generated_at_utc: datetime
    location_id: Optional[str] = "mumbai_offshore"
    location_name: Optional[str] = "Mumbai Offshore (Reference Node)"
    is_reference: bool = True
    validation_status: str = "reference_validated"
    data_status: str = "fresh"
    forecast_issue_time_utc: Optional[datetime] = None

    # Phase 5: Enhanced residual reporting attributes
    forecast_mode: str = "residual"
    current_swh: Optional[float] = None
    current_swh_m: Optional[float] = None
    predicted_swh: Optional[float] = None
    predicted_residual: Optional[float] = None
    forecast_horizon: Optional[int] = None
    selected_model_architecture: Optional[str] = None
    model_version: Optional[str] = None
    input_timestamp: Optional[datetime] = None
    data_source: Optional[str] = None
    data_freshness: Optional[str] = None
    data_freshness_minutes: Optional[int] = None
    forecast_generation_timestamp: Optional[datetime] = None
    model_validation_status: Optional[str] = None


class ForecastStatusResponse(BaseModel):
    live_api_online: bool
    demo_mode_available: bool
    latest_observation_time_utc: Optional[datetime] = None
    data_freshness_minutes: Optional[int] = None
    checkpoints_ready: Dict[str, bool]
    message: str
    location_id: Optional[str] = "mumbai_offshore"
    validation_status: Optional[str] = "reference_validated"


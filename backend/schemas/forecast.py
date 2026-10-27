"""
Forecast request and response Pydantic schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    horizon: int = Field(1, description="Forecast horizon in hours: 1, 6, or 24")
    model: str = Field("all", description="'ann', 'lstm', 'cnn1d', or 'all'")
    source: str = Field("live", description="'live' (Open-Meteo REST) or 'demo' (ERA5 historical window)")
    latitude: Optional[float] = Field(18.50, description="Target offshore latitude in °N")
    longitude: Optional[float] = Field(72.50, description="Target offshore longitude in °E")


class SingleModelForecast(BaseModel):
    model_name: str
    model_type: str
    checkpoint_file: str
    forecast_swh_m: float
    delta_vs_persistence_m: float
    delta_vs_nwp_m: Optional[float] = None
    sea_state_desc: str
    sea_state_color: str


class ObservationWindowStep(BaseModel):
    valid_time: str
    swh: float
    mwp: float
    mwd: Optional[float] = None
    wind_speed: float
    wind_dir: Optional[float] = None
    u10: Optional[float] = None
    v10: Optional[float] = None


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


class ForecastStatusResponse(BaseModel):
    live_api_online: bool
    demo_mode_available: bool
    latest_observation_time_utc: Optional[datetime] = None
    data_freshness_minutes: Optional[int] = None
    checkpoints_ready: Dict[str, bool]
    message: str

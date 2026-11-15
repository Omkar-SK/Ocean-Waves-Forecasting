"""
Pydantic schemas for the Ocean Hazard Simulation Subsystem.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ScenarioMetadataItem(BaseModel):
    scenario_id: str
    name: str
    category: str
    headline: str
    description: str
    severity_category: str
    coastal_impact_level: str
    recommended_action: str
    base_swh_m: float
    peak_swh_m: float
    peak_wind_kmh: float
    peak_mwp_s: float


class SimulationStatusResponse(BaseModel):
    is_active: bool
    state: str
    scenario_id: Optional[str] = None
    scenario_name: Optional[str] = None
    current_step: int
    total_steps: int
    elapsed_hours: int
    started_at_utc: Optional[str] = None
    updated_at_utc: Optional[str] = None
    disclaimer: str


class SimulationStartRequest(BaseModel):
    scenario_id: str = Field(..., description="'severe_wave', 'tropical_cyclone', 'coastal_flooding', or 'moderate_disturbance'")
    initial_step: int = Field(6, ge=0, le=24, description="Initial hour step along 24h timeline (default 6)")


class HeatmapPoint(BaseModel):
    latitude: float
    longitude: float
    label: str
    region: str
    swh_m: float
    max_wave_m: float
    mwp_s: float
    wind_kmh: float
    wind_dir_deg: float
    intensity: float
    risk_level: str
    color: str
    radius_km: float


class SimulationStats(BaseModel):
    active_alerts: int
    critical_risk_zones: int
    high_risk_zones: int
    max_swh_m: float
    max_wind_kmh: float
    coastal_risk_index: int
    affected_regions: List[str]
    scenario_name: str
    severity_category: str
    headline: str
    coastal_impact_level: str
    recommended_action: str


class SimulatedTimeSeriesPoint(BaseModel):
    step: int
    time: str
    timestamp: str
    actual: Optional[float] = None
    lstm: Optional[float] = None
    ann: Optional[float] = None
    cnn1d: Optional[float] = None
    persistence: Optional[float] = None
    wind_kmh: float
    mwp_s: float
    risk_level: str
    isForecast: bool


class SimulationResultsResponse(BaseModel):
    status: SimulationStatusResponse
    scenario: Optional[ScenarioMetadataItem] = None
    stats: Optional[SimulationStats] = None
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    heatmap_points: List[HeatmapPoint] = Field(default_factory=list)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    time_series: List[SimulatedTimeSeriesPoint] = Field(default_factory=list)


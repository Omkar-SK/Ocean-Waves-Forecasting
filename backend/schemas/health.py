"""
Health and metadata Pydantic schemas.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ComponentStatus(BaseModel):
    name: str
    status: str = Field(..., description="'healthy', 'degraded', or 'unhealthy'")
    message: str
    details: Optional[Dict[str, str]] = None


class SystemHealthResponse(BaseModel):
    status: str
    backend_online: bool = True
    checkpoints_available: int
    total_checkpoints: int = 9
    scaler_available: bool
    dataset_available: bool
    live_api_available: bool
    timestamp: datetime
    components: List[ComponentStatus]


class SystemMetadataResponse(BaseModel):
    project_name: str
    academic_title: str
    version: str
    selected_coordinates: Dict[str, float]
    location_name: str
    supported_horizons: List[int]
    supported_models: List[str]
    input_window_hours: int
    features: List[str]
    target: str
    units: Dict[str, str]
    dataset_temporal_span: Dict[str, str]
    best_overall_model: str

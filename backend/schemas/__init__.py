"""
Pydantic Schemas for OceanCast AI.
"""

from .health import SystemHealthResponse, SystemMetadataResponse
from .metrics import MetricEntry, MultiHorizonMetricsResponse
from .forecast import ForecastRequest, ForecastResponse, ForecastStatusResponse
from .historical import HistoricalTimeSeriesPoint, HistoricalSummaryResponse

__all__ = [
    "SystemHealthResponse",
    "SystemMetadataResponse",
    "MetricEntry",
    "MultiHorizonMetricsResponse",
    "ForecastRequest",
    "ForecastResponse",
    "ForecastStatusResponse",
    "HistoricalTimeSeriesPoint",
    "HistoricalSummaryResponse",
]

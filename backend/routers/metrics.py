"""
Model performance and multi-horizon benchmark metrics endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from backend.config import SUPPORTED_HORIZONS
from backend.schemas.metrics import (
    DetailedHorizonResponse,
    MetricEntry,
    MultiHorizonMetricsResponse,
)
from backend.services.metrics_service import metrics_service

router = APIRouter(prefix="/api/metrics", tags=["Metrics & Benchmarks"])


@router.get("/multi-horizon", response_model=MultiHorizonMetricsResponse)
def get_multi_horizon_metrics():
    """Returns the verified multi-horizon benchmark comparison table across 1h, 6h, and 24h horizons."""
    try:
        return metrics_service.get_multi_horizon_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {str(e)}")


@router.get("", response_model=List[MetricEntry])
def get_metrics_for_horizon(
    horizon: Optional[int] = Query(None, description="Optional filter by horizon: 1, 6, or 24")
):
    """Returns benchmark metric items, optionally filtered by horizon."""
    try:
        full_res = metrics_service.get_multi_horizon_metrics()
        if horizon is not None:
            if horizon not in SUPPORTED_HORIZONS:
                raise HTTPException(status_code=400, detail=f"Invalid horizon {horizon}h. Choose from {SUPPORTED_HORIZONS}.")
            return [m for m in full_res.items if m.horizon_hours == horizon]
        return full_res.items
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {str(e)}")


@router.get("/detailed/{horizon}", response_model=DetailedHorizonResponse)
def get_detailed_horizon_metrics(horizon: int):
    """Returns detailed latency, throughput, and error metrics for a specific horizon."""
    try:
        return metrics_service.get_detailed_horizon_metrics(horizon)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error loading detailed metrics: {str(e)}")

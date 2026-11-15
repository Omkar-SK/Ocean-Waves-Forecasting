"""
Operational live forecast and inference endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from backend.config import SUPPORTED_HORIZONS
from backend.schemas.forecast import (
    CoastalNodeSchema,
    ForecastRequest,
    ForecastResponse,
    ForecastStatusResponse,
)
from backend.services.live_service import live_forecast_service
from src.intelligence.nodes import get_coastal_registry

router = APIRouter(prefix="/api/forecast", tags=["Forecasting"])


@router.get("/locations", response_model=List[CoastalNodeSchema])
def list_coastal_locations():
    """List all registered coastal forecasting locations with validation status."""
    return get_coastal_registry().to_dict_list(enabled_only=False)


@router.get("/status", response_model=ForecastStatusResponse)
def get_forecast_status():
    """Checks real-time external API health, latest observation timestamp, and data freshness."""
    return live_forecast_service.check_status()


@router.post("", response_model=ForecastResponse)
def generate_forecast(req: ForecastRequest):
    """Executes wave height forecasting across selected horizons and deep learning architectures."""
    try:
        return live_forecast_service.generate_forecast(req)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except ConnectionError as ce:
        raise HTTPException(
            status_code=503,
            detail=f"Live observation service unavailable: {str(ce)}. Try switching to demo mode.",
        )
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast inference error: {str(e)}")


@router.get("/live", response_model=ForecastResponse)
def get_live_forecast_convenience(
    horizon: int = Query(6, description="Lead time: 1, 6, or 24 hours"),
    model: Optional[str] = Query(None, description="'ann', 'lstm', 'cnn1d', or 'all'"),
    model_name: Optional[str] = Query(None, description="Alias for model"),
    source: Optional[str] = Query("live", description="'live' or 'demo'"),
    location_id: Optional[str] = Query(None, description="Registered coastal node ID"),
    use_demo_fallback: Optional[bool] = Query(None, description="Fallback flag"),
):
    """Convenience GET endpoint for live forecast fetching."""
    selected_model = model or model_name or "all"
    # Normalize model string
    selected_model = selected_model.lower().replace("wave", "")
    if selected_model not in ["ann", "lstm", "cnn1d", "all"]:
        selected_model = "all"

    selected_source = "demo" if source == "demo" else "live"
    try:
        req = ForecastRequest(
            horizon=horizon,
            model=selected_model,
            source=selected_source,
            location_id=location_id,
        )
        return live_forecast_service.generate_forecast(req)
    except ConnectionError as ce:
        if use_demo_fallback or selected_source == "live":
            # Fall back to demo mode gracefully if network drops
            req = ForecastRequest(
                horizon=horizon,
                model=selected_model,
                source="demo",
                location_id=location_id,
            )
            fallback_res = live_forecast_service.generate_forecast(req)
            fallback_res.notes.append(
                f"⚠️ Live observation service unavailable ({ce}). Automatically fell back to offline ERA5 historical demo mode."
            )
            return fallback_res
        raise HTTPException(
            status_code=503,
            detail=f"Live observation service unavailable: {ce}. Try switching to demo mode.",
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast inference error: {str(e)}")


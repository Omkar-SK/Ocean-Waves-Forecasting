"""
REST API endpoints for Continuous Real-Time Forecast Verification and Model Performance Monitoring.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query

from backend.schemas.verification import (
    ContinuousAccuracyMetrics,
    ForecastArchivalRequest,
    ForecastArchivalResponse,
    ModelHealthMonitoringSummary,
    ObservationIngestRequest,
    ObservationIngestResponse,
    SevereEventTrackingMetrics,
    VerificationDashboardResponse,
    VerificationRecordSchema,
    VerificationRecordStatus,
    VerificationWindow,
)
from backend.services.forecast_verification_service import forecast_verification_service

router = APIRouter(prefix="/api/verification", tags=["Forecast Verification & Performance Monitoring"])


@router.get("/dashboard", response_model=VerificationDashboardResponse)
def get_verification_dashboard_endpoint(
    location_id: Optional[str] = Query(None, description="Optional coastal node ID (e.g., mumbai_offshore)"),
    limit: int = Query(50, ge=5, le=500, description="Max recent verification records to return"),
):
    """
    Returns full continuous forecast verification dashboard including:
    - Overall issuance and matching statistics
    - Multi-horizon accuracy metrics across rolling 24h, 7d, 30d, and all-time windows
    - Severe sea state (SWH >= 4.0m) tracking and historical vs operational comparison
    - Model health monitoring alerts and drift status
    - Recent prospective verification records
    """
    try:
        return forecast_verification_service.get_verification_dashboard(
            location_id=location_id,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error compiling verification dashboard: {str(e)}")


@router.get("/metrics", response_model=ContinuousAccuracyMetrics)
def get_horizon_verification_metrics(
    horizon: int = Query(6, description="Lead time in hours: 1, 6, or 24"),
    window: str = Query("all_time", description="'last_24h', 'last_7d', 'last_30d', or 'all_time'"),
    location_id: Optional[str] = Query(None, description="Optional coastal node ID"),
):
    """
    Retrieve continuous prospective accuracy metrics (MAE, RMSE, Bias, Skill Score vs. Persistence)
    for a specific forecast horizon and temporal window.
    """
    if horizon not in [1, 6, 24]:
        raise HTTPException(status_code=400, detail=f"Unsupported horizon {horizon}h. Choose 1, 6, or 24.")

    try:
        verif_win = VerificationWindow(window)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid window '{window}'. Choose from 'last_24h', 'last_7d', 'last_30d', 'all_time'.")

    return forecast_verification_service.calculate_window_metrics(
        horizon=horizon,
        window=verif_win,
        location_id=location_id,
    )


@router.get("/severe-events", response_model=SevereEventTrackingMetrics)
def get_severe_events_monitoring(
    location_id: Optional[str] = Query(None, description="Optional coastal node ID"),
):
    """
    Returns extreme-event verification metrics for wave heights >= 4.0m,
    strictly distinguishing the historical validation benchmark (N=22) from live operational evidence.
    """
    return forecast_verification_service.calculate_severe_event_metrics(location_id=location_id)


@router.get("/health", response_model=ModelHealthMonitoringSummary)
def get_model_health_monitoring(
    location_id: Optional[str] = Query(None, description="Optional coastal node ID"),
):
    """
    Returns automated ML operations health checks including missing observation rates,
    stale data frequency, and rolling error degradation against baseline.
    """
    return forecast_verification_service.get_model_health_monitoring_summary(location_id=location_id)


@router.post("/archive-forecast", response_model=ForecastArchivalResponse)
def archive_forecast_endpoint(req: ForecastArchivalRequest):
    """
    Archives an issued wave forecast into the verification queue with duplicate suppression.
    """
    try:
        return forecast_verification_service.archive_forecast(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast archival failed: {str(e)}")


@router.post("/ingest-observation", response_model=ObservationIngestResponse)
def ingest_observation_endpoint(req: ObservationIngestRequest):
    """
    Ingests an independent marine measurement and triggers prospective matching against pending forecasts.
    """
    try:
        res = forecast_verification_service.ingest_observation(req)
        if not res.success:
            raise HTTPException(status_code=422, detail=res.message)
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Observation ingestion failed: {str(e)}")


@router.post("/trigger-matching")
def trigger_matching_endpoint(
    location_id: Optional[str] = Query(None, description="Optional coastal node ID"),
    tolerance_seconds: int = Query(1800, ge=60, le=7200, description="Matching window in seconds (+/- tolerance)"),
):
    """
    Manually triggers matching between pending forecasts and stored observations.
    """
    matched = forecast_verification_service.match_and_verify(
        location_id=location_id,
        tolerance_seconds=tolerance_seconds,
    )
    return {
        "success": True,
        "matched_count": matched,
        "message": f"Processed verification queue. {matched} new forecast-observation pairs matched.",
    }


@router.post("/seed-demo-observations")
def seed_demo_observations_endpoint(
    count: int = Query(24, ge=5, le=100, description="Number of demo observations to ingest"),
):
    """
    Utility endpoint for test and demonstration mode:
    Ingests realistic prospective observation anchors and triggers matching to demonstrate
    real-time error distributions and verified history in development environments.
    """
    import numpy as np
    now = datetime.now(timezone.utc)
    ingested = 0

    # Ingest a sequence of hourly observations over the past 24-48 hours
    for i in range(count, 0, -1):
        obs_time = now - (i * __import__("datetime").timedelta(hours=1))
        # Realistic wave height curve around 1.35m with diurnal oscillation
        wave_height = round(1.30 + 0.35 * np.sin(i * 0.25) + 0.05 * np.cos(i * 0.5), 3)
        obs_req = ObservationIngestRequest(
            location_id="mumbai_offshore",
            observation_time=obs_time,
            observed_swh=float(wave_height),
            station_id="INCOIS_MUMBAI_01",
            source_name="INCOIS_MOORED_BUOY_NETWORK",
            source_type="in_situ_buoy",
            quality_status="valid",
            notes="Demonstration in-situ moored buoy telemetry",
        )
        res = forecast_verification_service.ingest_observation(obs_req)
        if res.success:
            ingested += 1

    return {
        "success": True,
        "observations_ingested": ingested,
        "message": f"Ingested {ingested} demonstration marine observations and matched pending forecasts.",
    }

"""
Monitoring and Model Drift REST API endpoints.
Provides read-only access to the forecast ledger, accuracy metrics,
data drift reports, and persistent operational monitoring events.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.intelligence.monitoring import (
    AccuracyMetrics,
    DataDriftReport,
    DriftStatus,
    MonitoringEventType,
    MonitoringSeverity,
    ObservationQuality,
    ObservationSourceType,
    VerificationStatus,
    calculate_forecast_accuracy,
    get_drift_manager,
    get_monitoring_store,
)
from src.intelligence.ingestion import get_ingestion_service

router = APIRouter(prefix="/api/monitoring", tags=["Monitoring & Model Drift"])


class MonitoringSummaryResponse(BaseModel):
    total_forecasts_issued: int
    total_observations_recorded: int
    active_monitoring_events_count: int
    independent_buoy_available: bool
    live_verification_status: str
    feature_drift_status: str
    evaluation_time_utc: str
    notices: List[str]


@router.get("/summary", response_model=MonitoringSummaryResponse)
def get_monitoring_overview():
    """High-level summary of forecast issuance volume, observation counts, and ML ops health."""
    store = get_monitoring_store()
    total_fc = store.count_forecasts()
    total_obs = store.count_observations()
    active_events = store.list_monitoring_events(status="ACTIVE")

    # In current deployment, live in-situ moored buoy APIs are not physically connected
    return MonitoringSummaryResponse(
        total_forecasts_issued=total_fc,
        total_observations_recorded=total_obs,
        active_monitoring_events_count=len(active_events),
        independent_buoy_available=False,
        live_verification_status=VerificationStatus.VERIFICATION_UNAVAILABLE.value,
        feature_drift_status=DriftStatus.NORMAL.value,
        evaluation_time_utc=datetime.now(timezone.utc).isoformat(),
        notices=[
            "Independent live in-situ moored buoy feeds are currently not configured.",
            "Historical ERA5 test split observations support retrospective backtesting.",
            "All candidate monitoring nodes outside Mumbai Offshore remain experimental.",
        ],
    )


@router.get("/accuracy")
def get_forecast_accuracy(
    location_id: Optional[str] = Query(None, description="Registered coastal node ID"),
    horizon: Optional[int] = Query(None, description="Forecast horizon: 1, 6, or 24"),
    model_name: Optional[str] = Query(None, description="'WaveLSTM', 'WaveANN', or 'WaveCNN1D'"),
    min_samples: int = Query(10, description="Minimum matched observations for statistical confidence"),
):
    """Retrieve verified or retrospective forecast accuracy metrics (MAE, RMSE, Bias, Skill Score)."""
    store = get_monitoring_store()
    matched = store.match_forecasts(
        location_id=location_id,
        horizon=horizon,
        model_name=model_name,
    )
    total_issued = store.count_forecasts(location_id=location_id)
    metrics = calculate_forecast_accuracy(
        matched_pairs=matched,
        total_issued=total_issued,
        min_sample_size=min_samples,
        location_id=location_id,
        horizon=horizon,
        model_name=model_name,
    )
    return metrics.to_dict()


@router.get("/drift")
def get_data_drift_report(
    location_id: Optional[str] = Query("mumbai_offshore", description="Coastal node ID to evaluate"),
    source: str = Query("demo", description="'demo' (ERA5 window) or 'live' (Open-Meteo REST)"),
):
    """Calculate Population Stability Index (PSI) and data quality metrics for canonical input features."""
    ingestion = get_ingestion_service()
    fetch_dict = ingestion.fetch_all_locations(source=source, enabled_only=True)
    fetch_res = fetch_dict.get(location_id)

    if fetch_res is None or fetch_res.raw_df is None or fetch_res.raw_df.empty:
        raise HTTPException(
            status_code=503,
            detail=f"Telemetry feed for node '{location_id}' is currently unavailable or empty.",
        )

    drift_mgr = get_drift_manager()
    report = drift_mgr.calculate_psi(fetch_res.raw_df)
    return report.to_dict()


@router.get("/events")
def list_monitoring_events(
    location_id: Optional[str] = Query(None, description="Filter by coastal location ID"),
    status: Optional[str] = Query("ACTIVE", description="'ACTIVE', 'ACKNOWLEDGED', 'RESOLVED', or 'ALL'"),
    event_type: Optional[str] = Query(None, description="'data_drift', 'performance_degradation', etc."),
    limit: int = Query(50, description="Maximum event records to return"),
):
    """List persistent ML operations events for data drift, quality issues, or performance drops."""
    store = get_monitoring_store()
    status_filter = None if status == "ALL" else status
    events = store.list_monitoring_events(
        location_id=location_id,
        status=status_filter,
        event_type=event_type,
        limit=limit,
    )
    return [e.to_dict() for e in events]


@router.post("/backtest")
def trigger_retrospective_backtest(
    location_id: Optional[str] = Query("mumbai_offshore", description="Location ID to populate ERA5 test observations for"),
    max_rows: int = Query(500, description="Maximum observations to ingest"),
):
    """Populate historical ERA5 test split observations for retrospective accuracy verification."""
    from src.intelligence.monitoring import populate_retrospective_reanalysis_observations
    store = get_monitoring_store()
    target_loc = "mumbai_offshore" if location_id in ["ALL", "all", "", None] else location_id
    inserted = populate_retrospective_reanalysis_observations(store, location_id=target_loc, max_rows=max_rows)
    return {
        "success": True,
        "location_id": target_loc,
        "inserted_observations": inserted,
        "message": f"Successfully ingested {inserted} retrospective ERA5 reanalysis observations for verification backtesting.",
    }


@router.get("/ledger")
def list_forecast_ledger(
    location_id: Optional[str] = Query(None, description="Filter by location ID"),
    limit: int = Query(50, description="Max ledger records to return"),
):
    """Query recent prospective records recorded in the immutable forecast ledger."""
    store = get_monitoring_store()
    filter_loc = None if location_id in ["ALL", "all", "", None] else location_id
    records = store.query_forecasts(location_id=filter_loc, limit=limit)
    return [r.to_dict() for r in records]

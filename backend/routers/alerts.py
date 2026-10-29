"""
API router for coastal risk alerts, notifications, lifecycle management, and thresholds.
"""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from backend.schemas.alerts import (
    AlertAcknowledgeRequest,
    AlertHistoryItem,
    AlertItem,
    AlertSummaryResponse,
    ThresholdConfigResponse,
)
from backend.services.model_service import model_service
from src.intelligence.alerts import AlertStatus, get_alert_engine, get_alert_store
from src.intelligence.ingestion import get_ingestion_service
from src.intelligence.nodes import get_coastal_registry
from src.intelligence.risk import assess_node_risk, get_threshold_manager

router = APIRouter(prefix="/api/alerts", tags=["Coastal Alerts"])


@router.get("", response_model=List[AlertItem])
def list_alerts(
    status: Optional[str] = Query(None, description="'ACTIVE', 'ACKNOWLEDGED', 'RESOLVED', 'EXPIRED', 'ACTIVE_ACKNOWLEDGED', or 'ALL'"),
    severity: Optional[str] = Query(None, description="'RED', 'ORANGE', 'YELLOW', or 'ALL'"),
    location_id: Optional[str] = Query(None, description="Registered coastal node ID"),
    limit: int = Query(100, description="Maximum alerts to retrieve"),
):
    """Retrieve coastal risk alerts filtered by lifecycle status, severity, and location."""
    store = get_alert_store()
    filter_loc = None if location_id in ["ALL", "all", "", None] else location_id
    filter_sev = None if severity in ["ALL", "all", "", None] else severity

    if status == "ACTIVE_ACKNOWLEDGED":
        pool = store.list_alerts(location_id=filter_loc, severity=filter_sev, limit=limit)
        alerts = [a for a in pool if a.is_active_or_acknowledged]
    elif status in ["ALL", "all", None]:
        alerts = store.list_alerts(location_id=filter_loc, severity=filter_sev, limit=limit)
    else:
        try:
            status_enum = AlertStatus(status.upper())
            alerts = store.list_alerts(status=status_enum, location_id=filter_loc, severity=filter_sev, limit=limit)
        except ValueError:
            alerts = store.list_alerts(location_id=filter_loc, severity=filter_sev, limit=limit)

    return [a.to_dict() for a in alerts]


@router.get("/summary", response_model=AlertSummaryResponse)
def get_alert_summary():
    """Retrieve statistical summary of active, severity-partitioned, and resolved alerts."""
    store = get_alert_store()
    sev_counts = store.count_active_by_severity()
    status_counts = store.count_alerts_by_status()
    total_active = sev_counts.get("RED", 0) + sev_counts.get("ORANGE", 0) + sev_counts.get("YELLOW", 0)

    return AlertSummaryResponse(
        total_active=total_active,
        critical_red=sev_counts.get("RED", 0),
        high_orange=sev_counts.get("ORANGE", 0),
        caution_yellow=sev_counts.get("YELLOW", 0),
        resolved_count=status_counts.get("RESOLVED", 0),
        status_counts=status_counts,
        last_assessment_utc=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/thresholds", response_model=ThresholdConfigResponse)
def get_threshold_configuration(
    location_id: Optional[str] = Query(None, description="Optional coastal node ID for location-specific overrides")
):
    """Retrieve current shared risk thresholds, hysteresis bounds, and scientific notice."""
    mgr = get_threshold_manager()
    t_set = mgr.get_thresholds(location_id)
    return ThresholdConfigResponse(**t_set.to_dict())


@router.post("/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, req: Optional[AlertAcknowledgeRequest] = None):
    """Transition an ACTIVE alert to ACKNOWLEDGED state with audit logging."""
    engine = get_alert_engine()
    notes = req.operator_notes if req else "Acknowledged by operator"
    success = engine.acknowledge_alert(alert_id=alert_id, operator_notes=notes)

    if not success:
        # Check if alert exists or was already acknowledged/resolved
        store = get_alert_store()
        rec = store.get_by_id(alert_id)
        if not rec:
            raise HTTPException(status_code=404, detail=f"Alert with ID '{alert_id}' was not found.")
        return {
            "success": False,
            "status": rec.status.value,
            "message": f"Alert is already in status '{rec.status.value}' and cannot be acknowledged.",
        }

    return {
        "success": True,
        "alert_id": alert_id,
        "status": "ACKNOWLEDGED",
        "acknowledged_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/{alert_id}/history", response_model=List[AlertHistoryItem])
def get_alert_transition_history(alert_id: str):
    """Retrieve chronological state transition log for a specific hazard alert."""
    store = get_alert_store()
    history = store.get_history(alert_id)
    return history


@router.post("/evaluate")
def trigger_alert_evaluation(
    source: str = Query("demo", description="'demo' (ERA5 replay) or 'live' (Open-Meteo REST)")
):
    """Trigger on-demand multi-location coastal telemetry ingestion and risk evaluation."""
    ingestion_svc = get_ingestion_service()
    coastal_reg = get_coastal_registry()
    thresh_mgr = get_threshold_manager()
    alert_engine = get_alert_engine()

    try:
        scaler_obj = model_service.get_scaler()
    except Exception:
        scaler_obj = None

    source_clean = "demo" if source.lower() == "demo" else "live"
    fetch_dict = ingestion_svc.fetch_all_locations(source=source_clean, enabled_only=True, force_refresh=True)

    assessments_to_run = []
    for lid, f_res in fetch_dict.items():
        try:
            node = coastal_reg.get(lid)
            t_set = thresh_mgr.get_thresholds(lid)
            ass_list = assess_node_risk(
                node=node,
                fetch_result=f_res,
                scaler=scaler_obj,
                thresholds=t_set,
                horizons=[1, 6, 24],
            )
            assessments_to_run.extend(ass_list)
        except Exception:
            continue

    processed_alerts = alert_engine.process_all_assessments(assessments_to_run)
    return {
        "evaluated_nodes_count": len(fetch_dict),
        "assessments_generated": len(assessments_to_run),
        "active_alerts_count": len(processed_alerts),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_clean,
    }

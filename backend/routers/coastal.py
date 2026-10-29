"""
API router for coastal risk heatmap, registered node telemetry, and geographic metrics.
"""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Query

from backend.schemas.alerts import ThresholdConfigResponse
from backend.schemas.coastal import CoastalMetricsResponse, NodeMapSummaryItem
from src.intelligence.mapping import compute_map_dashboard_metrics, prepare_coastal_node_map_data
from src.intelligence.nodes import get_coastal_registry
from src.intelligence.risk import get_threshold_manager

router = APIRouter(prefix="/api/coastal", tags=["Coastal Risk & Heatmap"])


@router.get("/locations")
def get_coastal_locations():
    """Retrieve all registered coastal monitoring candidate nodes and their status."""
    registry = get_coastal_registry()
    return [node.to_dict() for node in registry.list_all()]


@router.get("/map-data", response_model=List[NodeMapSummaryItem])
def get_coastal_map_data(
    horizon: int = Query(6, description="Lead time in hours: 1, 6, or 24"),
    source: str = Query("demo", description="'demo' (ERA5 window) or 'live' (Open-Meteo REST)"),
    model: str = Query("lstm", description="'lstm', 'ann', or 'cnn1d'"),
    force_refresh: bool = Query(False, description="Bypass telemetry cache"),
):
    """
    Retrieve real-time geographic data for all registered coastal candidate nodes.
    Includes coordinates, risk category, marker color, observed wave height,
    multi-horizon forecasts, data freshness, active alerts, and validation status.
    """
    summaries = prepare_coastal_node_map_data(
        source=source,
        selected_horizon=horizon,
        force_refresh=force_refresh,
    )

    items = []
    for s in summaries:
        items.append(
            NodeMapSummaryItem(
                location_id=s.location_id,
                name=s.name,
                latitude=s.latitude,
                longitude=s.longitude,
                coordinate_label=s.coordinate_label,
                region=s.region,
                is_reference=s.is_reference,
                validation_status=s.validation_status.value if hasattr(s.validation_status, "value") else str(s.validation_status),
                data_status=s.data_status.value if hasattr(s.data_status, "value") else str(s.data_status),
                data_source=s.data_source,
                retrieval_time_utc=s.retrieval_time_utc.isoformat() if s.retrieval_time_utc else None,
                forecast_issue_time_utc=s.forecast_issue_time_utc.isoformat() if s.forecast_issue_time_utc else None,
                latest_obs_swh_m=s.latest_obs_swh_m,
                latest_obs_time_utc=s.latest_obs_time_utc.isoformat() if s.latest_obs_time_utc else None,
                forecast_swh_m=s.forecast_swh_m,
                forecast_horizon_hours=s.forecast_horizon_hours,
                forecast_valid_time_utc=s.forecast_valid_time_utc.isoformat() if s.forecast_valid_time_utc else None,
                max_window_swh_m=s.max_window_swh_m,
                current_risk_level=s.current_risk_level.value if hasattr(s.current_risk_level, "value") else str(s.current_risk_level),
                highest_risk_level=s.highest_risk_level.value if hasattr(s.highest_risk_level, "value") else str(s.highest_risk_level),
                marker_hex_color=s.marker_hex_color,
                threshold_crossed_m=s.threshold_crossed_m,
                active_alert_count=s.active_alert_count,
                active_alerts=[a.to_dict() for a in s.active_alerts],
                is_degraded=s.is_degraded,
                trigger_explanation=s.trigger_explanation,
                limitations=s.limitations,
                multi_horizon_predictions=s.multi_horizon_predictions,
                external_nwp_forecasts=s.external_nwp_forecasts,
            )
        )
    return items


@router.get("/metrics", response_model=CoastalMetricsResponse)
def get_coastal_dashboard_metrics(
    horizon: int = Query(6, description="Lead time in hours: 1, 6, or 24"),
    source: str = Query("demo", description="'demo' (ERA5 window) or 'live' (Open-Meteo REST)"),
):
    """Retrieve high-level geographic summary KPIs across all coastal monitoring nodes."""
    summaries = prepare_coastal_node_map_data(source=source, selected_horizon=horizon)
    m = compute_map_dashboard_metrics(summaries)
    return CoastalMetricsResponse(
        total_nodes=m.get("total_nodes", len(summaries)),
        reference_nodes=m.get("reference_nodes", 1),
        active_alerts=m.get("active_alerts", 0),
        high_risk_nodes=m.get("high_risk_nodes", 0),
        degraded_feeds=m.get("degraded_feeds", 0),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/thresholds", response_model=ThresholdConfigResponse)
def get_shared_coastal_thresholds():
    """Retrieve global shared wave hazard threshold configuration."""
    t_set = get_threshold_manager().default_thresholds
    return ThresholdConfigResponse(**t_set.to_dict())

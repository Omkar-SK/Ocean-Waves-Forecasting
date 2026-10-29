"""
Coastal Mapping and Geospatial Risk Visualization Subsystem for Ocean Intelligence.

Prepares structured map data, markers, tooltips, and popups for candidate coastal
monitoring nodes along the Indian coastline.

IMPORTANT NOTICES:
1. POINT-BASED VISUALIZATION: Displays candidate offshore numerical forecast grid points
   (~5 to 25 km offshore). Does NOT interpolate or fabricate continuous spatial wave fields
   between sparse observation nodes.
2. RISK CONSISTENCY: Uses the exact same RiskAssessment and AlertStore subsystems as the
   Alert Centre to guarantee complete agreement between map markers and alert records.
3. DEGRADED / UNAVAILABLE DATA: Nodes with failed, stale, or unavailable telemetry are
   rendered in distinct neutral grey and are NEVER mislabeled as safe GREEN.
4. GEOGRAPHIC CALIBRATION: Clearly distinguishes Mumbai Offshore (Reference Calibrated)
   from candidate regional transfer nodes (Experimental Unvalidated).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.data.dataset import TimeSeriesScaler
from src.intelligence.alerts import (
    AlertRecord,
    AlertStatus,
    AlertStore,
    get_alert_store,
)
from src.intelligence.nodes import (
    CoastalNode,
    CoastalNodeRegistry,
    ValidationStatus,
    get_coastal_registry,
)
from src.intelligence.provider import DataStatus, ProviderFetchResult
from src.intelligence.ingestion import (
    MultiLocationIngestionService,
    get_ingestion_service,
)
from src.intelligence.risk import (
    EventType,
    RiskAssessment,
    RiskLevel,
    RiskThresholdSet,
    ThresholdConfigManager,
    assess_node_risk,
    get_threshold_manager,
)

logger = logging.getLogger("OceanIntelligence.Mapping")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class NodeMapSummary:
    """
    Standardized geographic data payload for a single coastal monitoring node.
    """
    location_id: str
    name: str
    latitude: float
    longitude: float
    coordinate_label: str
    region: str
    is_reference: bool
    validation_status: ValidationStatus
    data_status: DataStatus
    data_source: str
    retrieval_time_utc: Optional[datetime]
    forecast_issue_time_utc: Optional[datetime]
    latest_obs_swh_m: Optional[float]
    latest_obs_time_utc: Optional[datetime]
    forecast_swh_m: Optional[float]
    forecast_horizon_hours: int
    forecast_valid_time_utc: Optional[datetime]
    max_window_swh_m: Optional[float]
    current_risk_level: RiskLevel
    highest_risk_level: RiskLevel
    threshold_crossed_m: Optional[float]
    active_alert_count: int
    active_alerts: List[AlertRecord] = field(default_factory=list)
    is_degraded: bool = False
    trigger_explanation: str = ""
    limitations: List[str] = field(default_factory=list)
    multi_horizon_predictions: Dict[int, float] = field(default_factory=dict)
    external_nwp_forecasts: Dict[int, float] = field(default_factory=dict)

    @property
    def marker_hex_color(self) -> str:
        if self.is_degraded or self.data_status in [DataStatus.UNAVAILABLE, DataStatus.INVALID]:
            return "#64748b"  # Slate Grey
        elif self.data_status == DataStatus.STALE:
            return "#94a3b8"  # Light Muted Grey
        return self.current_risk_level.hex_color

    @property
    def marker_folium_color(self) -> str:
        """Map color compatible with Folium icon styles."""
        if self.is_degraded or self.data_status in [DataStatus.UNAVAILABLE, DataStatus.INVALID]:
            return "gray"
        elif self.data_status == DataStatus.STALE:
            return "lightgray"
        colors = {
            RiskLevel.RED: "red",
            RiskLevel.ORANGE: "orange",
            RiskLevel.YELLOW: "beige",
            RiskLevel.GREEN: "green",
        }
        return colors.get(self.current_risk_level, "blue")

    @property
    def marker_icon_name(self) -> str:
        if self.is_degraded or self.data_status in [DataStatus.UNAVAILABLE, DataStatus.INVALID]:
            return "question-sign"
        elif self.data_status == DataStatus.STALE:
            return "time"
        elif self.current_risk_level == RiskLevel.RED:
            return "exclamation-sign"
        elif self.current_risk_level == RiskLevel.ORANGE:
            return "warning-sign"
        elif self.current_risk_level == RiskLevel.YELLOW:
            return "bell"
        return "ok-sign"

    @property
    def status_label(self) -> str:
        if self.is_degraded or self.data_status in [DataStatus.UNAVAILABLE, DataStatus.INVALID]:
            return "UNAVAILABLE / DEGRADED"
        elif self.data_status == DataStatus.STALE:
            return "STALE TELEMETRY"
        return self.current_risk_level.display_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location_id": self.location_id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "coordinate_label": self.coordinate_label,
            "region": self.region,
            "is_reference": self.is_reference,
            "validation_status": self.validation_status.value,
            "data_status": self.data_status.value,
            "data_source": self.data_source,
            "retrieval_time_utc": self.retrieval_time_utc.isoformat() if self.retrieval_time_utc else None,
            "latest_obs_swh_m": round(self.latest_obs_swh_m, 2) if self.latest_obs_swh_m is not None else None,
            "forecast_swh_m": round(self.forecast_swh_m, 2) if self.forecast_swh_m is not None else None,
            "forecast_horizon_hours": self.forecast_horizon_hours,
            "max_window_swh_m": round(self.max_window_swh_m, 2) if self.max_window_swh_m is not None else None,
            "current_risk_level": self.current_risk_level.value,
            "highest_risk_level": self.highest_risk_level.value,
            "threshold_crossed_m": self.threshold_crossed_m,
            "active_alert_count": self.active_alert_count,
            "is_degraded": self.is_degraded,
            "marker_hex_color": self.marker_hex_color,
            "status_label": self.status_label,
        }


def prepare_coastal_node_map_data(
    selected_horizon: int = 6,
    source: str = "demo",
    force_refresh: bool = False,
    registry: Optional[CoastalNodeRegistry] = None,
    ingestion_svc: Optional[MultiLocationIngestionService] = None,
    alert_store: Optional[AlertStore] = None,
    thresh_mgr: Optional[ThresholdConfigManager] = None,
    scaler: Optional[TimeSeriesScaler] = None,
) -> List[NodeMapSummary]:
    """
    Build rich geospatial risk summary payloads for all enabled coastal nodes.

    Guarantees:
    - Queries the existing IngestionService, respecting its in-memory TTL caching.
    - Uses existing assess_node_risk logic, ensuring zero discrepancy with the Alert Centre.
    - Evaluates the selected horizon specifically while capturing peak window wave height.
    - Correlates with persistent active alert records from AlertStore.
    - Strictly labels unvalidated regional nodes and degraded telemetry states.

    Parameters
    ----------
    selected_horizon : int
        Target forecast lead time (1, 6, or 24 hours).
    source : str
        'demo' (offline ERA5 ground-truth) or 'live' (Open-Meteo REST).
    force_refresh : bool
        If True, bypasses cache and queries external provider.
    registry : Optional[CoastalNodeRegistry]
        Node registry (default: global coastal registry).
    ingestion_svc : Optional[MultiLocationIngestionService]
        Ingestion service instance.
    alert_store : Optional[AlertStore]
        Alert database store.
    thresh_mgr : Optional[ThresholdConfigManager]
        Threshold configuration manager.
    scaler : Optional[TimeSeriesScaler]
        Fitted training scaler.

    Returns
    -------
    List[NodeMapSummary]
        List of structured node summary objects for map rendering.
    """
    reg = registry or get_coastal_registry()
    ingestion = ingestion_svc or get_ingestion_service()
    store = alert_store or get_alert_store()
    mgr = thresh_mgr or get_threshold_manager()

    if scaler is None:
        scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
        if scaler_path.is_file():
            scaler = TimeSeriesScaler.load(scaler_path)

    # Ingest data for all enabled nodes with bounded concurrency and error isolation
    fetch_dict = ingestion.fetch_all_locations(
        source=source,
        enabled_only=True,
        force_refresh=force_refresh,
    )

    summaries: List[NodeMapSummary] = []
    nodes = reg.list_all(enabled_only=True)

    for node in nodes:
        f_res = fetch_dict.get(node.location_id)
        active_thresholds = mgr.get_thresholds(node.location_id)

        # Retrieve active alerts from database for this location
        loc_alerts_all = store.list_alerts(location_id=node.location_id, limit=50)
        active_alerts = [a for a in loc_alerts_all if a.is_active_or_acknowledged]
        alert_cnt = len(active_alerts)

        # Case 1: Ingestion failure or missing data
        if f_res is None or not f_res.is_success or f_res.raw_df is None or f_res.raw_df.empty:
            data_st = f_res.data_status if f_res else DataStatus.UNAVAILABLE
            err_msg = f_res.error_message if f_res and f_res.error_message else "Telemetry unavailable."
            summaries.append(
                NodeMapSummary(
                    location_id=node.location_id,
                    name=node.name,
                    latitude=node.latitude,
                    longitude=node.longitude,
                    coordinate_label=node.coordinate_label,
                    region=node.region,
                    is_reference=node.is_reference,
                    validation_status=node.validation_status,
                    data_status=data_st,
                    data_source="Unavailable" if f_res is None else f_res.provider_name,
                    retrieval_time_utc=f_res.retrieval_time_utc if f_res else None,
                    forecast_issue_time_utc=f_res.forecast_issue_time_utc if f_res else None,
                    latest_obs_swh_m=None,
                    latest_obs_time_utc=None,
                    forecast_swh_m=None,
                    forecast_horizon_hours=selected_horizon,
                    forecast_valid_time_utc=None,
                    max_window_swh_m=None,
                    current_risk_level=RiskLevel.GREEN,  # Tagged with is_degraded so not displayed as safe green!
                    highest_risk_level=RiskLevel.GREEN,
                    threshold_crossed_m=None,
                    active_alert_count=alert_cnt,
                    active_alerts=active_alerts,
                    is_degraded=True,
                    trigger_explanation=f"Telemetry feed offline or unreachable: {err_msg}",
                    limitations=[
                        "TELEMETRY UNAVAILABLE: Provider query failed.",
                        "Cannot verify ocean conditions; neutral grey indicates absence of data, NOT guaranteed safety.",
                        active_thresholds.guidance_notice,
                    ],
                )
            )
            continue

        # Case 2: Ingestion succeeded -> run risk assessment across horizons [1, 6, 24]
        try:
            assessments = assess_node_risk(
                node=node,
                fetch_result=f_res,
                scaler=scaler,
                thresholds=active_thresholds,
                horizons=[1, 6, 24],
                model_type="lstm",
            )
        except Exception as assess_err:
            logger.error("Error during risk assessment for %s: %s", node.location_id, assess_err)
            summaries.append(
                NodeMapSummary(
                    location_id=node.location_id,
                    name=node.name,
                    latitude=node.latitude,
                    longitude=node.longitude,
                    coordinate_label=node.coordinate_label,
                    region=node.region,
                    is_reference=node.is_reference,
                    validation_status=node.validation_status,
                    data_status=DataStatus.INVALID,
                    data_source=f_res.provider_name,
                    retrieval_time_utc=f_res.retrieval_time_utc,
                    forecast_issue_time_utc=f_res.forecast_issue_time_utc,
                    latest_obs_swh_m=None,
                    latest_obs_time_utc=None,
                    forecast_swh_m=None,
                    forecast_horizon_hours=selected_horizon,
                    forecast_valid_time_utc=None,
                    max_window_swh_m=None,
                    current_risk_level=RiskLevel.GREEN,
                    highest_risk_level=RiskLevel.GREEN,
                    threshold_crossed_m=None,
                    active_alert_count=alert_cnt,
                    active_alerts=active_alerts,
                    is_degraded=True,
                    trigger_explanation=f"Assessment error: {assess_err}",
                    limitations=[f"Assessment processing failed: {assess_err}"],
                )
            )
            continue

        # Extract current observation (horizon = 0)
        obs_assessment = next((a for a in assessments if a.forecast_horizon_hours == 0), None)
        latest_obs_val = obs_assessment.predicted_swh_m if obs_assessment else None
        latest_obs_dt = obs_assessment.forecast_valid_time_utc if obs_assessment else None

        # Extract multi-horizon predictions
        pred_dict: Dict[int, float] = {}
        for a in assessments:
            if a.forecast_horizon_hours > 0:
                pred_dict[a.forecast_horizon_hours] = a.predicted_swh_m

        # Selected horizon assessment
        sel_assessment = next((a for a in assessments if a.forecast_horizon_hours == selected_horizon), None)
        if sel_assessment is None and assessments:
            sel_assessment = assessments[-1]

        target_pred_val = sel_assessment.predicted_swh_m if sel_assessment else latest_obs_val
        target_valid_dt = sel_assessment.forecast_valid_time_utc if sel_assessment else latest_obs_dt
        current_risk = sel_assessment.risk_level if sel_assessment else RiskLevel.GREEN
        thresh_crossed = sel_assessment.threshold_crossed_m if sel_assessment else None

        # Find maximum wave height and highest risk across the entire 24h forecast window
        all_window_preds = list(pred_dict.values())
        if latest_obs_val is not None:
            all_window_preds.append(latest_obs_val)

        max_window_val = max(all_window_preds) if all_window_preds else target_pred_val
        highest_risk = max(
            [a.risk_level for a in assessments],
            key=lambda r: r.severity_rank,
            default=current_risk,
        )

        is_node_degraded = f_res.data_status in [DataStatus.STALE, DataStatus.INVALID]

        summaries.append(
            NodeMapSummary(
                location_id=node.location_id,
                name=node.name,
                latitude=node.latitude,
                longitude=node.longitude,
                coordinate_label=node.coordinate_label,
                region=node.region,
                is_reference=node.is_reference,
                validation_status=node.validation_status,
                data_status=f_res.data_status,
                data_source=f_res.provider_name,
                retrieval_time_utc=f_res.retrieval_time_utc,
                forecast_issue_time_utc=f_res.forecast_issue_time_utc,
                latest_obs_swh_m=latest_obs_val,
                latest_obs_time_utc=latest_obs_dt,
                forecast_swh_m=target_pred_val,
                forecast_horizon_hours=selected_horizon,
                forecast_valid_time_utc=target_valid_dt,
                max_window_swh_m=max_window_val,
                current_risk_level=current_risk,
                highest_risk_level=highest_risk,
                threshold_crossed_m=thresh_crossed,
                active_alert_count=alert_cnt,
                active_alerts=active_alerts,
                is_degraded=is_node_degraded,
                trigger_explanation=sel_assessment.trigger_explanation if sel_assessment else "",
                limitations=sel_assessment.limitations_and_uncertainty if sel_assessment else [],
                multi_horizon_predictions=pred_dict,
                external_nwp_forecasts=dict(f_res.external_nwp_forecasts),
            )
        )

    return summaries


def compute_map_dashboard_metrics(node_summaries: List[NodeMapSummary]) -> Dict[str, Any]:
    """
    Compute structured summary metrics for the top of the map dashboard.

    Guarantees:
    - Never counts unknown or degraded nodes as safe/low risk.
    - Explicitly separates fresh data count from stale or unavailable feeds.
    - Calculates active alert totals matching the Alert Store.
    """
    total_locations = len(node_summaries)
    if total_locations == 0:
        return {
            "total_locations": 0,
            "fresh_locations": 0,
            "stale_or_unavailable_locations": 0,
            "elevated_risk_locations": 0,
            "active_alert_locations": 0,
            "total_active_alerts": 0,
            "max_observed_or_forecast_swh_m": 0.0,
        }

    fresh_cnt = sum(1 for s in node_summaries if s.data_status == DataStatus.FRESH)
    degraded_cnt = sum(
        1
        for s in node_summaries
        if s.is_degraded or s.data_status in [DataStatus.STALE, DataStatus.UNAVAILABLE, DataStatus.INVALID]
    )

    elevated_cnt = sum(
        1
        for s in node_summaries
        if not s.is_degraded and s.current_risk_level in [RiskLevel.YELLOW, RiskLevel.ORANGE, RiskLevel.RED]
    )

    active_alert_locs = sum(1 for s in node_summaries if s.active_alert_count > 0)
    total_active_alerts = sum(s.active_alert_count for s in node_summaries)

    valid_heights = [
        s.forecast_swh_m
        for s in node_summaries
        if s.forecast_swh_m is not None and not s.is_degraded
    ]
    max_swh = max(valid_heights) if valid_heights else 0.0

    return {
        "total_locations": total_locations,
        "fresh_locations": fresh_cnt,
        "stale_or_unavailable_locations": degraded_cnt,
        "elevated_risk_locations": elevated_cnt,
        "active_alert_locations": active_alert_locs,
        "total_active_alerts": total_active_alerts,
        "max_observed_or_forecast_swh_m": round(max_swh, 2),
    }


def build_coastal_folium_map(
    node_summaries: List[NodeMapSummary],
    selected_node_id: Optional[str] = None,
    height: int = 550,
    thresholds: Optional[RiskThresholdSet] = None,
) -> Any:
    """
    Construct an interactive Folium map representing coastal monitoring points.

    Guarantees:
    - Dark ocean basemap (CartoDB Dark_Matter) matching dashboard theme.
    - Fits initial view bounds to all candidate coordinates.
    - Color-coded markers matching project risk categories (RED, ORANGE, YELLOW, GREEN, GREY).
    - Clear distinction between Reference Calibrated and Experimental Regional nodes.
    - Rich HTML popup with complete telemetry details and active alerts.
    - Integrated floating legend dynamically driven by active risk thresholds.
    """
    try:
        import folium
        from folium.plugins import Fullscreen
    except ImportError:
        logger.warning("Folium is not installed in the Python environment.")
        return None

    active_t = thresholds or get_threshold_manager().default_thresholds

    # Center around peninsular India
    m = folium.Map(
        location=[15.0, 78.5],
        zoom_start=5,
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        control_scale=True,
    )

    all_lats: List[float] = []
    all_lons: List[float] = []

    for s in node_summaries:
        all_lats.append(s.latitude)
        all_lons.append(s.longitude)

        is_selected = (selected_node_id == s.location_id)
        border_weight = 4 if is_selected else 2
        radius = 12 if is_selected else 9

        # Validation badge for popup
        if s.is_reference:
            val_html = "<span style='background:#065f46;color:#a7f3d0;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;'>REFERENCE CALIBRATED</span>"
        else:
            val_html = "<span style='background:#78350f;color:#fde68a;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;'>REGIONAL TRANSFER</span>"

        # Wave height text
        if s.is_degraded or s.forecast_swh_m is None:
            swh_str = "N/A"
            swh_color = "#94a3b8"
        else:
            swh_str = f"{s.forecast_swh_m:.2f} m"
            swh_color = s.marker_hex_color

        # Format popup HTML
        popup_html = f"""
        <div style="font-family: -apple-system, sans-serif; min-width: 230px; color: #1e293b; line-height: 1.4;">
            <div style="border-bottom: 2px solid {s.marker_hex_color}; padding-bottom: 6px; margin-bottom: 8px;">
                <h4 style="margin: 0; font-size: 15px; color: #0f172a;">{html.escape(s.name)}</h4>
                <div style="color: #64748b; font-size: 11px; margin-top: 2px;">{s.coordinate_label} | {html.escape(s.region)}</div>
                <div style="margin-top: 4px;">{val_html}</div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; background: #f8fafc; padding: 6px 8px; border-radius: 4px; margin-bottom: 8px;">
                <div>
                    <span style="font-size: 10px; color: #64748b; text-transform: uppercase;">Lead Time:</span><br>
                    <b style="font-size: 13px; color: #0284c7;">+{s.forecast_horizon_hours}h Horizon</b>
                </div>
                <div>
                    <span style="font-size: 10px; color: #64748b; text-transform: uppercase;">Predicted Wave:</span><br>
                    <b style="font-size: 15px; color: {swh_color};">{swh_str}</b>
                </div>
            </div>

            <div style="font-size: 11px; margin-bottom: 4px;">
                <b>Risk Level:</b> <span style="font-weight: 700; color: {s.marker_hex_color};">{s.status_label}</span>
            </div>
            <div style="font-size: 11px; margin-bottom: 4px;">
                <b>Telemetry Status:</b> <span style="font-weight: 600;">{s.data_status.value.upper()}</span>
            </div>
            <div style="font-size: 11px; margin-bottom: 6px;">
                <b>Active Alerts:</b> <span style="font-weight: 700; color: {'#ef4444' if s.active_alert_count > 0 else '#10b981'};">{s.active_alert_count} active</span>
            </div>
            <div style="font-size: 10px; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 4px;">
                <i>Click marker or use dropdown below to inspect deep learning trajectory.</i>
            </div>
        </div>
        """

        # Add outer pulsating halo if node is selected
        if is_selected:
            folium.CircleMarker(
                location=[s.latitude, s.longitude],
                radius=radius + 8,
                color="#38bdf8",
                weight=2,
                fill=True,
                fill_color="#38bdf8",
                fill_opacity=0.25,
            ).add_to(m)

        # Primary risk circle marker
        folium.CircleMarker(
            location=[s.latitude, s.longitude],
            radius=radius,
            color=s.marker_hex_color,
            weight=border_weight,
            fill=True,
            fill_color=s.marker_hex_color,
            fill_opacity=0.85,
            tooltip=f"{s.name}: {swh_str} ({s.status_label})",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(m)

    # Fit bounds if coordinates available
    if all_lats and all_lons:
        m.fit_bounds(
            [[min(all_lats) - 1.5, min(all_lons) - 2.0], [max(all_lats) + 1.5, max(all_lons) + 2.0]]
        )

    # Floating HTML Legend matching dashboard ocean palette and shared risk thresholds
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 25px; 
        right: 25px; 
        width: 220px;
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 14px;
        font-family: -apple-system, sans-serif;
        font-size: 11px;
        color: #f1f5f9;
        z-index: 9999;
        box-shadow: 0 4px 14px rgba(0,0,0,0.4);
    ">
        <div style="font-weight: 700; margin-bottom: 6px; color: #38bdf8; font-size: 12px; border-bottom: 1px solid #334155; padding-bottom: 4px;">
            🌊 Coastal Wave Risk Map
        </div>
        <div style="margin-bottom: 3px;"><span style="color: #ef4444; font-size: 14px;">●</span> <b>RED</b>: Critical (≥ {active_t.critical_m:.2f} m)</div>
        <div style="margin-bottom: 3px;"><span style="color: #f97316; font-size: 14px;">●</span> <b>ORANGE</b>: High (≥ {active_t.high_risk_m:.2f} m)</div>
        <div style="margin-bottom: 3px;"><span style="color: #f59e0b; font-size: 14px;">●</span> <b>YELLOW</b>: Caution (≥ {active_t.caution_m:.2f} m)</div>
        <div style="margin-bottom: 3px;"><span style="color: #10b981; font-size: 14px;">●</span> <b>GREEN</b>: Low Risk (< {active_t.caution_m:.2f} m)</div>
        <div style="margin-bottom: 5px;"><span style="color: #64748b; font-size: 14px;">●</span> <b>GREY</b>: Unavailable / Stale Data</div>
        <div style="color: #94a3b8; font-size: 10px; border-top: 1px solid #334155; padding-top: 4px;">
            ⚠️ <i>Config: {active_t.version}. Green is not a safety guarantee.</i>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

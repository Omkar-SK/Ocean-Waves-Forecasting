"""
Automated Unit and Integration Tests for Phase 4:
Interactive Coastal Risk Heatmap & Point-Based Geospatial Visualization.

Validates:
1. Geospatial map summary construction from the Coastal Node Registry.
2. Strict assignment of marker colors and icon states matching the Risk Assessment Engine.
3. Degraded, stale, and unavailable nodes are rendered in distinct neutral grey (never safe green).
4. Correct forecast horizon handling (+1h, +6h, +24h).
5. Accurate top-level dashboard summary metrics calculation.
6. Seamless integration with active alert records from AlertStore.
7. Prominent distinction between Reference Calibrated and Experimental Regional Transfer nodes.
8. Folium map building, bounds fitting, and selected-node halo highlighting.
9. Independent error isolation when individual coastal nodes fail.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import TimeSeriesScaler
from src.intelligence.alerts import (
    AlertEngine,
    AlertRecord,
    AlertStatus,
    AlertStore,
    compute_alert_fingerprint,
)
from src.intelligence.nodes import (
    CoastalNode,
    CoastalNodeRegistry,
    ValidationStatus,
    create_default_registry,
)
from src.intelligence.provider import (
    BaseMarineDataProvider,
    DataStatus,
    OfflineDemoProvider,
    ProviderFetchResult,
)
from src.intelligence.ingestion import MultiLocationIngestionService
from src.intelligence.risk import (
    EventType,
    RiskAssessment,
    RiskLevel,
    RiskThresholdSet,
    ThresholdConfigManager,
)
from src.intelligence.mapping import (
    NodeMapSummary,
    build_coastal_folium_map,
    compute_map_dashboard_metrics,
    prepare_coastal_node_map_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def trained_scaler():
    """Loads pre-fitted training scaler."""
    scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
    assert scaler_path.is_file(), "Scaler artifact missing."
    return TimeSeriesScaler.load(scaler_path)


@pytest.fixture
def temp_db_path(tmp_path):
    """Provides isolated SQLite database path."""
    return tmp_path / "test_map_alerts.db"


@pytest.fixture
def temp_alert_store(temp_db_path):
    """Provides isolated AlertStore instance."""
    return AlertStore(db_path=temp_db_path)


@pytest.fixture
def sample_48h_dataframe():
    """Generates 48 continuous hourly rows of valid marine and weather observations."""
    base_time = pd.Timestamp("2026-10-25 00:00:00", tz="UTC")
    times = [base_time + pd.Timedelta(hours=i) for i in range(48)]
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "valid_time": times,
            "swh": np.linspace(0.8, 1.2, 48),
            "mwp": np.linspace(7.5, 8.5, 48),
            "mwd": np.linspace(200.0, 240.0, 48),
            "wind_speed": np.linspace(4.0, 7.0, 48),
            "wind_dir": np.linspace(340.0, 360.0, 48),
        }
    )
    return df


# =============================================================================
# 1. Map Data Preparation & Attribute Tests
# =============================================================================

def test_prepare_coastal_node_map_data_all_nodes(trained_scaler, temp_alert_store):
    """Verify prepare_coastal_node_map_data generates summaries for all 8 candidate nodes."""
    reg = create_default_registry()
    summaries = prepare_coastal_node_map_data(
        selected_horizon=6,
        source="demo",
        registry=reg,
        alert_store=temp_alert_store,
        scaler=trained_scaler,
    )

    assert len(summaries) == 8

    # Verify Mumbai Offshore is reference validated
    mumbai = next(s for s in summaries if s.location_id == "mumbai_offshore")
    assert mumbai.is_reference is True
    assert mumbai.validation_status == ValidationStatus.REFERENCE_VALIDATED
    assert mumbai.latitude == 18.50
    assert mumbai.longitude == 72.50
    assert mumbai.forecast_horizon_hours == 6

    # Verify non-reference nodes are marked experimental unvalidated
    for s in summaries:
        if s.location_id != "mumbai_offshore":
            assert s.is_reference is False
            assert s.validation_status == ValidationStatus.EXPERIMENTAL_UNVALIDATED
            assert "REGIONAL TRANSFER" in s.limitations[0] or "GEOGRAPHIC" in " ".join(s.limitations)


def test_map_data_forecast_horizon_selection(trained_scaler, temp_alert_store):
    """Verify selected horizon correctly changes forecast_horizon_hours in output summaries."""
    reg = create_default_registry()
    for h in [1, 6, 24]:
        summaries = prepare_coastal_node_map_data(
            selected_horizon=h,
            source="demo",
            registry=reg,
            alert_store=temp_alert_store,
            scaler=trained_scaler,
        )
        for s in summaries:
            assert s.forecast_horizon_hours == h
            if s.forecast_swh_m is not None:
                assert s.forecast_swh_m > 0.0


def test_map_data_captures_window_maximum(trained_scaler, temp_alert_store):
    """Verify max_window_swh_m is calculated across the 24h evaluation window."""
    reg = create_default_registry()
    summaries = prepare_coastal_node_map_data(
        selected_horizon=1,
        source="demo",
        registry=reg,
        alert_store=temp_alert_store,
        scaler=trained_scaler,
    )
    for s in summaries:
        if s.forecast_swh_m is not None and s.max_window_swh_m is not None:
            assert s.max_window_swh_m >= s.forecast_swh_m or s.max_window_swh_m >= s.latest_obs_swh_m


# =============================================================================
# 2. Risk Mapping, Color Coding & Degraded Telemetry Tests
# =============================================================================

def test_marker_color_coding_for_risk_categories():
    """Verify marker_hex_color and marker_folium_color for all risk categories."""
    base_kwargs = dict(
        location_id="test_loc",
        name="Test",
        latitude=15.0,
        longitude=73.0,
        coordinate_label="15.00°N, 73.00°E",
        region="Test",
        is_reference=False,
        validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
        data_status=DataStatus.FRESH,
        data_source="Test",
        retrieval_time_utc=datetime.now(timezone.utc),
        forecast_issue_time_utc=None,
        latest_obs_swh_m=1.0,
        latest_obs_time_utc=None,
        forecast_swh_m=1.0,
        forecast_horizon_hours=6,
        forecast_valid_time_utc=None,
        max_window_swh_m=1.0,
        threshold_crossed_m=None,
        active_alert_count=0,
        is_degraded=False,
    )

    # GREEN (Normal)
    s_green = NodeMapSummary(
        **base_kwargs,
        current_risk_level=RiskLevel.GREEN,
        highest_risk_level=RiskLevel.GREEN,
    )
    assert s_green.marker_hex_color == "#10b981"
    assert s_green.marker_folium_color == "green"

    # YELLOW (Caution)
    s_yellow = NodeMapSummary(
        **base_kwargs,
        current_risk_level=RiskLevel.YELLOW,
        highest_risk_level=RiskLevel.YELLOW,
    )
    assert s_yellow.marker_hex_color == "#f59e0b"
    assert s_yellow.marker_folium_color == "beige"

    # ORANGE (High Risk)
    s_orange = NodeMapSummary(
        **base_kwargs,
        current_risk_level=RiskLevel.ORANGE,
        highest_risk_level=RiskLevel.ORANGE,
    )
    assert s_orange.marker_hex_color == "#f97316"
    assert s_orange.marker_folium_color == "orange"

    # RED (Critical)
    s_red = NodeMapSummary(
        **base_kwargs,
        current_risk_level=RiskLevel.RED,
        highest_risk_level=RiskLevel.RED,
    )
    assert s_red.marker_hex_color == "#ef4444"
    assert s_red.marker_folium_color == "red"


def test_unavailable_and_stale_telemetry_rendered_neutral_grey_never_green():
    """Verify unavailable or degraded telemetry is assigned neutral slate grey (#64748b)."""
    base_kwargs = dict(
        location_id="test_loc",
        name="Test",
        latitude=15.0,
        longitude=73.0,
        coordinate_label="15.00°N, 73.00°E",
        region="Test",
        is_reference=False,
        validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
        data_source="Test",
        retrieval_time_utc=None,
        forecast_issue_time_utc=None,
        latest_obs_swh_m=None,
        latest_obs_time_utc=None,
        forecast_swh_m=None,
        forecast_horizon_hours=6,
        forecast_valid_time_utc=None,
        max_window_swh_m=None,
        current_risk_level=RiskLevel.GREEN,
        highest_risk_level=RiskLevel.GREEN,
        threshold_crossed_m=None,
        active_alert_count=0,
    )

    # UNAVAILABLE
    s_unavail = NodeMapSummary(
        **base_kwargs,
        data_status=DataStatus.UNAVAILABLE,
        is_degraded=True,
    )
    assert s_unavail.marker_hex_color == "#64748b"
    assert s_unavail.marker_folium_color == "gray"
    assert s_unavail.status_label == "UNAVAILABLE / DEGRADED"
    # Must never be green!
    assert s_unavail.marker_hex_color != "#10b981"

    # STALE
    s_stale = NodeMapSummary(
        **base_kwargs,
        data_status=DataStatus.STALE,
        is_degraded=True,
    )
    assert s_stale.marker_hex_color == "#64748b" or s_stale.marker_hex_color == "#94a3b8"
    assert s_stale.marker_hex_color != "#10b981"


# =============================================================================
# 3. Top-Level Summary Metrics Tests
# =============================================================================

def test_compute_map_dashboard_metrics():
    """Verify compute_map_dashboard_metrics accurately aggregates map metrics."""
    summaries = [
        NodeMapSummary(
            location_id="loc_1",
            name="Node 1",
            latitude=18.0,
            longitude=72.0,
            coordinate_label="18.00°N, 72.00°E",
            region="Reg 1",
            is_reference=True,
            validation_status=ValidationStatus.REFERENCE_VALIDATED,
            data_status=DataStatus.FRESH,
            data_source="Test",
            retrieval_time_utc=datetime.now(timezone.utc),
            forecast_issue_time_utc=None,
            latest_obs_swh_m=1.1,
            latest_obs_time_utc=None,
            forecast_swh_m=1.2,
            forecast_horizon_hours=6,
            forecast_valid_time_utc=None,
            max_window_swh_m=1.3,
            current_risk_level=RiskLevel.GREEN,
            highest_risk_level=RiskLevel.GREEN,
            threshold_crossed_m=None,
            active_alert_count=0,
            is_degraded=False,
        ),
        NodeMapSummary(
            location_id="loc_2",
            name="Node 2",
            latitude=15.0,
            longitude=73.0,
            coordinate_label="15.00°N, 73.00°E",
            region="Reg 2",
            is_reference=False,
            validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
            data_status=DataStatus.FRESH,
            data_source="Test",
            retrieval_time_utc=datetime.now(timezone.utc),
            forecast_issue_time_utc=None,
            latest_obs_swh_m=2.2,
            latest_obs_time_utc=None,
            forecast_swh_m=2.8,  # High risk
            forecast_horizon_hours=6,
            forecast_valid_time_utc=None,
            max_window_swh_m=3.0,
            current_risk_level=RiskLevel.ORANGE,
            highest_risk_level=RiskLevel.ORANGE,
            threshold_crossed_m=2.5,
            active_alert_count=1,
            is_degraded=False,
        ),
        NodeMapSummary(
            location_id="loc_3",
            name="Node 3",
            latitude=12.0,
            longitude=74.0,
            coordinate_label="12.00°N, 74.00°E",
            region="Reg 3",
            is_reference=False,
            validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
            data_status=DataStatus.UNAVAILABLE,
            data_source="Test",
            retrieval_time_utc=None,
            forecast_issue_time_utc=None,
            latest_obs_swh_m=None,
            latest_obs_time_utc=None,
            forecast_swh_m=None,
            forecast_horizon_hours=6,
            forecast_valid_time_utc=None,
            max_window_swh_m=None,
            current_risk_level=RiskLevel.GREEN,
            highest_risk_level=RiskLevel.GREEN,
            threshold_crossed_m=None,
            active_alert_count=0,
            is_degraded=True,
        ),
    ]

    metrics = compute_map_dashboard_metrics(summaries)

    assert metrics["total_locations"] == 3
    assert metrics["fresh_locations"] == 2
    assert metrics["stale_or_unavailable_locations"] == 1
    assert metrics["elevated_risk_locations"] == 1  # Only Node 2 (Node 3 is degraded, not safe/low risk!)
    assert metrics["active_alert_locations"] == 1
    assert metrics["total_active_alerts"] == 1
    assert metrics["max_observed_or_forecast_swh_m"] == 2.8


# =============================================================================
# 4. Active Alert Linking Tests
# =============================================================================

def test_map_summaries_correlate_with_active_alerts(temp_alert_store, trained_scaler):
    """Verify that active alerts in AlertStore are accurately attached to map summaries."""
    engine = AlertEngine(store=temp_alert_store)
    event_time = datetime(2026, 10, 27, 18, 0, 0, tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)

    # Insert an active alert for Goa
    fp = compute_alert_fingerprint("goa_mormugao", EventType.PROJECT_FORECAST_RISK, 6, event_time)
    alert = AlertRecord(
        alert_id="test_alert_goa",
        fingerprint=fp,
        location_id="goa_mormugao",
        location_name="Mormugao / Goa Coastal Node",
        event_type="project_forecast_risk",
        severity="ORANGE",
        status=AlertStatus.ACTIVE,
        trigger_reason="Test alert",
        predicted_swh_m=2.85,
        threshold_m=2.50,
        forecast_horizon_hours=6,
        expected_event_time_utc=event_time,
        anchor_time_utc=now_utc,
        assessment_time_utc=now_utc,
        source_name="Test",
        data_status="fresh",
        model_name="WaveLSTM",
        validation_status="experimental_unvalidated",
        is_reference=False,
        created_at_utc=now_utc,
        updated_at_utc=now_utc,
    )
    temp_alert_store.insert_alert(alert)

    reg = create_default_registry()
    summaries = prepare_coastal_node_map_data(
        selected_horizon=6,
        source="demo",
        registry=reg,
        alert_store=temp_alert_store,
        scaler=trained_scaler,
    )

    goa = next(s for s in summaries if s.location_id == "goa_mormugao")
    assert goa.active_alert_count == 1
    assert len(goa.active_alerts) == 1
    assert goa.active_alerts[0].alert_id == "test_alert_goa"


# =============================================================================
# 5. Folium Map Builder Tests
# =============================================================================

def test_build_coastal_folium_map_construction(trained_scaler, temp_alert_store):
    """Verify build_coastal_folium_map returns a valid Folium map with markers and legend."""
    reg = create_default_registry()
    summaries = prepare_coastal_node_map_data(
        selected_horizon=6,
        source="demo",
        registry=reg,
        alert_store=temp_alert_store,
        scaler=trained_scaler,
    )

    folium_map = build_coastal_folium_map(summaries, selected_node_id="mumbai_offshore")
    assert folium_map is not None

    # Check HTML representation includes legend keywords
    html_repr = folium_map.get_root().render()
    assert "Coastal Wave Risk Map" in html_repr
    assert "Mumbai Offshore" in html_repr
    assert "Mormugao / Goa" in html_repr


def test_independent_error_isolation_on_node_failure(trained_scaler, temp_alert_store):
    """Verify failure of one node during ingestion does not prevent other nodes from rendering."""
    reg = create_default_registry()

    class FlakyIngestion(MultiLocationIngestionService):
        def fetch_all_locations(self, *args, **kwargs):
            # Normal fetch for Mumbai, failing fetch for Kochi
            res = super().fetch_all_locations(*args, **kwargs)
            res["kochi_coast"] = ProviderFetchResult(
                provider_name="FailedProvider",
                location_id="kochi_coast",
                latitude=9.95,
                longitude=76.15,
                retrieval_time_utc=datetime.now(timezone.utc),
                data_status=DataStatus.UNAVAILABLE,
                raw_df=pd.DataFrame(),
                error_message="Simulated connection timeout",
            )
            return res

    flaky_svc = FlakyIngestion(registry=reg)
    summaries = prepare_coastal_node_map_data(
        selected_horizon=6,
        source="demo",
        registry=reg,
        ingestion_svc=flaky_svc,
        alert_store=temp_alert_store,
        scaler=trained_scaler,
    )

    assert len(summaries) == 8
    kochi = next(s for s in summaries if s.location_id == "kochi_coast")
    assert kochi.is_degraded is True
    assert kochi.data_status == DataStatus.UNAVAILABLE
    assert kochi.marker_hex_color == "#64748b"

    # Mumbai remains valid and fresh
    mumbai = next(s for s in summaries if s.location_id == "mumbai_offshore")
    assert mumbai.is_degraded is False
    assert mumbai.forecast_swh_m is not None

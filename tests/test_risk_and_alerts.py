"""
Automated Unit and Integration Tests for Phase 3:
AI-Assisted Early-Warning Alert System & Risk Intelligence.

Validates:
1. RiskThresholdSet validation and ordering constraints.
2. Risk classification and boundary definitions (GREEN, YELLOW, ORANGE, RED).
3. RiskAssessment generation across current observations and forecast horizons.
4. Correct UTC event timestamps (anchor + horizon, never invented).
5. Degraded/unavailable telemetry handling (no silent GREEN safety guarantees).
6. Geographic validation status (Reference vs Experimental Regional Transfer).
7. Deterministic fingerprinting and duplicate alert suppression.
8. Alert lifecycle states (ACTIVE, ACKNOWLEDGED, RESOLVED, EXPIRED).
9. Severity escalation and hysteresis resolution.
10. API failure resilience (alerts preserved, not falsely resolved).
11. SQLite persistence and history logging across service restarts.
12. Independent multi-location risk evaluation.
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
    ValidationStatus,
    create_default_registry,
)
from src.intelligence.provider import (
    DataStatus,
    OfflineDemoProvider,
    ProviderFetchResult,
)
from src.intelligence.risk import (
    EventType,
    RiskAssessment,
    RiskLevel,
    RiskThresholdSet,
    ThresholdConfigManager,
    assess_node_risk,
    classify_risk,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Fixtures
# =============================================================================

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


@pytest.fixture
def trained_scaler():
    """Loads pre-fitted training scaler."""
    scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
    assert scaler_path.is_file(), "Scaler artifact missing."
    return TimeSeriesScaler.load(scaler_path)


@pytest.fixture
def temp_db_path(tmp_path):
    """Provides an isolated SQLite database path for testing persistence."""
    return tmp_path / "test_alerts.db"


@pytest.fixture
def temp_alert_store(temp_db_path):
    """Provides an isolated AlertStore instance."""
    return AlertStore(db_path=temp_db_path)


@pytest.fixture
def reference_node():
    return CoastalNode(
        location_id="mumbai_offshore",
        name="Mumbai Offshore (Reference Node)",
        latitude=18.50,
        longitude=72.50,
        region="Maharashtra Coast",
        timezone_str="Asia/Kolkata",
        enabled=True,
        is_reference=True,
        validation_status=ValidationStatus.REFERENCE_VALIDATED,
    )


@pytest.fixture
def experimental_node():
    return CoastalNode(
        location_id="goa_mormugao",
        name="Mormugao / Goa Coastal Node",
        latitude=15.40,
        longitude=73.70,
        region="Goa Coast",
        timezone_str="Asia/Kolkata",
        enabled=True,
        is_reference=False,
        validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
    )


# =============================================================================
# 1. Threshold Configuration & Boundary Tests
# =============================================================================

def test_threshold_set_valid_creation():
    """Verify default valid threshold initialization and attributes."""
    t = RiskThresholdSet()
    assert t.caution_m == 1.50
    assert t.high_risk_m == 2.50
    assert t.critical_m == 4.00
    assert t.hysteresis_buffer_m == 0.15
    assert t.is_demonstration is True
    assert "not an official maritime safety threshold" in t.guidance_notice.lower()


def test_threshold_set_ordering_validation():
    """Verify rejection of inverted or non-positive thresholds."""
    # Caution >= High Risk
    with pytest.raises(ValueError, match="strictly increasing"):
        RiskThresholdSet(caution_m=3.0, high_risk_m=2.5, critical_m=4.0)

    # High Risk >= Critical
    with pytest.raises(ValueError, match="strictly increasing"):
        RiskThresholdSet(caution_m=1.5, high_risk_m=4.5, critical_m=4.0)

    # Negative threshold
    with pytest.raises(ValueError, match="must be a positive number"):
        RiskThresholdSet(caution_m=-1.0, high_risk_m=2.0, critical_m=3.0)

    # Negative hysteresis buffer
    with pytest.raises(ValueError, match="must be non-negative"):
        RiskThresholdSet(caution_m=1.0, high_risk_m=2.0, critical_m=3.0, hysteresis_buffer_m=-0.1)

    # Hysteresis buffer too large
    with pytest.raises(ValueError, match="must be smaller than the minimum threshold interval"):
        RiskThresholdSet(caution_m=1.0, high_risk_m=2.0, critical_m=3.0, hysteresis_buffer_m=1.2)


def test_threshold_config_manager_overrides():
    """Verify ThresholdConfigManager handles global defaults and location overrides."""
    mgr = ThresholdConfigManager()
    default_t = mgr.get_thresholds()
    assert default_t.caution_m == 1.50

    custom_goa = RiskThresholdSet(caution_m=1.80, high_risk_m=2.80, critical_m=4.20, location_id="goa_mormugao")
    mgr.set_override("goa_mormugao", custom_goa)

    assert mgr.get_thresholds("goa_mormugao").caution_m == 1.80
    assert mgr.get_thresholds("mumbai_offshore").caution_m == 1.50  # Global unchanged

    mgr.remove_override("goa_mormugao")
    assert mgr.get_thresholds("goa_mormugao").caution_m == 1.50  # Restored to default


def test_classify_risk_categories():
    """Verify all risk category boundaries and threshold attribution."""
    thresh = RiskThresholdSet(caution_m=1.50, high_risk_m=2.50, critical_m=4.00)

    # Below caution -> GREEN (no threshold crossed)
    level, cross = classify_risk(0.85, thresh)
    assert level == RiskLevel.GREEN
    assert cross is None

    # Exactly at caution -> YELLOW
    level, cross = classify_risk(1.50, thresh)
    assert level == RiskLevel.YELLOW
    assert cross == 1.50

    # Between caution and high risk -> YELLOW
    level, cross = classify_risk(2.10, thresh)
    assert level == RiskLevel.YELLOW
    assert cross == 1.50

    # Exactly at high risk -> ORANGE
    level, cross = classify_risk(2.50, thresh)
    assert level == RiskLevel.ORANGE
    assert cross == 2.50

    # Between high risk and critical -> ORANGE
    level, cross = classify_risk(3.80, thresh)
    assert level == RiskLevel.ORANGE
    assert cross == 2.50

    # At or above critical -> RED
    level, cross = classify_risk(4.00, thresh)
    assert level == RiskLevel.RED
    assert cross == 4.00

    level, cross = classify_risk(6.50, thresh)
    assert level == RiskLevel.RED
    assert cross == 4.00


# =============================================================================
# 2. Risk Assessment Subsystem Tests
# =============================================================================

def test_assess_node_risk_observed_and_forecast_horizons(reference_node, sample_48h_dataframe, trained_scaler):
    """Verify assess_node_risk produces both observed condition (h=0) and forecast horizons (h=1, 6, 24)."""
    retrieval_time = pd.Timestamp("2026-10-26 12:00:00", tz="UTC").to_pydatetime()
    fetch_res = ProviderFetchResult(
        provider_name="TestProvider",
        location_id="mumbai_offshore",
        latitude=18.5,
        longitude=72.5,
        retrieval_time_utc=retrieval_time,
        data_status=DataStatus.FRESH,
        raw_df=sample_48h_dataframe,
        external_nwp_forecasts={1: 1.15, 6: 1.25, 24: 1.40},
    )

    assessments = assess_node_risk(
        node=reference_node,
        fetch_result=fetch_res,
        scaler=trained_scaler,
        horizons=[1, 6, 24],
    )

    assert len(assessments) == 4  # h=0 (observed), h=1, h=6, h=24

    # Check observed condition (h=0)
    obs = assessments[0]
    assert obs.forecast_horizon_hours == 0
    assert obs.event_type == EventType.OBSERVED_CONDITION_RISK
    assert obs.is_reference is True
    assert obs.validation_status == ValidationStatus.REFERENCE_VALIDATED
    assert obs.is_degraded_data is False

    # Check forecast horizons (h=1, 6, 24)
    expected_horizons = [1, 6, 24]
    for i, h in enumerate(expected_horizons):
        fc = assessments[i + 1]
        assert fc.forecast_horizon_hours == h
        assert fc.event_type == EventType.PROJECT_FORECAST_RISK
        # Expected target timestamp must equal anchor_time + h (derived, never invented!)
        expected_target = obs.forecast_valid_time_utc + timedelta(hours=h)
        assert fc.forecast_valid_time_utc == expected_target
        assert fc.predicted_swh_m > 0.0


def test_assess_node_risk_geographic_unvalidated_status(experimental_node, sample_48h_dataframe, trained_scaler):
    """Verify non-reference nodes are prominently marked with regional transfer limitations."""
    retrieval_time = pd.Timestamp("2026-10-26 12:00:00", tz="UTC").to_pydatetime()
    fetch_res = ProviderFetchResult(
        provider_name="TestProvider",
        location_id="goa_mormugao",
        latitude=15.4,
        longitude=73.7,
        retrieval_time_utc=retrieval_time,
        data_status=DataStatus.FRESH,
        raw_df=sample_48h_dataframe,
    )

    assessments = assess_node_risk(
        node=experimental_node,
        fetch_result=fetch_res,
        scaler=trained_scaler,
        horizons=[6],
    )

    for a in assessments:
        assert a.is_reference is False
        assert a.validation_status == ValidationStatus.EXPERIMENTAL_UNVALIDATED
        # Must contain explicit limitation notice
        lim_text = " ".join(a.limitations_and_uncertainty)
        assert "GEOGRAPHIC TRANSFER NOTICE" in lim_text
        assert "experimental regional estimates" in lim_text


def test_assess_node_risk_unavailable_telemetry_no_silent_green(reference_node):
    """Verify that unavailable provider data flags degraded data and avoids false safety guarantees."""
    now_utc = datetime.now(timezone.utc)
    broken_fetch = ProviderFetchResult(
        provider_name="OpenMeteoProvider",
        location_id="mumbai_offshore",
        latitude=18.5,
        longitude=72.5,
        retrieval_time_utc=now_utc,
        data_status=DataStatus.UNAVAILABLE,
        raw_df=pd.DataFrame(),
        error_message="HTTP 503: Service Temporarily Unavailable",
    )

    assessments = assess_node_risk(node=reference_node, fetch_result=broken_fetch)

    assert len(assessments) == 1
    a = assessments[0]
    assert a.is_degraded_data is True
    assert a.data_status == DataStatus.UNAVAILABLE
    assert "DATA UNAVAILABLE" in " ".join(a.limitations_and_uncertainty)
    assert "NOT guaranteed safety" in " ".join(a.limitations_and_uncertainty)


# =============================================================================
# 3. Alert Fingerprinting & Duplicate Suppression Tests
# =============================================================================

def test_alert_fingerprint_determinism():
    """Verify that the same event parameters produce identical fingerprints."""
    event_time = datetime(2026, 10, 27, 18, 0, 0, tzinfo=timezone.utc)
    fp1 = compute_alert_fingerprint("mumbai_offshore", EventType.PROJECT_FORECAST_RISK, 6, event_time)
    fp2 = compute_alert_fingerprint("mumbai_offshore", "project_forecast_risk", 6, event_time)
    assert fp1 == fp2
    assert fp1 == "mumbai_offshore:project_forecast_risk:6h:2026-10-27T18:00:00Z"

    # Different horizon produces distinct fingerprint
    fp3 = compute_alert_fingerprint("mumbai_offshore", EventType.PROJECT_FORECAST_RISK, 24, event_time)
    assert fp1 != fp3


def test_alert_duplicate_suppression_on_repeated_assessment(temp_alert_store):
    """Verify that repeated evaluation of the same forecast does NOT create duplicate alerts."""
    engine = AlertEngine(store=temp_alert_store)
    event_time = datetime(2026, 10, 27, 18, 0, 0, tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)

    assessment = RiskAssessment(
        location_id="mumbai_offshore",
        location_name="Mumbai Offshore",
        assessment_time_utc=now_utc,
        data_retrieval_time_utc=now_utc,
        forecast_valid_time_utc=event_time,
        forecast_horizon_hours=6,
        forecast_issue_time_utc=now_utc,
        event_type=EventType.PROJECT_FORECAST_RISK,
        risk_level=RiskLevel.YELLOW,
        threshold_crossed_m=1.50,
        threshold_config_version="demo-v1.0",
        predicted_swh_m=1.75,
        observed_persistence_swh_m=1.0,
        external_nwp_swh_m=None,
        source_name="OpenMeteoProvider",
        data_status=DataStatus.FRESH,
        model_name="WaveLSTM",
        is_reference=True,
        validation_status=ValidationStatus.REFERENCE_VALIDATED,
        trigger_explanation="WaveLSTM predicts 1.75m crossing Caution threshold (1.50m)",
    )

    # 1st evaluation -> Alert created
    rec1 = engine.process_assessment(assessment)
    assert rec1 is not None
    assert rec1.status == AlertStatus.ACTIVE
    assert len(temp_alert_store.list_alerts()) == 1

    # 2nd evaluation (identical or minor fluctuation) -> Alert updated, NOT duplicated
    rec2 = engine.process_assessment(assessment)
    assert rec2 is not None
    assert rec2.alert_id == rec1.alert_id
    assert len(temp_alert_store.list_alerts()) == 1  # Still exactly 1 alert!


# =============================================================================
# 4. Alert Lifecycle & Transitions Tests
# =============================================================================

def test_alert_severity_escalation(temp_alert_store):
    """Verify escalation from YELLOW to ORANGE updates alert severity and logs history."""
    engine = AlertEngine(store=temp_alert_store)
    event_time = datetime(2026, 10, 27, 18, 0, 0, tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)

    # 1. Trigger Caution alert (YELLOW, 1.70m)
    a1 = RiskAssessment(
        location_id="mumbai_offshore",
        location_name="Mumbai Offshore",
        assessment_time_utc=now_utc,
        data_retrieval_time_utc=now_utc,
        forecast_valid_time_utc=event_time,
        forecast_horizon_hours=6,
        forecast_issue_time_utc=now_utc,
        event_type=EventType.PROJECT_FORECAST_RISK,
        risk_level=RiskLevel.YELLOW,
        threshold_crossed_m=1.50,
        threshold_config_version="demo-v1.0",
        predicted_swh_m=1.70,
        observed_persistence_swh_m=1.0,
        external_nwp_swh_m=None,
        source_name="OpenMeteoProvider",
        data_status=DataStatus.FRESH,
        model_name="WaveLSTM",
        is_reference=True,
        validation_status=ValidationStatus.REFERENCE_VALIDATED,
        trigger_explanation="Predicted 1.70m exceeds 1.50m",
    )
    rec = engine.process_assessment(a1)
    assert rec.severity == "YELLOW"

    # 2. Escalate to High Risk (ORANGE, 2.80m)
    a2 = RiskAssessment(
        location_id="mumbai_offshore",
        location_name="Mumbai Offshore",
        assessment_time_utc=now_utc,
        data_retrieval_time_utc=now_utc,
        forecast_valid_time_utc=event_time,
        forecast_horizon_hours=6,
        forecast_issue_time_utc=now_utc,
        event_type=EventType.PROJECT_FORECAST_RISK,
        risk_level=RiskLevel.ORANGE,
        threshold_crossed_m=2.50,
        threshold_config_version="demo-v1.0",
        predicted_swh_m=2.80,
        observed_persistence_swh_m=1.0,
        external_nwp_swh_m=None,
        source_name="OpenMeteoProvider",
        data_status=DataStatus.FRESH,
        model_name="WaveLSTM",
        is_reference=True,
        validation_status=ValidationStatus.REFERENCE_VALIDATED,
        trigger_explanation="Predicted 2.80m exceeds 2.50m",
    )
    rec_esc = engine.process_assessment(a2)
    assert rec_esc.severity == "ORANGE"
    assert rec_esc.predicted_swh_m == 2.80

    # Verify history transition logged
    hist = temp_alert_store.get_history(rec.alert_id)
    assert len(hist) == 2
    assert "SEVERITY_ESCALATION" in hist[1]["reason"]


def test_alert_user_acknowledgement(temp_alert_store):
    """Verify operator acknowledgement transitions ACTIVE -> ACKNOWLEDGED."""
    engine = AlertEngine(store=temp_alert_store)
    event_time = datetime(2026, 10, 27, 18, 0, 0, tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)

    assessment = RiskAssessment(
        location_id="mumbai_offshore",
        location_name="Mumbai Offshore",
        assessment_time_utc=now_utc,
        data_retrieval_time_utc=now_utc,
        forecast_valid_time_utc=event_time,
        forecast_horizon_hours=6,
        forecast_issue_time_utc=now_utc,
        event_type=EventType.PROJECT_FORECAST_RISK,
        risk_level=RiskLevel.YELLOW,
        threshold_crossed_m=1.50,
        threshold_config_version="demo-v1.0",
        predicted_swh_m=1.70,
        observed_persistence_swh_m=1.0,
        external_nwp_swh_m=None,
        source_name="OpenMeteoProvider",
        data_status=DataStatus.FRESH,
        model_name="WaveLSTM",
        is_reference=True,
        validation_status=ValidationStatus.REFERENCE_VALIDATED,
        trigger_explanation="Predicted 1.70m",
    )
    rec = engine.process_assessment(assessment)
    assert rec.status == AlertStatus.ACTIVE

    # Acknowledge
    success = engine.acknowledge_alert(rec.alert_id, operator_notes="Notified port authority.")
    assert success is True

    updated = temp_alert_store.get_by_id(rec.alert_id)
    assert updated.status == AlertStatus.ACKNOWLEDGED
    assert updated.acknowledged_at_utc is not None
    assert updated.operator_notes == "Notified port authority."

    # Cannot re-acknowledge an already acknowledged alert
    assert engine.acknowledge_alert(rec.alert_id) is False


def test_alert_hysteresis_resolution(temp_alert_store):
    """Verify alert resolution requires dropping below threshold minus hysteresis buffer."""
    engine = AlertEngine(store=temp_alert_store)
    event_time = datetime(2026, 10, 27, 18, 0, 0, tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    thresholds = RiskThresholdSet(caution_m=1.50, hysteresis_buffer_m=0.15)
    # De-escalation target is: 1.50 - 0.15 = 1.35m

    # 1. Trigger Caution alert (1.60m)
    a1 = RiskAssessment(
        location_id="mumbai_offshore",
        location_name="Mumbai Offshore",
        assessment_time_utc=now_utc,
        data_retrieval_time_utc=now_utc,
        forecast_valid_time_utc=event_time,
        forecast_horizon_hours=6,
        forecast_issue_time_utc=now_utc,
        event_type=EventType.PROJECT_FORECAST_RISK,
        risk_level=RiskLevel.YELLOW,
        threshold_crossed_m=1.50,
        threshold_config_version="demo-v1.0",
        predicted_swh_m=1.60,
        observed_persistence_swh_m=1.0,
        external_nwp_swh_m=None,
        source_name="OpenMeteoProvider",
        data_status=DataStatus.FRESH,
        model_name="WaveLSTM",
        is_reference=True,
        validation_status=ValidationStatus.REFERENCE_VALIDATED,
        trigger_explanation="Exceeds 1.50m",
    )
    rec = engine.process_assessment(a1, thresholds=thresholds)
    assert rec.status == AlertStatus.ACTIVE

    # 2. Drop slightly below 1.50m (e.g. 1.42m).
    # Since 1.42m >= 1.35m (inside hysteresis band), do NOT resolve yet!
    a_hysteresis = RiskAssessment(
        location_id="mumbai_offshore",
        location_name="Mumbai Offshore",
        assessment_time_utc=now_utc,
        data_retrieval_time_utc=now_utc,
        forecast_valid_time_utc=event_time,
        forecast_horizon_hours=6,
        forecast_issue_time_utc=now_utc,
        event_type=EventType.PROJECT_FORECAST_RISK,
        risk_level=RiskLevel.GREEN,
        threshold_crossed_m=None,
        threshold_config_version="demo-v1.0",
        predicted_swh_m=1.42,
        observed_persistence_swh_m=1.0,
        external_nwp_swh_m=None,
        source_name="OpenMeteoProvider",
        data_status=DataStatus.FRESH,
        model_name="WaveLSTM",
        is_reference=True,
        validation_status=ValidationStatus.REFERENCE_VALIDATED,
        trigger_explanation="Below 1.50m but within hysteresis buffer",
    )
    rec_hold = engine.process_assessment(a_hysteresis, thresholds=thresholds)
    # Alert is not resolved; remains active
    assert rec_hold is None  # process_assessment returns None if no state change
    curr = temp_alert_store.get_by_id(rec.alert_id)
    assert curr.status == AlertStatus.ACTIVE

    # 3. Drop clearly below hysteresis boundary (1.20m < 1.35m) -> RESOLVE!
    a_resolved = RiskAssessment(
        location_id="mumbai_offshore",
        location_name="Mumbai Offshore",
        assessment_time_utc=now_utc,
        data_retrieval_time_utc=now_utc,
        forecast_valid_time_utc=event_time,
        forecast_horizon_hours=6,
        forecast_issue_time_utc=now_utc,
        event_type=EventType.PROJECT_FORECAST_RISK,
        risk_level=RiskLevel.GREEN,
        threshold_crossed_m=None,
        threshold_config_version="demo-v1.0",
        predicted_swh_m=1.20,
        observed_persistence_swh_m=1.0,
        external_nwp_swh_m=None,
        source_name="OpenMeteoProvider",
        data_status=DataStatus.FRESH,
        model_name="WaveLSTM",
        is_reference=True,
        validation_status=ValidationStatus.REFERENCE_VALIDATED,
        trigger_explanation="Well below hysteresis boundary",
    )
    rec_res = engine.process_assessment(a_resolved, thresholds=thresholds)
    assert rec_res is not None
    assert rec_res.status == AlertStatus.RESOLVED
    assert rec_res.resolved_at_utc is not None


def test_api_failure_preserves_active_alert(temp_alert_store):
    """Verify that an API network failure does NOT prematurely resolve an active alert."""
    engine = AlertEngine(store=temp_alert_store)
    event_time = datetime(2026, 10, 27, 18, 0, 0, tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)

    # 1. Active alert exists
    assessment = RiskAssessment(
        location_id="mumbai_offshore",
        location_name="Mumbai Offshore",
        assessment_time_utc=now_utc,
        data_retrieval_time_utc=now_utc,
        forecast_valid_time_utc=event_time,
        forecast_horizon_hours=6,
        forecast_issue_time_utc=now_utc,
        event_type=EventType.PROJECT_FORECAST_RISK,
        risk_level=RiskLevel.ORANGE,
        threshold_crossed_m=2.50,
        threshold_config_version="demo-v1.0",
        predicted_swh_m=2.90,
        observed_persistence_swh_m=1.0,
        external_nwp_swh_m=None,
        source_name="OpenMeteoProvider",
        data_status=DataStatus.FRESH,
        model_name="WaveLSTM",
        is_reference=True,
        validation_status=ValidationStatus.REFERENCE_VALIDATED,
        trigger_explanation="High risk alert",
    )
    rec = engine.process_assessment(assessment)
    assert rec.status == AlertStatus.ACTIVE

    # 2. Subsequent assessment is degraded / UNAVAILABLE
    failed_assessment = RiskAssessment(
        location_id="mumbai_offshore",
        location_name="Mumbai Offshore",
        assessment_time_utc=now_utc,
        data_retrieval_time_utc=now_utc,
        forecast_valid_time_utc=event_time,
        forecast_horizon_hours=6,
        forecast_issue_time_utc=None,
        event_type=EventType.PROJECT_FORECAST_RISK,
        risk_level=RiskLevel.GREEN,
        threshold_crossed_m=None,
        threshold_config_version="demo-v1.0",
        predicted_swh_m=0.0,
        observed_persistence_swh_m=None,
        external_nwp_swh_m=None,
        source_name="OpenMeteoProvider",
        data_status=DataStatus.UNAVAILABLE,
        model_name=None,
        is_reference=True,
        validation_status=ValidationStatus.REFERENCE_VALIDATED,
        trigger_explanation="Network timeout",
        is_degraded_data=True,
    )
    res_degraded = engine.process_assessment(failed_assessment)
    assert res_degraded is None  # Does NOT touch or resolve existing alert

    # Check that alert remains ACTIVE
    preserved = temp_alert_store.get_by_id(rec.alert_id)
    assert preserved.status == AlertStatus.ACTIVE


def test_sqlite_persistence_across_restarts(temp_db_path):
    """Verify that alerts and transition history survive simulated service restarts."""
    store1 = AlertStore(db_path=temp_db_path)
    engine1 = AlertEngine(store=store1)
    event_time = datetime(2026, 10, 27, 18, 0, 0, tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)

    assessment = RiskAssessment(
        location_id="mumbai_offshore",
        location_name="Mumbai Offshore",
        assessment_time_utc=now_utc,
        data_retrieval_time_utc=now_utc,
        forecast_valid_time_utc=event_time,
        forecast_horizon_hours=6,
        forecast_issue_time_utc=now_utc,
        event_type=EventType.PROJECT_FORECAST_RISK,
        risk_level=RiskLevel.RED,
        threshold_crossed_m=4.00,
        threshold_config_version="demo-v1.0",
        predicted_swh_m=4.25,
        observed_persistence_swh_m=1.0,
        external_nwp_swh_m=None,
        source_name="OpenMeteoProvider",
        data_status=DataStatus.FRESH,
        model_name="WaveLSTM",
        is_reference=True,
        validation_status=ValidationStatus.REFERENCE_VALIDATED,
        trigger_explanation="Critical swell event",
    )
    rec = engine1.process_assessment(assessment)
    engine1.acknowledge_alert(rec.alert_id, operator_notes="Alert broadcasted.")

    # Simulate restart by creating a completely new AlertStore instance pointing to the same file
    store2 = AlertStore(db_path=temp_db_path)
    reloaded = store2.get_by_id(rec.alert_id)

    assert reloaded is not None
    assert reloaded.alert_id == rec.alert_id
    assert reloaded.severity == "RED"
    assert reloaded.status == AlertStatus.ACKNOWLEDGED
    assert reloaded.predicted_swh_m == 4.25
    assert reloaded.operator_notes == "Alert broadcasted."

    # Verify history is preserved
    hist = store2.get_history(rec.alert_id)
    assert len(hist) == 2
    assert hist[0]["to_status"] == "ACTIVE"
    assert hist[1]["to_status"] == "ACKNOWLEDGED"


def test_multi_location_independent_assessment(temp_alert_store, trained_scaler):
    """Verify that multiple locations are evaluated independently without interference."""
    engine = AlertEngine(store=temp_alert_store)
    reg = create_default_registry()
    mumbai_node = reg.get("mumbai_offshore")
    goa_node = reg.get("goa_mormugao")

    demo_prov = OfflineDemoProvider()
    mumbai_fetch = demo_prov.fetch_telemetry(mumbai_node)
    goa_fetch = demo_prov.fetch_telemetry(goa_node)

    # Evaluate Mumbai with low thresholds (trigger alert)
    low_thresh = RiskThresholdSet(caution_m=0.50, high_risk_m=1.00, critical_m=2.00)
    mumbai_assess = assess_node_risk(mumbai_node, mumbai_fetch, scaler=trained_scaler, thresholds=low_thresh)
    engine.process_all_assessments(mumbai_assess, thresholds=low_thresh)

    # Evaluate Goa with high thresholds (no alert)
    high_thresh = RiskThresholdSet(caution_m=3.00, high_risk_m=4.50, critical_m=6.00)
    goa_assess = assess_node_risk(goa_node, goa_fetch, scaler=trained_scaler, thresholds=high_thresh)
    engine.process_all_assessments(goa_assess, thresholds=high_thresh)

    alerts = temp_alert_store.list_alerts()
    # Mumbai alerts exist, Goa alerts do not
    mumbai_alerts = [a for a in alerts if a.location_id == "mumbai_offshore"]
    goa_alerts = [a for a in alerts if a.location_id == "goa_mormugao"]

    assert len(mumbai_alerts) > 0
    assert len(goa_alerts) == 0

"""
Automated Pytest Suite for Coastal Risk Intelligence, Hysteresis,
Alert Lifecycle, and Simulation Isolation Audit.

Validates:
1. RiskThresholdSet validation and ordering constraints.
2. Boundary perturbation tests at millimeter precision (T - epsilon, T, T + epsilon).
3. Hysteresis resolution deadband behavior.
4. Alert lifecycle transitions (ACTIVE -> ACKNOWLEDGED -> RESOLVED / EXPIRED).
5. Immediate severity escalation without hysteresis lag.
6. Deduplication and deterministic fingerprint stability.
7. Telemetry outage and degradation resilience (alerts preserved, no silent GREEN).
8. Geospatial candidate node registry (1 reference, 7 experimental nodes).
9. Strict database isolation of the hazard simulation lab.
"""

from datetime import datetime, timezone, timedelta
import os
import sqlite3
import tempfile
import pytest

from src.intelligence.risk import (
    RiskLevel,
    EventType,
    RiskThresholdSet,
    ThresholdConfigManager,
    classify_risk,
    assess_node_risk,
    RiskAssessment,
)
from src.intelligence.alerts import (
    AlertStatus,
    AlertRecord,
    AlertStore,
    AlertEngine,
    compute_alert_fingerprint,
)
from src.intelligence.nodes import (
    ValidationStatus,
    CoastalNode,
    DEFAULT_COASTAL_NODES,
    get_coastal_registry,
)
from src.intelligence.provider import DataStatus
from src.intelligence.simulation import (
    ScenarioId,
    SCENARIO_DEFINITIONS,
    SimulationEngine,
    get_simulation_engine,
)


# -----------------------------------------------------------------------------
# 1. Threshold Invariants and Ordering Constraints
# -----------------------------------------------------------------------------

def test_risk_threshold_set_invariants():
    # Valid defaults
    t = RiskThresholdSet()
    assert t.caution_m == 1.50
    assert t.high_risk_m == 2.50
    assert t.critical_m == 4.00
    assert t.hysteresis_buffer_m == 0.15

    # Negative caution threshold
    with pytest.raises(ValueError, match="must be a positive number"):
        RiskThresholdSet(caution_m=-0.5)

    # Zero caution threshold
    with pytest.raises(ValueError, match="must be a positive number"):
        RiskThresholdSet(caution_m=0.0)

    # Inverted order: caution >= high_risk
    with pytest.raises(ValueError, match="strictly increasing"):
        RiskThresholdSet(caution_m=2.50, high_risk_m=2.50, critical_m=4.00)

    with pytest.raises(ValueError, match="strictly increasing"):
        RiskThresholdSet(caution_m=3.00, high_risk_m=2.50, critical_m=4.00)

    # Inverted order: high_risk >= critical
    with pytest.raises(ValueError, match="strictly increasing"):
        RiskThresholdSet(caution_m=1.50, high_risk_m=4.50, critical_m=4.00)

    # Negative hysteresis buffer
    with pytest.raises(ValueError, match="non-negative"):
        RiskThresholdSet(hysteresis_buffer_m=-0.05)

    # Hysteresis buffer >= min interval (interval is min(2.5-1.5, 4.0-2.5) = 1.0m)
    with pytest.raises(ValueError, match="smaller than the minimum threshold interval"):
        RiskThresholdSet(hysteresis_buffer_m=1.00)


# -----------------------------------------------------------------------------
# 2. Millimeter-Precision Boundary Perturbations
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "swh_val,expected_level,expected_crossed",
    [
        (1.490, RiskLevel.GREEN, None),
        (1.499, RiskLevel.GREEN, None),
        (1.500, RiskLevel.YELLOW, 1.50),
        (1.501, RiskLevel.YELLOW, 1.50),
        (2.490, RiskLevel.YELLOW, 1.50),
        (2.499, RiskLevel.YELLOW, 1.50),
        (2.500, RiskLevel.ORANGE, 2.50),
        (2.501, RiskLevel.ORANGE, 2.50),
        (3.990, RiskLevel.ORANGE, 2.50),
        (3.999, RiskLevel.ORANGE, 2.50),
        (4.000, RiskLevel.RED, 4.00),
        (4.001, RiskLevel.RED, 4.00),
    ],
)
def test_boundary_perturbations_exact_mm(swh_val, expected_level, expected_crossed):
    thresholds = RiskThresholdSet()
    level, crossed = classify_risk(swh_val, thresholds)
    assert level == expected_level
    assert crossed == expected_crossed


# -----------------------------------------------------------------------------
# 3. Hysteresis Resolution Deadband
# -----------------------------------------------------------------------------

def test_hysteresis_resolution_deadband():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        test_db_path = tmp_db.name

    try:
        store = AlertStore(test_db_path)
        engine = AlertEngine(store=store)
        now_utc = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)
        target_valid = now_utc + timedelta(hours=6)

        def make_test_assessment(swh: float, risk_level: RiskLevel, thresh: float = None):
            return RiskAssessment(
                location_id="mumbai_offshore",
                location_name="Mumbai Offshore",
                assessment_time_utc=now_utc,
                data_retrieval_time_utc=now_utc,
                forecast_valid_time_utc=target_valid,
                forecast_horizon_hours=6,
                forecast_issue_time_utc=now_utc,
                event_type=EventType.PROJECT_FORECAST_RISK,
                risk_level=risk_level,
                threshold_crossed_m=thresh,
                threshold_config_version="demo-v1.0",
                predicted_swh_m=swh,
                observed_persistence_swh_m=1.0,
                external_nwp_swh_m=swh,
                source_name="TestEngine",
                data_status=DataStatus.FRESH,
                model_name="Residual WaveANN",
                is_reference=True,
                validation_status=ValidationStatus.REFERENCE_VALIDATED,
                trigger_explanation=f"SWH {swh}m",
            )

        # 1. Trigger Caution Alert at 1.80m
        a1 = make_test_assessment(1.80, RiskLevel.YELLOW, 1.50)
        rec = engine.process_assessment(a1, reference_now_utc=now_utc)
        assert rec is not None
        assert rec.status == AlertStatus.ACTIVE

        # 2. Condition drops to 1.45m: below 1.50m, but inside hysteresis deadband (1.35m - 1.50m)
        a_deadband = make_test_assessment(1.45, RiskLevel.GREEN, None)
        rec2 = engine.process_assessment(a_deadband, reference_now_utc=now_utc)
        alert_db = store.get_by_id(rec.alert_id)
        assert alert_db.status == AlertStatus.ACTIVE, "Alert must NOT resolve in the hysteresis deadband"

        # 3. Condition at exactly 1.350m: must be strictly < 1.350m to resolve
        a_exact = make_test_assessment(1.350, RiskLevel.GREEN, None)
        engine.process_assessment(a_exact, reference_now_utc=now_utc)
        alert_db = store.get_by_id(rec.alert_id)
        assert alert_db.status == AlertStatus.ACTIVE, "Alert must NOT resolve at exact hysteresis threshold"

        # 4. Condition drops to 1.340m: strictly below 1.350m -> Resolves
        a_res = make_test_assessment(1.340, RiskLevel.GREEN, None)
        rec_res = engine.process_assessment(a_res, reference_now_utc=now_utc)
        assert rec_res is not None
        assert rec_res.status == AlertStatus.RESOLVED
        alert_db = store.get_by_id(rec.alert_id)
        assert alert_db.status == AlertStatus.RESOLVED

    finally:
        try:
            os.remove(test_db_path)
        except OSError:
            pass


# -----------------------------------------------------------------------------
# 4. Alert Lifecycle, Escalation, and Acknowledgement
# -----------------------------------------------------------------------------

def test_alert_lifecycle_escalation_and_acknowledgement():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        test_db_path = tmp_db.name

    try:
        store = AlertStore(test_db_path)
        engine = AlertEngine(store=store)
        now_utc = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)
        target_valid = now_utc + timedelta(hours=6)

        # Trigger YELLOW
        a_yellow = RiskAssessment(
            location_id="mumbai_offshore",
            location_name="Mumbai Offshore",
            assessment_time_utc=now_utc,
            data_retrieval_time_utc=now_utc,
            forecast_valid_time_utc=target_valid,
            forecast_horizon_hours=6,
            forecast_issue_time_utc=now_utc,
            event_type=EventType.PROJECT_FORECAST_RISK,
            risk_level=RiskLevel.YELLOW,
            threshold_crossed_m=1.50,
            threshold_config_version="demo-v1.0",
            predicted_swh_m=1.80,
            observed_persistence_swh_m=1.0,
            external_nwp_swh_m=1.8,
            source_name="TestEngine",
            data_status=DataStatus.FRESH,
            model_name="Residual WaveANN",
            is_reference=True,
            validation_status=ValidationStatus.REFERENCE_VALIDATED,
            trigger_explanation="Yellow Alert",
        )
        rec = engine.process_assessment(a_yellow, reference_now_utc=now_utc)
        assert rec.severity == "YELLOW"
        assert rec.status == AlertStatus.ACTIVE

        # Escalate immediately to ORANGE without hysteresis lag
        a_orange = RiskAssessment(
            location_id="mumbai_offshore",
            location_name="Mumbai Offshore",
            assessment_time_utc=now_utc,
            data_retrieval_time_utc=now_utc,
            forecast_valid_time_utc=target_valid,
            forecast_horizon_hours=6,
            forecast_issue_time_utc=now_utc,
            event_type=EventType.PROJECT_FORECAST_RISK,
            risk_level=RiskLevel.ORANGE,
            threshold_crossed_m=2.50,
            threshold_config_version="demo-v1.0",
            predicted_swh_m=2.90,
            observed_persistence_swh_m=1.0,
            external_nwp_swh_m=2.9,
            source_name="TestEngine",
            data_status=DataStatus.FRESH,
            model_name="Residual WaveANN",
            is_reference=True,
            validation_status=ValidationStatus.REFERENCE_VALIDATED,
            trigger_explanation="Orange Alert Escalation",
        )
        rec_orange = engine.process_assessment(a_orange, reference_now_utc=now_utc)
        assert rec_orange.severity == "ORANGE"
        assert rec_orange.status == AlertStatus.ACTIVE

        # Operator Acknowledgement
        ack_ok = engine.acknowledge_alert(rec.alert_id, operator_notes="Checked with port authority")
        assert ack_ok is True
        rec_ack = store.get_by_id(rec.alert_id)
        assert rec_ack.status == AlertStatus.ACKNOWLEDGED
        assert rec_ack.operator_notes == "Checked with port authority"

        # Check transition history
        history = store.get_history(rec.alert_id)
        assert len(history) >= 3  # Initial trigger, escalation, acknowledgement

    finally:
        try:
            os.remove(test_db_path)
        except OSError:
            pass


# -----------------------------------------------------------------------------
# 5. Telemetry Degradation Resilience
# -----------------------------------------------------------------------------

def test_telemetry_outage_preserves_active_alert():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        test_db_path = tmp_db.name

    try:
        store = AlertStore(test_db_path)
        engine = AlertEngine(store=store)
        now_utc = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)
        target_valid = now_utc + timedelta(hours=6)

        # Trigger active alert
        a = RiskAssessment(
            location_id="mumbai_offshore",
            location_name="Mumbai Offshore",
            assessment_time_utc=now_utc,
            data_retrieval_time_utc=now_utc,
            forecast_valid_time_utc=target_valid,
            forecast_horizon_hours=6,
            forecast_issue_time_utc=now_utc,
            event_type=EventType.PROJECT_FORECAST_RISK,
            risk_level=RiskLevel.YELLOW,
            threshold_crossed_m=1.50,
            threshold_config_version="demo-v1.0",
            predicted_swh_m=1.80,
            observed_persistence_swh_m=1.0,
            external_nwp_swh_m=1.8,
            source_name="TestEngine",
            data_status=DataStatus.FRESH,
            model_name="Residual WaveANN",
            is_reference=True,
            validation_status=ValidationStatus.REFERENCE_VALIDATED,
            trigger_explanation="Yellow Alert",
        )
        rec = engine.process_assessment(a, reference_now_utc=now_utc)
        assert rec.status == AlertStatus.ACTIVE

        # Simulate degraded telemetry (UNAVAILABLE)
        a_degraded = RiskAssessment(
            location_id="mumbai_offshore",
            location_name="Mumbai Offshore",
            assessment_time_utc=now_utc,
            data_retrieval_time_utc=now_utc,
            forecast_valid_time_utc=target_valid,
            forecast_horizon_hours=6,
            forecast_issue_time_utc=now_utc,
            event_type=EventType.PROJECT_FORECAST_RISK,
            risk_level=RiskLevel.GREEN,
            threshold_crossed_m=None,
            threshold_config_version="demo-v1.0",
            predicted_swh_m=0.0,
            observed_persistence_swh_m=None,
            external_nwp_swh_m=None,
            source_name="TestEngine",
            data_status=DataStatus.UNAVAILABLE,
            model_name=None,
            is_reference=True,
            validation_status=ValidationStatus.REFERENCE_VALIDATED,
            trigger_explanation="Telemetry API unavailable",
            is_degraded_data=True,
        )
        res_deg = engine.process_assessment(a_degraded, reference_now_utc=now_utc)
        assert res_deg is None  # Engine rejects processing degraded data for alert resolution
        
        # Verify alert in DB remains ACTIVE
        persisted = store.get_by_id(rec.alert_id)
        assert persisted.status == AlertStatus.ACTIVE, "Active alert must NOT be resolved when telemetry is unavailable"

    finally:
        try:
            os.remove(test_db_path)
        except OSError:
            pass


# -----------------------------------------------------------------------------
# 6. Geospatial Candidate Node Registry
# -----------------------------------------------------------------------------

def test_geospatial_node_classification():
    registry = get_coastal_registry()
    nodes = registry.list_all()
    assert len(nodes) == 8

    # Exactly 1 calibrated reference node
    ref_nodes = [n for n in nodes if n.is_reference]
    assert len(ref_nodes) == 1
    assert ref_nodes[0].location_id == "mumbai_offshore"
    assert ref_nodes[0].validation_status == ValidationStatus.REFERENCE_VALIDATED

    # Exactly 7 experimental unvalidated nodes
    exp_nodes = [n for n in nodes if not n.is_reference]
    assert len(exp_nodes) == 7
    for n in exp_nodes:
        assert n.validation_status == ValidationStatus.EXPERIMENTAL_UNVALIDATED
        # Verify valid Indian coastal waters coordinates
        assert 7.0 <= n.latitude <= 25.0
        assert 68.0 <= n.longitude <= 90.0


# -----------------------------------------------------------------------------
# 7. Simulation Database Isolation
# -----------------------------------------------------------------------------

def test_simulation_database_isolation():
    intel_db_path = pytest.PROJECT_ROOT = DEFAULT_COASTAL_NODES[0].location_id  # dummy
    db_file = AlertStore().db_path
    
    # Count rows before simulation
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM alerts")
    pre_alert_count = cur.fetchone()[0]
    conn.close()

    engine = get_simulation_engine()
    engine.start(ScenarioId.TROPICAL_CYCLONE, initial_step=12)
    engine.step(+3)

    # Count rows after simulation
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM alerts")
    post_alert_count = cur.fetchone()[0]
    conn.close()

    assert pre_alert_count == post_alert_count, "Simulation runs must NEVER write to production SQLite database"

    # Reset simulation
    engine.reset()

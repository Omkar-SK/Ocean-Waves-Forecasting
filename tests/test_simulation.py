"""
Integration and Unit Tests for the Interactive Ocean Hazard Simulation Subsystem.
Verifies all 4 hazard scenarios, state transitions (start, pause, resume, step, reset),
dynamic risk calculations, spatial heatmaps, and strict persistent DB isolation.
"""

import sqlite3
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.intelligence.alerts import DEFAULT_DB_PATH
from src.intelligence.simulation import (
    SCENARIO_DEFINITIONS,
    ScenarioId,
    get_simulation_engine,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_simulation_reset():
    """Ensure simulation engine is reset before and after each test."""
    engine = get_simulation_engine()
    engine.reset()
    yield
    engine.reset()


def test_list_scenarios_catalog():
    """Verify that all 4 predefined hypothetical hazard scenarios are returned with valid metadata."""
    res = client.get("/api/simulation/scenarios")
    assert res.status_code == 200
    scenarios = res.json()
    assert len(scenarios) == 4

    scenario_ids = {s["scenario_id"] for s in scenarios}
    expected_ids = {"severe_wave", "tropical_cyclone", "coastal_flooding", "moderate_disturbance"}
    assert scenario_ids == expected_ids

    for s in scenarios:
        assert s["base_swh_m"] > 0
        assert s["peak_swh_m"] > s["base_swh_m"]
        assert s["peak_wind_kmh"] > 0
        assert s["peak_mwp_s"] > 0
        assert len(s["headline"]) > 0
        assert len(s["recommended_action"]) > 0


def test_start_severe_wave_scenario():
    """Verify Scenario A (Severe Wave Conditions) generates expected hazard metrics and alerts."""
    payload = {"scenario_id": "severe_wave", "initial_step": 14}
    res = client.post("/api/simulation/start", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"]["is_active"] is True
    assert data["status"]["state"] == "RUNNING"
    assert data["status"]["scenario_id"] == "severe_wave"
    assert data["status"]["current_step"] == 14

    stats = data["stats"]
    assert 5.5 <= stats["max_swh_m"] <= 7.5
    assert 65.0 <= stats["max_wind_kmh"] <= 90.0
    assert stats["critical_risk_zones"] >= 1
    assert stats["active_alerts"] >= 1

    # Verify spatial heatmap points
    assert len(data["heatmap_points"]) >= 8
    for pt in data["heatmap_points"]:
        assert 0.0 <= pt["intensity"] <= 1.0
        assert pt["color"].startswith("#")
        assert pt["radius_km"] > 0

    # Verify time-series storm profile
    assert len(data["time_series"]) == 25  # 0 to 24 hours
    step_14 = data["time_series"][14]
    assert step_14["actual"] is not None
    assert step_14["lstm"] is not None


def test_start_tropical_cyclone_scenario():
    """Verify Scenario B (Tropical Cyclone Impact) generates extreme wave heights and hurricane winds."""
    payload = {"scenario_id": "tropical_cyclone", "initial_step": 14}
    res = client.post("/api/simulation/start", json=payload)
    assert res.status_code == 200
    data = res.json()

    stats = data["stats"]
    assert 7.0 <= stats["max_swh_m"] <= 10.0
    assert 100.0 <= stats["max_wind_kmh"] <= 140.0
    assert stats["severity_category"] == "CRITICAL"
    assert stats["coastal_risk_index"] >= 80


def test_start_coastal_flooding_scenario():
    """Verify Scenario C (Coastal Flooding & Surge) generates inundation alerts and elevated onshore winds."""
    payload = {"scenario_id": "coastal_flooding", "initial_step": 14}
    res = client.post("/api/simulation/start", json=payload)
    assert res.status_code == 200
    data = res.json()

    stats = data["stats"]
    assert 4.5 <= stats["max_swh_m"] <= 6.8
    assert 75.0 <= stats["max_wind_kmh"] <= 105.0


def test_start_moderate_disturbance_scenario():
    """Verify Scenario D (Moderate Marine Disturbance) generates moderate conditions and caution alerts."""
    payload = {"scenario_id": "moderate_disturbance", "initial_step": 14}
    res = client.post("/api/simulation/start", json=payload)
    assert res.status_code == 200
    data = res.json()

    stats = data["stats"]
    assert 2.6 <= stats["max_swh_m"] <= 4.2
    assert 40.0 <= stats["max_wind_kmh"] <= 60.0
    assert stats["severity_category"] == "MODERATE"


def test_invalid_scenario_id_rejection():
    """Verify that an invalid scenario ID returns HTTP 400 with a descriptive error message."""
    payload = {"scenario_id": "tsunami_asteroid", "initial_step": 6}
    res = client.post("/api/simulation/start", json=payload)
    assert res.status_code == 400
    assert "Invalid scenario_id" in res.json()["detail"]


def test_simulation_lifecycle_pause_resume_step_reset():
    """Verify full state machine: start -> pause -> resume -> step -> reset."""
    # 1. Start
    client.post("/api/simulation/start", json={"scenario_id": "severe_wave", "initial_step": 6})

    # 2. Pause
    pause_res = client.post("/api/simulation/pause")
    assert pause_res.status_code == 200
    assert pause_res.json()["state"] == "PAUSED"

    # 3. Resume
    resume_res = client.post("/api/simulation/resume")
    assert resume_res.status_code == 200
    assert resume_res.json()["state"] == "RUNNING"

    # 4. Step forward (+2 hours)
    step_res = client.post("/api/simulation/step?delta=2")
    assert step_res.status_code == 200
    assert step_res.json()["status"]["current_step"] == 8

    # 5. Status
    status_res = client.get("/api/simulation/status")
    assert status_res.status_code == 200
    assert status_res.json()["is_active"] is True
    assert status_res.json()["current_step"] == 8

    # 6. Results
    results_res = client.get("/api/simulation/results")
    assert results_res.status_code == 200
    assert results_res.json()["status"]["is_active"] is True

    # 7. Reset
    reset_res = client.post("/api/simulation/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["is_active"] is False
    assert reset_res.json()["state"] == "IDLE"

    # 8. Status after reset
    final_status = client.get("/api/simulation/status")
    assert final_status.json()["is_active"] is False


def test_strict_database_isolation():
    """
    Guarantees requirement 8 & 9:
    Running simulations must NEVER contaminate persistent SQLite database tables
    (alerts, forecast_ledger, observations, monitoring_events).
    """
    # Count records before simulation
    with sqlite3.connect(DEFAULT_DB_PATH) as conn:
        cursor = conn.cursor()
        alerts_before = cursor.execute("SELECT COUNT(*) FROM alerts;").fetchone()[0]
        ledger_before = cursor.execute("SELECT COUNT(*) FROM forecast_ledger;").fetchone()[0]

    # Run severe cyclone simulation with 8 simulated critical alerts
    client.post("/api/simulation/start", json={"scenario_id": "tropical_cyclone", "initial_step": 14})
    results = client.get("/api/simulation/results").json()
    assert len(results["alerts"]) > 0

    # Verify that persistent SQLite database tables have NOT increased
    with sqlite3.connect(DEFAULT_DB_PATH) as conn:
        cursor = conn.cursor()
        alerts_after = cursor.execute("SELECT COUNT(*) FROM alerts;").fetchone()[0]
        ledger_after = cursor.execute("SELECT COUNT(*) FROM forecast_ledger;").fetchone()[0]

    assert alerts_after == alerts_before, "Simulation contaminated persistent alerts table!"
    assert ledger_after == ledger_before, "Simulation contaminated persistent forecast ledger!"

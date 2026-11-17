"""
Automated Unit and Integration Tests for Continuous Forecast Verification & Model Monitoring.

Validates:
1. Forecast archival with complete operational metadata.
2. Duplicate forecast prevention via deterministic fingerprinting.
3. Timestamp matching within documented temporal tolerance (+/- 30 minutes).
4. Observation ingestion and physical quality validation (bounds 0-25m, non-negative).
5. Forecast error calculation: Error = Predicted SWH - Observed SWH.
6. Persistence error and skill score calculation against persistence baseline.
7. Rejection of invalid observations (non-physical bounds, non-physical source types).
8. Identification of missing observations when target timestamps pass without data.
9. Delayed observation ingestion matching retroactively within tolerance.
10. Multi-horizon tracking (+1h, +6h, +24h) and regional isolation.
11. Severe-event monitoring (SWH >= 4.0m) and distinction of historical sample (N=22).
12. FastAPI verification endpoint status codes, schemas, and response correctness.
"""

from datetime import datetime, timezone, timedelta
import math
from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.verification import (
    ForecastArchivalRequest,
    ObservationIngestRequest,
    VerificationRecordStatus,
    VerificationWindow,
)
from backend.services.forecast_verification_service import (
    ForecastVerificationService,
    MIN_RELIABLE_SAMPLES,
)


@pytest.fixture
def temp_verification_service():
    """Isolated verification service using a temporary SQLite database."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = Path(tmp_dir) / "test_verification.db"
        svc = ForecastVerificationService(db_path=db_path)
        yield svc


@pytest.fixture
def test_client():
    """FastAPI TestClient instance."""
    with TestClient(app) as client:
        yield client


# =============================================================================
# 1. Forecast Archival & Duplicate Prevention
# =============================================================================

def test_forecast_archival_success(temp_verification_service):
    svc = temp_verification_service
    now = datetime.now(timezone.utc)
    target = now + timedelta(hours=6)

    req = ForecastArchivalRequest(
        forecast_id="fc_test_001",
        location_id="mumbai_offshore",
        forecast_origin_time=now,
        target_timestamp=target,
        forecast_horizon=6,
        current_swh=1.20,
        predicted_swh=1.45,
        predicted_residual=0.25,
        model_architecture="Residual WaveANN",
        model_version="2.0.0-residual",
        input_data_timestamp=now,
        data_source="OpenMeteoProvider",
        data_freshness_minutes=15,
        forecast_validation_status="reference_validated",
        is_reference=True,
    )

    res = svc.archive_forecast(req)
    assert res.success is True
    assert res.is_duplicate is False
    assert res.status == VerificationRecordStatus.PENDING_VERIFICATION
    assert res.forecast_id == "fc_test_001"


def test_forecast_duplicate_prevention(temp_verification_service):
    svc = temp_verification_service
    now = datetime.now(timezone.utc)
    target = now + timedelta(hours=6)

    req = ForecastArchivalRequest(
        forecast_id="fc_test_dup",
        location_id="mumbai_offshore",
        forecast_origin_time=now,
        target_timestamp=target,
        forecast_horizon=6,
        current_swh=1.20,
        predicted_swh=1.45,
        predicted_residual=0.25,
        model_architecture="Residual WaveANN",
        model_version="2.0.0-residual",
        input_data_timestamp=now,
        data_source="OpenMeteoProvider",
    )

    res1 = svc.archive_forecast(req)
    assert res1.success is True
    assert res1.is_duplicate is False

    # Second issuance with identical origin, target, and location
    res2 = svc.archive_forecast(req)
    assert res2.success is True
    assert res2.is_duplicate is True
    assert "Duplicate" in res2.message


# =============================================================================
# 2. Observation Ingestion & Timestamp Matching
# =============================================================================

def test_observation_ingestion_and_timestamp_matching(temp_verification_service):
    svc = temp_verification_service
    origin = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
    target = origin + timedelta(hours=1)

    # 1. Archive +1h forecast
    fc_req = ForecastArchivalRequest(
        forecast_id="fc_h1_match",
        location_id="mumbai_offshore",
        forecast_origin_time=origin,
        target_timestamp=target,
        forecast_horizon=1,
        current_swh=1.50,
        predicted_swh=1.65,
        predicted_residual=0.15,
        model_architecture="Residual WaveLSTM",
        model_version="2.0.0-residual",
        input_data_timestamp=origin,
        data_source="OpenMeteoProvider",
    )
    svc.archive_forecast(fc_req)

    # 2. Ingest observation within tolerance (+10 minutes of target)
    obs_time = target + timedelta(minutes=10)
    obs_req = ObservationIngestRequest(
        location_id="mumbai_offshore",
        observation_time=obs_time,
        observed_swh=1.60,
        source_name="INCOIS_BUOY",
        source_type="in_situ_buoy",
        quality_status="valid",
    )
    obs_res = svc.ingest_observation(obs_req)
    assert obs_res.success is True
    assert obs_res.matched_forecasts_count == 1

    # 3. Verify matched record and error calculations
    dash = svc.get_verification_dashboard(location_id="mumbai_offshore")
    assert dash.total_verified_observations == 1
    matched_rec = next(r for r in dash.recent_verifications if r.forecast_id == "fc_h1_match")
    assert matched_rec.status == VerificationRecordStatus.VERIFIED
    assert matched_rec.observed_swh == 1.60
    # Error = Predicted (1.65) - Observed (1.60) = +0.05
    assert pytest.approx(matched_rec.forecast_error, abs=1e-4) == 0.05
    assert pytest.approx(matched_rec.absolute_error, abs=1e-4) == 0.05
    # Persistence Error = Current (1.50) - Observed (1.60) = -0.10
    assert pytest.approx(matched_rec.persistence_error, abs=1e-4) == -0.10
    assert matched_rec.is_prospective is True


def test_timestamp_tolerance_boundary_rejection(temp_verification_service):
    """Observations arriving outside the +/- 30m tolerance window must not match."""
    svc = temp_verification_service
    origin = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
    target = origin + timedelta(hours=6)

    svc.archive_forecast(
        ForecastArchivalRequest(
            forecast_id="fc_tol_test",
            location_id="mumbai_offshore",
            forecast_origin_time=origin,
            target_timestamp=target,
            forecast_horizon=6,
            current_swh=1.00,
            predicted_swh=1.20,
            model_architecture="Residual WaveANN",
            input_data_timestamp=origin,
        )
    )

    # Observation arrived 45 minutes after target (exceeds 30m tolerance)
    late_obs = target + timedelta(minutes=45)
    obs_res = svc.ingest_observation(
        ObservationIngestRequest(
            location_id="mumbai_offshore",
            observation_time=late_obs,
            observed_swh=1.18,
            source_type="in_situ_buoy",
        )
    )
    assert obs_res.matched_forecasts_count == 0

    dash = svc.get_verification_dashboard(location_id="mumbai_offshore")
    assert dash.total_verified_observations == 0
    assert dash.total_pending_verifications == 1


# =============================================================================
# 3. Invalid Observations & Edge Cases
# =============================================================================

def test_invalid_observation_bounds_rejection(temp_verification_service):
    """Negative wave heights or values exceeding 25m must be rejected."""
    svc = temp_verification_service
    now = datetime.now(timezone.utc)

    # Negative SWH
    res_neg = svc.ingest_observation(
        ObservationIngestRequest(
            location_id="mumbai_offshore",
            observation_time=now,
            observed_swh=-0.5,
            source_type="in_situ_buoy",
        )
    )
    assert res_neg.success is False
    assert "violates physical" in res_neg.message

    # SWH exceeding physical limit
    res_high = svc.ingest_observation(
        ObservationIngestRequest(
            location_id="mumbai_offshore",
            observation_time=now,
            observed_swh=32.0,
            source_type="in_situ_buoy",
        )
    )
    assert res_high.success is False


def test_reject_model_derived_pseudo_observations(temp_verification_service):
    """Model predictions (e.g., NWP) cannot masquerade as in-situ observations."""
    svc = temp_verification_service
    origin = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
    target = origin + timedelta(hours=1)

    svc.archive_forecast(
        ForecastArchivalRequest(
            forecast_id="fc_nwp_reject",
            location_id="mumbai_offshore",
            forecast_origin_time=origin,
            target_timestamp=target,
            forecast_horizon=1,
            current_swh=1.00,
            predicted_swh=1.20,
            model_architecture="Residual WaveLSTM",
            input_data_timestamp=origin,
        )
    )

    # Ingest with source_type = "external_nwp"
    svc.ingest_observation(
        ObservationIngestRequest(
            location_id="mumbai_offshore",
            observation_time=target,
            observed_swh=1.19,
            source_type="external_nwp",  # not a physical observation
            quality_status="valid",
        )
    )

    dash = svc.get_verification_dashboard(location_id="mumbai_offshore")
    rec = next(r for r in dash.recent_verifications if r.forecast_id == "fc_nwp_reject")
    assert rec.status == VerificationRecordStatus.INVALID_OBSERVATION
    assert "model-derived" in rec.status_reason


# =============================================================================
# 4. Multi-Horizon Tracking & Rolling Metrics
# =============================================================================

def test_multi_horizon_continuous_accuracy_metrics(temp_verification_service):
    svc = temp_verification_service
    base_time = datetime(2026, 11, 15, 0, 0, tzinfo=timezone.utc)

    # Ingest 10 consecutive hourly forecasts and matching observations for +6h
    for i in range(10):
        t_origin = base_time + timedelta(hours=i)
        t_target = t_origin + timedelta(hours=6)
        actual_swh = 1.0 + 0.1 * (i % 5)
        pred_swh = actual_swh + 0.04  # Error = +0.04m constant
        curr_swh = actual_swh + 0.10  # Persistence Error = +0.10m constant

        svc.archive_forecast(
            ForecastArchivalRequest(
                forecast_id=f"fc_seq_{i}",
                location_id="mumbai_offshore",
                forecast_origin_time=t_origin,
                target_timestamp=t_target,
                forecast_horizon=6,
                current_swh=curr_swh,
                predicted_swh=pred_swh,
                model_architecture="Residual WaveANN",
                input_data_timestamp=t_origin,
            )
        )

        svc.ingest_observation(
            ObservationIngestRequest(
                location_id="mumbai_offshore",
                observation_time=t_target,
                observed_swh=actual_swh,
                source_type="in_situ_buoy",
            )
        )

    # Check metrics calculation
    metrics = svc.calculate_window_metrics(
        horizon=6,
        window=VerificationWindow.ALL_TIME,
        location_id="mumbai_offshore",
    )
    assert metrics.is_statistically_reliable is True
    assert metrics.sample_count == 10
    assert pytest.approx(metrics.mae, abs=1e-3) == 0.04
    assert pytest.approx(metrics.rmse, abs=1e-3) == 0.04
    assert pytest.approx(metrics.bias, abs=1e-3) == 0.04
    assert pytest.approx(metrics.persistence_rmse, abs=1e-3) == 0.10
    # RMSE Skill = (1 - 0.04 / 0.10) * 100 = +60.0%
    assert pytest.approx(metrics.rmse_skill_pct, abs=0.1) == 60.0
    assert metrics.error_distribution.median is not None


def test_insufficient_sample_size_protection(temp_verification_service):
    """If verified sample count < MIN_RELIABLE_SAMPLES, metrics must be withheld."""
    svc = temp_verification_service
    base = datetime.now(timezone.utc) - timedelta(hours=2)

    # Archive and verify only 2 samples (less than minimum 5)
    for i in range(2):
        origin = base + timedelta(hours=i)
        target = origin + timedelta(hours=1)
        svc.archive_forecast(
            ForecastArchivalRequest(
                forecast_id=f"fc_insuf_{i}",
                location_id="mumbai_offshore",
                forecast_origin_time=origin,
                target_timestamp=target,
                forecast_horizon=1,
                current_swh=1.0,
                predicted_swh=1.1,
                model_architecture="Residual WaveLSTM",
                input_data_timestamp=origin,
            )
        )
        svc.ingest_observation(
            ObservationIngestRequest(
                location_id="mumbai_offshore",
                observation_time=target,
                observed_swh=1.05,
            )
        )

    metrics = svc.calculate_window_metrics(horizon=1, window=VerificationWindow.ALL_TIME)
    assert metrics.is_statistically_reliable is False
    assert metrics.sample_count == 2
    assert metrics.mae is None
    assert metrics.rmse is None
    assert "Insufficient operational sample" in metrics.reliability_notice


# =============================================================================
# 5. Severe-Event Monitoring (>= 4.0m) & Regional Isolation
# =============================================================================

def test_severe_wave_monitoring_and_historical_distinction(temp_verification_service):
    svc = temp_verification_service
    origin = datetime(2026, 11, 16, 6, 0, tzinfo=timezone.utc)
    target = origin + timedelta(hours=6)

    # Archive a severe event: observed = 4.30m, predicted = 3.90m (underprediction)
    svc.archive_forecast(
        ForecastArchivalRequest(
            forecast_id="fc_severe_01",
            location_id="mumbai_offshore",
            forecast_origin_time=origin,
            target_timestamp=target,
            forecast_horizon=6,
            current_swh=4.20,
            predicted_swh=3.90,  # Misses the >= 4.0m severe threshold
            model_architecture="Residual WaveANN",
            input_data_timestamp=origin,
        )
    )

    svc.ingest_observation(
        ObservationIngestRequest(
            location_id="mumbai_offshore",
            observation_time=target,
            observed_swh=4.30,  # Actual severe event
        )
    )

    severe_stats = svc.calculate_severe_event_metrics(location_id="mumbai_offshore")
    assert severe_stats.actual_severe_events_count == 1
    assert severe_stats.missed_severe_events_count == 1
    assert severe_stats.underprediction_frequency_pct == 100.0
    assert severe_stats.historical_validation_sample_size == 22
    assert severe_stats.operational_severe_status == "INSUFFICIENT_OPERATIONAL_SAMPLE"
    assert "HISTORICAL BENCHMARK NOTICE" in severe_stats.scientific_notice


def test_regional_isolation_in_verification(temp_verification_service):
    """Forecasts for Mumbai Offshore must not match observations from Chennai Port."""
    svc = temp_verification_service
    now = datetime.now(timezone.utc)
    target = now + timedelta(hours=1)

    svc.archive_forecast(
        ForecastArchivalRequest(
            forecast_id="fc_mumbai",
            location_id="mumbai_offshore",
            forecast_origin_time=now,
            target_timestamp=target,
            forecast_horizon=1,
            current_swh=1.0,
            predicted_swh=1.1,
            model_architecture="Residual WaveLSTM",
            input_data_timestamp=now,
        )
    )

    # Ingest observation for chennai_port at identical timestamp
    svc.ingest_observation(
        ObservationIngestRequest(
            location_id="chennai_port",
            observation_time=target,
            observed_swh=1.15,
        )
    )

    # Mumbai forecast must remain pending
    dash = svc.get_verification_dashboard(location_id="mumbai_offshore")
    assert dash.total_verified_observations == 0
    assert dash.total_pending_verifications == 1


# =============================================================================
# 6. REST API Endpoint Integration Tests
# =============================================================================

def test_api_verification_dashboard_endpoint(test_client):
    res = test_client.get("/api/verification/dashboard")
    assert res.status_code == 200
    data = res.json()
    assert "total_forecasts_issued" in data
    assert "horizon_metrics" in data
    assert "1h" in data["horizon_metrics"]
    assert "6h" in data["horizon_metrics"]
    assert "24h" in data["horizon_metrics"]
    assert "severe_events" in data
    assert "historical_vs_operational_summary" in data


def test_api_archive_forecast_endpoint(test_client):
    now = datetime.now(timezone.utc)
    payload = {
        "location_id": "mumbai_offshore",
        "forecast_origin_time": now.isoformat(),
        "target_timestamp": (now + timedelta(hours=1)).isoformat(),
        "forecast_horizon": 1,
        "current_swh": 1.25,
        "predicted_swh": 1.30,
        "model_architecture": "Residual WaveLSTM",
        "input_data_timestamp": now.isoformat(),
        "data_source": "OpenMeteoProvider",
    }
    res = test_client.post("/api/verification/archive-forecast", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "forecast_id" in data


def test_api_ingest_observation_endpoint(test_client):
    now = datetime.now(timezone.utc)
    payload = {
        "location_id": "mumbai_offshore",
        "observation_time": now.isoformat(),
        "observed_swh": 1.28,
        "source_name": "INCOIS_BUOY_TEST",
        "source_type": "in_situ_buoy",
        "quality_status": "valid",
    }
    res = test_client.post("/api/verification/ingest-observation", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "observation_id" in data


def test_api_metrics_endpoint(test_client):
    res = test_client.get("/api/verification/metrics?horizon=6&window=all_time")
    assert res.status_code == 200
    data = res.json()
    assert data["horizon_hours"] == 6
    assert data["window"] == "all_time"
    assert "sample_count" in data
    assert "is_statistically_reliable" in data


def test_missing_observation_status_transition(temp_verification_service):
    """Forecasts past target time by >3 hours with no observations transition to OBSERVATION_UNAVAILABLE."""
    svc = temp_verification_service
    origin = datetime(2026, 11, 10, 0, 0, tzinfo=timezone.utc)
    target = origin + timedelta(hours=6)

    svc.archive_forecast(
        ForecastArchivalRequest(
            forecast_id="fc_missing_test",
            location_id="mumbai_offshore",
            forecast_origin_time=origin,
            target_timestamp=target,
            forecast_horizon=6,
            current_swh=1.20,
            predicted_swh=1.35,
            model_architecture="Residual WaveANN",
            input_data_timestamp=origin,
        )
    )

    # Trigger matching without ingesting observations
    svc.match_and_verify(location_id="mumbai_offshore")

    dash = svc.get_verification_dashboard(location_id="mumbai_offshore")
    rec = next(r for r in dash.recent_verifications if r.forecast_id == "fc_missing_test")
    assert rec.status == VerificationRecordStatus.OBSERVATION_UNAVAILABLE
    assert "without matching telemetry" in rec.status_reason


def test_delayed_observation_retroactive_resolution(temp_verification_service):
    """A delayed observation arriving after deadline resolves previously UNAVAILABLE forecast to VERIFIED."""
    svc = temp_verification_service
    origin = datetime(2026, 11, 10, 0, 0, tzinfo=timezone.utc)
    target = origin + timedelta(hours=1)

    svc.archive_forecast(
        ForecastArchivalRequest(
            forecast_id="fc_delayed_test",
            location_id="mumbai_offshore",
            forecast_origin_time=origin,
            target_timestamp=target,
            forecast_horizon=1,
            current_swh=1.00,
            predicted_swh=1.15,
            model_architecture="Residual WaveLSTM",
            input_data_timestamp=origin,
        )
    )

    # Mark as unavailable
    svc.match_and_verify(location_id="mumbai_offshore")
    dash1 = svc.get_verification_dashboard(location_id="mumbai_offshore")
    assert any(r.forecast_id == "fc_delayed_test" and r.status == VerificationRecordStatus.OBSERVATION_UNAVAILABLE for r in dash1.recent_verifications)

    # Ingest delayed observation matching target
    svc.ingest_observation(
        ObservationIngestRequest(
            location_id="mumbai_offshore",
            observation_time=target,
            observed_swh=1.10,
            source_type="in_situ_buoy",
        )
    )

    dash2 = svc.get_verification_dashboard(location_id="mumbai_offshore")
    rec = next(r for r in dash2.recent_verifications if r.forecast_id == "fc_delayed_test")
    assert rec.status == VerificationRecordStatus.VERIFIED
    assert rec.observed_swh == 1.10
    assert pytest.approx(rec.forecast_error, abs=1e-4) == 0.05


def test_api_seed_demo_observations(test_client):
    res = test_client.post("/api/verification/seed-demo-observations?count=10")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["observations_ingested"] == 10

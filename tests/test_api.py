"""
API Integration Tests for OceanCast AI FastAPI Backend.
"""

from fastapi.testclient import TestClient
import pytest

from backend.main import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "OceanCast AI" in data["service"]
    assert data["documentation"] == "/docs"


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["backend_online"] is True
    assert data["checkpoints_available"] == 9
    assert data["total_checkpoints"] == 9
    assert data["scaler_available"] is True
    assert data["dataset_available"] is True
    assert len(data["components"]) >= 4


def test_metadata_endpoint():
    response = client.get("/api/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["project_name"] == "OceanCast AI"
    assert data["selected_coordinates"]["latitude"] == 18.5
    assert data["selected_coordinates"]["longitude"] == 72.5
    assert data["supported_horizons"] == [1, 6, 24]
    assert "WaveLSTM" in data["best_overall_model"]


def test_multi_horizon_metrics():
    response = client.get("/api/metrics/multi-horizon")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) >= 12  # 4 models * 3 horizons
    assert "best_by_horizon" in data
    # Check 6h best is WaveLSTM
    assert "WaveLSTM" in data["best_by_horizon"]["6"]["best_neural_net"]["model"]
    # Check 1h best persistence
    assert "Persistence" in data["best_by_horizon"]["1"]["best_on_rmse"]["model"]


def test_metrics_filtered_by_horizon():
    for h in [1, 6, 24]:
        resp = client.get(f"/api/metrics?horizon={h}")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 4
        assert all(item["horizon_hours"] == h for item in items)

    # Invalid horizon
    bad_resp = client.get("/api/metrics?horizon=12")
    assert bad_resp.status_code == 400


def test_detailed_metrics():
    resp = client.get("/api/metrics/detailed/6")
    assert resp.status_code == 200
    data = resp.json()
    assert data["horizon_hours"] == 6
    assert len(data["models"]) >= 3


def test_forecast_status():
    resp = client.get("/api/forecast/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["demo_mode_available"] is True
    assert all(data["checkpoints_ready"].values())


def test_forecast_demo_mode():
    payload = {
        "horizon": 1,
        "model": "all",
        "source": "demo",
        "latitude": 18.50,
        "longitude": 72.50,
    }
    resp = client.post("/api/forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["forecast_horizon_hours"] == 1
    assert data["is_demo"] is True
    assert "lstm" in data["predictions"]
    assert "ann" in data["predictions"]
    assert "cnn1d" in data["predictions"]
    assert len(data["observation_window"]) == 24
    assert data["persistence_swh_m"] > 0.0


def test_historical_summary():
    resp = client.get("/api/historical/summary?horizon=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_test_samples"] == 19455
    assert data["overall_min_swh"] > 0.0
    assert data["overall_max_swh"] > 3.0


def test_historical_predictions_downsampled():
    resp = client.get("/api/historical/predictions?horizon=1&downsample=100")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_samples"] == 19455
    assert len(data["points"]) <= 110


def test_historical_scatter():
    resp = client.get("/api/historical/scatter?horizon=6&model=lstm&sample_size=50")
    assert resp.status_code == 200
    points = resp.json()
    assert len(points) <= 60
    assert "actual" in points[0]
    assert "predicted" in points[0]


def test_historical_residuals():
    resp = client.get("/api/historical/residuals?horizon=6&model=lstm&num_bins=15")
    assert resp.status_code == 200
    bins = resp.json()
    assert len(bins) == 15
    assert sum(b["count"] for b in bins) == 19455

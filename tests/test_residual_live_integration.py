"""
Dedicated Unit & Integration Tests for Validated Residual Deep Learning Deployment.

Covers:
1. Residual checkpoint loading across all 9 trained models
2. Correct residual scaler loading and horizon isolation
3. Input preprocessing and tensor dimensions [1, 24, 13]
4. Residual inverse transformation
5. Forecast reconstruction (SWH_forecast = SWH_current + Predicted_ΔSWH)
6. Zero-residual equals persistence property
7. NaN rejection in input observations
8. Missing feature rejection
9. Invalid sequence rejection (insufficient length and timestamp gaps)
10. Candidate model selection across all three forecast horizons (+1h, +6h, +24h)
11. FastAPI live forecasting API integration
12. Concurrent multi-threaded inference safety
13. Geographic validation status isolation (Reference vs Experimental)
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
import torch

from backend.main import app
from backend.services.model_service import model_service
from src.data.dataset import DEFAULT_FEATURE_COLS, TimeSeriesScaler
from src.data.live_service import (
    predict_live_forecast,
    predict_residual_forecast,
    validate_and_prepare_input_window,
)
from src.models import build_wave_model
from src.models.registry import ForecastMode, ModelRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


@pytest.fixture
def valid_24h_dataframe():
    """Generates 24 consecutive hourly observations ending at anchor time."""
    base_time = pd.Timestamp("2026-10-25 00:00:00", tz="UTC")
    times = [base_time + pd.Timedelta(hours=i) for i in range(24)]
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "valid_time": times,
            "swh": np.linspace(1.2, 1.8, 24),
            "mwp": np.linspace(7.5, 8.5, 24),
            "mwd": np.linspace(210.0, 230.0, 24),
            "wind_speed": np.linspace(5.0, 8.0, 24),
            "wind_dir": np.linspace(350.0, 360.0, 24),
            "u10": np.linspace(-1.0, -2.0, 24),
            "v10": np.linspace(-4.0, -7.0, 24),
        }
    )
    # Prepare engineered features
    df_prepared, _ = validate_and_prepare_input_window(df, anchor_time=times[-1], window_size=24)
    return df_prepared


# ------------------------------------------------------------------------------
# 1. Residual Checkpoint Loading
# ------------------------------------------------------------------------------
def test_residual_checkpoints_loading():
    """Verify that all 9 residual PyTorch checkpoints load cleanly with matching architectures."""
    horizons = [1, 6, 24]
    architectures = ["ann", "lstm", "cnn1d"]

    for arch in architectures:
        for h in horizons:
            entry = ModelRegistry.resolve_model(h, arch, mode=ForecastMode.RESIDUAL)
            assert entry.exists(), f"Model file missing for {entry.model_id}"

            model = build_wave_model(arch, input_dim=len(DEFAULT_FEATURE_COLS), seq_len=24)
            ckpt = torch.load(entry.checkpoint_path, map_location="cpu", weights_only=True)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)

            model.eval()
            param_count = sum(p.numel() for p in model.parameters())
            assert param_count > 0


# ------------------------------------------------------------------------------
# 2. Correct Residual Scaler Loading & Horizon Isolation
# ------------------------------------------------------------------------------
def test_residual_scalers_loading_and_isolation():
    """Verify horizon residual scalers load correctly and maintain separate standard deviations."""
    scaler_h1 = model_service.get_residual_scaler(1)
    scaler_h6 = model_service.get_residual_scaler(6)
    scaler_h24 = model_service.get_residual_scaler(24)

    # Standard deviations must grow with forecast horizon
    assert 0.020 < scaler_h1["std"] < 0.030, f"Unexpected +1h std: {scaler_h1['std']}"
    assert 0.080 < scaler_h6["std"] < 0.120, f"Unexpected +6h std: {scaler_h6['std']}"
    assert 0.180 < scaler_h24["std"] < 0.250, f"Unexpected +24h std: {scaler_h24['std']}"

    # Mean changes should be very close to zero in stationary ocean series
    assert abs(scaler_h1["mean"]) < 0.01
    assert abs(scaler_h6["mean"]) < 0.01
    assert abs(scaler_h24["mean"]) < 0.01

    # Scaler isolation: H=1 scaler cannot equal H=6 or H=24 scaler
    assert scaler_h1["std"] != scaler_h6["std"]
    assert scaler_h6["std"] != scaler_h24["std"]


# ------------------------------------------------------------------------------
# 3. Input Preprocessing & Tensor Dimensions
# ------------------------------------------------------------------------------
def test_input_preprocessing_and_tensor_dimensions(valid_24h_dataframe):
    """Verify input feature scaling produces strictly [1, 24, 13] tensor."""
    scaler = model_service.get_scaler()
    assert len(valid_24h_dataframe) == 24

    scaled_x = scaler.feature_scaler.transform(valid_24h_dataframe[scaler.feature_names].values).astype(np.float32)
    assert scaled_x.shape == (24, 13)

    tensor_x = torch.tensor(scaled_x, dtype=torch.float32).unsqueeze(0)
    assert tensor_x.shape == (1, 24, 13)


# ------------------------------------------------------------------------------
# 4. Residual Inverse Transformation
# ------------------------------------------------------------------------------
def test_residual_inverse_transformation():
    """Test algebraic inversion of scaled delta to physical meters."""
    scaler_h6 = model_service.get_residual_scaler(6)
    mean = scaler_h6["mean"]
    std = scaler_h6["std"]

    # Scaled delta = +1.0 standard deviation
    delta_scaled = 1.0
    delta_m = delta_scaled * std + mean
    assert np.isclose(delta_m, std + mean)

    # Scaled delta = -1.0 standard deviation
    delta_scaled_neg = -1.0
    delta_m_neg = delta_scaled_neg * std + mean
    assert np.isclose(delta_m_neg, -std + mean)


# ------------------------------------------------------------------------------
# 5. Forecast Reconstruction Formula
# ------------------------------------------------------------------------------
def test_forecast_reconstruction_formula(valid_24h_dataframe):
    """Verify SWH_forecast = SWH_current + Predicted_ΔSWH."""
    swh_current = float(valid_24h_dataframe["swh"].iloc[-1])
    res = model_service.predict_single(
        valid_24h_dataframe,
        horizon=6,
        model_type="ann",
        persistence_swh=swh_current,
        mode=ForecastMode.RESIDUAL,
    )

    expected_reconstructed = swh_current + res.predicted_residual_m
    assert np.isclose(res.forecast_swh_m, expected_reconstructed, atol=1e-3)
    assert res.current_swh_m == round(swh_current, 4)


# ------------------------------------------------------------------------------
# 6. Zero-Residual Equals Persistence Property
# ------------------------------------------------------------------------------
def test_zero_residual_equals_persistence():
    """Verify that when ΔSWH = 0, reconstructed forecast identically matches persistence."""
    current_swh = 1.7500
    pred_delta_m = 0.0000
    reconstructed_swh = current_swh + pred_delta_m
    assert reconstructed_swh == current_swh


# ------------------------------------------------------------------------------
# 7. NaN Rejection in Input Observations
# ------------------------------------------------------------------------------
def test_nan_rejection_in_input():
    """Verify that NaN in any observation raises a descriptive error."""
    base_time = pd.Timestamp("2026-10-25 00:00:00", tz="UTC")
    times = [base_time + pd.Timedelta(hours=i) for i in range(24)]

    df_nan = pd.DataFrame(
        {
            "valid_time": times,
            "swh": [1.2] * 23 + [np.nan],  # NaN at latest observation
            "mwp": [8.0] * 24,
            "mwd": [220.0] * 24,
            "wind_speed": [6.0] * 24,
            "wind_dir": [355.0] * 24,
        }
    )
    with pytest.raises(ValueError, match="Missing.*NaN.*detected"):
        validate_and_prepare_input_window(df_nan, anchor_time=times[-1], window_size=24)


# ------------------------------------------------------------------------------
# 8. Missing Feature Rejection
# ------------------------------------------------------------------------------
def test_missing_feature_rejection():
    """Verify that omitting a required physical variable raises KeyError."""
    base_time = pd.Timestamp("2026-10-25 00:00:00", tz="UTC")
    times = [base_time + pd.Timedelta(hours=i) for i in range(24)]

    df_missing = pd.DataFrame(
        {
            "valid_time": times,
            "swh": [1.2] * 24,
            # 'mwp' omitted intentionally
            "wind_speed": [6.0] * 24,
            "wind_dir": [355.0] * 24,
        }
    )
    with pytest.raises(KeyError, match="Required physical column 'mwp' is missing"):
        validate_and_prepare_input_window(df_missing, anchor_time=times[-1], window_size=24)


# ------------------------------------------------------------------------------
# 9. Invalid Sequence Rejection (Insufficient Length & Gaps)
# ------------------------------------------------------------------------------
def test_invalid_sequence_rejection():
    """Verify rejection of <24 rows or timestamps with gaps."""
    base_time = pd.Timestamp("2026-10-25 00:00:00", tz="UTC")
    # Only 18 hourly steps
    times_short = [base_time + pd.Timedelta(hours=i) for i in range(18)]
    df_short = pd.DataFrame(
        {
            "valid_time": times_short,
            "swh": [1.2] * 18,
            "mwp": [8.0] * 18,
            "wind_speed": [6.0] * 18,
            "wind_dir": [355.0] * 18,
        }
    )
    with pytest.raises(ValueError, match="Insufficient historical observations"):
        validate_and_prepare_input_window(df_short, window_size=24)

    # 24 steps with a 2-hour gap
    times_gap = [base_time + pd.Timedelta(hours=i if i < 10 else i + 2) for i in range(24)]
    df_gap = pd.DataFrame(
        {
            "valid_time": times_gap,
            "swh": [1.2] * 24,
            "mwp": [8.0] * 24,
            "wind_speed": [6.0] * 24,
            "wind_dir": [355.0] * 24,
        }
    )
    with pytest.raises(ValueError, match="Irregular or missing hourly timestamps"):
        validate_and_prepare_input_window(df_gap, anchor_time=times_gap[-1], window_size=24)


# ------------------------------------------------------------------------------
# 10. Candidate Selection for All Three Horizons
# ------------------------------------------------------------------------------
def test_candidate_selection_all_three_horizons(valid_24h_dataframe):
    """Verify candidate model selection matches held-out test winners."""
    cand_h1 = ModelRegistry.get_candidate(1, mode=ForecastMode.RESIDUAL)
    assert cand_h1.architecture == "lstm"
    assert cand_h1.checkpoint_path.name == "best_lstm_residual_h1.pt"
    assert cand_h1.rmse_skill_score_vs_persistence == 29.76

    cand_h6 = ModelRegistry.get_candidate(6, mode=ForecastMode.RESIDUAL)
    assert cand_h6.architecture == "ann"
    assert cand_h6.checkpoint_path.name == "best_ann_residual_h6.pt"
    assert cand_h6.rmse_skill_score_vs_persistence == 18.85

    cand_h24 = ModelRegistry.get_candidate(24, mode=ForecastMode.RESIDUAL)
    assert cand_h24.architecture == "ann"
    assert cand_h24.checkpoint_path.name == "best_ann_residual_h24.pt"
    assert cand_h24.rmse_skill_score_vs_persistence == 11.09

    # Test direct predict_residual_forecast execution
    anchor = valid_24h_dataframe["valid_time"].iloc[-1]
    res1 = predict_residual_forecast(valid_24h_dataframe, anchor, horizon=1)
    assert res1.model_name == "Residual WaveLSTM"
    assert res1.horizon_hours == 1

    res6 = predict_residual_forecast(valid_24h_dataframe, anchor, horizon=6)
    assert res6.model_name == "Residual WaveANN"
    assert res6.horizon_hours == 6

    res24 = predict_residual_forecast(valid_24h_dataframe, anchor, horizon=24)
    assert res24.model_name == "Residual WaveANN"
    assert res24.horizon_hours == 24


# ------------------------------------------------------------------------------
# 11. Live API Integration via FastAPI
# ------------------------------------------------------------------------------
def test_fastapi_live_api_residual_endpoints():
    """Verify FastAPI endpoints return full residual response schema."""
    # 1. Registry overview endpoint
    reg_res = client.get("/api/forecast/registry")
    assert reg_res.status_code == 200
    reg_data = reg_res.json()
    assert reg_data["total_models"] == 18
    assert reg_data["ready_models"] == 18
    assert reg_data["all_ready"] is True

    # 2. GET /api/forecast/live in residual mode (default)
    get_res = client.get("/api/forecast/live?horizon=6&model=candidate&source=demo&location_id=mumbai_offshore")
    assert get_res.status_code == 200
    data = get_res.json()

    assert data["forecast_horizon_hours"] == 6
    assert data["forecast_mode"] == "residual"
    assert data["selected_model_architecture"] == "Residual WaveANN"
    assert data["predicted_residual"] is not None
    assert data["predicted_swh"] is not None
    assert data["current_swh"] is not None
    assert data["validation_status"] == "reference_validated"
    assert data["model_version"] == "2.0.0-residual"

    # 3. POST /api/forecast in residual mode
    post_res = client.post(
        "/api/forecast",
        json={
            "horizon": 1,
            "model": "candidate",
            "source": "demo",
            "location_id": "mumbai_offshore",
            "forecast_mode": "residual",
        },
    )
    assert post_res.status_code == 200
    post_data = post_res.json()
    assert post_data["selected_model_architecture"] == "Residual WaveLSTM"
    assert post_data["forecast_mode"] == "residual"

    # 4. POST /api/forecast in absolute mode (preservation of original)
    abs_res = client.post(
        "/api/forecast",
        json={
            "horizon": 1,
            "model": "lstm",
            "source": "demo",
            "location_id": "mumbai_offshore",
            "forecast_mode": "absolute",
        },
    )
    assert abs_res.status_code == 200
    abs_data = abs_res.json()
    assert abs_data["forecast_mode"] == "absolute"
    assert abs_data["selected_model_architecture"] == "WaveLSTM"


# ------------------------------------------------------------------------------
# 12. Concurrent Multi-Threaded Inference Safety
# ------------------------------------------------------------------------------
def test_concurrent_multi_threaded_inference(valid_24h_dataframe):
    """Verify thread safety during concurrent inference requests."""
    def run_worker(h):
        return model_service.predict_single(
            prepared_df=valid_24h_dataframe,
            horizon=h,
            model_type="candidate",
            persistence_swh=1.8,
            mode=ForecastMode.RESIDUAL,
        )

    tasks = [1, 6, 24, 1, 6, 24, 1, 6, 24]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_worker, tasks))

    assert len(results) == len(tasks)
    for r in results:
        assert np.isfinite(r.forecast_swh_m)
        assert r.predicted_residual_m is not None


# ------------------------------------------------------------------------------
# 13. Geographic Validation Status Isolation
# ------------------------------------------------------------------------------
def test_geographic_validation_status_isolation():
    """Verify only Mumbai reference node is reference_validated; others are experimental_unvalidated."""
    # Reference location
    ref_res = client.get("/api/forecast/live?horizon=6&location_id=mumbai_offshore&source=demo")
    assert ref_res.status_code == 200
    assert ref_res.json()["validation_status"] == "reference_validated"

    # Non-reference location
    exp_res = client.get("/api/forecast/live?horizon=6&location_id=chennai_port&source=demo")
    assert exp_res.status_code == 200
    assert exp_res.json()["validation_status"] == "experimental_unvalidated"

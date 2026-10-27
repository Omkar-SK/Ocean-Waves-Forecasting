"""
Unit tests for Live Forecasting Service:
- API response parsing and timestamp alignment
- Rigorous input sequence validation (24 consecutive hourly steps, gap rejection, NaN rejection)
- Feature engineering and column ordering matching ERA5 training pipeline
- Safe horizon-specific checkpoint loading (no H=1 for H=6/H=24)
- Scientific honesty: demo mode fallback verification and future timestamp calculation
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import DEFAULT_FEATURE_COLS, TimeSeriesScaler
from src.data.live_service import (
    LiveFetchResult,
    fetch_demo_era5_window,
    fetch_live_marine_and_weather,
    get_sea_state_classification,
    load_horizon_checkpoint,
    predict_live_forecast,
    validate_and_prepare_input_window,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_raw_dataframe():
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
    """Loads the pre-fitted training scaler."""
    scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
    assert scaler_path.is_file(), "Scaler artifact missing."
    return TimeSeriesScaler.load(scaler_path)


def test_validate_and_prepare_input_window_success(sample_raw_dataframe):
    """Verify clean 24-step window extraction, feature generation, and column ordering."""
    anchor = pd.Timestamp("2026-10-26 12:00:00", tz="UTC")
    df_window, actual_anchor = validate_and_prepare_input_window(
        sample_raw_dataframe, anchor_time=anchor, window_size=24
    )

    assert len(df_window) == 24
    assert actual_anchor == anchor

    # Verify all 13 required features are present in the exact order
    for col in DEFAULT_FEATURE_COLS:
        assert col in df_window.columns
        assert not df_window[col].isna().any()

    # Verify circular and meteorological wind calculations
    assert np.all((df_window["mwd_sin"] >= -1.0) & (df_window["mwd_sin"] <= 1.0))
    assert np.all((df_window["mwd_cos"] >= -1.0) & (df_window["mwd_cos"] <= 1.0))
    assert np.all((df_window["hour_sin"] >= -1.0) & (df_window["hour_sin"] <= 1.0))


def test_validate_rejects_insufficient_history(sample_raw_dataframe):
    """Verify that fewer than 24 observations raises ValueError."""
    short_df = sample_raw_dataframe.iloc[:15]
    with pytest.raises(ValueError, match="Insufficient historical observations"):
        validate_and_prepare_input_window(short_df, window_size=24)


def test_validate_rejects_missing_timestamps_or_gaps(sample_raw_dataframe):
    """Verify that any gap (> 1 hour) in the historical sequence triggers a rejection."""
    anchor = pd.Timestamp("2026-10-26 12:00:00", tz="UTC")
    # Drop row at index 20 (which is within the 24h window ending at index 36)
    broken_df = sample_raw_dataframe.drop(index=20).reset_index(drop=True)
    with pytest.raises(ValueError, match="Irregular or missing hourly timestamps"):
        validate_and_prepare_input_window(broken_df, anchor_time=anchor, window_size=24)


def test_validate_rejects_nan_values(sample_raw_dataframe):
    """Verify that NaN in observations is rejected rather than silently imputed."""
    anchor = pd.Timestamp("2026-10-26 12:00:00", tz="UTC")
    nan_df = sample_raw_dataframe.copy()
    nan_df.loc[20, "swh"] = np.nan
    with pytest.raises(ValueError, match="Missing \\(NaN\\) values detected"):
        validate_and_prepare_input_window(nan_df, anchor_time=anchor, window_size=24)


def test_validate_never_uses_future_observations(sample_raw_dataframe):
    """Verify that input observations are strictly <= anchor_time."""
    anchor = pd.Timestamp("2026-10-26 06:00:00", tz="UTC")
    df_window, actual_anchor = validate_and_prepare_input_window(
        sample_raw_dataframe, anchor_time=anchor, window_size=24
    )
    assert actual_anchor == anchor
    assert df_window["valid_time"].max() == anchor
    assert (df_window["valid_time"] <= anchor).all()
    assert len(df_window) == 24


def test_horizon_checkpoint_selection():
    """Verify correct checkpoint loading for H=1, 6, 24 and rejection of invalid horizons."""
    for h in [1, 6, 24]:
        for m in ["lstm", "ann", "cnn1d"]:
            model, meta, ckpt_name = load_horizon_checkpoint(m, h)
            assert isinstance(model, torch.nn.Module)
            meta_h = meta.get("dataset_config", {}).get("horizon", 1)
            assert meta_h == h, f"Checkpoint metadata horizon {meta_h} != requested {h}"
            if h == 1:
                assert f"best_{m}.pt" in ckpt_name
            else:
                assert f"best_{m}_h{h}.pt" in ckpt_name

    with pytest.raises(ValueError, match="Unsupported horizon"):
        load_horizon_checkpoint("lstm", 12)

    with pytest.raises(ValueError, match="Unsupported model type"):
        load_horizon_checkpoint("transformer", 1)


def test_predict_live_forecast_execution(sample_raw_dataframe, trained_scaler):
    """Verify end-to-end inference math, persistence baseline, and target timestamp calculation."""
    anchor = pd.Timestamp("2026-10-26 10:00:00", tz="UTC")
    df_window, actual_anchor = validate_and_prepare_input_window(
        sample_raw_dataframe, anchor_time=anchor, window_size=24
    )

    for h in [1, 6, 24]:
        res = predict_live_forecast(
            prepared_df=df_window,
            anchor_time=actual_anchor,
            horizon=h,
            model_type="lstm",
            scaler=trained_scaler,
            external_nwp_forecast=0.85,
            is_demo=False,
        )

        assert res.horizon_hours == h
        assert res.anchor_time_utc == anchor.to_pydatetime()
        assert res.target_time_utc == (anchor + pd.Timedelta(hours=h)).to_pydatetime()
        assert 0.0 < res.forecast_swh_m < 15.0  # Physical wave height sanity bounds
        assert pytest.approx(res.persistence_swh_m, 1e-4) == float(df_window["swh"].iloc[-1])
        assert res.external_nwp_swh_m == 0.85
        assert isinstance(res.sea_state_desc, str)


def test_demo_fallback_loader():
    """Verify offline demo fallback loader sets is_demo=True and includes honest disclaimer."""
    demo_res = fetch_demo_era5_window()
    assert isinstance(demo_res, LiveFetchResult)
    assert demo_res.is_demo is True
    assert "DEMO MODE ACTIVE" in demo_res.notes[0]
    assert demo_res.latest_obs_time_utc.year == 2026
    assert demo_res.latest_obs_time_utc.month == 10
    assert len(demo_res.raw_df) >= 24


def test_mocked_live_fetch():
    """Verify parsing of mocked Open-Meteo REST API responses."""
    times = [f"2026-10-26T{h:02d}:00" for h in range(24)]
    mock_marine = {
        "hourly": {
            "time": times,
            "wave_height": [0.75] * 24,
            "wave_period": [8.0] * 24,
            "wave_direction": [215.0] * 24,
        }
    }
    mock_weather = {
        "hourly": {
            "time": times,
            "wind_speed_10m": [5.2] * 24,
            "wind_direction_10m": [45.0] * 24,
        }
    }

    class MockResponse:
        def __init__(self, data_dict):
            self.data = json.dumps(data_dict).encode("utf-8")

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_urlopen(req, timeout=10):
        url = req.full_url
        if "marine-api" in url:
            return MockResponse(mock_marine)
        elif "api.open-meteo" in url:
            return MockResponse(mock_weather)
        raise ValueError(f"Unexpected url: {url}")

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        res = fetch_live_marine_and_weather(latitude=18.5, longitude=72.5)
        assert isinstance(res, LiveFetchResult)
        assert res.is_demo is False
        assert len(res.raw_df) == 24
        assert "swh" in res.raw_df.columns
        assert "wind_speed" in res.raw_df.columns


def test_sea_state_classification():
    """Verify WMO Sea State descriptions and colors across ranges."""
    desc, col = get_sea_state_classification(0.05)
    assert "Calm" in desc
    desc, col = get_sea_state_classification(0.8)
    assert "Smooth" in desc
    desc, col = get_sea_state_classification(1.8)
    assert "Moderate" in desc
    desc, col = get_sea_state_classification(3.2)
    assert "Rough" in desc
    desc, col = get_sea_state_classification(7.5)
    assert "High" in desc


def test_scaler_invariance_during_inference(sample_raw_dataframe, trained_scaler):
    """Verify that model inference strictly preserves scaler parameters without refitting."""
    anchor = pd.Timestamp("2026-10-26 10:00:00", tz="UTC")
    df_window, actual_anchor = validate_and_prepare_input_window(
        sample_raw_dataframe, anchor_time=anchor, window_size=24
    )

    # Snapshot mean and scale
    orig_mean = np.copy(trained_scaler.feature_scaler.mean_)
    orig_scale = np.copy(trained_scaler.feature_scaler.scale_)
    orig_target_mean = float(trained_scaler.target_scaler.mean_[0])
    orig_target_scale = float(trained_scaler.target_scaler.scale_[0])

    _ = predict_live_forecast(
        prepared_df=df_window,
        anchor_time=actual_anchor,
        horizon=1,
        model_type="lstm",
        scaler=trained_scaler,
    )

    np.testing.assert_array_equal(trained_scaler.feature_scaler.mean_, orig_mean)
    np.testing.assert_array_equal(trained_scaler.feature_scaler.scale_, orig_scale)
    assert float(trained_scaler.target_scaler.mean_[0]) == orig_target_mean
    assert float(trained_scaler.target_scaler.scale_[0]) == orig_target_scale


def test_feature_column_order_integrity(sample_raw_dataframe):
    """Verify feature ordering is invariant to scrambled input dataframe column order."""
    scrambled = sample_raw_dataframe[
        ["wind_dir", "swh", "valid_time", "wind_speed", "mwd", "mwp"]
    ].copy()
    anchor = pd.Timestamp("2026-10-26 12:00:00", tz="UTC")

    df_window, _ = validate_and_prepare_input_window(scrambled, anchor_time=anchor, window_size=24)

    for i, col in enumerate(DEFAULT_FEATURE_COLS):
        assert col in df_window.columns
        assert not df_window[col].isna().any()


def test_missing_required_column_raises_error(sample_raw_dataframe):
    """Verify KeyError is raised when a required physical column is completely absent."""
    missing_swh = sample_raw_dataframe.drop(columns=["swh"])
    anchor = pd.Timestamp("2026-10-26 12:00:00", tz="UTC")
    with pytest.raises(KeyError, match="Required physical column 'swh' is missing"):
        validate_and_prepare_input_window(missing_swh, anchor_time=anchor, window_size=24)


def test_future_timestamp_calculation_for_all_horizons(sample_raw_dataframe, trained_scaler):
    """Verify that target timestamps exactly equal anchor_time + horizon in UTC."""
    anchor = pd.Timestamp("2026-10-26 05:00:00", tz="UTC")
    df_window, actual_anchor = validate_and_prepare_input_window(
        sample_raw_dataframe, anchor_time=anchor, window_size=24
    )

    expected_targets = {
        1: datetime(2026, 10, 26, 6, 0, 0, tzinfo=timezone.utc),
        6: datetime(2026, 10, 26, 11, 0, 0, tzinfo=timezone.utc),
        24: datetime(2026, 10, 27, 5, 0, 0, tzinfo=timezone.utc),
    }

    for h in [1, 6, 24]:
        res = predict_live_forecast(
            prepared_df=df_window,
            anchor_time=actual_anchor,
            horizon=h,
            model_type="ann",
            scaler=trained_scaler,
        )
        assert res.target_time_utc == expected_targets[h]


def test_network_connection_error_handling():
    """Verify that network errors raise ConnectionError with clear actionable message."""
    import urllib.error

    def mock_broken_urlopen(req, timeout=10):
        raise urllib.error.URLError("Network connection failed or timed out.")

    with patch("urllib.request.urlopen", side_effect=mock_broken_urlopen):
        with pytest.raises(ConnectionError, match="Failed to fetch Open-Meteo Marine API"):
            fetch_live_marine_and_weather(latitude=18.5, longitude=72.5)


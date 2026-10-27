"""
Live Ocean Wave Forecasting Service:
- Ingestion from open-access marine & meteorological APIs (Open-Meteo Marine / ECMWF / GFS)
- Strict quality assurance: verification of 24 consecutive hourly observations, no NaNs, no gaps
- Leak-free feature alignment matching ERA5 training pipeline
- Inference using horizon-specific trained checkpoints (H=1, 6, 24) and pre-fitted training scaler
- Honest demarcation between real-time data, offline historical demo mode, and external NWP forecasts
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import torch

from src.data.dataset import DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL, TimeSeriesScaler
from src.data.features import add_cyclical_time_features, encode_circular_angle
from src.models import build_wave_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LiveFetchResult:
    """Encapsulates raw data fetched from real-time API or historical fallback demo."""
    raw_df: pd.DataFrame
    source_name: str
    attribution: str
    fetch_time_utc: datetime
    latest_obs_time_utc: datetime
    is_demo: bool = False
    external_nwp_forecasts: Dict[int, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class LiveForecastResult:
    """Encapsulates inference output from our trained deep learning models."""
    model_name: str
    horizon_hours: int
    checkpoint_file: str
    anchor_time_utc: datetime
    target_time_utc: datetime
    forecast_swh_m: float
    persistence_swh_m: float
    external_nwp_swh_m: Optional[float]
    sea_state_desc: str
    sea_state_color: str
    input_window_df: pd.DataFrame
    is_demo: bool
    source_info: Dict[str, Any]
    generated_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def fetch_live_marine_and_weather(
    latitude: float = 18.5,
    longitude: float = 72.5,
    timeout_s: int = 10,
    past_days: int = 2,
    forecast_days: int = 2,
) -> LiveFetchResult:
    """
    Fetch near-real-time marine wave conditions and 10m wind from Open-Meteo REST APIs.

    Sources:
    - Marine API: ECMWF WAM / Copernicus Marine Global Ocean Waves
    - Weather API: ECMWF IFS / GFS atmospheric wind components
    Licensing: Open Database License (ODbL) / CC BY 4.0

    Parameters
    ----------
    latitude : float
        Offshore target latitude (default 18.5° N).
    longitude : float
        Offshore target longitude (default 72.5° E).
    timeout_s : int
        Network request timeout in seconds.
    past_days : int
        Number of past days of hourly history to retrieve.
    forecast_days : int
        Number of future forecast days to retrieve for external NWP comparison.

    Returns
    -------
    LiveFetchResult
        Parsed raw observations and NWP forecast values.
    """
    headers = {"User-Agent": "WaveCast-DL-Forecasting/1.0 (Academic Project)"}

    # 1. Marine API Query
    marine_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "wave_height,wave_direction,wave_period",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    marine_url = f"https://marine-api.open-meteo.com/v1/marine?{urllib.parse.urlencode(marine_params)}"

    req_marine = urllib.request.Request(marine_url, headers=headers)
    try:
        with urllib.request.urlopen(req_marine, timeout=timeout_s) as resp:
            marine_json = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise ConnectionError(f"Failed to fetch Open-Meteo Marine API: {e}") from e

    # 2. Weather API Query (10m wind speed and direction in m/s)
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    weather_url = f"https://api.open-meteo.com/v1/forecast?{urllib.parse.urlencode(weather_params)}"

    req_weather = urllib.request.Request(weather_url, headers=headers)
    try:
        with urllib.request.urlopen(req_weather, timeout=timeout_s) as resp:
            weather_json = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise ConnectionError(f"Failed to fetch Open-Meteo Weather API: {e}") from e

    # 3. Parse and Align Timestamps
    df_marine = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(marine_json["hourly"]["time"], utc=True),
            "swh": marine_json["hourly"]["wave_height"],
            "mwp": marine_json["hourly"]["wave_period"],
            "mwd": marine_json["hourly"]["wave_direction"],
        }
    )

    df_weather = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(weather_json["hourly"]["time"], utc=True),
            "wind_speed": weather_json["hourly"]["wind_speed_10m"],
            "wind_dir": weather_json["hourly"]["wind_direction_10m"],
        }
    )

    merged = pd.merge(df_marine, df_weather, on="valid_time", how="inner").sort_values("valid_time").reset_index(drop=True)

    fetch_time_utc = datetime.now(timezone.utc)
    current_hour_utc = fetch_time_utc.replace(minute=0, second=0, microsecond=0)

    # Historical observations are strictly those at or before current time
    hist_mask = merged["valid_time"] <= current_hour_utc
    df_hist = merged[hist_mask]
    if df_hist.empty:
        raise ValueError("No historical observations found up to current UTC hour.")

    latest_obs_time = df_hist["valid_time"].max()

    # Extract external NWP wave forecasts for t+1, t+6, t+24
    external_nwp: Dict[int, float] = {}
    for h in [1, 6, 24]:
        target_t = latest_obs_time + pd.Timedelta(hours=h)
        match_row = merged[merged["valid_time"] == target_t]
        if not match_row.empty and not pd.isna(match_row["swh"].values[0]):
            external_nwp[h] = float(match_row["swh"].values[0])

    return LiveFetchResult(
        raw_df=merged,
        source_name="Open-Meteo Marine & Atmospheric API (ECMWF WAM / IFS Global Feeds)",
        attribution="Data provided by Open-Meteo under CC BY 4.0; underlying models by ECMWF & Copernicus Marine Service.",
        fetch_time_utc=fetch_time_utc,
        latest_obs_time_utc=latest_obs_time.to_pydatetime(),
        is_demo=False,
        external_nwp_forecasts=external_nwp,
        notes=[
            f"Location: {latitude:.2f}° N, {longitude:.2f}° E (Arabian Sea)",
            "Operational wave model: ECMWF Wave Action Model (WAM)",
            "Operational atmospheric model: ECMWF Integrated Forecasting System (IFS)",
        ],
    )


def fetch_demo_era5_window(
    window_size: int = 24,
    data_path: Optional[Union[str, Path]] = None,
) -> LiveFetchResult:
    """
    Offline fallback demo loader using genuine ERA5 historical observations.

    Strictly labeled as an offline historical demo, ensuring full scientific honesty
    when live internet connectivity is disabled or external APIs are unavailable.
    """
    if data_path is None:
        demo_csv = PROJECT_ROOT / "data" / "processed" / "era5_demo_window.csv"
        full_csv = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.csv"
        full_parquet = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"
        if demo_csv.is_file():
            data_path = demo_csv
        elif full_csv.is_file():
            data_path = full_csv
        elif full_parquet.is_file():
            data_path = full_parquet
        else:
            raise FileNotFoundError("No processed ERA5 dataset found in data/processed/")

    path = Path(data_path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df = df.sort_values("valid_time").reset_index(drop=True)

    latest_time = df["valid_time"].max()
    fetch_time_utc = datetime.now(timezone.utc)

    # Derive wind components if missing
    if "wind_speed" not in df.columns and "u10" in df.columns and "v10" in df.columns:
        df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    if "wind_dir" not in df.columns and "u10" in df.columns and "v10" in df.columns:
        df["wind_dir"] = (270.0 - np.rad2deg(np.arctan2(df["v10"].values, df["u10"].values))) % 360.0

    return LiveFetchResult(
        raw_df=df,
        source_name="Offline Demo Fallback: ERA5 Reanalysis Archive",
        attribution="ECMWF ERA5 Reanalysis Dataset (Historical Ground Truth Archive).",
        fetch_time_utc=fetch_time_utc,
        latest_obs_time_utc=latest_time.to_pydatetime(),
        is_demo=True,
        external_nwp_forecasts={},
        notes=[
            "DEMO MODE ACTIVE: Using historical ERA5 observation window ending 2026-10-17 23:00 UTC.",
            "This is NOT a real-time forecast. Provided for offline validation and pipeline inspection.",
        ],
    )


def validate_and_prepare_input_window(
    df: pd.DataFrame,
    anchor_time: Optional[pd.Timestamp] = None,
    window_size: int = 24,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Quality-assure and prepare the 24-hour observation sequence prior to model inference.

    Requirements:
    1. Filter strictly to observations at or before anchor_time (NEVER peek into future).
    2. Require at least window_size consecutive hourly time steps.
    3. Check for any missing timestamps or gaps (> 1 hour).
    4. Check for NaNs across all required physical variables (rejection on missing data).
    5. Construct circular directions and cyclical time encodings.
    6. Ensure column order strictly matches DEFAULT_FEATURE_COLS.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Timestamp]
        Prepared 24x13 DataFrame and verified anchor timestamp.
    """
    df_work = df.copy()
    df_work["valid_time"] = pd.to_datetime(df_work["valid_time"], utc=True)

    # If anchor_time not specified, select the latest observation at or before current time (or latest historical for demo)
    if anchor_time is None:
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        past_times = df_work.loc[df_work["valid_time"] <= now_utc, "valid_time"]
        if not past_times.empty:
            anchor_time = past_times.max()
        else:
            anchor_time = df_work["valid_time"].max()
    else:
        anchor_time = pd.to_datetime(anchor_time, utc=True)

    # Rule 1: Never use future observations as inputs
    df_window = df_work[df_work["valid_time"] <= anchor_time].sort_values("valid_time").reset_index(drop=True)

    if len(df_window) < window_size:
        raise ValueError(
            f"Insufficient historical observations. Required at least {window_size} hourly steps, "
            f"but found only {len(df_window)} observations at or before {anchor_time}."
        )

    # Slice the exact last window_size observations
    df_window = df_window.iloc[-window_size:].copy().reset_index(drop=True)
    actual_anchor = df_window["valid_time"].iloc[-1]

    # Rule 2: Verify timestamp continuity (strictly 1-hour delta between successive steps)
    time_diffs = df_window["valid_time"].diff().iloc[1:]
    expected_diff = pd.Timedelta(hours=1)
    irregular_diffs = time_diffs[time_diffs != expected_diff]
    if not irregular_diffs.empty:
        raise ValueError(
            f"Irregular or missing hourly timestamps detected in input window: "
            f"gaps found at indices {irregular_diffs.index.tolist()} with deltas {irregular_diffs.tolist()}. "
            f"Silent imputation is disabled for scientific integrity."
        )

    # Rule 3: Missing value check (no NaNs allowed in required inputs)
    core_inputs = ["swh", "mwp"]
    for col in core_inputs:
        if col not in df_window.columns:
            raise KeyError(f"Required physical column '{col}' is missing from observations.")
        if df_window[col].isna().any():
            raise ValueError(f"Missing (NaN) values detected in column '{col}'. Forecast aborted.")

    # Rule 4: Compute u10 and v10 if provided as wind speed & direction
    if "u10" not in df_window.columns or "v10" not in df_window.columns:
        if "wind_speed" not in df_window.columns or "wind_dir" not in df_window.columns:
            raise KeyError("Neither (u10, v10) nor (wind_speed, wind_dir) found in input observations.")
        speed = df_window["wind_speed"].to_numpy(dtype=float)
        direction_deg = df_window["wind_dir"].to_numpy(dtype=float)
        # Meteorological convention: wind coming FROM dir blows TOWARDS dir+180
        # u = -S * sin(dir), v = -S * cos(dir)
        rad = np.deg2rad(direction_deg)
        df_window["u10"] = -speed * np.sin(rad)
        df_window["v10"] = -speed * np.cos(rad)

    # Rule 5: Derive wind_speed and circular wind direction
    if "wind_speed" not in df_window.columns:
        df_window["wind_speed"] = np.sqrt(df_window["u10"] ** 2 + df_window["v10"] ** 2)

    if "wind_dir_sin" not in df_window.columns or "wind_dir_cos" not in df_window.columns:
        if "wind_dir" in df_window.columns:
            wind_dir_deg = df_window["wind_dir"]
        else:
            wind_dir_deg = (270.0 - np.rad2deg(np.arctan2(df_window["v10"].values, df_window["u10"].values))) % 360.0
        circ_wind = encode_circular_angle(wind_dir_deg, prefix="wind_dir")
        df_window["wind_dir_sin"] = circ_wind["wind_dir_sin"].values
        df_window["wind_dir_cos"] = circ_wind["wind_dir_cos"].values

    # Rule 6: Derive circular wave direction
    if "mwd_sin" not in df_window.columns or "mwd_cos" not in df_window.columns:
        if "mwd" not in df_window.columns:
            raise KeyError("Wave direction ('mwd' or 'mwd_sin'/'mwd_cos') missing from input observations.")
        circ_wave = encode_circular_angle(df_window["mwd"], prefix="mwd")
        df_window["mwd_sin"] = circ_wave["mwd_sin"].values
        df_window["mwd_cos"] = circ_wave["mwd_cos"].values

    # Rule 7: Cyclical temporal features
    df_window = add_cyclical_time_features(df_window, time_col="valid_time")

    # Rule 8: Enforce exact feature order
    for col in DEFAULT_FEATURE_COLS:
        if col not in df_window.columns:
            raise KeyError(f"Engineered feature '{col}' is missing.")
        if df_window[col].isna().any():
            raise ValueError(f"NaN detected in engineered feature '{col}'.")

    return df_window, actual_anchor


def load_horizon_checkpoint(
    model_type: str,
    horizon: int,
    models_dir: Optional[Union[str, Path]] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any], str]:
    """
    Safely load model architecture and weights for the specified horizon.

    Strictly guarantees that an H=1 checkpoint is NEVER loaded for H=6 or H=24.
    """
    if models_dir is None:
        models_dir = PROJECT_ROOT / "models"
    dir_path = Path(models_dir)

    m_code = model_type.lower().replace("wave", "").strip()
    if m_code not in ["lstm", "ann", "cnn1d"]:
        raise ValueError(f"Unsupported model type '{model_type}'. Choose 'lstm', 'ann', or 'cnn1d'.")

    if horizon not in [1, 6, 24]:
        raise ValueError(f"Unsupported horizon {horizon}h. Supported horizons: 1, 6, 24.")

    if horizon == 1:
        ckpt_filename = f"best_{m_code}.pt"
        meta_filename = f"best_{m_code}_metadata.json"
    else:
        ckpt_filename = f"best_{m_code}_h{horizon}.pt"
        meta_filename = f"best_{m_code}_h{horizon}_metadata.json"

    ckpt_path = dir_path / ckpt_filename
    meta_path = dir_path / meta_filename

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Required checkpoint file '{ckpt_filename}' does not exist at {dir_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Required metadata file '{meta_filename}' does not exist at {dir_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Double-check metadata horizon match
    meta_h = meta.get("dataset_config", {}).get("horizon", 1)
    if meta_h != horizon:
        raise ValueError(
            f"Checkpoint metadata mismatch! Requested horizon={horizon}h, "
            f"but checkpoint metadata declares horizon={meta_h}h."
        )

    cfg = meta.get("model_config", {})
    model = build_wave_model(
        model_type=cfg.get("model_type", m_code),
        input_dim=cfg.get("input_dim", len(DEFAULT_FEATURE_COLS)),
        seq_len=cfg.get("seq_len", 24),
        output_dim=cfg.get("output_dim", 1),
    )

    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
        model.load_state_dict(ckpt_data["model_state_dict"])
    else:
        model.load_state_dict(ckpt_data)

    model.eval()
    return model, meta, ckpt_filename


def predict_live_forecast(
    prepared_df: pd.DataFrame,
    anchor_time: pd.Timestamp,
    horizon: int,
    model_type: str,
    scaler: TimeSeriesScaler,
    models_dir: Optional[Union[str, Path]] = None,
    external_nwp_forecast: Optional[float] = None,
    is_demo: bool = False,
    source_info: Optional[Dict[str, Any]] = None,
) -> LiveForecastResult:
    """
    Execute end-to-end inference for a verified 24h input window.

    Guarantees:
    - Input shape is strictly (1, 24, 13)
    - Normalization applied using training-fitted scaler (never refitted)
    - Inverse scaling applied to obtain forecast in meters
    - Target time computed as anchor_time + H
    - Persistence baseline computed as swh at anchor_time
    """
    if len(prepared_df) != 24:
        raise ValueError(f"Prepared window must have exactly 24 rows, got {len(prepared_df)}.")

    # 1. Feature normalization using training scaler (no refit!)
    scaled_feats = scaler.feature_scaler.transform(prepared_df[scaler.feature_names].values).astype(np.float32)
    input_tensor = torch.tensor(scaled_feats, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 24, 13]

    # 2. Horizon-specific checkpoint loading
    model, meta, ckpt_name = load_horizon_checkpoint(model_type, horizon, models_dir=models_dir)

    # 3. Model inference forward pass
    with torch.no_grad():
        out_scaled = model(input_tensor).cpu().numpy().squeeze()

    # 4. Inverse transform to meters
    forecast_swh = float(scaler.inverse_transform_target(np.array([[out_scaled]]))[0])

    # 5. Persistence baseline: observed wave height at latest input time t
    persistence_swh = float(prepared_df["swh"].iloc[-1])

    # 6. Target timestamp
    target_time = anchor_time + pd.Timedelta(hours=horizon)

    # 7. Sea state classification
    sea_state_desc, sea_state_color = get_sea_state_classification(forecast_swh)

    return LiveForecastResult(
        model_name=meta.get("model_class", model_type),
        horizon_hours=horizon,
        checkpoint_file=ckpt_name,
        anchor_time_utc=anchor_time.to_pydatetime(),
        target_time_utc=target_time.to_pydatetime(),
        forecast_swh_m=forecast_swh,
        persistence_swh_m=persistence_swh,
        external_nwp_swh_m=external_nwp_forecast,
        sea_state_desc=sea_state_desc,
        sea_state_color=sea_state_color,
        input_window_df=prepared_df,
        is_demo=is_demo,
        source_info=source_info or {},
    )


def get_sea_state_classification(swh_meters: float) -> Tuple[str, str]:
    """WMO Sea State Code classification and color badge."""
    if swh_meters < 0.1:
        return "Calm (Glassy)", "#38bdf8"
    elif swh_meters < 0.5:
        return "Calm (Rippled)", "#38bdf8"
    elif swh_meters < 1.25:
        return "Smooth", "#22c55e"
    elif swh_meters < 2.5:
        return "Slight to Moderate", "#eab308"
    elif swh_meters < 4.0:
        return "Rough", "#f97316"
    elif swh_meters < 6.0:
        return "Very Rough", "#ef4444"
    elif swh_meters < 9.0:
        return "High", "#b91c1c"
    else:
        return "Phenomenal", "#7f1d1d"

"""
Configuration and constants for OceanCast AI FastAPI Backend.
"""

import os
from pathlib import Path
from typing import List

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PARQUET_TIMESERIES = PROCESSED_DATA_DIR / "era5_wave_offshore_timeseries.parquet"
DEMO_WINDOW_CSV = PROCESSED_DATA_DIR / "era5_demo_window.csv"

# Domain & Coordinates
DEFAULT_LATITUDE = 18.50
DEFAULT_LONGITUDE = 72.50
LOCATION_NAME = "Offshore Arabian Sea (Maharashtra Coast)"

# Model Constraints
SUPPORTED_HORIZONS: List[int] = [1, 6, 24]
SUPPORTED_MODELS: List[str] = ["ann", "lstm", "cnn1d"]
WINDOW_SIZE: int = 24

# Features
FEATURE_COLS: List[str] = [
    "swh",
    "mwp",
    "mwd_sin",
    "mwd_cos",
    "u10",
    "v10",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]
TARGET_COL: str = "swh"

# Units
VARIABLE_UNITS = {
    "swh": "m",
    "mwp": "s",
    "mwd": "deg",
    "u10": "m/s",
    "v10": "m/s",
    "wind_speed": "m/s",
    "wind_dir": "deg",
}

# CORS
_env_cors = os.getenv("CORS_ORIGINS", "")
ALLOWED_ORIGINS = [origin.strip() for origin in _env_cors.split(",") if origin.strip()] if _env_cors else [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:3000",
]

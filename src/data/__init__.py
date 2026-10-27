"""
ERA5 Ocean Wave & Atmospheric Ingestion, Validation, and Preprocessing Suite.
"""

from .loader import (
    load_netcdf_dataset,
    discover_era5_datasets,
    inspect_netcdf_variables,
)
from .validation import (
    validate_coordinates,
    check_temporal_alignment,
    verify_offshore_water_point,
    check_series_continuity_and_quality,
    PHYSICAL_BOUNDS,
)
from .features import (
    encode_circular_angle,
    compute_wind_speed_and_direction,
    add_cyclical_time_features,
)
from .preprocessing import (
    find_nearest_offshore_point,
    extract_point_timeseries,
    preprocess_era5_pipeline,
)
from .dataset import (
    WaveTimeSeriesDataset,
    TimeSeriesScaler,
    compute_leak_free_split_indices,
    create_wave_forecast_dataloaders,
    set_deterministic_seeds,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
)
from .live_service import (
    LiveFetchResult,
    LiveForecastResult,
    fetch_live_marine_and_weather,
    fetch_demo_era5_window,
    validate_and_prepare_input_window,
    load_horizon_checkpoint,
    predict_live_forecast,
    get_sea_state_classification,
)

__all__ = [
    "load_netcdf_dataset",
    "discover_era5_datasets",
    "inspect_netcdf_variables",
    "validate_coordinates",
    "check_temporal_alignment",
    "verify_offshore_water_point",
    "check_series_continuity_and_quality",
    "PHYSICAL_BOUNDS",
    "encode_circular_angle",
    "compute_wind_speed_and_direction",
    "add_cyclical_time_features",
    "find_nearest_offshore_point",
    "extract_point_timeseries",
    "preprocess_era5_pipeline",
    "WaveTimeSeriesDataset",
    "TimeSeriesScaler",
    "compute_leak_free_split_indices",
    "create_wave_forecast_dataloaders",
    "set_deterministic_seeds",
    "DEFAULT_FEATURE_COLS",
    "DEFAULT_TARGET_COL",
    "LiveFetchResult",
    "LiveForecastResult",
    "fetch_live_marine_and_weather",
    "fetch_demo_era5_window",
    "validate_and_prepare_input_window",
    "load_horizon_checkpoint",
    "predict_live_forecast",
    "get_sea_state_classification",
]


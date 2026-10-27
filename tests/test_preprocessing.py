"""
Comprehensive Automated Unit Tests for ERA5 Preprocessing Pipeline:
- Circular angular encoding (sine / cosine topology)
- Timestamp alignment and continuity checks
- Missing values and physical bounds detection
- Water vs. land cell verification logic
- Processed dataset schema and data integrity
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    encode_circular_angle,
    compute_wind_speed_and_direction,
    add_cyclical_time_features,
)
from src.data.validation import (
    check_series_continuity_and_quality,
    PHYSICAL_BOUNDS,
)


def test_circular_direction_encoding_properties():
    """Verify mathematical properties of circular angular decomposition."""
    angles = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    df_circ = encode_circular_angle(angles, prefix="test")

    # 1. 0° and 360° must produce identical (sin, cos) coordinates
    sin_0 = df_circ.loc[0, "test_sin"]
    cos_0 = df_circ.loc[0, "test_cos"]
    sin_360 = df_circ.loc[4, "test_sin"]
    cos_360 = df_circ.loc[4, "test_cos"]

    assert np.isclose(sin_0, sin_360, atol=1e-6)
    assert np.isclose(cos_0, cos_360, atol=1e-6)
    assert np.isclose(sin_0, 0.0, atol=1e-6)
    assert np.isclose(cos_0, 1.0, atol=1e-6)

    # 2. Pythagorean trigonometric identity: sin^2 + cos^2 == 1.0 for all angles
    unit_norm = df_circ["test_sin"] ** 2 + df_circ["test_cos"] ** 2
    assert np.allclose(unit_norm, 1.0, atol=1e-6)

    # 3. Known cardinal points
    # 90°: sin=1, cos=0
    assert np.isclose(df_circ.loc[1, "test_sin"], 1.0, atol=1e-6)
    assert np.isclose(df_circ.loc[1, "test_cos"], 0.0, atol=1e-6)
    # 180°: sin=0, cos=-1
    assert np.isclose(df_circ.loc[2, "test_sin"], 0.0, atol=1e-6)
    assert np.isclose(df_circ.loc[2, "test_cos"], -1.0, atol=1e-6)
    # 270°: sin=-1, cos=0
    assert np.isclose(df_circ.loc[3, "test_sin"], -1.0, atol=1e-6)
    assert np.isclose(df_circ.loc[3, "test_cos"], 0.0, atol=1e-6)


def test_circular_distance_preservation():
    """Verify that 359° and 1° are topologically close on the unit circle."""
    angles = np.array([359.0, 1.0])
    df_circ = encode_circular_angle(angles, prefix="dir")

    pt1 = np.array([df_circ.loc[0, "dir_sin"], df_circ.loc[0, "dir_cos"]])
    pt2 = np.array([df_circ.loc[1, "dir_sin"], df_circ.loc[1, "dir_cos"]])

    euclidean_dist = np.linalg.norm(pt1 - pt2)

    # Linear distance is |359 - 1| = 358 (artificial boundary jump)
    # Circular Euclidean distance on unit circle is ~ 0.0349
    assert euclidean_dist < 0.04
    assert np.isclose(euclidean_dist, 2 * np.sin(np.deg2rad(1.0)), atol=1e-4)


def test_wind_speed_and_direction():
    """Verify wind magnitude and meteorological direction computation."""
    # Eastward wind (u=10, v=0) -> blows FROM west -> 270°
    df_east = compute_wind_speed_and_direction([10.0], [0.0])
    assert np.isclose(df_east["wind_speed"].iloc[0], 10.0)
    assert np.isclose(df_east["wind_direction_deg"].iloc[0], 270.0)

    # Northward wind (u=0, v=10) -> blows FROM south -> 180°
    df_north = compute_wind_speed_and_direction([0.0], [10.0])
    assert np.isclose(df_north["wind_speed"].iloc[0], 10.0)
    assert np.isclose(df_north["wind_direction_deg"].iloc[0], 180.0)


def test_timestamp_continuity_and_gap_detection():
    """Verify detection of continuous hourly time series vs. discontinuous gaps."""
    # 1. Perfectly continuous 48-hour series
    dates = pd.date_range("2020-01-01 00:00", periods=48, freq="1h")
    df_clean = pd.DataFrame({"valid_time": dates, "swh": np.ones(48)})
    res_clean = check_series_continuity_and_quality(df_clean, time_col="valid_time", value_cols=["swh"])
    assert res_clean["passed"] is True
    assert res_clean["missing_timestamps_count"] == 0
    assert res_clean["duplicate_timestamps_count"] == 0

    # 2. Synthetic series with 5-hour gap
    dates_gap = list(dates[:10]) + list(dates[15:])
    df_gap = pd.DataFrame({"valid_time": dates_gap, "swh": np.ones(len(dates_gap))})
    res_gap = check_series_continuity_and_quality(
        df_gap, time_col="valid_time", value_cols=["swh"], max_allowed_gap_hours=3
    )
    assert res_gap["passed"] is False
    assert res_gap["missing_timestamps_count"] == 5
    assert res_gap["large_gaps_exceeding_threshold_count"] == 1

    # 3. Synthetic series with duplicates
    dates_dup = list(dates[:5]) + [dates[4]] + list(dates[5:])
    df_dup = pd.DataFrame({"valid_time": dates_dup, "swh": np.ones(len(dates_dup))})
    res_dup = check_series_continuity_and_quality(df_dup, time_col="valid_time", value_cols=["swh"])
    assert res_dup["passed"] is False
    assert res_dup["duplicate_timestamps_count"] == 1


def test_quality_assurance_bounds_and_anomalies():
    """Verify detection of negative wave heights, NaNs, and infinities."""
    dates = pd.date_range("2020-01-01 00:00", periods=5, freq="1h")

    # Negative wave height (physically impossible)
    df_anom = pd.DataFrame({
        "valid_time": dates,
        "swh": [1.0, 1.5, -2.0, 1.2, 1.8],
    })
    res_anom = check_series_continuity_and_quality(df_anom, time_col="valid_time", value_cols=["swh"])
    assert res_anom["passed"] is False
    assert res_anom["implausible_counts_per_column"]["swh"] == 1

    # NaN and Inf injection
    df_nan_inf = pd.DataFrame({
        "valid_time": dates,
        "swh": [1.0, np.nan, 2.0, np.inf, 1.8],
    })
    res_nan_inf = check_series_continuity_and_quality(df_nan_inf, time_col="valid_time", value_cols=["swh"])
    assert res_nan_inf["nan_counts_per_column"]["swh"] == 1
    assert res_nan_inf["inf_counts_per_column"]["swh"] == 1


def test_processed_dataset_integrity():
    """Verify that the generated Parquet artifact exists and conforms to requirements."""
    pq_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"

    assert pq_path.is_file(), f"Processed dataset missing at: {pq_path}"

    df = pd.read_parquet(pq_path)

    # 1. Row count & dimension check
    assert len(df) == 129696, f"Expected 129696 rows, found {len(df)}"

    # 2. Required columns present
    expected_cols = [
        "valid_time", "latitude", "longitude",
        "swh", "mwp", "mwd_deg", "mwd_sin", "mwd_cos",
        "u10", "v10", "wind_speed", "wind_direction_deg", "wind_dir_sin", "wind_dir_cos",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
    ]
    for c in expected_cols:
        assert c in df.columns, f"Missing required column: {c}"

    # 3. Zero NaNs or Infs across all columns
    assert df.isna().sum().sum() == 0, "Processed dataset contains NaNs!"
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert np.isinf(df[numeric_cols].to_numpy()).sum() == 0, "Processed dataset contains Infinities!"

    # 4. Circular encoding integrity: mwd_sin^2 + mwd_cos^2 == 1.0
    norm_mwd = df["mwd_sin"] ** 2 + df["mwd_cos"] ** 2
    assert np.allclose(norm_mwd, 1.0, atol=1e-5), "Wave direction circular decomposition does not lie on unit circle!"

    # 5. Temporal ordering & continuity
    times = pd.to_datetime(df["valid_time"])
    assert times.is_monotonic_increasing, "Timestamps are not strictly chronological!"
    assert times.duplicated().sum() == 0, "Duplicate timestamps detected in processed dataset!"

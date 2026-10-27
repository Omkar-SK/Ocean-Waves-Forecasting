"""
Validation and Quality Assurance Routines for ERA5 Wave and Atmospheric Time Series.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr


# Standard physical plausibility bounds for oceanographic and surface meteorological variables
PHYSICAL_BOUNDS = {
    "swh": {"min": 0.0, "max": 25.0, "units": "m", "desc": "Significant Wave Height"},
    "mwp": {"min": 1.0, "max": 30.0, "units": "s", "desc": "Mean Wave Period"},
    "mwd": {"min": 0.0, "max": 360.0, "units": "deg", "desc": "Mean Wave Direction"},
    "u10": {"min": -60.0, "max": 60.0, "units": "m/s", "desc": "10m Zonal Wind"},
    "v10": {"min": -60.0, "max": 60.0, "units": "m/s", "desc": "10m Meridional Wind"},
    "wind_speed": {"min": 0.0, "max": 75.0, "units": "m/s", "desc": "Wind Speed"},
    "wind_direction_deg": {"min": 0.0, "max": 360.0, "units": "deg", "desc": "Wind Direction"},
}


def validate_coordinates(ds: xr.Dataset) -> Dict[str, Any]:
    """Extract coordinate diagnostics (dimensions, time range, lat/lon bounds)."""
    coords = {}
    for coord_name, coord in ds.coords.items():
        arr = coord.values
        step = None
        if len(arr) > 1 and np.issubdtype(arr.dtype, np.number):
            diffs = np.diff(arr)
            step = float(diffs[0]) if np.allclose(diffs, diffs[0]) else "variable"

        coords[coord_name] = {
            "size": int(coord.size),
            "min": float(arr.min()) if np.issubdtype(arr.dtype, np.number) else str(arr[0]),
            "max": float(arr.max()) if np.issubdtype(arr.dtype, np.number) else str(arr[-1]),
            "dtype": str(coord.dtype),
            "step": step,
            "values": arr.tolist() if len(arr) <= 20 else None,
        }
    return coords


def check_temporal_alignment(ds1: xr.Dataset, ds2: xr.Dataset) -> Tuple[bool, Dict[str, Any]]:
    """Check if two datasets have aligned time coordinates and return diagnostics."""
    t1_name = "valid_time" if "valid_time" in ds1.coords else "time"
    t2_name = "valid_time" if "valid_time" in ds2.coords else "time"

    if t1_name not in ds1.coords or t2_name not in ds2.coords:
        return False, {"error": "Time coordinate missing from one or both datasets"}

    t1_vals = ds1[t1_name].values
    t2_vals = ds2[t2_name].values

    is_identical = bool(np.array_equal(t1_vals, t2_vals))
    overlap_count = int(len(np.intersect1d(t1_vals, t2_vals)))

    return is_identical, {
        "ds1_time_coord": t1_name,
        "ds2_time_coord": t2_name,
        "ds1_length": len(t1_vals),
        "ds2_length": len(t2_vals),
        "is_identical": is_identical,
        "overlap_count": overlap_count,
        "ds1_start": str(t1_vals[0]) if len(t1_vals) > 0 else None,
        "ds1_end": str(t1_vals[-1]) if len(t1_vals) > 0 else None,
        "ds2_start": str(t2_vals[0]) if len(t2_vals) > 0 else None,
        "ds2_end": str(t2_vals[-1]) if len(t2_vals) > 0 else None,
    }


def verify_offshore_water_point(
    ds_wave: xr.Dataset,
    latitude: float,
    longitude: float,
    wave_var: str = "swh",
) -> Dict[str, Any]:
    """
    Verify whether a specified (latitude, longitude) coordinate represents an offshore water cell.

    In ERA5 WAM (wave model), land cells are masked with NaN across 100% of time steps.
    Valid water cells contain continuous physical wave height and period observations.

    Parameters
    ----------
    ds_wave : xr.Dataset
        xarray Dataset containing ocean wave variables.
    latitude : float
        Target latitude in degrees North.
    longitude : float
        Target longitude in degrees East.
    wave_var : str
        Variable to verify (default 'swh').

    Returns
    -------
    Dict[str, Any]
        Diagnostic dictionary including:
        - 'is_water': bool
        - 'exact_match': bool
        - 'selected_lat': float
        - 'selected_lon': float
        - 'nan_count': int
        - 'valid_ratio': float
        - 'certainty': str ("high", "moderate", "uncertain")
        - 'reason': str
    """
    if "latitude" not in ds_wave.coords or "longitude" not in ds_wave.coords:
        raise ValueError("Dataset missing 'latitude' or 'longitude' coordinates.")

    lats = ds_wave["latitude"].values
    lons = ds_wave["longitude"].values

    exact_lat = latitude in lats
    exact_lon = longitude in lons
    exact_match = exact_lat and exact_lon

    # Select point using nearest neighbor
    point_ds = ds_wave.sel(latitude=latitude, longitude=longitude, method="nearest")
    selected_lat = float(point_ds["latitude"].values)
    selected_lon = float(point_ds["longitude"].values)

    if wave_var not in point_ds.data_vars:
        available_vars = list(point_ds.data_vars.keys())
        if not available_vars:
            raise ValueError(f"No data variables found in dataset to test water status.")
        wave_var = available_vars[0]

    vals = point_ds[wave_var].values
    total_steps = vals.size
    nan_count = int(np.isnan(vals).sum())
    valid_count = total_steps - nan_count
    valid_ratio = valid_count / total_steps if total_steps > 0 else 0.0

    min_val = float(np.nanmin(vals)) if valid_count > 0 else None
    max_val = float(np.nanmax(vals)) if valid_count > 0 else None
    mean_val = float(np.nanmean(vals)) if valid_count > 0 else None

    # Decision logic
    if valid_ratio == 1.0 and (min_val is not None and min_val > 0.0):
        is_water = True
        certainty = "high"
        reason = (
            f"Point ({selected_lat:.2f}°N, {selected_lon:.2f}°E) has 100% valid wave observations "
            f"({total_steps:,} hours, range: [{min_val:.2f}m, {max_val:.2f}m], mean: {mean_val:.2f}m). "
            f"Zero NaNs confirm this is an offshore ocean cell."
        )
    elif valid_ratio > 0.90:
        is_water = True
        certainty = "moderate"
        reason = (
            f"Point ({selected_lat:.2f}°N, {selected_lon:.2f}°E) has {valid_ratio*100:.2f}% valid observations. "
            f"Likely water with occasional missing steps or boundary effects."
        )
    elif valid_ratio == 0.0:
        is_water = False
        certainty = "high"
        reason = (
            f"Point ({selected_lat:.2f}°N, {selected_lon:.2f}°E) has 100% NaNs ({nan_count:,} steps). "
            f"This represents an ECMWF land-masked grid cell."
        )
    else:
        is_water = False
        certainty = "uncertain"
        reason = (
            f"Point ({selected_lat:.2f}°N, {selected_lon:.2f}°E) has {nan_count:,} NaNs ({nan_count/total_steps*100:.1f}%). "
            f"High uncertainty between coastal intertidal zone and land barrier."
        )

    return {
        "is_water": is_water,
        "exact_match": exact_match,
        "requested_lat": float(latitude),
        "requested_lon": float(longitude),
        "selected_lat": selected_lat,
        "selected_lon": selected_lon,
        "distance_deg": float(np.sqrt((latitude - selected_lat) ** 2 + (longitude - selected_lon) ** 2)),
        "wave_var_checked": wave_var,
        "total_steps": total_steps,
        "nan_count": nan_count,
        "valid_ratio": valid_ratio,
        "min_value": min_val,
        "max_value": max_val,
        "mean_value": mean_val,
        "certainty": certainty,
        "reason": reason,
    }


def check_series_continuity_and_quality(
    df: pd.DataFrame,
    time_col: str = "valid_time",
    value_cols: Optional[List[str]] = None,
    expected_freq: str = "1h",
    max_allowed_gap_hours: int = 3,
) -> Dict[str, Any]:
    """
    Perform rigorous data quality assurance checks on extracted time series:
    - Chronological sorting
    - Duplicate timestamps
    - Missing hours / continuity gaps against ideal date_range
    - NaNs and Nulls
    - Infinities (inf, -inf)
    - Physical plausibility range violations

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    time_col : str
        Name of timestamp column.
    value_cols : Optional[List[str]]
        Columns to check for anomalies (defaults to all numeric columns).
    expected_freq : str
        Expected pandas frequency string (default '1h').
    max_allowed_gap_hours : int
        Maximum gap duration in hours before flagging a critical discontinuity.

    Returns
    -------
    Dict[str, Any]
        Quality report including pass/fail status and detailed counts.
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame.")

    # Ensure datetime
    times = pd.to_datetime(df[time_col])

    # 1. Chronological order
    is_sorted = bool(times.is_monotonic_increasing)

    # 2. Duplicate timestamps
    dup_mask = times.duplicated()
    dup_count = int(dup_mask.sum())
    dup_samples = [str(t) for t in times[dup_mask].head(5)]

    # 3. Continuity check against ideal frequency
    t_start = times.min()
    t_end = times.max()
    expected_range = pd.date_range(start=t_start, end=t_end, freq=expected_freq)
    missing_timestamps = expected_range.difference(times)
    missing_count = len(missing_timestamps)

    # Step differences
    time_diffs_hours = (times.diff().dt.total_seconds() / 3600.0).dropna()
    max_step_hours = float(time_diffs_hours.max()) if len(time_diffs_hours) > 0 else 0.0
    min_step_hours = float(time_diffs_hours.min()) if len(time_diffs_hours) > 0 else 0.0

    large_gaps_count = int((time_diffs_hours > max_allowed_gap_hours).sum())

    # 4. Value-level checks
    if value_cols is None:
        value_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != time_col]

    nan_counts = {}
    inf_counts = {}
    implausible_counts = {}
    stats = {}

    for col in value_cols:
        series = df[col]
        # NaNs
        n_nan = int(series.isna().sum())
        nan_counts[col] = n_nan

        # Infs
        n_inf = int(np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum())
        inf_counts[col] = n_inf

        # Plausibility bounds (with float32 epsilon tolerance)
        bounds = PHYSICAL_BOUNDS.get(col)
        n_implausible = 0
        if bounds:
            eps = 1e-3
            below = (series < (bounds["min"] - eps)).sum()
            above = (series > (bounds["max"] + eps)).sum()
            n_implausible = int(below + above)
        implausible_counts[col] = n_implausible

        finite_vals = series.dropna()
        finite_vals = finite_vals[~np.isinf(finite_vals.to_numpy(dtype=float))]
        stats[col] = {
            "min": float(finite_vals.min()) if len(finite_vals) > 0 else None,
            "max": float(finite_vals.max()) if len(finite_vals) > 0 else None,
            "mean": float(finite_vals.mean()) if len(finite_vals) > 0 else None,
            "std": float(finite_vals.std()) if len(finite_vals) > 0 else None,
        }

    # Overall quality verdict
    passed = (
        is_sorted
        and dup_count == 0
        and missing_count == 0
        and sum(nan_counts.values()) == 0
        and sum(inf_counts.values()) == 0
        and sum(implausible_counts.values()) == 0
    )

    return {
        "passed": passed,
        "row_count": len(df),
        "time_start": str(t_start),
        "time_end": str(t_end),
        "is_chronologically_sorted": is_sorted,
        "duplicate_timestamps_count": dup_count,
        "duplicate_timestamp_samples": dup_samples,
        "missing_timestamps_count": missing_count,
        "expected_steps_count": len(expected_range),
        "min_step_hours": min_step_hours,
        "max_step_hours": max_step_hours,
        "large_gaps_exceeding_threshold_count": large_gaps_count,
        "max_allowed_gap_hours": max_allowed_gap_hours,
        "nan_counts_per_column": nan_counts,
        "total_nan_count": sum(nan_counts.values()),
        "inf_counts_per_column": inf_counts,
        "total_inf_count": sum(inf_counts.values()),
        "implausible_counts_per_column": implausible_counts,
        "total_implausible_count": sum(implausible_counts.values()),
        "column_summary_statistics": stats,
    }

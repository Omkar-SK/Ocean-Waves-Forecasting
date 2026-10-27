"""
Feature Engineering Utilities for Wave Forecasting:
- Circular angular transformations (sine / cosine decomposition)
- Wind vector decomposition, speed, and meteorological direction
- Cyclical calendar temporal features (hour of day, month of year)
"""

from typing import Union, Optional
import numpy as np
import pandas as pd


def encode_circular_angle(
    angles_deg: Union[pd.Series, np.ndarray],
    prefix: str = "angle",
) -> pd.DataFrame:
    """
    Decompose an angular variable in degrees [0, 360] into continuous sine and cosine components.

    Treating directional variables (such as wave or wind direction) as linear numbers induces
    an artificial discontinuity at the 0°/360° north boundary (where 359° and 1° appear 358 units apart).
    The (sin, cos) projection embeds directions on the unit circle S^1 where:
    - 0° and 360° map to identical coordinates (0.0, 1.0)
    - 359° and 1° are separated by a Euclidean distance of only ~0.0349
    - The identity sin²(θ) + cos²(θ) = 1 holds for all values.

    Parameters
    ----------
    angles_deg : Union[pd.Series, np.ndarray]
        Angles in degrees [0, 360].
    prefix : str
        Prefix for output columns (e.g. 'mwd' produces 'mwd_sin', 'mwd_cos').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns f'{prefix}_sin' and f'{prefix}_cos'.
    """
    if isinstance(angles_deg, pd.Series):
        vals = angles_deg.to_numpy(dtype=float)
        idx = angles_deg.index
    else:
        vals = np.asarray(angles_deg, dtype=float)
        idx = None

    rad = np.deg2rad(vals % 360.0)
    sin_vals = np.sin(rad)
    cos_vals = np.cos(rad)

    df = pd.DataFrame(
        {
            f"{prefix}_sin": sin_vals,
            f"{prefix}_cos": cos_vals,
        },
        index=idx,
    )
    return df


def compute_wind_speed_and_direction(
    u: Union[pd.Series, np.ndarray],
    v: Union[pd.Series, np.ndarray],
) -> pd.DataFrame:
    """
    Compute wind speed and meteorological direction from zonal (u) and meridional (v) components.

    Meteorological convention:
    Direction from which the wind blows, measured clockwise from true north:
    Dir = (270 - atan2(v, u) * 180 / π) % 360

    Parameters
    ----------
    u : Union[pd.Series, np.ndarray]
        Zonal 10m wind component (eastward in m/s).
    v : Union[pd.Series, np.ndarray]
        Meridional 10m wind component (northward in m/s).

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - 'wind_speed': scalar magnitude (m/s)
        - 'wind_direction_deg': meteorological direction [0, 360)
        - 'wind_dir_sin': sine of meteorological direction
        - 'wind_dir_cos': cosine of meteorological direction
    """
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)

    speed = np.sqrt(u_arr**2 + v_arr**2)
    # Meteorological direction
    dir_deg = (270.0 - np.rad2deg(np.arctan2(v_arr, u_arr))) % 360.0

    circ_df = encode_circular_angle(dir_deg, prefix="wind_dir")

    idx = u.index if isinstance(u, pd.Series) else None
    res = pd.DataFrame(
        {
            "wind_speed": speed,
            "wind_direction_deg": dir_deg,
            "wind_dir_sin": circ_df["wind_dir_sin"].values,
            "wind_dir_cos": circ_df["wind_dir_cos"].values,
        },
        index=idx,
    )
    return res


def add_cyclical_time_features(
    df: pd.DataFrame,
    time_col: str = "valid_time",
) -> pd.DataFrame:
    """
    Add cyclical calendar features (hour of day, month of year) using sine/cosine encoding.

    Captures diurnal sea breeze / wave variation and annual monsoon seasonality smoothly.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    time_col : str
        Timestamp column name.

    Returns
    -------
    pd.DataFrame
        DataFrame with added temporal columns:
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos'.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        times = pd.to_datetime(df[time_col])
    else:
        times = df[time_col]

    hour = times.dt.hour
    month = times.dt.month

    # Diurnal cycle: 24-hour periodicity
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    # Annual cycle: 12-month periodicity (month 1..12 -> 0..11)
    df["month_sin"] = np.sin(2.0 * np.pi * (month - 1) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0)

    return df

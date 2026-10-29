"""
Marine Data Provider Abstraction Layer for Real-Time Ocean Intelligence.

Defines standardized interfaces and result containers for external observational
and numerical weather prediction (NWP) providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from src.intelligence.nodes import CoastalNode

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DataStatus(str, Enum):
    """Explicit operational telemetry lifecycle states."""
    FRESH = "fresh"                  # Ingested from live provider within expected operational window
    CACHED = "cached"                # Served from local memory/disk cache within freshness TTL
    STALE = "stale"                  # Cached or retrieved data exceeding nominal freshness interval
    OFFLINE_DEMO = "offline_demo"    # Replayed historical archive for verified offline demonstration
    UNAVAILABLE = "unavailable"      # Upstream provider down, network timeout, or connection refused
    INVALID = "invalid"              # Ingested telemetry failed quality gates (gaps, NaNs, missing columns)


@dataclass
class ProviderFetchResult:
    """
    Standardized result envelope returned by all marine data providers.
    """
    provider_name: str
    location_id: str
    latitude: float
    longitude: float
    retrieval_time_utc: datetime
    data_status: DataStatus
    raw_df: pd.DataFrame
    forecast_issue_time_utc: Optional[datetime] = None
    valid_time_start_utc: Optional[datetime] = None
    valid_time_end_utc: Optional[datetime] = None
    latest_obs_time_utc: Optional[datetime] = None
    is_forecast: bool = False
    is_demo: bool = False
    external_nwp_forecasts: Dict[int, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    attribution: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Indicates whether retrieval was successful and data is usable."""
        return self.data_status in [DataStatus.FRESH, DataStatus.CACHED, DataStatus.OFFLINE_DEMO] and not self.raw_df.empty

    @property
    def freshness_minutes(self) -> Optional[int]:
        """Minutes elapsed between current time and latest observation."""
        if self.latest_obs_time_utc is None:
            return None
        now = datetime.now(timezone.utc)
        delta_s = (now - self.latest_obs_time_utc).total_seconds()
        return max(0, int(delta_s / 60))


class BaseMarineDataProvider(ABC):
    """
    Abstract interface for marine and meteorological telemetry providers.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier."""
        pass

    @property
    @abstractmethod
    def attribution(self) -> str:
        """Licensing and data attribution notice."""
        pass

    @abstractmethod
    def fetch_telemetry(
        self,
        node: CoastalNode,
        past_days: int = 2,
        forecast_days: int = 2,
        timeout_s: int = 10,
    ) -> ProviderFetchResult:
        """
        Retrieve marine wave and surface atmospheric telemetry for a specific coastal node.

        Parameters
        ----------
        node : CoastalNode
            Target coastal monitoring location.
        past_days : int
            Past observation history to retrieve (minimum 2 days required for 24h input window).
        forecast_days : int
            Future horizon days for external NWP benchmarks.
        timeout_s : int
            Network timeout limit in seconds.

        Returns
        -------
        ProviderFetchResult
            Standardized telemetry envelope with explicit data status.
        """
        pass


class OpenMeteoProvider(BaseMarineDataProvider):
    """
    Operational REST client for Open-Meteo Marine & Atmospheric APIs.
    Sources:
    - Marine API: ECMWF Wave Action Model (WAM) & Copernicus Marine Global Ocean Waves.
    - Weather API: ECMWF Integrated Forecasting System (IFS) / GFS 10m wind fields.
    Licensing: Open Database License (ODbL) / CC BY 4.0.
    """

    def __init__(self, max_retries: int = 2, retry_backoff_s: float = 1.0):
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

    @property
    def provider_name(self) -> str:
        return "Open-Meteo Marine & Atmospheric API (ECMWF WAM / IFS)"

    @property
    def attribution(self) -> str:
        return "Data provided by Open-Meteo under CC BY 4.0; underlying models by ECMWF & Copernicus Marine."

    def _execute_http_get(self, url: str, headers: Dict[str, str], timeout_s: int) -> Dict[str, Any]:
        """Execute HTTP GET with bounded retries on transient errors."""
        req = urllib.request.Request(url, headers=headers)
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    raw_data = resp.read().decode("utf-8")
                    return json.loads(raw_data)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionResetError) as e:
                last_exc = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * (attempt + 1))
            except Exception as e:
                last_exc = e
                break

        raise ConnectionError(f"Network error querying {url}: {last_exc}") from last_exc

    def fetch_telemetry(
        self,
        node: CoastalNode,
        past_days: int = 2,
        forecast_days: int = 2,
        timeout_s: int = 10,
    ) -> ProviderFetchResult:
        retrieval_time = datetime.now(timezone.utc)
        headers = {"User-Agent": "WaveCast-DL-Intelligence/2.0 (Academic Early-Warning Foundation)"}

        try:
            # 1. Marine API Query
            marine_params = {
                "latitude": node.latitude,
                "longitude": node.longitude,
                "hourly": "wave_height,wave_direction,wave_period",
                "past_days": past_days,
                "forecast_days": forecast_days,
                "timezone": "UTC",
            }
            marine_url = f"https://marine-api.open-meteo.com/v1/marine?{urllib.parse.urlencode(marine_params)}"
            marine_json = self._execute_http_get(marine_url, headers=headers, timeout_s=timeout_s)

            # 2. Weather API Query (10m wind speed and direction in m/s)
            weather_params = {
                "latitude": node.latitude,
                "longitude": node.longitude,
                "hourly": "wind_speed_10m,wind_direction_10m",
                "wind_speed_unit": "ms",
                "past_days": past_days,
                "forecast_days": forecast_days,
                "timezone": "UTC",
            }
            weather_url = f"https://api.open-meteo.com/v1/forecast?{urllib.parse.urlencode(weather_params)}"
            weather_json = self._execute_http_get(weather_url, headers=headers, timeout_s=timeout_s)

            # 3. Validate presence of hourly keys
            if "hourly" not in marine_json or "time" not in marine_json["hourly"]:
                raise ValueError("Malformed response from Open-Meteo Marine API: 'hourly.time' missing.")
            if "hourly" not in weather_json or "time" not in weather_json["hourly"]:
                raise ValueError("Malformed response from Open-Meteo Weather API: 'hourly.time' missing.")

            # 4. Parse timestamps to UTC
            times_marine = pd.to_datetime(marine_json["hourly"]["time"], utc=True)
            times_weather = pd.to_datetime(weather_json["hourly"]["time"], utc=True)

            df_marine = pd.DataFrame(
                {
                    "valid_time": times_marine,
                    "swh": marine_json["hourly"].get("wave_height", []),
                    "mwp": marine_json["hourly"].get("wave_period", []),
                    "mwd": marine_json["hourly"].get("wave_direction", []),
                }
            )

            df_weather = pd.DataFrame(
                {
                    "valid_time": times_weather,
                    "wind_speed": weather_json["hourly"].get("wind_speed_10m", []),
                    "wind_dir": weather_json["hourly"].get("wind_direction_10m", []),
                }
            )

            # Inner join on UTC valid_time
            merged = pd.merge(df_marine, df_weather, on="valid_time", how="inner").sort_values("valid_time").reset_index(drop=True)

            if merged.empty:
                raise ValueError("Zero overlapping hourly timestamps between marine and weather API responses.")

            current_hour_utc = retrieval_time.replace(minute=0, second=0, microsecond=0)
            hist_mask = merged["valid_time"] <= current_hour_utc
            df_hist = merged[hist_mask]

            if df_hist.empty:
                raise ValueError("No historical observations returned prior to or matching current UTC hour.")

            latest_obs_time = df_hist["valid_time"].max()

            # Extract external NWP forecast benchmarks for t+1, t+6, t+24
            external_nwp: Dict[int, float] = {}
            for h in [1, 6, 24]:
                target_t = latest_obs_time + pd.Timedelta(hours=h)
                match_row = merged[merged["valid_time"] == target_t]
                if not match_row.empty and not pd.isna(match_row["swh"].values[0]):
                    external_nwp[h] = float(match_row["swh"].values[0])

            # Detect nominal forecast generation/issue time from API metadata
            issue_time_utc: Optional[datetime] = None
            if "generationtime_ms" in marine_json:
                # Open-Meteo runs models 4 times daily (00, 06, 12, 18 UTC)
                # Compute latest model cycle anchor
                cycle_hour = (current_hour_utc.hour // 6) * 6
                issue_time_utc = current_hour_utc.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

            # Assess freshness (if latest observation is > 3 hours old, flag as STALE)
            age_hours = (retrieval_time - latest_obs_time.to_pydatetime()).total_seconds() / 3600.0
            data_status = DataStatus.FRESH if age_hours <= 3.5 else DataStatus.STALE

            return ProviderFetchResult(
                provider_name=self.provider_name,
                location_id=node.location_id,
                latitude=node.latitude,
                longitude=node.longitude,
                retrieval_time_utc=retrieval_time,
                data_status=data_status,
                raw_df=merged,
                forecast_issue_time_utc=issue_time_utc,
                valid_time_start_utc=merged["valid_time"].min().to_pydatetime(),
                valid_time_end_utc=merged["valid_time"].max().to_pydatetime(),
                latest_obs_time_utc=latest_obs_time.to_pydatetime(),
                is_forecast=False,
                is_demo=False,
                external_nwp_forecasts=external_nwp,
                error_message=None,
                attribution=self.attribution,
                notes=[
                    f"Location: {node.name} ({node.coordinate_label})",
                    f"Validation Status: {node.validation_status.value}",
                    f"Operational Marine Model: ECMWF WAM (~0.10° grid)",
                    f"Operational Atmospheric Model: ECMWF IFS (~0.10° grid)",
                ],
            )

        except Exception as e:
            return ProviderFetchResult(
                provider_name=self.provider_name,
                location_id=node.location_id,
                latitude=node.latitude,
                longitude=node.longitude,
                retrieval_time_utc=retrieval_time,
                data_status=DataStatus.UNAVAILABLE,
                raw_df=pd.DataFrame(),
                forecast_issue_time_utc=None,
                valid_time_start_utc=None,
                valid_time_end_utc=None,
                latest_obs_time_utc=None,
                is_forecast=False,
                is_demo=False,
                external_nwp_forecasts={},
                error_message=str(e),
                attribution=self.attribution,
                notes=[f"Failed to query Open-Meteo for {node.name}: {e}"],
            )


class OfflineDemoProvider(BaseMarineDataProvider):
    """
    Offline fallback provider replaying genuine ERA5 historical observation windows.

    Guarantees:
    - Zero external network dependency.
    - Explicitly labeled with DataStatus.OFFLINE_DEMO.
    - Preserves historical ground-truth observations ending 2026-10-17 23:00 UTC.
    - Never masquerades as a live real-time observation.
    """

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self.data_path = Path(data_path) if data_path else None

    @property
    def provider_name(self) -> str:
        return "Offline Historical Demo: ECMWF ERA5 Reanalysis Archive"

    @property
    def attribution(self) -> str:
        return "ECMWF ERA5 Reanalysis Dataset (Verified Historical Ground Truth Archive)."

    def _resolve_data_path(self) -> Path:
        """Identify available demo data files in data/processed/."""
        if self.data_path and self.data_path.is_file():
            return self.data_path

        demo_csv = PROJECT_ROOT / "data" / "processed" / "era5_demo_window.csv"
        full_csv = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.csv"
        full_parquet = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"

        if demo_csv.is_file():
            return demo_csv
        elif full_csv.is_file():
            return full_csv
        elif full_parquet.is_file():
            return full_parquet

        raise FileNotFoundError("No ERA5 dataset artifacts found in data/processed/ for offline demo.")

    def fetch_telemetry(
        self,
        node: CoastalNode,
        past_days: int = 2,
        forecast_days: int = 2,
        timeout_s: int = 10,
    ) -> ProviderFetchResult:
        retrieval_time = datetime.now(timezone.utc)
        try:
            path = self._resolve_data_path()
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)

            df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
            df = df.sort_values("valid_time").reset_index(drop=True)

            latest_time = df["valid_time"].max()

            # Ensure wind speed & direction exist
            if "wind_speed" not in df.columns and "u10" in df.columns and "v10" in df.columns:
                df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
            if "wind_dir" not in df.columns and "u10" in df.columns and "v10" in df.columns:
                df["wind_dir"] = (270.0 - np.rad2deg(np.arctan2(df["v10"].values, df["u10"].values))) % 360.0

            return ProviderFetchResult(
                provider_name=self.provider_name,
                location_id=node.location_id,
                latitude=node.latitude,
                longitude=node.longitude,
                retrieval_time_utc=retrieval_time,
                data_status=DataStatus.OFFLINE_DEMO,
                raw_df=df,
                forecast_issue_time_utc=latest_time.to_pydatetime(),
                valid_time_start_utc=df["valid_time"].min().to_pydatetime(),
                valid_time_end_utc=latest_time.to_pydatetime(),
                latest_obs_time_utc=latest_time.to_pydatetime(),
                is_forecast=False,
                is_demo=True,
                external_nwp_forecasts={},
                error_message=None,
                attribution=self.attribution,
                notes=[
                    f"OFFLINE DEMO MODE ACTIVE for {node.name}.",
                    "Using historical ERA5 observation window ending 2026-10-17 23:00 UTC.",
                    "This is NOT a real-time observation. Replays verified ground truth for pipeline inspection.",
                ],
            )
        except Exception as e:
            return ProviderFetchResult(
                provider_name=self.provider_name,
                location_id=node.location_id,
                latitude=node.latitude,
                longitude=node.longitude,
                retrieval_time_utc=retrieval_time,
                data_status=DataStatus.UNAVAILABLE,
                raw_df=pd.DataFrame(),
                forecast_issue_time_utc=None,
                valid_time_start_utc=None,
                valid_time_end_utc=None,
                latest_obs_time_utc=None,
                is_forecast=False,
                is_demo=True,
                external_nwp_forecasts={},
                error_message=str(e),
                attribution=self.attribution,
                notes=[f"Failed to load offline demo data: {e}"],
            )

"""
Live operational forecasting and external telemetry service.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.config import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, SUPPORTED_HORIZONS, SUPPORTED_MODELS
from backend.schemas.forecast import (
    ForecastRequest,
    ForecastResponse,
    ForecastStatusResponse,
    SingleModelForecast,
)
from backend.services.model_service import model_service
from src.data.live_service import (
    LiveFetchResult,
    fetch_demo_era5_window,
    fetch_live_marine_and_weather,
    validate_and_prepare_input_window,
)


class LiveForecastService:
    """Manages real-time data ingestion, input validation, and operational inference."""

    def __init__(self):
        self._cached_live_result: Optional[LiveFetchResult] = None
        self._last_fetch_time: Optional[datetime] = None

    def fetch_data(
        self,
        source: str = "live",
        latitude: float = DEFAULT_LATITUDE,
        longitude: float = DEFAULT_LONGITUDE,
        force_refresh: bool = False,
    ) -> LiveFetchResult:
        """Fetches live or offline demo observations with caching."""
        if source == "demo":
            return fetch_demo_era5_window()

        now = datetime.now(timezone.utc)
        # 10-minute cache on live API queries
        if (
            not force_refresh
            and self._cached_live_result is not None
            and self._last_fetch_time is not None
            and (now - self._last_fetch_time).total_seconds() < 600
        ):
            return self._cached_live_result

        result = fetch_live_marine_and_weather(latitude=latitude, longitude=longitude, timeout_s=10)
        self._cached_live_result = result
        self._last_fetch_time = now
        return result

    def generate_forecast(self, req: ForecastRequest) -> ForecastResponse:
        """Validates input sequence and generates single or multi-model forecasts."""
        if req.horizon not in SUPPORTED_HORIZONS:
            raise ValueError(f"Invalid horizon {req.horizon}h. Choose from {SUPPORTED_HORIZONS}.")

        # 1. Fetch raw telemetry
        fetch_res = self.fetch_data(
            source=req.source,
            latitude=req.latitude or DEFAULT_LATITUDE,
            longitude=req.longitude or DEFAULT_LONGITUDE,
        )

        # 2. Strict Input Quality Validation (24 consecutive hourly steps, no NaNs, no gaps)
        df_window, verified_anchor = validate_and_prepare_input_window(
            fetch_res.raw_df,
            window_size=24,
        )

        persistence_swh = float(df_window["swh"].iloc[-1])
        target_time = verified_anchor + pd.Timedelta(hours=req.horizon)
        nwp_val = fetch_res.external_nwp_forecasts.get(req.horizon)

        # 3. Model predictions
        predictions: Dict[str, SingleModelForecast] = {}
        models_to_run = SUPPORTED_MODELS if req.model == "all" else [req.model.lower().replace("wave", "")]

        for m in models_to_run:
            pred = model_service.predict_single(
                prepared_df=df_window,
                horizon=req.horizon,
                model_type=m,
                persistence_swh=persistence_swh,
                external_nwp=nwp_val,
            )
            predictions[m] = pred

        # 4. Clean observation window points for chart
        obs_points: List[Dict[str, Any]] = []
        for _, r in df_window.iterrows():
            obs_points.append(
                {
                    "valid_time": r["valid_time"].strftime("%Y-%m-%d %H:%M"),
                    "swh": round(float(r["swh"]), 3),
                    "mwp": round(float(r["mwp"]), 2),
                    "wind_speed": round(float(r["wind_speed"]), 2),
                    "u10": round(float(r["u10"]), 2),
                    "v10": round(float(r["v10"]), 2),
                }
            )

        return ForecastResponse(
            anchor_time_utc=verified_anchor.to_pydatetime(),
            target_time_utc=target_time.to_pydatetime(),
            forecast_horizon_hours=req.horizon,
            persistence_swh_m=round(persistence_swh, 4),
            external_nwp_swh_m=round(nwp_val, 4) if nwp_val is not None else None,
            predictions=predictions,
            observation_window=obs_points,
            source_name=fetch_res.source_name,
            attribution=fetch_res.attribution,
            is_demo=fetch_res.is_demo,
            notes=fetch_res.notes,
            generated_at_utc=datetime.now(timezone.utc),
        )

    def check_status(self) -> ForecastStatusResponse:
        """Checks live API accessibility and data freshness."""
        api_online = False
        latest_time = None
        freshness_mins = None
        message = "Live API connectivity unverified."

        try:
            res = self.fetch_data(source="live", force_refresh=False)
            api_online = True
            latest_time = res.latest_obs_time_utc
            now = datetime.now(timezone.utc)
            freshness_mins = max(0, int((now - latest_time).total_seconds() / 60))
            message = f"Live operational feed online. Freshness: ~{freshness_mins}m ago."
        except Exception as e:
            api_online = False
            message = f"Live feed offline or unreachable: {str(e)[:120]}. Fallback demo available."

        checkpoints = model_service.verify_checkpoints_status()

        return ForecastStatusResponse(
            live_api_online=api_online,
            demo_mode_available=True,
            latest_observation_time_utc=latest_time,
            data_freshness_minutes=freshness_mins,
            checkpoints_ready=checkpoints,
            message=message,
        )


live_forecast_service = LiveForecastService()

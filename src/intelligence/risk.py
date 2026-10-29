"""
Risk Assessment and Threshold Evaluation Subsystem for Ocean Intelligence.

Evaluates marine telemetry and model-generated wave forecasts against configurable
engineering risk thresholds to provide transparent, explainable decision support.

IMPORTANT NOTICES:
1. RISK CATEGORIES (GREEN, YELLOW, ORANGE, RED) are internal project engineering
   thresholds, NOT official maritime safety warnings or disaster management advisories.
2. GREEN indicates no configured warning threshold is exceeded in available inputs;
   it is NEVER a guarantee of maritime safety.
3. Official maritime advisories (e.g., INCOIS / IMD) are not integrated in this prototype.
4. Non-reference locations (all nodes other than Mumbai Offshore) carry unvalidated
   regional transfer status and must be compared with ECMWF NWP forecasts.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.dataset import TimeSeriesScaler
from src.data.live_service import (
    get_sea_state_classification,
    predict_live_forecast,
    validate_and_prepare_input_window,
)
from src.intelligence.nodes import CoastalNode, ValidationStatus
from src.intelligence.provider import DataStatus, ProviderFetchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RiskLevel(str, Enum):
    """Project-level engineering risk categories."""
    GREEN = "GREEN"    # No configured threshold exceeded (Caution: NOT a safety guarantee)
    YELLOW = "YELLOW"  # Caution threshold exceeded
    ORANGE = "ORANGE"  # High risk threshold exceeded
    RED = "RED"        # Critical risk threshold exceeded

    @property
    def display_name(self) -> str:
        names = {
            "GREEN": "Low Risk / Normal",
            "YELLOW": "Caution (Elevated Swell)",
            "ORANGE": "High Risk (Rough Seas)",
            "RED": "Critical Risk (Severe Wave Hazard)",
        }
        return names.get(self.value, self.value)

    @property
    def hex_color(self) -> str:
        colors = {
            "GREEN": "#10b981",   # Emerald Green
            "YELLOW": "#f59e0b",  # Amber Yellow
            "ORANGE": "#f97316",  # Coral Orange
            "RED": "#ef4444",     # Crimson Red
        }
        return colors.get(self.value, "#94a3b8")

    @property
    def bg_color(self) -> str:
        bgs = {
            "GREEN": "rgba(16, 185, 129, 0.15)",
            "YELLOW": "rgba(245, 158, 11, 0.18)",
            "ORANGE": "rgba(249, 115, 22, 0.20)",
            "RED": "rgba(239, 68, 68, 0.25)",
        }
        return bgs.get(self.value, "rgba(148, 163, 184, 0.15)")

    @property
    def severity_rank(self) -> int:
        ranks = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
        return ranks.get(self.value, 0)


class EventType(str, Enum):
    """Categorization of maritime assessment triggers."""
    PROJECT_FORECAST_RISK = "project_forecast_risk"
    OBSERVED_CONDITION_RISK = "observed_condition_risk"
    OFFICIAL_ADVISORY = "official_advisory"  # Reserved for authoritative feeds (none in prototype)


@dataclass(frozen=True)
class RiskThresholdSet:
    """
    Configurable significant wave height ($H_s$) risk thresholds in meters.

    Guarantees:
    - Stored outside core alert logic.
    - Strict validation: 0 < caution_m < high_risk_m < critical_m.
    - Explicit hysteresis buffer for alert resolution.
    - Explicitly labelled as demonstration engineering defaults.
    """
    caution_m: float = 1.50
    high_risk_m: float = 2.50
    critical_m: float = 4.00
    hysteresis_buffer_m: float = 0.15
    units: str = "meters (Significant Wave Height, Hs)"
    version: str = "demo-v1.0"
    is_demonstration: bool = True
    guidance_notice: str = (
        "Demonstration engineering thresholds for decision-support prototype. "
        "Not an official maritime safety threshold and not a substitute for national "
        "meteorological warnings (e.g., INCOIS/IMD)."
    )
    location_id: Optional[str] = None

    def __post_init__(self):
        # Validate positive numeric values
        for val_name, val in [
            ("caution_m", self.caution_m),
            ("high_risk_m", self.high_risk_m),
            ("critical_m", self.critical_m),
        ]:
            if not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(f"Threshold '{val_name}' must be a positive number, got {val}.")

        # Enforce strict ascending order
        if not (self.caution_m < self.high_risk_m < self.critical_m):
            raise ValueError(
                f"Thresholds must be strictly increasing: 0 < caution ({self.caution_m}) < "
                f"high_risk ({self.high_risk_m}) < critical ({self.critical_m})."
            )

        if not isinstance(self.hysteresis_buffer_m, (int, float)) or self.hysteresis_buffer_m < 0:
            raise ValueError("hysteresis_buffer_m must be non-negative.")

        # Ensure hysteresis buffer does not exceed interval spans
        min_interval = min(self.high_risk_m - self.caution_m, self.critical_m - self.high_risk_m)
        if self.hysteresis_buffer_m >= min_interval:
            raise ValueError(
                f"hysteresis_buffer_m ({self.hysteresis_buffer_m}) must be smaller than "
                f"the minimum threshold interval ({min_interval})."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "caution_m": self.caution_m,
            "high_risk_m": self.high_risk_m,
            "critical_m": self.critical_m,
            "hysteresis_buffer_m": self.hysteresis_buffer_m,
            "units": self.units,
            "version": self.version,
            "is_demonstration": self.is_demonstration,
            "guidance_notice": self.guidance_notice,
            "location_id": self.location_id,
        }


class ThresholdConfigManager:
    """Manages global default and per-location threshold configurations."""

    def __init__(self, default_thresholds: Optional[RiskThresholdSet] = None):
        self._default = default_thresholds or RiskThresholdSet()
        self._overrides: Dict[str, RiskThresholdSet] = {}

    def get_thresholds(self, location_id: Optional[str] = None) -> RiskThresholdSet:
        if location_id and location_id in self._overrides:
            return self._overrides[location_id]
        return self._default

    def set_override(self, location_id: str, thresholds: RiskThresholdSet) -> None:
        if not isinstance(thresholds, RiskThresholdSet):
            raise TypeError("thresholds must be a RiskThresholdSet instance.")
        self._overrides[location_id] = thresholds

    def remove_override(self, location_id: str) -> None:
        self._overrides.pop(location_id, None)

    @property
    def default_thresholds(self) -> RiskThresholdSet:
        return self._default


# Global threshold manager singleton
GLOBAL_THRESHOLD_MANAGER = ThresholdConfigManager()


def get_threshold_manager() -> ThresholdConfigManager:
    return GLOBAL_THRESHOLD_MANAGER


def classify_risk(wave_height_m: float, thresholds: RiskThresholdSet) -> Tuple[RiskLevel, Optional[float]]:
    """
    Classify significant wave height against configured thresholds.

    Returns
    -------
    Tuple[RiskLevel, Optional[float]]
        The assigned RiskLevel and the specific threshold value that was crossed (or None for GREEN).
    """
    if wave_height_m >= thresholds.critical_m:
        return RiskLevel.RED, thresholds.critical_m
    elif wave_height_m >= thresholds.high_risk_m:
        return RiskLevel.ORANGE, thresholds.high_risk_m
    elif wave_height_m >= thresholds.caution_m:
        return RiskLevel.YELLOW, thresholds.caution_m
    else:
        return RiskLevel.GREEN, None


@dataclass
class RiskAssessment:
    """
    Structured, explainable evaluation of wave hazard for a single point in time.
    """
    location_id: str
    location_name: str
    assessment_time_utc: datetime
    data_retrieval_time_utc: datetime
    forecast_valid_time_utc: datetime
    forecast_horizon_hours: int
    forecast_issue_time_utc: Optional[datetime]
    event_type: EventType
    risk_level: RiskLevel
    threshold_crossed_m: Optional[float]
    threshold_config_version: str
    predicted_swh_m: float
    observed_persistence_swh_m: Optional[float]
    external_nwp_swh_m: Optional[float]
    source_name: str
    data_status: DataStatus
    model_name: Optional[str]
    is_reference: bool
    validation_status: ValidationStatus
    trigger_explanation: str
    limitations_and_uncertainty: List[str] = field(default_factory=list)
    is_degraded_data: bool = False

    @property
    def is_warning_triggered(self) -> bool:
        """True if caution, high, or critical risk threshold was exceeded."""
        return self.risk_level in [RiskLevel.YELLOW, RiskLevel.ORANGE, RiskLevel.RED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location_id": self.location_id,
            "location_name": self.location_name,
            "assessment_time_utc": self.assessment_time_utc.isoformat(),
            "data_retrieval_time_utc": self.data_retrieval_time_utc.isoformat(),
            "forecast_valid_time_utc": self.forecast_valid_time_utc.isoformat(),
            "forecast_horizon_hours": self.forecast_horizon_hours,
            "forecast_issue_time_utc": self.forecast_issue_time_utc.isoformat() if self.forecast_issue_time_utc else None,
            "event_type": self.event_type.value,
            "risk_level": self.risk_level.value,
            "risk_display_name": self.risk_level.display_name,
            "threshold_crossed_m": self.threshold_crossed_m,
            "threshold_config_version": self.threshold_config_version,
            "predicted_swh_m": round(self.predicted_swh_m, 3),
            "observed_persistence_swh_m": round(self.observed_persistence_swh_m, 3) if self.observed_persistence_swh_m is not None else None,
            "external_nwp_swh_m": round(self.external_nwp_swh_m, 3) if self.external_nwp_swh_m is not None else None,
            "source_name": self.source_name,
            "data_status": self.data_status.value,
            "model_name": self.model_name,
            "is_reference": self.is_reference,
            "validation_status": self.validation_status.value,
            "trigger_explanation": self.trigger_explanation,
            "limitations_and_uncertainty": list(self.limitations_and_uncertainty),
            "is_degraded_data": self.is_degraded_data,
        }


def assess_node_risk(
    node: CoastalNode,
    fetch_result: ProviderFetchResult,
    scaler: Optional[TimeSeriesScaler] = None,
    thresholds: Optional[RiskThresholdSet] = None,
    horizons: Optional[List[int]] = None,
    model_type: str = "lstm",
    models_dir: Optional[Union[str, Path]] = None,
) -> List[RiskAssessment]:
    """
    Evaluate wave hazard risk for a coastal node across current observations and future horizons.

    Guarantees:
    - Reuses existing provider results, feature preprocessing, and model inference.
    - Generates distinct assessments for observed conditions (h=0) and forecast lead times (h=1, 6, 24).
    - Degraded, stale, or unavailable telemetry is explicitly marked and never reported as safe GREEN.
    - Non-reference nodes are prominently tagged with unvalidated regional transfer limitations.
    - Output timestamps are strictly derived from verified anchors and horizons.

    Parameters
    ----------
    node : CoastalNode
        Monitored coastal location.
    fetch_result : ProviderFetchResult
        Raw telemetry from live or demo provider.
    scaler : Optional[TimeSeriesScaler]
        Fitted training scaler artifact.
    thresholds : Optional[RiskThresholdSet]
        Active threshold set (defaults to global manager configuration).
    horizons : Optional[List[int]]
        Forecast horizons to evaluate (default: [1, 6, 24]).
    model_type : str
        Model architecture code (default: 'lstm').
    models_dir : Optional[Union[str, Path]]
        Directory containing model checkpoints.

    Returns
    -------
    List[RiskAssessment]
        List of structured risk assessments.
    """
    now_utc = datetime.now(timezone.utc)
    target_horizons = horizons or [1, 6, 24]
    active_thresholds = thresholds or get_threshold_manager().get_thresholds(node.location_id)
    assessments: List[RiskAssessment] = []

    # 1. Handle unavailable or empty provider telemetry
    if not fetch_result.is_success or fetch_result.raw_df is None or fetch_result.raw_df.empty:
        reason = fetch_result.error_message or "Telemetry feed is unavailable or returned empty data."
        assessments.append(
            RiskAssessment(
                location_id=node.location_id,
                location_name=node.name,
                assessment_time_utc=now_utc,
                data_retrieval_time_utc=fetch_result.retrieval_time_utc,
                forecast_valid_time_utc=now_utc,
                forecast_horizon_hours=0,
                forecast_issue_time_utc=fetch_result.forecast_issue_time_utc,
                event_type=EventType.OBSERVED_CONDITION_RISK,
                risk_level=RiskLevel.GREEN,
                threshold_crossed_m=None,
                threshold_config_version=active_thresholds.version,
                predicted_swh_m=0.0,
                observed_persistence_swh_m=None,
                external_nwp_swh_m=None,
                source_name=fetch_result.provider_name,
                data_status=fetch_result.data_status,
                model_name=None,
                is_reference=node.is_reference,
                validation_status=node.validation_status,
                trigger_explanation=f"Telemetry unavailable: {reason}",
                limitations_and_uncertainty=[
                    "DATA UNAVAILABLE: Provider query failed or timed out.",
                    "Cannot verify ocean conditions; GREEN category indicates absence of data, NOT guaranteed safety.",
                    active_thresholds.guidance_notice,
                ],
                is_degraded_data=True,
            )
        )
        return assessments

    # 2. Extract and validate 24h input window
    try:
        raw_df = fetch_result.raw_df.copy()
        current_hour_utc = fetch_result.retrieval_time_utc.replace(minute=0, second=0, microsecond=0)
        hist_rows = raw_df[raw_df["valid_time"] <= current_hour_utc]
        if hist_rows.empty:
            anchor_time = raw_df["valid_time"].max()
        else:
            anchor_time = hist_rows["valid_time"].max()

        window_df, actual_anchor = validate_and_prepare_input_window(
            raw_df,
            anchor_time=anchor_time,
            window_size=24,
        )
    except Exception as val_exc:
        # Quality gate failure (e.g. gaps, NaNs, missing columns)
        assessments.append(
            RiskAssessment(
                location_id=node.location_id,
                location_name=node.name,
                assessment_time_utc=now_utc,
                data_retrieval_time_utc=fetch_result.retrieval_time_utc,
                forecast_valid_time_utc=now_utc,
                forecast_horizon_hours=0,
                forecast_issue_time_utc=fetch_result.forecast_issue_time_utc,
                event_type=EventType.OBSERVED_CONDITION_RISK,
                risk_level=RiskLevel.GREEN,
                threshold_crossed_m=None,
                threshold_config_version=active_thresholds.version,
                predicted_swh_m=0.0,
                observed_persistence_swh_m=None,
                external_nwp_swh_m=None,
                source_name=fetch_result.provider_name,
                data_status=DataStatus.INVALID,
                model_name=None,
                is_reference=node.is_reference,
                validation_status=node.validation_status,
                trigger_explanation=f"Telemetry failed quality gates: {val_exc}",
                limitations_and_uncertainty=[
                    f"QUALITY REJECTION: {val_exc}",
                    "Assessment aborted due to input data discontinuity or missing variables.",
                    "GREEN category indicates untestable telemetry, NOT guaranteed maritime safety.",
                    active_thresholds.guidance_notice,
                ],
                is_degraded_data=True,
            )
        )
        return assessments

    is_degraded = fetch_result.data_status in [DataStatus.STALE, DataStatus.INVALID]
    current_obs_swh = float(window_df["swh"].iloc[-1])
    obs_time = window_df["valid_time"].iloc[-1]
    if isinstance(obs_time, pd.Timestamp):
        obs_time_dt = obs_time.to_pydatetime()
    else:
        obs_time_dt = obs_time

    # Base common limitations
    common_limitations: List[str] = [active_thresholds.guidance_notice]
    if not node.is_reference:
        common_limitations.append(
            f"GEOGRAPHIC TRANSFER NOTICE: Models were trained on historical ERA5 data at Mumbai Offshore "
            f"(18.50°N, 72.50°E). Forecasts at {node.name} ({node.coordinate_label}) are experimental "
            f"regional estimates without localized bathymetric calibration."
        )
    if fetch_result.is_demo:
        common_limitations.append(
            "OFFLINE REPLAY NOTICE: Telemetry is replayed from historical ERA5 ground-truth archives. "
            "This is not a live real-time observation."
        )
    if fetch_result.data_status == DataStatus.STALE:
        common_limitations.append(
            "DATA FRESHNESS NOTICE: Telemetry exceeds nominal 3.5h freshness interval; assessment may lag actual conditions."
        )

    # 3. Assess observed condition (horizon = 0)
    obs_risk, obs_thresh = classify_risk(current_obs_swh, active_thresholds)
    obs_sea_desc, _ = get_sea_state_classification(current_obs_swh)
    obs_explanation = (
        f"Observed Significant Wave Height is {current_obs_swh:.2f} m ({obs_sea_desc} sea state) "
        f"at {obs_time_dt.strftime('%Y-%m-%d %H:%M UTC')}. "
    )
    if obs_thresh is not None:
        obs_explanation += f"Exceeds configured {obs_risk.display_name} threshold ({obs_thresh:.2f} m)."
    else:
        obs_explanation += f"Below caution threshold ({active_thresholds.caution_m:.2f} m)."

    assessments.append(
        RiskAssessment(
            location_id=node.location_id,
            location_name=node.name,
            assessment_time_utc=now_utc,
            data_retrieval_time_utc=fetch_result.retrieval_time_utc,
            forecast_valid_time_utc=obs_time_dt,
            forecast_horizon_hours=0,
            forecast_issue_time_utc=fetch_result.forecast_issue_time_utc,
            event_type=EventType.OBSERVED_CONDITION_RISK,
            risk_level=obs_risk,
            threshold_crossed_m=obs_thresh,
            threshold_config_version=active_thresholds.version,
            predicted_swh_m=current_obs_swh,
            observed_persistence_swh_m=current_obs_swh,
            external_nwp_swh_m=None,
            source_name=fetch_result.provider_name,
            data_status=fetch_result.data_status,
            model_name="Observed Telemetry",
            is_reference=node.is_reference,
            validation_status=node.validation_status,
            trigger_explanation=obs_explanation,
            limitations_and_uncertainty=list(common_limitations),
            is_degraded_data=is_degraded,
        )
    )

    # 4. Assess each forecast horizon (H = 1, 6, 24)
    # Scaler fallback if not provided
    if scaler is None:
        scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
        if scaler_path.is_file():
            scaler = TimeSeriesScaler.load(scaler_path)

    for h in target_horizons:
        nwp_val = fetch_result.external_nwp_forecasts.get(h)
        target_valid_time = actual_anchor + pd.Timedelta(hours=h)
        if isinstance(target_valid_time, pd.Timestamp):
            target_valid_dt = target_valid_time.to_pydatetime()
        else:
            target_valid_dt = target_valid_time

        pred_swh: float = current_obs_swh
        used_model_name: str = "Persistence Baseline"

        if scaler is not None:
            try:
                forecast_res = predict_live_forecast(
                    prepared_df=window_df,
                    anchor_time=actual_anchor,
                    horizon=h,
                    model_type=model_type,
                    scaler=scaler,
                    models_dir=models_dir,
                    external_nwp_forecast=nwp_val,
                    is_demo=fetch_result.is_demo,
                    location_id=node.location_id,
                    validation_status=node.validation_status.value,
                )
                pred_swh = float(forecast_res.forecast_swh_m)
                used_model_name = forecast_res.model_name
            except Exception as infer_err:
                # If model inference fails, fallback to NWP or persistence with limitation notice
                if nwp_val is not None:
                    pred_swh = float(nwp_val)
                    used_model_name = "ECMWF NWP Benchmark"
                common_limitations.append(f"Inference error for horizon +{h}h: {infer_err}")

        # Classify forecast risk
        fc_risk, fc_thresh = classify_risk(pred_swh, active_thresholds)
        sea_desc, _ = get_sea_state_classification(pred_swh)

        explanation = (
            f"{used_model_name} predicts Significant Wave Height of {pred_swh:.2f} m ({sea_desc} sea state) "
            f"at lead time +{h}h (valid: {target_valid_dt.strftime('%Y-%m-%d %H:%M UTC')}). "
        )
        if fc_thresh is not None:
            explanation += f"Crosses configured {fc_risk.display_name} threshold ({fc_thresh:.2f} m)."
        else:
            explanation += f"Remains below caution threshold ({active_thresholds.caution_m:.2f} m)."

        if nwp_val is not None:
            explanation += f" (ECMWF NWP comparison: {nwp_val:.2f} m)."

        h_limitations = list(common_limitations)
        h_limitations.append(
            f"DISCRETE PREDICTION NOTICE: Single +{h}h horizon prediction. Does not model sudden sub-hourly wave surges."
        )

        assessments.append(
            RiskAssessment(
                location_id=node.location_id,
                location_name=node.name,
                assessment_time_utc=now_utc,
                data_retrieval_time_utc=fetch_result.retrieval_time_utc,
                forecast_valid_time_utc=target_valid_dt,
                forecast_horizon_hours=h,
                forecast_issue_time_utc=fetch_result.forecast_issue_time_utc,
                event_type=EventType.PROJECT_FORECAST_RISK,
                risk_level=fc_risk,
                threshold_crossed_m=fc_thresh,
                threshold_config_version=active_thresholds.version,
                predicted_swh_m=pred_swh,
                observed_persistence_swh_m=current_obs_swh,
                external_nwp_swh_m=nwp_val,
                source_name=fetch_result.provider_name,
                data_status=fetch_result.data_status,
                model_name=used_model_name,
                is_reference=node.is_reference,
                validation_status=node.validation_status,
                trigger_explanation=explanation,
                limitations_and_uncertainty=h_limitations,
                is_degraded_data=is_degraded,
            )
        )

    return assessments

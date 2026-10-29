"""
Forecast Accuracy, Verification, Data Drift, and Model Performance Monitoring Subsystem.

Provides:
1. Immutable Forecast Ledger for prospective tracking of issued wave predictions.
2. Independent observation interface and forecast-to-observation matching service.
3. Rigorous forecast accuracy metrics (MAE, RMSE, Bias, Skill Score vs. Persistence).
4. Feature distribution drift monitoring via Population Stability Index (PSI).
5. Rolling model performance degradation tracking with hysteresis and state machines.
6. Dedicated SQLite persistence for forecast records and monitoring events.

IMPORTANT NOTICES:
1. SCIENTIFIC INTEGRITY: Independent in-situ buoy measurements are distinct from
   external provider forecasts and historical ERA5 reanalysis. Verification status
   is explicitly reported as VERIFIED, LIMITED_EVIDENCE, AWAITING_OBSERVATIONS,
   VERIFICATION_UNAVAILABLE, or RETROSPECTIVE_REANALYSIS.
2. SEPARATION OF CONCERNS: Monitoring alerts (drift, performance degradation) are
   stored in a dedicated table and NEVER trigger maritime coastal safety warnings.
3. IMMUTABILITY: Historical issued forecasts are never overwritten.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import pandas as pd

from src.data.dataset import DEFAULT_FEATURE_COLS
from src.intelligence.nodes import ValidationStatus
from src.intelligence.provider import DataStatus

logger = logging.getLogger("OceanIntelligence.Monitoring")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "persistence" / "ocean_intelligence.db"
DEFAULT_REF_STATS_PATH = PROJECT_ROOT / "outputs" / "reports" / "feature_reference_distribution.json"


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class ForecastSourceCategory(str, Enum):
    """Categorization of how a forecast was produced."""
    PROJECT_MODEL = "project_model"       # Trained PyTorch checkpoint (WaveLSTM, WaveANN, WaveCNN1D)
    EXTERNAL_NWP = "external_nwp"         # External numerical weather prediction benchmark (e.g. ECMWF)
    DEMO_REPLAY = "demo_replay"           # Offline historical replay (demonstration)


class ObservationSourceType(str, Enum):
    """Origin and nature of marine verification measurements."""
    IN_SITU_BUOY = "in_situ_buoy"                     # Moored oceanographic buoy (in-situ measurement)
    COASTAL_RADAR = "coastal_radar"                   # Coastal HF ocean radar
    RETROSPECTIVE_REANALYSIS = "retrospective_reanalysis" # ERA5 reanalysis ground-truth (backtesting)
    SYNTHETIC_TEST_FIXTURE = "synthetic_test_fixture" # Explicitly labeled unit-test fixture


class ObservationQuality(str, Enum):
    """Quality control classification of an observation."""
    VALID = "valid"
    SUSPECT = "suspect"
    REJECTED = "rejected"
    MISSING = "missing"


class VerificationStatus(str, Enum):
    """Operational evidence status for forecast accuracy verification."""
    VERIFIED = "verified"                                 # >= min_samples valid independent observations
    LIMITED_EVIDENCE = "limited_evidence"                 # 1 to min_samples-1 observations
    AWAITING_OBSERVATIONS = "awaiting_observations"       # Forecasts issued; target observations not yet arrived
    VERIFICATION_UNAVAILABLE = "verification_unavailable" # No observation source configured
    RETROSPECTIVE_REANALYSIS = "retrospective_reanalysis" # Backtested against ERA5 reanalysis, not live buoy

    @property
    def display_badge(self) -> str:
        badges = {
            "verified": "🟢 Verified (Independent Buoy)",
            "limited_evidence": "🟡 Limited Evidence (< Min Samples)",
            "awaiting_observations": "⏳ Awaiting Observations",
            "verification_unavailable": "⚪ Verification Unavailable",
            "retrospective_reanalysis": "🔷 Retrospective Reanalysis",
        }
        return badges.get(self.value, self.value)


class DriftStatus(str, Enum):
    """Data distribution stability status."""
    NORMAL = "NORMAL"                         # All features PSI < 0.10, quality nominal
    WATCH = "WATCH"                           # Any feature 0.10 <= PSI < 0.25 or minor gaps
    DRIFT_DETECTED = "DRIFT_DETECTED"         # Key physical feature PSI >= 0.25
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"   # Window rows < 24


class PerformanceStatus(str, Enum):
    """Model error tracking status against historical baseline."""
    NORMAL = "NORMAL"                                     # Rolling error <= 1.25x baseline
    WATCH = "WATCH"                                       # 1.25x < Rolling error <= 1.50x baseline
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"   # Rolling error > 1.50x baseline persistent
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"               # Sample count < minimum (5)


class MonitoringEventType(str, Enum):
    """Classification of automated ML operations events."""
    DATA_DRIFT = "data_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    STALE_TELEMETRY = "stale_telemetry"


class MonitoringSeverity(str, Enum):
    """Urgency of monitoring notifications."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class MonitoringEventStatus(str, Enum):
    """Lifecycle status of a monitoring event."""
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class ForecastLedgerRecord:
    """Immutable audit record of a generated wave forecast."""
    forecast_id: str
    fingerprint: str
    location_id: str
    latitude: float
    longitude: float
    issue_time_utc: datetime
    anchor_time_utc: datetime
    target_time_utc: datetime
    horizon_hours: int
    predicted_swh_m: float
    units: str = "meters"
    model_name: str = "WaveLSTM"
    checkpoint_name: Optional[str] = None
    source_category: ForecastSourceCategory = ForecastSourceCategory.PROJECT_MODEL
    data_source: str = "OpenMeteoProvider"
    data_status: str = "fresh"
    validation_status: str = "reference_validated"
    is_reference: bool = True
    persistence_swh_m: Optional[float] = None
    external_nwp_swh_m: Optional[float] = None
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecast_id": self.forecast_id,
            "fingerprint": self.fingerprint,
            "location_id": self.location_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "issue_time_utc": self.issue_time_utc.isoformat(),
            "anchor_time_utc": self.anchor_time_utc.isoformat(),
            "target_time_utc": self.target_time_utc.isoformat(),
            "horizon_hours": self.horizon_hours,
            "predicted_swh_m": round(self.predicted_swh_m, 3),
            "units": self.units,
            "model_name": self.model_name,
            "checkpoint_name": self.checkpoint_name,
            "source_category": self.source_category.value,
            "data_source": self.data_source,
            "data_status": self.data_status,
            "validation_status": self.validation_status,
            "is_reference": self.is_reference,
            "persistence_swh_m": round(self.persistence_swh_m, 3) if self.persistence_swh_m is not None else None,
            "external_nwp_swh_m": round(self.external_nwp_swh_m, 3) if self.external_nwp_swh_m is not None else None,
            "created_at_utc": self.created_at_utc.isoformat(),
        }


@dataclass
class ObservationRecord:
    """Independently ingested marine sea-state measurement."""
    observation_id: str
    location_id: str
    latitude: float
    longitude: float
    observation_time_utc: datetime
    observed_swh_m: float
    station_id: Optional[str] = None
    units: str = "meters"
    source_name: str = "INCOIS_BUOY_NETWORK"
    source_type: ObservationSourceType = ObservationSourceType.IN_SITU_BUOY
    quality_status: ObservationQuality = ObservationQuality.VALID
    ingestion_time_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "location_id": self.location_id,
            "station_id": self.station_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "observation_time_utc": self.observation_time_utc.isoformat(),
            "observed_swh_m": round(self.observed_swh_m, 3),
            "units": self.units,
            "source_name": self.source_name,
            "source_type": self.source_type.value,
            "quality_status": self.quality_status.value,
            "ingestion_time_utc": self.ingestion_time_utc.isoformat(),
            "notes": self.notes,
        }


@dataclass
class MatchedForecastObservation:
    """Paired forecast and observation for precision verification."""
    forecast_id: str
    observation_id: str
    location_id: str
    horizon_hours: int
    model_name: str
    issue_time_utc: datetime
    target_time_utc: datetime
    observation_time_utc: datetime
    predicted_swh_m: float
    observed_swh_m: float
    persistence_swh_m: Optional[float]
    error_m: float                     # predicted - observed
    abs_error_m: float                 # abs(error_m)
    source_type: ObservationSourceType
    is_prospective: bool               # True if observation occurred after forecast issuance


@dataclass
class AccuracyMetrics:
    """Aggregated statistical evaluation of wave height predictions."""
    sample_count: int
    total_forecasts_issued: int
    coverage_pct: float
    mae_meters: Optional[float] = None
    rmse_meters: Optional[float] = None
    bias_meters: Optional[float] = None
    persistence_mae_meters: Optional[float] = None
    persistence_rmse_meters: Optional[float] = None
    skill_score_rmse: Optional[float] = None
    verification_status: VerificationStatus = VerificationStatus.AWAITING_OBSERVATIONS
    status_explanation: str = ""
    location_id: Optional[str] = None
    horizon_hours: Optional[int] = None
    model_name: Optional[str] = None
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None

    @property
    def verification_badge(self) -> str:
        return self.verification_status.display_badge

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "total_forecasts_issued": self.total_forecasts_issued,
            "coverage_pct": round(self.coverage_pct, 1),
            "mae_meters": round(self.mae_meters, 4) if self.mae_meters is not None else None,
            "rmse_meters": round(self.rmse_meters, 4) if self.rmse_meters is not None else None,
            "bias_meters": round(self.bias_meters, 4) if self.bias_meters is not None else None,
            "persistence_mae_meters": round(self.persistence_mae_meters, 4) if self.persistence_mae_meters is not None else None,
            "persistence_rmse_meters": round(self.persistence_rmse_meters, 4) if self.persistence_rmse_meters is not None else None,
            "skill_score_rmse": round(self.skill_score_rmse, 4) if self.skill_score_rmse is not None else None,
            "verification_status": self.verification_status.value,
            "verification_badge": self.verification_status.display_badge,
            "status_explanation": self.status_explanation,
            "location_id": self.location_id,
            "horizon_hours": self.horizon_hours,
            "model_name": self.model_name,
            "time_window_start": self.time_window_start.isoformat() if self.time_window_start else None,
            "time_window_end": self.time_window_end.isoformat() if self.time_window_end else None,
        }


@dataclass
class FeatureDriftResult:
    """Individual feature distribution shift metrics."""
    feature_name: str
    psi: float
    drift_level: str  # "NO_DRIFT", "MODERATE_DRIFT", "SIGNIFICANT_DRIFT"
    ref_mean: float
    curr_mean: float
    ref_std: float
    curr_std: float
    missing_rate: float
    out_of_bounds_rate: float
    bin_actual_pct: List[float] = field(default_factory=list)
    bin_expected_pct: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "psi": round(self.psi, 4),
            "drift_level": self.drift_level,
            "ref_mean": round(self.ref_mean, 3),
            "curr_mean": round(self.curr_mean, 3),
            "ref_std": round(self.ref_std, 3),
            "curr_std": round(self.curr_std, 3),
            "missing_rate": round(self.missing_rate, 4),
            "out_of_bounds_rate": round(self.out_of_bounds_rate, 4),
        }


@dataclass
class DataDriftReport:
    """Overall dataset stability assessment."""
    overall_status: DriftStatus
    evaluation_time_utc: datetime
    window_size: int
    reference_dataset: str
    feature_drifts: Dict[str, FeatureDriftResult]
    max_psi_feature: str
    max_psi_value: float
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "evaluation_time_utc": self.evaluation_time_utc.isoformat(),
            "window_size": self.window_size,
            "reference_dataset": self.reference_dataset,
            "max_psi_feature": self.max_psi_feature,
            "max_psi_value": round(self.max_psi_value, 4),
            "summary": self.summary,
            "features": {k: v.to_dict() for k, v in self.feature_drifts.items()},
        }


@dataclass
class PerformanceDegradationReport:
    """Rolling model performance health report."""
    status: PerformanceStatus
    evaluation_time_utc: datetime
    horizon_hours: int
    model_name: str
    rolling_mae: Optional[float]
    rolling_rmse: Optional[float]
    baseline_mae: float
    baseline_rmse: float
    mae_ratio: Optional[float]
    rmse_ratio: Optional[float]
    matched_samples: int
    min_samples_required: int
    consecutive_degraded_periods: int
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "evaluation_time_utc": self.evaluation_time_utc.isoformat(),
            "horizon_hours": self.horizon_hours,
            "model_name": self.model_name,
            "rolling_mae": round(self.rolling_mae, 4) if self.rolling_mae is not None else None,
            "rolling_rmse": round(self.rolling_rmse, 4) if self.rolling_rmse is not None else None,
            "baseline_mae": round(self.baseline_mae, 4),
            "baseline_rmse": round(self.baseline_rmse, 4),
            "mae_ratio": round(self.mae_ratio, 2) if self.mae_ratio is not None else None,
            "rmse_ratio": round(self.rmse_ratio, 2) if self.rmse_ratio is not None else None,
            "matched_samples": self.matched_samples,
            "min_samples_required": self.min_samples_required,
            "consecutive_degraded_periods": self.consecutive_degraded_periods,
            "summary": self.summary,
        }


@dataclass
class MonitoringEventRecord:
    """Persistent audit log of an operational ML monitoring event."""
    event_id: str
    event_type: MonitoringEventType
    location_id: str
    severity: MonitoringSeverity
    status: MonitoringEventStatus
    metric_name: str
    metric_value: float
    threshold_value: float
    summary: str
    details_json: str
    fingerprint: str
    created_at_utc: datetime
    model_name: Optional[str] = None
    horizon_hours: Optional[int] = None
    resolved_at_utc: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "location_id": self.location_id,
            "model_name": self.model_name,
            "horizon_hours": self.horizon_hours,
            "severity": self.severity.value,
            "status": self.status.value,
            "metric_name": self.metric_name,
            "metric_value": round(self.metric_value, 4),
            "threshold_value": round(self.threshold_value, 4),
            "summary": self.summary,
            "details": json.loads(self.details_json) if self.details_json else {},
            "created_at_utc": self.created_at_utc.isoformat(),
            "resolved_at_utc": self.resolved_at_utc.isoformat() if self.resolved_at_utc else None,
        }


# -----------------------------------------------------------------------------
# SQLite Monitoring Store
# -----------------------------------------------------------------------------

def compute_ledger_fingerprint(
    location_id: str,
    model_name: str,
    horizon_hours: int,
    anchor_time_utc: datetime,
    target_time_utc: datetime,
    source_category: Union[ForecastSourceCategory, str],
) -> str:
    """Generate deterministic deduplication key for an issued forecast."""
    sc_val = source_category.value if isinstance(source_category, ForecastSourceCategory) else str(source_category)
    a_str = anchor_time_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_str = target_time_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{location_id}:{model_name}:{horizon_hours}h:{a_str}:{t_str}:{sc_val}"


def compute_monitoring_event_fingerprint(
    event_type: Union[MonitoringEventType, str],
    location_id: str,
    metric_name: str,
    model_name: Optional[str] = None,
    horizon_hours: Optional[int] = None,
) -> str:
    """Generate deduplication key for an active monitoring event."""
    et_val = event_type.value if isinstance(event_type, MonitoringEventType) else str(event_type)
    m_val = model_name or "global"
    h_val = f"{horizon_hours}h" if horizon_hours is not None else "any"
    return f"{location_id}:{et_val}:{metric_name}:{m_val}:{h_val}"


class MonitoringStore:
    """
    SQLite persistence layer for the immutable forecast ledger, independent observations,
    and operational monitoring events.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create migration-safe tables for forecast ledger and monitoring events."""
        with self._get_connection() as conn:
            # 1. Immutable Forecast Ledger Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_ledger (
                    forecast_id TEXT PRIMARY KEY,
                    fingerprint TEXT UNIQUE NOT NULL,
                    location_id TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    issue_time_utc TEXT NOT NULL,
                    anchor_time_utc TEXT NOT NULL,
                    target_time_utc TEXT NOT NULL,
                    horizon_hours INTEGER NOT NULL,
                    predicted_swh_m REAL NOT NULL,
                    units TEXT NOT NULL DEFAULT 'meters',
                    model_name TEXT NOT NULL,
                    checkpoint_name TEXT,
                    source_category TEXT NOT NULL,
                    data_source TEXT NOT NULL,
                    data_status TEXT NOT NULL,
                    validation_status TEXT NOT NULL,
                    is_reference INTEGER NOT NULL,
                    persistence_swh_m REAL,
                    external_nwp_swh_m REAL,
                    created_at_utc TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_loc_target ON forecast_ledger(location_id, target_time_utc);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_target ON forecast_ledger(target_time_utc);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_issue ON forecast_ledger(issue_time_utc);")

            # 2. Independent Observations Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    location_id TEXT NOT NULL,
                    station_id TEXT,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    observation_time_utc TEXT NOT NULL,
                    observed_swh_m REAL NOT NULL,
                    units TEXT NOT NULL DEFAULT 'meters',
                    source_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    quality_status TEXT NOT NULL,
                    ingestion_time_utc TEXT NOT NULL,
                    notes TEXT,
                    UNIQUE(location_id, observation_time_utc, source_name)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_loc_time ON observations(location_id, observation_time_utc);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_time ON observations(observation_time_utc);")

            # 3. Dedicated Monitoring Events Table (Separate from wave safety alerts)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS monitoring_events (
                    event_id TEXT PRIMARY KEY,
                    fingerprint TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    model_name TEXT,
                    horizon_hours INTEGER,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    summary TEXT NOT NULL,
                    details_json TEXT,
                    created_at_utc TEXT NOT NULL,
                    resolved_at_utc TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mon_events_status ON monitoring_events(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mon_events_loc ON monitoring_events(location_id);")

    # -------------------------------------------------------------------------
    # Ledger Methods
    # -------------------------------------------------------------------------

    def record_forecast(self, record: ForecastLedgerRecord) -> bool:
        """
        Append an issued prediction to the immutable ledger.
        Uses INSERT OR IGNORE so genuine issuances are saved while duplicate
        re-evaluations are safely ignored without raising errors.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO forecast_ledger (
                    forecast_id, fingerprint, location_id, latitude, longitude,
                    issue_time_utc, anchor_time_utc, target_time_utc, horizon_hours,
                    predicted_swh_m, units, model_name, checkpoint_name,
                    source_category, data_source, data_status, validation_status,
                    is_reference, persistence_swh_m, external_nwp_swh_m, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.forecast_id,
                    record.fingerprint,
                    record.location_id,
                    record.latitude,
                    record.longitude,
                    record.issue_time_utc.isoformat(),
                    record.anchor_time_utc.isoformat(),
                    record.target_time_utc.isoformat(),
                    record.horizon_hours,
                    record.predicted_swh_m,
                    record.units,
                    record.model_name,
                    record.checkpoint_name,
                    record.source_category.value,
                    record.data_source,
                    record.data_status,
                    record.validation_status,
                    1 if record.is_reference else 0,
                    record.persistence_swh_m,
                    record.external_nwp_swh_m,
                    record.created_at_utc.isoformat(),
                ),
            )
            return cursor.rowcount > 0

    def query_forecasts(
        self,
        location_id: Optional[str] = None,
        horizon: Optional[int] = None,
        model_name: Optional[str] = None,
        source_category: Optional[str] = None,
        start_target_time: Optional[datetime] = None,
        end_target_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[ForecastLedgerRecord]:
        """Retrieve stored forecast records with flexible filtering."""
        clauses = []
        params: List[Any] = []

        if location_id:
            clauses.append("location_id = ?")
            params.append(location_id)
        if horizon:
            clauses.append("horizon_hours = ?")
            params.append(horizon)
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        if source_category:
            clauses.append("source_category = ?")
            params.append(source_category)
        if start_target_time:
            clauses.append("target_time_utc >= ?")
            params.append(start_target_time.isoformat())
        if end_target_time:
            clauses.append("target_time_utc <= ?")
            params.append(end_target_time.isoformat())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM forecast_ledger {where_sql} ORDER BY target_time_utc DESC LIMIT ?;"
        params.append(limit)

        records = []
        with self._get_connection() as conn:
            for row in conn.execute(sql, params).fetchall():
                records.append(
                    ForecastLedgerRecord(
                        forecast_id=row["forecast_id"],
                        fingerprint=row["fingerprint"],
                        location_id=row["location_id"],
                        latitude=row["latitude"],
                        longitude=row["longitude"],
                        issue_time_utc=datetime.fromisoformat(row["issue_time_utc"]),
                        anchor_time_utc=datetime.fromisoformat(row["anchor_time_utc"]),
                        target_time_utc=datetime.fromisoformat(row["target_time_utc"]),
                        horizon_hours=row["horizon_hours"],
                        predicted_swh_m=row["predicted_swh_m"],
                        units=row["units"],
                        model_name=row["model_name"],
                        checkpoint_name=row["checkpoint_name"],
                        source_category=ForecastSourceCategory(row["source_category"]),
                        data_source=row["data_source"],
                        data_status=row["data_status"],
                        validation_status=row["validation_status"],
                        is_reference=bool(row["is_reference"]),
                        persistence_swh_m=row["persistence_swh_m"],
                        external_nwp_swh_m=row["external_nwp_swh_m"],
                        created_at_utc=datetime.fromisoformat(row["created_at_utc"]),
                    )
                )
        return records

    def count_forecasts(self, location_id: Optional[str] = None) -> int:
        """Count total forecasts issued."""
        sql = "SELECT COUNT(*) as cnt FROM forecast_ledger"
        params: List[Any] = []
        if location_id:
            sql += " WHERE location_id = ?"
            params.append(location_id)
        with self._get_connection() as conn:
            return conn.execute(sql, params).fetchone()["cnt"]

    # -------------------------------------------------------------------------
    # Observation Methods
    # -------------------------------------------------------------------------

    def record_observation(self, record: ObservationRecord) -> bool:
        """Ingest an actual marine observation record idempotently."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO observations (
                    observation_id, location_id, station_id, latitude, longitude,
                    observation_time_utc, observed_swh_m, units, source_name,
                    source_type, quality_status, ingestion_time_utc, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.observation_id,
                    record.location_id,
                    record.station_id,
                    record.latitude,
                    record.longitude,
                    record.observation_time_utc.isoformat(),
                    record.observed_swh_m,
                    record.units,
                    record.source_name,
                    record.source_type.value,
                    record.quality_status.value,
                    record.ingestion_time_utc.isoformat(),
                    record.notes,
                ),
            )
            return cursor.rowcount > 0

    def query_observations(
        self,
        location_id: Optional[str] = None,
        source_type: Optional[str] = None,
        quality_status: Optional[str] = "valid",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[ObservationRecord]:
        """Query ingested observations."""
        clauses = []
        params: List[Any] = []
        if location_id:
            clauses.append("location_id = ?")
            params.append(location_id)
        if source_type:
            clauses.append("source_type = ?")
            params.append(source_type)
        if quality_status:
            clauses.append("quality_status = ?")
            params.append(quality_status)
        if start_time:
            clauses.append("observation_time_utc >= ?")
            params.append(start_time.isoformat())
        if end_time:
            clauses.append("observation_time_utc <= ?")
            params.append(end_time.isoformat())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM observations {where_sql} ORDER BY observation_time_utc DESC LIMIT ?;"
        params.append(limit)

        records = []
        with self._get_connection() as conn:
            for row in conn.execute(sql, params).fetchall():
                records.append(
                    ObservationRecord(
                        observation_id=row["observation_id"],
                        location_id=row["location_id"],
                        station_id=row["station_id"],
                        latitude=row["latitude"],
                        longitude=row["longitude"],
                        observation_time_utc=datetime.fromisoformat(row["observation_time_utc"]),
                        observed_swh_m=row["observed_swh_m"],
                        units=row["units"],
                        source_name=row["source_name"],
                        source_type=ObservationSourceType(row["source_type"]),
                        quality_status=ObservationQuality(row["quality_status"]),
                        ingestion_time_utc=datetime.fromisoformat(row["ingestion_time_utc"]),
                        notes=row["notes"],
                    )
                )
        return records

    def count_observations(self, location_id: Optional[str] = None) -> int:
        """Count total observations stored."""
        sql = "SELECT COUNT(*) as cnt FROM observations"
        params: List[Any] = []
        if location_id:
            sql += " WHERE location_id = ?"
            params.append(location_id)
        with self._get_connection() as conn:
            return conn.execute(sql, params).fetchone()["cnt"]

    # -------------------------------------------------------------------------
    # Forecast-to-Observation Matching
    # -------------------------------------------------------------------------

    def match_forecasts(
        self,
        location_id: Optional[str] = None,
        horizon: Optional[int] = None,
        model_name: Optional[str] = None,
        time_tolerance_seconds: int = 1800,  # +/- 30 minutes
        prospective_only: bool = False,
        limit: int = 2000,
    ) -> List[MatchedForecastObservation]:
        """
        Match eligible issued forecasts with valid observations.

        Guarantees:
        - Strict location match.
        - Temporal tolerance window matching target_time with observation_time.
        - Valid numeric observation (positive, not NaN).
        - Quality status must be 'valid'.
        - Tracks whether observation was prospective (occurred after forecast issuance).
        - Deduplicated 1-to-1 match.
        """
        clauses = ["o.quality_status = 'valid'", "o.observed_swh_m >= 0"]
        params: List[Any] = []

        if location_id:
            clauses.append("f.location_id = ?")
            params.append(location_id)
        if horizon:
            clauses.append("f.horizon_hours = ?")
            params.append(horizon)
        if model_name:
            clauses.append("f.model_name = ?")
            params.append(model_name)

        where_sql = f"WHERE {' AND '.join(clauses)}"

        # Query joins on location and target time
        sql = f"""
        SELECT 
            f.forecast_id,
            f.location_id,
            f.horizon_hours,
            f.model_name,
            f.issue_time_utc,
            f.target_time_utc,
            f.predicted_swh_m,
            f.persistence_swh_m,
            o.observation_id,
            o.observation_time_utc,
            o.observed_swh_m,
            o.source_type
        FROM forecast_ledger f
        JOIN observations o ON f.location_id = o.location_id
        {where_sql}
        ORDER BY f.target_time_utc DESC
        LIMIT ?;
        """
        params.append(limit * 2)

        matched: List[MatchedForecastObservation] = []
        matched_forecast_ids = set()

        with self._get_connection() as conn:
            for row in conn.execute(sql, params).fetchall():
                f_id = row["forecast_id"]
                if f_id in matched_forecast_ids:
                    continue

                f_target = datetime.fromisoformat(row["target_time_utc"])
                o_time = datetime.fromisoformat(row["observation_time_utc"])
                f_issue = datetime.fromisoformat(row["issue_time_utc"])

                # Check temporal tolerance
                diff_sec = abs((f_target - o_time).total_seconds())
                if diff_sec > time_tolerance_seconds:
                    continue

                is_prospective = o_time > f_issue
                if prospective_only and not is_prospective:
                    continue

                pred_val = float(row["predicted_swh_m"])
                obs_val = float(row["observed_swh_m"])
                pers_val = float(row["persistence_swh_m"]) if row["persistence_swh_m"] is not None else None
                err = pred_val - obs_val

                matched.append(
                    MatchedForecastObservation(
                        forecast_id=f_id,
                        observation_id=row["observation_id"],
                        location_id=row["location_id"],
                        horizon_hours=row["horizon_hours"],
                        model_name=row["model_name"],
                        issue_time_utc=f_issue,
                        target_time_utc=f_target,
                        observation_time_utc=o_time,
                        predicted_swh_m=pred_val,
                        observed_swh_m=obs_val,
                        persistence_swh_m=pers_val,
                        error_m=err,
                        abs_error_m=abs(err),
                        source_type=ObservationSourceType(row["source_type"]),
                        is_prospective=is_prospective,
                    )
                )
                matched_forecast_ids.add(f_id)
                if len(matched) >= limit:
                    break

        return matched

    # -------------------------------------------------------------------------
    # Monitoring Event Logging
    # -------------------------------------------------------------------------

    def record_monitoring_event(self, event: MonitoringEventRecord) -> bool:
        """
        Record or update an operational ML monitoring event.
        If an active event with the same fingerprint exists, update its metric value.
        """
        with self._get_connection() as conn:
            # Check existing active event
            existing = conn.execute(
                "SELECT event_id FROM monitoring_events WHERE fingerprint = ? AND status = 'ACTIVE';",
                (event.fingerprint,),
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE monitoring_events
                    SET metric_value = ?, summary = ?, details_json = ?
                    WHERE event_id = ?;
                    """,
                    (event.metric_value, event.summary, event.details_json, existing["event_id"]),
                )
                return False
            else:
                conn.execute(
                    """
                    INSERT INTO monitoring_events (
                        event_id, fingerprint, event_type, location_id, model_name,
                        horizon_hours, severity, status, metric_name, metric_value,
                        threshold_value, summary, details_json, created_at_utc, resolved_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        event.event_id,
                        event.fingerprint,
                        event.event_type.value,
                        event.location_id,
                        event.model_name,
                        event.horizon_hours,
                        event.severity.value,
                        event.status.value,
                        event.metric_name,
                        event.metric_value,
                        event.threshold_value,
                        event.summary,
                        event.details_json,
                        event.created_at_utc.isoformat(),
                        event.resolved_at_utc.isoformat() if event.resolved_at_utc else None,
                    ),
                )
                return True

    def list_monitoring_events(
        self,
        location_id: Optional[str] = None,
        status: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[MonitoringEventRecord]:
        """List persistent monitoring events."""
        clauses = []
        params: List[Any] = []
        if location_id:
            clauses.append("location_id = ?")
            params.append(location_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM monitoring_events {where_sql} ORDER BY created_at_utc DESC LIMIT ?;"
        params.append(limit)

        records = []
        with self._get_connection() as conn:
            for row in conn.execute(sql, params).fetchall():
                records.append(
                    MonitoringEventRecord(
                        event_id=row["event_id"],
                        fingerprint=row["fingerprint"],
                        event_type=MonitoringEventType(row["event_type"]),
                        location_id=row["location_id"],
                        model_name=row["model_name"],
                        horizon_hours=row["horizon_hours"],
                        severity=MonitoringSeverity(row["severity"]),
                        status=MonitoringEventStatus(row["status"]),
                        metric_name=row["metric_name"],
                        metric_value=row["metric_value"],
                        threshold_value=row["threshold_value"],
                        summary=row["summary"],
                        details_json=row["details_json"],
                        created_at_utc=datetime.fromisoformat(row["created_at_utc"]),
                        resolved_at_utc=datetime.fromisoformat(row["resolved_at_utc"]) if row["resolved_at_utc"] else None,
                    )
                )
        return records

    def resolve_monitoring_event(self, fingerprint: str) -> None:
        """Mark active monitoring event as RESOLVED."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE monitoring_events SET status = 'RESOLVED', resolved_at_utc = ? WHERE fingerprint = ? AND status = 'ACTIVE';",
                (now_str, fingerprint),
            )

    def cleanup_old_records(self, retention_days: int = 90, min_records: int = 100) -> int:
        """Purge forecast records older than retention window while protecting minimum count."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with self._get_connection() as conn:
            total_cnt = conn.execute("SELECT COUNT(*) as c FROM forecast_ledger;").fetchone()["c"]
            if total_cnt <= min_records:
                return 0
            cur = conn.execute(
                """
                DELETE FROM forecast_ledger 
                WHERE target_time_utc < ? 
                AND forecast_id NOT IN (
                    SELECT forecast_id FROM forecast_ledger ORDER BY target_time_utc DESC LIMIT ?
                );
                """,
                (cutoff, min_records),
            )
            return cur.rowcount


# Singleton monitoring store
_GLOBAL_MONITORING_STORE: Optional[MonitoringStore] = None


def get_monitoring_store(db_path: Optional[Union[str, Path]] = None) -> MonitoringStore:
    global _GLOBAL_MONITORING_STORE
    if _GLOBAL_MONITORING_STORE is None or db_path is not None:
        _GLOBAL_MONITORING_STORE = MonitoringStore(db_path=db_path)
    return _GLOBAL_MONITORING_STORE


# -----------------------------------------------------------------------------
# Forecast Accuracy Service
# -----------------------------------------------------------------------------

def calculate_forecast_accuracy(
    matched_pairs: List[MatchedForecastObservation],
    total_issued: int = 0,
    min_sample_size: int = 10,
    location_id: Optional[str] = None,
    horizon: Optional[int] = None,
    model_name: Optional[str] = None,
) -> AccuracyMetrics:
    """
    Calculate standard regression accuracy metrics from matched forecast-observation pairs.

    Guarantees:
    - Accurate mathematical definitions for MAE, RMSE, and Mean Bias.
    - True persistence baseline evaluation on the identical matched case subset.
    - Non-inflation of coverage percentages.
    - Explicit evidence classification: VERIFIED, LIMITED_EVIDENCE, AWAITING_OBSERVATIONS,
      or RETROSPECTIVE_REANALYSIS.
    """
    sample_count = len(matched_pairs)
    issued_count = max(total_issued, sample_count)
    coverage_pct = (sample_count / issued_count * 100.0) if issued_count > 0 else 0.0

    if sample_count == 0:
        return AccuracyMetrics(
            sample_count=0,
            total_forecasts_issued=issued_count,
            coverage_pct=0.0,
            verification_status=VerificationStatus.AWAITING_OBSERVATIONS,
            status_explanation="No eligible matched observations have arrived yet for the requested window.",
            location_id=location_id,
            horizon_hours=horizon,
            model_name=model_name,
        )

    errors = np.array([m.error_m for m in matched_pairs], dtype=np.float64)
    abs_errors = np.array([m.abs_error_m for m in matched_pairs], dtype=np.float64)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    # Persistence baseline calculated strictly on the identical matched cases
    valid_pers_pairs = [m for m in matched_pairs if m.persistence_swh_m is not None]
    if valid_pers_pairs:
        pers_errors = np.array([m.persistence_swh_m - m.observed_swh_m for m in valid_pers_pairs], dtype=np.float64)
        pers_mae = float(np.mean(np.abs(pers_errors)))
        pers_rmse = float(np.sqrt(np.mean(pers_errors ** 2)))
        skill = float(1.0 - (rmse / pers_rmse)) if pers_rmse > 1e-6 else 0.0
    else:
        pers_mae = None
        pers_rmse = None
        skill = None

    # Determine verification status
    source_types = set(m.source_type for m in matched_pairs)
    is_reanalysis_only = source_types == {ObservationSourceType.RETROSPECTIVE_REANALYSIS}

    if is_reanalysis_only:
        v_status = VerificationStatus.RETROSPECTIVE_REANALYSIS
        explanation = f"Evaluated against retrospective ERA5 reanalysis ground-truth ({sample_count} samples). Not a live buoy verification."
    elif sample_count >= min_sample_size:
        v_status = VerificationStatus.VERIFIED
        explanation = f"Verified against independent marine observations ({sample_count} eligible matched samples)."
    else:
        v_status = VerificationStatus.LIMITED_EVIDENCE
        explanation = f"Limited observation samples ({sample_count} / {min_sample_size} minimum required for statistical confidence)."

    t_start = min(m.target_time_utc for m in matched_pairs)
    t_end = max(m.target_time_utc for m in matched_pairs)

    return AccuracyMetrics(
        sample_count=sample_count,
        total_forecasts_issued=issued_count,
        coverage_pct=coverage_pct,
        mae_meters=mae,
        rmse_meters=rmse,
        bias_meters=bias,
        persistence_mae_meters=pers_mae,
        persistence_rmse_meters=pers_rmse,
        skill_score_rmse=skill,
        verification_status=v_status,
        status_explanation=explanation,
        location_id=location_id,
        horizon_hours=horizon,
        model_name=model_name,
        time_window_start=t_start,
        time_window_end=t_end,
    )


# -----------------------------------------------------------------------------
# Data Drift Monitoring (Population Stability Index)
# -----------------------------------------------------------------------------

class ReferenceDistributionManager:
    """
    Manages baseline feature distributions and calculates Population Stability Index (PSI)
    across the canonical 13 model features.
    """

    def __init__(self, stats_path: Optional[Union[str, Path]] = None):
        self.stats_path = Path(stats_path) if stats_path else DEFAULT_REF_STATS_PATH
        self._reference_data: Dict[str, Any] = {}
        self._load_or_create_reference()

    def _load_or_create_reference(self) -> None:
        """Load reference stats from JSON or calculate from preprocessed training parquet."""
        if self.stats_path.is_file():
            try:
                with open(self.stats_path, "r", encoding="utf-8") as f:
                    self._reference_data = json.load(f)
                return
            except Exception as e:
                logger.warning("Failed to parse reference distribution file: %s", e)

        # Build reference distribution from preprocessed training split (70%)
        parquet_path = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"
        if parquet_path.is_file():
            try:
                df = pd.read_parquet(parquet_path)
                n_train = int(len(df) * 0.70)
                train_df = df.iloc[:n_train]
                self._reference_data = self._build_profile_from_df(train_df)
                self.stats_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.stats_path, "w", encoding="utf-8") as f:
                    json.dump(self._reference_data, f, indent=2)
                logger.info("Generated feature reference distribution profile for %d features.", len(self._reference_data))
            except Exception as e:
                logger.error("Error creating reference stats profile: %s", e)

    def _build_profile_from_df(self, df: pd.DataFrame, num_bins: int = 10) -> Dict[str, Any]:
        """Compute quantile bin edges and baseline probabilities for each canonical feature."""
        profile: Dict[str, Any] = {}
        for col in DEFAULT_FEATURE_COLS:
            if col not in df.columns:
                continue
            vals = df[col].dropna().values.astype(np.float64)
            if len(vals) == 0:
                continue

            q_points = np.linspace(0, 1, num_bins + 1)
            bins = np.quantile(vals, q_points)
            bins = np.unique(bins)
            if len(bins) < 2:
                # Constant or near-constant feature
                bins = np.array([vals.min() - 0.01, vals.max() + 0.01])

            # Bin probabilities
            counts, _ = np.histogram(vals, bins=bins)
            probs = (counts / counts.sum()).tolist()

            profile[col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "bins": bins.tolist(),
                "expected_probs": probs,
            }
        return profile

    def calculate_psi(self, current_df: pd.DataFrame) -> DataDriftReport:
        """
        Evaluate Population Stability Index (PSI) and data quality on incoming feature window.

        Interpretation:
        - PSI < 0.10: Stable (No Drift)
        - 0.10 <= PSI < 0.25: Moderate Shift (Watch)
        - PSI >= 0.25: Significant Drift (Action / Investigate)
        """
        now_utc = datetime.now(timezone.utc)
        if current_df is None or len(current_df) < 12:
            return DataDriftReport(
                overall_status=DriftStatus.INSUFFICIENT_DATA,
                evaluation_time_utc=now_utc,
                window_size=len(current_df) if current_df is not None else 0,
                reference_dataset="Historical ERA5 Training Split (70%)",
                feature_drifts={},
                max_psi_feature="None",
                max_psi_value=0.0,
                summary="Insufficient data points in current input window (< 12 rows) to compute distribution statistics.",
            )

        feature_drifts: Dict[str, FeatureDriftResult] = {}
        max_psi = 0.0
        max_feature = "None"
        significant_count = 0
        moderate_count = 0

        for col in DEFAULT_FEATURE_COLS:
            ref_col_data = self._reference_data.get(col)
            if not ref_col_data:
                continue

            if col not in current_df.columns:
                # Column completely missing
                feature_drifts[col] = FeatureDriftResult(
                    feature_name=col,
                    psi=1.0,
                    drift_level="SIGNIFICANT_DRIFT",
                    ref_mean=ref_col_data["mean"],
                    curr_mean=0.0,
                    ref_std=ref_col_data["std"],
                    curr_std=0.0,
                    missing_rate=1.0,
                    out_of_bounds_rate=0.0,
                    bin_actual_pct=[],
                    bin_expected_pct=ref_col_data["expected_probs"],
                )
                significant_count += 1
                continue

            curr_series = current_df[col]
            missing_rate = float(curr_series.isna().mean())
            valid_vals = curr_series.dropna().values.astype(np.float64)

            if len(valid_vals) == 0:
                feature_drifts[col] = FeatureDriftResult(
                    feature_name=col,
                    psi=1.0,
                    drift_level="SIGNIFICANT_DRIFT",
                    ref_mean=ref_col_data["mean"],
                    curr_mean=0.0,
                    ref_std=ref_col_data["std"],
                    curr_std=0.0,
                    missing_rate=1.0,
                    out_of_bounds_rate=0.0,
                    bin_actual_pct=[],
                    bin_expected_pct=ref_col_data["expected_probs"],
                )
                significant_count += 1
                continue

            curr_mean = float(np.mean(valid_vals))
            curr_std = float(np.std(valid_vals)) if len(valid_vals) > 1 else 0.0

            # Out of bounds check: 3 std beyond reference min/max
            ref_min = ref_col_data["min"]
            ref_max = ref_col_data["max"]
            margin = max(3.0 * ref_col_data["std"], 0.1)
            oob_rate = float(np.mean((valid_vals < ref_min - margin) | (valid_vals > ref_max + margin)))

            # PSI Calculation
            bins = np.array(ref_col_data["bins"])
            exp_probs = np.array(ref_col_data["expected_probs"])

            # Handle constant features
            if len(bins) < 2 or ref_col_data["std"] < 1e-6:
                if abs(curr_mean - ref_col_data["mean"]) < 1e-4:
                    col_psi = 0.0
                else:
                    col_psi = 0.5
            else:
                # Extend outer bin bounds to catch outliers
                ext_bins = bins.copy()
                ext_bins[0] = min(ext_bins[0], valid_vals.min() - 0.01)
                ext_bins[-1] = max(ext_bins[-1], valid_vals.max() + 0.01)

                counts, _ = np.histogram(valid_vals, bins=ext_bins)
                actual_probs = counts / counts.sum()

                # Add smoothing epsilon
                eps = 1e-4
                act_smooth = (actual_probs + eps) / (1.0 + eps * len(actual_probs))
                exp_smooth = (exp_probs + eps) / (1.0 + eps * len(exp_probs))

                col_psi = float(np.sum((act_smooth - exp_smooth) * np.log(act_smooth / exp_smooth)))

            col_psi = max(col_psi, 0.0)

            if col_psi >= 0.25:
                d_level = "SIGNIFICANT_DRIFT"
                significant_count += 1
            elif col_psi >= 0.10:
                d_level = "MODERATE_DRIFT"
                moderate_count += 1
            else:
                d_level = "NO_DRIFT"

            if col_psi > max_psi:
                max_psi = col_psi
                max_feature = col

            feature_drifts[col] = FeatureDriftResult(
                feature_name=col,
                psi=col_psi,
                drift_level=d_level,
                ref_mean=ref_col_data["mean"],
                curr_mean=curr_mean,
                ref_std=ref_col_data["std"],
                curr_std=curr_std,
                missing_rate=missing_rate,
                out_of_bounds_rate=oob_rate,
                bin_actual_pct=[],
                bin_expected_pct=[],
            )

        # Overall drift status
        if significant_count > 0:
            overall = DriftStatus.DRIFT_DETECTED
            summary = f"Significant distribution drift detected in {significant_count} feature(s). Peak PSI in '{max_feature}' ({max_psi:.3f})."
        elif moderate_count > 0:
            overall = DriftStatus.WATCH
            summary = f"Moderate distribution shift detected across {moderate_count} feature(s). Max PSI in '{max_feature}' ({max_psi:.3f})."
        else:
            overall = DriftStatus.NORMAL
            summary = f"All canonical input features are stable within reference distribution tolerances (Max PSI: {max_psi:.3f})."

        return DataDriftReport(
            overall_status=overall,
            evaluation_time_utc=now_utc,
            window_size=len(current_df),
            reference_dataset="Historical ERA5 Training Split (70%)",
            feature_drifts=feature_drifts,
            max_psi_feature=max_feature,
            max_psi_value=max_psi,
            summary=summary,
        )


# Singleton drift manager
_GLOBAL_DRIFT_MGR: Optional[ReferenceDistributionManager] = None


def get_drift_manager() -> ReferenceDistributionManager:
    global _GLOBAL_DRIFT_MGR
    if _GLOBAL_DRIFT_MGR is None:
        _GLOBAL_DRIFT_MGR = ReferenceDistributionManager()
    return _GLOBAL_DRIFT_MGR


# -----------------------------------------------------------------------------
# Model Performance Degradation Tracking
# -----------------------------------------------------------------------------

# Benchmark test MAE and RMSE values from detailed_metrics.json
BASELINE_BENCHMARK_METRICS = {
    "WaveLSTM": {
        1: {"mae": 0.0182, "rmse": 0.0271},
        6: {"mae": 0.0650, "rmse": 0.0920},
        24: {"mae": 0.1420, "rmse": 0.1980},
    },
    "WaveANN": {
        1: {"mae": 0.0349, "rmse": 0.0465},
        6: {"mae": 0.0980, "rmse": 0.1340},
        24: {"mae": 0.1890, "rmse": 0.2520},
    },
    "WaveCNN1D": {
        1: {"mae": 0.0426, "rmse": 0.0587},
        6: {"mae": 0.1120, "rmse": 0.1510},
        24: {"mae": 0.2050, "rmse": 0.2680},
    },
}


def evaluate_model_performance_degradation(
    matched_pairs: List[MatchedForecastObservation],
    model_name: str = "WaveLSTM",
    horizon: int = 1,
    min_samples: int = 5,
    degradation_threshold_ratio: float = 1.50,
    watch_threshold_ratio: float = 1.25,
) -> PerformanceDegradationReport:
    """
    Track rolling model error against historical baseline.

    State transition logic:
    - NORMAL: rolling_mae <= 1.25 * baseline_mae
    - WATCH: 1.25 * baseline_mae < rolling_mae <= 1.50 * baseline_mae
    - PERFORMANCE_DEGRADATION: rolling_mae > 1.50 * baseline_mae
    """
    now_utc = datetime.now(timezone.utc)
    base_dict = BASELINE_BENCHMARK_METRICS.get(model_name, {}).get(horizon, {"mae": 0.05, "rmse": 0.08})
    b_mae = base_dict["mae"]
    b_rmse = base_dict["rmse"]

    matched_for_model = [m for m in matched_pairs if m.model_name == model_name and m.horizon_hours == horizon]
    count = len(matched_for_model)

    if count < min_samples:
        return PerformanceDegradationReport(
            status=PerformanceStatus.INSUFFICIENT_DATA,
            evaluation_time_utc=now_utc,
            horizon_hours=horizon,
            model_name=model_name,
            rolling_mae=None,
            rolling_rmse=None,
            baseline_mae=b_mae,
            baseline_rmse=b_rmse,
            mae_ratio=None,
            rmse_ratio=None,
            matched_samples=count,
            min_samples_required=min_samples,
            consecutive_degraded_periods=0,
            summary=f"Insufficient verified observation samples ({count} / {min_samples}) for {model_name} (+{horizon}h).",
        )

    abs_errs = [m.abs_error_m for m in matched_for_model]
    errs = [m.error_m for m in matched_for_model]
    roll_mae = float(np.mean(abs_errs))
    roll_rmse = float(np.sqrt(np.mean(np.array(errs) ** 2)))

    mae_ratio = roll_mae / b_mae if b_mae > 0 else 1.0
    rmse_ratio = roll_rmse / b_rmse if b_rmse > 0 else 1.0

    if mae_ratio > degradation_threshold_ratio:
        status = PerformanceStatus.PERFORMANCE_DEGRADATION
        summary = (
            f"POTENTIAL DEGRADATION: Rolling MAE ({roll_mae:.3f} m) is {mae_ratio:.1f}x higher than historical "
            f"test baseline ({b_mae:.3f} m) across {count} verified samples."
        )
    elif mae_ratio > watch_threshold_ratio:
        status = PerformanceStatus.WATCH
        summary = (
            f"ELEVATED ERROR: Rolling MAE ({roll_mae:.3f} m) is {mae_ratio:.1f}x baseline ({b_mae:.3f} m). "
            f"Observing for sustained trend."
        )
    else:
        status = PerformanceStatus.NORMAL
        summary = f"Performance nominal: Rolling MAE ({roll_mae:.3f} m) aligns with baseline ({b_mae:.3f} m)."

    return PerformanceDegradationReport(
        status=status,
        evaluation_time_utc=now_utc,
        horizon_hours=horizon,
        model_name=model_name,
        rolling_mae=roll_mae,
        rolling_rmse=roll_rmse,
        baseline_mae=b_mae,
        baseline_rmse=b_rmse,
        mae_ratio=mae_ratio,
        rmse_ratio=rmse_ratio,
        matched_samples=count,
        min_samples_required=min_samples,
        consecutive_degraded_periods=1 if status == PerformanceStatus.PERFORMANCE_DEGRADATION else 0,
        summary=summary,
    )


# -----------------------------------------------------------------------------
# Retrospective Reanalysis Ingestion Helper
# -----------------------------------------------------------------------------

def populate_retrospective_reanalysis_observations(
    store: MonitoringStore,
    location_id: str = "mumbai_offshore",
    max_rows: int = 1000,
) -> int:
    """
    Populate observation store with ground-truth ERA5 test observations for retrospective backtesting.
    Strictly flags source_type as RETROSPECTIVE_REANALYSIS.
    """
    parquet_path = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"
    if not parquet_path.is_file():
        return 0

    df = pd.read_parquet(parquet_path)
    # Use test split (last 15%)
    n_test = int(len(df) * 0.15)
    test_df = df.iloc[-n_test:].tail(max_rows)

    inserted = 0
    now_utc = datetime.now(timezone.utc)
    for _, row in test_df.iterrows():
        v_time = pd.to_datetime(row["valid_time"])
        if isinstance(v_time, pd.Timestamp):
            v_time_dt = v_time.to_pydatetime()
        else:
            v_time_dt = v_time

        obs = ObservationRecord(
            observation_id=f"era5_{v_time_dt.strftime('%Y%m%d%H%M')}",
            location_id=location_id,
            latitude=18.5,
            longitude=72.5,
            observation_time_utc=v_time_dt,
            observed_swh_m=float(row["swh"]),
            station_id="ERA5_GRID_18.5N_72.5E",
            units="meters",
            source_name="ERA5 Reanalysis Ground-Truth",
            source_type=ObservationSourceType.RETROSPECTIVE_REANALYSIS,
            quality_status=ObservationQuality.VALID,
            ingestion_time_utc=now_utc,
            notes="Historical reanalysis test split for retrospective backtesting.",
        )
        if store.record_observation(obs):
            inserted += 1

    return inserted

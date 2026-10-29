"""
Alert Management and Lifecycle Subsystem for Ocean Intelligence.

Manages persistent alert lifecycle states (ACTIVE, ACKNOWLEDGED, RESOLVED, EXPIRED),
fingerprint-based duplicate suppression, severity escalation, hysteresis resolution,
and SQLite history logging.

IMPORTANT NOTICES:
1. Alerts are generated based on project-level engineering risk thresholds.
2. Official warnings from maritime agencies (INCOIS, IMD) are not integrated in this prototype.
3. Degradation in telemetry (API timeout, stale data) preserves existing alerts rather
   than prematurely or falsely resolving them.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from src.intelligence.nodes import ValidationStatus
from src.intelligence.provider import DataStatus
from src.intelligence.risk import (
    EventType,
    RiskAssessment,
    RiskLevel,
    RiskThresholdSet,
    get_threshold_manager,
)

logger = logging.getLogger("OceanIntelligence.Alerts")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "persistence" / "ocean_intelligence.db"


class AlertStatus(str, Enum):
    """Lifecycle states of an operational wave hazard alert."""
    ACTIVE = "ACTIVE"                  # Newly triggered or continuing unacknowledged alert
    ACKNOWLEDGED = "ACKNOWLEDGED"      # Reviewed and acknowledged by human operator
    RESOLVED = "RESOLVED"              # De-escalated below threshold minus hysteresis margin
    EXPIRED = "EXPIRED"                # Event forecast window has elapsed past grace period

    @property
    def display_badge(self) -> str:
        badges = {
            "ACTIVE": "🔴 ACTIVE",
            "ACKNOWLEDGED": "🟡 ACKNOWLEDGED",
            "RESOLVED": "🟢 RESOLVED",
            "EXPIRED": "⚪ EXPIRED",
        }
        return badges.get(self.value, self.value)


@dataclass
class AlertRecord:
    """Persistent representation of a wave hazard alert."""
    alert_id: str
    fingerprint: str
    location_id: str
    location_name: str
    event_type: str
    severity: str  # 'YELLOW', 'ORANGE', 'RED'
    status: AlertStatus
    trigger_reason: str
    predicted_swh_m: float
    threshold_m: float
    forecast_horizon_hours: int
    expected_event_time_utc: datetime
    anchor_time_utc: datetime
    assessment_time_utc: datetime
    source_name: str
    data_status: str
    validation_status: str
    is_reference: bool
    created_at_utc: datetime
    updated_at_utc: datetime
    forecast_issue_time_utc: Optional[datetime] = None
    model_name: Optional[str] = None
    acknowledged_at_utc: Optional[datetime] = None
    resolved_at_utc: Optional[datetime] = None
    expired_at_utc: Optional[datetime] = None
    operator_notes: Optional[str] = None

    @property
    def severity_level(self) -> RiskLevel:
        return RiskLevel(self.severity)

    @property
    def is_active_or_acknowledged(self) -> bool:
        return self.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "fingerprint": self.fingerprint,
            "location_id": self.location_id,
            "location_name": self.location_name,
            "event_type": self.event_type,
            "severity": self.severity,
            "severity_display": self.severity_level.display_name,
            "status": self.status.value,
            "status_badge": self.status.display_badge,
            "trigger_reason": self.trigger_reason,
            "predicted_swh_m": round(self.predicted_swh_m, 2),
            "threshold_m": round(self.threshold_m, 2),
            "forecast_horizon_hours": self.forecast_horizon_hours,
            "expected_event_time_utc": self.expected_event_time_utc.isoformat(),
            "anchor_time_utc": self.anchor_time_utc.isoformat(),
            "forecast_issue_time_utc": self.forecast_issue_time_utc.isoformat() if self.forecast_issue_time_utc else None,
            "assessment_time_utc": self.assessment_time_utc.isoformat(),
            "source_name": self.source_name,
            "data_status": self.data_status,
            "model_name": self.model_name,
            "validation_status": self.validation_status,
            "is_reference": self.is_reference,
            "created_at_utc": self.created_at_utc.isoformat(),
            "updated_at_utc": self.updated_at_utc.isoformat(),
            "acknowledged_at_utc": self.acknowledged_at_utc.isoformat() if self.acknowledged_at_utc else None,
            "resolved_at_utc": self.resolved_at_utc.isoformat() if self.resolved_at_utc else None,
            "expired_at_utc": self.expired_at_utc.isoformat() if self.expired_at_utc else None,
            "operator_notes": self.operator_notes,
        }


def compute_alert_fingerprint(
    location_id: str,
    event_type: Union[EventType, str],
    forecast_horizon_hours: int,
    expected_event_time_utc: datetime,
) -> str:
    """
    Generate a deterministic, stable fingerprint identifying a specific event.

    Guarantees:
    - Same location, event type, horizon, and target timestamp produce the exact same key.
    - Prevents duplicate alerts upon repeated evaluations of the same forecast window.
    """
    ev_str = event_type.value if isinstance(event_type, EventType) else str(event_type)
    # Normalize timestamp to hourly ISO format (UTC)
    norm_time = expected_event_time_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")
    return f"{location_id}:{ev_str}:{forecast_horizon_hours}h:{norm_time}"


class AlertStore:
    """
    SQLite persistence layer for coastal wave hazard alerts and transition history.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        """Create schema if it does not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    fingerprint TEXT UNIQUE NOT NULL,
                    location_id TEXT NOT NULL,
                    location_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    predicted_swh_m REAL NOT NULL,
                    threshold_m REAL NOT NULL,
                    forecast_horizon_hours INTEGER NOT NULL,
                    expected_event_time_utc TEXT NOT NULL,
                    anchor_time_utc TEXT NOT NULL,
                    forecast_issue_time_utc TEXT,
                    assessment_time_utc TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    data_status TEXT NOT NULL,
                    model_name TEXT,
                    validation_status TEXT NOT NULL,
                    is_reference INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    acknowledged_at_utc TEXT,
                    resolved_at_utc TEXT,
                    expired_at_utc TEXT,
                    operator_notes TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_loc_status ON alerts(location_id, status);
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    changed_at_utc TEXT NOT NULL,
                    details_json TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alert_history_alert ON alert_history(alert_id);
                """
            )

    def _row_to_record(self, row: sqlite3.Row) -> AlertRecord:
        return AlertRecord(
            alert_id=row["alert_id"],
            fingerprint=row["fingerprint"],
            location_id=row["location_id"],
            location_name=row["location_name"],
            event_type=row["event_type"],
            severity=row["severity"],
            status=AlertStatus(row["status"]),
            trigger_reason=row["trigger_reason"],
            predicted_swh_m=float(row["predicted_swh_m"]),
            threshold_m=float(row["threshold_m"]),
            forecast_horizon_hours=int(row["forecast_horizon_hours"]),
            expected_event_time_utc=datetime.fromisoformat(row["expected_event_time_utc"]),
            anchor_time_utc=datetime.fromisoformat(row["anchor_time_utc"]),
            forecast_issue_time_utc=datetime.fromisoformat(row["forecast_issue_time_utc"]) if row["forecast_issue_time_utc"] else None,
            assessment_time_utc=datetime.fromisoformat(row["assessment_time_utc"]),
            source_name=row["source_name"],
            data_status=row["data_status"],
            model_name=row["model_name"],
            validation_status=row["validation_status"],
            is_reference=bool(row["is_reference"]),
            created_at_utc=datetime.fromisoformat(row["created_at_utc"]),
            updated_at_utc=datetime.fromisoformat(row["updated_at_utc"]),
            acknowledged_at_utc=datetime.fromisoformat(row["acknowledged_at_utc"]) if row["acknowledged_at_utc"] else None,
            resolved_at_utc=datetime.fromisoformat(row["resolved_at_utc"]) if row["resolved_at_utc"] else None,
            expired_at_utc=datetime.fromisoformat(row["expired_at_utc"]) if row["expired_at_utc"] else None,
            operator_notes=row["operator_notes"],
        )

    def get_by_fingerprint(self, fingerprint: str) -> Optional[AlertRecord]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM alerts WHERE fingerprint = ?", (fingerprint,))
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None

    def get_by_id(self, alert_id: str) -> Optional[AlertRecord]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None

    def list_alerts(
        self,
        status: Optional[Union[AlertStatus, str]] = None,
        location_id: Optional[str] = None,
        severity: Optional[Union[RiskLevel, str]] = None,
        limit: int = 100,
    ) -> List[AlertRecord]:
        query = "SELECT * FROM alerts WHERE 1=1"
        params: List[Any] = []

        if status:
            s_val = status.value if isinstance(status, AlertStatus) else str(status)
            query += " AND status = ?"
            params.append(s_val)

        if location_id:
            query += " AND location_id = ?"
            params.append(location_id)

        if severity:
            sev_val = severity.value if isinstance(severity, RiskLevel) else str(severity)
            query += " AND severity = ?"
            params.append(sev_val)

        query += " ORDER BY updated_at_utc DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_record(r) for r in cursor.fetchall()]

    def insert_alert(self, alert: AlertRecord, transition_reason: str = "INITIAL_TRIGGER") -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO alerts (
                    alert_id, fingerprint, location_id, location_name, event_type,
                    severity, status, trigger_reason, predicted_swh_m, threshold_m,
                    forecast_horizon_hours, expected_event_time_utc, anchor_time_utc,
                    forecast_issue_time_utc, assessment_time_utc, source_name,
                    data_status, model_name, validation_status, is_reference,
                    created_at_utc, updated_at_utc, acknowledged_at_utc,
                    resolved_at_utc, expired_at_utc, operator_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.alert_id,
                    alert.fingerprint,
                    alert.location_id,
                    alert.location_name,
                    alert.event_type,
                    alert.severity,
                    alert.status.value,
                    alert.trigger_reason,
                    alert.predicted_swh_m,
                    alert.threshold_m,
                    alert.forecast_horizon_hours,
                    alert.expected_event_time_utc.isoformat(),
                    alert.anchor_time_utc.isoformat(),
                    alert.forecast_issue_time_utc.isoformat() if alert.forecast_issue_time_utc else None,
                    alert.assessment_time_utc.isoformat(),
                    alert.source_name,
                    alert.data_status,
                    alert.model_name,
                    alert.validation_status,
                    1 if alert.is_reference else 0,
                    alert.created_at_utc.isoformat(),
                    alert.updated_at_utc.isoformat(),
                    alert.acknowledged_at_utc.isoformat() if alert.acknowledged_at_utc else None,
                    alert.resolved_at_utc.isoformat() if alert.resolved_at_utc else None,
                    alert.expired_at_utc.isoformat() if alert.expired_at_utc else None,
                    alert.operator_notes,
                ),
            )
            # Record initial history
            conn.execute(
                """
                INSERT INTO alert_history (alert_id, from_status, to_status, reason, changed_at_utc, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.alert_id,
                    None,
                    alert.status.value,
                    transition_reason,
                    alert.created_at_utc.isoformat(),
                    json.dumps(
                        {
                            "predicted_swh_m": alert.predicted_swh_m,
                            "threshold_m": alert.threshold_m,
                            "severity": alert.severity,
                        }
                    ),
                ),
            )

    def update_alert(
        self,
        alert: AlertRecord,
        from_status: Optional[AlertStatus] = None,
        reason: str = "UPDATE",
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE alerts SET
                    severity = ?,
                    status = ?,
                    trigger_reason = ?,
                    predicted_swh_m = ?,
                    threshold_m = ?,
                    assessment_time_utc = ?,
                    data_status = ?,
                    updated_at_utc = ?,
                    acknowledged_at_utc = ?,
                    resolved_at_utc = ?,
                    expired_at_utc = ?,
                    operator_notes = ?
                WHERE alert_id = ?
                """,
                (
                    alert.severity,
                    alert.status.value,
                    alert.trigger_reason,
                    alert.predicted_swh_m,
                    alert.threshold_m,
                    alert.assessment_time_utc.isoformat(),
                    alert.data_status,
                    alert.updated_at_utc.isoformat(),
                    alert.acknowledged_at_utc.isoformat() if alert.acknowledged_at_utc else None,
                    alert.resolved_at_utc.isoformat() if alert.resolved_at_utc else None,
                    alert.expired_at_utc.isoformat() if alert.expired_at_utc else None,
                    alert.operator_notes,
                    alert.alert_id,
                ),
            )
            # Record state transition
            conn.execute(
                """
                INSERT INTO alert_history (alert_id, from_status, to_status, reason, changed_at_utc, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.alert_id,
                    from_status.value if from_status else None,
                    alert.status.value,
                    reason,
                    alert.updated_at_utc.isoformat(),
                    json.dumps(
                        {
                            "predicted_swh_m": alert.predicted_swh_m,
                            "severity": alert.severity,
                            "notes": alert.operator_notes,
                        }
                    ),
                ),
            )

    def get_history(self, alert_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM alert_history WHERE alert_id = ? ORDER BY history_id ASC",
                (alert_id,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "history_id": r["history_id"],
                    "from_status": r["from_status"],
                    "to_status": r["to_status"],
                    "reason": r["reason"],
                    "changed_at_utc": r["changed_at_utc"],
                    "details": json.loads(r["details_json"]) if r["details_json"] else {},
                }
                for r in rows
            ]

    def count_alerts_by_status(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM alerts GROUP BY status"
            )
            counts = {s.value: 0 for s in AlertStatus}
            for row in cursor.fetchall():
                counts[row["status"]] = row["cnt"]
            return counts

    def count_active_by_severity(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT severity, COUNT(*) as cnt FROM alerts WHERE status IN ('ACTIVE', 'ACKNOWLEDGED') GROUP BY severity"
            )
            counts = {"RED": 0, "ORANGE": 0, "YELLOW": 0}
            for row in cursor.fetchall():
                counts[row["severity"]] = row["cnt"]
            return counts


# Singleton store instance
ALERT_STORE = AlertStore()


def get_alert_store() -> AlertStore:
    return ALERT_STORE


class AlertEngine:
    """
    Evaluates risk assessments, orchestrates alert lifecycle, and persists state.
    """

    def __init__(
        self,
        store: Optional[AlertStore] = None,
        expiration_grace_hours: float = 6.0,
    ):
        self.store = store or get_alert_store()
        self.expiration_grace_hours = expiration_grace_hours

    def process_assessment(
        self,
        assessment: RiskAssessment,
        thresholds: Optional[RiskThresholdSet] = None,
        reference_now_utc: Optional[datetime] = None,
    ) -> Optional[AlertRecord]:
        """
        Process a single risk assessment through the alert lifecycle.

        Guarantees:
        - Deduplication: uses deterministic fingerprint to find existing alert.
        - Escalation: upgrades severity if risk level increased.
        - Hysteresis: requires wave height to drop below threshold - hysteresis_buffer before resolving.
        - Expiration: auto-expires past events.
        - Resilience: if data is degraded/unavailable, existing alerts are preserved and NOT resolved.
        """
        now_utc = datetime.now(timezone.utc)
        active_thresh = thresholds or get_threshold_manager().get_thresholds(assessment.location_id)

        # 1. Do not resolve or dismiss alerts on degraded or unavailable data
        if assessment.is_degraded_data or assessment.data_status == DataStatus.UNAVAILABLE:
            return None

        # 2. Check if event is past its expiration window
        if reference_now_utc is not None:
            ref_now = reference_now_utc
        elif assessment.data_status == DataStatus.OFFLINE_DEMO:
            ref_now = assessment.forecast_valid_time_utc
        else:
            ref_now = now_utc

        is_past_event = ref_now > (assessment.forecast_valid_time_utc + timedelta(hours=self.expiration_grace_hours))

        # 3. Compute deterministic fingerprint
        fp = compute_alert_fingerprint(
            location_id=assessment.location_id,
            event_type=assessment.event_type,
            forecast_horizon_hours=assessment.forecast_horizon_hours,
            expected_event_time_utc=assessment.forecast_valid_time_utc,
        )

        existing = self.store.get_by_fingerprint(fp)

        # 4. Handle event expiration
        if existing and existing.is_active_or_acknowledged and is_past_event:
            old_status = existing.status
            existing.status = AlertStatus.EXPIRED
            existing.expired_at_utc = now_utc
            existing.updated_at_utc = now_utc
            self.store.update_alert(existing, from_status=old_status, reason="EVENT_WINDOW_ELAPSED")
            return existing

        # 5. Case A: Threshold is exceeded -> Create or update alert
        if assessment.is_warning_triggered:
            if is_past_event:
                # Do not trigger brand new alerts for events already elapsed
                return None

            if existing is None:
                # Create NEW alert
                alert_id = str(uuid.uuid4())
                new_alert = AlertRecord(
                    alert_id=alert_id,
                    fingerprint=fp,
                    location_id=assessment.location_id,
                    location_name=assessment.location_name,
                    event_type=assessment.event_type.value,
                    severity=assessment.risk_level.value,
                    status=AlertStatus.ACTIVE,
                    trigger_reason=assessment.trigger_explanation,
                    predicted_swh_m=assessment.predicted_swh_m,
                    threshold_m=assessment.threshold_crossed_m or active_thresh.caution_m,
                    forecast_horizon_hours=assessment.forecast_horizon_hours,
                    expected_event_time_utc=assessment.forecast_valid_time_utc,
                    anchor_time_utc=assessment.data_retrieval_time_utc,
                    forecast_issue_time_utc=assessment.forecast_issue_time_utc,
                    assessment_time_utc=assessment.assessment_time_utc,
                    source_name=assessment.source_name,
                    data_status=assessment.data_status.value,
                    model_name=assessment.model_name,
                    validation_status=assessment.validation_status.value,
                    is_reference=assessment.is_reference,
                    created_at_utc=now_utc,
                    updated_at_utc=now_utc,
                )
                self.store.insert_alert(new_alert, transition_reason="THRESHOLD_EXCEEDED")
                return new_alert

            elif existing.is_active_or_acknowledged:
                # Check for severity escalation or forecast update
                old_sev_rank = existing.severity_level.severity_rank
                new_sev_rank = assessment.risk_level.severity_rank

                old_status = existing.status
                existing.predicted_swh_m = assessment.predicted_swh_m
                existing.threshold_m = assessment.threshold_crossed_m or existing.threshold_m
                existing.assessment_time_utc = assessment.assessment_time_utc
                existing.trigger_reason = assessment.trigger_explanation
                existing.data_status = assessment.data_status.value
                existing.updated_at_utc = now_utc

                if new_sev_rank > old_sev_rank:
                    # Severity escalation!
                    old_sev = existing.severity
                    existing.severity = assessment.risk_level.value
                    self.store.update_alert(
                        existing,
                        from_status=old_status,
                        reason=f"SEVERITY_ESCALATION: {old_sev} -> {assessment.risk_level.value}",
                    )
                else:
                    # General update (same or lower severity without crossing hysteresis)
                    self.store.update_alert(
                        existing,
                        from_status=old_status,
                        reason="FORECAST_UPDATE",
                    )
                return existing

        # 6. Case B: Risk assessment is GREEN (Threshold not exceeded) -> Check for hysteresis resolution
        else:
            if existing and existing.is_active_or_acknowledged:
                # Hysteresis check: wave height must drop below threshold minus buffer
                res_threshold = active_thresh.caution_m - active_thresh.hysteresis_buffer_m
                if assessment.predicted_swh_m < res_threshold:
                    old_status = existing.status
                    existing.status = AlertStatus.RESOLVED
                    existing.resolved_at_utc = now_utc
                    existing.updated_at_utc = now_utc
                    existing.predicted_swh_m = assessment.predicted_swh_m
                    existing.trigger_reason = (
                        f"Resolved: Wave height dropped to {assessment.predicted_swh_m:.2f} m, "
                        f"below de-escalation threshold ({res_threshold:.2f} m with hysteresis)."
                    )
                    self.store.update_alert(
                        existing,
                        from_status=old_status,
                        reason=f"DE_ESCALATION_RESOLVED: Dropped below {res_threshold:.2f}m",
                    )
                    return existing

        return None

    def process_all_assessments(
        self,
        assessments: List[RiskAssessment],
        thresholds: Optional[RiskThresholdSet] = None,
        reference_now_utc: Optional[datetime] = None,
    ) -> List[AlertRecord]:
        """Process a batch of assessments and return updated/created alerts."""
        processed: List[AlertRecord] = []
        for a in assessments:
            rec = self.process_assessment(a, thresholds=thresholds, reference_now_utc=reference_now_utc)
            if rec:
                processed.append(rec)
        return processed

    def acknowledge_alert(self, alert_id: str, operator_notes: Optional[str] = None) -> bool:
        """
        Transition an ACTIVE alert to ACKNOWLEDGED state.
        """
        alert = self.store.get_by_id(alert_id)
        if not alert or alert.status != AlertStatus.ACTIVE:
            return False

        now_utc = datetime.now(timezone.utc)
        old_status = alert.status
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at_utc = now_utc
        alert.updated_at_utc = now_utc
        if operator_notes:
            alert.operator_notes = operator_notes

        self.store.update_alert(
            alert,
            from_status=old_status,
            reason="OPERATOR_ACKNOWLEDGED",
        )
        return True


# Global alert engine singleton
ALERT_ENGINE = AlertEngine()


def get_alert_engine() -> AlertEngine:
    return ALERT_ENGINE

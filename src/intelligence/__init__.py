"""
Ocean Intelligence & Coastal Early-Warning Subsystem.
Provides coastal node registries, data provider abstractions, multi-location ingestion,
risk assessment, and persistent alert lifecycle management.
"""

from src.intelligence.nodes import (
    CoastalNode,
    ValidationStatus,
    CoastalNodeRegistry,
    get_coastal_registry,
    COASTAL_NODE_REGISTRY,
)
from src.intelligence.provider import (
    DataStatus,
    ProviderFetchResult,
    BaseMarineDataProvider,
    OpenMeteoProvider,
    OfflineDemoProvider,
)
from src.intelligence.ingestion import (
    MultiLocationIngestionService,
    get_ingestion_service,
)
from src.intelligence.risk import (
    RiskLevel,
    EventType,
    RiskThresholdSet,
    ThresholdConfigManager,
    RiskAssessment,
    classify_risk,
    assess_node_risk,
    get_threshold_manager,
)
from src.intelligence.alerts import (
    AlertStatus,
    AlertRecord,
    AlertStore,
    AlertEngine,
    compute_alert_fingerprint,
    get_alert_store,
    get_alert_engine,
)
from src.intelligence.mapping import (
    NodeMapSummary,
    prepare_coastal_node_map_data,
    compute_map_dashboard_metrics,
    build_coastal_folium_map,
)
from src.intelligence.monitoring import (
    ForecastSourceCategory,
    ObservationSourceType,
    ObservationQuality,
    VerificationStatus,
    DriftStatus,
    PerformanceStatus,
    MonitoringEventType,
    MonitoringSeverity,
    MonitoringEventStatus,
    ForecastLedgerRecord,
    ObservationRecord,
    MatchedForecastObservation,
    AccuracyMetrics,
    FeatureDriftResult,
    DataDriftReport,
    PerformanceDegradationReport,
    MonitoringEventRecord,
    compute_ledger_fingerprint,
    compute_monitoring_event_fingerprint,
    MonitoringStore,
    get_monitoring_store,
    calculate_forecast_accuracy,
    ReferenceDistributionManager,
    get_drift_manager,
    evaluate_model_performance_degradation,
    populate_retrospective_reanalysis_observations,
)

__all__ = [
    # Nodes
    "CoastalNode",
    "ValidationStatus",
    "CoastalNodeRegistry",
    "get_coastal_registry",
    "COASTAL_NODE_REGISTRY",
    # Providers
    "DataStatus",
    "ProviderFetchResult",
    "BaseMarineDataProvider",
    "OpenMeteoProvider",
    "OfflineDemoProvider",
    # Ingestion
    "MultiLocationIngestionService",
    "get_ingestion_service",
    # Risk
    "RiskLevel",
    "EventType",
    "RiskThresholdSet",
    "ThresholdConfigManager",
    "RiskAssessment",
    "classify_risk",
    "assess_node_risk",
    "get_threshold_manager",
    # Alerts
    "AlertStatus",
    "AlertRecord",
    "AlertStore",
    "AlertEngine",
    "compute_alert_fingerprint",
    "get_alert_store",
    "get_alert_engine",
    # Mapping
    "NodeMapSummary",
    "prepare_coastal_node_map_data",
    "compute_map_dashboard_metrics",
    "build_coastal_folium_map",
    # Monitoring
    "ForecastSourceCategory",
    "ObservationSourceType",
    "ObservationQuality",
    "VerificationStatus",
    "DriftStatus",
    "PerformanceStatus",
    "MonitoringEventType",
    "MonitoringSeverity",
    "MonitoringEventStatus",
    "ForecastLedgerRecord",
    "ObservationRecord",
    "MatchedForecastObservation",
    "AccuracyMetrics",
    "FeatureDriftResult",
    "DataDriftReport",
    "PerformanceDegradationReport",
    "MonitoringEventRecord",
    "compute_ledger_fingerprint",
    "compute_monitoring_event_fingerprint",
    "MonitoringStore",
    "get_monitoring_store",
    "calculate_forecast_accuracy",
    "ReferenceDistributionManager",
    "get_drift_manager",
    "evaluate_model_performance_degradation",
    "populate_retrospective_reanalysis_observations",
]


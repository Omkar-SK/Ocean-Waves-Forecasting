export type Horizon = 1 | 6 | 24
export type ModelName = 'WaveLSTM' | 'WaveANN' | 'WaveCNN1D' | 'persistence' | 'all' | 'lstm' | 'ann' | 'cnn1d'

export type RiskLevel = 'GREEN' | 'YELLOW' | 'ORANGE' | 'RED'
export type AlertLifecycleStatus = 'ACTIVE' | 'ACKNOWLEDGED' | 'RESOLVED' | 'EXPIRED'
export type VerificationStatusType =
  | 'verified'
  | 'limited_evidence'
  | 'awaiting_observations'
  | 'verification_unavailable'
  | 'retrospective_reanalysis'

export interface ComponentHealth {
  name: string
  status: 'healthy' | 'degraded' | 'unhealthy'
  message: string
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy'
  backend_online: boolean
  checkpoints_available: number
  total_checkpoints: number
  scaler_available: boolean
  dataset_available: boolean
  live_api_available: boolean
  timestamp: string
  components: ComponentHealth[]
}

export interface MetadataResponse {
  project_name: string
  academic_title: string
  version: string
  selected_coordinates: {
    latitude: number
    longitude: number
  }
  location_name: string
  supported_horizons: number[]
  supported_models: string[]
  input_window_hours: number
  features: string[]
  target: string
  units: Record<string, string>
  dataset_temporal_span: {
    start: string
    end: string
    total_hours: string
    test_start: string
    test_hours: string
  }
  best_overall_model: string
}

export interface ModelMetric {
  horizon: number
  model: string
  rmse: number
  mae: number
  r2: number
  rmse_skill_pct?: number | null
  mae_skill_pct?: number | null
  is_best?: boolean
}

export interface MultiHorizonResponse {
  horizons: number[]
  metrics: ModelMetric[]
  summary?: {
    best_overall_model: string
    best_horizon_models: Record<string, { model: string; rmse: number; mae: number; r2: number }>
  }
  items?: Array<{
    horizon_hours: number
    model_name: string
    mae_meters: number
    rmse_meters: number
    r2_score: number
    rmse_skill_vs_persistence: number
    is_best_in_horizon?: boolean
  }>
  best_by_horizon?: Record<string, any>
}

export interface CoastalNode {
  location_id: string
  name: string
  latitude: number
  longitude: number
  region: string
  timezone_str: string
  coordinate_label: string
  is_reference: boolean
  validation_status: string
  description: string
  notes: string[]
  enabled: boolean
}

export interface SingleModelForecast {
  model_name: string
  model_type: string
  checkpoint_file: string
  forecast_swh_m: number
  delta_vs_persistence_m: number
  delta_vs_nwp_m?: number | null
  sea_state_desc: string
  sea_state_color: string
  // Residual learning integration fields
  current_swh_m?: number | null
  predicted_residual_m?: number | null
  forecast_mode?: 'residual' | 'absolute'
  model_version?: string
  validation_status?: string
  rmse_skill_score?: number | null
  is_candidate_model?: boolean
  reconstruction_formula?: string
  predicted_swh?: number | null
  current_swh?: number | null
  predicted_residual?: number | null
  selected_model_architecture?: string
  model_validation_status?: string
}

export interface ObservationWindowStep {
  valid_time: string
  swh: number
  mwp: number
  wind_speed: number
  mwd?: number | null
  wind_dir?: number | null
  u10?: number | null
  v10?: number | null
}

export interface ForecastResponse {
  anchor_time_utc: string
  target_time_utc: string
  forecast_horizon_hours: number
  persistence_swh_m: number
  external_nwp_swh_m?: number | null
  predictions: Record<string, SingleModelForecast>
  observation_window: ObservationWindowStep[]
  source_name: string
  attribution: string
  is_demo: boolean
  notes: string[]
  generated_at_utc: string
  location_id?: string
  location_name?: string
  is_reference?: boolean
  validation_status?: string
  // Enhanced residual reporting attributes
  forecast_mode?: 'residual' | 'absolute'
  current_swh?: number | null
  current_swh_m?: number | null
  predicted_swh?: number | null
  predicted_residual?: number | null
  forecast_horizon?: number
  selected_model_architecture?: string
  model_version?: string
  input_timestamp?: string
  data_source?: string
  data_freshness?: string
  data_freshness_minutes?: number | null
  forecast_generation_timestamp?: string
  model_validation_status?: string
}

export interface ForecastStatusResponse {
  live_api_online: boolean
  demo_mode_available: boolean
  latest_observation_time_utc?: string | null
  data_freshness_minutes?: number | null
  checkpoints_ready: Record<string, boolean>
  message: string
}

// ----------------------------------------------------------------------------
// Coastal Risk Heatmap & Nodes Telemetry
// ----------------------------------------------------------------------------
export interface NodeMapSummaryItem {
  location_id: string
  name: string
  latitude: number
  longitude: number
  coordinate_label: string
  region: string
  is_reference: boolean
  validation_status: string
  data_status: string
  data_source: string
  retrieval_time_utc?: string | null
  forecast_issue_time_utc?: string | null
  latest_obs_swh_m?: number | null
  latest_obs_time_utc?: string | null
  forecast_swh_m?: number | null
  forecast_horizon_hours: number
  forecast_valid_time_utc?: string | null
  max_window_swh_m?: number | null
  current_risk_level: RiskLevel
  highest_risk_level: RiskLevel
  marker_hex_color: string
  threshold_crossed_m?: number | null
  active_alert_count: number
  active_alerts: AlertItem[]
  is_degraded: boolean
  trigger_explanation: string
  limitations: string[]
  multi_horizon_predictions: Record<number, number>
  external_nwp_forecasts: Record<number, number>
}

export interface CoastalMetricsResponse {
  total_nodes: number
  reference_nodes: number
  active_alerts: number
  high_risk_nodes: number
  degraded_feeds: number
  timestamp_utc: string
}

// ----------------------------------------------------------------------------
// Coastal Early Warning Alerts
// ----------------------------------------------------------------------------
export interface AlertItem {
  alert_id: string
  fingerprint: string
  location_id: string
  location_name: string
  event_type: string
  severity: 'YELLOW' | 'ORANGE' | 'RED'
  status: AlertLifecycleStatus
  trigger_reason: string
  predicted_swh_m: number
  threshold_m: number
  forecast_horizon_hours: number
  expected_event_time_utc: string
  anchor_time_utc: string
  forecast_issue_time_utc?: string | null
  assessment_time_utc: string
  source_name: string
  data_status: string
  model_name?: string | null
  validation_status: string
  is_reference: boolean
  created_at_utc: string
  updated_at_utc: string
  acknowledged_at_utc?: string | null
  resolved_at_utc?: string | null
  expired_at_utc?: string | null
  operator_notes?: string | null
}

export interface AlertSummaryResponse {
  total_active: number
  critical_red: number
  high_orange: number
  caution_yellow: number
  resolved_count: number
  status_counts: Record<string, number>
  last_assessment_utc?: string | null
}

export interface AlertHistoryItem {
  history_id: number
  from_status?: string | null
  to_status: string
  reason: string
  changed_at_utc: string
  details: Record<string, any>
}

export interface ThresholdConfigResponse {
  caution_m: number
  high_risk_m: number
  critical_m: number
  hysteresis_buffer_m: number
  units: string
  version: string
  is_demonstration: boolean
  guidance_notice: string
  location_id?: string | null
}

// ----------------------------------------------------------------------------
// Forecast Accuracy, Drift & Monitoring
// ----------------------------------------------------------------------------
export interface MonitoringSummaryResponse {
  total_forecasts_issued: number
  total_observations_recorded: number
  active_monitoring_events_count: number
  independent_buoy_available: boolean
  live_verification_status: string
  feature_drift_status: string
  evaluation_time_utc: string
  notices: string[]
}

export interface AccuracyMetricsResponse {
  sample_count: number
  total_forecasts_issued: number
  coverage_pct: number
  mae_meters?: number | null
  rmse_meters?: number | null
  bias_meters?: number | null
  persistence_mae_meters?: number | null
  persistence_rmse_meters?: number | null
  skill_score_rmse?: number | null
  verification_status: VerificationStatusType
  verification_badge: string
  status_explanation: string
  location_id?: string | null
  horizon_hours?: number | null
  model_name?: string | null
  time_window_start?: string | null
  time_window_end?: string | null
}

export interface FeatureDriftItem {
  feature_name: string
  psi: number
  drift_level: 'NO_DRIFT' | 'MODERATE_DRIFT' | 'SIGNIFICANT_DRIFT'
  ref_mean: number
  curr_mean: number
  ref_std: number
  curr_std: number
  missing_rate: number
  out_of_bounds_rate: number
  bin_actual_pct?: number[]
  bin_expected_pct?: number[]
}

export interface DataDriftResponse {
  status: 'NORMAL' | 'WATCH' | 'DRIFT_DETECTED' | 'INSUFFICIENT_DATA'
  evaluation_time_utc: string
  window_rows: number
  reference_dataset: string
  feature_drifts: Record<string, FeatureDriftItem>
  max_psi_feature: string
  max_psi_value: number
  summary: string
}

export interface MonitoringEventItem {
  event_id: string
  fingerprint: string
  event_type: string
  location_id: string
  model_name?: string | null
  horizon_hours?: number | null
  severity: 'INFO' | 'WARNING' | 'CRITICAL'
  status: 'ACTIVE' | 'ACKNOWLEDGED' | 'RESOLVED'
  metric_name: string
  metric_value: number
  threshold_value: number
  summary: string
  details_json?: string | null
  created_at_utc: string
  resolved_at_utc?: string | null
}

export interface ForecastLedgerItem {
  forecast_id: string
  fingerprint: string
  location_id: string
  latitude: number
  longitude: number
  issue_time_utc: string
  anchor_time_utc: string
  target_time_utc: string
  horizon_hours: number
  predicted_swh_m: number
  units: string
  model_name: string
  checkpoint_name?: string | null
  source_category: string
  data_source: string
  data_status: string
  validation_status: string
  is_reference: boolean
  persistence_swh_m?: number | null
  external_nwp_swh_m?: number | null
  created_at_utc: string
}

// ----------------------------------------------------------------------------
// Historical Analysis & Evaluation
// ----------------------------------------------------------------------------
export interface HistoricalSummaryResponse {
  total_test_samples?: number
  total_samples?: number
  start_date?: string
  end_date?: string
  available_models?: string[]
  available_horizons?: number[]
  overall_min_swh?: number
  overall_max_swh?: number
  overall_mean_swh?: number
  overall_std_swh?: number
  metrics_summary?: Record<string, { rmse: number; mae: number; r2: number }>
}

export interface TimeSeriesPoint {
  timestamp: string
  actual: number
  predicted?: number
  residual?: number
  error_abs?: number
  ann?: number
  lstm?: number
  cnn1d?: number
  persistence?: number
}

export interface HistoricalPredictionsResponse {
  horizon: number
  model?: string
  start_date?: string
  end_date?: string
  total_points?: number
  total_samples?: number
  points?: TimeSeriesPoint[]
  data?: TimeSeriesPoint[]
}

export interface ScatterPoint {
  actual: number
  predicted: number
}

export interface ScatterDataResponse {
  horizon?: number
  model?: string
  total_points?: number
  min_val?: number
  max_val?: number
  points?: ScatterPoint[]
}

export interface ResidualBin {
  bin_center?: number
  lower?: number
  upper?: number
  count: number
  frequency?: number
}

export interface ResidualDataResponse {
  horizon?: number
  model?: string
  mean_error?: number
  std_error?: number
  bins?: ResidualBin[]
}

// ----------------------------------------------------------------------------
// Hazard Simulation Mode
// ----------------------------------------------------------------------------
export interface ScenarioMetadataItem {
  scenario_id: string
  name: string
  category: string
  headline: string
  description: string
  severity_category: string
  coastal_impact_level: string
  recommended_action: string
  base_swh_m: number
  peak_swh_m: number
  peak_wind_kmh: number
  peak_mwp_s: number
}

export interface SimulationStatusResponse {
  is_active: boolean
  state: 'IDLE' | 'RUNNING' | 'PAUSED'
  scenario_id?: string | null
  scenario_name?: string | null
  current_step: number
  total_steps: number
  elapsed_hours: number
  started_at_utc?: string | null
  updated_at_utc?: string | null
  disclaimer: string
}

export interface HeatmapPoint {
  latitude: number
  longitude: number
  label: string
  region: string
  swh_m: number
  max_wave_m: number
  mwp_s: number
  wind_kmh: number
  wind_dir_deg: number
  intensity: number
  risk_level: string
  color: string
  radius_km: number
}

export interface SimulationStats {
  active_alerts: number
  critical_risk_zones: number
  high_risk_zones: number
  max_swh_m: number
  max_wind_kmh: number
  coastal_risk_index: number
  affected_regions: string[]
  scenario_name: string
  severity_category: string
  headline: string
  coastal_impact_level: string
  recommended_action: string
}

export interface SimulatedTimeSeriesPoint {
  step: number
  time: string
  timestamp: string
  actual?: number | null
  lstm?: number | null
  ann?: number | null
  cnn1d?: number | null
  persistence?: number | null
  wind_kmh: number
  mwp_s: number
  risk_level: string
  isForecast: boolean
}

export interface SimulationResultsResponse {
  status: SimulationStatusResponse
  scenario?: ScenarioMetadataItem | null
  stats?: SimulationStats | null
  nodes: NodeMapSummaryItem[]
  heatmap_points: HeatmapPoint[]
  alerts: AlertItem[]
  time_series: SimulatedTimeSeriesPoint[]
}

// ----------------------------------------------------------------------------
// Continuous Forecast Verification & Model Performance Monitoring
// ----------------------------------------------------------------------------
export type VerificationRecordStatus =
  | 'PENDING_VERIFICATION'
  | 'VERIFIED'
  | 'OBSERVATION_UNAVAILABLE'
  | 'INVALID_OBSERVATION'

export type VerificationWindow = 'last_24h' | 'last_7d' | 'last_30d' | 'all_time'

export interface VerificationRecord {
  forecast_id: string
  location_id: string
  forecast_origin_time: string
  target_timestamp: string
  forecast_horizon: number
  model_architecture: string
  model_version: string
  current_swh: number
  predicted_swh: number
  predicted_residual?: number | null
  observed_swh?: number | null
  observation_time?: string | null
  forecast_error?: number | null
  absolute_error?: number | null
  persistence_error?: number | null
  status: VerificationRecordStatus
  status_reason?: string | null
  is_prospective: boolean
  is_severe_actual: boolean
  is_severe_forecast: boolean
  data_source: string
  validation_status: string
  created_at_utc: string
  verified_at_utc?: string | null
}

export interface AbsoluteErrorDistribution {
  p25?: number | null
  median?: number | null
  p75?: number | null
  p90?: number | null
  p95?: number | null
  max_error?: number | null
}

export interface ContinuousAccuracyMetrics {
  horizon_hours: number
  model_architecture: string
  window: string
  sample_count: number
  total_issued_count: number
  coverage_pct: number
  mae?: number | null
  mse?: number | null
  rmse?: number | null
  r2?: number | null
  bias?: number | null
  persistence_mae?: number | null
  persistence_rmse?: number | null
  rmse_skill_pct?: number | null
  mae_skill_pct?: number | null
  error_distribution: AbsoluteErrorDistribution
  is_statistically_reliable: boolean
  min_samples_required: number
  reliability_notice: string
}

export interface SevereEventTrackingMetrics {
  actual_severe_events_count: number
  forecasted_severe_events_count: number
  underprediction_frequency_pct?: number | null
  missed_severe_events_count: number
  false_alarms_count: number
  average_detection_lead_time_hours?: number | null
  historical_validation_sample_size: number
  historical_severe_rmse_skill: Record<string, number>
  operational_severe_sample_size: number
  operational_severe_status: string
  scientific_notice: string
}

export interface ModelHealthMonitoringSummary {
  prediction_drift_status: string
  input_feature_drift_status: string
  missing_observations_count: number
  missing_observations_rate_pct: number
  stale_data_alerts_count: number
  increasing_forecast_errors_warning: boolean
  error_degradation_ratio?: number | null
  extreme_underprediction_warning: boolean
  model_inference_failures_count: number
  active_warnings: string[]
}

export interface VerificationDashboardResponse {
  evaluation_time_utc: string
  location_id?: string | null
  total_forecasts_issued: number
  total_verified_observations: number
  total_pending_verifications: number
  total_unavailable_observations: number
  total_invalid_observations: number
  horizon_metrics: Record<string, Record<string, ContinuousAccuracyMetrics>>
  severe_events: SevereEventTrackingMetrics
  monitoring_health: ModelHealthMonitoringSummary
  recent_verifications: VerificationRecord[]
  historical_vs_operational_summary: Record<string, any>
}

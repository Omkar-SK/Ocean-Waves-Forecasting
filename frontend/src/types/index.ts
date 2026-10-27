export type Horizon = 1 | 6 | 24
export type ModelName = 'WaveLSTM' | 'WaveANN' | 'WaveCNN1D' | 'persistence' | 'all' | 'lstm' | 'ann' | 'cnn1d'

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'error'
  timestamp: string
  models_loaded: number
  checkpoints_available: Record<string, boolean>
  data_status: {
    era5_available: boolean
    test_predictions_available: boolean
    multi_horizon_metrics_available: boolean
    scaler_available: boolean
  }
  location: {
    lat: number
    lon: number
    description: string
  }
  active_horizons: number[]
}

export interface MetadataResponse {
  project_name: string
  academic_title: string
  location: {
    latitude: number
    longitude: number
    description: string
  }
  sequence_length: number
  features: string[]
  horizons: number[]
  models: string[]
  test_period: {
    start: string
    end: string
    samples: number
  }
  target_variable: string
  units: Record<string, string>
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
  summary: {
    best_overall_model: string
    best_horizon_models: Record<string, { model: string; rmse: number; mae: number; r2: number }>
  }
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
}

export interface ForecastStatusResponse {
  live_api_online: boolean
  demo_mode_available: boolean
  latest_observation_time_utc?: string | null
  data_freshness_minutes?: number | null
  checkpoints_ready: Record<string, boolean>
  message: string
}

export interface HistoricalSummaryResponse {
  total_samples: number
  start_date: string
  end_date: string
  available_models: string[]
  available_horizons: number[]
  metrics_summary: Record<string, { rmse: number; mae: number; r2: number }>
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
  model: string
  start_date: string
  end_date: string
  total_points: number
  data: TimeSeriesPoint[]
}

export interface ScatterPoint {
  actual: number
  predicted: number
}

export interface ScatterDataResponse {
  horizon: number
  model: string
  total_points: number
  min_val: number
  max_val: number
  points: ScatterPoint[]
}

export interface ResidualBin {
  bin_center: number
  count: number
  frequency: number
}

export interface ResidualDataResponse {
  horizon: number
  model: string
  mean_error: number
  std_error: number
  bins: ResidualBin[]
}

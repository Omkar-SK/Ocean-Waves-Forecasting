import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import App from '../App'
import { api } from '../services/api'

// Mock api methods to return deterministic test data
vi.mock('../services/api', () => ({
  api: {
    getHealth: vi.fn(),
    getMetadata: vi.fn(),
    getMetricsByHorizon: vi.fn(),
    getMultiHorizonMetrics: vi.fn(),
    getLiveForecast: vi.fn(),
    getLiveStatus: vi.fn(),
    getAlertSummary: vi.fn(),
    getCoastalMetrics: vi.fn(),
    getCoastalLocations: vi.fn(),
    getCoastalMapData: vi.fn(),
    getCoastalThresholds: vi.fn(),
    listAlerts: vi.fn(),
    getAccuracyMetrics: vi.fn(),
    getDataDrift: vi.fn(),
    getMonitoringSummary: vi.fn(),
    getForecastLedger: vi.fn(),
    getMonitoringEvents: vi.fn(),
    getHistoricalSummary: vi.fn(),
    getHistoricalPredictions: vi.fn(),
    getHistoricalScatter: vi.fn(),
    getHistoricalResiduals: vi.fn(),
  },
}))

describe('Application Shell & Navigation Routing', () => {
  beforeEach(() => {
    vi.clearAllMocks()

    // Default mock implementations
    vi.mocked(api.getHealth).mockResolvedValue({
      status: 'healthy',
      backend_online: true,
      checkpoints_available: 9,
      total_checkpoints: 9,
      scaler_available: true,
      dataset_available: true,
      live_api_available: true,
      timestamp: '2026-10-27T12:00:00Z',
      components: [{ name: 'PyTorch Inference Engine', status: 'healthy', message: '9/9 weights ready' }],
    })

    vi.mocked(api.getMetricsByHorizon).mockResolvedValue([
      { horizon: 6, model: 'WaveLSTM', rmse: 0.0769, mae: 0.052, r2: 0.9886, is_best: true },
    ])

    vi.mocked(api.getMultiHorizonMetrics).mockResolvedValue({
      horizons: [1, 6, 24],
      metrics: [],
      items: [],
    })

    vi.mocked(api.getLiveForecast).mockResolvedValue({
      anchor_time_utc: '2026-10-27T12:00:00Z',
      target_time_utc: '2026-10-27T18:00:00Z',
      forecast_horizon_hours: 6,
      persistence_swh_m: 0.68,
      external_nwp_swh_m: 0.72,
      predictions: {
        lstm: {
          model_name: 'WaveLSTM',
          model_type: 'LSTM',
          checkpoint_file: 'best_lstm_h6.pt',
          forecast_swh_m: 0.74,
          delta_vs_persistence_m: 0.06,
          sea_state_desc: 'Smooth (Wavelets)',
          sea_state_color: 'text-cyan-400',
        },
      },
      observation_window: [
        { valid_time: '2026-10-27T11:00:00Z', swh: 0.65, mwp: 6.8, wind_speed: 4.2 },
        { valid_time: '2026-10-27T12:00:00Z', swh: 0.68, mwp: 6.9, wind_speed: 4.5 },
      ],
      source_name: 'Open-Meteo ECMWF',
      attribution: 'ECMWF WAM',
      is_demo: false,
      notes: [],
      generated_at_utc: '2026-10-27T12:00:05Z',
      location_id: 'mumbai',
      location_name: 'Mumbai Reference',
      is_reference: true,
      validation_status: 'reference',
    })

    vi.mocked(api.getAlertSummary).mockResolvedValue({
      total_active: 0,
      critical_red: 0,
      high_orange: 0,
      caution_yellow: 0,
      resolved_count: 0,
      status_counts: { ACTIVE: 0, ACKNOWLEDGED: 0, RESOLVED: 0 },
      last_assessment_utc: '2026-10-27T12:00:00Z',
    })

    vi.mocked(api.getCoastalMetrics).mockResolvedValue({
      total_nodes: 8,
      reference_nodes: 1,
      active_alerts: 0,
      high_risk_nodes: 0,
      degraded_feeds: 0,
      timestamp_utc: '2026-10-27T12:00:00Z',
    })

    vi.mocked(api.getCoastalLocations).mockResolvedValue([
      {
        location_id: 'mumbai',
        name: 'Mumbai Offshore Grid',
        latitude: 18.5,
        longitude: 72.5,
        region: 'Arabian Sea (Central West)',
        timezone_str: 'Asia/Kolkata',
        coordinate_label: '18.50° N, 72.50° E',
        is_reference: true,
        validation_status: 'reference',
        description: 'Reference training point',
        notes: [],
        enabled: true,
      },
      {
        location_id: 'gujarat_khambhat',
        name: 'Gujarat Gulf of Khambhat',
        latitude: 20.8,
        longitude: 71.8,
        region: 'Arabian Sea (North West)',
        timezone_str: 'Asia/Kolkata',
        coordinate_label: '20.80° N, 71.80° E',
        is_reference: false,
        validation_status: 'experimental',
        description: 'Experimental coastal node',
        notes: [],
        enabled: true,
      },
    ])
  })

  it('renders application brand title and sidebar navigation links', async () => {
    render(<App />)

    // Check header / brand
    expect(screen.getByText('OceanCast')).toBeInTheDocument()

    // Check all primary navigation items are present
    expect(screen.getByText('Overview')).toBeInTheDocument()
    expect(screen.getByText('Live Forecast')).toBeInTheDocument()
    expect(screen.getByText('Coastal Risk Map')).toBeInTheDocument()
    expect(screen.getByText('Coastal Alerts')).toBeInTheDocument()
    expect(screen.getByText('Coastal Locations')).toBeInTheDocument()
    expect(screen.getByText('Accuracy & Drift')).toBeInTheDocument()
    expect(screen.getByText('Historical Analysis')).toBeInTheDocument()
    expect(screen.getByText('Model Comparison')).toBeInTheDocument()
    expect(screen.getByText('System & Docs')).toBeInTheDocument()
  })

  it('renders overview dashboard cards with live telemetry data', async () => {
    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('Mumbai Offshore Grid (18.50° N, 72.50° E)')).toBeInTheDocument()
      expect(screen.getByText('Validated Reference')).toBeInTheDocument()
    })
  })
})

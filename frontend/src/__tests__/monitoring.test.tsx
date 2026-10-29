import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ForecastAccuracyDriftPage } from '../pages/ForecastAccuracyDriftPage'
import { api } from '../services/api'
import type {
  AccuracyMetricsResponse,
  CoastalNode,
  DataDriftResponse,
  MonitoringSummaryResponse,
} from '../types'

vi.mock('../services/api', () => ({
  api: {
    getMonitoringSummary: vi.fn(),
    getForecastAccuracy: vi.fn(),
    getDataDriftReport: vi.fn(),
    listMonitoringEvents: vi.fn(),
    getForecastLedger: vi.fn(),
    getCoastalLocations: vi.fn(),
    triggerBacktest: vi.fn(),
  },
}))

describe('Forecast Accuracy & Model Drift Monitoring Page', () => {
  const mockSummary: MonitoringSummaryResponse = {
    total_forecasts_issued: 1420,
    total_observations_recorded: 1205,
    active_monitoring_events_count: 0,
    independent_buoy_available: false,
    live_verification_status: 'AWAITING_OBSERVATIONS',
    feature_drift_status: 'STABLE',
    evaluation_time_utc: '2026-10-27T12:00:00Z',
    notices: ['Independent waverider buoy telemetry is unconfigured.'],
  }

  const mockAccuracy: AccuracyMetricsResponse = {
    sample_count: 1205,
    total_forecasts_issued: 1420,
    coverage_pct: 85.2,
    mae_meters: 0.052,
    rmse_meters: 0.0769,
    bias_meters: -0.011,
    persistence_mae_meters: 0.061,
    persistence_rmse_meters: 0.089,
    skill_score_rmse: 13.88,
    verification_status: 'verified',
    verification_badge: 'VERIFIED',
    status_explanation: 'Matched with prospectively verified observations',
  }

  const mockDrift: DataDriftResponse = {
    status: 'NORMAL',
    evaluation_time_utc: '2026-10-27T12:00:00Z',
    window_rows: 24,
    reference_dataset: 'ERA5 Arabian Sea Baseline',
    max_psi_feature: 'swh',
    max_psi_value: 0.042,
    summary: 'Distribution shift within acceptable operational bounds',
    feature_drifts: {
      swh: {
        feature_name: 'swh',
        psi: 0.021,
        drift_level: 'NO_DRIFT',
        ref_mean: 1.25,
        curr_mean: 1.28,
        ref_std: 0.45,
        curr_std: 0.46,
        missing_rate: 0,
        out_of_bounds_rate: 0,
      },
      wind_speed: {
        feature_name: 'wind_speed',
        psi: 0.035,
        drift_level: 'NO_DRIFT',
        ref_mean: 5.4,
        curr_mean: 5.6,
        ref_std: 1.2,
        curr_std: 1.25,
        missing_rate: 0,
        out_of_bounds_rate: 0,
      },
    },
  }

  const mockLocations: CoastalNode[] = [
    {
      location_id: 'mumbai',
      name: 'Mumbai Offshore Grid',
      latitude: 18.5,
      longitude: 72.5,
      region: 'Arabian Sea',
      timezone_str: 'Asia/Kolkata',
      coordinate_label: '18.50° N, 72.50° E',
      is_reference: true,
      validation_status: 'reference',
      description: 'Reference',
      notes: [],
      enabled: true,
    },
  ]

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getMonitoringSummary).mockResolvedValue(mockSummary)
    vi.mocked(api.getForecastAccuracy).mockResolvedValue(mockAccuracy)
    vi.mocked(api.getDataDriftReport).mockResolvedValue(mockDrift)
    vi.mocked(api.listMonitoringEvents).mockResolvedValue([])
    vi.mocked(api.getForecastLedger).mockResolvedValue([])
    vi.mocked(api.getCoastalLocations).mockResolvedValue(mockLocations)
  })

  it('renders forecast accuracy metrics, verified evidence badge, and verification coverage', async () => {
    render(<ForecastAccuracyDriftPage />)

    await waitFor(() => {
      expect(screen.getByText('Forecast Accuracy & Model Drift Monitoring')).toBeInTheDocument()
      expect(screen.getAllByText('VERIFIED').length).toBeGreaterThan(0)
      expect(screen.getByText('85.2%')).toBeInTheDocument()
      expect(screen.getByText('0.052 m')).toBeInTheDocument()
    })
  })

  it('renders features drift table when switching to drift tab', async () => {
    render(<ForecastAccuracyDriftPage />)

    await waitFor(() => {
      expect(screen.getByText(/Canonical Feature Drift/)).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText(/Canonical Feature Drift/))

    await waitFor(() => {
      expect(screen.getAllByText('swh').length).toBeGreaterThan(0)
      expect(screen.getAllByText('wind_speed').length).toBeGreaterThan(0)
    })
  })
})

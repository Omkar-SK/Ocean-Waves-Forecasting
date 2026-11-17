import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ForecastVerificationPage } from '../pages/ForecastVerificationPage'
import { api } from '../services/api'
import type { VerificationDashboardResponse } from '../types'

vi.mock('../services/api', () => ({
  api: {
    getVerificationDashboard: vi.fn(),
    triggerVerificationMatching: vi.fn(),
    seedDemoObservations: vi.fn(),
  },
}))

describe('Continuous Real-Time Forecast Verification Page', () => {
  const mockDashboard: VerificationDashboardResponse = {
    evaluation_time_utc: '2026-11-16T12:00:00Z',
    location_id: 'mumbai_offshore',
    total_forecasts_issued: 25,
    total_verified_observations: 18,
    total_pending_verifications: 5,
    total_unavailable_observations: 1,
    total_invalid_observations: 1,
    horizon_metrics: {
      '6h': {
        all_time: {
          horizon_hours: 6,
          model_architecture: 'Residual WaveANN',
          window: 'all_time',
          sample_count: 18,
          total_issued_count: 25,
          coverage_pct: 72.0,
          mae: 0.045,
          mse: 0.0035,
          rmse: 0.059,
          r2: 0.985,
          bias: -0.008,
          persistence_mae: 0.062,
          persistence_rmse: 0.088,
          rmse_skill_pct: 32.95,
          mae_skill_pct: 27.42,
          error_distribution: {
            p25: 0.02,
            median: 0.04,
            p75: 0.06,
            p90: 0.08,
            p95: 0.09,
            max_error: 0.11,
          },
          is_statistically_reliable: true,
          min_samples_required: 5,
          reliability_notice: 'Empirically validated across N=18 verified prospective observations.',
        },
      },
    },
    severe_events: {
      actual_severe_events_count: 0,
      forecasted_severe_events_count: 0,
      underprediction_frequency_pct: null,
      missed_severe_events_count: 0,
      false_alarms_count: 0,
      average_detection_lead_time_hours: null,
      historical_validation_sample_size: 22,
      historical_severe_rmse_skill: { '1h': 11.91, '6h': -5.58, '24h': -35.61 },
      operational_severe_sample_size: 0,
      operational_severe_status: 'INSUFFICIENT_OPERATIONAL_SAMPLE',
      scientific_notice:
        'HISTORICAL BENCHMARK NOTICE: Independent ERA5 validation contains exactly N=22 severe events (>=4.0m). Currently, 0 live operational severe events have been recorded.',
    },
    monitoring_health: {
      prediction_drift_status: 'NORMAL',
      input_feature_drift_status: 'NORMAL',
      missing_observations_count: 1,
      missing_observations_rate_pct: 4.0,
      stale_data_alerts_count: 0,
      increasing_forecast_errors_warning: false,
      error_degradation_ratio: 0.82,
      extreme_underprediction_warning: false,
      model_inference_failures_count: 0,
      active_warnings: [],
    },
    recent_verifications: [
      {
        forecast_id: 'fc_demo_101',
        location_id: 'mumbai_offshore',
        forecast_origin_time: '2026-11-16T06:00:00Z',
        target_timestamp: '2026-11-16T12:00:00Z',
        forecast_horizon: 6,
        model_architecture: 'Residual WaveANN',
        model_version: '2.0.0-residual',
        current_swh: 1.25,
        predicted_swh: 1.35,
        predicted_residual: 0.1,
        observed_swh: 1.32,
        observation_time: '2026-11-16T12:05:00Z',
        forecast_error: 0.03,
        absolute_error: 0.03,
        persistence_error: -0.07,
        status: 'VERIFIED',
        status_reason: 'Verified against independent buoy measurement.',
        is_prospective: true,
        is_severe_actual: false,
        is_severe_forecast: false,
        data_source: 'OpenMeteoProvider',
        validation_status: 'reference_validated',
        created_at_utc: '2026-11-16T06:00:00Z',
        verified_at_utc: '2026-11-16T12:10:00Z',
      },
    ],
    historical_vs_operational_summary: {
      operational_status: 'MEASURING_LIVE_PERFORMANCE',
    },
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getVerificationDashboard).mockResolvedValue(mockDashboard)
  })

  it('renders page header, summary KPI cards, and scientific notices', async () => {
    render(<ForecastVerificationPage />)

    await waitFor(() => {
      expect(
        screen.getByText('Continuous Real-Time Forecast Verification')
      ).toBeInTheDocument()
    })

    expect(screen.getByText('Total Issued')).toBeInTheDocument()
    expect(screen.getByText('Verified Prospective')).toBeInTheDocument()
    expect(screen.getByText('Pending Target Time')).toBeInTheDocument()
    expect(screen.getByText('Telemetry Missing')).toBeInTheDocument()

    // Counts from mock
    expect(screen.getByText('25')).toBeInTheDocument()
    expect(screen.getByText('18')).toBeInTheDocument()
  })

  it('displays severe sea state monitoring panel and historical N=22 caveat', async () => {
    render(<ForecastVerificationPage />)

    await waitFor(() => {
      expect(
        screen.getByText(/Severe Sea-State Verification & Peak Underprediction Monitoring/i)
      ).toBeInTheDocument()
    })

    expect(screen.getByText(/Historical N=22/i)).toBeInTheDocument()
  })

  it('renders recent verification history table with verification badges', async () => {
    render(<ForecastVerificationPage />)

    await waitFor(() => {
      expect(screen.getByText('fc_demo_101')).toBeInTheDocument()
    })

    expect(screen.getAllByText('Verified').length).toBeGreaterThan(0)
    expect(screen.getByText('✓ Prospective')).toBeInTheDocument()
  })

  it('allows switching forecast horizons', async () => {
    render(<ForecastVerificationPage />)

    await waitFor(() => {
      expect(screen.getByText('+1h (Residual WaveLSTM)')).toBeInTheDocument()
    })

    const h1Btn = screen.getByText('+1h (Residual WaveLSTM)')
    fireEvent.click(h1Btn)
    expect(h1Btn).toHaveClass('bg-[#102A43]')
  })
})

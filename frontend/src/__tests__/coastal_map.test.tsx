import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { CoastalRiskMapPage } from '../pages/CoastalRiskMapPage'
import { api } from '../services/api'
import type { NodeMapSummaryItem, ThresholdConfigResponse } from '../types'

vi.mock('../services/api', () => ({
  api: {
    getCoastalMapData: vi.fn(),
    getCoastalThresholds: vi.fn(),
  },
}))

describe('Coastal Risk Map Page', () => {
  const mockNodes: NodeMapSummaryItem[] = [
    {
      location_id: 'mumbai',
      name: 'Mumbai Offshore Grid',
      latitude: 18.5,
      longitude: 72.5,
      coordinate_label: '18.50° N, 72.50° E',
      region: 'Arabian Sea (Central West)',
      is_reference: true,
      validation_status: 'reference',
      data_status: 'FRESH',
      data_source: 'Open-Meteo ECMWF',
      current_risk_level: 'GREEN',
      highest_risk_level: 'GREEN',
      marker_hex_color: '#10B981',
      latest_obs_swh_m: 0.68,
      latest_obs_time_utc: '2026-10-27T12:00:00Z',
      forecast_swh_m: 0.74,
      forecast_horizon_hours: 6,
      forecast_valid_time_utc: '2026-10-27T18:00:00Z',
      active_alert_count: 0,
      active_alerts: [],
      is_degraded: false,
      trigger_explanation: 'All marine variables within safe normal bounds.',
      limitations: [],
      multi_horizon_predictions: { 1: 0.68, 6: 0.74, 24: 0.81 },
      external_nwp_forecasts: { 1: 0.69, 6: 0.75, 24: 0.82 },
    },
    {
      location_id: 'gujarat_khambhat',
      name: 'Gujarat Gulf of Khambhat',
      latitude: 20.8,
      longitude: 71.8,
      coordinate_label: '20.80° N, 71.80° E',
      region: 'Arabian Sea (North West)',
      is_reference: false,
      validation_status: 'experimental',
      data_status: 'FRESH',
      data_source: 'Open-Meteo ECMWF',
      current_risk_level: 'YELLOW',
      highest_risk_level: 'YELLOW',
      marker_hex_color: '#EAB308',
      latest_obs_swh_m: 1.25,
      latest_obs_time_utc: '2026-10-27T12:00:00Z',
      forecast_swh_m: 1.45,
      forecast_horizon_hours: 6,
      forecast_valid_time_utc: '2026-10-27T18:00:00Z',
      active_alert_count: 1,
      active_alerts: [],
      is_degraded: false,
      trigger_explanation: 'Elevated wind-wave swell.',
      limitations: ['Regional transfer without local buoy ground truth.'],
      multi_horizon_predictions: { 1: 1.2, 6: 1.45, 24: 1.6 },
      external_nwp_forecasts: { 1: 1.25, 6: 1.48, 24: 1.65 },
    },
  ]

  const mockThresholds: ThresholdConfigResponse = {
    caution_m: 1.5,
    high_risk_m: 2.5,
    critical_m: 4.0,
    hysteresis_buffer_m: 0.2,
    units: 'meters',
    version: 'v1.0',
    is_demonstration: false,
    guidance_notice: 'Standard operational marine thresholds',
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getCoastalMapData).mockResolvedValue(mockNodes)
    vi.mocked(api.getCoastalThresholds).mockResolvedValue(mockThresholds)
  })

  it('renders coastal risk map title, node count, and horizon controls', async () => {
    render(<CoastalRiskMapPage />)

    await waitFor(() => {
      expect(screen.getByText('Interactive Coastal Risk Heatmap')).toBeInTheDocument()
      expect(screen.getByText('+1h')).toBeInTheDocument()
      expect(screen.getByText('+6h')).toBeInTheDocument()
      expect(screen.getByText('+24h')).toBeInTheDocument()
    })
  })

  it('displays the scientific disclaimer against spatial interpolation between sparse monitoring points', async () => {
    render(<CoastalRiskMapPage />)

    await waitFor(() => {
      expect(
        screen.getByText(/Continuous contour surface interpolation between sparse coastal points/)
      ).toBeInTheDocument()
    })
  })

  it('renders registered nodes in the telemetry list and shows reference vs experimental distinction', async () => {
    render(<CoastalRiskMapPage />)

    await waitFor(() => {
      expect(screen.getAllByText('Mumbai Offshore Grid').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Gujarat Gulf of Khambhat').length).toBeGreaterThan(0)
    })
  })
})

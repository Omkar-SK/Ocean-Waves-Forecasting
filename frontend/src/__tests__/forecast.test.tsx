import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { LiveForecastPage } from '../pages/LiveForecastPage'
import { api } from '../services/api'
import type { CoastalNode, ForecastResponse, ForecastStatusResponse } from '../types'

vi.mock('../services/api', () => ({
  api: {
    getLiveForecast: vi.fn(),
    getLiveStatus: vi.fn(),
    getCoastalLocations: vi.fn(),
  },
}))

describe('Live Forecast Page', () => {
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
      description: 'Experimental',
      notes: [],
      enabled: true,
    },
  ]

  const mockLiveStatus: ForecastStatusResponse = {
    live_api_online: true,
    demo_mode_available: true,
    latest_observation_time_utc: '2026-10-27T12:00:00Z',
    data_freshness_minutes: 8,
    checkpoints_ready: {
      ann_h1: true,
      ann_h6: true,
      ann_h24: true,
      lstm_h1: true,
      lstm_h6: true,
      lstm_h24: true,
      cnn1d_h1: true,
      cnn1d_h6: true,
      cnn1d_h24: true,
    },
    message: '9 PyTorch checkpoints ready',
  }

  const mockForecast: ForecastResponse = {
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
      ann: {
        model_name: 'WaveANN',
        model_type: 'ANN',
        checkpoint_file: 'best_ann_h6.pt',
        forecast_swh_m: 0.71,
        delta_vs_persistence_m: 0.03,
        sea_state_desc: 'Smooth (Wavelets)',
        sea_state_color: 'text-teal-400',
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
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getCoastalLocations).mockResolvedValue(mockLocations)
    vi.mocked(api.getLiveStatus).mockResolvedValue(mockLiveStatus)
    vi.mocked(api.getLiveForecast).mockResolvedValue(mockForecast)
  })

  it('renders live wave forecasting page with parameters and predicted wave height', async () => {
    render(<LiveForecastPage />)

    await waitFor(() => {
      expect(screen.getByText('Forecasting Parameters & Operational Controls')).toBeInTheDocument()
      expect(screen.getByText('0.740 m')).toBeInTheDocument()
      expect(screen.getByText('Live Operational Telemetry Active')).toBeInTheDocument()
    })
  })

  it('displays experimental warning when an unvalidated location is chosen', async () => {
    render(<LiveForecastPage />)

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })

    const select = screen.getByRole('combobox')
    fireEvent.change(select, { target: { value: 'gujarat_khambhat' } })

    await waitFor(() => {
      expect(screen.getByText(/Geographic Validation Notice: Experimental Node/)).toBeInTheDocument()
    })
  })
})

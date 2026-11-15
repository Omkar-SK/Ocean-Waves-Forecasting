import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { CinematicHomePage } from '../pages/CinematicHomePage'
import { api } from '../services/api'

vi.mock('../services/api', () => ({
  api: {
    getHealth: vi.fn(),
    getLiveForecast: vi.fn(),
    getAlertSummary: vi.fn(),
    getCoastalLocations: vi.fn(),
    getMultiHorizonMetrics: vi.fn(),
  },
}))

// Mock 3D Canvas scenes in JSDOM environment
vi.mock('../components/home/OceanScene3D', () => ({
  OceanScene3D: () => <div data-testid="mock-ocean-scene-3d" />,
}))

vi.mock('../components/home/InteractiveOceanGlobe', () => ({
  InteractiveOceanGlobe: () => <div data-testid="mock-ocean-globe-3d" />,
}))

describe('Cinematic Ocean Intelligence Homepage (/home)', () => {
  beforeEach(() => {
    vi.clearAllMocks()

    vi.mocked(api.getHealth).mockResolvedValue({
      status: 'healthy',
      backend_online: true,
      checkpoints_available: 9,
      total_checkpoints: 9,
      scaler_available: true,
      dataset_available: true,
      live_api_available: true,
      timestamp: '2026-11-15T18:00:00Z',
      components: [],
    })

    vi.mocked(api.getLiveForecast).mockResolvedValue({
      status: 'operational',
      location: {
        id: 'mumbai_offshore',
        name: 'Mumbai Offshore',
        latitude: 18.5,
        longitude: 72.5,
        region: 'Maharashtra Coast',
        is_reference: true,
        validation_status: 'reference_validated',
      },
      predictions: {
        lstm: { forecast_swh_m: 0.68, horizon_hours: 6, model_name: 'WaveLSTM' },
      },
      observation_window: [
        {
          timestamp: '2026-11-15T18:00:00Z',
          valid_time: '2026-11-15T18:00:00Z',
          swh: 0.52,
          mwp: 8.5,
          wind_speed: 4.6,
        },
      ],
      is_demo: false,
    } as any)

    vi.mocked(api.getAlertSummary).mockResolvedValue({
      total_active: 0,
      critical_red: 0,
      high_orange: 0,
      caution_yellow: 0,
      resolved_count: 5,
      status_counts: { ACTIVE: 0 },
    })

    vi.mocked(api.getCoastalLocations).mockResolvedValue([
      {
        location_id: 'mumbai_offshore',
        name: 'Mumbai Offshore (Reference Node)',
        latitude: 18.5,
        longitude: 72.5,
        region: 'Maharashtra Coast',
        coordinate_label: '18.50°N, 72.50°E',
        is_reference: true,
        validation_status: 'reference_validated',
        description: 'Verified deep Arabian Sea offshore node.',
        notes: ['ERA5 reference calibration station.'],
        enabled: true,
        timezone_str: 'Asia/Kolkata',
      },
    ])

    vi.mocked(api.getMultiHorizonMetrics).mockResolvedValue({
      horizons: [1, 6, 24],
      metrics: [],
      items: [
        {
          horizon_hours: 6,
          model: 'WaveLSTM',
          model_type: 'LSTM',
          rmse_meters: 0.0769,
          mae_meters: 0.0533,
          r2_score: 0.9886,
          rmse_skill_pct: 13.88,
          is_best: true,
        },
      ],
    } as any)
  })

  it('renders the cinematic hero section with required typography and CTAs', async () => {
    render(
      <BrowserRouter>
        <CinematicHomePage />
      </BrowserRouter>
    )

    // Hero title & labels
    expect(screen.getByText(/OCEAN INTELLIGENCE PLATFORM/i)).toBeInTheDocument()
    expect(screen.getByText(/Understand the Ocean\./i)).toBeInTheDocument()
    expect(screen.getByText(/Anticipate What's Next\./i)).toBeInTheDocument()

    // Supporting description
    expect(
      screen.getByText(/Advanced deep learning and live marine intelligence for wave forecasting/i)
    ).toBeInTheDocument()

    // Action buttons
    expect(screen.getByText(/Explore Ocean Intelligence/i)).toBeInTheDocument()
    const dashboardButtons = screen.getAllByText(/Enter Dashboard/i)
    expect(dashboardButtons.length).toBeGreaterThanOrEqual(1)
  })

  it('renders floating navbar with links and logo', () => {
    render(
      <BrowserRouter>
        <CinematicHomePage />
      </BrowserRouter>
    )

    expect(screen.getAllByText(/OceanCast/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Intelligence/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Technology/i).length).toBeGreaterThan(0)
  })

  it('renders live ocean intelligence section with live data metrics', async () => {
    render(
      <BrowserRouter>
        <CinematicHomePage />
      </BrowserRouter>
    )

    await waitFor(() => {
      expect(screen.getByText(/Observed Wave Height/i)).toBeInTheDocument()
      expect(screen.getByText(/\+6h Forecast Horizon/i)).toBeInTheDocument()
      expect(screen.getAllByText(/Coastal Grid Stations/i).length).toBeGreaterThan(0)
      expect(screen.getByText(/Active Hazard Alerts/i)).toBeInTheDocument()
    })
  })

  it('renders all four core capability cards', () => {
    render(
      <BrowserRouter>
        <CinematicHomePage />
      </BrowserRouter>
    )

    expect(screen.getAllByText(/Wave Forecasting/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Coastal Risk Intelligence/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Early Warning System/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Ocean Hazard Simulation/i).length).toBeGreaterThan(0)
  })

  it('renders deep learning technology section explaining ANN, LSTM, 1D-CNN', async () => {
    render(
      <BrowserRouter>
        <CinematicHomePage />
      </BrowserRouter>
    )

    expect(screen.getAllByText(/WaveLSTM/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/WaveANN/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/WaveCNN1D/i).length).toBeGreaterThan(0)
    expect(screen.getByText(/Empirical Test Set Performance/i)).toBeInTheDocument()
  })

  it('renders Call to Action and Footer with scientific notices', () => {
    render(
      <BrowserRouter>
        <CinematicHomePage />
      </BrowserRouter>
    )

    expect(screen.getByText(/The Ocean Is Constantly Changing\./i)).toBeInTheDocument()
    expect(screen.getByText(/Stay One Step Ahead\./i)).toBeInTheDocument()
    expect(screen.getByText(/Enter OceanCast Dashboard/i)).toBeInTheDocument()
    expect(screen.getByText(/Scientific Notices/i)).toBeInTheDocument()
  })
})

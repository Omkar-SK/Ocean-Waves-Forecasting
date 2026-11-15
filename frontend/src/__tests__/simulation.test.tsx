import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { SimulationProvider, useSimulation } from '../context/SimulationContext'
import { SimulationControlBar } from '../components/simulation/SimulationControlBar'
import { SimulationModal } from '../components/simulation/SimulationModal'
import { HazardHeatmapLegend } from '../components/simulation/HazardHeatmapLegend'
import { SimulationLabPage } from '../pages/SimulationLabPage'
import { api } from '../services/api'
import type {
  ScenarioMetadataItem,
  SimulationResultsResponse,
  SimulationStatusResponse,
} from '../types'

// Mock Three.js Canvas and Drei Html in JSDOM test environment
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="mock-simulation-r3f-canvas">{children}</div>
  ),
  useFrame: vi.fn(),
  useThree: () => ({
    camera: {
      position: { x: 0, y: 7.5, z: 12, lerp: vi.fn(), copy: vi.fn(), distanceTo: vi.fn(() => 0) },
      lookAt: vi.fn(),
    },
  }),
}))

vi.mock('@react-three/drei', () => ({
  Html: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="mock-simulation-drei-html">{children}</div>
  ),
}))

vi.mock('../services/api', () => ({
  api: {
    getSimulationScenarios: vi.fn(),
    getSimulationStatus: vi.fn(),
    getSimulationResults: vi.fn(),
    startSimulation: vi.fn(),
    pauseSimulation: vi.fn(),
    resumeSimulation: vi.fn(),
    stepSimulation: vi.fn(),
    resetSimulation: vi.fn(),
  },
}))

const mockScenarios: ScenarioMetadataItem[] = [
  {
    scenario_id: 'severe_wave',
    name: 'Severe Wave Conditions',
    category: 'High-Energy Swell Event',
    headline: 'Extreme Deep-Sea Swell & Destructive Wave Action (5.5–7.5m)',
    description: 'Simulates intense oceanic storm generating long-period swell.',
    severity_category: 'CRITICAL',
    coastal_impact_level: 'Severe Offshore Wave Action',
    recommended_action: 'Complete prohibition of small craft operations.',
    base_swh_m: 5.5,
    peak_swh_m: 7.4,
    peak_wind_kmh: 85.0,
    peak_mwp_s: 15.8,
  },
  {
    scenario_id: 'tropical_cyclone',
    name: 'Tropical Cyclone Impact',
    category: 'Tropical Cyclonic Storm',
    headline: 'Approaching Category 3/4 Tropical Cyclone with Catastrophic Waves',
    description: 'Simulates tropical cyclone landfall.',
    severity_category: 'CRITICAL',
    coastal_impact_level: 'Catastrophic Storm Surge',
    recommended_action: 'Immediate coastal evacuation.',
    base_swh_m: 7.0,
    peak_swh_m: 9.85,
    peak_wind_kmh: 135.0,
    peak_mwp_s: 17.5,
  },
]

const mockActiveResults: SimulationResultsResponse = {
  status: {
    is_active: true,
    state: 'RUNNING',
    scenario_id: 'severe_wave',
    scenario_name: 'Severe Wave Conditions',
    current_step: 6,
    total_steps: 24,
    elapsed_hours: 0,
    started_at_utc: '2026-10-30T01:00:00Z',
    updated_at_utc: '2026-10-30T01:00:00Z',
    disclaimer: 'Simulation results are hypothetical.',
  },
  scenario: mockScenarios[0],
  stats: {
    active_alerts: 8,
    critical_risk_zones: 3,
    high_risk_zones: 4,
    max_swh_m: 6.74,
    max_wind_kmh: 78.5,
    coastal_risk_index: 82,
    affected_regions: ['Maharashtra Coast', 'Goa Coast'],
    scenario_name: 'Severe Wave Conditions',
    severity_category: 'CRITICAL',
    headline: 'Extreme Deep-Sea Swell & Destructive Wave Action (5.5–7.5m)',
    coastal_impact_level: 'Severe Offshore Wave Action',
    recommended_action: 'Complete prohibition of small craft operations.',
  },
  nodes: [
    {
      location_id: 'mumbai_offshore',
      name: 'Mumbai Offshore Grid',
      latitude: 18.5,
      longitude: 72.5,
      coordinate_label: '18.50° N, 72.50° E',
      region: 'Arabian Sea (Central West)',
      is_reference: true,
      validation_status: 'reference',
      data_status: 'FRESH',
      data_source: 'Hypothetical Simulation',
      current_risk_level: 'CRITICAL',
      highest_risk_level: 'CRITICAL',
      marker_hex_color: '#EF4444',
      latest_obs_swh_m: 6.74,
      latest_obs_time_utc: '2026-10-30T01:00:00Z',
      forecast_swh_m: 6.85,
      forecast_horizon_hours: 6,
      forecast_valid_time_utc: '2026-10-30T07:00:00Z',
      active_alert_count: 1,
      active_alerts: [],
      is_degraded: false,
      trigger_explanation: 'Extreme swell',
      limitations: [],
    },
  ],
  heatmap_points: [
    {
      latitude: 18.5,
      longitude: 72.5,
      label: 'Mumbai Offshore',
      region: 'Maharashtra Coast',
      swh_m: 6.74,
      max_wave_m: 11.12,
      mwp_s: 15.5,
      wind_kmh: 78.5,
      wind_dir_deg: 240,
      intensity: 0.67,
      risk_level: 'CRITICAL',
      color: '#ef4444',
      radius_km: 55,
    },
  ],
  alerts: [
    {
      alert_id: 'sim_alt_001',
      fingerprint: 'sim_fp_1',
      location_id: 'mumbai_offshore',
      location_name: 'Mumbai Offshore Grid',
      event_type: 'HIGH_WAVE',
      severity: 'RED',
      status: 'ACTIVE',
      trigger_reason: 'Simulated wave height 6.74m exceeds critical threshold (4.00m)',
      predicted_swh_m: 6.74,
      threshold_m: 4.0,
      forecast_horizon_hours: 6,
      expected_event_time_utc: '2026-10-30T07:00:00Z',
      anchor_time_utc: '2026-10-30T01:00:00Z',
      assessment_time_utc: '2026-10-30T01:00:00Z',
      source_name: 'Hypothetical Simulation Engine',
      data_status: 'FRESH',
      validation_status: 'simulation',
      is_reference: true,
      created_at_utc: '2026-10-30T01:00:00Z',
      updated_at_utc: '2026-10-30T01:00:00Z',
    },
  ],
  time_series: [
    {
      step: 6,
      time: 'T+06h',
      timestamp: '2026-10-30T07:00:00Z',
      actual: 6.74,
      lstm: 6.85,
      ann: 6.62,
      cnn1d: 6.7,
      persistence: 6.5,
      wind_kmh: 78.5,
      mwp_s: 15.5,
      risk_level: 'CRITICAL',
      isForecast: true,
    },
  ],
}

function ModalOpener() {
  const { openModal } = useSimulation()
  return (
    <button onClick={openModal} data-testid="open-modal-btn">
      Open Modal
    </button>
  )
}

describe('Interactive Ocean Hazard Simulation Mode', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getSimulationScenarios).mockResolvedValue(mockScenarios)
    vi.mocked(api.getSimulationStatus).mockResolvedValue({
      is_active: false,
      state: 'IDLE',
      current_step: 6,
      total_steps: 24,
      elapsed_hours: 0,
      disclaimer: 'Simulation results are hypothetical.',
    })
  })

  it('renders simulation control bar when simulation is active', async () => {
    vi.mocked(api.getSimulationStatus).mockResolvedValue(mockActiveResults.status)
    vi.mocked(api.getSimulationResults).mockResolvedValue(mockActiveResults)

    render(
      <BrowserRouter>
        <SimulationProvider>
          <SimulationControlBar />
        </SimulationProvider>
      </BrowserRouter>
    )

    await waitFor(() => {
      expect(screen.getByText(/SIMULATION MODE ACTIVE/i)).toBeInTheDocument()
      expect(screen.getByText(/Severe Wave Conditions/i)).toBeInTheDocument()
      expect(screen.getByText(/T\+06h/i)).toBeInTheDocument()
    })
  })

  it('renders simulation modal, displays scenario options, and initiates simulation', async () => {
    vi.mocked(api.startSimulation).mockResolvedValue(mockActiveResults)

    render(
      <SimulationProvider>
        <ModalOpener />
        <SimulationModal />
      </SimulationProvider>
    )

    fireEvent.click(screen.getByTestId('open-modal-btn'))

    await waitFor(() => {
      expect(screen.getByText(/Interactive Ocean Hazard Simulation Mode/i)).toBeInTheDocument()
      expect(screen.getByText(/Severe Wave Conditions/i)).toBeInTheDocument()
      expect(screen.getByText(/Tropical Cyclone Impact/i)).toBeInTheDocument()
    })

    const runBtn = screen.getByRole('button', { name: /Run Hazard Simulation/i })
    expect(runBtn).toBeInTheDocument()
    fireEvent.click(runBtn)

    await waitFor(() => {
      expect(api.startSimulation).toHaveBeenCalledWith('severe_wave', 6)
    })
  })

  it('renders hazard heatmap legend and toggles visibility state', async () => {
    let showHeatmap = true
    const handleToggle = vi.fn(() => {
      showHeatmap = !showHeatmap
    })

    const { rerender } = render(
      <SimulationProvider>
        <HazardHeatmapLegend showHeatmap={showHeatmap} onToggleHeatmap={handleToggle} />
      </SimulationProvider>
    )

    expect(screen.getByText(/Coastal Hazard Heatmap/i)).toBeInTheDocument()
    const toggleBtn = screen.getByTitle(/Hide heatmap layer/i)
    fireEvent.click(toggleBtn)
    expect(handleToggle).toHaveBeenCalled()

    rerender(
      <SimulationProvider>
        <HazardHeatmapLegend showHeatmap={false} onToggleHeatmap={handleToggle} />
      </SimulationProvider>
    )

    expect(screen.getByTitle(/Show heatmap layer/i)).toBeInTheDocument()
  })

  it('deactivates simulation and restores live data on reset', async () => {
    vi.mocked(api.getSimulationStatus).mockResolvedValue(mockActiveResults.status)
    vi.mocked(api.getSimulationResults).mockResolvedValue(mockActiveResults)
    vi.mocked(api.resetSimulation).mockResolvedValue({
      is_active: false,
      state: 'IDLE',
      current_step: 6,
      total_steps: 24,
      elapsed_hours: 0,
      disclaimer: 'Simulation results are hypothetical.',
    })

    render(
      <BrowserRouter>
        <SimulationProvider>
          <SimulationControlBar />
        </SimulationProvider>
      </BrowserRouter>
    )

    await waitFor(() => {
      expect(screen.getByText(/Return to Live Data/i)).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText(/Return to Live Data/i))

    await waitFor(() => {
      expect(api.resetSimulation).toHaveBeenCalled()
    })
  })

  describe('Ocean Hazard Simulation Laboratory Page (SimulationLabPage)', () => {
    it('renders master simulation laboratory with 3D ocean, timeline, and intelligence panel', async () => {
      vi.mocked(api.getSimulationStatus).mockResolvedValue(mockActiveResults.status)
      vi.mocked(api.getSimulationResults).mockResolvedValue(mockActiveResults)
      vi.mocked(api.startSimulation).mockResolvedValue(mockActiveResults)

      render(
        <BrowserRouter>
          <SimulationProvider>
            <SimulationLabPage />
          </SimulationProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        expect(screen.getByText(/Ocean Hazard Simulation Laboratory/i)).toBeInTheDocument()
        expect(screen.getByTestId('mock-simulation-r3f-canvas')).toBeInTheDocument()
        expect(screen.getByText(/24-Hour Storm Evolution Timeline/i)).toBeInTheDocument()
        expect(screen.getByText(/Real-Time Simulation State/i)).toBeInTheDocument()
      })
    })

    it('displays scenario cards with backend metrics, description, and recommended action', async () => {
      vi.mocked(api.getSimulationStatus).mockResolvedValue(mockActiveResults.status)
      vi.mocked(api.getSimulationResults).mockResolvedValue(mockActiveResults)
      vi.mocked(api.startSimulation).mockResolvedValue(mockActiveResults)

      render(
        <BrowserRouter>
          <SimulationProvider>
            <SimulationLabPage />
          </SimulationProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        expect(screen.getByText('Select Ocean Hazard Scenario')).toBeInTheDocument()
        expect(screen.getAllByText('Severe Wave Conditions').length).toBeGreaterThan(0)
        expect(screen.getAllByText('7.40 m').length).toBeGreaterThan(0)
        expect(screen.getAllByText('85 km/h').length).toBeGreaterThan(0)
        expect(screen.getAllByText('Severe Offshore Wave Action').length).toBeGreaterThan(0)
        expect(screen.getAllByText(/Complete prohibition of small craft operations/i).length).toBeGreaterThan(0)
      })
    })

    it('renders right-side intelligence panel with wave height, wind, and simulation event log', async () => {
      vi.mocked(api.getSimulationStatus).mockResolvedValue(mockActiveResults.status)
      vi.mocked(api.getSimulationResults).mockResolvedValue(mockActiveResults)
      vi.mocked(api.startSimulation).mockResolvedValue(mockActiveResults)

      render(
        <BrowserRouter>
          <SimulationProvider>
            <SimulationLabPage />
          </SimulationProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        expect(screen.getAllByText('6.74 m').length).toBeGreaterThan(0)
        expect(screen.getByText(/(78|79) km\/h/)).toBeInTheDocument()
        expect(screen.getByText(/82\/100/)).toBeInTheDocument()
        expect(screen.getByText(/Simulation Event Log/i)).toBeInTheDocument()
        expect(screen.getByText(/Outer Continental Shelf Impact/i)).toBeInTheDocument()
      })
    })

    it('renders hypothetical early-warning alerts feed with clear data isolation notice', async () => {
      vi.mocked(api.getSimulationStatus).mockResolvedValue(mockActiveResults.status)
      vi.mocked(api.getSimulationResults).mockResolvedValue(mockActiveResults)
      vi.mocked(api.startSimulation).mockResolvedValue(mockActiveResults)

      render(
        <BrowserRouter>
          <SimulationProvider>
            <SimulationLabPage />
          </SimulationProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        expect(screen.getByText(/Simulated Coastal Hazard Early Warnings/i)).toBeInTheDocument()
        expect(screen.getByText(/Hypothetical Model Output/i)).toBeInTheDocument()
        expect(screen.getByText(/Simulated wave height 6.74m exceeds critical threshold/i)).toBeInTheDocument()
      })
    })

    it('allows timeline stepping and view mode toggling between 3D and Geospatial Heatmap', async () => {
      vi.mocked(api.getSimulationStatus).mockResolvedValue(mockActiveResults.status)
      vi.mocked(api.getSimulationResults).mockResolvedValue(mockActiveResults)
      vi.mocked(api.startSimulation).mockResolvedValue(mockActiveResults)
      vi.mocked(api.stepSimulation).mockResolvedValue({
        ...mockActiveResults,
        status: { ...mockActiveResults.status, current_step: 7 },
      })

      render(
        <BrowserRouter>
          <SimulationProvider>
            <SimulationLabPage />
          </SimulationProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        expect(screen.getByText('3D Ocean Dynamics')).toBeInTheDocument()
        expect(screen.getByText('Geospatial Heatmap')).toBeInTheDocument()
      })

      // Toggle to geospatial heatmap
      fireEvent.click(screen.getByText('Geospatial Heatmap'))
      await waitFor(() => {
        expect(screen.getByText(/Spatial Risk Heatmap \(Simulated\)/i)).toBeInTheDocument()
      })

      // Step forward along timeline
      const stepForwardBtn = screen.getByTitle('Step forward 1 hour')
      fireEvent.click(stepForwardBtn)

      await waitFor(() => {
        expect(api.stepSimulation).toHaveBeenCalledWith(1)
      })
    })
  })
})

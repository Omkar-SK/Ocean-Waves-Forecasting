import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { AlertsPage } from '../pages/AlertsPage'
import { api } from '../services/api'
import type { AlertItem, AlertSummaryResponse, ThresholdConfigResponse, CoastalNode } from '../types'

vi.mock('../services/api', () => ({
  api: {
    listAlerts: vi.fn(),
    getAlertSummary: vi.fn(),
    getCoastalLocations: vi.fn(),
    getCoastalThresholds: vi.fn(),
    getAlertThresholds: vi.fn(),
    acknowledgeAlert: vi.fn(),
    getAlertHistory: vi.fn(),
    triggerAlertEvaluation: vi.fn(),
  },
}))

describe('Coastal Alerts & Early Warnings Page', () => {
  const mockAlerts: AlertItem[] = [
    {
      alert_id: 'alt_mumbai_001',
      fingerprint: 'fp_123',
      location_id: 'mumbai',
      location_name: 'Mumbai Reference',
      event_type: 'HIGH_WAVE',
      severity: 'RED',
      status: 'ACTIVE',
      trigger_reason: 'Wave height 3.80m exceeds critical threshold (3.50m)',
      predicted_swh_m: 3.8,
      threshold_m: 3.5,
      forecast_horizon_hours: 6,
      expected_event_time_utc: '2026-10-27T18:00:00Z',
      anchor_time_utc: '2026-10-27T12:00:00Z',
      assessment_time_utc: '2026-10-27T12:00:00Z',
      source_name: 'Open-Meteo ECMWF',
      data_status: 'FRESH',
      validation_status: 'reference',
      is_reference: true,
      created_at_utc: '2026-10-27T12:00:00Z',
      updated_at_utc: '2026-10-27T12:00:00Z',
    },
  ]

  const mockSummary: AlertSummaryResponse = {
    total_active: 1,
    critical_red: 1,
    high_orange: 0,
    caution_yellow: 0,
    resolved_count: 4,
    status_counts: { ACTIVE: 1, ACKNOWLEDGED: 0, RESOLVED: 4 },
    last_assessment_utc: '2026-10-27T12:00:00Z',
  }

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
    vi.mocked(api.listAlerts).mockResolvedValue(mockAlerts)
    vi.mocked(api.getAlertSummary).mockResolvedValue(mockSummary)
    vi.mocked(api.getCoastalLocations).mockResolvedValue(mockLocations)
    vi.mocked(api.getCoastalThresholds).mockResolvedValue(mockThresholds)
    vi.mocked(api.getAlertThresholds).mockResolvedValue(mockThresholds)
  })

  it('renders active alerts, summary count, and critical badges', async () => {
    render(<AlertsPage />)

    await waitFor(() => {
      expect(screen.getByText('Mumbai Reference')).toBeInTheDocument()
      expect(screen.getByText('RED RISK')).toBeInTheDocument()
      expect(screen.getByText(/Wave height 3.80m exceeds critical threshold/)).toBeInTheDocument()
    })
  })

  it('allows marine operator to acknowledge an active alert calling backend endpoint', async () => {
    vi.mocked(api.acknowledgeAlert).mockResolvedValue({
      alert_id: 'alt_mumbai_001',
      status: 'ACKNOWLEDGED',
      success: true,
    } as any)

    render(<AlertsPage />)

    await waitFor(() => {
      expect(screen.getByText('✍️ Acknowledge')).toBeInTheDocument()
    })

    const operatorInput = screen.getByPlaceholderText(/Operator notes/)
    fireEvent.change(operatorInput, { target: { value: 'Harbor Master notified' } })

    const confirmBtn = screen.getByText('✍️ Acknowledge')
    fireEvent.click(confirmBtn)

    await waitFor(() => {
      expect(api.acknowledgeAlert).toHaveBeenCalledWith(
        'alt_mumbai_001',
        'Harbor Master notified'
      )
    })
  })

  it('renders backend-configured threshold values correctly', async () => {
    render(<AlertsPage />)

    await waitFor(() => {
      expect(screen.getByText('≥ 4 m')).toBeInTheDocument()
      expect(screen.getByText('≥ 2.5 m')).toBeInTheDocument()
      expect(screen.getByText('≥ 1.5 m')).toBeInTheDocument()
    })
  })
})

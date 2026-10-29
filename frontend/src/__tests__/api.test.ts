import { describe, it, expect, vi, beforeEach } from 'vitest'
import { api } from '../services/api'

describe('Centralized API Client', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('handles successful API GET requests and parses JSON response', async () => {
    const mockHealth = {
      status: 'healthy',
      backend_online: true,
      checkpoints_available: 9,
      total_checkpoints: 9,
      scaler_available: true,
      dataset_available: true,
      live_api_available: true,
      timestamp: '2026-10-27T12:00:00Z',
      components: [],
    }

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => mockHealth,
    } as Response)

    const result = await api.getHealth()
    expect(result.backend_online).toBe(true)
    expect(result.checkpoints_available).toBe(9)
  })

  it('extracts detail string on HTTP 400/404 error responses', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: async () => ({ detail: 'Node id xyz not found in coastal registry' }),
    } as Response)

    await expect(api.getCoastalLocations()).rejects.toThrow('Node id xyz not found in coastal registry')
  })

  it('handles network failure with clear communication error', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValueOnce(new Error('Failed to fetch'))

    await expect(api.getHealth()).rejects.toThrow('Failed to fetch')
  })

  it('constructs correct query parameters for listAlerts', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => [],
    } as Response)

    await api.listAlerts('ACTIVE', 'CRITICAL', 'mumbai', 50)
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringContaining('/alerts?limit=50&status=ACTIVE&severity=CRITICAL&location_id=mumbai'),
      expect.any(Object)
    )
  })

  it('sends POST body correctly when acknowledging an alert', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        alert_id: 'alt_123',
        status: 'ACKNOWLEDGED',
        acknowledged_by: 'Marine Operator A',
      }),
    } as Response)

    const result = await api.acknowledgeAlert('alt_123', 'Action in progress')
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringContaining('/alerts/alt_123/acknowledge'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          operator_notes: 'Action in progress',
        }),
      })
    )
    expect(result.status).toBe('ACKNOWLEDGED')
  })

  it('calls coastal map data with horizon and source params', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => [],
    } as Response)

    await api.getCoastalMapData(24, 'live', 'lstm', true)
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringContaining('/coastal/map-data?horizon=24&source=live&model=lstm&force_refresh=true'),
      expect.any(Object)
    )
  })
})

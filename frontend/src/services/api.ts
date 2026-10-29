import type {
  AccuracyMetricsResponse,
  AlertHistoryItem,
  AlertItem,
  AlertSummaryResponse,
  CoastalMetricsResponse,
  CoastalNode,
  DataDriftResponse,
  ForecastLedgerItem,
  ForecastResponse,
  ForecastStatusResponse,
  HealthResponse,
  HistoricalPredictionsResponse,
  HistoricalSummaryResponse,
  Horizon,
  MetadataResponse,
  ModelMetric,
  ModelName,
  MonitoringEventItem,
  MonitoringSummaryResponse,
  MultiHorizonResponse,
  NodeMapSummaryItem,
  ResidualDataResponse,
  ScatterDataResponse,
  ThresholdConfigResponse,
} from '../types'

const BASE_URL = import.meta.env.VITE_API_URL ?? '/api'

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${endpoint}`
  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    })

    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`
      try {
        const errorJson = await response.json()
        if (errorJson.detail) {
          errorMessage = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail)
        } else if (errorJson.message) {
          errorMessage = errorJson.message
        }
      } catch {
        // Fall back to default error
      }
      throw new Error(errorMessage)
    }

    return await response.json()
  } catch (err: unknown) {
    if (err instanceof Error) {
      throw err
    }
    throw new Error('An unexpected network error occurred while communicating with the backend.')
  }
}

export const api = {
  // 1. System Health & Metadata
  getHealth: () => request<HealthResponse>('/health'),
  getMetadata: () => request<MetadataResponse>('/metadata'),

  // 2. Metrics & Benchmarks
  getMultiHorizonMetrics: () => request<MultiHorizonResponse>('/metrics/multi-horizon'),
  getMetricsByHorizon: (horizon: Horizon) => request<ModelMetric[]>(`/metrics?horizon=${horizon}`),
  getDetailedHorizonMetrics: (horizon: Horizon) => request<any>(`/metrics/detailed/${horizon}`),

  // 3. Live Forecasting & Registered Locations
  getCoastalLocations: () => request<CoastalNode[]>('/forecast/locations'),
  getLiveStatus: () => request<ForecastStatusResponse>('/forecast/status'),
  getLiveForecast: (
    horizon: Horizon = 6,
    model: string = 'all',
    source: string = 'live',
    locationId?: string
  ) => {
    const cleanModel = model.toLowerCase().replace('wave', '')
    const params = new URLSearchParams({
      horizon: horizon.toString(),
      model: cleanModel,
      source: source.toLowerCase(),
    })
    if (locationId) params.append('location_id', locationId)
    return request<ForecastResponse>(`/forecast/live?${params.toString()}`)
  },

  // 4. Coastal Risk Heatmap & Multi-Node Telemetry
  getCoastalMapData: (
    horizon: Horizon = 6,
    source: string = 'demo',
    model: string = 'lstm',
    forceRefresh: boolean = false
  ) => {
    const params = new URLSearchParams({
      horizon: horizon.toString(),
      source: source.toLowerCase(),
      model: model.toLowerCase(),
      force_refresh: forceRefresh ? 'true' : 'false',
    })
    return request<NodeMapSummaryItem[]>(`/coastal/map-data?${params.toString()}`)
  },
  getCoastalMetrics: (horizon: Horizon = 6, source: string = 'demo') => {
    return request<CoastalMetricsResponse>(`/coastal/metrics?horizon=${horizon}&source=${source}`)
  },
  getCoastalThresholds: () => request<ThresholdConfigResponse>('/coastal/thresholds'),

  // 5. Coastal Alerts & Early Warnings
  listAlerts: (status?: string, severity?: string, locationId?: string, limit: number = 100) => {
    const params = new URLSearchParams({ limit: limit.toString() })
    if (status && status !== 'ALL') params.append('status', status)
    if (severity && severity !== 'ALL') params.append('severity', severity)
    if (locationId && locationId !== 'ALL') params.append('location_id', locationId)
    return request<AlertItem[]>(`/alerts?${params.toString()}`)
  },
  getAlertSummary: () => request<AlertSummaryResponse>('/alerts/summary'),
  acknowledgeAlert: (alertId: string, operatorNotes?: string) => {
    return request<{ success: boolean; alert_id: string; status: string }>(`/alerts/${alertId}/acknowledge`, {
      method: 'POST',
      body: JSON.stringify({ operator_notes: operatorNotes || 'Acknowledged by operator in web interface.' }),
    })
  },
  getAlertHistory: (alertId: string) => request<AlertHistoryItem[]>(`/alerts/${alertId}/history`),
  triggerAlertEvaluation: (source: string = 'demo') => {
    return request<{ evaluated_nodes_count: number; active_alerts_count: number; timestamp_utc: string }>(
      `/alerts/evaluate?source=${source}`,
      { method: 'POST' }
    )
  },
  getAlertThresholds: (locationId?: string) => {
    const param = locationId ? `?location_id=${locationId}` : ''
    return request<ThresholdConfigResponse>(`/alerts/thresholds${param}`)
  },

  // 6. Forecast Accuracy, Verification & Model Drift
  getMonitoringSummary: () => request<MonitoringSummaryResponse>('/monitoring/summary'),
  getForecastAccuracy: (
    locationId?: string,
    horizon?: number,
    modelName?: string,
    minSamples: number = 10
  ) => {
    const params = new URLSearchParams({ min_samples: minSamples.toString() })
    if (locationId && locationId !== 'ALL') params.append('location_id', locationId)
    if (horizon && horizon !== 0) params.append('horizon', horizon.toString())
    if (modelName && modelName !== 'ALL') params.append('model_name', modelName)
    return request<AccuracyMetricsResponse>(`/monitoring/accuracy?${params.toString()}`)
  },
  getDataDriftReport: (locationId: string = 'mumbai_offshore', source: string = 'demo') => {
    const params = new URLSearchParams({
      location_id: locationId,
      source: source.toLowerCase(),
    })
    return request<DataDriftResponse>(`/monitoring/drift?${params.toString()}`)
  },
  listMonitoringEvents: (locationId?: string, status: string = 'ALL', eventType?: string, limit: number = 50) => {
    const params = new URLSearchParams({ status, limit: limit.toString() })
    if (locationId && locationId !== 'ALL') params.append('location_id', locationId)
    if (eventType) params.append('event_type', eventType)
    return request<MonitoringEventItem[]>(`/monitoring/events?${params.toString()}`)
  },
  triggerBacktest: (locationId: string = 'mumbai_offshore', maxRows: number = 500) => {
    const params = new URLSearchParams({
      location_id: locationId,
      max_rows: maxRows.toString(),
    })
    return request<{ success: boolean; inserted_observations: number; message: string }>(
      `/monitoring/backtest?${params.toString()}`,
      { method: 'POST' }
    )
  },
  getForecastLedger: (locationId?: string, limit: number = 50) => {
    const params = new URLSearchParams({ limit: limit.toString() })
    if (locationId && locationId !== 'ALL') params.append('location_id', locationId)
    return request<ForecastLedgerItem[]>(`/monitoring/ledger?${params.toString()}`)
  },

  // 7. Historical Analytics & Test Split Evaluation
  getHistoricalSummary: (horizon: Horizon = 1) => request<HistoricalSummaryResponse>(`/historical/summary?horizon=${horizon}`),
  getHistoricalPredictions: (
    horizon: Horizon = 1,
    model: ModelName = 'all',
    startDate?: string,
    endDate?: string,
    maxPoints: number = 400
  ) => {
    const params = new URLSearchParams({
      horizon: horizon.toString(),
      model: model.toString(),
      max_points: maxPoints.toString(),
    })
    if (startDate) params.append('start_date', startDate)
    if (endDate) params.append('end_date', endDate)
    return request<HistoricalPredictionsResponse>(`/historical/predictions?${params.toString()}`)
  },
  getHistoricalScatter: (horizon: Horizon = 1, model: ModelName = 'WaveLSTM', maxPoints: number = 600) => {
    const cleanModel = model.toString().toLowerCase().replace('wave', '')
    return request<ScatterDataResponse>(
      `/historical/scatter?horizon=${horizon}&model=${cleanModel}&max_points=${maxPoints}`
    )
  },
  getHistoricalResiduals: (horizon: Horizon = 1, model: ModelName = 'WaveLSTM', numBins: number = 30) => {
    const cleanModel = model.toString().toLowerCase().replace('wave', '')
    return request<ResidualDataResponse>(
      `/historical/residuals?horizon=${horizon}&model=${cleanModel}&num_bins=${numBins}`
    )
  },

  // 8. Export URL helper
  getExportUrl: (horizon: Horizon) => {
    return `${BASE_URL}/historical/export?horizon=${horizon}`
  },
}

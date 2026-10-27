import type {
  ForecastResponse,
  ForecastStatusResponse,
  HealthResponse,
  HistoricalPredictionsResponse,
  HistoricalSummaryResponse,
  Horizon,
  MetadataResponse,
  ModelMetric,
  ModelName,
  MultiHorizonResponse,
  ResidualDataResponse,
  ScatterDataResponse,
} from '../types'

const BASE_URL = import.meta.env.VITE_API_URL || '/api'

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
        }
      } catch {
        // Fall back to HTTP status
      }
      throw new Error(errorMessage)
    }

    return await response.json()
  } catch (err: unknown) {
    if (err instanceof Error) {
      throw err
    }
    throw new Error('An unknown network error occurred')
  }
}

export const api = {
  // System Health & Metadata
  getHealth: () => request<HealthResponse>('/health'),
  getMetadata: () => request<MetadataResponse>('/metadata'),

  // Metrics
  getMultiHorizonMetrics: () => request<MultiHorizonResponse>('/metrics/multi-horizon'),
  getMetricsByHorizon: (horizon: Horizon) => request<ModelMetric[]>(`/metrics?horizon=${horizon}`),

  // Live Forecasting
  getLiveStatus: () => request<ForecastStatusResponse>('/forecast/status'),
  getLiveForecast: (horizon: Horizon = 6, model: string = 'all', source: string = 'live') => {
    return request<ForecastResponse>(
      `/forecast/live?horizon=${horizon}&model=${model.toLowerCase().replace('wave', '')}&source=${source}`
    )
  },

  // Historical Test Evaluation
  getHistoricalSummary: () => request<HistoricalSummaryResponse>('/historical/summary'),
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
    return request<ScatterDataResponse>(
      `/historical/scatter?horizon=${horizon}&model=${model}&max_points=${maxPoints}`
    )
  },
  getHistoricalResiduals: (horizon: Horizon = 1, model: ModelName = 'WaveLSTM', numBins: number = 30) => {
    return request<ResidualDataResponse>(
      `/historical/residuals?horizon=${horizon}&model=${model}&num_bins=${numBins}`
    )
  },

  // Export URL helper
  getExportUrl: (horizon: Horizon, model?: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams({ horizon: horizon.toString() })
    if (model) params.append('model', model)
    if (startDate) params.append('start_date', startDate)
    if (endDate) params.append('end_date', endDate)
    return `${BASE_URL}/historical/export?${params.toString()}`
  },
}

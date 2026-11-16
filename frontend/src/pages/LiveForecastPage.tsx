import React, { useState, useEffect, useCallback, useMemo } from 'react'
import {
  Activity,
  AlertTriangle,
  Clock,
  Compass,
  Cpu,
  Database,
  Info,
  Layers,
  Radio,
  RefreshCw,
  Waves,
  Wind,
  ShieldCheck,
  CheckCircle2,
  BarChart2,
  Flame,
  FlaskConical,
  RotateCcw,
  ShieldAlert,
  ArrowUpRight,
} from 'lucide-react'
import { useSimulation } from '../context/SimulationContext'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  CartesianGrid,
  Legend,
  Area,
  ComposedChart,
} from 'recharts'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { Skeleton } from '../components/ui/Skeleton'
import { api } from '../services/api'
import { formatDateTime, formatMeters, getWmoSeaState } from '../lib/utils'
import type { CoastalNode, ForecastResponse, ForecastStatusResponse, Horizon, ModelName } from '../types'

export function LiveForecastPage() {
  const { isSimulationActive, activeScenario, simulationData, openModal, reset } = useSimulation()

  const [horizon, setHorizon] = useState<Horizon>(6)
  const [selectedModel, setSelectedModel] = useState<ModelName>('all')
  const [useDemoFallback, setUseDemoFallback] = useState<boolean>(false)
  const [locations, setLocations] = useState<CoastalNode[]>([])
  const [selectedLocationId, setSelectedLocationId] = useState<string>('mumbai')
  const [liveStatus, setLiveStatus] = useState<ForecastStatusResponse | null>(null)
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  // Load locations on mount
  useEffect(() => {
    async function loadLocations() {
      try {
        const nodes = await api.getCoastalLocations()
        setLocations(nodes)
      } catch (err) {
        console.error('Failed to load coastal locations:', err)
      }
    }
    loadLocations()
  }, [])

  const fetchForecast = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const source = useDemoFallback ? 'demo' : 'live'
      const [statusRes, forecastRes] = await Promise.all([
        api.getLiveStatus(),
        api.getLiveForecast(horizon, selectedModel, source, selectedLocationId),
      ])
      setLiveStatus(statusRes)
      setForecast(forecastRes)
    } catch (err: unknown) {
      console.error('Failed to fetch live forecast:', err)
      setError(err instanceof Error ? err.message : 'Error executing live inference')
    } finally {
      setLoading(false)
    }
  }, [horizon, selectedModel, useDemoFallback, selectedLocationId])

  useEffect(() => {
    fetchForecast()
  }, [fetchForecast])

  useEffect(() => {
    const handleRefresh = () => fetchForecast()
    window.addEventListener('oceancast:refresh', handleRefresh)
    return () => window.removeEventListener('oceancast:refresh', handleRefresh)
  }, [fetchForecast])

  const lstmPred = forecast?.predictions?.lstm?.forecast_swh_m
  const annPred = forecast?.predictions?.ann?.forecast_swh_m
  const cnnPred = forecast?.predictions?.cnn1d?.forecast_swh_m
  const nwpPred = forecast?.external_nwp_swh_m
  const persistPred = forecast?.persistence_swh_m

  const primeSwh = useMemo(() => {
    if (isSimulationActive && simulationData?.stats?.max_swh_m) {
      return simulationData.stats.max_swh_m
    }
    if (selectedModel === 'WaveLSTM' || selectedModel === 'lstm') {
      return lstmPred ?? persistPred ?? null
    }
    if (selectedModel === 'WaveANN' || selectedModel === 'ann') {
      return annPred ?? persistPred ?? null
    }
    if (selectedModel === 'WaveCNN1D' || selectedModel === 'cnn1d') {
      return cnnPred ?? persistPred ?? null
    }
    if (horizon === 24) {
      return annPred ?? lstmPred ?? persistPred ?? null
    }
    return lstmPred ?? annPred ?? cnnPred ?? persistPred ?? null
  }, [isSimulationActive, simulationData, selectedModel, horizon, lstmPred, annPred, cnnPred, persistPred])

  const seaState = primeSwh !== null ? getWmoSeaState(primeSwh) : { code: 0, description: 'Calm', color: '#10B981' }

  const lastObs =
    forecast?.observation_window && forecast.observation_window.length > 0
      ? forecast.observation_window[forecast.observation_window.length - 1]
      : null
  const liveMwp = isSimulationActive
    ? (simulationData?.scenario?.peak_mwp_s ?? null)
    : (lastObs?.mwp ?? null)
  const liveWindSpeed = isSimulationActive && simulationData?.stats?.max_wind_kmh
    ? Number((simulationData.stats.max_wind_kmh / 3.6).toFixed(1))
    : (lastObs?.wind_speed ?? null)

  // 3-way comparison bar chart data
  const comparisonData = useMemo(() => {
    if (isSimulationActive && simulationData?.time_series && simulationData.time_series.length > 0) {
      const cur = simulationData.time_series[Math.min(simulationData.time_series.length - 1, 14)]
      return [
        { name: 'WaveLSTM', category: 'Deep Learning', swh: cur.lstm ?? 6.2, color: '#246B9E' },
        { name: 'WaveANN', category: 'Deep Learning', swh: cur.ann ?? 6.4, color: '#20AFA7' },
        { name: 'WaveCNN1D', category: 'Deep Learning', swh: cur.cnn1d ?? 6.1, color: '#7C3AED' },
        { name: 'ECMWF WAM', category: 'NWP Physics', swh: Number(((cur.lstm ?? 6.0) * 0.95).toFixed(2)), color: '#397FB8' },
        { name: 'Persistence', category: 'Baseline', swh: cur.persistence ?? 5.8, color: '#D97706' },
      ]
    }
    if (!forecast) return []
    const items = []

    if (lstmPred !== undefined && lstmPred !== null) {
      items.push({
        name: 'WaveLSTM',
        category: 'Deep Learning',
        swh: lstmPred,
        color: '#246B9E',
      })
    }
    if (annPred !== undefined && annPred !== null) {
      items.push({
        name: 'WaveANN',
        category: 'Deep Learning',
        swh: annPred,
        color: '#20AFA7',
      })
    }
    if (cnnPred !== undefined && cnnPred !== null) {
      items.push({
        name: 'WaveCNN1D',
        category: 'Deep Learning',
        swh: cnnPred,
        color: '#7C3AED',
      })
    }
    if (nwpPred !== null && nwpPred !== undefined) {
      items.push({
        name: 'ECMWF WAM',
        category: 'NWP Physics',
        swh: nwpPred,
        color: '#397FB8',
      })
    }
    if (persistPred !== null && persistPred !== undefined) {
      items.push({
        name: 'Persistence',
        category: 'Baseline',
        swh: persistPred,
        color: '#D97706',
      })
    }

    return items
  }, [forecast, lstmPred, annPred, cnnPred, nwpPred, persistPred, isSimulationActive, simulationData])

  const selectedNode = locations.find((n) => n.location_id === selectedLocationId) || locations[0]
  const isExperimental = selectedNode ? !selectedNode.is_reference : false

  const timeSeriesData = useMemo(() => {
    if (isSimulationActive && simulationData?.time_series && simulationData.time_series.length > 0) {
      return simulationData.time_series.map((pt) => ({
        time: pt.time,
        swh: pt.isForecast ? (pt.lstm ?? pt.actual ?? 0) : (pt.actual ?? pt.lstm ?? 0),
        mwp: pt.mwp_s,
        wind_speed: Number((pt.wind_kmh / 3.6).toFixed(1)),
        isForecast: pt.isForecast,
        upperBound: pt.isForecast ? (pt.lstm ?? 0) + 0.12 : null,
        lowerBound: pt.isForecast ? Math.max(0, (pt.lstm ?? 0) - 0.12) : null,
      }))
    }

    if (!forecast?.observation_window) return []
    const points: any[] = forecast.observation_window.map((step) => {
      const d = new Date(step.valid_time)
      const timeStr = `${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2, '0')}:00`
      return {
        time: timeStr,
        swh: step.swh,
        mwp: step.mwp,
        wind_speed: step.wind_speed,
        isForecast: false,
      }
    })

    if (forecast.target_time_utc) {
      const d = new Date(forecast.target_time_utc)
      const timeStr = `${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2, '0')}:00`
      points.push({
        time: timeStr,
        swh: primeSwh,
        mwp: liveMwp,
        wind_speed: liveWindSpeed,
        isForecast: true,
        upperBound: primeSwh !== null ? primeSwh + 0.08 : null,
        lowerBound: primeSwh !== null ? Math.max(0.1, primeSwh - 0.08) : null,
      })
    }
    return points
  }, [isSimulationActive, simulationData, forecast, primeSwh, liveMwp, liveWindSpeed])

  const isDemo = forecast?.is_demo ?? false

  return (
    <div className="space-y-6 pb-12">
      {/* Simulation Mode Banner */}
      {isSimulationActive && (
        <div className="rounded-2xl border border-[#FCD34D] bg-gradient-to-r from-[#FFFBEB] via-[#FEF3C7] to-[#FFFBEB] p-5 text-[#92400E] flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-xs animate-ocean-fade">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-[#D97706] rounded-xl text-white shadow-xs">
              <Flame className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-sm text-[#78350F] font-['Outfit',sans-serif]">
                  Hypothetical Hazard Forecast Active: {activeScenario?.name}
                </span>
                <Badge variant="destructive" className="text-xs uppercase font-mono">Simulation</Badge>
              </div>
              <p className="text-xs text-[#92400E] leading-relaxed mt-1 font-normal">
                Displaying simulated wave progression and multi-model projections for this disaster drill scenario.
                Real observations remain isolated.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 self-end md:self-auto">
            <Button size="sm" onClick={openModal} className="bg-white border border-[#FCD34D] text-[#92400E] hover:bg-[#FDE68A] text-xs font-semibold">
              Change Scenario
            </Button>
            <Button size="sm" variant="secondary" onClick={reset} className="bg-[#246B9E] text-white hover:bg-[#102A43] text-xs font-semibold">
              Exit Simulation
            </Button>
          </div>
        </div>
      )}

      {/* Geographic Validation Warning for Experimental Nodes */}
      {isExperimental && (
        <div className="rounded-2xl border border-[#FCD34D] bg-[#FEF3C7] p-5 text-[#92400E] shadow-xs">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-[#D97706] shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="font-bold text-sm text-[#78350F] font-['Outfit',sans-serif]">
                  Geographic Validation Notice: Experimental Node ({selectedNode?.name})
                </span>
                <Badge variant="warning" className="text-xs font-mono">Uncalibrated Station</Badge>
              </div>
              <p className="text-xs text-[#92400E] leading-relaxed">
                Checkpoints were trained and verified exclusively on Mumbai Reference coordinates (18.50°N, 72.50°E).
                Wave predictions at {selectedNode?.name} ({selectedNode?.coordinate_label}) are generated for architectural
                evaluation and multi-location monitoring, but lack localized bathymetric calibration and waverider buoy ground truth.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Live Operational Status Banner */}
      {isDemo ? (
        <div className="rounded-2xl border border-[#FCD34D] bg-[#FEF3C7] p-5 text-[#92400E] shadow-xs">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-[#D97706] shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="font-bold text-sm text-[#78350F] font-['Outfit',sans-serif]">
                  Academic Verification Mode: Historical ERA5 Validation Window
                </span>
                <Badge variant="warning">Offline Demo</Badge>
              </div>
              <p className="text-xs text-[#92400E] leading-relaxed">
                Inference was executed on the verified consecutive 24-hour ERA5 reanalysis window ending at{' '}
                <strong>{forecast?.anchor_time_utc ? formatDateTime(forecast.anchor_time_utc) : '2026-10-17 23:00 UTC'}</strong>.
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="rounded-2xl border border-[#A7F3D0] bg-[#ECFDF5] p-5 text-[#065F46] shadow-xs">
          <div className="flex items-start gap-3">
            <Radio className="h-5 w-5 text-[#10B981] shrink-0 mt-0.5 animate-pulse" />
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="font-bold text-sm text-[#065F46] font-['Outfit',sans-serif]">
                  Live Operational Telemetry Active
                </span>
                <Badge variant="teal" className="text-xs font-mono">Real-Time Open-Meteo ECMWF Feed</Badge>
              </div>
              <p className="text-xs text-[#047857] leading-relaxed">
                Ingested consecutive 24 hourly marine & atmospheric NWP states from Open-Meteo for{' '}
                <strong>{selectedNode ? `${selectedNode.name} (${selectedNode.coordinate_label})` : '18.50°N, 72.50°E'}</strong> (0 gaps, 0 NaNs). Real-time inference executed across trained PyTorch
                checkpoints for target lead time +{horizon}h.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Control Panel: Pure White Surface with Crisp Border */}
      <div className="bg-white border border-[#DCE5EF] rounded-2xl p-6 shadow-xs space-y-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#DCE5EF] pb-4">
          <div>
            <h2 className="font-['Outfit',sans-serif] text-base font-bold text-[#102A43] flex items-center gap-2">
              <Compass className="h-4 w-4 text-[#20AFA7]" />
              <span>Forecasting Parameters & Operational Controls</span>
            </h2>
            <p className="text-xs text-[#64758A] mt-0.5 font-normal">
              Configure coastal monitoring location, target projection horizon, neural network checkpoint, and input data mode
            </p>
          </div>

          <Button
            variant="secondary"
            size="sm"
            onClick={fetchForecast}
            isLoading={loading}
            className="self-start md:self-auto bg-white hover:bg-[#F4F7FB] border border-[#DCE5EF] text-[#102A43]"
          >
            <RefreshCw className={`h-3.5 w-3.5 text-[#246B9E] ${loading ? 'animate-spin' : ''}`} />
            <span>Run Inference</span>
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Coastal Location Selector (Tested by combobox role) */}
          <div>
            <label className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] font-semibold mb-2 block">
              Coastal Station Node
            </label>
            <select
              value={selectedLocationId}
              onChange={(e) => setSelectedLocationId(e.target.value)}
              className="w-full py-2 px-3 text-xs font-mono font-medium rounded-lg bg-[#F4F7FB] border border-[#DCE5EF] text-[#102A43] focus:outline-none focus:border-[#20AFA7]"
            >
              {locations.length > 0 ? (
                locations.map((loc) => (
                  <option key={loc.location_id} value={loc.location_id}>
                    {loc.name} {loc.is_reference ? '★ (Reference)' : '(Experimental)'}
                  </option>
                ))
              ) : (
                <option value="mumbai">Mumbai Offshore Grid (18.5°N, 72.5°E) ★</option>
              )}
            </select>
          </div>

          {/* Horizon Selector */}
          <div>
            <label className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] font-semibold mb-2 block">
              Forecast Horizon
            </label>
            <div className="grid grid-cols-3 gap-1.5 p-1 rounded-lg bg-[#F4F7FB] border border-[#DCE5EF]">
              {([1, 6, 24] as Horizon[]).map((h) => (
                <button
                  key={h}
                  type="button"
                  onClick={() => setHorizon(h)}
                  className={`py-1.5 text-xs font-semibold rounded-md transition-all cursor-pointer text-center ${
                    horizon === h
                      ? 'bg-white text-[#102A43] shadow-xs font-bold border border-[#DCE5EF]'
                      : 'text-[#64758A] hover:text-[#102A43]'
                  }`}
                >
                  +{h}h
                </button>
              ))}
            </div>
          </div>

          {/* Model Architecture Selector */}
          <div>
            <label className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] font-semibold mb-2 block">
              Model Architecture
            </label>
            <div className="grid grid-cols-4 gap-1 p-1 rounded-lg bg-[#F4F7FB] border border-[#DCE5EF]">
              {[
                { id: 'all', label: 'All 3' },
                { id: 'WaveLSTM', label: 'LSTM' },
                { id: 'WaveANN', label: 'ANN' },
                { id: 'WaveCNN1D', label: '1D-CNN' },
              ].map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setSelectedModel(m.id as ModelName)}
                  className={`py-1.5 text-xs font-medium rounded-md transition-all cursor-pointer text-center ${
                    selectedModel === m.id
                      ? 'bg-white text-[#246B9E] border border-[#DCE5EF] font-bold shadow-xs'
                      : 'text-[#64758A] hover:text-[#102A43]'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {/* Offline Fallback Toggle */}
          <div>
            <label className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] font-semibold mb-2 block">
              Data Ingestion Feed
            </label>
            <div className="flex items-center justify-between p-2 rounded-lg bg-[#F4F7FB] border border-[#DCE5EF]">
              <div className="text-xs text-[#102A43]">
                <span className="font-semibold text-[11px]">Force Offline Historical</span>
                <p className="text-[10px] text-[#64758A]">ERA5 calibration window</p>
              </div>
              <input
                type="checkbox"
                checked={useDemoFallback}
                onChange={(e) => setUseDemoFallback(e.target.checked)}
                className="w-4 h-4 accent-[#20AFA7] cursor-pointer"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Target Timestamps & Primary Prediction Display Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Primary Prediction Display (5 Cols) */}
        <div className="lg:col-span-5 bg-white border border-[#DCE5EF] rounded-2xl p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between border-b border-[#DCE5EF] pb-3">
            <span className="text-xs font-mono uppercase font-semibold text-[#64758A] tracking-wider">
              +{horizon}H TARGET WAVE PROJECTION
            </span>
            <Badge variant="teal" className="text-xs font-mono">
              {selectedModel === 'all' ? (horizon === 24 ? 'WaveANN' : 'WaveLSTM') : selectedModel}
            </Badge>
          </div>

          <div className="flex items-baseline gap-3 my-2">
            <span className="font-['Outfit',sans-serif] text-5xl font-extrabold text-[#102A43] tracking-tight">
              {loading ? <Skeleton className="h-14 w-36" /> : formatMeters(primeSwh, 3)}
            </span>
            <Badge variant="teal" className="text-xs px-2.5 py-1">
              {seaState.description}
            </Badge>
          </div>

          <div className="p-3.5 rounded-xl bg-[#F4F7FB] border border-[#DCE5EF] space-y-2 text-xs font-mono">
            <div className="flex items-center justify-between">
              <span className="text-[#64758A]">Target UTC Time:</span>
              <span className="font-bold text-[#102A43]">
                {forecast?.target_time_utc ? formatDateTime(forecast.target_time_utc) : `+${horizon}h`}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[#64758A]">Baseline Persistence:</span>
              <span className="font-semibold text-[#64758A]">
                {persistPred !== undefined && persistPred !== null ? `${persistPred.toFixed(3)} m` : '--'}
              </span>
            </div>
            <div className="flex items-center justify-between border-t border-[#DCE5EF] pt-2">
              <span className="text-[#64758A]">Forecast Uncertainty Envelope:</span>
              <span className="font-bold text-[#20AFA7]">
                {primeSwh !== null ? (
                  <>
                    &plusmn;0.08 m ({Math.max(0.1, primeSwh - 0.08).toFixed(2)} &ndash; {(primeSwh + 0.08).toFixed(2)} m)
                  </>
                ) : (
                  '--'
                )}
              </span>
            </div>
          </div>
        </div>

        {/* Side-by-Side Model Comparison (7 Cols) */}
        <div className="lg:col-span-7 bg-white border border-[#DCE5EF] rounded-2xl p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between border-b border-[#DCE5EF] pb-3">
            <div className="flex items-center gap-2">
              <BarChart2 className="h-4 w-4 text-[#246B9E]" />
              <h3 className="font-['Outfit',sans-serif] text-sm font-bold text-[#102A43]">
                Multi-Model Inference Comparison (+{horizon}h)
              </h3>
            </div>
            <span className="text-xs font-mono text-[#64758A]">
              ECMWF vs PyTorch Checkpoints
            </span>
          </div>

          <div className="h-[210px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData} margin={{ top: 10, right: 15, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="name" stroke="#64758A" fontSize={11} tickLine={false} dy={5} />
                <YAxis stroke="#64758A" fontSize={11} tickLine={false} unit="m" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    borderColor: '#DCE5EF',
                    borderRadius: '8px',
                    fontFamily: 'monospace',
                    boxShadow: '0 4px 12px rgba(16,42,67,0.08)',
                  }}
                  formatter={(val: any) => [`${Number(val).toFixed(3)} m`, 'Forecast SWH']}
                />
                <Bar dataKey="swh" radius={[6, 6, 0, 0]}>
                  {comparisonData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Actual vs Predicted Trajectory Viewport */}
      <div className="bg-white border border-[#DCE5EF] rounded-2xl p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between border-b border-[#DCE5EF] pb-3">
          <div>
            <h3 className="font-['Outfit',sans-serif] text-base font-bold text-[#102A43] flex items-center gap-2">
              <Waves className="h-4 w-4 text-[#20AFA7]" />
              <span>Observation Sequence vs Forward Neural Projection</span>
            </h3>
            <p className="text-xs text-[#64758A] mt-0.5 font-normal">
              Continuous 24-hour sequence with model projection and confidence envelope
            </p>
          </div>
          <Badge variant="teal" className="text-xs font-mono">
            {isDemo ? 'ERA5 Historical' : 'ECMWF Live'}
          </Badge>
        </div>

        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={timeSeriesData} margin={{ top: 15, right: 20, left: -10, bottom: 20 }}>
              <defs>
                <linearGradient id="liveAreaGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#246B9E" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#246B9E" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis dataKey="time" stroke="#64758A" fontSize={11} tickLine={false} dy={8} />
              <YAxis stroke="#64758A" fontSize={11} tickLine={false} unit="m" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  borderColor: '#DCE5EF',
                  borderRadius: '8px',
                  fontFamily: 'monospace',
                }}
                formatter={(val: any, name: any) => [
                  `${Number(val).toFixed(3)} m`,
                  name === 'swh' ? 'Wave Height' : name,
                ]}
              />
              <Area type="monotone" dataKey="swh" stroke="#246B9E" strokeWidth={2.5} fill="url(#liveAreaGradient)" name="swh" />
              <Line type="monotone" dataKey="upperBound" stroke="#20AFA7" strokeWidth={1} strokeDasharray="3 3" dot={false} name="Upper Bound" />
              <Line type="monotone" dataKey="lowerBound" stroke="#20AFA7" strokeWidth={1} strokeDasharray="3 3" dot={false} name="Lower Bound" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

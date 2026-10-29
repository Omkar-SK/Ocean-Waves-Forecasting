import React, { useState, useEffect, useCallback, useMemo } from 'react'
import {
  History,
  Download,
  ScatterChart as ScatterIcon,
  BarChart2,
  Table as TableIcon,
  Layers,
  Calendar,
  ShieldCheck,
  TrendingUp,
} from 'lucide-react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from 'recharts'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { Skeleton } from '../components/ui/Skeleton'
import { api } from '../services/api'
import { formatMeters } from '../lib/utils'
import type {
  HistoricalPredictionsResponse,
  HistoricalSummaryResponse,
  Horizon,
  ModelName,
} from '../types'

export function HistoricalAnalysisPage() {
  const [horizon, setHorizon] = useState<Horizon>(1)
  const [model, setModel] = useState<ModelName>('WaveLSTM')
  const [summary, setSummary] = useState<HistoricalSummaryResponse | null>(null)
  const [predictions, setPredictions] = useState<HistoricalPredictionsResponse | null>(null)
  const [scatterData, setScatterData] = useState<any[]>([])
  const [residualsData, setResidualsData] = useState<any[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  // Date range presets
  const [datePreset, setDatePreset] = useState<'all' | 'monsoon' | 'calm' | 'cyclone'>('all')

  const fetchSummary = useCallback(async () => {
    try {
      const sum = await api.getHistoricalSummary()
      setSummary(sum)
    } catch (err) {
      console.error('Failed to load historical summary:', err)
    }
  }, [])

  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      let startDate: string | undefined = undefined
      let endDate: string | undefined = undefined

      if (datePreset === 'monsoon') {
        startDate = '2025-07-01'
        endDate = '2025-08-15'
      } else if (datePreset === 'calm') {
        startDate = '2024-11-01'
        endDate = '2024-12-15'
      } else if (datePreset === 'cyclone') {
        startDate = '2025-05-15'
        endDate = '2025-06-15'
      }

      const mClean = model.toLowerCase().replace('wave', '').trim()

      const [predData, scatRes, resRes] = await Promise.all([
        api.getHistoricalPredictions(horizon, model, startDate, endDate, 400),
        api.getHistoricalScatter(horizon, mClean as any, 400),
        api.getHistoricalResiduals(horizon, mClean as any, 25),
      ])

      setPredictions(predData)
      setScatterData(Array.isArray(scatRes) ? scatRes : (scatRes as any)?.points ?? [])
      setResidualsData(Array.isArray(resRes) ? resRes : (resRes as any)?.bins ?? [])
    } catch (err: unknown) {
      console.error('Failed to fetch historical evaluation data:', err)
      setError(err instanceof Error ? err.message : 'Error loading historical evaluation')
    } finally {
      setLoading(false)
    }
  }, [horizon, model, datePreset])

  useEffect(() => {
    fetchSummary()
  }, [fetchSummary])

  useEffect(() => {
    loadData()
  }, [loadData])

  useEffect(() => {
    const handleRefresh = () => loadData()
    window.addEventListener('oceancast:refresh', handleRefresh)
    return () => window.removeEventListener('oceancast:refresh', handleRefresh)
  }, [loadData])

  const exportUrl = api.getExportUrl(horizon)

  // Prepare time-series chart data
  const timeSeriesData = useMemo(() => {
    const rawList = (predictions as any)?.points || (predictions as any)?.data || []
    if (!rawList || rawList.length === 0) return []

    const mKey = model.toLowerCase().replace('wave', '').trim()

    return rawList.map((p: any) => {
      const ts = p.valid_time || p.timestamp || ''
      const d = new Date(ts)
      const timeStr = `${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2, '0')}:00`
      const actualVal = p.actual_swh ?? p.actual ?? 0

      let predVal = p.lstm_pred ?? p.lstm ?? p.predicted
      let resVal = p.lstm_error ?? p.residual

      if (mKey === 'ann') {
        predVal = p.ann_pred ?? p.ann ?? p.predicted
        resVal = p.ann_error ?? p.residual
      } else if (mKey === 'cnn1d' || mKey === 'cnn') {
        predVal = p.cnn1d_pred ?? p.cnn1d ?? p.predicted
        resVal = p.cnn1d_error ?? p.residual
      } else if (mKey === 'persistence') {
        predVal = actualVal
        resVal = 0
      }

      return {
        ...p,
        time: timeStr,
        timestamp: ts,
        actual: actualVal,
        predicted: predVal,
        residual: resVal,
        error_abs: resVal !== undefined && resVal !== null ? Math.abs(resVal) : undefined,
      }
    })
  }, [predictions, model])

  // Min and max for 1:1 reference diagonal
  const scatterMin = useMemo(() => {
    if (!scatterData.length) return 0
    return Math.max(0, Math.min(...scatterData.map((p) => p.actual)))
  }, [scatterData])

  const scatterMax = useMemo(() => {
    if (!scatterData.length) return 5
    return Math.max(...scatterData.map((p) => Math.max(p.actual, p.predicted)))
  }, [scatterData])

  return (
    <div className="space-y-6 pb-8">
      {/* Header Banner */}
      <div className="bg-white p-6 rounded-2xl border border-[#DCE5EF] shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#EAF2F9] text-[#245B8F] border border-[#DCE5EF]">
                <History className="h-4 w-4" />
              </span>
              <h1 className="text-xl font-bold tracking-tight text-[#15304B] font-['Outfit',sans-serif]">
                Historical Test Set Evaluation
              </h1>
            </div>
            <p className="mt-1 text-xs text-[#64758A] max-w-3xl leading-relaxed">
              Deep evaluation across 19,455 untouched chronological ERA5 reanalysis test records (2024–2026).
              Analyze actual vs predicted wave heights, 1:1 scatter correlations, and residual error distributions.
            </p>
          </div>

          <div className="flex items-center gap-2.5">
            <a
              href={exportUrl}
              download
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#F0F6FC] border border-[#DCE5EF] hover:border-[#20AFA7]/60 text-xs font-mono text-[#245B8F] transition-colors shadow-xs"
            >
              <Download className="h-3.5 w-3.5 text-[#20AFA7]" />
              <span>Export Test CSV (+{horizon}h)</span>
            </a>
          </div>
        </div>
      </div>

      {/* Control Panel: Horizon, Model, Presets */}
      <div className="bg-white p-5 rounded-xl border border-[#DCE5EF] shadow-xs">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Horizon Selector */}
          <div>
            <label className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] block mb-2">
              Forecast Horizon
            </label>
            <div className="grid grid-cols-3 gap-1 bg-[#F0F6FC] p-1 rounded-lg border border-[#DCE5EF]">
              {([1, 6, 24] as Horizon[]).map((h) => (
                <button
                  key={h}
                  onClick={() => setHorizon(h)}
                  className={`py-1.5 text-xs font-semibold rounded-md transition-colors ${
                    horizon === h
                      ? 'bg-[#20AFA7] text-white font-bold shadow-xs'
                      : 'text-[#64758A] hover:text-[#192B3C] hover:bg-white/50'
                  }`}
                >
                  +{h}h
                </button>
              ))}
            </div>
          </div>

          {/* Model Selector */}
          <div>
            <label className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] block mb-2">
              Model Architecture
            </label>
            <div className="grid grid-cols-3 gap-1 bg-[#F0F6FC] p-1 rounded-lg border border-[#DCE5EF]">
              {(['WaveLSTM', 'WaveANN', 'WaveCNN1D'] as ModelName[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setModel(m)}
                  className={`py-1.5 text-xs font-semibold rounded-md transition-colors ${
                    model === m
                      ? 'bg-white text-[#245B8F] border border-[#DCE5EF] font-bold shadow-xs'
                      : 'text-[#64758A] hover:text-[#192B3C]'
                  }`}
                >
                  {m.replace('Wave', '')}
                </button>
              ))}
            </div>
          </div>

          {/* Seasonal Presets */}
          <div>
            <label className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] block mb-2">
              Atmospheric Regime Presets
            </label>
            <div className="grid grid-cols-4 gap-1 bg-[#F0F6FC] p-1 rounded-lg border border-[#DCE5EF]">
              {[
                { id: 'all', label: 'All' },
                { id: 'monsoon', label: 'Monsoon' },
                { id: 'calm', label: 'Calm' },
                { id: 'cyclone', label: 'Cyclone' },
              ].map((p) => (
                <button
                  key={p.id}
                  onClick={() => setDatePreset(p.id as any)}
                  className={`py-1.5 text-xs font-medium rounded-md transition-colors ${
                    datePreset === p.id
                      ? 'bg-white text-[#245B8F] border border-[#DCE5EF] font-bold shadow-xs'
                      : 'text-[#64758A] hover:text-[#192B3C]'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 1. Time-Series Actual vs Predicted Track */}
      <div className="bg-white border border-[#DCE5EF] rounded-2xl p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between border-b border-[#DCE5EF] pb-3">
          <div>
            <h3 className="font-['Outfit',sans-serif] text-base font-bold text-[#15304B]">
              Ground-Truth Verification: Actual vs {model} (+{horizon}h)
            </h3>
            <p className="text-xs text-[#64758A] mt-0.5">
              Historical reanalysis test sample slice ({timeSeriesData.length} records)
            </p>
          </div>
          <Badge variant="teal" className="font-mono text-[10px]">
            Zero Data Leakage
          </Badge>
        </div>

        <div className="h-[300px] w-full">
          {loading ? (
            <div className="h-full flex items-center justify-center">
              <Skeleton className="h-full w-full rounded-xl" />
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timeSeriesData} margin={{ top: 10, right: 20, left: -10, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="time" stroke="#64758A" fontSize={11} tickLine={false} dy={8} />
                <YAxis stroke="#64758A" fontSize={11} tickLine={false} unit="m" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    borderColor: '#DCE5EF',
                    borderRadius: '8px',
                    color: '#192B3C',
                    boxShadow: '0 4px 6px -1px rgba(0,0,0,0.07)',
                    fontFamily: 'var(--font-mono)',
                  }}
                  formatter={(val: any, name: any) => [
                    `${Number(val).toFixed(3)} m`,
                    name === 'actual' ? 'Ground-Truth SWH' : `${model} Predicted`,
                  ]}
                />
                <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '10px', fontSize: '11px', fontFamily: 'var(--font-mono)' }} />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#245B8F"
                  strokeWidth={2}
                  dot={false}
                  name="Ground-Truth SWH"
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#20AFA7"
                  strokeWidth={2}
                  strokeDasharray="3 3"
                  dot={false}
                  name={`${model} Predicted`}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* 2. Scatter Plot & Residual Error Distribution Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Scatter Plot */}
        <div className="bg-white border border-[#DCE5EF] rounded-2xl p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between border-b border-[#DCE5EF] pb-3">
            <h3 className="font-['Outfit',sans-serif] text-sm font-bold text-[#15304B] flex items-center gap-2">
              <ScatterIcon className="h-4 w-4 text-[#20AFA7]" />
              <span>1:1 Actual vs Predicted Correlation</span>
            </h3>
            <span className="text-xs font-mono text-[#64758A]">Ideal 45° line</span>
          </div>

          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, left: -10, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="actual" stroke="#64758A" fontSize={11} name="Actual SWH" unit="m" />
                <YAxis dataKey="predicted" stroke="#64758A" fontSize={11} name="Predicted SWH" unit="m" />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    borderColor: '#DCE5EF',
                    borderRadius: '8px',
                    color: '#192B3C',
                    boxShadow: '0 4px 6px -1px rgba(0,0,0,0.07)',
                    fontFamily: 'var(--font-mono)',
                  }}
                  formatter={(val: any, name: any) => [`${Number(val).toFixed(3)} m`, name]}
                />
                <ReferenceLine
                  segment={[{ x: scatterMin, y: scatterMin }, { x: scatterMax, y: scatterMax }]}
                  stroke="#8291A2"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                />
                <Scatter data={scatterData} fill="#245B8F" fillOpacity={0.6} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Residuals Distribution Histogram */}
        <div className="bg-white border border-[#DCE5EF] rounded-2xl p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between border-b border-[#DCE5EF] pb-3">
            <h3 className="font-['Outfit',sans-serif] text-sm font-bold text-[#15304B] flex items-center gap-2">
              <BarChart2 className="h-4 w-4 text-[#245B8F]" />
              <span>Residual Error Distribution (Actual - Predicted)</span>
            </h3>
            <span className="text-xs font-mono text-[#64758A]">Gaussian Error</span>
          </div>

          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={residualsData} margin={{ top: 10, right: 10, left: -10, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                <XAxis dataKey="bin_center" stroke="#64758A" fontSize={11} tickLine={false} dy={8} />
                <YAxis stroke="#64758A" fontSize={11} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    borderColor: '#DCE5EF',
                    borderRadius: '8px',
                    color: '#192B3C',
                    boxShadow: '0 4px 6px -1px rgba(0,0,0,0.07)',
                    fontFamily: 'var(--font-mono)',
                  }}
                  formatter={(val: any) => [`${val} records`, 'Frequency']}
                />
                <Bar dataKey="count" fill="#397FB8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}

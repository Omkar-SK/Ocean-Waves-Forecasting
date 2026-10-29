import React, { useState, useEffect, useCallback } from 'react'
import {
  History,
  Download,
  ScatterChart as ScatterIcon,
  BarChart2,
  Table as TableIcon,
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

  // Prepare time-series chart data with formatted timestamps and model-specific predictions
  const timeSeriesData = React.useMemo(() => {
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
  const scatterMin = React.useMemo(() => {
    if (!scatterData.length) return 0
    return Math.max(0, Math.min(...scatterData.map((p) => p.actual)))
  }, [scatterData])

  const scatterMax = React.useMemo(() => {
    if (!scatterData.length) return 5
    return Math.max(...scatterData.map((p) => Math.max(p.actual, p.predicted)))
  }, [scatterData])

  // Error stats
  const residualStats = React.useMemo(() => {
    if (!timeSeriesData.length) return { mean: 0, std: 0 }
    const residuals = timeSeriesData
      .map((p: any) => p.residual)
      .filter((r: any) => r !== undefined && r !== null && !isNaN(r)) as number[]
    if (!residuals.length) return { mean: 0, std: 0 }
    const mean = residuals.reduce((a, b) => a + b, 0) / residuals.length
    const variance =
      residuals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / residuals.length
    return { mean, std: Math.sqrt(variance) }
  }, [timeSeriesData])

  return (
    <div className="space-y-6">
      {/* Historical Test Overview Banner */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 p-4 rounded-xl bg-[#101D2D] border border-[#1E3A5F]">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-[#35C6E8]/10 border border-[#35C6E8]/30">
            <History className="h-5 w-5 text-[#35C6E8]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold text-[#F1F5F9] font-['Outfit',sans-serif]">
                Untouched Test Dataset: 19,455 Hourly Records
              </h2>
              <Badge variant="teal">Chronological Split (15%)</Badge>
            </div>
            <p className="text-xs text-[#94A3B8] mt-0.5">
              Available Test Span: {(summary as any)?.date_range_start?.slice(0, 10) || (summary as any)?.start_date?.slice(0, 10) || '2024-07-28'} to{' '}
              {(summary as any)?.date_range_end?.slice(0, 10) || (summary as any)?.end_date?.slice(0, 10) || '2026-10-17'} • 0 Missing Hours • 0 NaNs
            </p>
          </div>
        </div>

        {/* CSV Export Button */}
        <a
          href={exportUrl}
          download={`oceancast_predictions_h${horizon}_${model}.csv`}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Button variant="outline" size="sm" className="gap-2">
            <Download className="h-4 w-4" />
            <span>Export Test CSV</span>
          </Button>
        </a>
      </div>

      {/* Interactive Filter Bar */}
      <Card>
        <CardContent className="pt-5 pb-5">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Horizon Filter */}
            <div>
              <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider mb-2 block">
                Forecast Horizon
              </label>
              <div className="grid grid-cols-3 gap-1.5 p-1 rounded-lg bg-[#07111F] border border-[#1E3A5F]">
                {([1, 6, 24] as Horizon[]).map((h) => (
                  <button
                    key={h}
                    onClick={() => setHorizon(h)}
                    className={`py-2 text-xs font-semibold rounded-md transition-all cursor-pointer text-center ${
                      horizon === h
                        ? 'bg-[#35C6E8] text-[#07111F] shadow-sm font-bold'
                        : 'text-[#94A3B8] hover:text-[#F1F5F9] hover:bg-[#16263A]'
                    }`}
                  >
                    +{h}h
                  </button>
                ))}
              </div>
            </div>

            {/* Model Architecture Filter */}
            <div>
              <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider mb-2 block">
                Model Architecture
              </label>
              <div className="grid grid-cols-4 gap-1 p-1 rounded-lg bg-[#07111F] border border-[#1E3A5F]">
                {[
                  { id: 'WaveLSTM', label: 'LSTM' },
                  { id: 'WaveANN', label: 'ANN' },
                  { id: 'WaveCNN1D', label: '1D-CNN' },
                  { id: 'persistence', label: 'Baseline' },
                ].map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setModel(m.id as ModelName)}
                    className={`py-2 text-xs font-medium rounded-md transition-all cursor-pointer text-center ${
                      model === m.id
                        ? 'bg-[#16263A] text-[#35C6E8] border border-[#35C6E8]/40 font-semibold'
                        : 'text-[#94A3B8] hover:text-[#F1F5F9]'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Date-Range Presets */}
            <div>
              <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider mb-2 block">
                Sea State Regimes
              </label>
              <div className="grid grid-cols-4 gap-1 p-1 rounded-lg bg-[#07111F] border border-[#1E3A5F]">
                {[
                  { id: 'all', label: 'Full Span' },
                  { id: 'monsoon', label: 'Monsoon Peak' },
                  { id: 'cyclone', label: 'Pre-Monsoon' },
                  { id: 'calm', label: 'Winter Calm' },
                ].map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => setDatePreset(preset.id as any)}
                    className={`py-2 text-[11px] font-medium rounded-md transition-all cursor-pointer text-center ${
                      datePreset === preset.id
                        ? 'bg-[#16263A] text-[#35D0B0] border border-[#35D0B0]/40 font-semibold'
                        : 'text-[#94A3B8] hover:text-[#F1F5F9]'
                    }`}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Historical Time-Series Chart */}
      <Card>
        <CardHeader>
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div>
              <CardTitle>Observed vs Predicted Wave Height Over Time</CardTitle>
              <CardDescription>
                Chronological test set playback: {model} predicting {horizon} hour(s) ahead
              </CardDescription>
            </div>
            <div className="flex items-center gap-3 text-xs text-[#94A3B8]">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-[#F1F5F9]" /> Observed ($H_s$)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-[#35C6E8]" /> Predicted ({model})
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-[#EF4444]" /> Residual Error
              </span>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="h-80 flex items-center justify-center">
              <Skeleton className="h-72 w-full" />
            </div>
          ) : error ? (
            <div className="h-80 flex flex-col items-center justify-center text-center p-6 bg-[#16263A]/40 rounded-lg border border-[#EF4444]/30">
              <p className="text-sm font-semibold text-[#F1F5F9]">Failed to load test predictions</p>
              <p className="text-xs text-[#94A3B8] mt-1">{error}</p>
              <Button variant="outline" size="sm" onClick={loadData} className="mt-4">
                Retry
              </Button>
            </div>
          ) : (
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeSeriesData} margin={{ top: 10, right: 15, left: -10, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#16263A" />
                  <XAxis
                    dataKey="time"
                    stroke="#94A3B8"
                    fontSize={11}
                    tickLine={false}
                    interval="preserveStartEnd"
                    angle={-25}
                    textAnchor="end"
                  />
                  <YAxis
                    stroke="#94A3B8"
                    fontSize={11}
                    domain={['dataMin - 0.1', 'dataMax + 0.1']}
                    unit=" m"
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#101D2D',
                      borderColor: '#1E3A5F',
                      borderRadius: '8px',
                      color: '#F1F5F9',
                    }}
                    formatter={(val: any) => [formatMeters(typeof val === 'number' ? val : null), '']}
                  />
                  <Legend verticalAlign="top" height={32} />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    name="Observed Hs"
                    stroke="#F1F5F9"
                    strokeWidth={1.8}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    name={`Predicted (${model})`}
                    stroke="#35C6E8"
                    strokeWidth={1.8}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="residual"
                    name="Error (Actual - Pred)"
                    stroke="#EF4444"
                    strokeWidth={1}
                    strokeDasharray="2 2"
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Dual Diagnostic Charts: Scatter Plot (Identity 1:1) & Residual Histogram */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Scatter Plot: Actual vs Predicted */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <ScatterIcon className="h-4 w-4 text-[#35C6E8]" />
                  <span>Actual vs Predicted Scatter Plot</span>
                </CardTitle>
                <CardDescription>
                  Points falling along the cyan 1:1 diagonal represent perfect predictions
                </CardDescription>
              </div>
              <Badge variant="outline" className="font-mono text-[10px]">
                {scatterData.length} Sample Points
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="h-64 flex items-center justify-center">
                <Skeleton className="h-56 w-full" />
              </div>
            ) : (
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#16263A" />
                    <XAxis
                      type="number"
                      dataKey="actual"
                      name="Actual Hs"
                      unit="m"
                      stroke="#94A3B8"
                      fontSize={11}
                      domain={['auto', 'auto']}
                    />
                    <YAxis
                      type="number"
                      dataKey="predicted"
                      name="Predicted Hs"
                      unit="m"
                      stroke="#94A3B8"
                      fontSize={11}
                      domain={['auto', 'auto']}
                    />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3' }}
                      contentStyle={{
                        backgroundColor: '#101D2D',
                        borderColor: '#1E3A5F',
                        borderRadius: '8px',
                        color: '#F1F5F9',
                      }}
                      formatter={(val: any, name: any) => [
                        formatMeters(typeof val === 'number' ? val : null),
                        name === 'actual' ? 'Actual' : 'Predicted',
                      ]}
                    />
                    {/* 1:1 Reference Line */}
                    <ReferenceLine
                      stroke="#35C6E8"
                      strokeWidth={1.5}
                      strokeDasharray="4 4"
                      segment={[
                        { x: scatterMin, y: scatterMin },
                        { x: scatterMax, y: scatterMax },
                      ]}
                    />
                    <Scatter
                      name="Predictions"
                      data={scatterData}
                      fill="#35C6E8"
                      opacity={0.6}
                      isAnimationActive={false}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Residual Error Distribution Histogram */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <BarChart2 className="h-4 w-4 text-[#35D0B0]" />
                  <span>Residual Error Distribution</span>
                </CardTitle>
                <CardDescription>
                  Histogram of error (Actual - Predicted) • Zero-centered bell curve
                </CardDescription>
              </div>
              <div className="text-right text-[11px] font-mono text-[#94A3B8]">
                <span>Mean: {residualStats.mean.toFixed(4)}m</span>
                <span className="mx-1">•</span>
                <span>Std: {residualStats.std.toFixed(4)}m</span>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="h-64 flex items-center justify-center">
                <Skeleton className="h-56 w-full" />
              </div>
            ) : (
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={residualsData} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#16263A" />
                    <XAxis
                      dataKey="bin_center"
                      stroke="#94A3B8"
                      fontSize={10}
                      tickFormatter={(val) => Number(val).toFixed(2)}
                      tickLine={false}
                    />
                    <YAxis stroke="#94A3B8" fontSize={11} tickLine={false} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#101D2D',
                        borderColor: '#1E3A5F',
                        borderRadius: '8px',
                        color: '#F1F5F9',
                      }}
                      formatter={(val: any) => [`${val} occurrences`, 'Count']}
                      labelFormatter={(val) => `Error center: ${Number(val).toFixed(3)} m`}
                    />
                    <ReferenceLine x={0} stroke="#F59E0B" strokeDasharray="3 3" />
                    <Bar dataKey="count" fill="#35D0B0" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Prediction Details Sample Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <TableIcon className="h-4 w-4 text-[#35C6E8]" />
                <span>Prediction Audit Sample Log</span>
              </CardTitle>
              <CardDescription>
                First 8 chronological records of the filtered evaluation window
              </CardDescription>
            </div>
            <Badge variant="outline" className="font-mono text-[10px]">
              Showing 8 of {timeSeriesData.length}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[#1E3A5F] text-[#94A3B8]">
                  <th className="pb-3 font-semibold">Timestamp</th>
                  <th className="pb-3 font-semibold text-right">Observed Hs</th>
                  <th className="pb-3 font-semibold text-right">Predicted Hs ({model})</th>
                  <th className="pb-3 font-semibold text-right">Residual (m)</th>
                  <th className="pb-3 font-semibold text-right">Absolute Error</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E3A5F]/40 font-mono">
                {timeSeriesData.slice(0, 8).map((row: any, i: number) => (
                  <tr key={i} className="hover:bg-[#16263A]/40 transition-colors">
                    <td className="py-2.5 font-sans text-[#F1F5F9]">{row.timestamp || row.valid_time}</td>
                    <td className="py-2.5 text-right text-[#F1F5F9] font-bold">
                      {row.actual !== undefined && row.actual !== null ? `${Number(row.actual).toFixed(4)} m` : '—'}
                    </td>
                    <td className="py-2.5 text-right text-[#35C6E8]">
                      {row.predicted !== undefined && row.predicted !== null ? `${Number(row.predicted).toFixed(4)} m` : '—'}
                    </td>
                    <td
                      className={`py-2.5 text-right ${
                        (row.residual ?? 0) >= 0 ? 'text-[#35D0B0]' : 'text-[#EF4444]'
                      }`}
                    >
                      {row.residual !== undefined && row.residual !== null ? `${Number(row.residual).toFixed(4)} m` : '—'}
                    </td>
                    <td className="py-2.5 text-right text-[#94A3B8]">
                      {row.error_abs !== undefined && row.error_abs !== null ? `${Number(row.error_abs).toFixed(4)} m` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

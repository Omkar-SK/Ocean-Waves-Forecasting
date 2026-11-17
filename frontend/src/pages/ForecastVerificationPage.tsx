import React, { useState, useEffect, useMemo } from 'react'
import {
  ShieldCheck,
  Activity,
  AlertTriangle,
  RefreshCw,
  Info,
  Layers,
  Database,
  BarChart2,
  Calendar,
  Clock,
  CheckCircle2,
  Table as TableIcon,
  Radio,
  TrendingDown,
  TrendingUp,
  AlertCircle,
  HelpCircle,
  Zap,
} from 'lucide-react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  ReferenceLine,
} from 'recharts'
import { api } from '../services/api'
import type {
  ContinuousAccuracyMetrics,
  SevereEventTrackingMetrics,
  VerificationDashboardResponse,
  VerificationRecord,
  VerificationRecordStatus,
  VerificationWindow,
} from '../types'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { StatusIndicator } from '../components/ui/StatusIndicator'

export function ForecastVerificationPage() {
  const [loading, setLoading] = useState<boolean>(true)
  const [refreshing, setRefreshing] = useState<boolean>(false)
  const [dashboard, setDashboard] = useState<VerificationDashboardResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Filters
  const [selectedHorizon, setSelectedHorizon] = useState<1 | 6 | 24>(6)
  const [selectedWindow, setSelectedWindow] = useState<VerificationWindow>('all_time')
  const [statusFilter, setStatusFilter] = useState<string>('ALL')
  const [locationFilter, setLocationFilter] = useState<string>('mumbai_offshore')

  const fetchDashboard = async (showRefreshing = false) => {
    if (showRefreshing) setRefreshing(true)
    setError(null)
    try {
      const data = await api.getVerificationDashboard(
        locationFilter === 'ALL' ? undefined : locationFilter,
        100
      )
      setDashboard(data)
    } catch (err: any) {
      setError(err?.message || 'Failed to load forecast verification dashboard.')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    fetchDashboard()
  }, [locationFilter])

  const handleTriggerMatching = async () => {
    setRefreshing(true)
    try {
      await api.triggerVerificationMatching(
        locationFilter === 'ALL' ? undefined : locationFilter,
        1800
      )
      await fetchDashboard()
    } catch (err: any) {
      setError(err?.message || 'Error triggering verification matching.')
    } finally {
      setRefreshing(false)
    }
  }

  const handleSeedDemoObservations = async () => {
    setRefreshing(true)
    try {
      await api.seedDemoObservations(24)
      await fetchDashboard()
    } catch (err: any) {
      setError(err?.message || 'Error seeding demo observations.')
    } finally {
      setRefreshing(false)
    }
  }

  // Active Horizon Metrics
  const currentMetrics = useMemo<ContinuousAccuracyMetrics | null>(() => {
    if (!dashboard) return null
    const hKey = `${selectedHorizon}h`
    return dashboard.horizon_metrics?.[hKey]?.[selectedWindow] || null
  }, [dashboard, selectedHorizon, selectedWindow])

  // Filtered Verification Records
  const filteredRecords = useMemo(() => {
    if (!dashboard?.recent_verifications) return []
    return dashboard.recent_verifications.filter((r) => {
      if (r.forecast_horizon !== selectedHorizon) return false
      if (statusFilter !== 'ALL' && r.status !== statusFilter) return false
      return true
    })
  }, [dashboard, selectedHorizon, statusFilter])

  // Chart Data: Forecast vs Observed timeline for verified or pending records
  const timelineChartData = useMemo(() => {
    if (!dashboard?.recent_verifications) return []
    const sorted = [...dashboard.recent_verifications]
      .filter((r) => r.forecast_horizon === selectedHorizon)
      .sort((a, b) => new Date(a.target_timestamp).getTime() - new Date(b.target_timestamp).getTime())
      .slice(-30) // last 30 chronologically

    return sorted.map((r) => {
      const timeLabel = new Date(r.target_timestamp).toLocaleTimeString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
      return {
        time: timeLabel,
        predicted: r.predicted_swh,
        observed: r.observed_swh !== null && r.observed_swh !== undefined ? r.observed_swh : null,
        persistence: r.current_swh,
        error: r.forecast_error !== null && r.forecast_error !== undefined ? r.forecast_error : null,
        status: r.status,
      }
    })
  }, [dashboard, selectedHorizon])

  const getStatusBadge = (status: VerificationRecordStatus) => {
    switch (status) {
      case 'VERIFIED':
        return (
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-semibold text-emerald-700 border border-emerald-200">
            <CheckCircle2 className="h-3 w-3" />
            Verified
          </span>
        )
      case 'PENDING_VERIFICATION':
        return (
          <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-700 border border-amber-200">
            <Clock className="h-3 w-3" />
            Pending Target
          </span>
        )
      case 'OBSERVATION_UNAVAILABLE':
        return (
          <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-600 border border-slate-300">
            <AlertCircle className="h-3 w-3" />
            Telemetry Unavailable
          </span>
        )
      case 'INVALID_OBSERVATION':
        return (
          <span className="inline-flex items-center gap-1 rounded-full bg-red-50 px-2 py-0.5 text-xs font-semibold text-red-700 border border-red-200">
            <AlertTriangle className="h-3 w-3" />
            Invalid Telemetry
          </span>
        )
    }
  }

  if (loading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <RefreshCw className="h-8 w-8 animate-spin text-[#246B9E]" />
          <p className="text-sm font-medium text-[#486581]">Loading prospective forecast verification telemetry...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* 1. Header & Controls */}
      <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center border-b border-slate-200 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-[#102A43]">
              Continuous Real-Time Forecast Verification
            </h1>
            <span className="rounded-md bg-teal-50 px-2.5 py-0.5 text-xs font-semibold text-teal-700 border border-teal-200">
              Live MLOps
            </span>
          </div>
          <p className="mt-1 text-sm text-[#486581]">
            Prospective matching of issued wave forecasts against independent physical observations with automated error tracking and extreme-event monitoring.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <select
            value={locationFilter}
            onChange={(e) => setLocationFilter(e.target.value)}
            className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-[#102A43] shadow-xs focus:border-[#246B9E] focus:outline-none"
          >
            <option value="mumbai_offshore">Mumbai Offshore (Reference Node)</option>
            <option value="chennai_port">Chennai Port (Experimental)</option>
            <option value="kochi_offshore">Kochi Offshore (Experimental)</option>
            <option value="ALL">All Coastal Monitoring Nodes</option>
          </select>

          <Button
            variant="outline"
            size="sm"
            onClick={handleTriggerMatching}
            disabled={refreshing}
            className="flex items-center gap-1.5 text-xs font-medium text-[#102A43]"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? 'animate-spin' : ''}`} />
            Match & Verify
          </Button>

          <Button
            variant="primary"
            size="sm"
            onClick={handleSeedDemoObservations}
            disabled={refreshing}
            className="flex items-center gap-1.5 text-xs font-medium bg-[#246B9E] hover:bg-[#1B4E72] text-white"
          >
            <Zap className="h-3.5 w-3.5" />
            Seed Demo Buoy Telemetry
          </Button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700 flex items-center gap-2">
          <AlertCircle className="h-5 w-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* 2. Operational Summary KPI Cards */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <Card className="p-4 border-slate-200 bg-white shadow-xs">
          <p className="text-xs font-medium text-[#486581] uppercase tracking-wider">Total Issued</p>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold text-[#102A43]">{dashboard?.total_forecasts_issued ?? 0}</span>
            <span className="text-xs text-slate-400">Forecasts</span>
          </div>
        </Card>

        <Card className="p-4 border-emerald-100 bg-emerald-50/30 shadow-xs">
          <p className="text-xs font-medium text-emerald-800 uppercase tracking-wider">Verified Prospective</p>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold text-emerald-700">{dashboard?.total_verified_observations ?? 0}</span>
            <span className="text-xs font-semibold text-emerald-600">Matched Pairs</span>
          </div>
        </Card>

        <Card className="p-4 border-amber-100 bg-amber-50/30 shadow-xs">
          <p className="text-xs font-medium text-amber-800 uppercase tracking-wider">Pending Target Time</p>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold text-amber-700">{dashboard?.total_pending_verifications ?? 0}</span>
            <span className="text-xs text-amber-600">In Horizon</span>
          </div>
        </Card>

        <Card className="p-4 border-slate-200 bg-slate-50/50 shadow-xs">
          <p className="text-xs font-medium text-slate-600 uppercase tracking-wider">Telemetry Missing</p>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold text-slate-700">{dashboard?.total_unavailable_observations ?? 0}</span>
            <span className="text-xs text-slate-500">Unavailable</span>
          </div>
        </Card>

        <Card className="p-4 border-red-100 bg-red-50/30 shadow-xs">
          <p className="text-xs font-medium text-red-800 uppercase tracking-wider">Rejected Telemetry</p>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold text-red-700">{dashboard?.total_invalid_observations ?? 0}</span>
            <span className="text-xs text-red-500">Quality / Bounds</span>
          </div>
        </Card>
      </div>

      {/* 3. Horizon & Rolling Window Selectors */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-white p-3 border border-slate-200 shadow-xs">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-[#486581] mr-1">Forecast Horizon:</span>
          {[
            { h: 1, label: '+1h (Residual WaveLSTM)' },
            { h: 6, label: '+6h (Residual WaveANN)' },
            { h: 24, label: '+24h (Residual WaveANN)' },
          ].map((item) => (
            <button
              key={item.h}
              onClick={() => setSelectedHorizon(item.h as 1 | 6 | 24)}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                selectedHorizon === item.h
                  ? 'bg-[#102A43] text-white shadow-xs'
                  : 'bg-slate-100 text-[#486581] hover:bg-slate-200'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-xs font-bold uppercase tracking-wider text-[#486581] mr-1">Window:</span>
          {(['last_24h', 'last_7d', 'last_30d', 'all_time'] as VerificationWindow[]).map((win) => {
            const labels: Record<string, string> = {
              last_24h: '24h',
              last_7d: '7 Days',
              last_30d: '30 Days',
              all_time: 'All Verified',
            }
            return (
              <button
                key={win}
                onClick={() => setSelectedWindow(win)}
                className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                  selectedWindow === win
                    ? 'bg-[#246B9E] text-white font-semibold'
                    : 'bg-slate-50 text-[#486581] hover:bg-slate-100 border border-slate-200'
                }`}
              >
                {labels[win]}
              </button>
            )
          })}
        </div>
      </div>

      {/* 4. Operational Performance Metrics vs Persistence */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        {/* Metric 1: RMSE */}
        <Card className="p-4 border-slate-200 bg-white shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs font-medium text-[#486581]">
              <span>OPERATIONAL RMSE</span>
              <span className="rounded-sm bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-slate-700">
                N = {currentMetrics?.sample_count ?? 0}
              </span>
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-bold text-[#102A43]">
                {currentMetrics?.rmse !== null && currentMetrics?.rmse !== undefined
                  ? `${currentMetrics.rmse.toFixed(3)} m`
                  : '—'}
              </span>
              {currentMetrics?.persistence_rmse !== null && currentMetrics?.persistence_rmse !== undefined && (
                <span className="text-xs text-[#486581]">
                  (Persist: {currentMetrics.persistence_rmse.toFixed(3)}m)
                </span>
              )}
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between">
            <span className="text-xs text-[#486581]">Skill vs Persistence:</span>
            {currentMetrics?.rmse_skill_pct !== null && currentMetrics?.rmse_skill_pct !== undefined ? (
              <span
                className={`inline-flex items-center gap-0.5 text-xs font-bold ${
                  currentMetrics.rmse_skill_pct > 0 ? 'text-emerald-700' : 'text-red-600'
                }`}
              >
                {currentMetrics.rmse_skill_pct > 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                {currentMetrics.rmse_skill_pct > 0 ? '+' : ''}
                {currentMetrics.rmse_skill_pct.toFixed(1)}%
              </span>
            ) : (
              <span className="text-xs text-slate-400">Insufficient Data</span>
            )}
          </div>
        </Card>

        {/* Metric 2: MAE */}
        <Card className="p-4 border-slate-200 bg-white shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs font-medium text-[#486581]">
              <span>OPERATIONAL MAE</span>
              <span className="rounded-sm bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-slate-700">
                N = {currentMetrics?.sample_count ?? 0}
              </span>
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-bold text-[#102A43]">
                {currentMetrics?.mae !== null && currentMetrics?.mae !== undefined
                  ? `${currentMetrics.mae.toFixed(3)} m`
                  : '—'}
              </span>
              {currentMetrics?.persistence_mae !== null && currentMetrics?.persistence_mae !== undefined && (
                <span className="text-xs text-[#486581]">
                  (Persist: {currentMetrics.persistence_mae.toFixed(3)}m)
                </span>
              )}
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between">
            <span className="text-xs text-[#486581]">MAE Skill:</span>
            {currentMetrics?.mae_skill_pct !== null && currentMetrics?.mae_skill_pct !== undefined ? (
              <span
                className={`inline-flex items-center gap-0.5 text-xs font-bold ${
                  currentMetrics.mae_skill_pct > 0 ? 'text-emerald-700' : 'text-red-600'
                }`}
              >
                {currentMetrics.mae_skill_pct > 0 ? '+' : ''}
                {currentMetrics.mae_skill_pct.toFixed(1)}%
              </span>
            ) : (
              <span className="text-xs text-slate-400">Insufficient Data</span>
            )}
          </div>
        </Card>

        {/* Metric 3: Mean Bias Error */}
        <Card className="p-4 border-slate-200 bg-white shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs font-medium text-[#486581]">
              <span>MEAN BIAS ERROR (MBE)</span>
              <span className="text-[10px] font-mono text-slate-400">Pred - Obs</span>
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-bold text-[#102A43]">
                {currentMetrics?.bias !== null && currentMetrics?.bias !== undefined
                  ? `${currentMetrics.bias > 0 ? '+' : ''}${currentMetrics.bias.toFixed(3)} m`
                  : '—'}
              </span>
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 text-xs text-[#486581]">
            {currentMetrics?.bias !== null && currentMetrics?.bias !== undefined ? (
              currentMetrics.bias > 0.02 ? (
                <span className="text-amber-700 font-medium">Mild positive overprediction bias</span>
              ) : currentMetrics.bias < -0.02 ? (
                <span className="text-amber-700 font-medium">Mild negative underprediction bias</span>
              ) : (
                <span className="text-emerald-700 font-medium">Zero-mean calibrated error</span>
              )
            ) : (
              <span className="text-slate-400">Awaiting matched data</span>
            )}
          </div>
        </Card>

        {/* Metric 4: R² & Sample Reliability */}
        <Card className="p-4 border-slate-200 bg-white shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs font-medium text-[#486581]">
              <span>COEFFICIENT OF DET. (R²)</span>
              <span className="text-[10px] font-mono text-slate-400">Variance Expl.</span>
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-bold text-[#102A43]">
                {currentMetrics?.r2 !== null && currentMetrics?.r2 !== undefined
                  ? currentMetrics.r2.toFixed(4)
                  : '—'}
              </span>
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100">
            {currentMetrics?.is_statistically_reliable ? (
              <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700">
                <CheckCircle2 className="h-3 w-3" /> Statistically Verified (N={currentMetrics.sample_count})
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-700">
                <AlertCircle className="h-3 w-3" /> Insufficient Samples (Min {currentMetrics?.min_samples_required ?? 5})
              </span>
            )}
          </div>
        </Card>
      </div>

      {/* 5. Historical Benchmarks vs Operational Reality Comparison */}
      <Card className="p-5 border-slate-200 bg-gradient-to-r from-slate-50 to-blue-50/20 shadow-xs">
        <div className="flex items-center gap-2 mb-3">
          <Info className="h-4 w-4 text-[#246B9E]" />
          <h3 className="text-sm font-bold uppercase tracking-wider text-[#102A43]">
            Scientific Audit: Historical Benchmark vs Live Operational Accuracy
          </h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="rounded-lg bg-white p-3.5 border border-slate-200">
            <span className="font-bold text-[#102A43] block text-sm mb-1">+1h Horizon</span>
            <div className="space-y-1 text-[#486581]">
              <div className="flex justify-between">
                <span>Historical Held-Out Skill:</span>
                <span className="font-bold text-emerald-700">+29.76% (19.4k anchors)</span>
              </div>
              <div className="flex justify-between">
                <span>Operational Live Skill:</span>
                <span className="font-bold text-[#246B9E]">
                  {dashboard?.horizon_metrics?.['1h']?.all_time?.rmse_skill_pct !== null &&
                  dashboard?.horizon_metrics?.['1h']?.all_time?.rmse_skill_pct !== undefined
                    ? `+${dashboard.horizon_metrics['1h'].all_time.rmse_skill_pct.toFixed(1)}%`
                    : 'Accumulating'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">Recurrent state tracking maintains sub-centimeter fidelity.</p>
            </div>
          </div>

          <div className="rounded-lg bg-white p-3.5 border border-slate-200">
            <span className="font-bold text-[#102A43] block text-sm mb-1">+6h Horizon</span>
            <div className="space-y-1 text-[#486581]">
              <div className="flex justify-between">
                <span>Historical Held-Out Skill:</span>
                <span className="font-bold text-teal-700">+18.85% (19.4k anchors)</span>
              </div>
              <div className="flex justify-between">
                <span>Operational Live Skill:</span>
                <span className="font-bold text-[#246B9E]">
                  {dashboard?.horizon_metrics?.['6h']?.all_time?.rmse_skill_pct !== null &&
                  dashboard?.horizon_metrics?.['6h']?.all_time?.rmse_skill_pct !== undefined
                    ? `+${dashboard.horizon_metrics['6h'].all_time.rmse_skill_pct.toFixed(1)}%`
                    : 'Accumulating'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">Optimal lead time for coastal port operational planning.</p>
            </div>
          </div>

          <div className="rounded-lg bg-white p-3.5 border border-slate-200">
            <span className="font-bold text-[#102A43] block text-sm mb-1">+24h Horizon</span>
            <div className="space-y-1 text-[#486581]">
              <div className="flex justify-between">
                <span>Historical Held-Out Skill:</span>
                <span className="font-bold text-[#246B9E]">+11.09% (19.4k anchors)</span>
              </div>
              <div className="flex justify-between">
                <span>Operational Live Skill:</span>
                <span className="font-bold text-[#246B9E]">
                  {dashboard?.horizon_metrics?.['24h']?.all_time?.rmse_skill_pct !== null &&
                  dashboard?.horizon_metrics?.['24h']?.all_time?.rmse_skill_pct !== undefined
                    ? `+${dashboard.horizon_metrics['24h'].all_time.rmse_skill_pct.toFixed(1)}%`
                    : 'Accumulating'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">Multi-step diurnal cycle predictions exhibit higher variance.</p>
            </div>
          </div>
        </div>
      </Card>

      {/* 6. Charts: Timeline Forecast vs Observed & Error Trend */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Chart 1: Wave Height Timeline */}
        <Card className="p-5 border-slate-200 bg-white shadow-xs">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-[#102A43]">
                Forecast vs. Observed Significant Wave Height (+{selectedHorizon}h)
              </h3>
              <p className="text-xs text-[#486581]">Comparison of issued predictions, observed in-situ buoy telemetry, and persistence.</p>
            </div>
            <span className="text-xs font-semibold text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
              SWH [m]
            </span>
          </div>

          <div className="h-64 w-full">
            {timelineChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timelineChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#486581' }} stroke="#CBD5E1" />
                  <YAxis tick={{ fontSize: 10, fill: '#486581' }} stroke="#CBD5E1" domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#102A43', color: '#FFF', borderRadius: '8px', fontSize: '11px', border: 'none' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                  <Line
                    type="monotone"
                    dataKey="observed"
                    name="Observed Buoy SWH"
                    stroke="#10B981"
                    strokeWidth={2.2}
                    dot={{ r: 3, fill: '#10B981' }}
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    name={`Predicted SWH (+${selectedHorizon}h)`}
                    stroke="#246B9E"
                    strokeWidth={2}
                    dot={{ r: 2.5, fill: '#246B9E' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="persistence"
                    name="Persistence Baseline"
                    stroke="#DC2626"
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-xs text-slate-400">
                No recent prospective forecasts generated for +{selectedHorizon}h horizon.
              </div>
            )}
          </div>
        </Card>

        {/* Chart 2: Forecast Error Trend */}
        <Card className="p-5 border-slate-200 bg-white shadow-xs">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-[#102A43]">
                Operational Forecast Error Trend (Predicted - Observed)
              </h3>
              <p className="text-xs text-[#486581]">Visual verification of residual zero-centering and temporal deviation.</p>
            </div>
            <span className="text-xs font-semibold text-slate-600 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              ΔSWH [m]
            </span>
          </div>

          <div className="h-64 w-full">
            {timelineChartData.filter((d) => d.error !== null).length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={timelineChartData.filter((d) => d.error !== null)}
                  margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#486581' }} stroke="#CBD5E1" />
                  <YAxis tick={{ fontSize: 10, fill: '#486581' }} stroke="#CBD5E1" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#102A43', color: '#FFF', borderRadius: '8px', fontSize: '11px', border: 'none' }}
                  />
                  <ReferenceLine y={0} stroke="#486581" strokeWidth={1.5} />
                  <Bar dataKey="error" name="Forecast Error (m)" fill="#20AFA7" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-xs text-slate-400">
                Awaiting matched observations to compute prospective error trend.
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* 7. Extreme-Event Monitoring (>= 4.0m) & Historical Sample Notice */}
      <Card className="p-5 border-amber-200 bg-amber-50/20 shadow-xs">
        <div className="flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="space-y-3 w-full">
            <div>
              <h3 className="text-sm font-bold text-[#102A43]">
                Severe Sea-State Verification & Peak Underprediction Monitoring (SWH ≥ 4.0m)
              </h3>
              <p className="text-xs text-[#486581] mt-0.5 leading-relaxed">
                {dashboard?.severe_events?.scientific_notice}
              </p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-2 border-t border-amber-200/50">
              <div className="rounded bg-white p-2.5 border border-amber-100">
                <span className="text-[10px] uppercase font-semibold text-[#486581] block">Actual Severe</span>
                <span className="text-lg font-bold text-[#102A43]">
                  {dashboard?.severe_events?.actual_severe_events_count ?? 0}
                </span>
                <span className="text-[10px] text-slate-400 block">Obs ≥ 4.0m</span>
              </div>

              <div className="rounded bg-white p-2.5 border border-amber-100">
                <span className="text-[10px] uppercase font-semibold text-[#486581] block">Forecast Severe</span>
                <span className="text-lg font-bold text-[#102A43]">
                  {dashboard?.severe_events?.forecasted_severe_events_count ?? 0}
                </span>
                <span className="text-[10px] text-slate-400 block">Pred ≥ 4.0m</span>
              </div>

              <div className="rounded bg-white p-2.5 border border-amber-100">
                <span className="text-[10px] uppercase font-semibold text-[#486581] block">Missed Events</span>
                <span className="text-lg font-bold text-red-600">
                  {dashboard?.severe_events?.missed_severe_events_count ?? 0}
                </span>
                <span className="text-[10px] text-red-400 block">Obs ≥ 4.0 & Pred &lt; 4.0</span>
              </div>

              <div className="rounded bg-white p-2.5 border border-amber-100">
                <span className="text-[10px] uppercase font-semibold text-[#486581] block">False Alarms</span>
                <span className="text-lg font-bold text-amber-700">
                  {dashboard?.severe_events?.false_alarms_count ?? 0}
                </span>
                <span className="text-[10px] text-amber-500 block">Pred ≥ 4.0 & Obs &lt; 4.0</span>
              </div>

              <div className="rounded bg-white p-2.5 border border-amber-100">
                <span className="text-[10px] uppercase font-semibold text-[#486581] block">Severe Sample Status</span>
                <span className="text-xs font-bold text-slate-700 block mt-1">
                  {dashboard?.severe_events?.operational_severe_status === 'INSUFFICIENT_OPERATIONAL_SAMPLE'
                    ? '🟡 Low Operational N'
                    : '🟢 Evidenced'}
                </span>
                <span className="text-[10px] text-slate-400 block">Historical N=22</span>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* 8. Verification History Table */}
      <Card className="p-5 border-slate-200 bg-white shadow-xs">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-4">
          <div>
            <h3 className="text-sm font-bold text-[#102A43]">Prospective Forecast Verification History</h3>
            <p className="text-xs text-[#486581]">
              Audit ledger of issued predictions and paired physical in-situ measurements.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-[#486581]">Filter Status:</span>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs text-[#102A43] focus:border-[#246B9E] focus:outline-none"
            >
              <option value="ALL">All Statuses</option>
              <option value="VERIFIED">Verified</option>
              <option value="PENDING_VERIFICATION">Pending Target</option>
              <option value="OBSERVATION_UNAVAILABLE">Observation Unavailable</option>
              <option value="INVALID_OBSERVATION">Invalid Observation</option>
            </select>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/75 text-[#486581]">
                <th className="py-2.5 px-3 font-semibold">Forecast ID</th>
                <th className="py-2.5 px-3 font-semibold">Origin Time</th>
                <th className="py-2.5 px-3 font-semibold">Target Time</th>
                <th className="py-2.5 px-3 font-semibold">Lead Time</th>
                <th className="py-2.5 px-3 font-semibold">Model Architecture</th>
                <th className="py-2.5 px-3 font-semibold">Persist SWH</th>
                <th className="py-2.5 px-3 font-semibold">Predicted</th>
                <th className="py-2.5 px-3 font-semibold">Observed</th>
                <th className="py-2.5 px-3 font-semibold">Error (Δ)</th>
                <th className="py-2.5 px-3 font-semibold">Status</th>
                <th className="py-2.5 px-3 font-semibold">Prospective</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredRecords.length > 0 ? (
                filteredRecords.map((r) => {
                  const errFormatted =
                    r.forecast_error !== null && r.forecast_error !== undefined
                      ? `${r.forecast_error > 0 ? '+' : ''}${r.forecast_error.toFixed(3)} m`
                      : '—'
                  return (
                    <tr key={r.forecast_id} className="hover:bg-slate-50/50 transition-colors">
                      <td className="py-2.5 px-3 font-mono font-medium text-slate-700">{r.forecast_id}</td>
                      <td className="py-2.5 px-3 text-slate-500">
                        {new Date(r.forecast_origin_time).toLocaleString([], {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </td>
                      <td className="py-2.5 px-3 text-slate-700 font-medium">
                        {new Date(r.target_timestamp).toLocaleString([], {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </td>
                      <td className="py-2.5 px-3 font-semibold text-[#246B9E]">+{r.forecast_horizon}h</td>
                      <td className="py-2.5 px-3 text-slate-600">{r.model_architecture}</td>
                      <td className="py-2.5 px-3 text-slate-500">{r.current_swh.toFixed(2)} m</td>
                      <td className="py-2.5 px-3 font-bold text-[#102A43]">{r.predicted_swh.toFixed(2)} m</td>
                      <td className="py-2.5 px-3 font-bold text-emerald-700">
                        {r.observed_swh !== null && r.observed_swh !== undefined
                          ? `${r.observed_swh.toFixed(2)} m`
                          : '—'}
                      </td>
                      <td
                        className={`py-2.5 px-3 font-mono font-medium ${
                          r.forecast_error && r.forecast_error > 0
                            ? 'text-amber-700'
                            : r.forecast_error && r.forecast_error < 0
                            ? 'text-blue-700'
                            : 'text-slate-500'
                        }`}
                      >
                        {errFormatted}
                      </td>
                      <td className="py-2.5 px-3">{getStatusBadge(r.status)}</td>
                      <td className="py-2.5 px-3">
                        {r.is_prospective ? (
                          <span className="text-[10px] font-semibold text-emerald-600">✓ Prospective</span>
                        ) : (
                          <span className="text-[10px] text-slate-400">Backtest</span>
                        )}
                      </td>
                    </tr>
                  )
                })
              ) : (
                <tr>
                  <td colSpan={11} className="py-8 text-center text-xs text-slate-400">
                    No forecast verification records matching the selected horizon and status filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

import React, { useState, useEffect } from 'react'
import {
  TrendingUp,
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
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  CartesianGrid,
} from 'recharts'
import { api } from '../services/api'
import type {
  AccuracyMetricsResponse,
  DataDriftResponse,
  ForecastLedgerItem,
  MonitoringEventItem,
  MonitoringSummaryResponse,
} from '../types'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { StatusIndicator } from '../components/ui/StatusIndicator'

export function ForecastAccuracyDriftPage() {
  const [activeTab, setActiveTab] = useState<'accuracy' | 'drift' | 'degradation' | 'events'>('accuracy')
  const [locFilter, setLocFilter] = useState<string>('mumbai_offshore')
  const [horizonFilter, setHorizonFilter] = useState<number>(0)
  const [modelFilter, setModelFilter] = useState<string>('ALL')
  const [driftSource, setDriftSource] = useState<'demo' | 'live'>('demo')

  const [summary, setSummary] = useState<MonitoringSummaryResponse | null>(null)
  const [accuracy, setAccuracy] = useState<AccuracyMetricsResponse | null>(null)
  const [drift, setDrift] = useState<DataDriftResponse | null>(null)
  const [events, setEvents] = useState<MonitoringEventItem[]>([])
  const [ledger, setLedger] = useState<ForecastLedgerItem[]>([])

  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [isBacktesting, setIsBacktesting] = useState<boolean>(false)
  const [backtestMessage, setBacktestMessage] = useState<string | null>(null)

  const fetchData = async () => {
    try {
      setLoading(true)
      setError(null)

      const [sumData, accData, driftData, eventsData, ledgerData] = await Promise.all([
        api.getMonitoringSummary(),
        api.getForecastAccuracy(locFilter, horizonFilter, modelFilter),
        api.getDataDriftReport(locFilter, driftSource),
        api.listMonitoringEvents(locFilter, 'ALL'),
        api.getForecastLedger(locFilter, 30),
      ])

      setSummary(sumData)
      setAccuracy(accData)
      setDrift(driftData)
      setEvents(eventsData)
      setLedger(ledgerData)
    } catch (err: unknown) {
      console.error('Failed to load monitoring data:', err)
      setError(err instanceof Error ? err.message : 'Failed to query monitoring metrics.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [locFilter, horizonFilter, modelFilter, driftSource])

  const handleBacktest = async () => {
    try {
      setIsBacktesting(true)
      const res = await api.triggerBacktest(locFilter, 500)
      setBacktestMessage(res.message)
      await fetchData()
      setTimeout(() => setBacktestMessage(null), 5000)
    } catch (err: unknown) {
      console.error('Backtest trigger failed:', err)
      alert(err instanceof Error ? err.message : 'Backtest request failed.')
    } finally {
      setIsBacktesting(false)
    }
  }

  // Bar chart data for feature drift PSI
  const driftChartData = React.useMemo(() => {
    if (!drift?.feature_drifts) return []
    return Object.entries(drift.feature_drifts).map(([feature, item]) => ({
      feature,
      psi: Number(item.psi.toFixed(4)),
      level: item.drift_level,
    }))
  }, [drift])

  return (
    <div className="space-y-6">
      {/* Top Banner & Scientific Integrity Policy */}
      <div className="rounded-xl bg-gradient-to-r from-[#0F172A] via-[#0369A1] to-[#0284C7] p-5 border border-[#0284C7] shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#38BDF8]/20 text-[#38BDF8] border border-[#38BDF8]/40">
                <TrendingUp className="h-5 w-5" />
              </span>
              <h1 className="text-xl font-bold tracking-tight text-[#F1F5F9] font-['Outfit',sans-serif]">
                Forecast Accuracy & Model Drift Monitoring
              </h1>
            </div>
            <p className="mt-1 text-xs text-[#CBD5E1] max-w-3xl leading-relaxed">
              Prospective forecast ledger tracking, independent observation matching, canonical feature drift (PSI), and rolling degradation tracking against ERA5 benchmarks.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleBacktest}
              isLoading={isBacktesting}
              className="text-xs bg-[#0F172A] border-[#38BDF8] text-[#38BDF8] hover:bg-[#1E293B]"
            >
              <Database className="h-3.5 w-3.5 text-[#38BDF8]" />
              <span>🔄 Backtest ERA5</span>
            </Button>
          </div>
        </div>

        {/* Backtest Success Notification */}
        {backtestMessage && (
          <div className="mt-3 p-2.5 rounded-lg bg-emerald-950/80 border border-emerald-500/50 text-xs text-emerald-200">
            {backtestMessage}
          </div>
        )}

        {/* Scientific Observation Integrity Notice */}
        <div className="mt-4 rounded-lg bg-[#07111F]/80 p-3.5 border border-[#38BDF8]/40 text-xs text-[#CBD5E1] space-y-1">
          <div className="font-semibold text-[#38BDF8] flex items-center gap-1.5">
            <Info className="h-4 w-4" />
            <span>Scientific Verification & Observation Integrity Policy</span>
          </div>
          <div className="text-[11px] leading-relaxed text-[#94A3B8]">
            • <b>Observation Sourcing</b>: True operational verification requires independent, in-situ oceanographic measurements (moored buoys / coastal radar). External provider forecasts (e.g. Open-Meteo / ECMWF) are <i>forecast comparisons</i>, not verified accuracy.<br />
            • <b>Retrospective Reanalysis</b>: Comparing predictions against historical ERA5 is classified as <i>retrospective reanalysis backtesting</i>, not live buoy verification.<br />
            • <b>Separation of Alerts</b>: ML operations events (drift, degradation) are strictly separated and <b>never generate maritime safety warnings</b>.
          </div>
        </div>
      </div>

      {/* KPI Ribbon */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="bg-[#101D2D] p-3.5 rounded-xl border border-[#1E3A5F]">
          <div className="text-[11px] text-[#94A3B8] uppercase font-semibold">Forecasts Issued</div>
          <div className="text-2xl font-bold font-mono text-[#38BDF8] mt-1">
            {summary?.total_forecasts_issued ?? 0}
          </div>
          <div className="text-[10px] text-[#64748B] mt-0.5">Immutable Audit Ledger</div>
        </div>

        <div className="bg-[#101D2D] p-3.5 rounded-xl border border-[#1E3A5F]">
          <div className="text-[11px] text-[#94A3B8] uppercase font-semibold">Matched Pairs</div>
          <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
            {accuracy?.sample_count ?? 0}
          </div>
          <div className="text-[10px] text-[#64748B] mt-0.5">
            Total Obs: {summary?.total_observations_recorded ?? 0}
          </div>
        </div>

        <div className="bg-[#101D2D] p-3.5 rounded-xl border border-[#1E3A5F]">
          <div className="text-[11px] text-[#94A3B8] uppercase font-semibold">Verification Coverage</div>
          <div className="text-2xl font-bold font-mono text-[#F1F5F9] mt-1">
            {accuracy ? `${accuracy.coverage_pct.toFixed(1)}%` : '0.0%'}
          </div>
          <div className="text-[10px] text-[#38BDF8] mt-0.5 truncate">
            {accuracy?.verification_badge || 'Awaiting Observations'}
          </div>
        </div>

        <div className="bg-[#101D2D] p-3.5 rounded-xl border border-[#1E3A5F]">
          <div className="text-[11px] text-[#94A3B8] uppercase font-semibold">Verified MAE</div>
          <div className="text-2xl font-bold font-mono text-[#38BDF8] mt-1">
            {accuracy?.mae_meters !== null && accuracy?.mae_meters !== undefined
              ? `${accuracy.mae_meters.toFixed(3)} m`
              : 'N/A'}
          </div>
          <div className="text-[10px] text-[#64748B] mt-0.5">
            {accuracy?.rmse_meters !== null && accuracy?.rmse_meters !== undefined
              ? `RMSE: ${accuracy.rmse_meters.toFixed(3)} m`
              : 'Awaiting matched pairs'}
          </div>
        </div>

        <div className="bg-[#101D2D] p-3.5 rounded-xl border border-[#1E3A5F] col-span-2 md:col-span-1">
          <div className="text-[11px] text-[#94A3B8] uppercase font-semibold">Active ML Events</div>
          <div className={`text-2xl font-bold font-mono mt-1 ${summary?.active_monitoring_events_count ? 'text-amber-400' : 'text-emerald-400'}`}>
            {summary?.active_monitoring_events_count ?? 0}
          </div>
          <div className="text-[10px] text-[#64748B] mt-0.5">Drift & Degradation Logs</div>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 bg-[#101D2D] p-3.5 rounded-xl border border-[#1E3A5F]">
        <div>
          <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider block mb-1.5">
            Monitored Coastal Node
          </label>
          <select
            value={locFilter}
            onChange={(e) => setLocFilter(e.target.value)}
            className="w-full bg-[#07111F] border border-[#1E3A5F] rounded-lg px-3 py-2 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#35C6E8]"
          >
            <option value="ALL">All Coastal Nodes</option>
            <option value="mumbai_offshore">Mumbai High Offshore (Reference)</option>
            <option value="goa_mormugao">Mormugao / Goa Coastal Node</option>
            <option value="mangalore_port">Mangalore Coastal Node</option>
            <option value="kochi_coast">Kochi Coastal Node</option>
            <option value="veraval_gujarat">Veraval / Gujarat Coast</option>
            <option value="ratnagiri_coast">Ratnagiri Coastal Node</option>
            <option value="kanyakumari_point">Kanyakumari Point</option>
            <option value="lakshadweep_kavaratti">Lakshadweep / Kavaratti</option>
          </select>
        </div>

        <div>
          <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider block mb-1.5">
            Forecast Horizon
          </label>
          <select
            value={horizonFilter}
            onChange={(e) => setHorizonFilter(Number(e.target.value))}
            className="w-full bg-[#07111F] border border-[#1E3A5F] rounded-lg px-3 py-2 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#35C6E8]"
          >
            <option value={0}>All Horizons</option>
            <option value={1}>+1h Ahead</option>
            <option value={6}>+6h Ahead</option>
            <option value={24}>+24h Ahead</option>
          </select>
        </div>

        <div>
          <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider block mb-1.5">
            Forecasting Architecture
          </label>
          <select
            value={modelFilter}
            onChange={(e) => setModelFilter(e.target.value)}
            className="w-full bg-[#07111F] border border-[#1E3A5F] rounded-lg px-3 py-2 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#35C6E8]"
          >
            <option value="ALL">All Models</option>
            <option value="WaveLSTM">WaveLSTM</option>
            <option value="WaveANN">WaveANN</option>
            <option value="WaveCNN1D">WaveCNN1D</option>
          </select>
        </div>

        <div>
          <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider block mb-1.5">
            Drift Stream Evaluated
          </label>
          <select
            value={driftSource}
            onChange={(e) => setDriftSource(e.target.value as 'demo' | 'live')}
            className="w-full bg-[#07111F] border border-[#1E3A5F] rounded-lg px-3 py-2 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#35C6E8]"
          >
            <option value="demo">📂 Offline Historical (ERA5)</option>
            <option value="live">🌐 Live Open-Meteo REST API</option>
          </select>
        </div>
      </div>

      {/* Tabs Navigation */}
      <div className="flex border-b border-[#1E3A5F] gap-2">
        {[
          { key: 'accuracy', label: '🎯 Verified Accuracy & Persistence' },
          { key: 'drift', label: '🌊 Canonical Feature Drift (PSI)' },
          { key: 'degradation', label: '📉 Performance Degradation' },
          { key: 'events', label: '📋 ML Ops Event Log' },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as any)}
            className={`py-2.5 px-4 text-xs font-semibold border-b-2 transition-all cursor-pointer ${
              activeTab === tab.key
                ? 'border-[#35C6E8] text-[#35C6E8] bg-[#101D2D]/60'
                : 'border-transparent text-[#94A3B8] hover:text-[#F1F5F9]'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab 1: Verified Accuracy */}
      {activeTab === 'accuracy' && (
        <div className="space-y-4">
          <div className="rounded-xl bg-[#101D2D] p-5 border border-[#1E3A5F] space-y-4">
            <div className="flex items-center justify-between border-b border-[#1E3A5F] pb-3">
              <div>
                <h3 className="text-sm font-bold text-[#F1F5F9]">
                  Verified Forecast Accuracy vs. Naive Persistence Baseline
                </h3>
                <p className="text-xs text-[#94A3B8] mt-0.5">
                  Evaluated strictly on identical matched observation cases. Zero metric extrapolation.
                </p>
              </div>
              <span className="text-xs px-2.5 py-1 rounded bg-[#07111F] border border-[#1E3A5F] text-[#38BDF8] font-semibold">
                {accuracy?.verification_badge || 'Awaiting Observations'}
              </span>
            </div>

            {accuracy && accuracy.sample_count > 0 ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                  <div className="bg-[#07111F] p-3 rounded-lg border border-[#1E3A5F]">
                    <div className="text-[10px] text-[#94A3B8] uppercase">Mean Absolute Error (MAE)</div>
                    <div className="text-xl font-bold font-mono text-[#38BDF8] mt-0.5">
                      {accuracy.mae_meters?.toFixed(4)} m
                    </div>
                  </div>
                  <div className="bg-[#07111F] p-3 rounded-lg border border-[#1E3A5F]">
                    <div className="text-[10px] text-[#94A3B8] uppercase">Root Mean Squared Error (RMSE)</div>
                    <div className="text-xl font-bold font-mono text-[#38BDF8] mt-0.5">
                      {accuracy.rmse_meters?.toFixed(4)} m
                    </div>
                  </div>
                  <div className="bg-[#07111F] p-3 rounded-lg border border-[#1E3A5F]">
                    <div className="text-[10px] text-[#94A3B8] uppercase">Mean Bias Error</div>
                    <div className="text-xl font-bold font-mono text-[#F1F5F9] mt-0.5">
                      {accuracy.bias_meters ? (accuracy.bias_meters > 0 ? `+${accuracy.bias_meters.toFixed(4)}` : accuracy.bias_meters.toFixed(4)) : '0.0000'} m
                    </div>
                  </div>
                  <div className="bg-[#07111F] p-3 rounded-lg border border-[#1E3A5F]">
                    <div className="text-[10px] text-[#94A3B8] uppercase">Skill vs Persistence</div>
                    <div className={`text-xl font-bold font-mono mt-0.5 ${(accuracy.skill_score_rmse ?? 0) > 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {accuracy.skill_score_rmse ? `${(accuracy.skill_score_rmse * 100).toFixed(1)}%` : 'N/A'}
                    </div>
                  </div>
                </div>

                {/* Persistence Benchmark Comparison Table */}
                <div className="rounded-lg border border-[#1E3A5F] overflow-hidden">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-[#07111F] text-[#94A3B8] border-b border-[#1E3A5F]">
                      <tr>
                        <th className="py-2.5 px-3">Methodology</th>
                        <th className="py-2.5 px-3">MAE (meters)</th>
                        <th className="py-2.5 px-3">RMSE (meters)</th>
                        <th className="py-2.5 px-3">Matched Sample Size</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1E3A5F] bg-[#101D2D]">
                      <tr>
                        <td className="py-2.5 px-3 font-semibold text-[#F1F5F9]">Deep Learning Model Forecast</td>
                        <td className="py-2.5 px-3 font-mono text-[#38BDF8] font-bold">{accuracy.mae_meters?.toFixed(4)} m</td>
                        <td className="py-2.5 px-3 font-mono text-[#38BDF8] font-bold">{accuracy.rmse_meters?.toFixed(4)} m</td>
                        <td className="py-2.5 px-3 font-mono text-[#CBD5E1]">{accuracy.sample_count}</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 px-3 font-semibold text-[#94A3B8]">Naive Persistence Baseline (t)</td>
                        <td className="py-2.5 px-3 font-mono text-[#94A3B8]">{accuracy.persistence_mae_meters ? `${accuracy.persistence_mae_meters.toFixed(4)} m` : 'N/A'}</td>
                        <td className="py-2.5 px-3 font-mono text-[#94A3B8]">{accuracy.persistence_rmse_meters ? `${accuracy.persistence_rmse_meters.toFixed(4)} m` : 'N/A'}</td>
                        <td className="py-2.5 px-3 font-mono text-[#CBD5E1]">{accuracy.sample_count}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="rounded-lg bg-[#07111F] p-8 text-center border border-dashed border-[#1E3A5F] text-xs text-[#64748B] space-y-2">
                <Clock className="h-6 w-6 mx-auto text-[#38BDF8]" />
                <h4 className="font-semibold text-[#CBD5E1]">No Matched Forecast-Observation Pairs Yet</h4>
                <p className="max-w-md mx-auto">
                  Click <b>'🔄 Backtest ERA5'</b> above to populate retrospective ground-truth observations, or wait for prospective observations to arrive.
                </p>
              </div>
            )}
          </div>

          {/* Immutable Forecast Ledger Audit Feed */}
          <div className="rounded-xl bg-[#101D2D] p-5 border border-[#1E3A5F] space-y-3">
            <h3 className="text-sm font-bold text-[#F1F5F9] flex items-center justify-between">
              <span>Immutable Forecast Issuance Ledger (Recent Records)</span>
              <span className="text-xs text-[#94A3B8] font-normal">Deduplicated Prospective Audit Trail</span>
            </h3>

            <div className="rounded-lg border border-[#1E3A5F] overflow-x-auto">
              <table className="w-full text-xs text-left whitespace-nowrap">
                <thead className="bg-[#07111F] text-[#94A3B8] border-b border-[#1E3A5F]">
                  <tr>
                    <th className="py-2 px-3">Forecast ID</th>
                    <th className="py-2 px-3">Location</th>
                    <th className="py-2 px-3">Model</th>
                    <th className="py-2 px-3">Horizon</th>
                    <th className="py-2 px-3">Predicted (Hs)</th>
                    <th className="py-2 px-3">Persistence</th>
                    <th className="py-2 px-3">Target Time (UTC)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1E3A5F] bg-[#101D2D]">
                  {ledger.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="py-4 text-center text-[#64748B]">
                        No issued forecasts recorded in ledger yet.
                      </td>
                    </tr>
                  ) : (
                    ledger.map((rec) => (
                      <tr key={rec.forecast_id} className="hover:bg-[#16263A]">
                        <td className="py-2 px-3 font-mono text-[#38BDF8]">{rec.forecast_id.slice(0, 8)}...</td>
                        <td className="py-2 px-3 font-semibold text-[#F1F5F9]">{rec.location_id}</td>
                        <td className="py-2 px-3">{rec.model_name}</td>
                        <td className="py-2 px-3 font-bold text-[#35C6E8]">+{rec.horizon_hours}h</td>
                        <td className="py-2 px-3 font-mono font-bold text-[#F1F5F9]">{rec.predicted_swh_m.toFixed(2)} m</td>
                        <td className="py-2 px-3 font-mono text-[#94A3B8]">{rec.persistence_swh_m !== null ? `${rec.persistence_swh_m?.toFixed(2)} m` : 'N/A'}</td>
                        <td className="py-2 px-3 text-[#CBD5E1]">{new Date(rec.target_time_utc).toUTCString().slice(5, 22)}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Feature Distribution Drift (PSI) */}
      {activeTab === 'drift' && (
        <div className="space-y-4">
          <div className="rounded-xl bg-[#101D2D] p-5 border border-[#1E3A5F] space-y-4">
            <div className="flex items-center justify-between border-b border-[#1E3A5F] pb-3">
              <div>
                <h3 className="text-sm font-bold text-[#F1F5F9]">
                  Canonical Input Feature Distribution Drift (Population Stability Index)
                </h3>
                <p className="text-xs text-[#94A3B8] mt-0.5">
                  Benchmarks incoming 24h meteorological telemetry against the 14.8-year ERA5 training split reference distribution (70%).
                </p>
              </div>
              <span
                className={`text-xs px-2.5 py-1 rounded font-bold border ${
                  drift?.status === 'NORMAL'
                    ? 'bg-emerald-950/60 text-emerald-400 border-emerald-500/40'
                    : drift?.status === 'WATCH'
                    ? 'bg-amber-950/60 text-amber-300 border-amber-500/40'
                    : 'bg-red-950/60 text-red-400 border-red-500/40'
                }`}
              >
                Status: {drift?.status || 'NORMAL'}
              </span>
            </div>

            {/* Recharts Bar Chart of PSI */}
            <div className="h-64 w-full bg-[#07111F] p-4 rounded-xl border border-[#1E3A5F]">
              <div className="text-xs font-semibold text-[#94A3B8] mb-2 flex items-center justify-between">
                <span>PSI Score per Canonical Feature (Threshold: 0.10 Watch / 0.25 Drift)</span>
                <span className="text-[10px] text-[#64748B]">Reference: ERA5 Training Set</span>
              </div>
              <ResponsiveContainer width="100%" height="90%">
                <BarChart data={driftChartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E3A5F" />
                  <XAxis dataKey="feature" stroke="#94A3B8" fontSize={10} interval={0} angle={-30} textAnchor="end" />
                  <YAxis stroke="#94A3B8" fontSize={10} domain={[0, 0.35]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#07111F', borderColor: '#1E3A5F', borderRadius: '8px', fontSize: '11px' }}
                    labelStyle={{ color: '#F1F5F9', fontWeight: 'bold' }}
                  />
                  <Bar dataKey="psi" radius={[4, 4, 0, 0]}>
                    {driftChartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.psi >= 0.25 ? '#EF4444' : entry.psi >= 0.1 ? '#F59E0B' : '#10B981'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Feature Drift Detailed Matrix Table */}
            <div className="rounded-lg border border-[#1E3A5F] overflow-x-auto">
              <table className="w-full text-xs text-left whitespace-nowrap">
                <thead className="bg-[#07111F] text-[#94A3B8] border-b border-[#1E3A5F]">
                  <tr>
                    <th className="py-2 px-3">Feature Name</th>
                    <th className="py-2 px-3">PSI Score</th>
                    <th className="py-2 px-3">Drift Classification</th>
                    <th className="py-2 px-3">Incoming Mean</th>
                    <th className="py-2 px-3">Reference Mean</th>
                    <th className="py-2 px-3">Missing Rate</th>
                    <th className="py-2 px-3">Out of Bounds</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1E3A5F] bg-[#101D2D]">
                  {drift?.feature_drifts ? (
                    Object.entries(drift.feature_drifts).map(([fname, fitem]) => {
                      const color =
                        fitem.psi >= 0.25 ? '#EF4444' : fitem.psi >= 0.1 ? '#F59E0B' : '#10B981'
                      return (
                        <tr key={fname} className="hover:bg-[#16263A]">
                          <td className="py-2 px-3 font-semibold text-[#F1F5F9] font-mono">{fname}</td>
                          <td className="py-2 px-3 font-bold font-mono" style={{ color }}>
                            {fitem.psi.toFixed(4)}
                          </td>
                          <td className="py-2 px-3">
                            <span
                              className="px-2 py-0.5 rounded text-[10px] font-bold border"
                              style={{ color, borderColor: `${color}44`, backgroundColor: `${color}15` }}
                            >
                              {fitem.drift_level}
                            </span>
                          </td>
                          <td className="py-2 px-3 font-mono">{fitem.curr_mean.toFixed(2)}</td>
                          <td className="py-2 px-3 font-mono text-[#94A3B8]">{fitem.ref_mean.toFixed(2)}</td>
                          <td className="py-2 px-3 font-mono">{(fitem.missing_rate * 100).toFixed(1)}%</td>
                          <td className="py-2 px-3 font-mono">{(fitem.out_of_bounds_rate * 100).toFixed(1)}%</td>
                        </tr>
                      )
                    })
                  ) : (
                    <tr>
                      <td colSpan={7} className="py-4 text-center text-[#64748B]">
                        Loading feature drift metrics...
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Standards Explanation */}
            <div className="rounded-lg bg-[#07111F]/60 p-3 border border-[#1E3A5F] text-[11px] text-[#94A3B8] space-y-0.5">
              <span className="font-semibold text-[#F1F5F9]">Population Stability Index (PSI) Threshold Standards:</span>
              <div>• <b>PSI &lt; 0.10 (🟢 NO_DRIFT)</b>: Incoming feature distributions conform to the training baseline.</div>
              <div>• <b>0.10 ≤ PSI &lt; 0.25 (🟡 MODERATE_DRIFT / WATCH)</b>: Moderate distribution shift; monitor incoming feed.</div>
              <div>• <b>PSI ≥ 0.25 (🔴 SIGNIFICANT_DRIFT)</b>: Significant distribution anomaly (e.g. seasonal monsoon peak or sensor calibration drift).</div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Model Performance Degradation */}
      {activeTab === 'degradation' && (
        <div className="space-y-4">
          <div className="rounded-xl bg-[#101D2D] p-5 border border-[#1E3A5F] space-y-4">
            <div className="border-b border-[#1E3A5F] pb-3">
              <h3 className="text-sm font-bold text-[#F1F5F9]">
                Model Error Degradation Tracking (Hysteresis & Benchmark Ratio)
              </h3>
              <p className="text-xs text-[#94A3B8] mt-0.5">
                Tracks rolling inference error against historical test set ground-truth benchmarks. Requires persistent deviation before triggering alerts.
              </p>
            </div>

            {/* Benchmarks Reference Table */}
            <div className="rounded-lg border border-[#1E3A5F] overflow-hidden">
              <div className="p-2.5 bg-[#07111F] text-xs font-semibold text-[#94A3B8] border-b border-[#1E3A5F]">
                🏛️ Historical Test Set Benchmarks (Reference Ground-Truth)
              </div>
              <table className="w-full text-xs text-left">
                <thead className="bg-[#07111F] text-[#64748B] border-b border-[#1E3A5F]">
                  <tr>
                    <th className="py-2 px-3">Model Architecture</th>
                    <th className="py-2 px-3">+1h Test MAE</th>
                    <th className="py-2 px-3">+6h Test MAE</th>
                    <th className="py-2 px-3">+24h Test MAE</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1E3A5F] bg-[#101D2D] text-[#CBD5E1]">
                  <tr>
                    <td className="py-2 px-3 font-semibold text-[#35C6E8]">WaveLSTM (Primary)</td>
                    <td className="py-2 px-3 font-mono">0.0182 m</td>
                    <td className="py-2 px-3 font-mono">0.0650 m</td>
                    <td className="py-2 px-3 font-mono">0.1420 m</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-[#F1F5F9]">WaveANN</td>
                    <td className="py-2 px-3 font-mono">0.0349 m</td>
                    <td className="py-2 px-3 font-mono">0.0980 m</td>
                    <td className="py-2 px-3 font-mono">0.1890 m</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-[#F1F5F9]">WaveCNN1D</td>
                    <td className="py-2 px-3 font-mono">0.0426 m</td>
                    <td className="py-2 px-3 font-mono">0.1120 m</td>
                    <td className="py-2 px-3 font-mono">0.2050 m</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Status Policy Callout */}
            <div className="p-3.5 rounded-lg bg-[#07111F] border border-[#1E3A5F] text-xs text-[#94A3B8] space-y-1">
              <span className="font-semibold text-[#F1F5F9]">Engineering Degradation Safeguards:</span>
              <div>• <b>NORMAL</b>: Rolling MAE ≤ 1.25x baseline test benchmark.</div>
              <div>• <b>WATCH</b>: 1.25x &lt; Rolling MAE ≤ 1.50x baseline benchmark.</div>
              <div>• <b>PERFORMANCE_DEGRADATION</b>: Sustained error &gt; 1.50x baseline across minimum 5 matched samples.</div>
              <div className="pt-1 text-[11px] text-[#64748B]">
                <b>Policy:</b> Model performance degradation logs are engineering diagnostic signals. They do NOT automatically delete, retrain, or substitute production checkpoints.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: Operational ML Events Log */}
      {activeTab === 'events' && (
        <div className="space-y-4">
          <div className="rounded-xl bg-[#101D2D] p-5 border border-[#1E3A5F] space-y-3">
            <div className="flex items-center justify-between border-b border-[#1E3A5F] pb-2">
              <div>
                <h3 className="text-sm font-bold text-[#F1F5F9]">
                  Operational ML Monitoring Event Log
                </h3>
                <p className="text-xs text-[#94A3B8] mt-0.5">
                  Persistent SQLite event history strictly isolated from coastal maritime wave hazard alerts.
                </p>
              </div>
              <span className="text-xs text-[#94A3B8] font-mono">
                {events.length} Recorded Events
              </span>
            </div>

            <div className="rounded-lg border border-[#1E3A5F] overflow-x-auto">
              <table className="w-full text-xs text-left whitespace-nowrap">
                <thead className="bg-[#07111F] text-[#94A3B8] border-b border-[#1E3A5F]">
                  <tr>
                    <th className="py-2 px-3">Event ID</th>
                    <th className="py-2 px-3">Type</th>
                    <th className="py-2 px-3">Location</th>
                    <th className="py-2 px-3">Severity</th>
                    <th className="py-2 px-3">Status</th>
                    <th className="py-2 px-3">Metric Value</th>
                    <th className="py-2 px-3">Threshold</th>
                    <th className="py-2 px-3">Summary</th>
                    <th className="py-2 px-3">Timestamp (UTC)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1E3A5F] bg-[#101D2D]">
                  {events.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="py-4 text-center text-[#64748B]">
                        No ML monitoring events logged. Telemetry distributions and model performance are nominal.
                      </td>
                    </tr>
                  ) : (
                    events.map((ev) => (
                      <tr key={ev.event_id} className="hover:bg-[#16263A]">
                        <td className="py-2 px-3 font-mono text-[#38BDF8]">{ev.event_id.slice(0, 8)}</td>
                        <td className="py-2 px-3 font-semibold text-[#F1F5F9]">{ev.event_type.toUpperCase()}</td>
                        <td className="py-2 px-3">{ev.location_id}</td>
                        <td className="py-2 px-3">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                              ev.severity === 'CRITICAL'
                                ? 'bg-red-500/20 text-red-400'
                                : ev.severity === 'WARNING'
                                ? 'bg-amber-500/20 text-amber-300'
                                : 'bg-blue-500/20 text-blue-300'
                            }`}
                          >
                            {ev.severity}
                          </span>
                        </td>
                        <td className="py-2 px-3 font-mono">{ev.status}</td>
                        <td className="py-2 px-3 font-mono font-bold text-[#F1F5F9]">{ev.metric_value.toFixed(3)}</td>
                        <td className="py-2 px-3 font-mono text-[#64748B]">{ev.threshold_value.toFixed(3)}</td>
                        <td className="py-2 px-3 text-[#CBD5E1] truncate max-w-xs">{ev.summary}</td>
                        <td className="py-2 px-3 text-[#64748B]">
                          {new Date(ev.created_at_utc).toUTCString().slice(5, 22)}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

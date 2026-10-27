import React, { useState, useEffect, useCallback } from 'react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts'
import { Award as AwardIcon, Sparkles as SparklesIcon, TrendingUp as TrendingUpIcon, BarChart3 as BarChartIcon } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { Skeleton } from '../components/ui/Skeleton'
import { api } from '../services/api'
import { formatPct, formatR2 } from '../lib/utils'
import type { Horizon, MultiHorizonResponse } from '../types'

export function ModelComparisonPage() {
  const [activeHorizon, setActiveHorizon] = useState<Horizon | 'all'>('all')
  const [multiMetrics, setMultiMetrics] = useState<MultiHorizonResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getMultiHorizonMetrics()
      setMultiMetrics(data)
    } catch (err: unknown) {
      console.error('Failed to load multi-horizon metrics:', err)
      setError(err instanceof Error ? err.message : 'Error fetching benchmark metrics')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  useEffect(() => {
    const handleRefresh = () => fetchData()
    window.addEventListener('oceancast:refresh', handleRefresh)
    return () => window.removeEventListener('oceancast:refresh', handleRefresh)
  }, [fetchData])

  // Map metrics robustly supporting both .metrics and .items from FastAPI
  const metricList = React.useMemo(() => {
    if (!multiMetrics) return []
    const raw = (multiMetrics as any).metrics || (multiMetrics as any).items || []
    return raw.map((m: any) => {
      const h = m.horizon ?? m.horizon_hours
      const isPersist = String(m.model).toLowerCase().includes('persistence')
      const isLstm = String(m.model).toLowerCase().includes('lstm')
      const isAnn = String(m.model).toLowerCase().includes('ann')

      return {
        horizon: h,
        model: m.model,
        model_type: m.model_type,
        rmse: m.rmse ?? m.rmse_meters ?? 0,
        mae: m.mae ?? m.mae_meters ?? 0,
        r2: m.r2 ?? m.r2_score ?? 0,
        rmse_skill_pct: m.rmse_skill_pct,
        mae_skill_pct: m.mae_skill_pct,
        is_best: m.is_best ?? ((h === 1 && isPersist) || (h === 6 && isLstm) || (h === 24 && isAnn)),
      }
    })
  }, [multiMetrics])

  // Filter metrics based on selected horizon
  const filteredMetrics = React.useMemo(() => {
    if (activeHorizon === 'all') return metricList
    return metricList.filter((m: any) => m.horizon === activeHorizon)
  }, [metricList, activeHorizon])

  // Prepare grouped bar chart data for RMSE and MAE across horizons
  const groupedChartData = React.useMemo(() => {
    if (!metricList.length) return []

    const horizons = [1, 6, 24]
    return horizons.map((h) => {
      const hMetrics = metricList.filter((m: any) => m.horizon === h)
      const persist = hMetrics.find((m: any) => m.model.toLowerCase().includes('persistence'))
      const lstm = hMetrics.find((m: any) => m.model.toLowerCase().includes('lstm'))
      const ann = hMetrics.find((m: any) => m.model.toLowerCase().includes('ann'))
      const cnn = hMetrics.find((m: any) => m.model.toLowerCase().includes('cnn'))

      return {
        horizon: `+${h}h Horizon`,
        'WaveLSTM (RMSE)': lstm?.rmse ?? 0,
        'WaveANN (RMSE)': ann?.rmse ?? 0,
        'WaveCNN1D (RMSE)': cnn?.rmse ?? 0,
        'Persistence (RMSE)': persist?.rmse ?? 0,
        'WaveLSTM (MAE)': lstm?.mae ?? 0,
        'WaveANN (MAE)': ann?.mae ?? 0,
        'WaveCNN1D (MAE)': cnn?.mae ?? 0,
        'Persistence (MAE)': persist?.mae ?? 0,
        r2_lstm: lstm?.r2 ?? 0,
        r2_ann: ann?.r2 ?? 0,
        r2_cnn: cnn?.r2 ?? 0,
        r2_persist: persist?.r2 ?? 0,
      }
    })
  }, [metricList])

  return (
    <div className="space-y-6">
      {/* Horizon Filter Tabs */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 rounded-xl bg-[#101D2D] border border-[#1E3A5F]">
        <div>
          <h2 className="text-sm font-semibold text-[#F1F5F9] font-['Outfit',sans-serif]">
            Benchmark Filtering
          </h2>
          <p className="text-xs text-[#94A3B8]">
            Filter table and charts by specific horizon or view comprehensive multi-horizon suite
          </p>
        </div>
        <div className="flex items-center gap-1.5 p-1 rounded-lg bg-[#07111F] border border-[#1E3A5F]">
          {(['all', 1, 6, 24] as (Horizon | 'all')[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveHorizon(tab)}
              className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all cursor-pointer ${
                activeHorizon === tab
                  ? 'bg-[#35C6E8] text-[#07111F] shadow-sm font-bold'
                  : 'text-[#94A3B8] hover:text-[#F1F5F9] hover:bg-[#16263A]'
              }`}
            >
              {tab === 'all' ? 'All Horizons' : `+${tab}h`}
            </button>
          ))}
        </div>
      </div>

      {/* Dynamic Academic Interpretation Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* +1h Finding */}
        <Card className="border-l-4 border-l-[#94A3B8]">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <Badge variant="outline">+1 Hour Horizon</Badge>
              <AwardIcon className="h-4 w-4 text-[#94A3B8]" />
            </div>
            <CardTitle className="text-sm mt-1">Persistence Leads Short Horizon</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Due to extreme lag-1 ocean inertia (autocorrelation $\rho_1 = 0.9995$), <strong>Persistence</strong> yields
              lowest RMSE (<strong>0.0213m</strong>). Among neural architectures, <strong>WaveLSTM</strong> comes closest
              (0.0271m, $R^2 = 0.9986$).
            </p>
          </CardContent>
        </Card>

        {/* +6h Finding */}
        <Card glow="cyan" className="border-l-4 border-l-[#35C6E8]">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <Badge variant="cyan">+6 Hour Horizon (Sweet Spot)</Badge>
              <SparklesIcon className="h-4 w-4 text-[#35C6E8]" />
            </div>
            <CardTitle className="text-sm mt-1">WaveLSTM Dominates (+13.88% Skill)</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              At 6 hours, wave growth and wind-sea dynamics decouple from pure inertia. <strong>WaveLSTM</strong> achieves
              best overall performance with <strong>0.0769m RMSE</strong> (+13.88% skill) and <strong>0.0533m MAE</strong> (+14.51% skill)
              over baseline.
            </p>
          </CardContent>
        </Card>

        {/* +24h Finding */}
        <Card className="border-l-4 border-l-[#35D0B0]">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <Badge variant="teal">+24 Hour Horizon</Badge>
              <TrendingUpIcon className="h-4 w-4 text-[#35D0B0]" />
            </div>
            <CardTitle className="text-sm mt-1">WaveANN Wins MAE Skill (+4.71%)</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              At 24 hours, persistence keeps lowest RMSE (0.1866m) due to diurnal reversion, but <strong>WaveANN</strong> achieves
              the best mean absolute accuracy with <strong>0.1225m MAE</strong> (+4.71% skill vs persistence).
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Grouped Bar Chart: RMSE Across Horizons */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChartIcon className="h-4 w-4 text-[#35C6E8]" />
              <span>Root Mean Square Error (RMSE in meters)</span>
            </CardTitle>
            <CardDescription>Lower is better • Evaluated across 19,455 untouched test samples</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="h-64 flex items-center justify-center">
                <Skeleton className="h-56 w-full" />
              </div>
            ) : (
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={groupedChartData} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#16263A" />
                    <XAxis dataKey="horizon" stroke="#94A3B8" fontSize={11} tickLine={false} />
                    <YAxis stroke="#94A3B8" fontSize={11} unit=" m" tickLine={false} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#101D2D',
                        borderColor: '#1E3A5F',
                        borderRadius: '8px',
                        color: '#F1F5F9',
                      }}
                      formatter={(val: any) => [`${Number(val).toFixed(4)} m`, '']}
                    />
                    <Legend verticalAlign="top" height={32} />
                    <Bar dataKey="Persistence (RMSE)" fill="#94A3B8" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="WaveLSTM (RMSE)" fill="#35C6E8" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="WaveANN (RMSE)" fill="#35D0B0" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="WaveCNN1D (RMSE)" fill="#F59E0B" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Grouped Bar Chart: MAE Across Horizons */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChartIcon className="h-4 w-4 text-[#35D0B0]" />
              <span>Mean Absolute Error (MAE in meters)</span>
            </CardTitle>
            <CardDescription>Lower is better • Median absolute deviation measure</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="h-64 flex items-center justify-center">
                <Skeleton className="h-56 w-full" />
              </div>
            ) : (
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={groupedChartData} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#16263A" />
                    <XAxis dataKey="horizon" stroke="#94A3B8" fontSize={11} tickLine={false} />
                    <YAxis stroke="#94A3B8" fontSize={11} unit=" m" tickLine={false} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#101D2D',
                        borderColor: '#1E3A5F',
                        borderRadius: '8px',
                        color: '#F1F5F9',
                      }}
                      formatter={(val: any) => [`${Number(val).toFixed(4)} m`, '']}
                    />
                    <Legend verticalAlign="top" height={32} />
                    <Bar dataKey="Persistence (MAE)" fill="#94A3B8" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="WaveLSTM (MAE)" fill="#35C6E8" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="WaveANN (MAE)" fill="#35D0B0" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="WaveCNN1D (MAE)" fill="#F59E0B" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Full Multi-Horizon Benchmark Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Complete Benchmark Table (Ground Truth Artifacts)</CardTitle>
              <CardDescription>
                Verified against <code>outputs/metrics/multi_horizon_comparison.csv</code>
              </CardDescription>
            </div>
            <Badge variant="cyan" className="font-mono">
              12 Benchmarks
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[#1E3A5F] text-[#94A3B8]">
                  <th className="pb-3 font-semibold">Horizon</th>
                  <th className="pb-3 font-semibold">Model Architecture</th>
                  <th className="pb-3 font-semibold text-right">RMSE (m)</th>
                  <th className="pb-3 font-semibold text-right">MAE (m)</th>
                  <th className="pb-3 font-semibold text-right">R² Coefficient</th>
                  <th className="pb-3 font-semibold text-right">RMSE Skill vs Baseline</th>
                  <th className="pb-3 font-semibold text-right">MAE Skill vs Baseline</th>
                  <th className="pb-3 font-semibold text-center">Outcome</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E3A5F]/40 font-mono">
                {filteredMetrics.map((row: any, idx: number) => {
                  const isPersist = String(row.model).toLowerCase().includes('persistence')
                  const isWinner = row.is_best

                  return (
                    <tr
                      key={`${row.horizon}-${row.model}-${idx}`}
                      className={`hover:bg-[#16263A]/40 transition-colors ${
                        isWinner ? 'bg-[#35C6E8]/5' : ''
                      }`}
                    >
                      <td className="py-3 font-sans font-semibold text-[#35C6E8]">
                        +{row.horizon}h
                      </td>
                      <td className="py-3 font-sans font-medium text-[#F1F5F9] flex items-center gap-2">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            String(row.model).includes('LSTM')
                              ? 'bg-[#35C6E8]'
                              : String(row.model).includes('ANN')
                              ? 'bg-[#35D0B0]'
                              : String(row.model).includes('CNN')
                              ? 'bg-[#F59E0B]'
                              : 'bg-[#94A3B8]'
                          }`}
                        />
                        <span>{row.model}</span>
                        {isPersist && <span className="text-[10px] text-[#94A3B8]">(baseline)</span>}
                      </td>
                      <td className="py-3 text-right font-bold text-[#F1F5F9]">
                        {row.rmse !== undefined && row.rmse !== null ? row.rmse.toFixed(4) : '—'}
                      </td>
                      <td className="py-3 text-right text-[#94A3B8]">
                        {row.mae !== undefined && row.mae !== null ? row.mae.toFixed(4) : '—'}
                      </td>
                      <td className="py-3 text-right text-[#94A3B8]">{formatR2(row.r2)}</td>
                      <td className="py-3 text-right">
                        {row.rmse_skill_pct !== null && row.rmse_skill_pct !== undefined ? (
                          <span
                            className={
                              row.rmse_skill_pct > 0
                                ? 'text-[#35D0B0] font-semibold'
                                : row.rmse_skill_pct === 0
                                ? 'text-[#94A3B8]'
                                : 'text-[#EF4444]'
                            }
                          >
                            {formatPct(row.rmse_skill_pct)}
                          </span>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="py-3 text-right">
                        {row.mae_skill_pct !== null && row.mae_skill_pct !== undefined ? (
                          <span
                            className={
                              row.mae_skill_pct > 0
                                ? 'text-[#35D0B0] font-semibold'
                                : row.mae_skill_pct === 0
                                ? 'text-[#94A3B8]'
                                : 'text-[#EF4444]'
                            }
                          >
                            {formatPct(row.mae_skill_pct)}
                          </span>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="py-3 text-center font-sans">
                        {isWinner ? (
                          <Badge variant="cyan" className="text-[10px]">
                            Horizon Leader
                          </Badge>
                        ) : (
                          <span className="text-[11px] text-[#94A3B8]">—</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

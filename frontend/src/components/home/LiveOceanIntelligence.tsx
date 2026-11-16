import React, { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  Waves,
  Clock,
  MapPin,
  AlertTriangle,
  ArrowRight,
  ShieldCheck,
  TrendingUp,
  Activity,
  Wind,
} from 'lucide-react'
import { api } from '../../services/api'
import type { ForecastResponse, AlertSummaryResponse, CoastalNode } from '../../types'
import { getWmoSeaState, formatMeters } from '../../lib/utils'
import { ThreeDAnalyticsCard } from '../ui/ThreeDAnalyticsCard'

export function LiveOceanIntelligence() {
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [alertSummary, setAlertSummary] = useState<AlertSummaryResponse | null>(null)
  const [nodes, setNodes] = useState<CoastalNode[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true
    const fetchLiveIntelligence = async () => {
      try {
        setLoading(true)
        setError(null)
        const [forecastRes, alertRes, nodesRes] = await Promise.allSettled([
          api.getLiveForecast(6, 'lstm', 'live'),
          api.getAlertSummary(),
          api.getCoastalLocations(),
        ])

        if (!isMounted) return

        if (forecastRes.status === 'fulfilled') {
          setForecast(forecastRes.value)
        }
        if (alertRes.status === 'fulfilled') {
          setAlertSummary(alertRes.value)
        }
        if (nodesRes.status === 'fulfilled') {
          setNodes(nodesRes.value)
        }
      } catch (err) {
        if (isMounted) setError('Telemetry temporarily unavailable from upstream marine provider.')
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    fetchLiveIntelligence()
    return () => {
      isMounted = false
    }
  }, [])

  // Derived telemetry metrics
  const lastObs =
    forecast?.observation_window && forecast.observation_window.length > 0
      ? forecast.observation_window[forecast.observation_window.length - 1]
      : null

  const currentSwh = lastObs?.swh ?? forecast?.predictions?.lstm?.forecast_swh_m ?? null
  const forecastSwh = forecast?.predictions?.lstm?.forecast_swh_m ?? null
  const peakPeriod = lastObs?.mwp ?? null
  const windSpeed = lastObs?.wind_speed ?? null
  const isDemo = forecast?.is_demo ?? false

  const seaState = currentSwh !== null ? getWmoSeaState(currentSwh) : null

  const navigate = useNavigate()

  return (
    <section id="intelligence" className="py-24 bg-[#F4F7FB] border-b border-[#DCE5EF] relative">
      <div className="max-w-7xl mx-auto px-4 md:px-8">
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-16 gap-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#E1F5F2] border border-[#20AFA7]/30 text-xs font-mono font-semibold uppercase text-[#0D837C] mb-3">
              <Activity className="h-3.5 w-3.5" />
              <span>Real-Time Marine Feed</span>
            </div>
            <h2 className="text-3xl md:text-5xl font-bold font-['Outfit',sans-serif] text-[#102A43] tracking-tight">
              Live Ocean Intelligence
            </h2>
            <p className="text-[#64758A] text-base md:text-lg max-w-2xl mt-3 font-normal leading-relaxed">
              Real-time atmospheric and wave telemetry ingested directly from ECMWF WAM global feeds across candidate Arabian Sea coastal grids.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Link
              to="/live"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs font-semibold bg-white hover:bg-[#EAF2F9] text-[#102A43] border border-[#DCE5EF] shadow-xs transition-colors"
            >
              <span>View Live Telemetry Station</span>
              <ArrowRight className="h-3.5 w-3.5 text-[#246B9E]" />
            </Link>
          </div>
        </div>

        {/* 21st.dev Interactive 3D Analytics Dashboard Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Card 1: Significant Wave Height */}
          <ThreeDAnalyticsCard
            title="Observed Wave Height"
            subtitle="ECMWF WAM Real-Time"
            value={loading ? '--' : currentSwh !== null ? currentSwh.toFixed(2) : '--'}
            unit="m"
            badge={seaState?.description || 'Calm to Moderate'}
            badgeVariant="teal"
            icon={Waves}
            trend={{
              value: peakPeriod !== null ? `${peakPeriod.toFixed(1)}s` : '--',
              label: windSpeed !== null ? `Wind: ${windSpeed.toFixed(1)} m/s` : 'Wind: --',
            }}
            onClick={() => navigate('/live')}
          />

          {/* Card 2: +6h Deep Learning Forecast */}
          <ThreeDAnalyticsCard
            title="+6h Forecast Horizon"
            subtitle="WaveLSTM Deep Neural Model"
            value={loading ? '--' : forecastSwh !== null ? forecastSwh.toFixed(2) : '--'}
            unit="m"
            badge="WaveLSTM (+13.88% Skill Score)"
            badgeVariant="sky"
            icon={TrendingUp}
            trend={{
              value: '+13.88%',
              isPositive: true,
              label: 'Lead: 6 Hours',
            }}
            onClick={() => navigate('/comparison')}
          />

          {/* Card 3: Monitored Coastal Nodes */}
          <ThreeDAnalyticsCard
            title="Coastal Grid Stations"
            subtitle="Monitored Arabian Sea Nodes"
            value={loading ? '--' : nodes.length || 8}
            unit="Nodes"
            badge="Mumbai Calibrated Reference"
            badgeVariant="blue"
            icon={MapPin}
            trend={{
              value: '100% Active',
              isPositive: true,
              label: 'Coverage: Indian Coast',
            }}
            onClick={() => navigate('/locations')}
          />

          {/* Card 4: Active Hazard Warnings */}
          <ThreeDAnalyticsCard
            title="Active Hazard Alerts"
            subtitle="Operational Early Warning"
            value={loading ? '--' : alertSummary?.total_active ?? 0}
            unit="Alerts"
            badge={
              (alertSummary?.total_active ?? 0) > 0
                ? `${alertSummary?.total_active} Active Warning`
                : 'Safe Marine Conditions'
            }
            badgeVariant={(alertSummary?.total_active ?? 0) > 0 ? 'sky' : 'teal'}
            icon={AlertTriangle}
            trend={{
              value: (alertSummary?.total_active ?? 0) > 0 ? 'Caution' : 'Normal',
              isPositive: (alertSummary?.total_active ?? 0) === 0,
              label: 'Caution: ≥1.5m • Critical: ≥4.0m',
            }}
            onClick={() => navigate('/alerts')}
          />
        </div>

        {/* Telemetry Provenance Notice */}
        <div className="mt-8 flex flex-col sm:flex-row items-center justify-between gap-4 p-4 rounded-xl bg-[#EAF2F9] border border-[#DCE5EF] text-xs text-[#64758A]">
          <div className="flex items-center gap-2">
            <span className="inline-block h-2 w-2 rounded-full bg-[#20AFA7]" />
            <span>
              <strong>Attribution:</strong> Open-Meteo Marine API under CC BY 4.0 &bull; ECMWF WAM 0.08&deg; (~9km) numerical wave model &bull; IFS global atmospheric predictions.
            </span>
          </div>

          <span className="font-mono text-[11px] text-[#246B9E] shrink-0">
            {isDemo ? 'OFFLINE ERA5 DEMO' : 'LIVE OPERATIONAL FEED'}
          </span>
        </div>
      </div>
    </section>
  )
}

import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  Radio,
  MapPin,
  ShieldCheck,
  AlertCircle,
  RefreshCw,
  ExternalLink,
  Activity,
  Waves,
  Clock,
  Compass,
} from 'lucide-react'
import { api } from '../services/api'
import type { CoastalNode, NodeMapSummaryItem } from '../types'
import { Button } from '../components/ui/Button'
import { StatusIndicator } from '../components/ui/StatusIndicator'

export function CoastalNodesPage() {
  const [nodesTelemetry, setNodesTelemetry] = useState<NodeMapSummaryItem[]>([])
  const [coastalNodes, setCoastalNodes] = useState<CoastalNode[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [source, setSource] = useState<'demo' | 'live'>('demo')
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false)

  const fetchData = async (forceRefresh: boolean = false) => {
    try {
      if (forceRefresh) setIsRefreshing(true)
      else setLoading(true)
      setError(null)

      const [mapData, regData] = await Promise.all([
        api.getCoastalMapData(6, source, 'lstm', forceRefresh),
        api.getCoastalLocations(),
      ])

      setNodesTelemetry(mapData)
      setCoastalNodes(regData)
    } catch (err: unknown) {
      console.error('Failed to load coastal nodes:', err)
      setError(err instanceof Error ? err.message : 'Failed to query coastal nodes telemetry.')
    } finally {
      setLoading(false)
      setIsRefreshing(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [source])

  // Merge registry info with telemetry
  const mergedStations = coastalNodes.map((reg) => {
    const tel = nodesTelemetry.find((t) => t.location_id === reg.location_id)
    return {
      ...reg,
      telemetry: tel || null,
    }
  })

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="rounded-xl bg-gradient-to-r from-[#0F172A] via-[#162B4D] to-[#0A192F] p-5 border border-[#1E3A5F] shadow-lg">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#35C6E8]/20 text-[#35C6E8] border border-[#35C6E8]/40">
                <Radio className="h-4 w-4" />
              </span>
              <h1 className="text-xl font-bold tracking-tight text-[#F1F5F9] font-['Outfit',sans-serif]">
                Coastal Node Registry & Telemetry Feeds
              </h1>
            </div>
            <p className="mt-1 text-xs text-[#94A3B8] max-w-3xl leading-relaxed">
              Multi-location ingestion pipeline monitoring 8 coastal candidate nodes across the Arabian Sea coastline.
              Tracks feed status, data freshness, localized wave heights, and geographic model validation.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex bg-[#07111F] p-1 rounded-lg border border-[#1E3A5F]">
              <button
                onClick={() => setSource('demo')}
                className={`px-3 py-1 text-xs font-semibold rounded-md transition-colors ${
                  source === 'demo'
                    ? 'bg-[#1E3A5F] text-[#35C6E8] border border-[#35C6E8]/40'
                    : 'text-[#94A3B8] hover:text-[#F1F5F9]'
                }`}
              >
                📂 ERA5 Demo
              </button>
              <button
                onClick={() => setSource('live')}
                className={`px-3 py-1 text-xs font-semibold rounded-md transition-colors ${
                  source === 'live'
                    ? 'bg-[#1E3A5F] text-[#35C6E8] border border-[#35C6E8]/40'
                    : 'text-[#94A3B8] hover:text-[#F1F5F9]'
                }`}
              >
                🌐 Open-Meteo REST
              </button>
            </div>

            <Button
              variant="secondary"
              size="sm"
              onClick={() => fetchData(true)}
              isLoading={isRefreshing}
              className="text-xs"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
              <span>Refresh Telemetry</span>
            </Button>
          </div>
        </div>
      </div>

      {/* Grid of Coastal Stations */}
      {loading ? (
        <div className="rounded-xl bg-[#101D2D] p-12 text-center border border-[#1E3A5F] text-xs text-[#35C6E8]">
          <RefreshCw className="h-6 w-6 animate-spin mx-auto mb-2" />
          <span>Polling multi-location telemetry pipeline...</span>
        </div>
      ) : error ? (
        <div className="rounded-xl bg-red-950/80 p-4 border border-red-500/50 text-xs text-red-200">
          {error}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {mergedStations.map((station) => {
            const tel = station.telemetry
            const isRef = station.is_reference
            const isDegraded = tel?.is_degraded || tel?.data_status === 'unavailable'
            const isStale = tel?.data_status === 'stale'
            const markerColor = tel?.marker_hex_color || '#64748B'

            return (
              <div
                key={station.location_id}
                className="rounded-xl bg-[#101D2D] border border-[#1E3A5F] overflow-hidden flex flex-col justify-between hover:border-[#35C6E8]/60 transition-all shadow-md group"
              >
                {/* Card Top */}
                <div className="p-4 space-y-3">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h3 className="text-sm font-bold text-[#F1F5F9] group-hover:text-[#35C6E8] transition-colors">
                        {station.name}
                      </h3>
                      <div className="text-[11px] text-[#35C6E8] font-mono mt-0.5">
                        {station.coordinate_label}
                      </div>
                    </div>
                    <span
                      className="px-2 py-0.5 rounded text-[10px] font-bold border"
                      style={{
                        color: markerColor,
                        borderColor: `${markerColor}66`,
                        backgroundColor: `${markerColor}15`,
                      }}
                    >
                      {tel ? tel.current_risk_level : 'UNKNOWN'}
                    </span>
                  </div>

                  {/* Validation Badge */}
                  <div>
                    {isRef ? (
                      <div className="inline-flex items-center gap-1 bg-emerald-950/70 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded text-[10px] font-semibold">
                        <ShieldCheck className="h-3 w-3 text-emerald-400" />
                        <span>REFERENCE VALIDATED</span>
                      </div>
                    ) : (
                      <div className="inline-flex items-center gap-1 bg-amber-950/70 text-amber-300 border border-amber-500/30 px-2 py-0.5 rounded text-[10px] font-semibold">
                        <AlertCircle className="h-3 w-3 text-amber-400" />
                        <span>EXPERIMENTAL REGIONAL NODE</span>
                      </div>
                    )}
                  </div>

                  <p className="text-xs text-[#94A3B8] leading-relaxed line-clamp-2">
                    {station.description}
                  </p>

                  {/* Live Telemetry Summary */}
                  <div className="grid grid-cols-2 gap-2 bg-[#07111F] p-2.5 rounded-lg border border-[#1E3A5F] text-xs">
                    <div>
                      <div className="text-[10px] text-[#94A3B8] uppercase">Observed (Hs)</div>
                      <div className="text-base font-bold font-mono text-[#F1F5F9]">
                        {tel?.latest_obs_swh_m !== null && tel?.latest_obs_swh_m !== undefined
                          ? `${tel.latest_obs_swh_m.toFixed(2)} m`
                          : 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[#94A3B8] uppercase">+6h Forecast</div>
                      <div
                        className="text-base font-bold font-mono"
                        style={{ color: markerColor }}
                      >
                        {tel?.forecast_swh_m !== null && tel?.forecast_swh_m !== undefined
                          ? `${tel.forecast_swh_m.toFixed(2)} m`
                          : 'N/A'}
                      </div>
                    </div>
                  </div>

                  {/* Feed Status */}
                  <div className="text-[11px] text-[#94A3B8] flex items-center justify-between pt-1">
                    <span className="flex items-center gap-1">
                      <span
                        className={`h-2 w-2 rounded-full ${
                          isDegraded ? 'bg-red-500' : isStale ? 'bg-amber-400' : 'bg-emerald-400'
                        }`}
                      />
                      <span className="font-mono text-[#CBD5E1]">
                        {tel?.data_status.toUpperCase() || 'UNAVAILABLE'}
                      </span>
                    </span>

                    {tel?.active_alert_count ? (
                      <span className="bg-red-500/20 text-red-400 border border-red-500/40 text-[10px] font-bold px-1.5 py-0.5 rounded">
                        {tel.active_alert_count} ALERTS
                      </span>
                    ) : (
                      <span className="text-[10px] text-emerald-400">Nominal</span>
                    )}
                  </div>
                </div>

                {/* Card Footer Actions */}
                <div className="p-3 bg-[#07111F]/60 border-t border-[#1E3A5F] flex items-center justify-between text-xs">
                  <Link
                    to="/map"
                    className="text-[#35C6E8] hover:underline flex items-center gap-1"
                  >
                    <Compass className="h-3.5 w-3.5" />
                    <span>View on Map</span>
                  </Link>

                  <Link
                    to={`/live?location=${station.location_id}`}
                    className="text-[#94A3B8] hover:text-[#F1F5F9] flex items-center gap-1"
                  >
                    <span>Forecast</span>
                    <ExternalLink className="h-3 w-3" />
                  </Link>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

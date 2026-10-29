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
import { Badge } from '../components/ui/Badge'
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
    <div className="space-y-6 pb-8">
      {/* Header Banner */}
      <div className="bg-white p-6 rounded-2xl border border-[#DCE5EF] shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#EAF2F9] text-[#245B8F] border border-[#DCE5EF]">
                <Radio className="h-4 w-4" />
              </span>
              <h1 className="text-xl font-bold tracking-tight text-[#15304B] font-['Outfit',sans-serif]">
                Coastal Node Registry & Telemetry Feeds
              </h1>
            </div>
            <p className="mt-1 text-xs text-[#64758A] max-w-3xl leading-relaxed">
              Multi-location ingestion pipeline monitoring 8 coastal candidate nodes across the Arabian Sea coastline.
              Tracks feed status, data freshness, localized wave heights, and geographic model validation.
            </p>
          </div>

          <div className="flex items-center gap-2.5">
            <div className="flex bg-[#F0F6FC] p-1 rounded-lg border border-[#DCE5EF]">
              <button
                onClick={() => setSource('demo')}
                className={`px-3 py-1 text-xs font-semibold rounded-md transition-colors ${
                  source === 'demo'
                    ? 'bg-white text-[#245B8F] border border-[#DCE5EF] shadow-xs'
                    : 'text-[#64758A] hover:text-[#192B3C]'
                }`}
              >
                ERA5 Demo
              </button>
              <button
                onClick={() => setSource('live')}
                className={`px-3 py-1 text-xs font-semibold rounded-md transition-colors ${
                  source === 'live'
                    ? 'bg-white text-[#245B8F] border border-[#DCE5EF] shadow-xs'
                    : 'text-[#64758A] hover:text-[#192B3C]'
                }`}
              >
                Open-Meteo REST
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
        <div className="bg-white p-12 text-center rounded-2xl border border-[#DCE5EF] text-xs text-[#20AFA7] shadow-xs">
          <RefreshCw className="h-6 w-6 animate-spin mx-auto mb-2 text-[#20AFA7]" />
          <span className="font-mono text-[#64758A]">Polling multi-location telemetry pipeline...</span>
        </div>
      ) : error ? (
        <div className="rounded-xl bg-red-50 p-4 border border-red-200 text-xs text-red-700 font-mono">
          {error}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          {mergedStations.map((station) => {
            const tel = station.telemetry
            const isRef = station.is_reference
            const isDegraded = tel?.is_degraded || tel?.data_status === 'unavailable'
            const isStale = tel?.data_status === 'stale'
            const markerColor = tel?.marker_hex_color || '#64748B'

            return (
              <div
                key={station.location_id}
                className="bg-white border border-[#DCE5EF] rounded-2xl p-5 overflow-hidden flex flex-col justify-between hover:border-[#20AFA7]/60 hover:shadow-md transition-all shadow-xs group"
              >
                {/* Station Top */}
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h3 className="text-sm font-bold text-[#15304B] group-hover:text-[#245B8F] transition-colors font-['Outfit',sans-serif]">
                        {station.name}
                      </h3>
                      <div className="text-[11px] text-[#245B8F] font-mono mt-0.5">
                        {station.coordinate_label}
                      </div>
                    </div>
                    <span
                      className="px-2 py-0.5 rounded text-[10px] font-mono font-bold border"
                      style={{
                        color: markerColor,
                        borderColor: `${markerColor}44`,
                        backgroundColor: `${markerColor}15`,
                      }}
                    >
                      {tel?.current_risk_level || 'UNKNOWN'}
                    </span>
                  </div>

                  <p className="text-[11px] text-[#64758A] leading-relaxed line-clamp-2">
                    {station.description}
                  </p>

                  {/* Telemetry Metrics */}
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono p-3 rounded-xl bg-[#F0F6FC] border border-[#DCE5EF]">
                    <div>
                      <span className="text-[10px] text-[#64758A] block">OBSERVED $H_s$</span>
                      <span className="text-base font-bold text-[#15304B]">
                        {tel?.latest_obs_swh_m != null ? `${Number(tel.latest_obs_swh_m).toFixed(2)} m` : 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] text-[#64758A] block">+6h FORECAST</span>
                      <span className="text-base font-bold text-[#20AFA7]">
                        {tel?.forecast_swh_m != null ? `${Number(tel.forecast_swh_m).toFixed(2)} m` : 'N/A'}
                      </span>
                    </div>
                  </div>

                  {/* Ingestion & Feed Status */}
                  <div className="flex items-center justify-between text-[11px] font-mono pt-1">
                    <span className="text-[#64758A]">Feed Freshness:</span>
                    <span className={`font-semibold ${isDegraded ? 'text-red-600' : isStale ? 'text-amber-600' : 'text-emerald-600'}`}>
                      {tel?.data_status || 'FRESH'}
                    </span>
                  </div>

                  {/* Validation status badge */}
                  <div>
                    {isRef ? (
                      <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-[#E1F5F2] text-[#0D9488] text-[10px] font-mono border border-[#20AFA7]/30 w-full justify-center font-medium">
                        <ShieldCheck className="h-3.5 w-3.5" /> Reference Calibration Station
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-[#FEF3C7] text-[#D97706] text-[10px] font-mono border border-[#F59E0B]/30 w-full justify-center font-medium">
                        <AlertCircle className="h-3.5 w-3.5" /> Regional Transfer Candidate
                      </span>
                    )}
                  </div>
                </div>

                {/* Footer Link to Map */}
                <div className="pt-4 border-t border-[#DCE5EF] mt-4 flex items-center justify-between text-xs">
                  <Link
                    to="/map"
                    className="text-[#20AFA7] hover:text-[#0D9488] hover:underline flex items-center gap-1 text-[11px] font-mono font-medium"
                  >
                    <span>View Station on Map</span>
                    <ExternalLink className="h-3 w-3" />
                  </Link>

                  <span className="text-[10px] font-mono text-[#64758A]">
                    {station.region}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

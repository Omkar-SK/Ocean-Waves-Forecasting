import React, { useState, useEffect, useRef, useMemo } from 'react'
import L from 'leaflet'
import {
  MapPin,
  AlertTriangle,
  Clock,
  Layers,
  Info,
  Radio,
  RefreshCw,
  Sliders,
  TrendingUp,
  ShieldCheck,
  AlertCircle,
  ExternalLink,
} from 'lucide-react'
import { api } from '../services/api'
import type { Horizon, NodeMapSummaryItem, RiskLevel, ThresholdConfigResponse } from '../types'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { StatusIndicator } from '../components/ui/StatusIndicator'

export function CoastalRiskMapPage() {
  const [horizon, setHorizon] = useState<Horizon>(6)
  const [source, setSource] = useState<'demo' | 'live'>('demo')
  const [selectedRiskFilter, setSelectedRiskFilter] = useState<string>('ALL')
  const [alertsOnly, setAlertsOnly] = useState<boolean>(false)

  const [nodes, setNodes] = useState<NodeMapSummaryItem[]>([])
  const [thresholds, setThresholds] = useState<ThresholdConfigResponse | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string>('mumbai_offshore')
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false)

  const mapContainerRef = useRef<HTMLDivElement | null>(null)
  const mapInstanceRef = useRef<L.Map | null>(null)
  const markersGroupRef = useRef<L.LayerGroup | null>(null)

  // Fetch data
  const fetchData = async (forceRefresh: boolean = false) => {
    try {
      if (forceRefresh) setIsRefreshing(true)
      else setLoading(true)
      setError(null)

      const [mapData, threshData] = await Promise.all([
        api.getCoastalMapData(horizon, source, 'lstm', forceRefresh),
        api.getCoastalThresholds(),
      ])

      setNodes(mapData)
      setThresholds(threshData)

      // Default selected node if current one is not in returned list
      if (!mapData.some((n) => n.location_id === selectedNodeId) && mapData.length > 0) {
        setSelectedNodeId(mapData[0].location_id)
      }
    } catch (err: unknown) {
      console.error('Failed to load coastal map data:', err)
      setError(err instanceof Error ? err.message : 'Failed to load coastal map telemetry')
    } finally {
      setLoading(false)
      setIsRefreshing(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [horizon, source])

  // Filtered nodes
  const filteredNodes = useMemo(() => {
    return nodes.filter((n) => {
      if (selectedRiskFilter !== 'ALL' && n.current_risk_level !== selectedRiskFilter) {
        return false
      }
      if (alertsOnly && n.active_alert_count === 0) {
        return false
      }
      return true
    })
  }, [nodes, selectedRiskFilter, alertsOnly])

  const selectedNode = useMemo(() => {
    return nodes.find((n) => n.location_id === selectedNodeId) || nodes[0] || null
  }, [nodes, selectedNodeId])

  // Initialize Leaflet Map once
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return

    // Center on Arabian Sea / Indian West Coast
    const map = L.map(mapContainerRef.current, {
      center: [15.2, 73.0],
      zoom: 6,
      minZoom: 4,
      maxZoom: 12,
      zoomControl: true,
    })

    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap contributors',
      maxZoom: 19,
    }).addTo(map)

    const markersGroup = L.layerGroup().addTo(map)
    mapInstanceRef.current = map
    markersGroupRef.current = markersGroup

    return () => {
      map.remove()
      mapInstanceRef.current = null
      markersGroupRef.current = null
    }
  }, [])

  // Update Leaflet Markers when nodes change
  useEffect(() => {
    if (!mapInstanceRef.current || !markersGroupRef.current) return

    markersGroupRef.current.clearLayers()

    filteredNodes.forEach((node) => {
      const isSelected = node.location_id === selectedNodeId
      const color = node.marker_hex_color

      // Outer circle (halo)
      const halo = L.circleMarker([node.latitude, node.longitude], {
        radius: isSelected ? 18 : 12,
        fillColor: color,
        fillOpacity: isSelected ? 0.35 : 0.2,
        color: color,
        weight: isSelected ? 2.5 : 1.5,
      })

      // Inner center dot
      const marker = L.circleMarker([node.latitude, node.longitude], {
        radius: isSelected ? 9 : 6,
        fillColor: color,
        fillOpacity: 0.95,
        color: '#FFFFFF',
        weight: 1.5,
      })

      // Popup
      const popupContent = `
        <div style="font-family: sans-serif; color: #0F172A; min-width: 190px; padding: 4px;">
          <div style="font-weight: 700; font-size: 13px; margin-bottom: 2px;">${node.name}</div>
          <div style="font-size: 11px; color: #64748B; margin-bottom: 6px;">${node.coordinate_label} • ${node.region}</div>
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
            <span>Current Risk:</span>
            <span style="font-weight: 700; color: ${color};">${node.current_risk_level}</span>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
            <span>Observed (Hs):</span>
            <span style="font-weight: 600;">${node.latest_obs_swh_m != null ? `${Number(node.latest_obs_swh_m).toFixed(2)} m` : 'N/A'}</span>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
            <span>+${node.forecast_horizon_hours}h Forecast:</span>
            <span style="font-weight: 700; color: #0284C7;">${node.forecast_swh_m != null ? `${Number(node.forecast_swh_m).toFixed(2)} m` : 'N/A'}</span>
          </div>
          ${
            node.is_reference
              ? `<div style="background: #E0F2FE; color: #0369A1; font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px; margin-top: 6px; text-align: center;">REFERENCE VALIDATED</div>`
              : `<div style="background: #FEF3C7; color: #92400E; font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px; margin-top: 6px; text-align: center;">EXPERIMENTAL REGIONAL NODE</div>`
          }
        </div>
      `
      marker.bindPopup(popupContent)

      // Click to select node in side drawer
      const onMarkerClick = () => {
        setSelectedNodeId(node.location_id)
      }
      marker.on('click', onMarkerClick)
      halo.on('click', onMarkerClick)

      markersGroupRef.current?.addLayer(halo)
      markersGroupRef.current?.addLayer(marker)
    })
  }, [filteredNodes, selectedNodeId])

  // Center on selected node
  const handleNodeCardClick = (lid: string, lat: number, lon: number) => {
    setSelectedNodeId(lid)
    mapInstanceRef.current?.flyTo([lat, lon], 7.5, { duration: 1.0 })
  }

  return (
    <div className="space-y-6">
      {/* Top Banner & Scientific Disclaimer */}
      <div className="rounded-xl bg-gradient-to-r from-[#0F172A] via-[#102A43] to-[#0A192F] p-5 border border-[#1E3A5F] shadow-lg">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#35C6E8]/20 text-[#35C6E8] border border-[#35C6E8]/40">
                <MapPin className="h-4 w-4" />
              </span>
              <h1 className="text-xl font-bold tracking-tight text-[#F1F5F9] font-['Outfit',sans-serif]">
                Interactive Coastal Risk Heatmap
              </h1>
            </div>
            <p className="mt-1 text-xs text-[#94A3B8] max-w-3xl leading-relaxed">
              Geographic wave risk assessment across 8 registered candidate monitoring nodes along the Indian West Coast.
              Evaluates localized swell hazard against engineering bounds with zero spatial interpolation.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => fetchData(true)}
              isLoading={isRefreshing}
              className="text-xs"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
              <span>Refresh Map</span>
            </Button>
          </div>
        </div>

        {/* Notice on Spatial Interpolation */}
        <div className="mt-4 rounded-lg bg-[#07111F]/80 p-3 border border-[#1E3A5F]/70 text-xs text-[#94A3B8] flex items-start gap-2.5">
          <Info className="h-4 w-4 text-[#35C6E8] shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-[#F1F5F9]">Scientific Integrity Policy:</span> Point-forecast markers are rendered strictly at registered offshore/port coordinates. Continuous contour surface interpolation between sparse coastal points is scientifically disabled to prevent misleading intermediate hazard claims.
          </div>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 bg-[#101D2D] p-3.5 rounded-xl border border-[#1E3A5F]">
        {/* Horizon Selector */}
        <div>
          <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider block mb-1.5">
            Forecast Horizon
          </label>
          <div className="grid grid-cols-3 gap-1 bg-[#07111F] p-1 rounded-lg border border-[#1E3A5F]">
            {[1, 6, 24].map((h) => (
              <button
                key={h}
                onClick={() => setHorizon(h as Horizon)}
                className={`py-1.5 text-xs font-semibold rounded-md transition-colors ${
                  horizon === h
                    ? 'bg-[#35C6E8] text-[#07111F]'
                    : 'text-[#94A3B8] hover:text-[#F1F5F9] hover:bg-[#16263A]'
                }`}
              >
                +{h}h
              </button>
            ))}
          </div>
        </div>

        {/* Telemetry Stream */}
        <div>
          <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider block mb-1.5">
            Data Stream
          </label>
          <div className="grid grid-cols-2 gap-1 bg-[#07111F] p-1 rounded-lg border border-[#1E3A5F]">
            <button
              onClick={() => setSource('demo')}
              className={`py-1.5 text-xs font-semibold rounded-md transition-colors ${
                source === 'demo'
                  ? 'bg-[#1E3A5F] text-[#35C6E8] border border-[#35C6E8]/40'
                  : 'text-[#94A3B8] hover:text-[#F1F5F9]'
              }`}
            >
              📂 ERA5 Demo
            </button>
            <button
              onClick={() => setSource('live')}
              className={`py-1.5 text-xs font-semibold rounded-md transition-colors ${
                source === 'live'
                  ? 'bg-[#1E3A5F] text-[#35C6E8] border border-[#35C6E8]/40'
                  : 'text-[#94A3B8] hover:text-[#F1F5F9]'
              }`}
            >
              🌐 Open-Meteo
            </button>
          </div>
        </div>

        {/* Risk Level Filter */}
        <div>
          <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider block mb-1.5">
            Filter Risk Category
          </label>
          <select
            value={selectedRiskFilter}
            onChange={(e) => setSelectedRiskFilter(e.target.value)}
            className="w-full bg-[#07111F] border border-[#1E3A5F] rounded-lg px-3 py-2 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#35C6E8]"
          >
            <option value="ALL">All Categories</option>
            <option value="RED">🔴 Critical Risk (≥ 4.0m)</option>
            <option value="ORANGE">🟠 High Risk (≥ 2.5m)</option>
            <option value="YELLOW">🟡 Caution (≥ 1.5m)</option>
            <option value="GREEN">🟢 Low Hazard (&lt; 1.5m)</option>
          </select>
        </div>

        {/* Alerts Only Toggle */}
        <div className="flex flex-col justify-end">
          <label className="flex items-center gap-2 cursor-pointer bg-[#07111F] p-2.5 rounded-lg border border-[#1E3A5F] text-xs text-[#F1F5F9]">
            <input
              type="checkbox"
              checked={alertsOnly}
              onChange={(e) => setAlertsOnly(e.target.checked)}
              className="h-4 w-4 rounded border-[#1E3A5F] text-[#35C6E8] focus:ring-0 bg-[#16263A]"
            />
            <span className="font-medium">Active Alerts Only</span>
          </label>
        </div>
      </div>

      {/* Main Map + Synchronized Drawer Split */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Leaflet Interactive Map Container */}
        <div className="lg:col-span-2 space-y-3">
          <div className="relative h-[560px] w-full rounded-xl overflow-hidden border border-[#1E3A5F] shadow-xl bg-[#0B1523]">
            {loading && (
              <div className="absolute inset-0 z-20 flex items-center justify-center bg-[#07111F]/70 backdrop-blur-xs">
                <div className="flex flex-col items-center gap-2 text-xs text-[#35C6E8]">
                  <RefreshCw className="h-6 w-6 animate-spin" />
                  <span>Loading coastal telemetry...</span>
                </div>
              </div>
            )}

            {error && (
              <div className="absolute top-4 left-4 right-4 z-20 rounded-lg bg-red-950/80 p-3 border border-red-500/50 text-xs text-red-200">
                {error}
              </div>
            )}

            {/* Map Element */}
            <div ref={mapContainerRef} className="h-full w-full z-10" />

            {/* Shared Risk Legend Overlay */}
            <div className="absolute bottom-4 left-4 z-20 rounded-lg bg-[#07111F]/90 p-3 border border-[#1E3A5F] shadow-md backdrop-blur-sm text-xs text-[#F1F5F9] max-w-xs">
              <div className="font-semibold text-xs text-[#94A3B8] mb-1.5 flex items-center justify-between">
                <span>Configured Risk Legend</span>
                <span className="text-[10px] text-[#35C6E8] font-mono">v1.0</span>
              </div>
              <div className="space-y-1 font-mono text-[11px]">
                <div className="flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full bg-[#EF4444]" />
                  <span>Critical Risk: ≥ {thresholds?.critical_m ?? 4.0}m</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full bg-[#F97316]" />
                  <span>High Risk: ≥ {thresholds?.high_risk_m ?? 2.5}m</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full bg-[#F59E0B]" />
                  <span>Caution: ≥ {thresholds?.caution_m ?? 1.5}m</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full bg-[#10B981]" />
                  <span>Low Risk: &lt; {thresholds?.caution_m ?? 1.5}m</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full bg-[#64748B]" />
                  <span className="text-[#94A3B8]">Offline / Stale Telemetry</span>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Nodes Selector Carousel / Pills */}
          <div className="flex items-center gap-2 overflow-x-auto py-1">
            {filteredNodes.map((n) => (
              <button
                key={n.location_id}
                onClick={() => handleNodeCardClick(n.location_id, n.latitude, n.longitude)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap border transition-all flex items-center gap-2 ${
                  n.location_id === selectedNodeId
                    ? 'bg-[#1E3A5F] text-[#F1F5F9] border-[#35C6E8] shadow-xs'
                    : 'bg-[#101D2D] text-[#94A3B8] border-[#1E3A5F]/70 hover:border-[#94A3B8]'
                }`}
              >
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: n.marker_hex_color }}
                />
                <span>{n.name}</span>
                {n.active_alert_count > 0 && (
                  <span className="rounded bg-red-500/20 text-red-400 px-1 text-[10px] font-bold">
                    {n.active_alert_count}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Right: Synchronized Location Detail Drawer */}
        <div className="space-y-4">
          {selectedNode ? (
            <div className="rounded-xl bg-[#101D2D] border border-[#1E3A5F] p-5 shadow-lg space-y-4">
              {/* Header */}
              <div className="border-b border-[#1E3A5F] pb-3">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-base font-bold text-[#F1F5F9] leading-snug">
                      {selectedNode.name}
                    </h3>
                    <p className="text-xs text-[#35C6E8] font-mono mt-0.5">
                      {selectedNode.coordinate_label} • {selectedNode.region}
                    </p>
                  </div>
                  <span
                    className="px-2.5 py-1 rounded text-xs font-bold border"
                    style={{
                      color: selectedNode.marker_hex_color,
                      borderColor: `${selectedNode.marker_hex_color}55`,
                      backgroundColor: `${selectedNode.marker_hex_color}15`,
                    }}
                  >
                    {selectedNode.current_risk_level}
                  </span>
                </div>

                {/* Validation Status Pill */}
                <div className="mt-3">
                  {selectedNode.is_reference ? (
                    <div className="inline-flex items-center gap-1.5 bg-emerald-950/60 text-emerald-400 border border-emerald-500/30 px-2.5 py-1 rounded-md text-xs font-semibold">
                      <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
                      <span>Reference Grid (18.50°N, 72.50°E Validated)</span>
                    </div>
                  ) : (
                    <div className="inline-flex items-center gap-1.5 bg-amber-950/60 text-amber-300 border border-amber-500/30 px-2.5 py-1 rounded-md text-xs font-semibold">
                      <AlertCircle className="h-3.5 w-3.5 text-amber-400" />
                      <span>Experimental Node (Regional Transfer)</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Conditions Metrics Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-[#07111F] p-3 rounded-lg border border-[#1E3A5F]">
                  <div className="text-[11px] text-[#94A3B8] uppercase">Observed Wave (Hs)</div>
                  <div className="text-xl font-bold text-[#F1F5F9] font-mono mt-0.5">
                    {selectedNode.latest_obs_swh_m != null
                      ? `${selectedNode.latest_obs_swh_m.toFixed(2)} m`
                      : 'Unavailable'}
                  </div>
                  <div className="text-[10px] text-[#64748B] mt-0.5">
                    {selectedNode.latest_obs_time_utc
                      ? new Date(selectedNode.latest_obs_time_utc).toUTCString().slice(17, 22) + ' UTC'
                      : 'N/A'}
                  </div>
                </div>

                <div className="bg-[#07111F] p-3 rounded-lg border border-[#1E3A5F]">
                  <div className="text-[11px] text-[#94A3B8] uppercase">+{selectedNode.forecast_horizon_hours}h DL Forecast</div>
                  <div
                    className="text-xl font-bold font-mono mt-0.5"
                    style={{ color: selectedNode.marker_hex_color }}
                  >
                    {selectedNode.forecast_swh_m != null
                      ? `${selectedNode.forecast_swh_m.toFixed(2)} m`
                      : 'Calculating...'}
                  </div>
                  <div className="text-[10px] text-[#64748B] mt-0.5">
                    Target: +{selectedNode.forecast_horizon_hours}h Lead Time
                  </div>
                </div>
              </div>

              {/* Multi-Horizon Breakdown Table */}
              <div>
                <h4 className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider mb-2">
                  ⏱️ Multi-Horizon Lead Times vs ECMWF NWP
                </h4>
                <div className="rounded-lg border border-[#1E3A5F] overflow-hidden">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-[#07111F] text-[#94A3B8] border-b border-[#1E3A5F]">
                      <tr>
                        <th className="py-2 px-3">Horizon</th>
                        <th className="py-2 px-3">DL Forecast</th>
                        <th className="py-2 px-3">ECMWF NWP</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1E3A5F] bg-[#101D2D]">
                      {[1, 6, 24].map((hKey) => {
                        const dlVal = selectedNode.multi_horizon_predictions?.[hKey]
                        const nwpVal = selectedNode.external_nwp_forecasts?.[hKey]
                        return (
                          <tr key={hKey} className="hover:bg-[#16263A]">
                            <td className="py-2 px-3 font-semibold text-[#F1F5F9]">t + {hKey}h</td>
                            <td className="py-2 px-3 font-mono text-[#35C6E8] font-bold">
                              {dlVal !== undefined ? `${dlVal.toFixed(2)} m` : 'N/A'}
                            </td>
                            <td className="py-2 px-3 font-mono text-[#94A3B8]">
                              {nwpVal !== undefined ? `${nwpVal.toFixed(2)} m` : 'N/A'}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Active Alerts for this Node */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider">
                    🚨 Active Alerts ({selectedNode.active_alert_count})
                  </h4>
                </div>
                {selectedNode.active_alerts && selectedNode.active_alerts.length > 0 ? (
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {selectedNode.active_alerts.map((alt) => (
                      <div
                        key={alt.alert_id}
                        className="rounded-lg bg-[#07111F] p-2.5 border border-red-500/40 text-xs text-[#F1F5F9]"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-red-400">{alt.severity} ALERT</span>
                          <span className="text-[10px] text-[#94A3B8] font-mono">+{alt.forecast_horizon_hours}h</span>
                        </div>
                        <div className="text-[11px] text-[#CBD5E1] mt-1 leading-snug">
                          {alt.trigger_reason}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-lg bg-[#07111F] p-3 text-center border border-[#1E3A5F] text-xs text-[#64748B]">
                    No active hazard alerts for this station.
                  </div>
                )}
              </div>

              {/* Station Metadata & Limitations */}
              <div className="rounded-lg bg-[#07111F]/60 p-3 border border-[#1E3A5F]/60 text-xs text-[#94A3B8] space-y-1">
                <div>
                  <span className="text-[#64748B]">Data Source:</span>{' '}
                  <span className="text-[#F1F5F9]">{selectedNode.data_source}</span>
                </div>
                <div>
                  <span className="text-[#64748B]">Telemetry Status:</span>{' '}
                  <span className="text-[#F1F5F9] font-mono">{selectedNode.data_status.toUpperCase()}</span>
                </div>
                {selectedNode.limitations && selectedNode.limitations.length > 0 && (
                  <div className="mt-2 text-[11px] text-amber-400/90 leading-tight">
                    • {selectedNode.limitations[0]}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="rounded-xl bg-[#101D2D] border border-[#1E3A5F] p-8 text-center text-xs text-[#64748B]">
              Select a coastal node on the map to inspect telemetry and hazard details.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

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
  Waves,
  Flame,
  FlaskConical,
} from 'lucide-react'
import { api } from '../services/api'
import { useSimulation } from '../context/SimulationContext'
import { HazardHeatmapLegend } from '../components/simulation/HazardHeatmapLegend'
import type { Horizon, NodeMapSummaryItem, RiskLevel, ThresholdConfigResponse } from '../types'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { StatusIndicator } from '../components/ui/StatusIndicator'

export function CoastalRiskMapPage() {
  const { isSimulationActive, activeScenario, simulationData, openModal } = useSimulation()

  const [horizon, setHorizon] = useState<Horizon>(6)
  const [source, setSource] = useState<'demo' | 'live'>('demo')
  const [selectedRiskFilter, setSelectedRiskFilter] = useState<string>('ALL')
  const [alertsOnly, setAlertsOnly] = useState<boolean>(false)
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true)

  const [nodes, setNodes] = useState<NodeMapSummaryItem[]>([])
  const [thresholds, setThresholds] = useState<ThresholdConfigResponse | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string>('mumbai_offshore')
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false)

  const mapContainerRef = useRef<HTMLDivElement | null>(null)
  const mapInstanceRef = useRef<L.Map | null>(null)
  const markersGroupRef = useRef<L.LayerGroup | null>(null)
  const heatmapGroupRef = useRef<L.LayerGroup | null>(null)

  const fetchData = async (forceRefresh: boolean = false) => {
    if (isSimulationActive && simulationData?.nodes) {
      setNodes(simulationData.nodes)
      setLoading(false)
      setIsRefreshing(false)
      return
    }
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
    if (isSimulationActive && simulationData?.nodes) {
      setNodes(simulationData.nodes)
    } else if (!isSimulationActive) {
      fetchData()
    }
  }, [isSimulationActive, simulationData])

  useEffect(() => {
    fetchData()
  }, [horizon, source])

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

  // Initialize Leaflet Map once with clean keyless tiles or optional CARTO key
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return

    const map = L.map(mapContainerRef.current, {
      center: [16.5, 73.0],
      zoom: 6,
      minZoom: 4,
      maxZoom: 12,
      zoomControl: true,
    })

    const cartoKey = import.meta.env.VITE_CARTO_API_KEY
    const tileUrl = cartoKey
      ? `https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png?key=${cartoKey}`
      : 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'

    const attribution = cartoKey
      ? '&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      : '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

    L.tileLayer(tileUrl, {
      attribution,
      maxZoom: 19,
    }).addTo(map)

    const heatmapGroup = L.layerGroup().addTo(map)
    const markersGroup = L.layerGroup().addTo(map)
    mapInstanceRef.current = map
    markersGroupRef.current = markersGroup
    heatmapGroupRef.current = heatmapGroup

    return () => {
      map.remove()
      mapInstanceRef.current = null
      markersGroupRef.current = null
      heatmapGroupRef.current = null
    }
  }, [])

  // Update Simulated Heatmap Layer
  useEffect(() => {
    if (!mapInstanceRef.current || !heatmapGroupRef.current) return

    heatmapGroupRef.current.clearLayers()

    if (isSimulationActive && showHeatmap && simulationData?.heatmap_points) {
      simulationData.heatmap_points.forEach((pt) => {
        // Outer broad plume (gradient halo)
        const outerCircle = L.circle([pt.latitude, pt.longitude], {
          radius: pt.radius_km * 1000,
          color: pt.color,
          fillColor: pt.color,
          fillOpacity: Math.min(0.25, pt.intensity * 0.32),
          weight: 1,
        })

        // Inner core marker
        const innerCore = L.circleMarker([pt.latitude, pt.longitude], {
          radius: Math.max(5, Math.min(16, pt.intensity * 18)),
          color: '#FFFFFF',
          fillColor: pt.color,
          fillOpacity: 0.85,
          weight: 1.5,
        })

        const popupContent = `
          <div style="font-family: sans-serif; background: #FFFFFF; color: #192B3C; min-width: 175px; padding: 6px; border-radius: 6px; border: 1px solid #DCE5EF;">
            <div style="font-weight: 700; font-size: 13px; color: #15304B; margin-bottom: 2px;">${pt.label}</div>
            <div style="font-size: 11px; color: #64758A; margin-bottom: 6px;">${pt.region} • Simulated Hazard</div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px;">
              <span style="color: #64758A;">Simulated Hs:</span>
              <span style="font-weight: 700; color: ${pt.color}; font-family: monospace;">${pt.swh_m.toFixed(2)} m</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px;">
              <span style="color: #64758A;">Peak Period:</span>
              <span style="font-weight: 600; font-family: monospace;">${pt.mwp_s.toFixed(1)} s</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px;">
              <span style="color: #64758A;">Wind Gusts:</span>
              <span style="font-weight: 600; font-family: monospace;">${pt.wind_kmh.toFixed(1)} km/h</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 12px;">
              <span style="color: #64758A;">Risk Level:</span>
              <span style="font-weight: 700; color: ${pt.color};">${pt.risk_level}</span>
            </div>
          </div>
        `
        innerCore.bindPopup(popupContent)
        outerCircle.bindPopup(popupContent)

        heatmapGroupRef.current?.addLayer(outerCircle)
        heatmapGroupRef.current?.addLayer(innerCore)
      })
    }
  }, [isSimulationActive, showHeatmap, simulationData])

  // Update Leaflet Markers when nodes change
  useEffect(() => {
    if (!mapInstanceRef.current || !markersGroupRef.current) return

    markersGroupRef.current.clearLayers()

    filteredNodes.forEach((node) => {
      const isSelected = node.location_id === selectedNodeId
      const color = node.marker_hex_color

      // Outer pulse ring (halo)
      const halo = L.circleMarker([node.latitude, node.longitude], {
        radius: isSelected ? 18 : 12,
        fillColor: color,
        fillOpacity: isSelected ? 0.35 : 0.2,
        color: color,
        weight: isSelected ? 2.5 : 1.5,
      })

      // Inner center dot
      const marker = L.circleMarker([node.latitude, node.longitude], {
        radius: isSelected ? 8 : 6,
        fillColor: color,
        fillOpacity: 0.95,
        color: '#FFFFFF',
        weight: 1.5,
      })

      // High-contrast clean popup
      const popupContent = `
        <div style="font-family: sans-serif; background: #FFFFFF; color: #192B3C; min-width: 190px; padding: 6px; border-radius: 6px; border: 1px solid #DCE5EF;">
          <div style="font-weight: 700; font-size: 13px; margin-bottom: 2px; color: #15304B;">${node.name}</div>
          <div style="font-size: 11px; color: #64758A; margin-bottom: 6px; font-family: monospace;">${node.coordinate_label} • ${node.region}</div>
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
            <span style="color: #64758A;">Current Risk:</span>
            <span style="font-weight: 700; color: ${color};">${node.current_risk_level}</span>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
            <span style="color: #64758A;">Observed (Hs):</span>
            <span style="font-weight: 600; font-family: monospace; color: #15304B;">${node.latest_obs_swh_m != null ? `${Number(node.latest_obs_swh_m).toFixed(2)} m` : 'N/A'}</span>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
            <span style="color: #64758A;">+${node.forecast_horizon_hours}h Forecast:</span>
            <span style="font-weight: 700; color: #245B8F; font-family: monospace;">${node.forecast_swh_m != null ? `${Number(node.forecast_swh_m).toFixed(2)} m` : 'N/A'}</span>
          </div>
          ${
            node.is_reference
              ? `<div style="background: #EAF2F9; color: #245B8F; font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px; margin-top: 6px; text-align: center; border: 1px solid #DCECF9;">REFERENCE VALIDATED</div>`
              : `<div style="background: #FEF3C7; color: #92400E; font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px; margin-top: 6px; text-align: center; border: 1px solid #FCD34D;">EXPERIMENTAL REGIONAL NODE</div>`
          }
        </div>
      `
      marker.bindPopup(popupContent)

      const onMarkerClick = () => {
        setSelectedNodeId(node.location_id)
      }
      marker.on('click', onMarkerClick)
      halo.on('click', onMarkerClick)

      markersGroupRef.current?.addLayer(halo)
      markersGroupRef.current?.addLayer(marker)
    })
  }, [filteredNodes, selectedNodeId])

  const handleNodeCardClick = (lid: string, lat: number, lon: number) => {
    setSelectedNodeId(lid)
    mapInstanceRef.current?.flyTo([lat, lon], 7.5, { duration: 1.0 })
  }

  return (
    <div className="space-y-6 pb-8">
      {/* Top Banner & Scientific Disclaimer */}
      <div className="ocean-panel bg-white p-6 rounded-2xl border border-[#DCE5EF] shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#EAF2F9] text-[#245B8F] border border-[#DCECF9]">
                <MapPin className="h-4 w-4" />
              </span>
              <h1 className="text-xl font-bold tracking-tight text-[#15304B] font-['Outfit',sans-serif]">
                Interactive Coastal Risk Heatmap
              </h1>
            </div>
            <p className="mt-1 text-xs text-[#64758A] max-w-3xl leading-relaxed">
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

        {/* Notice on Spatial Interpolation (tested by vitest) */}
        <div className="mt-4 rounded-lg bg-[#F0F6FC] p-3 border border-[#DCECF9] text-xs text-[#64758A] flex items-start gap-2.5">
          <Info className="h-4 w-4 text-[#20AFA7] shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-[#15304B]">Scientific Integrity Policy:</span> Point-forecast markers are rendered strictly at registered offshore/port coordinates. Continuous contour surface interpolation between sparse coastal points is scientifically disabled to prevent misleading intermediate hazard claims.
          </div>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 bg-white p-4 rounded-xl border border-[#DCE5EF] shadow-xs">
        {/* Horizon Selector */}
        <div>
          <label className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] font-semibold block mb-1.5">
            Forecast Horizon
          </label>
          <div className="grid grid-cols-3 gap-1 bg-[#F0F6FC] p-1 rounded-lg border border-[#DCE5EF]">
            {[1, 6, 24].map((h) => (
              <button
                key={h}
                onClick={() => setHorizon(h as Horizon)}
                className={`py-1.5 text-xs font-semibold rounded-md transition-colors ${
                  horizon === h
                    ? 'bg-white text-[#15304B] shadow-xs font-bold border border-[#DCE5EF]'
                    : 'text-[#64758A] hover:text-[#15304B]'
                }`}
              >
                +{h}h
              </button>
            ))}
          </div>
        </div>

        {/* Telemetry Stream */}
        <div>
          <label className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] font-semibold block mb-1.5">
            Data Stream
          </label>
          <div className="grid grid-cols-2 gap-1 bg-[#F0F6FC] p-1 rounded-lg border border-[#DCE5EF]">
            <button
              onClick={() => setSource('demo')}
              className={`py-1.5 text-xs font-semibold rounded-md transition-colors ${
                source === 'demo'
                  ? 'bg-white text-[#245B8F] border border-[#DCE5EF] shadow-xs font-bold'
                  : 'text-[#64758A] hover:text-[#15304B]'
              }`}
            >
              ERA5 Demo
            </button>
            <button
              onClick={() => setSource('live')}
              className={`py-1.5 text-xs font-semibold rounded-md transition-colors ${
                source === 'live'
                  ? 'bg-white text-[#245B8F] border border-[#DCE5EF] shadow-xs font-bold'
                  : 'text-[#64758A] hover:text-[#15304B]'
              }`}
            >
              Open-Meteo
            </button>
          </div>
        </div>

        {/* Risk Level Filter */}
        <div>
          <label className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] font-semibold block mb-1.5">
            Filter Risk Category
          </label>
          <select
            value={selectedRiskFilter}
            onChange={(e) => setSelectedRiskFilter(e.target.value)}
            className="w-full bg-[#F0F6FC] border border-[#DCE5EF] rounded-lg px-3 py-2 text-xs text-[#15304B] font-mono focus:outline-none focus:border-[#20AFA7]"
          >
            <option value="ALL">All Categories</option>
            <option value="RED">Critical Risk (≥ 4.0m)</option>
            <option value="ORANGE">High Risk (≥ 2.5m)</option>
            <option value="YELLOW">Caution (≥ 1.5m)</option>
            <option value="GREEN">Low Hazard (&lt; 1.5m)</option>
          </select>
        </div>

        {/* Alerts Only Toggle & Simulation Trigger */}
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 cursor-pointer bg-[#F0F6FC] p-2 rounded-lg border border-[#DCE5EF] text-xs text-[#15304B] flex-1">
            <input
              type="checkbox"
              checked={alertsOnly}
              onChange={(e) => setAlertsOnly(e.target.checked)}
              className="h-4 w-4 rounded border-[#DCE5EF] accent-[#20AFA7] focus:ring-0 bg-white"
            />
            <span className="font-semibold text-[11px] whitespace-nowrap">Active Alerts Only</span>
          </label>

          <Button
            size="sm"
            onClick={openModal}
            className="bg-linear-to-r from-[#245B8F] to-[#1D4A75] hover:from-[#1D4A75] text-white text-xs font-semibold px-3 py-2 rounded-lg flex items-center gap-1.5 shadow-2xs whitespace-nowrap"
          >
            <FlaskConical className="h-3.5 w-3.5" />
            <span>{isSimulationActive ? 'Scenario' : 'Simulate'}</span>
          </Button>
        </div>
      </div>

      {/* Main Map + Synchronized Drawer Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Leaflet Interactive Map Container (8 Cols) */}
        <div className="lg:col-span-8 ocean-panel bg-white rounded-2xl overflow-hidden border border-[#DCE5EF] shadow-xs relative">
          <HazardHeatmapLegend
            showHeatmap={showHeatmap}
            onToggleHeatmap={() => setShowHeatmap(!showHeatmap)}
          />

          <div
            ref={mapContainerRef}
            className="h-[560px] w-full z-10"
            style={{ background: '#EAF2F9' }}
          />

          {/* Map floating legend */}
          <div className="absolute bottom-4 left-4 z-20 bg-white/95 border border-[#DCE5EF] p-3 rounded-lg backdrop-blur-md shadow-md text-xs space-y-1.5 font-mono">
            <span className="text-[10px] uppercase font-bold text-[#64758A] block tracking-wider">Risk Threshold Legend</span>
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-[#10B981]" />
              <span className="text-[#192B3C]">&lt; 1.5m (Safe Green)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-[#EAB308]" />
              <span className="text-[#192B3C]">1.5m - 2.5m (Caution Yellow)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-[#F97316]" />
              <span className="text-[#192B3C]">2.5m - 4.0m (High Orange)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-[#EF4444]" />
              <span className="text-[#192B3C]">≥ 4.0m (Critical Red)</span>
            </div>
          </div>
        </div>

        {/* Right: Selected Node Inspection Drawer (4 Cols) */}
        <div className="lg:col-span-4 space-y-4">
          {selectedNode ? (
            <div className="ocean-panel bg-white border border-[#DCE5EF] rounded-2xl p-5 shadow-xs space-y-4">
              <div className="flex items-start justify-between border-b border-[#DCE5EF] pb-3">
                <div>
                  <h3 className="font-['Outfit',sans-serif] text-base font-bold text-[#15304B]">
                    {selectedNode.name}
                  </h3>
                  <p className="text-xs font-mono text-[#64758A]">
                    {selectedNode.coordinate_label}
                  </p>
                </div>
                <Badge
                  variant={selectedNode.current_risk_level === 'GREEN' ? 'teal' : selectedNode.current_risk_level === 'YELLOW' ? 'warning' : 'destructive'}
                  className="font-mono text-[10px]"
                >
                  {selectedNode.current_risk_level}
                </Badge>
              </div>

              {/* Station metrics */}
              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div className="p-3 rounded-lg bg-[#F0F6FC] border border-[#DCECF9]">
                  <span className="text-[10px] text-[#64758A] font-semibold block">OBSERVED SWH</span>
                  <span className="text-lg font-bold text-[#15304B]">
                    {selectedNode.latest_obs_swh_m != null ? `${Number(selectedNode.latest_obs_swh_m).toFixed(2)}m` : 'N/A'}
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-[#F0F6FC] border border-[#DCECF9]">
                  <span className="text-[10px] text-[#64758A] font-semibold block">+{horizon}h FORECAST</span>
                  <span className="text-lg font-bold text-[#245B8F]">
                    {selectedNode.forecast_swh_m != null ? `${Number(selectedNode.forecast_swh_m).toFixed(2)}m` : 'N/A'}
                  </span>
                </div>
              </div>

              {/* Multi-horizon projections */}
              <div className="space-y-1.5 pt-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] font-semibold block">
                  Multi-Horizon Forecasts (m)
                </span>
                <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
                  <div className="p-2 rounded bg-[#F0F6FC] border border-[#DCE5EF]">
                    <span className="text-[10px] text-[#64758A] block">+1h</span>
                    <span className="text-[#15304B] font-bold">{selectedNode.multi_horizon_predictions?.[1]?.toFixed(2) ?? '—'}</span>
                  </div>
                  <div className="p-2 rounded bg-[#F0F6FC] border border-[#DCE5EF]">
                    <span className="text-[10px] text-[#64758A] block">+6h</span>
                    <span className="text-[#20AFA7] font-bold">{selectedNode.multi_horizon_predictions?.[6]?.toFixed(2) ?? '—'}</span>
                  </div>
                  <div className="p-2 rounded bg-[#F0F6FC] border border-[#DCE5EF]">
                    <span className="text-[10px] text-[#64758A] block">+24h</span>
                    <span className="text-[#15304B] font-bold">{selectedNode.multi_horizon_predictions?.[24]?.toFixed(2) ?? '—'}</span>
                  </div>
                </div>
              </div>

              {/* Station classification */}
              <div className="p-3 rounded-lg bg-[#F0F6FC] border border-[#DCE5EF] text-xs">
                {selectedNode.is_reference ? (
                  <div className="flex items-center gap-2 text-[#0D837C] font-medium">
                    <ShieldCheck className="h-4 w-4 text-[#20AFA7]" />
                    <span>Primary Reference Model Station</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-[#B45309] font-medium">
                    <AlertTriangle className="h-4 w-4 text-[#D97706]" />
                    <span>Experimental Bathymetric Transfer Node</span>
                  </div>
                )}
              </div>
            </div>
          ) : null}

          {/* Registered Nodes Quick Navigator */}
          <div className="ocean-panel bg-white border border-[#DCE5EF] rounded-2xl p-4 shadow-xs space-y-2 max-h-[280px] overflow-y-auto">
            <span className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] font-bold block px-1 pb-1 border-b border-[#DCE5EF]">
              Registered Coastal Nodes ({filteredNodes.length})
            </span>
            <div className="space-y-1.5 pt-1">
              {filteredNodes.map((n) => (
                <div
                  key={n.location_id}
                  onClick={() => handleNodeCardClick(n.location_id, n.latitude, n.longitude)}
                  className={`p-2.5 rounded-lg border text-xs cursor-pointer transition-colors flex items-center justify-between ${
                    selectedNodeId === n.location_id
                      ? 'bg-[#EAF2F9] border-[#79B7E5] text-[#15304B] font-semibold'
                      : 'bg-white border-[#DCE5EF] text-[#192B3C] hover:bg-[#F0F6FC]'
                  }`}
                >
                  <div>
                    <span className="font-semibold block">{n.name}</span>
                    <span className="text-[10px] font-mono text-[#64758A]">{n.coordinate_label}</span>
                  </div>
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: n.marker_hex_color }}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

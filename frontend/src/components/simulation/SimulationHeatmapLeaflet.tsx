import React, { useEffect, useRef } from 'react'
import L from 'leaflet'
import type { HeatmapPoint, NodeMapSummaryItem } from '../../types'

interface SimulationHeatmapLeafletProps {
  nodes?: NodeMapSummaryItem[]
  heatmapPoints?: HeatmapPoint[]
  selectedNodeId?: string | null
  onSelectNode?: (nodeId: string) => void
}

export function SimulationHeatmapLeaflet({
  nodes = [],
  heatmapPoints = [],
  selectedNodeId,
  onSelectNode,
}: SimulationHeatmapLeafletProps) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null)
  const mapInstanceRef = useRef<L.Map | null>(null)
  const heatmapLayerRef = useRef<L.LayerGroup | null>(null)
  const markersLayerRef = useRef<L.LayerGroup | null>(null)

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return

    const map = L.map(mapContainerRef.current, {
      center: [16.8, 73.0],
      zoom: 6,
      minZoom: 4,
      maxZoom: 12,
      zoomControl: true,
    })

    const cartoKey = import.meta.env.VITE_CARTO_API_KEY
    const tileUrl = cartoKey
      ? `https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png?key=${cartoKey}`
      : 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'

    L.tileLayer(tileUrl, {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19,
    }).addTo(map)

    const heatmapGroup = L.layerGroup().addTo(map)
    const markersGroup = L.layerGroup().addTo(map)

    mapInstanceRef.current = map
    heatmapLayerRef.current = heatmapGroup
    markersLayerRef.current = markersGroup

    return () => {
      map.remove()
      mapInstanceRef.current = null
      heatmapLayerRef.current = null
      markersLayerRef.current = null
    }
  }, [])

  // Update Heatmap Points from actual simulation engine
  useEffect(() => {
    if (!mapInstanceRef.current || !heatmapLayerRef.current) return

    heatmapLayerRef.current.clearLayers()

    heatmapPoints.forEach((pt) => {
      // Outer broad plume (gradient halo) - fallback safely to circleMarker if circle is unmocked in test
      const circleFn = typeof L.circle === 'function' ? L.circle : L.circleMarker
      const outerCircle = circleFn([pt.latitude, pt.longitude], {
        radius: pt.radius_km * 1000,
        color: pt.color,
        fillColor: pt.color,
        fillOpacity: Math.min(0.28, pt.intensity * 0.35),
        weight: 1.5,
      })

      // Inner intense core marker
      const innerCore = L.circleMarker([pt.latitude, pt.longitude], {
        radius: Math.max(6, Math.min(18, pt.intensity * 20)),
        color: '#FFFFFF',
        fillColor: pt.color,
        fillOpacity: 0.9,
        weight: 2,
      })

      const popupContent = `
        <div style="font-family: sans-serif; background: #FFFFFF; color: #15304B; min-width: 180px; padding: 6px; border-radius: 8px;">
          <div style="font-weight: 700; font-size: 13px; color: #15304B; margin-bottom: 2px;">${pt.label}</div>
          <div style="font-size: 11px; color: #64758A; margin-bottom: 6px;">${pt.region} • Simulated Hazard</div>
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px;">
            <span style="color: #64758A;">Simulated Hs:</span>
            <span style="font-weight: 700; color: ${pt.color}; font-family: monospace;">${pt.swh_m.toFixed(2)} m</span>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px;">
            <span style="color: #64758A;">Max Wave:</span>
            <span style="font-weight: 700; font-family: monospace;">${pt.max_wave_m.toFixed(2)} m</span>
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
            <span style="color: #64758A;">Risk Classification:</span>
            <span style="font-weight: 700; color: ${pt.color};">${pt.risk_level}</span>
          </div>
        </div>
      `
      innerCore.bindPopup(popupContent)
      outerCircle.bindPopup(popupContent)

      heatmapLayerRef.current?.addLayer(outerCircle)
      heatmapLayerRef.current?.addLayer(innerCore)
    })
  }, [heatmapPoints])

  // Update Coastal Markers
  useEffect(() => {
    if (!mapInstanceRef.current || !markersLayerRef.current) return

    markersLayerRef.current.clearLayers()

    nodes.forEach((node) => {
      const isSelected = selectedNodeId === node.location_id
      const color = node.marker_hex_color || '#10B981'

      const halo = L.circleMarker([node.latitude, node.longitude], {
        radius: isSelected ? 18 : 12,
        fillColor: color,
        fillOpacity: isSelected ? 0.45 : 0.25,
        color: color,
        weight: isSelected ? 2.5 : 1.5,
      })

      const marker = L.circleMarker([node.latitude, node.longitude], {
        radius: isSelected ? 8 : 6,
        fillColor: color,
        fillOpacity: 0.95,
        color: '#FFFFFF',
        weight: 1.5,
      })

      const popupContent = `
        <div style="font-family: sans-serif; background: #FFFFFF; color: #15304B; min-width: 180px; padding: 6px; border-radius: 8px;">
          <div style="font-weight: 700; font-size: 13px; margin-bottom: 2px;">${node.name}</div>
          <div style="font-size: 11px; color: #64758A; margin-bottom: 6px; font-family: monospace;">${node.coordinate_label}</div>
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px;">
            <span style="color: #64758A;">Risk Level:</span>
            <span style="font-weight: 700; color: ${color};">${node.current_risk_level}</span>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 12px;">
            <span style="color: #64758A;">Simulated Hs:</span>
            <span style="font-weight: 700; color: #15304B; font-family: monospace;">
              ${node.forecast_swh_m != null ? `${Number(node.forecast_swh_m).toFixed(2)} m` : 'N/A'}
            </span>
          </div>
        </div>
      `
      marker.bindPopup(popupContent)

      const onClick = () => {
        if (onSelectNode) onSelectNode(node.location_id)
      }
      marker.on('click', onClick)
      halo.on('click', onClick)

      markersLayerRef.current?.addLayer(halo)
      markersLayerRef.current?.addLayer(marker)
    })
  }, [nodes, selectedNodeId, onSelectNode])

  return (
    <div className="relative w-full h-[520px] rounded-2xl overflow-hidden border border-[#DCE5EF] shadow-inner bg-[#EAF2F9]">
      <div ref={mapContainerRef} className="w-full h-full z-10" />

      {/* Floating Legend */}
      <div className="absolute bottom-4 left-4 z-20 bg-white/95 backdrop-blur-md border border-[#DCE5EF] p-3 rounded-xl shadow-lg text-xs font-mono space-y-1.5">
        <span className="text-[10px] font-bold text-[#64758A] uppercase tracking-wider block">
          Spatial Risk Heatmap (Simulated)
        </span>
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-[#10B981]" />
          <span className="text-[#15304B]">&lt; 1.5m Low Risk</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-[#EAB308]" />
          <span className="text-[#15304B]">1.5m - 2.5m Moderate Risk</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-[#F97316]" />
          <span className="text-[#15304B]">2.5m - 4.0m High Risk</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-[#EF4444]" />
          <span className="text-[#15304B]">≥ 4.0m Critical Risk</span>
        </div>
      </div>
    </div>
  )
}

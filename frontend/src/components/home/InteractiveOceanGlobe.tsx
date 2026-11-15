import React, { useRef, useState, useMemo, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import * as THREE from 'three'
import { MapPin, Waves, Compass, ShieldCheck, AlertCircle, ArrowUpRight, Globe2, Radio, Play, Pause } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../../services/api'
import type { CoastalNode } from '../../types'
import { WireframeDottedGlobeMesh } from './WireframeDottedGlobe'

const GLOBE_RADIUS = 3.2

// Convert latitude and longitude to 3D Cartesian coordinates on sphere
function latLonToVector3(lat: number, lon: number, radius: number): [number, number, number] {
  const phi = (90 - lat) * (Math.PI / 180)
  // Center India (~78° E) towards the front camera
  const theta = (lon + 195) * (Math.PI / 180)

  const x = -(radius * Math.sin(phi) * Math.cos(theta))
  const z = radius * Math.sin(phi) * Math.sin(theta)
  const y = radius * Math.cos(phi)

  return [x, y, z]
}

interface NodeMarkerProps {
  node: CoastalNode
  isSelected: boolean
  onSelect: (node: CoastalNode) => void
}

function NodeMarker({ node, isSelected, onSelect }: NodeMarkerProps) {
  const position = useMemo(() => {
    return latLonToVector3(node.latitude, node.longitude, GLOBE_RADIUS + 0.05)
  }, [node.latitude, node.longitude])

  const [hovered, setHovered] = useState(false)
  const isReference = node.is_reference

  const markerColor = isSelected ? '#35B8AD' : isReference ? '#10B981' : '#79B7E5'

  return (
    <group position={position}>
      {/* Outer pulsing ring */}
      <mesh
        onClick={(e) => {
          e.stopPropagation()
          onSelect(node)
        }}
        onPointerOver={(e) => {
          e.stopPropagation()
          setHovered(true)
        }}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[isSelected || hovered ? 0.12 : 0.08, 16, 16]} />
        <meshBasicMaterial
          color={markerColor}
          transparent
          opacity={isSelected ? 1.0 : hovered ? 0.9 : 0.75}
        />
      </mesh>

      {/* Floating HTML label on hover or selection */}
      {(hovered || isSelected) && (
        <Html distanceFactor={8} position={[0, 0.2, 0]} center>
          <div className="bg-[#102A43]/95 backdrop-blur-md border border-[#35B8AD]/40 text-white text-[11px] font-sans px-2.5 py-1 rounded-md shadow-lg pointer-events-none whitespace-nowrap">
            <span className="font-bold">{node.name.split('(')[0].trim()}</span>
            {isReference && <span className="ml-1.5 text-[9px] text-[#35B8AD] font-mono">REF</span>}
          </div>
        </Html>
      )}
    </group>
  )
}

function EarthGlobe({
  nodes,
  selectedNode,
  onSelectNode,
}: {
  nodes: CoastalNode[]
  selectedNode: CoastalNode | null
  onSelectNode: (node: CoastalNode) => void
}) {
  const globeRef = useRef<THREE.Group>(null)

  // Graticule grid lines (Latitude & Longitude)
  const gridLines = useMemo(() => {
    const lines: THREE.BufferGeometry[] = []

    // Latitude circles
    for (let lat = -60; lat <= 60; lat += 20) {
      const phi = (90 - lat) * (Math.PI / 180)
      const r = GLOBE_RADIUS * Math.sin(phi)
      const y = GLOBE_RADIUS * Math.cos(phi)
      const points: THREE.Vector3[] = []

      for (let i = 0; i <= 64; i++) {
        const theta = (i / 64) * Math.PI * 2
        points.push(new THREE.Vector3(r * Math.cos(theta), y, r * Math.sin(theta)))
      }
      lines.push(new THREE.BufferGeometry().setFromPoints(points))
    }

    // Longitude circles
    for (let lon = 0; lon < 360; lon += 30) {
      const points: THREE.Vector3[] = []
      const rad = (lon * Math.PI) / 180
      for (let i = 0; i <= 64; i++) {
        const lat = -80 + (i / 64) * 160
        const phi = (90 - lat) * (Math.PI / 180)
        const x = -(GLOBE_RADIUS * Math.sin(phi) * Math.cos(rad))
        const z = GLOBE_RADIUS * Math.sin(phi) * Math.sin(rad)
        const y = GLOBE_RADIUS * Math.cos(phi)
        points.push(new THREE.Vector3(x, y, z))
      }
      lines.push(new THREE.BufferGeometry().setFromPoints(points))
    }

    return lines
  }, [])

  return (
    <group ref={globeRef}>
      {/* Base Ocean Sphere */}
      <mesh receiveShadow>
        <sphereGeometry args={[GLOBE_RADIUS, 64, 64]} />
        <meshStandardMaterial
          color="#102A43"
          roughness={0.45}
          metalness={0.15}
          emissive="#0a1a2c"
          emissiveIntensity={0.35}
        />
      </mesh>

      {/* Atmospheric glow envelope */}
      <mesh>
        <sphereGeometry args={[GLOBE_RADIUS * 1.025, 48, 48]} />
        <meshBasicMaterial
          color="#35B8AD"
          transparent
          opacity={0.08}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Lat/Lon Graticule Lines */}
      {gridLines.map((geom, idx) => (
        <lineLoop key={idx} geometry={geom}>
          <lineBasicMaterial color="#246B9E" transparent opacity={0.25} />
        </lineLoop>
      ))}

      {/* Indian Coastline Outline Anchor Accent (Equator / Tropic of Cancer) */}
      <mesh position={[0, 0, 0]} rotation={[0, 0, 0]}>
        <ringGeometry args={[GLOBE_RADIUS * 1.002, GLOBE_RADIUS * 1.006, 64]} />
        <meshBasicMaterial color="#35B8AD" transparent opacity={0.15} side={THREE.DoubleSide} />
      </mesh>

      {/* Render 8 Coastal Nodes on the 3D Sphere */}
      {nodes.map((node) => (
        <NodeMarker
          key={node.location_id}
          node={node}
          isSelected={selectedNode?.location_id === node.location_id}
          onSelect={onSelectNode}
        />
      ))}
    </group>
  )
}

export function InteractiveOceanGlobe() {
  const [nodes, setNodes] = useState<CoastalNode[]>([])
  const [selectedNode, setSelectedNode] = useState<CoastalNode | null>(null)
  const [loading, setLoading] = useState(true)
  const [globeMode, setGlobeMode] = useState<'bathymetric' | 'wireframe'>('bathymetric')
  const [autoRotate, setAutoRotate] = useState(true)

  useEffect(() => {
    let isMounted = true
    const fetchNodes = async () => {
      try {
        const data = await api.getCoastalLocations()
        if (isMounted && data && data.length > 0) {
          setNodes(data)
          // Default to Mumbai Reference
          const ref = data.find((n) => n.is_reference) || data[0]
          setSelectedNode(ref)
        }
      } catch (err) {
        console.warn('Failed to fetch coastal locations for 3D Globe:', err)
      } finally {
        if (isMounted) setLoading(false)
      }
    }
    fetchNodes()
    return () => {
      isMounted = false
    }
  }, [])

  return (
    <div className="relative w-full rounded-3xl bg-gradient-to-b from-[#0e243a] via-[#102A43] to-[#0a1829] border border-[#246B9E]/30 overflow-hidden shadow-2xl">
      {/* Background radial accent */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-[#35B8AD]/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-96 h-96 bg-[#246B9E]/15 rounded-full blur-3xl pointer-events-none" />

      {/* Top Banner Ribbon */}
      <div className="border-b border-white/10 px-6 py-2.5 flex items-center justify-between text-[11px] font-mono text-[#8CC8E8]/80 bg-[#102A43]/50 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <Radio className="h-3.5 w-3.5 text-[#35B8AD] animate-pulse" />
          <span>GLOBAL OCEAN OBSERVATION MESH</span>
        </div>
        <div className="hidden sm:flex items-center gap-4">
          <span>8 STATIONS TRACKED</span>
          <span>&bull;</span>
          <span className="text-[#35B8AD]">ERA5 CALIBRATED REFERENCE: MUMBAI</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 items-center min-h-[620px]">
        {/* Left Side: 3D Canvas (7 Cols) */}
        <div className="lg:col-span-7 h-[420px] lg:h-[620px] w-full relative cursor-grab active:cursor-grabbing">
          {/* Globe Mode Switcher & Auto-Rotate Controls */}
          <div className="absolute top-4 left-6 z-10 flex items-center gap-2">
            <div className="flex items-center p-1 rounded-xl bg-[#102A43]/90 border border-[#246B9E]/40 backdrop-blur-md shadow-lg">
              <button
                onClick={() => setGlobeMode('bathymetric')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  globeMode === 'bathymetric'
                    ? 'bg-[#246B9E] text-white shadow-xs font-semibold'
                    : 'text-[#8CC8E8] hover:text-white'
                }`}
              >
                Bathymetric Shaded
              </button>
              <button
                onClick={() => setGlobeMode('wireframe')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  globeMode === 'wireframe'
                    ? 'bg-[#35B8AD] text-[#102A43] shadow-xs font-semibold'
                    : 'text-[#8CC8E8] hover:text-white'
                }`}
              >
                Wireframe Dotted
              </button>
            </div>

            <button
              onClick={() => setAutoRotate(!autoRotate)}
              title={autoRotate ? 'Pause Rotation' : 'Resume Rotation'}
              className="p-2 rounded-xl bg-[#102A43]/90 border border-[#246B9E]/40 backdrop-blur-md text-[#8CC8E8] hover:text-white hover:border-[#35B8AD]/40 transition-all shadow-lg"
            >
              {autoRotate ? (
                <Pause className="h-3.5 w-3.5 text-[#35B8AD]" />
              ) : (
                <Play className="h-3.5 w-3.5 text-[#8CC8E8]" />
              )}
            </button>
          </div>

          <Canvas
            camera={{ position: [0, 1.2, 7.8], fov: 45 }}
            dpr={[1, 1.5]}
            gl={{ antialias: true, alpha: true }}
          >
            <ambientLight intensity={0.65} />
            <directionalLight position={[10, 15, 12]} intensity={1.5} color="#eef8ff" />
            <pointLight position={[-10, -5, -8]} intensity={0.5} color="#35B8AD" />

            {globeMode === 'wireframe' ? (
              <group>
                <WireframeDottedGlobeMesh
                  radius={GLOBE_RADIUS}
                  pointsCount={1400}
                  color="#246B9E"
                  accentColor="#35B8AD"
                />
                {nodes.map((node) => (
                  <NodeMarker
                    key={node.location_id}
                    node={node}
                    isSelected={selectedNode?.location_id === node.location_id}
                    onSelect={setSelectedNode}
                  />
                ))}
              </group>
            ) : (
              <EarthGlobe
                nodes={nodes}
                selectedNode={selectedNode}
                onSelectNode={setSelectedNode}
              />
            )}

            <OrbitControls
              enableZoom={false}
              enablePan={false}
              autoRotate={autoRotate}
              autoRotateSpeed={0.6}
              minPolarAngle={Math.PI / 4}
              maxPolarAngle={(Math.PI * 3) / 4}
            />
          </Canvas>

          {/* Helper Badge */}
          <div className="absolute bottom-4 left-6 z-10 flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#102A43]/80 border border-[#246B9E]/40 backdrop-blur-md text-xs text-[#8CC8E8]">
            <Compass className="h-3.5 w-3.5 text-[#35B8AD] animate-spin" style={{ animationDuration: '8s' }} />
            <span>
              {globeMode === 'wireframe'
                ? 'Scientific Dotted Grid &bull; Click node to inspect telemetry'
                : 'Drag to rotate globe &bull; Click node to inspect telemetry'}
            </span>
          </div>
        </div>

        {/* Right Side: Node Details Card (5 Cols) */}
        <div className="lg:col-span-5 p-6 lg:p-10 z-10 flex flex-col justify-center">
          {selectedNode ? (
            <div className="space-y-6 animate-ocean-fade">
              {/* Header Badge */}
              <div className="flex items-center gap-2.5">
                <span className="flex h-2.5 w-2.5 rounded-full bg-[#35B8AD] animate-pulse" />
                <span className="text-xs font-mono font-semibold uppercase tracking-wider text-[#35B8AD]">
                  {selectedNode.is_reference ? 'Primary Reference Station' : 'Regional Coastal Candidate'}
                </span>
              </div>

              {/* Node Title & Coordinates */}
              <div>
                <h3 className="text-2xl lg:text-3xl font-bold font-['Outfit',sans-serif] text-white tracking-tight">
                  {selectedNode.name}
                </h3>
                <div className="flex items-center gap-2 mt-2 text-sm text-[#8CC8E8] font-mono">
                  <MapPin className="h-4 w-4 text-[#35B8AD] shrink-0" />
                  <span>{selectedNode.coordinate_label}</span>
                  <span>&bull;</span>
                  <span>{selectedNode.region}</span>
                </div>
              </div>

              {/* Status Chips */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-white/5 border border-white/10 backdrop-blur-md">
                  <div className="text-[11px] text-[#8CC8E8] uppercase tracking-wider">Calibration</div>
                  <div className="text-sm font-semibold text-white mt-0.5 flex items-center gap-1.5">
                    {selectedNode.is_reference ? (
                      <>
                        <ShieldCheck className="h-4 w-4 text-[#10B981]" />
                        <span>ERA5 Calibrated</span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="h-4 w-4 text-[#F59E0B]" />
                        <span>Regional Transfer</span>
                      </>
                    )}
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-white/5 border border-white/10 backdrop-blur-md">
                  <div className="text-[11px] text-[#8CC8E8] uppercase tracking-wider">Telemetry Origin</div>
                  <div className="text-sm font-semibold text-white mt-0.5 flex items-center gap-1.5">
                    <Waves className="h-4 w-4 text-[#35B8AD]" />
                    <span>Open-Meteo REST</span>
                  </div>
                </div>
              </div>

              {/* Description */}
              <p className="text-sm text-[#8CC8E8]/90 leading-relaxed">
                {selectedNode.description ||
                  'Candidate offshore numerical wave forecasting node in the Arabian Sea grid (~5 to 25km offshore).'}
              </p>

              {/* Notes */}
              {selectedNode.notes && selectedNode.notes.length > 0 && (
                <div className="space-y-1.5 text-xs text-[#8CC8E8]/75 font-mono">
                  {selectedNode.notes.slice(0, 2).map((note, i) => (
                    <div key={i} className="flex items-start gap-2">
                      <span className="text-[#35B8AD] font-bold">&gt;</span>
                      <span>{note}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Node selector buttons */}
              <div className="pt-2">
                <label className="block text-[11px] font-mono text-[#8CC8E8] uppercase mb-2">
                  Select from 8 Monitored Stations:
                </label>
                <div className="flex flex-wrap gap-1.5">
                  {nodes.map((n) => (
                    <button
                      key={n.location_id}
                      onClick={() => setSelectedNode(n)}
                      className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                        selectedNode.location_id === n.location_id
                          ? 'bg-[#35B8AD] text-[#102A43] font-bold shadow-md'
                          : 'bg-white/10 hover:bg-white/20 text-white border border-white/10'
                      }`}
                    >
                      {n.name.split(' ')[0]}
                    </button>
                  ))}
                </div>
              </div>

              {/* Action Link to Dashboard Map */}
              <div className="pt-4 border-t border-white/10">
                <Link
                  to="/map"
                  className="inline-flex items-center gap-2 text-sm font-semibold text-[#35B8AD] hover:text-white transition-colors"
                >
                  <span>Open Station on Interactive Coastal Map</span>
                  <ArrowUpRight className="h-4 w-4" />
                </Link>
              </div>
            </div>
          ) : (
            <div className="text-center text-[#8CC8E8] py-12">
              <Compass className="h-8 w-8 text-[#35B8AD] mx-auto animate-spin mb-3" />
              <p className="text-sm">Connecting to coastal node telemetry...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

import React, { useRef, useMemo, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import {
  RotateCcw,
  Activity,
  Layers,
  ShieldAlert,
  Wind,
  Compass,
  AlertTriangle,
  ZoomIn,
  ZoomOut,
  Maximize2,
} from 'lucide-react'
import type { HeatmapPoint, NodeMapSummaryItem } from '../../types'

interface SimulationOcean3DProps {
  swh?: number
  windKmh?: number
  mwpSeconds?: number
  scenarioName?: string
  severityCategory?: string
  currentStep?: number
  nodes?: NodeMapSummaryItem[]
  heatmapPoints?: HeatmapPoint[]
  onSelectNode?: (nodeId: string) => void
}

// -----------------------------------------------------------------------------
// 1. DYNAMIC PROCEDURAL GERSTNER HAZARD WAVE COMPONENT
// -----------------------------------------------------------------------------

function HazardOceanMesh({
  swh = 2.5,
  windKmh = 40,
  mwpSeconds = 10,
  severity = 'CRITICAL',
}: {
  swh: number
  windKmh: number
  mwpSeconds: number
  severity: string
}) {
  const meshRef = useRef<THREE.Mesh>(null)

  // Calibrate amplitude based on significant wave height (1.0m to 10.0m)
  const amplitudeScale = useMemo(() => {
    return Math.max(0.4, Math.min(2.8, swh * 0.28))
  }, [swh])

  // Frequency inversely proportional to wave period
  const waveFrequency = useMemo(() => {
    return Math.max(0.08, Math.min(0.35, 1.8 / Math.max(6, mwpSeconds)))
  }, [mwpSeconds])

  // Speed scales with wind speed
  const waveSpeed = useMemo(() => {
    return Math.max(0.8, Math.min(3.2, (windKmh / 50.0) * 1.5))
  }, [windKmh])

  // Custom vertex & fragment shaders with dynamic wave deformation & foam
  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uAmplitude: { value: amplitudeScale },
      uFrequency: { value: waveFrequency },
      uSpeed: { value: waveSpeed },
      uDeepColor: { value: new THREE.Color(severity === 'CRITICAL' ? '#081726' : '#0B2545') },
      uShallowColor: { value: new THREE.Color(severity === 'CRITICAL' ? '#1D4A75' : '#20AFA7') },
      uFoamColor: { value: new THREE.Color('#FFFFFF') },
      uHazardIntensity: { value: swh > 5.0 ? 1.0 : swh > 2.5 ? 0.6 : 0.2 },
    }),
    [severity]
  )

  // Update uniforms when props change
  uniforms.uAmplitude.value = amplitudeScale
  uniforms.uFrequency.value = waveFrequency
  uniforms.uSpeed.value = waveSpeed
  uniforms.uHazardIntensity.value = swh > 5.0 ? 1.0 : swh > 2.5 ? 0.6 : 0.2

  useFrame((state) => {
    const t = state.clock.getElapsedTime()
    uniforms.uTime.value = t
  })

  // Vertex Shader for physical Gerstner trochoidal ocean deformation
  const vertexShader = `
    uniform float uTime;
    uniform float uAmplitude;
    uniform float uFrequency;
    uniform float uSpeed;
    uniform float uHazardIntensity;
    
    varying vec3 vNormal;
    varying vec3 vPosition;
    varying float vWaveHeight;
    varying float vFoam;

    void main() {
      vPosition = position;
      vec3 pos = position;

      float t = uTime * uSpeed;
      
      // Primary long-crested hazard swell
      float w1 = sin(pos.x * uFrequency + t * 1.2) * cos(pos.y * (uFrequency * 0.6) + t * 0.4);
      
      // Secondary cross-chop swell (wind driven)
      float w2 = sin(pos.x * (uFrequency * 1.8) - t * 1.6 + 1.2) * sin(pos.y * (uFrequency * 1.4) + t * 0.9);
      
      // Tertiary micro-turbulent crest peak
      float w3 = cos((pos.x + pos.y) * (uFrequency * 2.8) + t * 2.2);

      float totalDisplacement = (w1 * 0.65 + w2 * 0.25 + w3 * 0.10) * uAmplitude * 4.0;
      
      // Crest steepening (trochoidal pinch)
      float pinch = pow(abs(w1 * 0.5 + 0.5), 2.2) * uHazardIntensity;
      pos.z += totalDisplacement + pinch * 1.5;

      vWaveHeight = pos.z;
      vFoam = clamp((pos.z - (uAmplitude * 2.0)) / (uAmplitude * 2.2), 0.0, 1.0);
      
      // Calculate analytical normal
      vec3 calculatedNormal = normalize(vec3(
        -cos(pos.x * uFrequency + t) * uAmplitude * 0.6,
        -sin(pos.y * uFrequency + t) * uAmplitude * 0.6,
        1.0
      ));
      vNormal = normalMatrix * calculatedNormal;

      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
  `

  // Fragment Shader for realistic Fresnel reflections, depth gradient, and foam
  const fragmentShader = `
    uniform vec3 uDeepColor;
    uniform vec3 uShallowColor;
    uniform vec3 uFoamColor;
    uniform float uHazardIntensity;

    varying vec3 vNormal;
    varying vec3 vPosition;
    varying float vWaveHeight;
    varying float vFoam;

    void main() {
      vec3 viewDir = normalize(-vPosition);
      float fresnel = pow(1.0 - max(dot(viewDir, normalize(vNormal)), 0.0), 3.5);

      // Height-based depth color interpolation
      float heightFactor = clamp((vWaveHeight + 3.0) / 7.0, 0.0, 1.0);
      vec3 waterColor = mix(uDeepColor, uShallowColor, heightFactor * 0.7);
      
      // Add Fresnel reflection highlight
      waterColor = mix(waterColor, vec3(0.65, 0.85, 0.95), fresnel * 0.55);

      // Foam overlay on highest crests
      waterColor = mix(waterColor, uFoamColor, vFoam * (0.4 + uHazardIntensity * 0.45));

      gl_FragColor = vec4(waterColor, 0.94);
    }
  `

  return (
    <mesh ref={meshRef} rotation={[-Math.PI / 2.2, 0, 0]} position={[0, -0.8, -2]} receiveShadow>
      <planeGeometry args={[75, 75, 128, 128]} />
      <shaderMaterial
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

// -----------------------------------------------------------------------------
// 2. 3D COASTAL NODES & FLOATING TELEMETRY BUOYS
// -----------------------------------------------------------------------------

function CoastalNodeMarker3D({
  node,
  isSelected,
  onSelect,
}: {
  node: NodeMapSummaryItem
  isSelected: boolean
  onSelect: () => void
}) {
  const markerRef = useRef<THREE.Group>(null)
  const [hovered, setHovered] = useState(false)

  // Map latitude/longitude along Indian West Coast to 3D grid space
  // Ref: Lat 8.5° to 22° N, Lon 70° to 77° E
  const position3D = useMemo<[number, number, number]>(() => {
    const x = (node.longitude - 72.8) * 4.2
    const z = -(node.latitude - 16.0) * 1.8
    const y = 0.4
    return [x, y, z]
  }, [node.latitude, node.longitude])

  const riskColor = useMemo(() => {
    const lvl = String(node.current_risk_level)
    switch (lvl) {
      case 'CRITICAL':
      case 'RED':
        return '#EF4444'
      case 'HIGH':
      case 'ORANGE':
        return '#F97316'
      case 'CAUTION':
      case 'YELLOW':
        return '#EAB308'
      default:
        return '#10B981'
    }
  }, [node.current_risk_level])

  // Bobbing animation for buoy
  useFrame((state) => {
    if (!markerRef.current) return
    const t = state.clock.getElapsedTime()
    markerRef.current.position.y = position3D[1] + Math.sin(t * 2.2 + position3D[0]) * 0.12
  })

  return (
    <group
      ref={markerRef}
      position={position3D}
      onClick={(e) => {
        e.stopPropagation()
        onSelect()
      }}
      onPointerOver={(e) => {
        e.stopPropagation()
        setHovered(true)
      }}
      onPointerOut={() => setHovered(false)}
    >
      {/* Outer Beacon Pulsing Ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.35, isSelected ? 0.75 : 0.55, 32]} />
        <meshBasicMaterial
          color={riskColor}
          transparent
          opacity={isSelected ? 0.8 : hovered ? 0.6 : 0.3}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Buoy Cylinder Core */}
      <mesh position={[0, 0.3, 0]}>
        <cylinderGeometry args={[0.16, 0.22, 0.6, 16]} />
        <meshStandardMaterial color={riskColor} roughness={0.3} metalness={0.6} />
      </mesh>

      {/* Mast & Warning Light */}
      <mesh position={[0, 0.75, 0]}>
        <sphereGeometry args={[0.09, 16, 16]} />
        <meshBasicMaterial color="#FFFFFF" />
      </mesh>

      {/* Interactive Tooltip Overlay */}
      {(hovered || isSelected) && (
        <Html position={[0, 1.2, 0]} center distanceFactor={14}>
          <div className="bg-white/95 backdrop-blur-md p-2.5 rounded-xl border border-[#DCE5EF] shadow-xl text-xs font-mono min-w-[170px] pointer-events-none animate-in fade-in zoom-in-95">
            <div className="font-bold text-[#15304B] font-['Outfit',sans-serif] leading-tight">
              {node.name}
            </div>
            <div className="text-[10px] text-[#64758A] mt-0.5">{node.coordinate_label}</div>
            <div className="mt-2 pt-2 border-t border-[#DCE5EF] flex items-center justify-between">
              <span className="text-[10px] text-[#64758A]">Simulated $H_s$:</span>
              <span className="font-bold text-[#15304B]">
                {node.forecast_swh_m != null ? `${Number(node.forecast_swh_m).toFixed(2)}m` : 'N/A'}
              </span>
            </div>
            <div className="flex items-center justify-between mt-1">
              <span className="text-[10px] text-[#64758A]">Hazard State:</span>
              <span className="font-bold text-[10px]" style={{ color: riskColor }}>
                {node.current_risk_level}
              </span>
            </div>
          </div>
        </Html>
      )}
    </group>
  )
}

// -----------------------------------------------------------------------------
// 3. 3D STORM EPICENTER & IMPACT PLUME RADIUS
// -----------------------------------------------------------------------------

function StormEpicenter3D({
  heatmapPoints = [],
  step = 6,
}: {
  heatmapPoints: HeatmapPoint[]
  step: number
}) {
  const epicenterRef = useRef<THREE.Group>(null)

  // Find the primary maximum hazard point
  const primaryPoint = useMemo(() => {
    if (!heatmapPoints.length) return null
    return heatmapPoints.reduce((max, p) => (p.swh_m > max.swh_m ? p : max), heatmapPoints[0])
  }, [heatmapPoints])

  const centerPos = useMemo<[number, number, number]>(() => {
    if (!primaryPoint) return [0, 0, -3]
    const x = (primaryPoint.longitude - 72.8) * 4.2
    const z = -(primaryPoint.latitude - 16.0) * 1.8
    return [x, 0.1, z]
  }, [primaryPoint])

  useFrame((state) => {
    if (!epicenterRef.current) return
    const t = state.clock.getElapsedTime()
    epicenterRef.current.rotation.y = t * 0.4
  })

  if (!primaryPoint) return null

  return (
    <group position={centerPos} ref={epicenterRef}>
      {/* Broad Outer Swell Reach Ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]}>
        <ringGeometry args={[3.2, 3.4, 64]} />
        <meshBasicMaterial color="#EF4444" transparent opacity={0.25} side={THREE.DoubleSide} />
      </mesh>

      {/* Middle Concentric Hazard Reach */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.03, 0]}>
        <ringGeometry args={[1.8, 1.95, 48]} />
        <meshBasicMaterial color="#F97316" transparent opacity={0.4} side={THREE.DoubleSide} />
      </mesh>

      {/* Core Epicenter Vortex */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <ringGeometry args={[0.6, 0.85, 32]} />
        <meshBasicMaterial color="#EF4444" transparent opacity={0.7} side={THREE.DoubleSide} />
      </mesh>

      {/* Storm Core Vortex Column */}
      <mesh position={[0, 1.8, 0]}>
        <cylinderGeometry args={[0.08, 0.6, 3.5, 16, 1, true]} />
        <meshBasicMaterial color="#EF4444" transparent opacity={0.18} side={THREE.DoubleSide} />
      </mesh>
    </group>
  )
}

// -----------------------------------------------------------------------------
// 4. MAIN THREE.JS SIMULATION SCENE CONTROLLER
// -----------------------------------------------------------------------------

function CameraRig({ isResetting, onResetComplete }: { isResetting: boolean; onResetComplete: () => void }) {
  const { camera } = useThree()
  const initialPos = useMemo(() => new THREE.Vector3(0, 7.5, 12), [])

  useFrame(() => {
    if (isResetting) {
      camera.position.lerp(initialPos, 0.08)
      camera.lookAt(0, 0, -1.5)
      if (camera.position.distanceTo(initialPos) < 0.1) {
        onResetComplete()
      }
    }
  })

  return null
}

export function SimulationOcean3D({
  swh = 2.5,
  windKmh = 45,
  mwpSeconds = 11,
  scenarioName = 'Simulation Scenario',
  severityCategory = 'CRITICAL',
  currentStep = 6,
  nodes = [],
  heatmapPoints = [],
  onSelectNode,
}: SimulationOcean3DProps) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [isResetting, setIsResetting] = useState(false)
  const [zoomLevel, setZoomLevel] = useState(1)

  const handleNodeClick = (nodeId: string) => {
    setSelectedNodeId(nodeId)
    if (onSelectNode) onSelectNode(nodeId)
  }

  return (
    <div className="relative w-full h-[520px] rounded-2xl overflow-hidden bg-linear-to-b from-[#0F2236] to-[#081524] border border-[#DCE5EF] shadow-inner">
      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [0, 7.5, 12], fov: 45, near: 0.1, far: 1000 }}
        shadows
        className="w-full h-full cursor-grab active:cursor-grabbing"
      >
        {/* Dynamic Storm Atmospheric Lighting */}
        <ambientLight intensity={severityCategory === 'CRITICAL' ? 0.35 : 0.65} color="#DCE5EF" />
        <directionalLight
          position={[12, 18, 10]}
          intensity={severityCategory === 'CRITICAL' ? 0.7 : 1.3}
          color="#EAF2F9"
          castShadow
        />
        <pointLight position={[-10, 6, -5]} intensity={0.4} color="#20AFA7" />

        {/* Camera Reset Rig */}
        <CameraRig isResetting={isResetting} onResetComplete={() => setIsResetting(false)} />

        {/* Procedural Gerstner Waves */}
        <HazardOceanMesh
          swh={swh}
          windKmh={windKmh}
          mwpSeconds={mwpSeconds}
          severity={severityCategory}
        />

        {/* 3D Storm Epicenter & Plumes */}
        <StormEpicenter3D heatmapPoints={heatmapPoints} step={currentStep} />

        {/* Coastal Monitoring Nodes */}
        {nodes.map((node) => (
          <CoastalNodeMarker3D
            key={node.location_id}
            node={node}
            isSelected={selectedNodeId === node.location_id}
            onSelect={() => handleNodeClick(node.location_id)}
          />
        ))}
      </Canvas>

      {/* Top Left: Scenario HUD Overlay */}
      <div className="absolute top-4 left-4 z-20 pointer-events-none flex flex-col gap-1.5 font-mono text-xs">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#0F2236]/85 backdrop-blur-md border border-[#DCE5EF]/30 text-white shadow-lg">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#EF4444] opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-[#EF4444]"></span>
          </span>
          <span className="font-bold font-['Outfit',sans-serif] text-sm tracking-wide text-white">
            {scenarioName}
          </span>
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#EF4444]/30 text-[#FCA5A5] border border-[#EF4444]/40 font-bold">
            {severityCategory}
          </span>
        </div>

        <div className="inline-flex items-center gap-3 px-3 py-1 rounded-md bg-[#0F2236]/75 backdrop-blur-md border border-[#DCE5EF]/20 text-white/80 text-[11px]">
          <span>Timeline: <strong className="text-[#35B8AD]">T+{String(currentStep).padStart(2, '0')}h / 24h</strong></span>
          <span>Wave $H_s$: <strong className="text-[#EF4444]">{swh.toFixed(2)}m</strong></span>
          <span>Wind: <strong className="text-[#E2E8F0]">{windKmh.toFixed(0)} km/h</strong></span>
        </div>
      </div>

      {/* Top Right: Interactive Camera Controls */}
      <div className="absolute top-4 right-4 z-20 flex items-center gap-1.5 bg-[#0F2236]/85 backdrop-blur-md border border-[#DCE5EF]/30 p-1 rounded-lg shadow-lg">
        <button
          onClick={() => setIsResetting(true)}
          className="p-1.5 rounded text-white/80 hover:text-white hover:bg-white/10 transition-colors"
          title="Reset Camera View"
          aria-label="Reset Camera View"
        >
          <RotateCcw className="h-4 w-4" />
        </button>
      </div>

      {/* Bottom Floating Legend Bar */}
      <div className="absolute bottom-4 left-4 right-4 z-20 flex flex-wrap items-center justify-between gap-2 p-2.5 rounded-xl bg-[#0F2236]/90 backdrop-blur-md border border-[#DCE5EF]/30 text-white text-[11px] font-mono shadow-xl">
        <div className="flex items-center gap-4">
          <span className="text-white/60 uppercase font-semibold text-[10px]">3D Coastal Risk Status:</span>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-[#10B981]" />
            <span className="text-white/80">Safe (&lt;1.5m)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-[#EAB308]" />
            <span className="text-white/80">Caution (1.5-2.5m)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-[#F97316]" />
            <span className="text-white/80">High (2.5-4.0m)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-[#EF4444]" />
            <span className="text-white/80 font-bold">Critical (≥4.0m)</span>
          </div>
        </div>

        <div className="text-[10px] text-white/50">
          Interactive View • Click buoy to inspect • Drag to orbit
        </div>
      </div>
    </div>
  )
}

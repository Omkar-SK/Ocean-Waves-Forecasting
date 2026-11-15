import React, { useRef, useMemo, useEffect, useState, useCallback } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import {
  RotateCcw,
  Layers,
  Activity,
  Radio,
  Waves,
  ShieldCheck,
  Compass,
  ArrowUpRight,
} from 'lucide-react'

// ============================================================================
// 1. GERSTNER WAVE MATHEMATICS & PHYSICS MODEL
// ============================================================================

export interface WaveDisplacement {
  x: number
  y: number
  z: number
  slopeX: number
  slopeZ: number
}

interface HarmonicWave {
  dir: [number, number]
  steepness: number
  wavelength: number
}

// 6 Multi-frequency harmonic wave components matching oceanic dispersion
export const HARMONIC_WAVES: HarmonicWave[] = [
  { dir: [1.0, 0.35], steepness: 0.16, wavelength: 26.0 },   // Deep ocean primary swell
  { dir: [-0.6, 0.8], steepness: 0.12, wavelength: 15.0 },   // Secondary cross-swell
  { dir: [0.4, 0.9], steepness: 0.09, wavelength: 9.0 },     // Wind-sea
  { dir: [-0.8, -0.4], steepness: 0.06, wavelength: 5.2 },   // Surface chop
  { dir: [0.9, -0.3], steepness: 0.035, wavelength: 3.0 },   // Capillary ripple
  { dir: [-0.2, 0.95], steepness: 0.025, wavelength: 1.8 },  // Surface micro-texture
]

/**
 * Calculates instantaneous 3D Gerstner wave displacement and slopes for any world coordinate.
 * Used for physically synchronizing floating sensor buoys, datum rods, and particle fields
 * identically with the GPU vertex shader.
 */
export function calculateWaveDisplacement(
  x: number,
  z: number,
  time: number,
  speedMultiplier: number = 1.0
): WaveDisplacement {
  let dx = 0
  let dy = 0
  let dz = 0
  let slopeX = 0
  let slopeZ = 0
  const t = time * speedMultiplier

  for (let i = 0; i < HARMONIC_WAVES.length; i++) {
    const w = HARMONIC_WAVES[i]
    const len = Math.hypot(w.dir[0], w.dir[1])
    const dX = w.dir[0] / len
    const dZ = w.dir[1] / len
    const k = (2.0 * Math.PI) / w.wavelength
    const c = Math.sqrt(9.80665 / k)
    const f = k * (dX * x + dZ * z - c * t * 0.45)
    const a = w.steepness / k

    dx += dX * a * Math.cos(f)
    dy += a * Math.sin(f)
    dz += dZ * a * Math.cos(f)

    slopeX += a * k * Math.cos(f) * dX
    slopeZ += a * k * Math.cos(f) * dZ
  }

  return {
    x: x + dx,
    y: dy,
    z: z + dz,
    slopeX,
    slopeZ,
  }
}

// ============================================================================
// 2. GLSL OCEAN SHADERS
// ============================================================================

const oceanVertexShader = `
  uniform float uTime;
  uniform float uSpeedMultiplier;
  varying vec3 vWorldPosition;
  varying vec3 vNormal;
  varying float vWaveHeight;
  varying vec2 vUv;

  vec3 gerstnerWave(vec4 wave, vec3 p, inout vec3 tangent, inout vec3 binormal, float time) {
    float steepness = wave.z;
    float wavelength = wave.w;
    float k = 2.0 * 3.14159265359 / wavelength;
    float c = sqrt(9.80665 / k);
    vec2 d = normalize(wave.xy);
    float f = k * (dot(d, p.xz) - c * time * 0.45);
    float a = steepness / k;

    tangent += vec3(
      -d.x * d.x * (steepness * sin(f)),
      d.x * (steepness * cos(f)),
      -d.x * d.y * (steepness * sin(f))
    );
    binormal += vec3(
      -d.x * d.y * (steepness * sin(f)),
      d.y * (steepness * cos(f)),
      -d.y * d.y * (steepness * sin(f))
    );

    return vec3(
      d.x * (a * cos(f)),
      a * sin(f),
      d.y * (a * cos(f))
    );
  }

  void main() {
    vUv = uv;
    vec3 p = position;
    vec3 tangent = vec3(1.0, 0.0, 0.0);
    vec3 binormal = vec3(0.0, 0.0, 1.0);
    float t = uTime * uSpeedMultiplier;

    // 6 Harmonic wave vectors
    vec3 w1 = gerstnerWave(vec4(1.0, 0.35, 0.16, 26.0), p, tangent, binormal, t);
    vec3 w2 = gerstnerWave(vec4(-0.6, 0.8, 0.12, 15.0), p, tangent, binormal, t);
    vec3 w3 = gerstnerWave(vec4(0.4, 0.9, 0.09, 9.0), p, tangent, binormal, t);
    vec3 w4 = gerstnerWave(vec4(-0.8, -0.4, 0.06, 5.2), p, tangent, binormal, t);
    vec3 w5 = gerstnerWave(vec4(0.9, -0.3, 0.035, 3.0), p, tangent, binormal, t);
    vec3 w6 = gerstnerWave(vec4(-0.2, 0.95, 0.025, 1.8), p, tangent, binormal, t);

    vec3 finalPos = p + w1 + w2 + w3 + w4 + w5 + w6;
    vec3 norm = normalize(cross(binormal, tangent));

    vNormal = norm;
    vWorldPosition = (modelMatrix * vec4(finalPos, 1.0)).xyz;
    vWaveHeight = finalPos.y;

    gl_Position = projectionMatrix * viewMatrix * vec4(vWorldPosition, 1.0);
  }
`

const oceanFragmentShader = `
  uniform vec3 uColorDeep;
  uniform vec3 uColorSurface;
  uniform vec3 uColorCrest;
  uniform vec3 uColorTeal;
  uniform vec3 uSunPosition;
  uniform vec3 uSunColor;
  uniform vec3 uCameraPos;
  uniform float uShowContours;
  uniform float uTime;

  varying vec3 vWorldPosition;
  varying vec3 vNormal;
  varying float vWaveHeight;
  varying vec2 vUv;

  void main() {
    vec3 viewDir = normalize(uCameraPos - vWorldPosition);
    vec3 normal = normalize(vNormal);

    // Fresnel reflectance approximation (Schlick's formulation)
    float fresnel = 0.03 + 0.97 * pow(1.0 - max(0.0, dot(viewDir, normal)), 4.5);

    // Sun directional specular reflection (sharp core + wide ocean glint)
    vec3 lightDir = normalize(uSunPosition - vWorldPosition);
    vec3 reflectDir = reflect(-lightDir, normal);
    float specSharp = pow(max(dot(viewDir, reflectDir), 0.0), 64.0);
    float specWide = pow(max(dot(viewDir, reflectDir), 0.0), 16.0);
    vec3 specular = uSunColor * (specSharp * 1.6 + specWide * 0.4);

    // Deep bathymetric color gradient based on wave height
    float heightFactor = smoothstep(-1.4, 1.4, vWaveHeight);
    vec3 waterColor = mix(uColorDeep, uColorSurface, heightFactor);

    // Shallow / crest teal refraction nuance
    float tealFactor = smoothstep(0.4, 1.2, vWaveHeight) * max(0.0, normal.y);
    waterColor = mix(waterColor, uColorTeal, tealFactor * 0.35);

    // Soft crest foam accents (highest peaks + high curvature slope)
    float crestCurvature = max(0.0, 1.0 - normal.y);
    float crestFactor = smoothstep(0.75, 1.45, vWaveHeight) * smoothstep(0.15, 0.45, crestCurvature);
    waterColor = mix(waterColor, uColorCrest, crestFactor * 0.55);

    // Environmental sky reflection
    vec3 skyColor = vec3(0.07, 0.18, 0.32);
    vec3 finalColor = mix(waterColor, skyColor, fresnel * 0.62) + specular;

    // Optional scientific bathymetric contour lines
    if (uShowContours > 0.5) {
      float contourFreq = 3.5;
      float contourVal = abs(fract(vWaveHeight * contourFreq - 0.5) - 0.5);
      float contourLine = 1.0 - smoothstep(0.0, 0.06, contourVal);
      finalColor += uColorTeal * contourLine * 0.35;
    }

    // Atmospheric horizon fog blending
    float dist = length(vWorldPosition.xz);
    float fog = smoothstep(22.0, 85.0, dist);
    vec3 fogColor = vec3(0.024, 0.075, 0.133); // Matches #061322

    gl_FragColor = vec4(mix(finalColor, fogColor, fog), 0.96);
  }
`

// ============================================================================
// 3. SCIENTIFIC MONITORING STATION DATA
// ============================================================================

export interface BuoyStation {
  id: string
  name: string
  stationCode: string
  anchor: [number, number]
  swh: number
  period: number
  status: 'calibrated' | 'active' | 'sentinel'
  depthM: number
  coordinateLabel: string
}

export const SCIENTIFIC_BUOYS: BuoyStation[] = [
  {
    id: 'station-alpha',
    name: 'Mumbai Deep Offshore (Ref)',
    stationCode: 'NODE-01-MUM',
    anchor: [-7.5, -4.0],
    swh: 0.68,
    period: 8.9,
    status: 'calibrated',
    depthM: 85,
    coordinateLabel: '18.50°N, 72.50°E',
  },
  {
    id: 'station-beta',
    name: 'Goa Shelf Sentinel',
    stationCode: 'NODE-02-GOA',
    anchor: [8.5, -7.5],
    swh: 0.74,
    period: 7.4,
    status: 'active',
    depthM: 62,
    coordinateLabel: '15.40°N, 73.70°E',
  },
  {
    id: 'station-gamma',
    name: 'Kochi Deep Basin Profiler',
    stationCode: 'NODE-04-KOC',
    anchor: [-1.2, -13.5],
    swh: 0.82,
    period: 9.5,
    status: 'sentinel',
    depthM: 110,
    coordinateLabel: '9.95°N, 76.15°E',
  },
]

// ============================================================================
// 4. THREE.JS SCENE SUBCOMPONENTS
// ============================================================================

interface OceanWaterSurfaceProps {
  showContours: boolean
  reducedMotion: boolean
  isVisible: boolean
}

function OceanWaterSurface({ showContours, reducedMotion, isVisible }: OceanWaterSurfaceProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const materialRef = useRef<THREE.ShaderMaterial>(null)
  const { camera } = useThree()

  const uniforms = useMemo(() => {
    return {
      uTime: { value: 0 },
      uSpeedMultiplier: { value: reducedMotion ? 0.08 : 0.85 },
      uColorDeep: { value: new THREE.Color('#061526') },
      uColorSurface: { value: new THREE.Color('#103559') },
      uColorCrest: { value: new THREE.Color('#d9eef9') },
      uColorTeal: { value: new THREE.Color('#20afa7') },
      uSunPosition: { value: new THREE.Vector3(35, 40, -45) },
      uSunColor: { value: new THREE.Color('#e6f2fc') },
      uCameraPos: { value: new THREE.Vector3(0, 4.2, 16.5) },
      uShowContours: { value: showContours ? 1.0 : 0.0 },
    }
  }, [reducedMotion, showContours])

  useEffect(() => {
    if (materialRef.current?.uniforms?.uShowContours) {
      materialRef.current.uniforms.uShowContours.value = showContours ? 1.0 : 0.0
    }
  }, [showContours])

  useEffect(() => {
    if (materialRef.current?.uniforms?.uSpeedMultiplier) {
      materialRef.current.uniforms.uSpeedMultiplier.value = reducedMotion ? 0.08 : 0.85
    }
  }, [reducedMotion])

  useFrame((_, delta) => {
    if (!isVisible || !materialRef.current?.uniforms) return
    materialRef.current.uniforms.uTime.value += delta
    materialRef.current.uniforms.uCameraPos.value.copy(camera.position)
  })

  // Cleanup geometry and shader material on unmount
  useEffect(() => {
    return () => {
      if (meshRef.current?.geometry?.dispose) {
        meshRef.current.geometry.dispose()
      }
      if (materialRef.current?.dispose) {
        materialRef.current.dispose()
      }
    }
  }, [])

  return (
    <mesh
      ref={meshRef}
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, -0.6, 0]}
      receiveShadow
    >
      <planeGeometry args={[140, 140, 150, 150]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={oceanVertexShader}
        fragmentShader={oceanFragmentShader}
        uniforms={uniforms}
        transparent
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

/**
 * Subsurface bathymetric grid plane positioned underneath the water surface.
 */
function BathymetricDepthGrid({ visible }: { visible: boolean }) {
  if (!visible) return null

  return (
    <group position={[0, -4.2, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      {/* Primary Bathymetric Wireframe Grid */}
      <mesh>
        <planeGeometry args={[120, 120, 30, 30]} />
        <meshBasicMaterial
          color="#1b4267"
          wireframe
          transparent
          opacity={0.25}
        />
      </mesh>

      {/* Subsurface Sea-floor Solid Plane */}
      <mesh position={[0, 0, -0.05]}>
        <planeGeometry args={[130, 130]} />
        <meshBasicMaterial color="#040c17" />
      </mesh>
    </group>
  )
}

/**
 * Subsurface ADCP Acoustic Drift Particles.
 */
function CurrentDriftParticles({ isVisible, reducedMotion }: { isVisible: boolean; reducedMotion: boolean }) {
  const pointsRef = useRef<THREE.Points>(null)
  const count = 220

  const [positions, phases, speeds] = useMemo(() => {
    const pos = new Float32Array(count * 3)
    const ph = new Float32Array(count)
    const sp = new Float32Array(count)
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 60
      pos[i * 3 + 1] = -(Math.random() * 3.2 + 0.6)
      pos[i * 3 + 2] = (Math.random() - 0.5) * 50 - 5
      ph[i] = Math.random() * Math.PI * 2
      sp[i] = 0.5 + Math.random() * 0.8
    }
    return [pos, ph, sp]
  }, [count])

  useFrame((state, delta) => {
    if (!pointsRef.current || !isVisible || reducedMotion) return
    const pos = pointsRef.current.geometry.attributes.position.array as Float32Array
    const time = state.clock.elapsedTime

    for (let i = 0; i < count; i++) {
      // Flow along swell vector (1.0, 0.35)
      pos[i * 3] += delta * speeds[i] * 0.9
      pos[i * 3 + 2] += delta * speeds[i] * 0.35
      pos[i * 3 + 1] += Math.sin(time * 0.8 + phases[i]) * 0.003

      // Wrap around within bounding box
      if (pos[i * 3] > 30) pos[i * 3] = -30
      if (pos[i * 3 + 2] > 20) pos[i * 3 + 2] = -30
    }
    pointsRef.current.geometry.attributes.position.needsUpdate = true
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.14}
        color="#35B8AD"
        transparent
        opacity={0.35}
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

/**
 * Floating Spar Sensor Buoy with real-time dynamic pitch, roll, and elevation.
 */
interface SensorBuoyProps {
  station: BuoyStation
  isSelected: boolean
  onSelect: (st: BuoyStation) => void
  reducedMotion: boolean
  isVisible: boolean
  showBadge: boolean
}

function SensorBuoy({
  station,
  isSelected,
  onSelect,
  reducedMotion,
  isVisible,
  showBadge,
}: SensorBuoyProps) {
  const groupRef = useRef<THREE.Group>(null)
  const ringRef = useRef<THREE.Mesh>(null)

  const statusColor =
    station.status === 'calibrated'
      ? '#10B981'
      : station.status === 'sentinel'
      ? '#35B8AD'
      : '#79B7E5'

  useFrame((state) => {
    if (!groupRef.current || !isVisible) return
    const t = state.clock.elapsedTime
    const disp = calculateWaveDisplacement(
      station.anchor[0],
      station.anchor[1],
      t,
      reducedMotion ? 0.08 : 0.85
    )

    groupRef.current.position.set(disp.x, disp.y, disp.z)
    // Dynamic pitch and roll matching local wave gradient
    groupRef.current.rotation.x = -disp.slopeZ * 0.75
    groupRef.current.rotation.z = disp.slopeX * 0.75

    if (ringRef.current) {
      const pulse = 1.0 + (Math.sin(t * 2.2 + station.anchor[0]) + 1.0) * 0.35
      ringRef.current.scale.set(pulse, pulse, 1)
    }
  })

  return (
    <group
      ref={groupRef}
      position={[station.anchor[0], 0, station.anchor[1]]}
      onClick={(e) => {
        e.stopPropagation()
        onSelect(station)
      }}
    >
      {/* Subsurface Mooring Cable */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array([0, 0, 0, 0, -3.8, 0]), 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#1b4267" transparent opacity={0.4} />
      </line>

      {/* Surface Expanding Ripple Ring */}
      <mesh ref={ringRef} position={[0, 0.02, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.3, 0.44, 24]} />
        <meshBasicMaterial
          color={statusColor}
          transparent
          opacity={0.35}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Buoy Flotation Collar Body */}
      <mesh position={[0, 0.04, 0]}>
        <cylinderGeometry args={[0.26, 0.32, 0.22, 16]} />
        <meshStandardMaterial color="#0f263e" roughness={0.3} metalness={0.7} />
      </mesh>

      {/* Solar Collector Ring */}
      <mesh position={[0, 0.16, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.2, 0.44, 16]} />
        <meshBasicMaterial color="#244b70" side={THREE.DoubleSide} />
      </mesh>

      {/* Telemetry Mast */}
      <mesh position={[0, 0.7, 0]}>
        <cylinderGeometry args={[0.02, 0.035, 1.1, 8]} />
        <meshStandardMaterial color="#88a2bd" metalness={0.8} roughness={0.2} />
      </mesh>

      {/* Top Flashing Beacon Sphere */}
      <mesh position={[0, 1.28, 0]}>
        <sphereGeometry args={[0.07, 12, 12]} />
        <meshStandardMaterial
          color={statusColor}
          emissive={statusColor}
          emissiveIntensity={isSelected ? 3.0 : 1.8}
        />
      </mesh>

      {/* Beacon Light */}
      <pointLight position={[0, 1.3, 0]} color={statusColor} intensity={0.7} distance={4} />

      {/* HTML Telemetry Label / Badge */}
      {showBadge && (
        <Html distanceFactor={14} position={[0, 1.7, 0]} center>
          <div
            onClick={() => onSelect(station)}
            className={`cursor-pointer px-2.5 py-1 rounded-lg backdrop-blur-md border text-[11px] font-mono whitespace-nowrap transition-all select-none shadow-xl ${
              isSelected
                ? 'bg-[#061526]/95 border-[#35B8AD] text-white ring-1 ring-[#35B8AD]/40 scale-105'
                : 'bg-[#061526]/80 border-white/20 text-[#8CC8E8] hover:text-white hover:border-[#35B8AD]/60'
            }`}
          >
            <div className="flex items-center gap-1.5 font-semibold">
              <span
                className="h-1.5 w-1.5 rounded-full animate-ping"
                style={{ backgroundColor: statusColor }}
              />
              <span>{station.stationCode}</span>
              <span className="text-white/60">({station.swh.toFixed(2)}m)</span>
            </div>
          </div>
        </Html>
      )}
    </group>
  )
}

/**
 * Vertical Scientific Wave-Height Calibration Staff.
 */
function ElevationDatumStaff({ isVisible, reducedMotion }: { isVisible: boolean; reducedMotion: boolean }) {
  const pointerRef = useRef<THREE.Mesh>(null)
  const staffAnchor = [-9.2, -4.5]

  useFrame((state) => {
    if (!pointerRef.current || !isVisible) return
    const t = state.clock.elapsedTime
    const disp = calculateWaveDisplacement(
      staffAnchor[0],
      staffAnchor[1],
      t,
      reducedMotion ? 0.08 : 0.85
    )
    pointerRef.current.position.y = disp.y
  })

  return (
    <group position={[staffAnchor[0], 0, staffAnchor[1]]}>
      {/* Vertical Calibration Rod (-2m to +2.5m) */}
      <mesh position={[0, 0.25, 0]}>
        <cylinderGeometry args={[0.03, 0.03, 4.5, 8]} />
        <meshStandardMaterial color="#47688a" metalness={0.7} roughness={0.3} />
      </mesh>

      {/* Reference Base Collar */}
      <mesh position={[0, -2.0, 0]}>
        <cylinderGeometry args={[0.3, 0.35, 0.4, 12]} />
        <meshStandardMaterial color="#0c1e30" />
      </mesh>

      {/* Floating Instantaneous Elevation Collar Pointer */}
      <mesh ref={pointerRef} position={[0, 0, 0]}>
        <cylinderGeometry args={[0.09, 0.09, 0.06, 12]} />
        <meshStandardMaterial color="#35B8AD" emissive="#35B8AD" emissiveIntensity={1.5} />
      </mesh>
    </group>
  )
}

/**
 * Camera Controller with autonomous gentle orbital breathing, cursor parallax,
 * scroll tilt integration, and smooth reset interpolation.
 */
interface CameraControllerProps {
  mousePos: { x: number; y: number }
  scrollY: number
  isResetting: boolean
  onResetComplete: () => void
  reducedMotion: boolean
  isVisible: boolean
}

function CinematicCameraController({
  mousePos,
  scrollY,
  isResetting,
  onResetComplete,
  reducedMotion,
  isVisible,
}: CameraControllerProps) {
  const { camera } = useThree()
  const initialPos = useMemo(() => new THREE.Vector3(0, 4.2, 16.5), [])
  const targetLookAt = useMemo(() => new THREE.Vector3(0, 0.6, -7.0), [])

  useFrame((state) => {
    if (!isVisible) return

    if (isResetting) {
      camera.position.lerp(initialPos, 0.08)
      if (camera.position.distanceTo(initialPos) < 0.05) {
        camera.position.copy(initialPos)
        onResetComplete()
      }
      camera.lookAt(targetLookAt)
      return
    }

    if (reducedMotion) {
      camera.position.lerp(initialPos, 0.05)
      camera.lookAt(targetLookAt)
      return
    }

    // Scroll factor tilts camera forward and slightly elevates perspective
    const scrollFactor = Math.min(scrollY / 700, 1.0)
    const autonomousX = Math.sin(state.clock.elapsedTime * 0.22) * 0.35
    const autonomousY = Math.cos(state.clock.elapsedTime * 0.18) * 0.22

    const targetX = mousePos.x * 2.4 + autonomousX
    const targetY = 4.2 + mousePos.y * 1.3 + autonomousY + scrollFactor * 1.6
    const targetZ = 16.5 - scrollFactor * 2.2

    camera.position.x = THREE.MathUtils.lerp(camera.position.x, targetX, 0.04)
    camera.position.y = THREE.MathUtils.lerp(camera.position.y, targetY, 0.04)
    camera.position.z = THREE.MathUtils.lerp(camera.position.z, targetZ, 0.04)

    const lookY = 0.6 - scrollFactor * 0.6
    camera.lookAt(0, lookY, -7.0)
  })

  return null
}

// ============================================================================
// 5. MAIN COMPONENT: UPGRADED 3D SCIENTIFIC OCEAN SCENE
// ============================================================================

export interface OceanScene3DProps {
  className?: string
  initialShowGrid?: boolean
  initialShowContours?: boolean
  initialShowBuoys?: boolean
}

export function OceanScene3D({
  className = '',
  initialShowGrid = true,
  initialShowContours = false,
  initialShowBuoys = true,
}: OceanScene3DProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })
  const [scrollY, setScrollY] = useState(0)
  const [isSupported, setIsSupported] = useState(true)
  const [isVisible, setIsVisible] = useState(true)
  const [reducedMotion, setReducedMotion] = useState(false)

  // Interactive Scene HUD Toggles
  const [showGrid, setShowGrid] = useState(initialShowGrid)
  const [showContours, setShowContours] = useState(initialShowContours)
  const [showBuoys, setShowBuoys] = useState(initialShowBuoys)
  const [selectedStation, setSelectedStation] = useState<BuoyStation | null>(null)
  const [isResetting, setIsResetting] = useState(false)

  // 1. Check WebGL and Reduced-Motion preference
  useEffect(() => {
    try {
      const canvas = document.createElement('canvas')
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl')
      if (!gl) setIsSupported(false)
    } catch {
      setIsSupported(false)
    }

    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    setReducedMotion(motionQuery.matches)
    const handleMotionChange = (e: MediaQueryListEvent) => setReducedMotion(e.matches)
    motionQuery.addEventListener('change', handleMotionChange)

    return () => motionQuery.removeEventListener('change', handleMotionChange)
  }, [])

  // 2. Track Window Scroll and Mouse Move
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const x = (e.clientX / window.innerWidth) * 2 - 1
      const y = -(e.clientY / window.innerHeight) * 2 + 1
      setMousePos({ x, y })
    }

    const handleScroll = () => {
      setScrollY(window.scrollY || window.pageYOffset || 0)
    }

    window.addEventListener('mousemove', handleMouseMove, { passive: true })
    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('scroll', handleScroll)
    }
  }, [])

  // 3. Pause Rendering when Off-Screen via IntersectionObserver
  useEffect(() => {
    const el = containerRef.current
    if (!el || typeof IntersectionObserver === 'undefined') return

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          setIsVisible(entry.isIntersecting)
        })
      },
      { threshold: 0.05 }
    )

    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const handleResetCamera = useCallback(() => {
    setMousePos({ x: 0, y: 0 })
    setIsResetting(true)
  }, [])

  if (!isSupported) {
    return (
      <div className="absolute inset-0 bg-gradient-to-b from-[#061322] via-[#0d2238] to-[#061322] overflow-hidden">
        <div className="absolute inset-0 opacity-20 bg-[radial-gradient(circle_at_50%_40%,#35B8AD,transparent_70%)]" />
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className={`absolute inset-0 w-full h-full select-none ${className}`}
    >
      {/* 3D WebGL Canvas */}
      <Canvas
        camera={{ position: [0, 4.2, 16.5], fov: 50 }}
        dpr={[1, 1.5]}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: 'high-performance',
        }}
      >
        <color attach="background" args={['#061322']} />
        <fog attach="fog" args={['#061322', 20, 85]} />

        {/* Cinematic Directional & Environmental Lighting */}
        <ambientLight intensity={0.5} />
        <directionalLight
          position={[35, 40, -45]}
          intensity={1.5}
          color="#e6f2fc"
          castShadow
        />
        <pointLight position={[0, 8, 4]} intensity={0.65} color="#35B8AD" />
        <pointLight position={[-15, 6, -10]} intensity={0.4} color="#246B9E" />

        {/* Camera Controller */}
        <CinematicCameraController
          mousePos={mousePos}
          scrollY={scrollY}
          isResetting={isResetting}
          onResetComplete={() => setIsResetting(false)}
          reducedMotion={reducedMotion}
          isVisible={isVisible}
        />

        {/* Procedural Gerstner Water Mesh */}
        <OceanWaterSurface
          showContours={showContours}
          reducedMotion={reducedMotion}
          isVisible={isVisible}
        />

        {/* Subsurface Bathymetric Depth Grid */}
        <BathymetricDepthGrid visible={showGrid} />

        {/* ADCP Current Drift Particles */}
        <CurrentDriftParticles isVisible={isVisible} reducedMotion={reducedMotion} />

        {/* Elevation Calibration Staff */}
        <ElevationDatumStaff isVisible={isVisible} reducedMotion={reducedMotion} />

        {/* Floating Scientific Sensor Buoys */}
        {showBuoys &&
          SCIENTIFIC_BUOYS.map((st) => (
            <SensorBuoy
              key={st.id}
              station={st}
              isSelected={selectedStation?.id === st.id}
              onSelect={(station) => setSelectedStation(station)}
              reducedMotion={reducedMotion}
              isVisible={isVisible}
              showBadge={showBuoys}
            />
          ))}
      </Canvas>

      {/* Floating Scientific HUD Controls (Bottom-Right) */}
      <div className="absolute bottom-6 right-6 z-20 flex flex-wrap items-center gap-2 pointer-events-auto">
        <button
          type="button"
          onClick={handleResetCamera}
          title="Reset Camera Framing"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#061526]/85 hover:bg-[#0c243d] text-white/80 hover:text-white border border-[#246B9E]/40 text-xs font-mono backdrop-blur-md shadow-lg transition-all hover:scale-105 active:scale-95"
        >
          <RotateCcw className="h-3.5 w-3.5 text-[#35B8AD]" />
          <span className="hidden sm:inline">Reset View</span>
        </button>

        <button
          type="button"
          onClick={() => setShowGrid(!showGrid)}
          title="Toggle Subsurface Bathymetric Depth Grid"
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-mono backdrop-blur-md shadow-lg transition-all hover:scale-105 active:scale-95 ${
            showGrid
              ? 'bg-[#35B8AD]/20 border-[#35B8AD] text-white'
              : 'bg-[#061526]/85 border-[#246B9E]/40 text-white/60 hover:text-white'
          }`}
        >
          <Layers className="h-3.5 w-3.5 text-[#35B8AD]" />
          <span className="hidden sm:inline">Depth Grid</span>
        </button>

        <button
          type="button"
          onClick={() => setShowContours(!showContours)}
          title="Toggle Wave Elevation Contour Lines"
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-mono backdrop-blur-md shadow-lg transition-all hover:scale-105 active:scale-95 ${
            showContours
              ? 'bg-[#35B8AD]/20 border-[#35B8AD] text-white'
              : 'bg-[#061526]/85 border-[#246B9E]/40 text-white/60 hover:text-white'
          }`}
        >
          <Activity className="h-3.5 w-3.5 text-[#35B8AD]" />
          <span className="hidden sm:inline">Contours</span>
        </button>

        <button
          type="button"
          onClick={() => setShowBuoys(!showBuoys)}
          title="Toggle Scientific Monitoring Buoys"
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-mono backdrop-blur-md shadow-lg transition-all hover:scale-105 active:scale-95 ${
            showBuoys
              ? 'bg-[#35B8AD]/20 border-[#35B8AD] text-white'
              : 'bg-[#061526]/85 border-[#246B9E]/40 text-white/60 hover:text-white'
          }`}
        >
          <Radio className="h-3.5 w-3.5 text-[#35B8AD]" />
          <span className="hidden sm:inline">Buoys</span>
        </button>
      </div>

      {/* Selected Sensor Buoy Telemetry Card Overlay */}
      {selectedStation && (
        <div className="absolute bottom-20 right-6 z-20 max-w-sm w-full bg-[#061526]/95 border border-[#35B8AD]/50 rounded-xl p-4 shadow-2xl backdrop-blur-lg pointer-events-auto text-white animate-ocean-fade">
          <div className="flex items-start justify-between gap-3 mb-2">
            <div>
              <div className="text-[10px] font-mono uppercase tracking-wider text-[#35B8AD] flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-[#10B981] animate-pulse" />
                <span>IN-SITU ACOUSTIC SENSOR PROBE</span>
              </div>
              <h4 className="text-sm font-semibold font-['Outfit',sans-serif] mt-0.5">
                {selectedStation.name}
              </h4>
            </div>
            <button
              type="button"
              onClick={() => setSelectedStation(null)}
              className="text-white/50 hover:text-white text-xs p-1"
            >
              ✕
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono my-3 bg-white/5 p-2 rounded-lg border border-white/10">
            <div>
              <span className="text-white/60 block text-[10px]">SIGNIFICANT WAVE</span>
              <span className="text-[#35B8AD] font-bold text-sm">
                {selectedStation.swh.toFixed(2)} m
              </span>
            </div>
            <div>
              <span className="text-white/60 block text-[10px]">WAVE PERIOD</span>
              <span className="text-white font-bold text-sm">
                {selectedStation.period.toFixed(1)} s
              </span>
            </div>
            <div>
              <span className="text-white/60 block text-[10px]">DEPTH DATUM</span>
              <span className="text-white font-medium">-{selectedStation.depthM} m</span>
            </div>
            <div>
              <span className="text-white/60 block text-[10px]">GRID COORD</span>
              <span className="text-white/80 text-[11px]">{selectedStation.coordinateLabel}</span>
            </div>
          </div>

          <div className="text-[11px] text-[#8CC8E8]/90 leading-relaxed font-sans">
            Calibrated waverider spar buoy feeding real-time Arabian Sea boundary conditions into the WaveLSTM model.
          </div>
        </div>
      )}
    </div>
  )
}

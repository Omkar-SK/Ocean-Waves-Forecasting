import React, { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { CoastalNode } from '../../types'

interface WireframeDottedGlobeProps {
  radius?: number
  pointsCount?: number
  color?: string
  accentColor?: string
  nodes?: CoastalNode[]
  selectedNodeId?: string | null
  onSelectNode?: (node: CoastalNode) => void
}

export function WireframeDottedGlobeMesh({
  radius = 3.2,
  pointsCount = 1200,
  color = '#246B9E',
  accentColor = '#35B8AD',
}: {
  radius?: number
  pointsCount?: number
  color?: string
  accentColor?: string
}) {
  const pointsRef = useRef<THREE.Points>(null)
  const ringsRef = useRef<THREE.Group>(null)

  // 1. Generate Fibonacci sphere dot matrix
  const { positions, colors } = useMemo(() => {
    const pos = new Float32Array(pointsCount * 3)
    const cols = new Float32Array(pointsCount * 3)
    const c1 = new THREE.Color(color)
    const c2 = new THREE.Color(accentColor)
    const phi = Math.PI * (3 - Math.sqrt(5)) // Golden angle

    for (let i = 0; i < pointsCount; i++) {
      const y = 1 - (i / (pointsCount - 1)) * 2 // y from 1 to -1
      const radiusAtY = Math.sqrt(1 - y * y) // Radius at y
      const theta = phi * i

      const x = Math.cos(theta) * radiusAtY
      const z = Math.sin(theta) * radiusAtY

      pos[i * 3] = x * radius
      pos[i * 3 + 1] = y * radius
      pos[i * 3 + 2] = z * radius

      // Give equatorial and Northern Indian Ocean region higher teal luminescence
      const isNorthernOcean = y > 0.1 && y < 0.6 && z > 0
      const mixed = isNorthernOcean ? c2 : c1

      cols[i * 3] = mixed.r
      cols[i * 3 + 1] = mixed.g
      cols[i * 3 + 2] = mixed.b
    }

    return { positions: pos, colors: cols }
  }, [pointsCount, radius, color, accentColor])

  // 2. Graticule Ring Geometry
  const ringGeometries = useMemo(() => {
    const geoms: THREE.BufferGeometry[] = []
    // 5 Latitude rings
    ;[-40, -20, 0, 20, 40].forEach((lat) => {
      const phi = (90 - lat) * (Math.PI / 180)
      const r = radius * Math.sin(phi)
      const y = radius * Math.cos(phi)
      const pts: THREE.Vector3[] = []
      for (let i = 0; i <= 64; i++) {
        const theta = (i / 64) * Math.PI * 2
        pts.push(new THREE.Vector3(r * Math.cos(theta), y, r * Math.sin(theta)))
      }
      geoms.push(new THREE.BufferGeometry().setFromPoints(pts))
    })
    return geoms
  }, [radius])

  useFrame((state) => {
    const t = state.clock.getElapsedTime()
    if (ringsRef.current) {
      ringsRef.current.rotation.y = t * 0.05
    }
  })

  return (
    <group>
      {/* Inner subtle core sphere */}
      <mesh>
        <sphereGeometry args={[radius * 0.985, 32, 32]} />
        <meshBasicMaterial color="#0A1829" />
      </mesh>

      {/* Atmospheric glowing rim */}
      <mesh>
        <sphereGeometry args={[radius * 1.035, 32, 32]} />
        <meshBasicMaterial color="#35B8AD" transparent opacity={0.06} side={THREE.BackSide} />
      </mesh>

      {/* Dotted Sphere Points */}
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[positions, 3]}
          />
          <bufferAttribute
            attach="attributes-color"
            args={[colors, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.055}
          vertexColors
          transparent
          opacity={0.85}
          sizeAttenuation
        />
      </points>

      {/* Equatorial & Latitude Wireframe Rings */}
      <group ref={ringsRef}>
        {ringGeometries.map((geom, idx) => (
          <lineLoop key={idx} geometry={geom}>
            <lineBasicMaterial color="#8CC8E8" transparent opacity={0.18} />
          </lineLoop>
        ))}
      </group>
    </group>
  )
}

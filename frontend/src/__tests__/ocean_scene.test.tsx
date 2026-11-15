import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import {
  calculateWaveDisplacement,
  HARMONIC_WAVES,
  SCIENTIFIC_BUOYS,
  OceanScene3D,
} from '../components/home/OceanScene3D'

// Mock Three.js Canvas and Drei Html in JSDOM test environment
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="mock-r3f-canvas">{children}</div>
  ),
  useFrame: vi.fn(),
  useThree: () => ({
    camera: {
      position: { x: 0, y: 4.2, z: 16.5, lerp: vi.fn(), copy: vi.fn(), distanceTo: vi.fn(() => 0) },
      lookAt: vi.fn(),
    },
  }),
}))

vi.mock('@react-three/drei', () => ({
  Html: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="mock-drei-html">{children}</div>
  ),
}))

describe('Upgraded 3D Ocean Experience & Wave Physics Engine', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Mock WebGL context in JSDOM for 3D scene testing
    HTMLCanvasElement.prototype.getContext = vi.fn().mockImplementation((type: string) => {
      if (type === 'webgl' || type === 'experimental-webgl') {
        return {}
      }
      return null
    })
  })

  describe('Gerstner Wave Physics Model (calculateWaveDisplacement)', () => {
    it('defines exactly 6 harmonic wave vectors with physically plausible parameters', () => {
      expect(HARMONIC_WAVES).toHaveLength(6)

      HARMONIC_WAVES.forEach((wave) => {
        // Steepness should be well below Stokes breaking limit (0.44)
        expect(wave.steepness).toBeGreaterThan(0.01)
        expect(wave.steepness).toBeLessThan(0.3)

        // Wavelengths should span from swell (26m) down to capillary ripples (1.8m)
        expect(wave.wavelength).toBeGreaterThan(1.0)
        expect(wave.wavelength).toBeLessThanOrEqual(30.0)

        // Direction vectors must be non-zero
        const mag = Math.hypot(wave.dir[0], wave.dir[1])
        expect(mag).toBeGreaterThan(0.1)
      })
    })

    it('calculates bounded elevation displacement and realistic slopes at origin', () => {
      const disp0 = calculateWaveDisplacement(0, 0, 0)

      // Vertical elevation should be within realistic SWH physical envelope
      expect(disp0.y).toBeGreaterThanOrEqual(-1.5)
      expect(disp0.y).toBeLessThanOrEqual(1.5)

      // Slopes should be smooth and differentiable
      expect(Number.isFinite(disp0.slopeX)).toBe(true)
      expect(Number.isFinite(disp0.slopeZ)).toBe(true)
    })

    it('demonstrates temporal propagation over time', () => {
      const dispT0 = calculateWaveDisplacement(5.0, 5.0, 0.0)
      const dispT1 = calculateWaveDisplacement(5.0, 5.0, 2.0)

      // Position should change as waves propagate
      expect(dispT0.y).not.toBe(dispT1.y)
      expect(dispT0.x).not.toBe(dispT1.x)
    })

    it('respects reduced-motion speed multiplier scaling', () => {
      const normalDisp = calculateWaveDisplacement(10.0, -10.0, 5.0, 1.0)
      const reducedDisp = calculateWaveDisplacement(10.0, -10.0, 5.0, 0.08)

      expect(Number.isFinite(normalDisp.y)).toBe(true)
      expect(Number.isFinite(reducedDisp.y)).toBe(true)
      expect(normalDisp.y).not.toEqual(reducedDisp.y)
    })
  })

  describe('Scientific Monitoring Station Metadata', () => {
    it('registers all 3 scientific sensor buoys with accurate coordinates and depths', () => {
      expect(SCIENTIFIC_BUOYS).toHaveLength(3)

      const refStation = SCIENTIFIC_BUOYS.find((b) => b.id === 'station-alpha')
      expect(refStation).toBeDefined()
      expect(refStation?.status).toBe('calibrated')
      expect(refStation?.swh).toBe(0.68)
      expect(refStation?.depthM).toBe(85)

      const goaStation = SCIENTIFIC_BUOYS.find((b) => b.id === 'station-beta')
      expect(goaStation).toBeDefined()
      expect(goaStation?.status).toBe('active')
      expect(goaStation?.swh).toBe(0.74)

      const kochiStation = SCIENTIFIC_BUOYS.find((b) => b.id === 'station-gamma')
      expect(kochiStation).toBeDefined()
      expect(kochiStation?.status).toBe('sentinel')
      expect(kochiStation?.swh).toBe(0.82)
    })
  })

  describe('OceanScene3D Component & Interactive HUD Controls', () => {
    it('renders the 3D Canvas container and scientific HUD control buttons', () => {
      render(<OceanScene3D />)

      expect(screen.getByTestId('mock-r3f-canvas')).toBeInTheDocument()
      expect(screen.getByTitle(/Reset Camera Framing/i)).toBeInTheDocument()
      expect(screen.getByTitle(/Toggle Subsurface Bathymetric Depth Grid/i)).toBeInTheDocument()
      expect(screen.getByTitle(/Toggle Wave Elevation Contour Lines/i)).toBeInTheDocument()
      expect(screen.getByTitle(/Toggle Scientific Monitoring Buoys/i)).toBeInTheDocument()
    })

    it('toggles Depth Grid button state on click', () => {
      render(<OceanScene3D />)

      const gridBtn = screen.getByTitle(/Toggle Subsurface Bathymetric Depth Grid/i)
      expect(gridBtn).toBeInTheDocument()

      // Default state has grid active
      expect(gridBtn).toHaveClass('border-[#35B8AD]')

      // Click toggles off
      fireEvent.click(gridBtn)
      expect(gridBtn).not.toHaveClass('border-[#35B8AD]')

      // Click toggles back on
      fireEvent.click(gridBtn)
      expect(gridBtn).toHaveClass('border-[#35B8AD]')
    })

    it('toggles Wave Contours button state on click', () => {
      render(<OceanScene3D />)

      const contourBtn = screen.getByTitle(/Toggle Wave Elevation Contour Lines/i)
      expect(contourBtn).toBeInTheDocument()

      // Default state is false
      expect(contourBtn).not.toHaveClass('border-[#35B8AD]')

      // Click toggles on
      fireEvent.click(contourBtn)
      expect(contourBtn).toHaveClass('border-[#35B8AD]')
    })

    it('triggers Reset View interaction without throwing errors', () => {
      render(<OceanScene3D />)

      const resetBtn = screen.getByTitle(/Reset Camera Framing/i)
      expect(resetBtn).toBeInTheDocument()

      expect(() => {
        fireEvent.click(resetBtn)
      }).not.toThrow()
    })

    it('renders graceful fallback gradient when WebGL is unavailable', () => {
      HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue(null)
      const { container } = render(<OceanScene3D />)
      expect(screen.queryByTestId('mock-r3f-canvas')).not.toBeInTheDocument()
      expect(container.querySelector('.bg-gradient-to-b')).toBeInTheDocument()
    })
  })
})

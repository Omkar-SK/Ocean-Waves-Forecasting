import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

// Mock ResizeObserver for Recharts
;(globalThis as any).ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
})

// Mock Leaflet for jsdom environment
vi.mock('leaflet', () => {
  const mockLayer = {
    addTo: vi.fn().mockReturnThis(),
    bindPopup: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
    clearLayers: vi.fn().mockReturnThis(),
    addLayer: vi.fn().mockReturnThis(),
  }

  return {
    default: {
      map: vi.fn().mockReturnValue({
        remove: vi.fn(),
        setView: vi.fn().mockReturnThis(),
        on: vi.fn().mockReturnThis(),
      }),
      tileLayer: vi.fn().mockReturnValue(mockLayer),
      layerGroup: vi.fn().mockReturnValue(mockLayer),
      circleMarker: vi.fn().mockReturnValue(mockLayer),
    },
    map: vi.fn().mockReturnValue({
      remove: vi.fn(),
      setView: vi.fn().mockReturnThis(),
      on: vi.fn().mockReturnThis(),
    }),
    tileLayer: vi.fn().mockReturnValue(mockLayer),
    layerGroup: vi.fn().mockReturnValue(mockLayer),
    circleMarker: vi.fn().mockReturnValue(mockLayer),
  }
})


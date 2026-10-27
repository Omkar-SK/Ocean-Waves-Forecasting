import React, { useState, useEffect, useCallback } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { Header } from './Header'
import { api } from '../../services/api'
import type { HealthResponse } from '../../types'

const pageTitles: Record<string, { title: string; subtitle: string }> = {
  '/': {
    title: 'Mission Overview & Forecast Hub',
    subtitle: 'Real-time telemetry, model readiness, and primary forecast projection',
  },
  '/live': {
    title: 'Live Operational Forecasting',
    subtitle: 'Near-real-time Arabian Sea wave height predictions vs Open-Meteo ECMWF NWP',
  },
  '/comparison': {
    title: 'Model Benchmarking & Skill Analysis',
    subtitle: 'Rigorous comparative evaluation across 1h, 6h, and 24h horizons vs Persistence',
  },
  '/historical': {
    title: 'Historical Test Set Evaluation',
    subtitle: 'Deep performance drill-down across 19,455 untouched test samples (2024–2026)',
  },
  '/about': {
    title: 'About OceanCast & Methodology',
    subtitle: 'Deep learning pipeline, ERA5 reanalysis provenance, and academic design',
  },
}

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthLoading, setHealthLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const location = useLocation()

  const fetchHealth = useCallback(async () => {
    try {
      const data = await api.getHealth()
      setHealth(data)
    } catch (err) {
      console.error('Failed to fetch backend health status:', err)
      setHealth(null)
    } finally {
      setHealthLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchHealth()
    const timer = setInterval(fetchHealth, 30000)
    return () => clearInterval(timer)
  }, [fetchHealth])

  const handleManualRefresh = async () => {
    setIsRefreshing(true)
    await fetchHealth()
    // Dispatch a custom event so active pages can refresh their own local data
    window.dispatchEvent(new CustomEvent('oceancast:refresh'))
    setTimeout(() => setIsRefreshing(false), 600)
  }

  const currentMeta = pageTitles[location.pathname] || {
    title: 'OceanCast AI',
    subtitle: 'Ocean Wave Forecasting Dashboard',
  }

  return (
    <div className="flex min-h-screen bg-[#07111F] text-[#F1F5F9]">
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <div className="flex flex-1 flex-col overflow-hidden">
        <Header
          onMenuClick={() => setSidebarOpen(true)}
          title={currentMeta.title}
          subtitle={currentMeta.subtitle}
          health={health}
          healthLoading={healthLoading}
          onRefresh={handleManualRefresh}
          isRefreshing={isRefreshing}
        />

        <main className="flex-1 overflow-y-auto px-4 py-6 md:px-8 md:py-8">
          <div className="mx-auto max-w-7xl">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}

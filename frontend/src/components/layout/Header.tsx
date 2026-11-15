import React, { useState, useEffect } from 'react'
import {
  Menu,
  RefreshCw,
  Flame,
  FlaskConical,
  Radio,
  Clock,
  ShieldCheck,
  Cpu,
  Layers,
  Activity,
} from 'lucide-react'
import { StatusIndicator } from '../ui/StatusIndicator'
import { Button } from '../ui/Button'
import { useSimulation } from '../../context/SimulationContext'
import type { HealthResponse } from '../../types'

interface HeaderProps {
  onMenuClick: () => void
  title: string
  subtitle?: string
  health: HealthResponse | null
  healthLoading: boolean
  onRefresh: () => void
  isRefreshing: boolean
}

export function Header({
  onMenuClick,
  title,
  subtitle,
  health,
  healthLoading,
  onRefresh,
  isRefreshing,
}: HeaderProps) {
  const isHealthy = health?.status === 'healthy'
  const isDegraded = health?.status === 'degraded'
  const checkpointsCount = health?.checkpoints_available ?? 9
  const totalCheckpoints = health?.total_checkpoints ?? 9

  const { isSimulationActive, activeScenario, openModal, reset, isLoading } = useSimulation()

  const [utcTime, setUtcTime] = useState<string>('')
  const [lastUpdateStr, setLastUpdateStr] = useState<string>('Just now')

  useEffect(() => {
    const updateTime = () => {
      const now = new Date()
      setUtcTime(now.toISOString().substring(11, 19) + ' UTC')
    }
    updateTime()
    const timer = setInterval(updateTime, 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    if (health?.timestamp) {
      try {
        const updateDate = new Date(health.timestamp)
        const diffMinutes = Math.max(0, Math.floor((Date.now() - updateDate.getTime()) / 60000))
        if (diffMinutes === 0) setLastUpdateStr('Just now')
        else if (diffMinutes === 1) setLastUpdateStr('1 min ago')
        else setLastUpdateStr(`${diffMinutes} mins ago`)
      } catch {
        setLastUpdateStr('Recent')
      }
    }
  }, [health?.timestamp, isRefreshing])

  return (
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-[#DCE5EF] bg-white px-4 md:px-8 shadow-xs">
      {/* Left Title & Command Context */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onMenuClick}
          className="p-2 text-[#64758A] hover:text-[#102A43] md:hidden rounded-lg hover:bg-[#F0F6FC]"
          aria-label="Open sidebar"
        >
          <Menu className="h-5 w-5" />
        </button>

        <div>
          <h1 className="font-['Outfit',sans-serif] text-base font-bold text-[#102A43] leading-tight md:text-lg tracking-tight">
            {title}
          </h1>
          {subtitle && (
            <p className="hidden text-xs text-[#64758A] md:block mt-0.5 font-normal">
              {subtitle}
            </p>
          )}
        </div>
      </div>

      {/* Right Controls & Telemetry Heartbeat */}
      <div className="flex items-center gap-2.5">
        {/* Mode Indicator: LIVE TELEMETRY vs SIMULATION MODE */}
        {isSimulationActive ? (
          <button
            type="button"
            onClick={openModal}
            className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#FEF3C7] border border-[#FCD34D] text-[#92400E] text-xs font-bold hover:bg-[#FDE68A] transition-all shadow-2xs"
            title="Hazard simulation active — click to manage scenario"
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#D97706] opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[#D97706]" />
            </span>
            <span className="truncate max-w-[130px] sm:max-w-none">
              SIMULATION MODE{activeScenario?.name ? `: ${activeScenario.name}` : ''}
            </span>
          </button>
        ) : (
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#ECFDF5] border border-[#A7F3D0] text-[#065F46] text-xs font-bold shadow-2xs">
            <span className="inline-block h-2 w-2 rounded-full bg-[#10B981] animate-pulse" />
            <span>LIVE DATA</span>
          </div>
        )}

        {/* Run Hazard Simulation Button (when in live mode) */}
        {!isSimulationActive && (
          <Button
            size="sm"
            onClick={openModal}
            disabled={isLoading}
            className="bg-gradient-to-r from-[#246B9E] to-[#102A43] hover:from-[#1d557e] hover:to-[#0c1f32] text-white shadow-xs text-xs font-semibold px-3 py-1.5 rounded-lg flex items-center gap-1.5"
          >
            <FlaskConical className="h-3.5 w-3.5 text-[#35B8AD]" />
            <span className="hidden sm:inline">Run Hazard Simulation</span>
            <span className="sm:hidden">Simulate</span>
          </Button>
        )}

        {/* Telemetry Freshness & Last Update */}
        <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-lg bg-[#F4F7FB] border border-[#DCE5EF] text-xs font-mono text-[#64758A]">
          <span className="text-[10px] text-[#246B9E] font-semibold uppercase">Freshness:</span>
          <span className="text-[#102A43] font-medium">{lastUpdateStr}</span>
        </div>

        {/* Live UTC Clock */}
        <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[#F0F6FC] border border-[#DCE5EF] text-xs font-mono text-[#102A43]">
          <span className="inline-block h-2 w-2 rounded-full bg-[#20AFA7] animate-pulse" />
          <span className="font-semibold">{utcTime || 'LIVE UTC'}</span>
        </div>

        {/* Backend & Model Health Status */}
        <div className="flex items-center gap-1.5">
          <StatusIndicator
            status={
              healthLoading
                ? 'syncing'
                : isHealthy
                ? 'healthy'
                : isDegraded
                ? 'degraded'
                : 'offline'
            }
            label={
              healthLoading
                ? 'Connecting...'
                : isHealthy
                ? `FastAPI (${checkpointsCount}/${totalCheckpoints} Models)`
                : isDegraded
                ? 'Degraded'
                : 'Backend Offline'
            }
            size="sm"
          />
        </div>

        {/* Manual Refresh Trigger */}
        <Button
          variant="secondary"
          size="sm"
          onClick={onRefresh}
          isLoading={isRefreshing}
          className="hidden sm:inline-flex bg-white hover:bg-[#F4F7FB] border border-[#DCE5EF] text-[#102A43] text-xs font-medium px-3 py-1.5"
        >
          <RefreshCw className={`h-3.5 w-3.5 text-[#246B9E] ${isRefreshing ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </Button>
      </div>
    </header>
  )
}

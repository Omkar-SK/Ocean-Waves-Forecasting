import React, { useState, useEffect } from 'react'
import { Menu, RefreshCw } from 'lucide-react'
import { StatusIndicator } from '../ui/StatusIndicator'
import { Button } from '../ui/Button'
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

  const [utcTime, setUtcTime] = useState<string>('')

  useEffect(() => {
    const updateTime = () => {
      const now = new Date()
      setUtcTime(now.toISOString().substring(11, 19) + ' UTC')
    }
    updateTime()
    const timer = setInterval(updateTime, 1000)
    return () => clearInterval(timer)
  }, [])

  return (
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-[#DCE5EF] bg-white px-4 md:px-8 shadow-xs">
      {/* Left Title & Context */}
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuClick}
          className="p-2 text-[#64758A] hover:text-[#15304B] md:hidden rounded-lg hover:bg-[#F0F6FC]"
          aria-label="Open sidebar"
        >
          <Menu className="h-5 w-5" />
        </button>

        <div>
          <h1 className="font-['Outfit',sans-serif] text-base font-bold text-[#15304B] leading-tight md:text-lg tracking-tight">
            {title}
          </h1>
          {subtitle && (
            <p className="hidden text-xs text-[#64758A] md:block mt-0.5">
              {subtitle}
            </p>
          )}
        </div>
      </div>

      {/* Right Controls & Telemetry Heartbeat */}
      <div className="flex items-center gap-3">
        {/* Live UTC Clock */}
        <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-md bg-[#F0F6FC] border border-[#DCE5EF] text-xs font-mono text-[#15304B]">
          <span className="inline-block h-2 w-2 rounded-full bg-[#20AFA7] animate-pulse" />
          <span className="font-semibold">{utcTime || 'LIVE UTC'}</span>
        </div>

        {/* Backend Status Indicator */}
        <StatusIndicator
          status={healthLoading ? 'syncing' : isHealthy ? 'healthy' : isDegraded ? 'degraded' : 'offline'}
          label={
            healthLoading
              ? 'Connecting...'
              : isHealthy
              ? 'FastAPI Connected'
              : isDegraded
              ? 'Degraded'
              : 'Backend Offline'
          }
          size="sm"
        />

        {/* Manual Refresh Trigger */}
        <Button
          variant="secondary"
          size="sm"
          onClick={onRefresh}
          isLoading={isRefreshing}
          className="hidden sm:inline-flex"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </Button>
      </div>
    </header>
  )
}

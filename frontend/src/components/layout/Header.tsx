import React from 'react'
import { Menu, RefreshCw, Radio, Database, ShieldCheck } from 'lucide-react'
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

  return (
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-[#1E3A5F] bg-[#07111F]/90 px-4 md:px-8 backdrop-blur-md">
      {/* Left Title Section */}
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuClick}
          className="p-2 text-[#94A3B8] hover:text-[#F1F5F9] md:hidden rounded-lg hover:bg-[#16263A]"
          aria-label="Open sidebar"
        >
          <Menu className="h-5 w-5" />
        </button>

        <div>
          <h1 className="font-['Outfit',sans-serif] text-lg font-bold text-[#F1F5F9] leading-tight md:text-xl">
            {title}
          </h1>
          {subtitle && (
            <p className="hidden text-xs text-[#94A3B8] md:block mt-0.5">
              {subtitle}
            </p>
          )}
        </div>
      </div>

      {/* Right Controls & Status Section */}
      <div className="flex items-center gap-3">
        {/* Backend & Checkpoints Status */}
        <div className="hidden items-center gap-2 lg:flex">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#101D2D] border border-[#1E3A5F] text-xs text-[#94A3B8]">
            <Database className="h-3.5 w-3.5 text-[#35C6E8]" />
            <span>ERA5 Clean</span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#101D2D] border border-[#1E3A5F] text-xs text-[#94A3B8]">
            <ShieldCheck className="h-3.5 w-3.5 text-[#35D0B0]" />
            <span>9/9 Checkpoints</span>
          </div>
        </div>

        {/* Live Backend Connection Indicator */}
        <StatusIndicator
          status={healthLoading ? 'syncing' : isHealthy ? 'healthy' : isDegraded ? 'degraded' : 'offline'}
          label={
            healthLoading
              ? 'Checking...'
              : isHealthy
              ? 'FastAPI Connected'
              : isDegraded
              ? 'Degraded'
              : 'Backend Offline'
          }
          size="sm"
        />

        {/* Refresh Action */}
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

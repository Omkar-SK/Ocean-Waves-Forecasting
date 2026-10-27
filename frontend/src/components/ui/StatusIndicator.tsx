import React from 'react'
import { cn } from '../../lib/utils'

interface StatusIndicatorProps {
  status: 'online' | 'healthy' | 'degraded' | 'offline' | 'error' | 'syncing'
  label?: string
  className?: string
  size?: 'sm' | 'md'
}

export function StatusIndicator({ status, label, className, size = 'sm' }: StatusIndicatorProps) {
  const statusConfig = {
    online: {
      color: 'bg-[#35D0B0]',
      glow: 'shadow-[0_0_8px_rgba(53,208,176,0.8)]',
      text: 'text-[#35D0B0]',
      defaultLabel: 'Live Online',
    },
    healthy: {
      color: 'bg-[#35D0B0]',
      glow: 'shadow-[0_0_8px_rgba(53,208,176,0.8)]',
      text: 'text-[#35D0B0]',
      defaultLabel: 'Healthy',
    },
    degraded: {
      color: 'bg-[#F59E0B]',
      glow: 'shadow-[0_0_8px_rgba(245,158,11,0.8)]',
      text: 'text-[#F59E0B]',
      defaultLabel: 'Degraded / Demo',
    },
    offline: {
      color: 'bg-[#EF4444]',
      glow: 'shadow-[0_0_8px_rgba(239,68,68,0.8)]',
      text: 'text-[#EF4444]',
      defaultLabel: 'Offline',
    },
    error: {
      color: 'bg-[#EF4444]',
      glow: 'shadow-[0_0_8px_rgba(239,68,68,0.8)]',
      text: 'text-[#EF4444]',
      defaultLabel: 'Error',
    },
    syncing: {
      color: 'bg-[#35C6E8]',
      glow: 'shadow-[0_0_8px_rgba(53,198,232,0.8)]',
      text: 'text-[#35C6E8]',
      defaultLabel: 'Syncing',
    },
  }

  const current = statusConfig[status] || statusConfig.online
  const dotSize = size === 'sm' ? 'w-2 h-2' : 'w-2.5 h-2.5'

  return (
    <div
      className={cn(
        'inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-[#101D2D] border border-[#1E3A5F] text-xs font-medium',
        className
      )}
    >
      <span className="relative flex items-center justify-center">
        <span
          className={cn(
            'absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping',
            current.color
          )}
        />
        <span className={cn('relative inline-flex rounded-full', dotSize, current.color, current.glow)} />
      </span>
      <span className={cn('leading-none', current.text)}>{label || current.defaultLabel}</span>
    </div>
  )
}

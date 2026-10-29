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
      color: 'bg-[#10B981]',
      pulse: true,
      text: 'text-[#065F46]',
      bg: 'bg-[#EBFBF5]',
      border: 'border-[#A7F3D0]',
      defaultLabel: 'Live Stream',
    },
    healthy: {
      color: 'bg-[#10B981]',
      pulse: false,
      text: 'text-[#065F46]',
      bg: 'bg-[#EBFBF5]',
      border: 'border-[#A7F3D0]',
      defaultLabel: 'Operational',
    },
    degraded: {
      color: 'bg-[#D97706]',
      pulse: false,
      text: 'text-[#92400E]',
      bg: 'bg-[#FEF3C7]',
      border: 'border-[#FDE68A]',
      defaultLabel: 'Degraded Feed',
    },
    offline: {
      color: 'bg-[#E11D48]',
      pulse: false,
      text: 'text-[#9F1239]',
      bg: 'bg-[#FFE4E6]',
      border: 'border-[#FECDD3]',
      defaultLabel: 'Offline',
    },
    error: {
      color: 'bg-[#E11D48]',
      pulse: false,
      text: 'text-[#9F1239]',
      bg: 'bg-[#FFE4E6]',
      border: 'border-[#FECDD3]',
      defaultLabel: 'System Error',
    },
    syncing: {
      color: 'bg-[#397FB8]',
      pulse: true,
      text: 'text-[#1E40AF]',
      bg: 'bg-[#EFF6FF]',
      border: 'border-[#BFDBFE]',
      defaultLabel: 'Syncing',
    },
  }

  const current = statusConfig[status] || statusConfig.online
  const dotSize = size === 'sm' ? 'w-1.5 h-1.5' : 'w-2 h-2'

  return (
    <div
      className={cn(
        'inline-flex items-center gap-2 px-2.5 py-1 rounded-full border text-xs font-mono font-medium transition-colors',
        current.bg,
        current.border,
        className
      )}
    >
      <span className="relative flex h-2 w-2 items-center justify-center">
        {current.pulse && (
          <span
            className={cn(
              'absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping',
              current.color
            )}
          />
        )}
        <span className={cn('relative inline-flex rounded-full', dotSize, current.color)} />
      </span>
      <span className={cn('tracking-tight font-semibold', current.text)}>
        {label || current.defaultLabel}
      </span>
    </div>
  )
}

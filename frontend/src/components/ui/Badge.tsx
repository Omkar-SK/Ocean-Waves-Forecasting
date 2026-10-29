import React from 'react'
import { cn } from '../../lib/utils'

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'cyan' | 'teal' | 'warning' | 'destructive' | 'outline' | 'surface' | 'radar'
}

export function Badge({ className, variant = 'default', children, ...props }: BadgeProps) {
  const variantStyles = {
    default: 'bg-[#EAF2F9] text-[#15304B] border-[#DCE5EF]',
    cyan: 'bg-[#DCECF9] text-[#245B8F] border-[#79B7E5]/50',
    teal: 'bg-[#E1F5F2] text-[#0D837C] border-[#20AFA7]/40',
    warning: 'bg-[#FEF3C7] text-[#B45309] border-[#FCD34D]',
    destructive: 'bg-[#FFE4E6] text-[#BE123C] border-[#FDA4AF]',
    outline: 'bg-white text-[#192B3C] border-[#DCE5EF]',
    surface: 'bg-[#F0F6FC] text-[#15304B] border-[#DCECF9]',
    radar: 'bg-[#E1F5F2] text-[#0D837C] border-[#20AFA7]/50 shadow-xs',
  }

  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium tracking-wide border transition-colors',
        variantStyles[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  )
}

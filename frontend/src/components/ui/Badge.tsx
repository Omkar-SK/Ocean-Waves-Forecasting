import React from 'react'
import { cn } from '../../lib/utils'

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'cyan' | 'teal' | 'warning' | 'destructive' | 'outline' | 'surface'
}

export function Badge({ className, variant = 'default', children, ...props }: BadgeProps) {
  const variantStyles = {
    default: 'bg-[#16263A] text-[#94A3B8] border-[#1E3A5F]',
    cyan: 'bg-[#35C6E8]/10 text-[#35C6E8] border-[#35C6E8]/30',
    teal: 'bg-[#35D0B0]/10 text-[#35D0B0] border-[#35D0B0]/30',
    warning: 'bg-[#F59E0B]/10 text-[#F59E0B] border-[#F59E0B]/30',
    destructive: 'bg-[#EF4444]/10 text-[#EF4444] border-[#EF4444]/30',
    outline: 'bg-transparent text-[#F1F5F9] border-[#1E3A5F]',
    surface: 'bg-[#101D2D] text-[#F1F5F9] border-[#1E3A5F]',
  }

  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border transition-colors',
        variantStyles[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  )
}

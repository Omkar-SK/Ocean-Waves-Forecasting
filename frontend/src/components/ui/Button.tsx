import React from 'react'
import { cn } from '../../lib/utils'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'teal'
  size?: 'sm' | 'md' | 'lg'
  isLoading?: boolean
}

export function Button({
  className,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  disabled,
  children,
  ...props
}: ButtonProps) {
  const base =
    'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-[#20AFA7]/30 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer'

  const variants = {
    primary:
      'bg-[#20AFA7] hover:bg-[#1A9C94] text-white font-semibold shadow-xs border border-transparent active:scale-[0.98]',
    secondary:
      'bg-white hover:bg-[#F0F6FC] text-[#15304B] border border-[#DCE5EF] hover:border-[#CBD5E1] shadow-xs active:scale-[0.98]',
    outline:
      'bg-transparent hover:bg-[#EAF2F9] text-[#245B8F] border border-[#245B8F]/35 hover:border-[#245B8F] active:scale-[0.98]',
    teal:
      'bg-[#20AFA7] hover:bg-[#1A9C94] text-white font-semibold shadow-xs border border-transparent active:scale-[0.98]',
    ghost:
      'bg-transparent hover:bg-[#F0F6FC] text-[#64758A] hover:text-[#15304B] border border-transparent',
  }

  const sizes = {
    sm: 'text-xs px-2.5 py-1.5 gap-1.5',
    md: 'text-sm px-4 py-2 gap-2',
    lg: 'text-base px-5 py-2.5 gap-2.5',
  }

  return (
    <button
      className={cn(base, variants[variant], sizes[size], className)}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <span className="inline-block w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin mr-1.5" />
      ) : null}
      {children}
    </button>
  )
}

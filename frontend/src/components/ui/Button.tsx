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
    'inline-flex items-center justify-center font-medium rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-[#35C6E8]/40 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer'

  const variants = {
    primary:
      'bg-[#35C6E8] hover:bg-[#28B0D0] text-[#07111F] font-semibold shadow-md shadow-[#35C6E8]/20 border border-transparent',
    secondary:
      'bg-[#16263A] hover:bg-[#1E3A5F] text-[#F1F5F9] border border-[#1E3A5F]',
    outline:
      'bg-transparent hover:bg-[#16263A] text-[#35C6E8] border border-[#35C6E8]/50 hover:border-[#35C6E8]',
    teal:
      'bg-[#35D0B0] hover:bg-[#28B999] text-[#07111F] font-semibold shadow-md shadow-[#35D0B0]/20 border border-transparent',
    ghost:
      'bg-transparent hover:bg-[#16263A] text-[#94A3B8] hover:text-[#F1F5F9]',
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

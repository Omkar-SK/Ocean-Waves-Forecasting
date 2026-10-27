import React from 'react'
import { cn } from '../../lib/utils'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  glow?: 'cyan' | 'teal' | 'none'
}

export function Card({ className, glow = 'none', children, ...props }: CardProps) {
  return (
    <div
      className={cn(
        'ocean-card bg-[#101D2D] border border-[#1E3A5F] rounded-xl text-[#F1F5F9] shadow-lg transition-all duration-200',
        glow === 'cyan' && 'ocean-glow-cyan border-[#35C6E8]/40',
        glow === 'teal' && 'ocean-glow-teal border-[#35D0B0]/40',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('p-5 pb-3 flex flex-col space-y-1.5', className)} {...props}>
      {children}
    </div>
  )
}

export function CardTitle({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn('text-base font-semibold leading-tight text-[#F1F5F9] font-["Outfit",sans-serif]', className)}
      {...props}
    >
      {children}
    </h3>
  )
}

export function CardDescription({ className, children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p className={cn('text-xs text-[#94A3B8] font-normal leading-relaxed', className)} {...props}>
      {children}
    </p>
  )
}

export function CardContent({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('p-5 pt-2', className)} {...props}>
      {children}
    </div>
  )
}

export function CardFooter({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('p-5 pt-0 flex items-center border-t border-[#1E3A5F]/40 mt-4', className)} {...props}>
      {children}
    </div>
  )
}

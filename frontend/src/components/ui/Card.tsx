import React from 'react'
import { cn } from '../../lib/utils'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  glow?: 'teal' | 'cyan' | 'coral' | 'none'
  variant?: 'panel' | 'raised' | 'ice' | 'navy'
}

export function Card({
  className,
  glow = 'none',
  variant = 'panel',
  children,
  ...props
}: CardProps) {
  const variantClasses = {
    panel: 'bg-white border-[#DCE5EF] text-[#192B3C] shadow-xs',
    raised: 'bg-white border-[#DCE5EF] text-[#192B3C] shadow-sm',
    ice: 'bg-[#F0F6FC] border-[#DCECF9] text-[#192B3C]',
    navy: 'bg-[#15304B] border-[#245B8F] text-white',
  }

  return (
    <div
      className={cn(
        'ocean-card rounded-xl transition-all duration-200',
        variantClasses[variant],
        glow === 'teal' && 'border-[#20AFA7]/50 shadow-[0_0_15px_rgba(32,175,167,0.15)]',
        glow === 'cyan' && 'border-[#397FB8]/40 shadow-[0_0_15px_rgba(57,127,184,0.15)]',
        glow === 'coral' && 'border-[#E11D48]/40 shadow-[0_0_15px_rgba(225,29,72,0.12)]',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('p-5 pb-3 flex flex-col space-y-1.5', className)} {...props}>
      {children}
    </div>
  )
}

export function CardTitle({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn(
        'text-base font-bold leading-tight text-[#15304B] font-["Outfit",sans-serif] tracking-tight',
        className
      )}
      {...props}
    >
      {children}
    </h3>
  )
}

export function CardDescription({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p className={cn('text-xs text-[#64758A] font-normal leading-relaxed', className)} {...props}>
      {children}
    </p>
  )
}

export function CardContent({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('p-5 pt-2', className)} {...props}>
      {children}
    </div>
  )
}

export function CardFooter({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('p-5 pt-0 flex items-center border-t border-[#DCE5EF] mt-4', className)}
      {...props}
    >
      {children}
    </div>
  )
}

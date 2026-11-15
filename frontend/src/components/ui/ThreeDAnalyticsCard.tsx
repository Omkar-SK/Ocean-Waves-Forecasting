import React, { useRef, useState } from 'react'
import { motion } from 'motion/react'
import type { LucideIcon } from 'lucide-react'
import { TrendingUp, TrendingDown, ArrowUpRight } from 'lucide-react'
import { cn } from '../../lib/utils'

export interface ThreeDAnalyticsCardProps {
  title: string
  subtitle?: string
  value: string | number
  unit?: string
  trend?: {
    value: string
    isPositive?: boolean
    label?: string
  }
  icon?: LucideIcon
  badge?: string
  badgeVariant?: 'teal' | 'blue' | 'sky' | 'default'
  children?: React.ReactNode
  className?: string
  onClick?: () => void
}

export function ThreeDAnalyticsCard({
  title,
  subtitle,
  value,
  unit,
  trend,
  icon: Icon,
  badge,
  badgeVariant = 'teal',
  children,
  className = '',
  onClick,
}: ThreeDAnalyticsCardProps) {
  const cardRef = useRef<HTMLDivElement | null>(null)
  const [rotateX, setRotateX] = useState(0)
  const [rotateY, setRotateY] = useState(0)
  const [glarePosition, setGlarePosition] = useState({ x: 50, y: 50 })
  const [isHovered, setIsHovered] = useState(false)

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return
    const rect = cardRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    // Calculate rotation (-12deg to +12deg)
    const normX = (x / rect.width - 0.5) * 2
    const normY = (y / rect.height - 0.5) * 2

    setRotateX(-normY * 10)
    setRotateY(normX * 10)
    setGlarePosition({
      x: (x / rect.width) * 100,
      y: (y / rect.height) * 100,
    })
  }

  const handleMouseEnter = () => setIsHovered(true)

  const handleMouseLeave = () => {
    setIsHovered(false)
    setRotateX(0)
    setRotateY(0)
  }

  const getBadgeClass = () => {
    switch (badgeVariant) {
      case 'teal':
        return 'bg-[#35B8AD]/15 text-[#35B8AD] border-[#35B8AD]/30'
      case 'blue':
        return 'bg-[#246B9E]/15 text-[#246B9E] border-[#246B9E]/30'
      case 'sky':
        return 'bg-[#8CC8E8]/20 text-[#246B9E] border-[#8CC8E8]/40'
      default:
        return 'bg-[#F3F8FC] text-[#102A43] border-[#DCE5EF]'
    }
  }

  return (
    <div
      style={{ perspective: '1000px' }}
      className={cn('relative transition-transform duration-200 select-none', className)}
    >
      <div
        ref={cardRef}
        onMouseMove={handleMouseMove}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onClick={onClick}
        style={{
          transform: `rotateX(${rotateX}deg) rotateY(${rotateY}deg) ${
            isHovered ? 'scale3d(1.02, 1.02, 1.02)' : 'scale3d(1, 1, 1)'
          }`,
          transformStyle: 'preserve-3d',
        }}
        className={cn(
          'relative rounded-2xl bg-white border border-[#DCE5EF] p-5 shadow-xs transition-shadow duration-300 overflow-hidden cursor-pointer group',
          isHovered && 'shadow-xl shadow-[#246B9E]/10 border-[#35B8AD]/50'
        )}
      >
        {/* Dynamic Specular Glare Layer */}
        {isHovered && (
          <div
            className="pointer-events-none absolute inset-0 z-30 transition-opacity duration-300 rounded-2xl"
            style={{
              background: `radial-gradient(circle 280px at ${glarePosition.x}% ${glarePosition.y}%, rgba(255, 255, 255, 0.45), transparent 70%)`,
            }}
          />
        )}

        {/* Ambient Corner Accent */}
        <div className="absolute -top-12 -right-12 w-32 h-32 rounded-full bg-gradient-to-br from-[#8CC8E8]/20 to-[#35B8AD]/20 blur-xl pointer-events-none" />

        {/* Card Content with 3D Depth Elevation */}
        <div style={{ transform: 'translateZ(20px)' }} className="relative z-10 flex flex-col justify-between h-full">
          {/* Header Row */}
          <div className="flex items-start justify-between gap-3 mb-3">
            <div>
              <span className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] font-semibold block">
                {title}
              </span>
              {subtitle && <p className="text-xs text-[#8291A2] mt-0.5">{subtitle}</p>}
            </div>

            <div className="flex items-center gap-1.5">
              {badge && (
                <span className={cn('px-2 py-0.5 rounded-full text-[10px] font-mono font-bold border', getBadgeClass())}>
                  {badge}
                </span>
              )}
              {Icon && (
                <div className="p-2 rounded-xl bg-[#F3F8FC] border border-[#DCE5EF] text-[#246B9E] group-hover:text-[#35B8AD] transition-colors">
                  <Icon className="h-4 w-4" />
                </div>
              )}
            </div>
          </div>

          {/* Core Metric Value with 3D Pop */}
          <div style={{ transform: 'translateZ(30px)' }} className="my-2 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold font-mono text-[#102A43] tracking-tight">
              {value}
            </span>
            {unit && <span className="text-sm font-bold font-mono text-[#64758A]">{unit}</span>}
          </div>

          {/* Trend Indicator or Extra Metric Content */}
          {trend && (
            <div className="flex items-center gap-1.5 text-xs font-mono mt-1">
              {trend.isPositive !== false ? (
                <span className="inline-flex items-center gap-0.5 text-emerald-600 font-bold">
                  <TrendingUp className="h-3.5 w-3.5" />
                  {trend.value}
                </span>
              ) : (
                <span className="inline-flex items-center gap-0.5 text-rose-600 font-bold">
                  <TrendingDown className="h-3.5 w-3.5" />
                  {trend.value}
                </span>
              )}
              {trend.label && <span className="text-[#8291A2] text-[11px]">{trend.label}</span>}
            </div>
          )}

          {/* Injected Children (e.g. mini chart or sparkline) */}
          {children && <div style={{ transform: 'translateZ(15px)' }} className="mt-3">{children}</div>}
        </div>
      </div>
    </div>
  )
}

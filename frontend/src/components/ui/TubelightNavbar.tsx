import React, { useState } from 'react'
import { motion } from 'motion/react'
import type { LucideIcon } from 'lucide-react'
import { cn } from '../../lib/utils'

export interface NavItem {
  name: string
  url: string
  icon?: LucideIcon
}

interface TubelightNavbarProps {
  items: NavItem[]
  activeTab?: string
  onTabChange?: (name: string) => void
  className?: string
}

export function TubelightNavbar({
  items,
  activeTab: controlledActiveTab,
  onTabChange,
  className,
}: TubelightNavbarProps) {
  const [internalActiveTab, setInternalActiveTab] = useState(items[0]?.name || '')
  const activeTab = controlledActiveTab !== undefined ? controlledActiveTab : internalActiveTab

  const handleSelect = (name: string) => {
    setInternalActiveTab(name)
    if (onTabChange) onTabChange(name)
  }

  return (
    <div
      className={cn(
        'flex items-center gap-1 bg-[#102A43]/85 border border-[#35B8AD]/30 backdrop-blur-lg py-1.5 px-2 rounded-full shadow-2xl shadow-[#102A43]/40',
        className
      )}
    >
      {items.map((item) => {
        const Icon = item.icon
        const isActive = activeTab === item.name

        return (
          <a
            key={item.name}
            href={item.url}
            onClick={() => handleSelect(item.name)}
            className={cn(
              'relative cursor-pointer text-xs md:text-sm font-medium px-4 py-1.5 rounded-full transition-colors flex items-center gap-1.5',
              isActive ? 'text-white font-semibold' : 'text-[#8CC8E8]/80 hover:text-white'
            )}
          >
            {Icon && <Icon className="h-4 w-4 shrink-0" />}
            <span>{item.name}</span>

            {isActive && (
              <motion.div
                layoutId="tubelightLamp"
                className="absolute inset-0 w-full bg-[#246B9E]/30 rounded-full -z-10"
                initial={false}
                transition={{
                  type: 'spring',
                  stiffness: 320,
                  damping: 30,
                }}
              >
                {/* Tubelight Glowing Filament */}
                <div className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-8 h-1 bg-[#35B8AD] rounded-t-full shadow-[0_0_12px_2px_#35B8AD]">
                  {/* Concentrated light beam */}
                  <div className="absolute w-12 h-4 bg-[#35B8AD]/30 rounded-full blur-xs -top-1 -left-2" />
                  <div className="absolute w-6 h-4 bg-[#8CC8E8]/40 rounded-full blur-md -top-1 left-1" />
                </div>
              </motion.div>
            )}
          </a>
        )
      })}
    </div>
  )
}

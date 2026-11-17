import React from 'react'
import { NavLink } from 'react-router-dom'
import {
  Compass,
  Activity,
  Map,
  AlertTriangle,
  Radio,
  TrendingUp,
  BarChart3,
  History,
  Info,
  Globe,
  ChevronLeft,
  ChevronRight,
  X,
  ShieldCheck,
  FlaskConical,
} from 'lucide-react'
import { cn } from '../../lib/utils'

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  isCollapsed?: boolean
  onToggleCollapse?: () => void
}

interface NavGroup {
  groupName: string
  items: {
    name: string
    path: string
    icon: React.ComponentType<{ className?: string }>
    badge?: string
    legacyAlias?: string
  }[]
}

const navGroups: NavGroup[] = [
  {
    groupName: 'Operational Intelligence',
    items: [
      { name: 'Overview', path: '/', icon: Compass, badge: 'Hub' },
      { name: 'Simulation Lab', path: '/simulation', icon: FlaskConical, badge: 'Lab' },
      { name: 'Live Forecast', path: '/live', icon: Activity, badge: 'DL' },
      { name: 'Coastal Risk Map', path: '/map', icon: Map, badge: '8 Nodes' },
      { name: 'Coastal Alerts', path: '/alerts', icon: AlertTriangle, badge: 'Alerts' },
    ],
  },
  {
    groupName: 'Monitoring',
    items: [
      { name: 'Forecast Verification', path: '/verification', icon: ShieldCheck, badge: 'Live' },
      { name: 'Coastal Locations', path: '/locations', icon: Radio, badge: 'Feeds' },
      { name: 'Accuracy & Drift', path: '/monitoring', icon: TrendingUp, badge: 'PSI' },
      { name: 'Historical Analysis', path: '/historical', icon: History, badge: '19.4k' },
      { name: 'Model Comparison', path: '/comparison', icon: BarChart3, badge: 'Bench' },
    ],
  },
  {
    groupName: 'Platform',
    items: [
      { name: 'System Status', path: '/about', icon: Info, badge: 'Ready', legacyAlias: 'System & Docs' },
      { name: 'Cinematic Landing', path: '/home', icon: Globe, badge: '3D' },
    ],
  },
]

export function Sidebar({
  isOpen,
  onClose,
  isCollapsed = false,
  onToggleCollapse,
}: SidebarProps) {
  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 backdrop-blur-xs md:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar Container: Grounding Deep Navy Marine Panel */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-50 flex flex-col bg-[#102A43] border-r border-[#245B8F]/30 transition-all duration-300 ease-in-out md:static md:translate-x-0 shadow-xl select-none',
          isOpen ? 'translate-x-0' : '-translate-x-full',
          isCollapsed ? 'md:w-20' : 'md:w-64 w-72'
        )}
      >
        {/* Brand Header with Custom Wave Radar Mark */}
        <div className="flex h-16 items-center justify-between border-b border-[#245B8F]/30 px-4">
          <NavLink
            to="/"
            onClick={onClose}
            className="flex items-center gap-3 group focus:outline-none"
          >
            {/* Custom Wave Mark */}
            <div className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#0a1829] border border-[#20AFA7]/40 shadow-xs group-hover:border-[#20AFA7] transition-all group-hover:scale-105">
              <svg viewBox="0 0 32 32" className="h-6 w-6" aria-hidden="true">
                <circle cx="16" cy="16" r="13" fill="none" stroke="#245B8F" strokeWidth="1" />
                <circle cx="16" cy="16" r="8" fill="none" stroke="#397FB8" strokeWidth="0.8" strokeDasharray="2 2" />
                <path
                  d="M6 16 C9 12, 11 12, 14 16 C17 20, 19 20, 22 16 C24 13, 25 13, 26 15"
                  fill="none"
                  stroke="#20AFA7"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
                <circle cx="16" cy="16" r="2.2" fill="#79B7E5" />
              </svg>
            </div>

            {!isCollapsed && (
              <div className="flex flex-col">
                <div className="flex items-center gap-1.5">
                  <span className="font-['Outfit',sans-serif] text-base font-extrabold tracking-tight text-white leading-none">
                    OceanCast
                  </span>
                  <span className="text-[10px] font-mono font-bold px-1 py-0.2 rounded bg-[#20AFA7]/20 text-[#20AFA7] border border-[#20AFA7]/40">
                    AI
                  </span>
                </div>
                <span className="text-[10px] uppercase font-mono tracking-widest text-[#79B7E5] mt-0.5">
                  Command Centre
                </span>
              </div>
            )}
          </NavLink>

          {/* Mobile Close Button */}
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 text-[#79B7E5] hover:text-white rounded-lg md:hidden hover:bg-[#245B8F]/30"
            aria-label="Close sidebar"
          >
            <X className="h-5 w-5" />
          </button>

          {/* Desktop Collapse Toggle */}
          {onToggleCollapse && (
            <button
              type="button"
              onClick={onToggleCollapse}
              className="hidden md:flex p-1.5 text-[#79B7E5] hover:text-[#20AFA7] rounded-lg hover:bg-[#245B8F]/30 transition-colors"
              title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
            </button>
          )}
        </div>

        {/* Navigation Item Groups */}
        <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
          {navGroups.map((group, gIdx) => (
            <div key={gIdx} className="space-y-1">
              {!isCollapsed && (
                <div className="px-3 pb-1.5 text-[10px] font-mono uppercase tracking-widest text-[#79B7E5]/75 font-semibold">
                  {group.groupName}
                </div>
              )}
              {group.items.map((item) => {
                const Icon = item.icon
                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    end={item.path === '/'}
                    onClick={onClose}
                    title={isCollapsed ? item.name : undefined}
                    className={({ isActive }) =>
                      cn(
                        'group relative flex items-center gap-3 rounded-xl px-3 py-2 text-xs font-medium transition-all duration-150',
                        isCollapsed ? 'justify-center px-2 py-2.5' : '',
                        isActive
                          ? 'bg-[#246B9E] text-white font-semibold shadow-xs'
                          : 'text-[#DCECF9] hover:bg-[#1A3D63] hover:text-white'
                      )
                    }
                  >
                    {({ isActive }) => (
                      <>
                        {/* Active vertical accent bar */}
                        {isActive && (
                          <span className="absolute left-0 top-1.5 bottom-1.5 w-1 rounded-r-md bg-[#20AFA7] shadow-[0_0_8px_#20AFA7]" />
                        )}

                        <Icon
                          className={cn(
                            'h-4 w-4 shrink-0 transition-transform group-hover:scale-110',
                            isActive ? 'text-[#20AFA7]' : 'text-[#79B7E5] group-hover:text-white'
                          )}
                        />

                        {!isCollapsed && (
                          <span className="flex-1 truncate tracking-tight font-medium">
                            {item.name}
                            {item.legacyAlias && (
                              <span className="sr-only">{item.legacyAlias}</span>
                            )}
                          </span>
                        )}

                        {!isCollapsed && item.badge && (
                          <span
                            className={cn(
                              'text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded-md transition-colors',
                              isActive
                                ? 'bg-[#20AFA7]/30 text-white'
                                : 'bg-[#0a1829] text-[#79B7E5]'
                            )}
                          >
                            {item.badge}
                          </span>
                        )}
                      </>
                    )}
                  </NavLink>
                )
              })}
            </div>
          ))}
        </div>

        {/* Sidebar Footer: Real-time Coordinates & Status */}
        <div className="border-t border-[#245B8F]/30 p-3 bg-[#0a1829]/60">
          {!isCollapsed ? (
            <div className="flex items-center justify-between text-[11px] text-[#79B7E5]">
              <div className="flex items-center gap-2">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#10B981] opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-[#10B981]" />
                </span>
                <span className="font-mono text-white text-[10px]">18.50°N 72.50°E</span>
              </div>
              <span className="text-[10px] font-mono text-[#20AFA7] font-semibold flex items-center gap-1">
                <ShieldCheck className="h-3 w-3" /> ECMWF
              </span>
            </div>
          ) : (
            <div className="flex justify-center" title="Telemetry Feed Active: 18.50°N 72.50°E">
              <div className="h-2.5 w-2.5 rounded-full bg-[#10B981] animate-pulse" />
            </div>
          )}
        </div>
      </aside>
    </>
  )
}

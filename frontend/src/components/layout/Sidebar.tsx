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
  Waves,
  MapPin,
  Cpu,
  X,
} from 'lucide-react'
import { cn } from '../../lib/utils'

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
}

const navItems = [
  {
    name: 'Overview',
    path: '/',
    icon: Compass,
    badge: 'Hub',
  },
  {
    name: 'Live Forecast',
    path: '/live',
    icon: Activity,
    badge: 'DL + NWP',
  },
  {
    name: 'Coastal Risk Map',
    path: '/map',
    icon: Map,
    badge: '8 Nodes',
  },
  {
    name: 'Coastal Alerts',
    path: '/alerts',
    icon: AlertTriangle,
    badge: 'Early Warning',
  },
  {
    name: 'Coastal Locations',
    path: '/locations',
    icon: Radio,
    badge: 'Telemetry',
  },
  {
    name: 'Accuracy & Drift',
    path: '/monitoring',
    icon: TrendingUp,
    badge: 'ML Ops',
  },
  {
    name: 'Historical Analysis',
    path: '/historical',
    icon: History,
    badge: '19.4k Test',
  },
  {
    name: 'Model Comparison',
    path: '/comparison',
    icon: BarChart3,
    badge: 'Benchmark',
  },
  {
    name: 'System & Docs',
    path: '/about',
    icon: Info,
    badge: 'Status',
  },
]

export function Sidebar({ isOpen, onClose }: SidebarProps) {
  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-xs md:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-50 flex w-72 flex-col bg-[#07111F] border-r border-[#1E3A5F] transition-transform duration-300 ease-in-out md:static md:translate-x-0',
          isOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        {/* Brand Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-[#1E3A5F]">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-[#35C6E8]/20 to-[#35D0B0]/20 border border-[#35C6E8]/40 shadow-sm shadow-[#35C6E8]/20">
              <Waves className="h-5 w-5 text-[#35C6E8]" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-['Outfit',sans-serif] text-lg font-bold tracking-tight text-[#F1F5F9]">
                  OceanCast
                </span>
                <span className="rounded bg-[#35C6E8]/20 px-1.5 py-0.5 text-[10px] font-semibold text-[#35C6E8] border border-[#35C6E8]/30">
                  AI
                </span>
              </div>
              <p className="text-[11px] text-[#94A3B8]">Deep Learning Wave Radar</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="md:hidden p-1.5 text-[#94A3B8] hover:text-[#F1F5F9] rounded-lg hover:bg-[#16263A]"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Location Indicator Card */}
        <div className="px-4 py-3">
          <div className="rounded-lg bg-[#101D2D] p-3 border border-[#1E3A5F]/80">
            <div className="flex items-center gap-2 text-[11px] font-medium text-[#35C6E8] uppercase tracking-wider mb-1">
              <MapPin className="h-3.5 w-3.5" />
              <span>Offshore Target Grid</span>
            </div>
            <div className="text-sm font-semibold text-[#F1F5F9] font-mono">
              18.50° N, 72.50° E
            </div>
            <div className="text-[11px] text-[#94A3B8] mt-0.5">
              Arabian Sea • Mumbai High Offshore
            </div>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 space-y-1.5 px-3 py-2 overflow-y-auto">
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={onClose}
                className={({ isActive }) =>
                  cn(
                    'group flex items-center justify-between rounded-lg px-3.5 py-2.5 text-sm font-medium transition-all duration-150',
                    isActive
                      ? 'bg-[#16263A] text-[#35C6E8] border-l-3 border-[#35C6E8] shadow-sm shadow-[#35C6E8]/10'
                      : 'text-[#94A3B8] hover:bg-[#101D2D] hover:text-[#F1F5F9]'
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <div className="flex items-center gap-3">
                      <Icon
                        className={cn(
                          'h-4 w-4 transition-colors',
                          isActive ? 'text-[#35C6E8]' : 'text-[#94A3B8] group-hover:text-[#F1F5F9]'
                        )}
                      />
                      <span>{item.name}</span>
                    </div>
                    <span
                      className={cn(
                        'text-[10px] px-1.5 py-0.5 rounded font-mono',
                        isActive
                          ? 'bg-[#35C6E8]/20 text-[#35C6E8] font-semibold'
                          : 'bg-[#16263A]/80 text-[#94A3B8]'
                      )}
                    >
                      {item.badge}
                    </span>
                  </>
                )}
              </NavLink>
            )
          })}
        </nav>

        {/* Models Badge Footer */}
        <div className="p-4 border-t border-[#1E3A5F] bg-[#101D2D]/40">
          <div className="flex items-center justify-between text-xs text-[#94A3B8] mb-2">
            <span className="flex items-center gap-1.5">
              <Cpu className="h-3.5 w-3.5 text-[#35D0B0]" />
              <span>Architectures</span>
            </span>
            <span className="font-mono text-[10px] text-[#35D0B0]">9 Checkpoints</span>
          </div>
          <div className="grid grid-cols-3 gap-1.5 text-center text-[10px] font-mono">
            <div className="bg-[#16263A] py-1 px-1 rounded border border-[#1E3A5F] text-[#F1F5F9]">
              ANN
            </div>
            <div className="bg-[#16263A] py-1 px-1 rounded border border-[#1E3A5F] text-[#35C6E8] font-semibold">
              LSTM
            </div>
            <div className="bg-[#16263A] py-1 px-1 rounded border border-[#1E3A5F] text-[#F1F5F9]">
              1D-CNN
            </div>
          </div>
          <div className="mt-3 text-[10px] text-center text-[#94A3B8]/80">
            Sem 5 Deep Learning Project
          </div>
        </div>
      </aside>
    </>
  )
}

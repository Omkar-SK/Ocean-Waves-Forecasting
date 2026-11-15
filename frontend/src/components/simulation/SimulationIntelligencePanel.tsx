import React from 'react'
import {
  Activity,
  AlertTriangle,
  Flame,
  Wind,
  Waves,
  ShieldAlert,
  ShieldCheck,
  Compass,
  Clock,
  CheckCircle2,
  AlertCircle,
  FileText,
} from 'lucide-react'
import type { SimulationStats } from '../../types'
import { Badge } from '../ui/Badge'

interface SimulationIntelligencePanelProps {
  stats?: SimulationStats | null
  currentStep: number
  scenarioName?: string
  severityCategory?: string
  headline?: string
  coastalImpactLevel?: string
  recommendedAction?: string
}

export function SimulationIntelligencePanel({
  stats,
  currentStep = 6,
  scenarioName = 'Simulation Active',
  severityCategory = 'CRITICAL',
  headline,
  coastalImpactLevel,
  recommendedAction,
}: SimulationIntelligencePanelProps) {
  const isCritical = severityCategory === 'CRITICAL'
  const isHigh = severityCategory === 'HIGH'

  // Dynamic Event Timeline milestones
  const eventMilestones = [
    {
      step: 0,
      title: 'Storm Genesis & Deposition',
      desc: 'Atmospheric instability initiates rapid deep-water swell propagation in central Arabian Sea.',
      status: currentStep >= 0 ? 'COMPLETE' : 'UPCOMING',
    },
    {
      step: 6,
      title: 'Outer Continental Shelf Impact',
      desc: 'First long-period wave trains reach offshore deep-water grids; Hs breaches 3.5m threshold.',
      status: currentStep >= 6 ? 'ACTIVE' : 'UPCOMING',
    },
    {
      step: 12,
      title: 'Regional Gale Acceleration',
      desc: 'Sustained winds exceed 75 km/h. Coastal stations issue small craft warnings and port alerts.',
      status: currentStep >= 12 ? 'ACTIVE' : 'UPCOMING',
    },
    {
      step: 16,
      title: 'Peak Hydrodynamic Force',
      desc: 'Peak wave crests exceed critical 6.0m+ engineering bounds. High surge risk on low-lying estuaries.',
      status: currentStep >= 16 ? 'ACTIVE' : 'UPCOMING',
    },
    {
      step: 20,
      title: 'Nearshore Breaker Shoaling',
      desc: 'Severe shallow-water wave transformation; dangerous rip currents and harbor entrance shoaling.',
      status: currentStep >= 20 ? 'ACTIVE' : 'UPCOMING',
    },
    {
      step: 24,
      title: 'Landfall & Energy Dissipation',
      desc: 'Storm center moves inland or tracks northward; sea state gradually relaxes toward safe baselines.',
      status: currentStep >= 24 ? 'ACTIVE' : 'UPCOMING',
    },
  ]

  return (
    <div className="space-y-4">
      {/* Primary Intelligence Card */}
      <div className="bg-white border border-[#DCE5EF] rounded-2xl p-5 shadow-xs space-y-4">
        {/* Title */}
        <div className="flex items-center justify-between border-b border-[#DCE5EF] pb-3">
          <div>
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#64758A] font-bold block">
              Hydrodynamic Intelligence
            </span>
            <h3 className="font-['Outfit',sans-serif] text-base font-bold text-[#15304B]">
              Real-Time Simulation State
            </h3>
          </div>
          <Badge
            variant={isCritical ? 'destructive' : isHigh ? 'warning' : 'teal'}
            className="font-mono text-[10px] font-bold"
          >
            {severityCategory} RISK
          </Badge>
        </div>

        {/* 4 Quantitative Metrics */}
        <div className="grid grid-cols-2 gap-3 text-xs font-mono">
          <div className="p-3 rounded-xl bg-[#F0F6FC] border border-[#DCE5EF]">
            <span className="text-[10px] text-[#64758A] font-semibold block uppercase">
              Current Wave $H_s$
            </span>
            <span className="text-xl font-bold text-[#15304B]">
              {stats?.max_swh_m != null ? `${Number(stats.max_swh_m).toFixed(2)} m` : '6.74 m'}
            </span>
            <span className="text-[10px] text-[#20AFA7] block mt-0.5 font-medium">Significant Height</span>
          </div>

          <div className="p-3 rounded-xl bg-[#F0F6FC] border border-[#DCE5EF]">
            <span className="text-[10px] text-[#64758A] font-semibold block uppercase">
              Peak Wind Gusts
            </span>
            <span className="text-xl font-bold text-[#245B8F]">
              {stats?.max_wind_kmh != null ? `${Number(stats.max_wind_kmh).toFixed(0)} km/h` : '78 km/h'}
            </span>
            <span className="text-[10px] text-[#64758A] block mt-0.5">Atmospheric Velocity</span>
          </div>

          <div className="p-3 rounded-xl bg-[#F0F6FC] border border-[#DCE5EF]">
            <span className="text-[10px] text-[#64758A] font-semibold block uppercase">
              Coastal Risk Index
            </span>
            <span className="text-xl font-bold text-[#B91C1C]">
              {stats?.coastal_risk_index != null ? `${stats.coastal_risk_index}` : '82'}/100
            </span>
            <span className="text-[10px] text-red-600 block mt-0.5 font-semibold">Elevated Hazard</span>
          </div>

          <div className="p-3 rounded-xl bg-[#F0F6FC] border border-[#DCE5EF]">
            <span className="text-[10px] text-[#64758A] font-semibold block uppercase">
              Active Warnings
            </span>
            <span className="text-xl font-bold text-[#D97706]">
              {stats?.active_alerts ?? 8}
            </span>
            <span className="text-[10px] text-amber-700 block mt-0.5">Automated Alerts</span>
          </div>
        </div>

        {/* Coastal Impact Assessment */}
        <div className="p-3.5 rounded-xl bg-[#FFFBEB] border border-[#FCD34D] text-xs space-y-1">
          <span className="text-[10px] font-mono text-[#92400E] font-bold uppercase block">
            Coastal Impact Classification
          </span>
          <p className="font-semibold text-[#78350F]">
            {stats?.coastal_impact_level || coastalImpactLevel || 'Severe Offshore Wave Action'}
          </p>
        </div>

        {/* Recommended Action */}
        <div className="p-3.5 rounded-xl bg-[#F0F6FC] border border-[#DCE5EF] text-xs space-y-1">
          <span className="text-[10px] font-mono text-[#245B8F] font-bold uppercase block">
            Recommended Maritime Action
          </span>
          <p className="text-[#15304B] leading-relaxed text-[11px]">
            {stats?.recommended_action ||
              recommendedAction ||
              'Complete prohibition of small craft operations, suspension of coastal port lighterage, and reinforcement of exposed breakwaters.'}
          </p>
        </div>
      </div>

      {/* Simulation Event Milestones Timeline */}
      <div className="bg-white border border-[#DCE5EF] rounded-2xl p-5 shadow-xs space-y-3">
        <div className="flex items-center justify-between border-b border-[#DCE5EF] pb-3">
          <h3 className="font-['Outfit',sans-serif] text-sm font-bold text-[#15304B] flex items-center gap-2">
            <Clock className="h-4 w-4 text-[#20AFA7]" />
            <span>Simulation Event Log (24h)</span>
          </h3>
          <span className="text-[10px] font-mono text-[#64758A]">
            Current: T+{String(currentStep).padStart(2, '0')}h
          </span>
        </div>

        <div className="space-y-2.5 max-h-[300px] overflow-y-auto pr-1">
          {eventMilestones.map((milestone) => {
            const isPassed = currentStep >= milestone.step
            const isNow =
              currentStep >= milestone.step &&
              (milestone.step === 24 || currentStep < milestone.step + 4)

            return (
              <div
                key={milestone.step}
                className={`p-2.5 rounded-lg border text-xs transition-colors flex items-start gap-2.5 ${
                  isNow
                    ? 'bg-[#EAF2F9] border-[#245B8F]/40'
                    : isPassed
                    ? 'bg-[#F0F6FC] border-[#DCE5EF]'
                    : 'bg-white border-[#E2E8F0] opacity-50'
                }`}
              >
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold shrink-0 mt-0.5 ${
                    isNow
                      ? 'bg-[#245B8F] text-white'
                      : isPassed
                      ? 'bg-[#DCE5EF] text-[#15304B]'
                      : 'bg-slate-100 text-slate-400'
                  }`}
                >
                  T+{String(milestone.step).padStart(2, '0')}h
                </span>

                <div>
                  <div className="font-bold text-[#15304B]">{milestone.title}</div>
                  <p className="text-[11px] text-[#64758A] leading-snug mt-0.5">
                    {milestone.desc}
                  </p>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

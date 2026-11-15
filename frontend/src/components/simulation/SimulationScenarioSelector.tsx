import React from 'react'
import {
  Flame,
  AlertTriangle,
  Wind,
  Waves,
  ShieldAlert,
  ShieldCheck,
  CheckCircle2,
  Clock,
  ArrowRight,
  Sparkles,
} from 'lucide-react'
import type { ScenarioMetadataItem } from '../../types'
import { Button } from '../ui/Button'
import { Badge } from '../ui/Badge'

interface SimulationScenarioSelectorProps {
  scenarios: ScenarioMetadataItem[]
  activeScenarioId?: string | null
  isLoading?: boolean
  onSelectScenario: (scenarioId: string) => void
}

export function SimulationScenarioSelector({
  scenarios = [],
  activeScenarioId,
  isLoading = false,
  onSelectScenario,
}: SimulationScenarioSelectorProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-['Outfit',sans-serif] text-base font-bold text-[#15304B] flex items-center gap-2">
          <Flame className="h-4 w-4 text-[#D97706]" />
          <span>Select Ocean Hazard Scenario</span>
        </h3>
        <span className="text-xs font-mono text-[#64758A]">
          {scenarios.length} Predefined Scenarios
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {scenarios.map((sc) => {
          const isActive = activeScenarioId === sc.scenario_id
          const isCritical = sc.severity_category === 'CRITICAL'
          const isHigh = sc.severity_category === 'HIGH'

          return (
            <div
              key={sc.scenario_id}
              onClick={() => onSelectScenario(sc.scenario_id)}
              className={`p-4 rounded-xl border text-left cursor-pointer transition-all flex flex-col justify-between ${
                isActive
                  ? 'bg-[#FFFBEB] border-[#F59E0B] shadow-md ring-2 ring-[#F59E0B]/30'
                  : 'bg-white border-[#DCE5EF] hover:border-[#20AFA7]/60 hover:shadow-xs'
              }`}
            >
              <div>
                {/* Header badge & title */}
                <div className="flex items-start justify-between gap-2 mb-2">
                  <Badge
                    variant={isCritical ? 'destructive' : isHigh ? 'warning' : 'teal'}
                    className="text-[10px] font-mono uppercase font-bold"
                  >
                    {sc.severity_category}
                  </Badge>

                  {isActive && (
                    <span className="flex items-center gap-1 text-[10px] font-mono font-bold text-[#92400E] bg-[#FEF3C7] px-2 py-0.5 rounded-full border border-[#FCD34D]">
                      <CheckCircle2 className="h-3 w-3 text-[#D97706]" />
                      Active
                    </span>
                  )}
                </div>

                <h4 className="font-['Outfit',sans-serif] text-sm font-bold text-[#15304B] mb-1">
                  {sc.name}
                </h4>

                <p className="text-[11px] text-[#64758A] line-clamp-2 leading-relaxed mb-3">
                  {sc.description}
                </p>

                {/* Quantitative metrics from backend */}
                <div className="grid grid-cols-2 gap-2 text-xs font-mono p-2.5 rounded-lg bg-[#F0F6FC] border border-[#DCE5EF] mb-3">
                  <div>
                    <span className="text-[9px] text-[#64758A] font-semibold block uppercase">Peak $H_s$</span>
                    <span className="font-bold text-[#15304B] text-sm">
                      {sc.peak_swh_m.toFixed(2)} m
                    </span>
                  </div>
                  <div>
                    <span className="text-[9px] text-[#64758A] font-semibold block uppercase">Peak Wind</span>
                    <span className="font-bold text-[#20AFA7] text-sm">
                      {sc.peak_wind_kmh.toFixed(0)} km/h
                    </span>
                  </div>
                </div>

                {/* Impact Level */}
                <div className="mb-2">
                  <span className="text-[10px] font-mono text-[#64758A] block uppercase font-semibold">Impact Level:</span>
                  <p className="text-xs font-medium text-[#15304B] leading-snug">
                    {sc.coastal_impact_level}
                  </p>
                </div>

                {/* Recommended Action */}
                <div>
                  <span className="text-[10px] font-mono text-[#64758A] block uppercase font-semibold">Recommended Action:</span>
                  <p className="text-[11px] text-[#64758A] leading-snug line-clamp-2">
                    {sc.recommended_action}
                  </p>
                </div>
              </div>

              {/* Action trigger */}
              <div className="pt-3 mt-3 border-t border-[#DCE5EF]">
                <button
                  type="button"
                  disabled={isLoading}
                  className={`w-full py-1.5 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors cursor-pointer ${
                    isActive
                      ? 'bg-[#D97706] text-white hover:bg-[#B45309]'
                      : 'bg-[#F0F6FC] text-[#245B8F] hover:bg-[#EAF2F9] border border-[#DCE5EF]'
                  }`}
                >
                  <span>{isActive ? 'Current Scenario' : 'Run Scenario'}</span>
                  <ArrowRight className="h-3 w-3" />
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

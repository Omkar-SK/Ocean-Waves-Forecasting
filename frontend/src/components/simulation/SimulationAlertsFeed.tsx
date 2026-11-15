import React from 'react'
import {
  AlertTriangle,
  Flame,
  ShieldAlert,
  Clock,
  Waves,
  MapPin,
  CheckCircle2,
  ExternalLink,
} from 'lucide-react'
import type { AlertItem } from '../../types'
import { Badge } from '../ui/Badge'

interface SimulationAlertsFeedProps {
  alerts?: AlertItem[]
  scenarioName?: string
}

export function SimulationAlertsFeed({
  alerts = [],
  scenarioName = 'Hazard Simulation',
}: SimulationAlertsFeedProps) {
  return (
    <div className="bg-white border border-[#DCE5EF] rounded-2xl p-6 shadow-xs space-y-4">
      {/* Header with mandatory hypothetical watermark */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#DCE5EF] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#FFE4E6] text-[#E11D48] border border-[#FECDD3]">
              <AlertTriangle className="h-4 w-4" />
            </span>
            <h3 className="font-['Outfit',sans-serif] text-base font-bold text-[#15304B]">
              Simulated Coastal Hazard Early Warnings
            </h3>
          </div>
          <p className="text-xs text-[#64758A] mt-0.5">
            Real-time threshold breaches automatically triggered by the active simulation model
          </p>
        </div>

        {/* Clear hypothetical badge */}
        <div className="flex items-center gap-2">
          <span className="px-3 py-1 rounded-full bg-[#FEF3C7] border border-[#FCD34D] text-[#92400E] text-[10px] font-mono font-extrabold uppercase tracking-wider">
            Hypothetical Model Output (Zero DB Contamination)
          </span>
        </div>
      </div>

      {/* Alerts list */}
      {alerts.length === 0 ? (
        <div className="p-8 text-center rounded-xl bg-[#F0F6FC] border border-[#DCE5EF] text-xs text-[#64758A]">
          <CheckCircle2 className="h-6 w-6 text-[#10B981] mx-auto mb-2" />
          <p className="font-semibold text-[#15304B]">No Active Threshold Breaches</p>
          <p className="text-[11px] mt-0.5">All coastal nodes are within safe engineering operational limits for this simulation hour.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {alerts.map((alt) => {
            const isRed = alt.severity === 'RED' || (alt.severity as string) === 'CRITICAL'
            const isOrange = alt.severity === 'ORANGE' || (alt.severity as string) === 'HIGH'

            return (
              <div
                key={alt.alert_id}
                className={`p-4 rounded-xl border text-xs space-y-2.5 transition-all shadow-2xs ${
                  isRed
                    ? 'bg-white border-[#FECDD3] border-l-4 border-l-[#E11D48]'
                    : isOrange
                    ? 'bg-white border-[#FDE68A] border-l-4 border-l-[#D97706]'
                    : 'bg-white border-[#DCE5EF] border-l-4 border-l-[#EAB308]'
                }`}
              >
                {/* Station name & severity */}
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-1.5 font-bold text-[#15304B]">
                    <MapPin className="h-3.5 w-3.5 text-[#245B8F] shrink-0" />
                    <span className="truncate">{alt.location_name}</span>
                  </div>

                  <Badge
                    variant={isRed ? 'destructive' : isOrange ? 'warning' : 'teal'}
                    className="text-[9px] font-mono font-bold uppercase shrink-0"
                  >
                    {alt.severity} RISK
                  </Badge>
                </div>

                {/* Trigger reason */}
                <p className="text-[#192B3C] font-medium leading-relaxed text-[11px]">
                  {alt.trigger_reason}
                </p>

                {/* Metrics */}
                <div className="grid grid-cols-3 gap-2 text-center font-mono text-[10px] p-2 rounded-lg bg-[#F0F6FC] border border-[#DCE5EF]">
                  <div>
                    <span className="text-[#64758A] block uppercase">Predicted $H_s$</span>
                    <span className="font-bold text-[#15304B] text-xs">
                      {alt.predicted_swh_m?.toFixed(2) ?? '—'} m
                    </span>
                  </div>
                  <div>
                    <span className="text-[#64758A] block uppercase">Threshold</span>
                    <span className="font-bold text-[#D97706] text-xs">
                      {alt.threshold_m?.toFixed(2) ?? '—'} m
                    </span>
                  </div>
                  <div>
                    <span className="text-[#64758A] block uppercase">Horizon</span>
                    <span className="font-bold text-[#245B8F] text-xs">
                      +{alt.forecast_horizon_hours}h
                    </span>
                  </div>
                </div>

                {/* Footnote */}
                <div className="flex items-center justify-between text-[10px] font-mono text-[#64758A] pt-1">
                  <span>Status: <strong className="text-[#E11D48]">{alt.status}</strong></span>
                  <span>Event Time: {alt.expected_event_time_utc ? new Date(alt.expected_event_time_utc).toLocaleTimeString() : 'Immediate'}</span>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

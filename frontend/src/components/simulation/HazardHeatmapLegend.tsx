import React from 'react'
import { Layers, Flame, Eye, EyeOff, ShieldAlert } from 'lucide-react'
import { useSimulation } from '../../context/SimulationContext'

interface HazardHeatmapLegendProps {
  showHeatmap: boolean
  onToggleHeatmap: () => void
}

export function HazardHeatmapLegend({ showHeatmap, onToggleHeatmap }: HazardHeatmapLegendProps) {
  const { isSimulationActive, activeScenario, simulationData } = useSimulation()

  return (
    <div className="absolute top-4 right-4 z-20 max-w-xs bg-white/95 backdrop-blur-md rounded-xl border border-[#DCE5EF] shadow-md p-3 text-xs font-sans text-[#15304B]">
      <div className="flex items-center justify-between gap-2 border-b border-[#DCE5EF] pb-2 mb-2">
        <div className="flex items-center gap-1.5 font-bold">
          <Layers className="h-4 w-4 text-[#245B8F]" />
          <span>Coastal Hazard Heatmap</span>
        </div>

        <button
          onClick={onToggleHeatmap}
          className={`flex items-center gap-1 px-2 py-0.5 rounded-md font-medium text-[11px] transition-colors border ${
            showHeatmap
              ? 'bg-[#EAF2F9] border-[#B9D3EC] text-[#245B8F]'
              : 'bg-white border-[#DCE5EF] text-[#64758A] hover:bg-[#F4F7FB]'
          }`}
          title={showHeatmap ? 'Hide heatmap layer' : 'Show heatmap layer'}
        >
          {showHeatmap ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
          <span>{showHeatmap ? 'ON' : 'OFF'}</span>
        </button>
      </div>

      {isSimulationActive ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[11px] bg-[#FEF3C7] border border-[#FCD34D] px-2 py-1 rounded-md text-[#92400E]">
            <div className="flex items-center gap-1 font-semibold truncate">
              <Flame className="h-3 w-3 text-[#D97706] shrink-0" />
              <span className="truncate">{activeScenario?.name || 'Hazard Simulation'}</span>
            </div>
            <span className="text-[10px] font-mono shrink-0 uppercase font-bold">Active</span>
          </div>

          <div className="space-y-1 text-[11px]">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-[#EF4444]" />
                <span className="font-medium text-[#15304B]">Critical Risk</span>
              </span>
              <span className="font-mono text-[#64758A]">&ge; 4.0m</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-[#F97316]" />
                <span className="font-medium text-[#15304B]">High Risk</span>
              </span>
              <span className="font-mono text-[#64758A]">2.5 &ndash; 4.0m</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-[#F59E0B]" />
                <span className="font-medium text-[#15304B]">Caution</span>
              </span>
              <span className="font-mono text-[#64758A]">1.5 &ndash; 2.5m</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-[#10B981]" />
                <span className="font-medium text-[#15304B]">Low Risk</span>
              </span>
              <span className="font-mono text-[#64758A]">&lt; 1.5m</span>
            </div>
          </div>

          {simulationData?.stats && (
            <div className="pt-1.5 border-t border-[#DCE5EF] text-[10px] text-[#64758A] flex justify-between">
              <span>Peak Swell: <strong className="text-[#EF4444] font-mono">{simulationData.stats.max_swh_m}m</strong></span>
              <span>Gusts: <strong className="text-[#15304B] font-mono">{simulationData.stats.max_wind_kmh}km/h</strong></span>
            </div>
          )}
        </div>
      ) : (
        <div className="text-[11px] text-[#64758A] space-y-1">
          <p>Heatmap overlays are activated during <strong>Simulation Mode</strong> to display synthetic coastal hazard plumes.</p>
          <div className="flex items-center gap-1 text-[10px] text-[#245B8F] font-semibold mt-1">
            <ShieldAlert className="h-3 w-3" />
            <span>Click &apos;Simulate&apos; to view storm plumes</span>
          </div>
        </div>
      )}
    </div>
  )
}


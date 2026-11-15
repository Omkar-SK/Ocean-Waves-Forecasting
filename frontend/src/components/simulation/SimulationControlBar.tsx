import React from 'react'
import { Link } from 'react-router-dom'
import {
  Flame,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  Sliders,
  AlertTriangle,
  Info,
  Waves,
  FlaskConical,
} from 'lucide-react'
import { useSimulation } from '../../context/SimulationContext'
import { Button } from '../ui/Button'

export function SimulationControlBar() {
  const {
    isSimulationActive,
    simulationState,
    activeScenario,
    simulationData,
    currentStep,
    totalSteps,
    pauseSimulation,
    resumeSimulation,
    stepSimulation,
    reset,
    openModal,
    isLoading,
  } = useSimulation()

  if (!isSimulationActive || !activeScenario) {
    return null
  }

  const isPaused = simulationState === 'PAUSED'

  return (
    <aside
      aria-label="Hazard simulation controls"
      className="bg-linear-to-r from-[#FFFBEB] via-[#FEF3C7] to-[#FFFBEB] border-b border-[#FCD34D] px-4 py-2.5 shadow-xs transition-all animate-ocean-fade"
    >
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3 text-xs">
        {/* Left: Mode Badge & Scenario Summary */}
        <div className="flex items-center gap-3 w-full md:w-auto">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#D97706] text-white font-extrabold tracking-wide uppercase text-[11px] shadow-xs shrink-0">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
            </span>
            <span>SIMULATION MODE ACTIVE</span>
          </div>

          <div className="truncate">
            <div className="flex items-center gap-1.5 font-bold text-[#92400E]">
              <Flame className="h-3.5 w-3.5 text-[#D97706] shrink-0" />
              <span className="truncate">{activeScenario.name}</span>
              <span className="hidden lg:inline text-[11px] font-normal text-[#B45309]">
                &bull; {activeScenario.headline}
              </span>
            </div>
          </div>
        </div>

        {/* Center: Timeline Progress & Stepper Controls */}
        <div className="flex items-center gap-2 bg-white/80 border border-[#FDE68A] px-3 py-1 rounded-lg shadow-2xs">
          <span className="font-mono font-bold text-[#92400E] text-xs">
            T+{String(currentStep).padStart(2, '0')}h
          </span>
          <span className="text-[#B45309] text-[11px]">/ 24h</span>

          <div className="h-3.5 w-px bg-[#FCD34D] mx-1" />

          {/* Step Backward */}
          <button
            onClick={() => stepSimulation(-1)}
            disabled={isLoading || currentStep <= 0}
            className="p-1 rounded text-[#92400E] hover:bg-[#FEF3C7] disabled:opacity-40 transition-colors"
            title="Step backward (-1h)"
            aria-label="Step backward 1 hour"
          >
            <SkipBack className="h-3.5 w-3.5" />
          </button>

          {/* Pause / Resume */}
          <button
            onClick={isPaused ? resumeSimulation : pauseSimulation}
            disabled={isLoading}
            className="p-1 rounded text-[#92400E] hover:bg-[#FEF3C7] transition-colors"
            title={isPaused ? 'Resume simulation progression' : 'Pause simulation progression'}
            aria-label={isPaused ? 'Resume simulation' : 'Pause simulation'}
          >
            {isPaused ? <Play className="h-3.5 w-3.5 fill-[#92400E]" /> : <Pause className="h-3.5 w-3.5 fill-[#92400E]" />}
          </button>

          {/* Step Forward */}
          <button
            onClick={() => stepSimulation(1)}
            disabled={isLoading || currentStep >= totalSteps}
            className="p-1 rounded text-[#92400E] hover:bg-[#FEF3C7] disabled:opacity-40 transition-colors"
            title="Step forward (+1h)"
            aria-label="Step forward 1 hour"
          >
            <SkipForward className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Right: Key Stats & Exit Trigger */}
        <div className="flex items-center gap-2.5 w-full md:w-auto justify-end">
          {simulationData?.stats && (
            <div className="hidden sm:flex items-center gap-3 text-[11px] text-[#92400E] font-medium mr-1">
              <span>Peak: <strong className="font-mono text-[#B91C1C]">{simulationData.stats.max_swh_m}m</strong></span>
              <span>Wind: <strong className="font-mono">{simulationData.stats.max_wind_kmh} km/h</strong></span>
              <span>Risk: <strong className="font-mono text-[#B91C1C]">{simulationData.stats.coastal_risk_index}/100</strong></span>
            </div>
          )}

          <Link
            to="/simulation"
            className="px-2.5 py-1 text-xs font-semibold rounded-md bg-[#245B8F] text-white hover:bg-[#15304B] transition-all flex items-center gap-1 shadow-2xs"
          >
            <FlaskConical className="h-3 w-3" />
            <span>Simulation Lab</span>
          </Link>

          <button
            onClick={openModal}
            className="px-2.5 py-1 text-xs font-semibold rounded-md bg-white border border-[#FCD34D] text-[#92400E] hover:bg-[#FEF3C7] transition-all flex items-center gap-1 shadow-2xs"
          >
            <Sliders className="h-3 w-3" />
            <span>Scenario</span>
          </button>

          <Button
            size="sm"
            variant="outline"
            onClick={reset}
            disabled={isLoading}
            className="text-xs bg-white text-[#B91C1C] border-[#FCA5A5] hover:bg-[#FEF2F2] hover:border-[#EF4444] px-2.5 py-1 h-auto font-semibold flex items-center gap-1"
          >
            <RotateCcw className="h-3 w-3" />
            <span>Return to Live Data</span>
          </Button>
        </div>
      </div>

      {/* Floating Disclaimer */}
      <div className="max-w-7xl mx-auto mt-1 flex items-center gap-1 text-[10px] text-[#B45309]">
        <Info className="h-2.5 w-2.5 shrink-0" />
        <span className="truncate">
          Hypothetical hazard scenario active for decision-support and stress-testing. Actual marine telemetry remains unmutated.
        </span>
      </div>
    </aside>
  )
}


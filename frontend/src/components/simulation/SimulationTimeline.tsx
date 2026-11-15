import React from 'react'
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  Clock,
  Flame,
  Activity,
  Layers,
} from 'lucide-react'
import { Button } from '../ui/Button'

interface SimulationTimelineProps {
  currentStep: number
  totalSteps?: number
  isPaused: boolean
  isLoading?: boolean
  scenarioName?: string
  onPlayResume: () => void
  onPause: () => void
  onStepForward: () => void
  onStepBackward: () => void
  onReset: () => void
  onSelectStep: (step: number) => void
}

export function SimulationTimeline({
  currentStep = 6,
  totalSteps = 24,
  isPaused,
  isLoading = false,
  scenarioName,
  onPlayResume,
  onPause,
  onStepForward,
  onStepBackward,
  onReset,
  onSelectStep,
}: SimulationTimelineProps) {
  // Determine storm phase
  const phaseLabel =
    currentStep <= 6
      ? 'Phase 1: Genesis & Swell Growth (0-6h)'
      : currentStep <= 14
      ? 'Phase 2: Rapid Intensification (6-14h)'
      : currentStep <= 18
      ? 'Phase 3: Peak Hazard & Nearshore Impact (14-18h)'
      : 'Phase 4: Shoaling & System Dissipation (18-24h)'

  return (
    <div className="bg-white border border-[#DCE5EF] rounded-2xl p-5 shadow-xs space-y-4">
      {/* Top row: Timeline status & Phase */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#DCE5EF] pb-3">
        <div className="flex items-center gap-2.5">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#FFFBEB] text-[#D97706] border border-[#FDE68A]">
            <Clock className="h-4 w-4" />
          </span>
          <div>
            <span className="text-xs font-mono font-bold text-[#15304B] block">
              24-Hour Storm Evolution Timeline
            </span>
            <span className="text-[11px] text-[#D97706] font-semibold">{phaseLabel}</span>
          </div>
        </div>

        <div className="flex items-center gap-2 self-end sm:self-auto font-mono text-xs">
          <span className="px-2.5 py-1 rounded bg-[#F0F6FC] border border-[#DCE5EF] text-[#15304B] font-bold">
            T+{String(currentStep).padStart(2, '0')}h / 24h
          </span>
        </div>
      </div>

      {/* Scrub Bar / Timeline Steps Grid */}
      <div className="space-y-1.5">
        <div className="relative w-full flex items-center gap-1">
          {Array.from({ length: totalSteps + 1 }).map((_, i) => {
            const isCurrent = i === currentStep
            const isPast = i < currentStep
            const isCriticalZone = i >= 12 && i <= 18

            return (
              <button
                key={i}
                type="button"
                onClick={() => onSelectStep(i)}
                disabled={isLoading}
                title={`Jump to T+${i}h`}
                className={`flex-1 h-8 rounded-sm transition-all cursor-pointer relative group flex flex-col items-center justify-center ${
                  isCurrent
                    ? 'bg-[#D97706] text-white font-bold ring-2 ring-[#D97706]/40 scale-105 z-10'
                    : isPast
                    ? isCriticalZone
                      ? 'bg-[#FCA5A5] hover:bg-[#F87171]'
                      : 'bg-[#93C5FD] hover:bg-[#60A5FA]'
                    : isCriticalZone
                    ? 'bg-[#FEE2E2] hover:bg-[#FECDD3]'
                    : 'bg-[#F0F6FC] hover:bg-[#E2E8F0]'
                }`}
              >
                <span className="text-[9px] font-mono leading-none">
                  {i % 4 === 0 || i === currentStep ? `${i}` : ''}
                </span>
              </button>
            )
          })}
        </div>

        {/* Phase Markers under scrub bar */}
        <div className="flex justify-between text-[10px] font-mono text-[#64758A] pt-1">
          <span>T+00h Genesis</span>
          <span>T+06h Deep Sea</span>
          <span>T+14h Peak Landfall</span>
          <span>T+18h Breakers</span>
          <span>T+24h Dissipation</span>
        </div>
      </div>

      {/* Control Buttons Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
        <div className="flex items-center gap-1.5">
          {/* Step Backward (-1h) */}
          <Button
            size="sm"
            variant="secondary"
            onClick={onStepBackward}
            disabled={isLoading || currentStep <= 0}
            className="text-xs flex items-center gap-1"
            title="Step backward 1 hour"
          >
            <SkipBack className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Step -1h</span>
          </Button>

          {/* Play / Pause Toggle */}
          <Button
            size="sm"
            variant={isPaused ? 'primary' : 'secondary'}
            onClick={isPaused ? onPlayResume : onPause}
            disabled={isLoading}
            className={`text-xs font-semibold flex items-center gap-1.5 ${
              isPaused
                ? 'bg-linear-to-r from-[#245B8F] to-[#15304B] text-white'
                : 'bg-amber-100 text-amber-900 border-amber-300'
            }`}
          >
            {isPaused ? (
              <>
                <Play className="h-3.5 w-3.5 fill-current" />
                <span>Resume Evolution</span>
              </>
            ) : (
              <>
                <Pause className="h-3.5 w-3.5 fill-current" />
                <span>Pause</span>
              </>
            )}
          </Button>

          {/* Step Forward (+1h) */}
          <Button
            size="sm"
            variant="secondary"
            onClick={onStepForward}
            disabled={isLoading || currentStep >= totalSteps}
            className="text-xs flex items-center gap-1"
            title="Step forward 1 hour"
          >
            <span className="hidden sm:inline">Step +1h</span>
            <SkipForward className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* Reset / Exit Simulation Button */}
        <div>
          <Button
            size="sm"
            variant="outline"
            onClick={onReset}
            disabled={isLoading}
            className="text-xs bg-white text-[#B91C1C] border-[#FCA5A5] hover:bg-[#FEF2F2] flex items-center gap-1.5 font-semibold"
          >
            <RotateCcw className="h-3.5 w-3.5 text-[#B91C1C]" />
            <span>Reset & Restore Live Data</span>
          </Button>
        </div>
      </div>
    </div>
  )
}

import React, { useState, useEffect } from 'react'
import {
  X,
  FlaskConical,
  Flame,
  AlertTriangle,
  Wind,
  Waves,
  Clock,
  ShieldCheck,
  ShieldAlert,
  RotateCcw,
  Sliders,
  CheckCircle2,
} from 'lucide-react'
import { useSimulation } from '../../context/SimulationContext'
import { Button } from '../ui/Button'
import { Badge } from '../ui/Badge'

export function SimulationModal() {
  const {
    isModalOpen,
    closeModal,
    scenarios,
    activeScenario,
    isSimulationActive,
    startSimulation,
    reset,
    isLoading,
  } = useSimulation()

  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('severe_wave')
  const [initialStep, setInitialStep] = useState<number>(6)

  // Sync selected scenario with active when modal opens
  useEffect(() => {
    if (activeScenario) {
      setSelectedScenarioId(activeScenario.scenario_id)
    } else if (scenarios.length > 0 && !selectedScenarioId) {
      setSelectedScenarioId(scenarios[0].scenario_id)
    }
  }, [activeScenario, scenarios, selectedScenarioId])

  if (!isModalOpen) return null

  const handleStart = async () => {
    if (!selectedScenarioId) return
    await startSimulation(selectedScenarioId, initialStep)
  }

  const handleExitSimulation = async () => {
    await reset()
    closeModal()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#15304B]/60 backdrop-blur-xs animate-ocean-fade">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="simulation-modal-title"
        className="relative w-full max-w-4xl max-h-[90vh] flex flex-col bg-white rounded-2xl shadow-2xl border border-[#DCE5EF] overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#DCE5EF] bg-linear-to-r from-[#F4F7FB] via-white to-[#F0F6FC]">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-linear-to-br from-[#245B8F] to-[#15304B] text-white shadow-xs">
              <FlaskConical className="h-5 w-5" />
            </div>
            <div>
              <h2 id="simulation-modal-title" className="text-lg font-bold text-[#15304B] tracking-tight">
                Interactive Ocean Hazard Simulation Mode
              </h2>
              <p className="text-xs text-[#64758A]">
                Stress-test OceanCast AI with hypothetical dangerous marine-weather scenarios
              </p>
            </div>
          </div>

          <button
            onClick={closeModal}
            className="p-1.5 text-[#64758A] hover:text-[#15304B] hover:bg-[#EAF2F9] rounded-lg transition-colors"
            title="Close modal"
            aria-label="Close modal"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Disclaimer Bar */}
        <div className="px-6 py-2.5 bg-[#FEF3C7] border-b border-[#FCD34D] text-[#92400E] text-xs flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 shrink-0 text-[#D97706]" />
          <span>
            <strong>Scientific & Safety Notice:</strong> Simulation scenarios generate hypothetical environmental data for engineering validation. They do NOT mutate live database records or represent official meteorological advisories.
          </span>
        </div>

        {/* Body: Scenario Cards & Timeline Slider */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Scenario Selection Grid */}
          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-[#64758A] mb-3">
              1. Select Hypothetical Marine Scenario
            </label>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {scenarios.map((sc) => {
                const isSelected = selectedScenarioId === sc.scenario_id
                const isCritical = sc.severity_category === 'CRITICAL'
                const isHigh = sc.severity_category === 'HIGH'

                return (
                  <div
                    key={sc.scenario_id}
                    onClick={() => setSelectedScenarioId(sc.scenario_id)}
                    className={`relative p-4 rounded-xl border-2 cursor-pointer transition-all ${
                      isSelected
                        ? 'border-[#245B8F] bg-[#F0F6FC] shadow-sm'
                        : 'border-[#DCE5EF] bg-white hover:border-[#B9D3EC] hover:bg-[#F8FAFC]'
                    }`}
                  >
                    {/* Top Row: Title & Severity */}
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div>
                        <div className="text-[11px] font-semibold text-[#64758A] uppercase tracking-wider">
                          {sc.category}
                        </div>
                        <h3 className="font-bold text-[#15304B] text-sm leading-snug">
                          {sc.name}
                        </h3>
                      </div>

                      <Badge
                        variant={isCritical ? 'destructive' : isHigh ? 'warning' : 'cyan'}
                        className="text-[10px] font-mono uppercase font-bold shrink-0"
                      >
                        {sc.severity_category}
                      </Badge>
                    </div>

                    {/* Headline */}
                    <p className="text-xs text-[#245B8F] font-medium mb-3">
                      {sc.headline}
                    </p>

                    {/* Metrics Preview Chips */}
                    <div className="grid grid-cols-3 gap-2 py-2 px-3 rounded-lg bg-white border border-[#DCE5EF] text-center mb-3">
                      <div>
                        <div className="text-[10px] text-[#64758A]">Peak SWH</div>
                        <div className="text-xs font-bold font-mono text-[#EF4444]">
                          {sc.peak_swh_m}m
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-[#64758A]">Max Wind</div>
                        <div className="text-xs font-bold font-mono text-[#15304B]">
                          {sc.peak_wind_kmh} km/h
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-[#64758A]">Peak Period</div>
                        <div className="text-xs font-bold font-mono text-[#245B8F]">
                          {sc.peak_mwp_s}s
                        </div>
                      </div>
                    </div>

                    {/* Description & Impact */}
                    <p className="text-xs text-[#475569] leading-relaxed mb-2">
                      {sc.description}
                    </p>

                    <div className="text-[11px] text-[#047857] bg-[#ECFDF5] border border-[#A7F3D0] p-2 rounded-md">
                      <strong>Recommended Action:</strong> {sc.recommended_action}
                    </div>

                    {/* Radio checkmark */}
                    {isSelected && (
                      <div className="absolute top-3 right-3 h-5 w-5 rounded-full bg-[#245B8F] text-white flex items-center justify-center">
                        <CheckCircle2 className="h-4 w-4" />
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Initial Timeline Step */}
          <div className="p-4 rounded-xl bg-[#F4F7FB] border border-[#DCE5EF] space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-xs font-bold uppercase tracking-wider text-[#15304B]">
                  2. Initial Storm Timeline Offset
                </label>
                <p className="text-xs text-[#64758A]">
                  Select where along the 24-hour storm progression curve the simulation will initialize
                </p>
              </div>

              <div className="px-3 py-1 bg-white border border-[#DCE5EF] rounded-lg font-mono font-bold text-sm text-[#245B8F]">
                T+{String(initialStep).padStart(2, '0')}h
              </div>
            </div>

            <input
              type="range"
              min="0"
              max="24"
              step="1"
              value={initialStep}
              onChange={(e) => setInitialStep(Number(e.target.value))}
              className="w-full accent-[#245B8F] cursor-pointer"
            />

            <div className="flex justify-between text-[10px] text-[#64758A] font-mono">
              <span>T+00h (Onset)</span>
              <span>T+06h (Initial Hazard)</span>
              <span>T+14h (Storm Peak)</span>
              <span>T+24h (Dissipation)</span>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-[#DCE5EF] bg-[#F4F7FB]">
          <div>
            {isSimulationActive && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleExitSimulation}
                disabled={isLoading}
                className="text-[#B91C1C] border-[#FCA5A5] hover:bg-[#FEF2F2] flex items-center gap-1.5"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                <span>Exit Simulation & Return to Live Data</span>
              </Button>
            )}
          </div>

          <div className="flex items-center gap-3">
            <Button
              variant="secondary"
              onClick={closeModal}
              disabled={isLoading}
            >
              Cancel
            </Button>

            <Button
              onClick={handleStart}
              disabled={isLoading || !selectedScenarioId}
              isLoading={isLoading}
              className="bg-linear-to-r from-[#245B8F] to-[#1D4A75] hover:from-[#1D4A75] hover:to-[#15304B] text-white shadow-md font-semibold px-5 flex items-center gap-2"
            >
              <FlaskConical className="h-4 w-4" />
              <span>{isSimulationActive ? 'Update Simulation Scenario' : 'Run Hazard Simulation'}</span>
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

import React, { useState, useEffect, useCallback } from 'react'
import {
  FlaskConical,
  Flame,
  Layers,
  Activity,
  RotateCcw,
  Sliders,
  MapPin,
  Waves,
  Wind,
  Compass,
  AlertTriangle,
  Info,
  CheckCircle2,
  Maximize2,
  Eye,
  BarChart2,
} from 'lucide-react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from 'recharts'
import { useSimulation } from '../context/SimulationContext'
import { SimulationOcean3D } from '../components/simulation/SimulationOcean3D'
import { SimulationHeatmapLeaflet } from '../components/simulation/SimulationHeatmapLeaflet'
import { SimulationTimeline } from '../components/simulation/SimulationTimeline'
import { SimulationIntelligencePanel } from '../components/simulation/SimulationIntelligencePanel'
import { SimulationScenarioSelector } from '../components/simulation/SimulationScenarioSelector'
import { SimulationAlertsFeed } from '../components/simulation/SimulationAlertsFeed'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'

export function SimulationLabPage() {
  const {
    isSimulationActive,
    simulationState,
    simulationStatus,
    activeScenario,
    simulationData,
    scenarios,
    currentStep,
    totalSteps,
    isLoading,
    startSimulation,
    pauseSimulation,
    resumeSimulation,
    stepSimulation,
    reset,
  } = useSimulation()

  // View Mode: '3d' | 'heatmap' | 'split'
  const [viewMode, setViewMode] = useState<'3d' | 'heatmap' | 'split'>('3d')
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)

  // If simulation is not active on initial visit, auto-start default scenario so lab is immediately live
  useEffect(() => {
    if (!isSimulationActive && scenarios.length > 0 && !isLoading) {
      startSimulation('severe_wave', 6)
    }
  }, [isSimulationActive, scenarios, isLoading, startSimulation])

  const handleScenarioChange = useCallback(
    async (scenarioId: string) => {
      await startSimulation(scenarioId, currentStep)
    },
    [startSimulation, currentStep]
  )

  const handleStepJump = useCallback(
    async (step: number) => {
      if (activeScenario) {
        await startSimulation(activeScenario.scenario_id, step)
      } else {
        await startSimulation('severe_wave', step)
      }
    },
    [activeScenario, startSimulation]
  )

  const isPaused = simulationState === 'PAUSED'

  // Extract primary telemetry metrics from simulationData
  const currentSwh = simulationData?.stats?.max_swh_m ?? activeScenario?.peak_swh_m ?? 5.5
  const currentWind = simulationData?.stats?.max_wind_kmh ?? activeScenario?.peak_wind_kmh ?? 75
  const currentMwp = activeScenario?.peak_mwp_s ?? 14.5
  const severityCategory = simulationData?.stats?.severity_category ?? activeScenario?.severity_category ?? 'CRITICAL'

  // Chart data from time_series
  const timeSeriesData = simulationData?.time_series ?? []

  return (
    <div className="space-y-6 pb-12 animate-ocean-fade">
      {/* 1. Header Banner & View Controls */}
      <div className="bg-white p-6 rounded-2xl border border-[#DCE5EF] shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2.5">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-linear-to-br from-[#245B8F] to-[#15304B] text-white shadow-xs">
                <FlaskConical className="h-4 w-4" />
              </span>
              <h1 className="text-xl font-bold tracking-tight text-[#15304B] font-['Outfit',sans-serif]">
                Ocean Hazard Simulation Laboratory
              </h1>
              <span className="px-2.5 py-0.5 rounded-full bg-[#FEF3C7] border border-[#FCD34D] text-[#92400E] text-[10px] font-mono font-bold uppercase tracking-wider">
                Hypothetical Environment
              </span>
            </div>
            <p className="mt-1 text-xs text-[#64758A] max-w-3xl leading-relaxed">
              Interactive 3D hydrodynamic environment and spatial risk model evaluating Arabian Sea swell dynamics, storm surge progression, and early-warning alerts across 8 coastal candidate nodes.
            </p>
          </div>

          {/* View Mode Switcher & Reset Trigger */}
          <div className="flex flex-wrap items-center gap-2.5">
            <div className="flex bg-[#F0F6FC] p-1 rounded-xl border border-[#DCE5EF]">
              <button
                onClick={() => setViewMode('3d')}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer ${
                  viewMode === '3d'
                    ? 'bg-white text-[#245B8F] border border-[#DCE5EF] shadow-xs'
                    : 'text-[#64758A] hover:text-[#15304B]'
                }`}
              >
                <Waves className="h-3.5 w-3.5" />
                <span>3D Ocean Dynamics</span>
              </button>

              <button
                onClick={() => setViewMode('heatmap')}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer ${
                  viewMode === 'heatmap'
                    ? 'bg-white text-[#245B8F] border border-[#DCE5EF] shadow-xs'
                    : 'text-[#64758A] hover:text-[#15304B]'
                }`}
              >
                <Layers className="h-3.5 w-3.5" />
                <span>Geospatial Heatmap</span>
              </button>

              <button
                onClick={() => setViewMode('split')}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer ${
                  viewMode === 'split'
                    ? 'bg-white text-[#245B8F] border border-[#DCE5EF] shadow-xs'
                    : 'text-[#64758A] hover:text-[#15304B]'
                }`}
              >
                <Maximize2 className="h-3.5 w-3.5" />
                <span>Dual Split</span>
              </button>
            </div>

            <Button
              size="sm"
              variant="outline"
              onClick={reset}
              disabled={isLoading}
              className="text-xs bg-white text-[#B91C1C] border-[#FCA5A5] hover:bg-[#FEF2F2] flex items-center gap-1.5 font-semibold"
            >
              <RotateCcw className="h-3.5 w-3.5 text-[#B91C1C]" />
              <span>Restore Live Data</span>
            </Button>
          </div>
        </div>

        {/* Scientific Disclaimer Strip */}
        <div className="mt-4 pt-3 border-t border-[#DCE5EF] flex items-center gap-2 text-xs text-[#92400E] bg-[#FFFBEB] p-2.5 rounded-lg border border-[#FCD34D]">
          <AlertTriangle className="h-4 w-4 shrink-0 text-[#D97706]" />
          <span className="leading-tight">
            <strong>Data Isolation Guarantee:</strong> Simulation scenarios run entirely in-memory using validated geophysical equations. Live observational caches, SQLite databases, and trained neural networks remain 100% unmutated.
          </span>
        </div>
      </div>

      {/* 2. Main Visualization & Intelligence Panel Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Central Interactive Visualization (8 Cols) */}
        <div className="lg:col-span-8 space-y-6">
          {viewMode === '3d' && (
            <SimulationOcean3D
              swh={currentSwh}
              windKmh={currentWind}
              mwpSeconds={currentMwp}
              scenarioName={activeScenario?.name ?? 'Severe Wave Conditions'}
              severityCategory={severityCategory}
              currentStep={currentStep}
              nodes={simulationData?.nodes ?? []}
              heatmapPoints={simulationData?.heatmap_points ?? []}
              onSelectNode={(nodeId) => setSelectedNodeId(nodeId)}
            />
          )}

          {viewMode === 'heatmap' && (
            <SimulationHeatmapLeaflet
              nodes={simulationData?.nodes ?? []}
              heatmapPoints={simulationData?.heatmap_points ?? []}
              selectedNodeId={selectedNodeId}
              onSelectNode={(nodeId) => setSelectedNodeId(nodeId)}
            />
          )}

          {viewMode === 'split' && (
            <div className="space-y-4">
              <SimulationOcean3D
                swh={currentSwh}
                windKmh={currentWind}
                mwpSeconds={currentMwp}
                scenarioName={activeScenario?.name ?? 'Severe Wave Conditions'}
                severityCategory={severityCategory}
                currentStep={currentStep}
                nodes={simulationData?.nodes ?? []}
                heatmapPoints={simulationData?.heatmap_points ?? []}
                onSelectNode={(nodeId) => setSelectedNodeId(nodeId)}
              />
              <SimulationHeatmapLeaflet
                nodes={simulationData?.nodes ?? []}
                heatmapPoints={simulationData?.heatmap_points ?? []}
                selectedNodeId={selectedNodeId}
                onSelectNode={(nodeId) => setSelectedNodeId(nodeId)}
              />
            </div>
          )}

          {/* Interactive 24-Hour Timeline */}
          <SimulationTimeline
            currentStep={currentStep}
            totalSteps={totalSteps}
            isPaused={isPaused}
            isLoading={isLoading}
            scenarioName={activeScenario?.name}
            onPlayResume={resumeSimulation}
            onPause={pauseSimulation}
            onStepForward={() => stepSimulation(1)}
            onStepBackward={() => stepSimulation(-1)}
            onReset={reset}
            onSelectStep={handleStepJump}
          />
        </div>

        {/* Right: Hydrodynamic Intelligence Panel (4 Cols) */}
        <div className="lg:col-span-4">
          <SimulationIntelligencePanel
            stats={simulationData?.stats}
            currentStep={currentStep}
            scenarioName={activeScenario?.name}
            severityCategory={severityCategory}
            headline={activeScenario?.headline}
            coastalImpactLevel={activeScenario?.coastal_impact_level}
            recommendedAction={activeScenario?.recommended_action}
          />
        </div>
      </div>

      {/* 3. Scenario Selection Grid */}
      <SimulationScenarioSelector
        scenarios={scenarios}
        activeScenarioId={activeScenario?.scenario_id}
        isLoading={isLoading}
        onSelectScenario={handleScenarioChange}
      />

      {/* 4. Multi-Horizon Time-Series Forecast Progression Chart */}
      {timeSeriesData.length > 0 && (
        <div className="bg-white border border-[#DCE5EF] rounded-2xl p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between border-b border-[#DCE5EF] pb-3">
            <div>
              <h3 className="font-['Outfit',sans-serif] text-base font-bold text-[#15304B] flex items-center gap-2">
                <BarChart2 className="h-4 w-4 text-[#20AFA7]" />
                <span>Simulated Multi-Model Wave Height Progression (24h)</span>
              </h3>
              <p className="text-xs text-[#64758A] mt-0.5">
                Comparison of WaveLSTM, WaveANN, WaveCNN1D, and Persistence along the active scenario trajectory
              </p>
            </div>
            <Badge variant="teal" className="font-mono text-[10px]">
              T+00h to T+24h Trajectory
            </Badge>
          </div>

          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timeSeriesData} margin={{ top: 10, right: 20, left: -10, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="time" stroke="#64758A" fontSize={11} tickLine={false} dy={8} />
                <YAxis stroke="#64758A" fontSize={11} tickLine={false} unit="m" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    borderColor: '#DCE5EF',
                    borderRadius: '8px',
                    color: '#15304B',
                    boxShadow: '0 4px 6px -1px rgba(0,0,0,0.07)',
                    fontFamily: 'var(--font-mono)',
                  }}
                  formatter={(val: any, name: any) => [`${Number(val).toFixed(2)} m`, name]}
                />
                <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '10px', fontSize: '11px', fontFamily: 'var(--font-mono)' }} />
                <ReferenceLine y={4.0} stroke="#EF4444" strokeDasharray="4 4" label={{ value: 'Critical (4.0m)', fill: '#EF4444', fontSize: 10 }} />
                <ReferenceLine y={2.5} stroke="#F97316" strokeDasharray="4 4" label={{ value: 'High (2.5m)', fill: '#F97316', fontSize: 10 }} />
                <Line type="monotone" dataKey="actual" stroke="#15304B" strokeWidth={2.5} dot={false} name="Simulated Swell Hs" />
                <Line type="monotone" dataKey="lstm" stroke="#20AFA7" strokeWidth={2} strokeDasharray="3 3" dot={false} name="WaveLSTM Forecast" />
                <Line type="monotone" dataKey="ann" stroke="#245B8F" strokeWidth={1.5} dot={false} name="WaveANN Forecast" />
                <Line type="monotone" dataKey="cnn1d" stroke="#79B7E5" strokeWidth={1.5} dot={false} name="WaveCNN1D Forecast" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* 5. Hypothetical Early-Warning Alerts Feed */}
      <SimulationAlertsFeed
        alerts={simulationData?.alerts ?? []}
        scenarioName={activeScenario?.name}
      />
    </div>
  )
}

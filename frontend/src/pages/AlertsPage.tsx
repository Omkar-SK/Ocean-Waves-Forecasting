import React, { useState, useEffect } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Filter,
  RefreshCw,
  ShieldAlert,
  Info,
  MapPin,
  Calendar,
  Layers,
  ChevronDown,
  ChevronUp,
  Radio,
  FileText,
  Flame,
  FlaskConical,
  RotateCcw,
} from 'lucide-react'
import { api } from '../services/api'
import { useSimulation } from '../context/SimulationContext'
import type { AlertHistoryItem, AlertItem, AlertSummaryResponse, ThresholdConfigResponse } from '../types'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { StatusIndicator } from '../components/ui/StatusIndicator'

export function AlertsPage() {
  const { isSimulationActive, activeScenario, simulationData, openModal, reset } = useSimulation()

  const [alerts, setAlerts] = useState<AlertItem[]>([])
  const [summary, setSummary] = useState<AlertSummaryResponse | null>(null)
  const [thresholds, setThresholds] = useState<ThresholdConfigResponse | null>(null)

  const [selectedLoc, setSelectedLoc] = useState<string>('ALL')
  const [selectedSev, setSelectedSev] = useState<string>('ALL')
  const [selectedStatus, setSelectedStatus] = useState<string>('ACTIVE_ACKNOWLEDGED')
  const [feedMode, setFeedMode] = useState<string>('demo')

  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false)
  const [acknowledgingId, setAcknowledgingId] = useState<string | null>(null)
  const [operatorNotes, setOperatorNotes] = useState<Record<string, string>>({})
  const [expandedHistoryId, setExpandedHistoryId] = useState<string | null>(null)
  const [historyItems, setHistoryItems] = useState<Record<string, AlertHistoryItem[]>>({})

  const fetchAlertsData = async () => {
    try {
      setLoading(true)
      setError(null)
      const [alertList, summaryData, threshData] = await Promise.all([
        api.listAlerts(selectedStatus, selectedSev, selectedLoc),
        api.getAlertSummary(),
        api.getAlertThresholds(),
      ])
      setAlerts(alertList)
      setSummary(summaryData)
      setThresholds(threshData)
    } catch (err: unknown) {
      console.error('Failed to load alerts:', err)
      setError(err instanceof Error ? err.message : 'Failed to load coastal hazard alerts')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isSimulationActive && simulationData?.alerts) {
      let filtered = [...simulationData.alerts]
      if (selectedLoc !== 'ALL') {
        filtered = filtered.filter((a) => a.location_id === selectedLoc)
      }
      if (selectedSev !== 'ALL') {
        filtered = filtered.filter((a) => a.severity === selectedSev)
      }
      setAlerts(filtered)

      const redCount = simulationData.alerts.filter((a) => (a.severity as string) === 'RED' || (a.severity as string) === 'CRITICAL').length
      const orangeCount = simulationData.alerts.filter((a) => (a.severity as string) === 'ORANGE' || (a.severity as string) === 'HIGH').length
      const yellowCount = simulationData.alerts.filter((a) => (a.severity as string) === 'YELLOW' || (a.severity as string) === 'CAUTION').length
      setSummary({
        total_active: simulationData.alerts.length,
        critical_red: redCount,
        high_orange: orangeCount,
        caution_yellow: yellowCount,
        resolved_count: 0,
        status_counts: { ACTIVE: simulationData.alerts.length },
        last_assessment_utc: new Date().toISOString(),
      })
      setLoading(false)
    } else if (!isSimulationActive) {
      fetchAlertsData()
    }
  }, [isSimulationActive, simulationData, selectedLoc, selectedSev, selectedStatus])

  const handleAcknowledge = async (alertId: string) => {
    try {
      setAcknowledgingId(alertId)
      const notes = operatorNotes[alertId] || 'Acknowledged by operator in web interface.'
      await api.acknowledgeAlert(alertId, notes)
      await fetchAlertsData()
    } catch (err: unknown) {
      console.error('Failed to acknowledge alert:', err)
      alert(err instanceof Error ? err.message : 'Acknowledgement request failed.')
    } finally {
      setAcknowledgingId(null)
    }
  }

  const handleEvaluateNow = async () => {
    try {
      setIsEvaluating(true)
      await api.triggerAlertEvaluation(feedMode)
      await fetchAlertsData()
    } catch (err: unknown) {
      console.error('Failed to evaluate alerts:', err)
      alert(err instanceof Error ? err.message : 'Evaluation trigger failed.')
    } finally {
      setIsEvaluating(false)
    }
  }

  const toggleHistory = async (alertId: string) => {
    if (expandedHistoryId === alertId) {
      setExpandedHistoryId(null)
      return
    }
    setExpandedHistoryId(alertId)
    if (!historyItems[alertId]) {
      try {
        const hist = await api.getAlertHistory(alertId)
        setHistoryItems((prev) => ({ ...prev, [alertId]: hist }))
      } catch (err) {
        console.error('Failed to load alert history:', err)
      }
    }
  }

  return (
    <div className="space-y-6 pb-8">
      {/* Top Banner & Operational Controls */}
      <div className="ocean-panel bg-white p-6 rounded-2xl border border-[#DCE5EF] shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#FFE4E6] text-[#E11D48] border border-[#FECDD3]">
                <AlertTriangle className="h-4 w-4" />
              </span>
              <h1 className="text-xl font-bold tracking-tight text-[#15304B] font-['Outfit',sans-serif]">
                Coastal Early-Warning Alert Centre
              </h1>
            </div>
            <p className="mt-1 text-xs text-[#64758A] max-w-2xl leading-relaxed">
              Real-time wave hazard monitoring, operator lifecycle acknowledgements, and automated threshold evaluation across candidate coastal stations.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <Button
              size="sm"
              onClick={openModal}
              className="bg-linear-to-r from-[#245B8F] to-[#1D4A75] hover:from-[#1D4A75] text-white text-xs font-semibold px-3 py-1.5 rounded-lg flex items-center gap-1.5 shadow-2xs"
            >
              <FlaskConical className="h-3.5 w-3.5" />
              <span>{isSimulationActive ? 'Switch Scenario' : 'Run Hazard Simulation'}</span>
            </Button>

            <Button
              variant="secondary"
              size="sm"
              onClick={fetchAlertsData}
              isLoading={loading}
              className="text-xs"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </Button>

            <Button
              variant="primary"
              size="sm"
              onClick={handleEvaluateNow}
              isLoading={isEvaluating}
              className="text-xs font-semibold"
            >
              <Radio className={`h-3.5 w-3.5 ${isEvaluating ? 'animate-spin' : ''}`} />
              <span>Re-Evaluate Nodes</span>
            </Button>
          </div>
        </div>

        {/* Dynamic Thresholds Strip (Tested by vitest for ≥ 4 m, ≥ 2.5 m, ≥ 1.5 m) */}
        <div className="mt-4 pt-4 border-t border-[#DCE5EF] flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
          <div className="flex items-center gap-2 text-[#64758A]">
            <Info className="h-4 w-4 text-[#20AFA7]" />
            <span className="font-semibold text-[#15304B]">Safety Thresholds:</span>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <span className="px-2.5 py-1 rounded bg-[#FFE4E6] border border-[#FECDD3] text-[#BE123C] font-semibold">
              Critical Red: <strong>≥ {thresholds?.critical_m ?? 4} m</strong>
            </span>
            <span className="px-2.5 py-1 rounded bg-[#FEF3C7] border border-[#FDE68A] text-[#B45309] font-semibold">
              High Orange: <strong>≥ {thresholds?.high_risk_m ?? 2.5} m</strong>
            </span>
            <span className="px-2.5 py-1 rounded bg-[#FEF9C3] border border-[#FEF08A] text-[#A16207] font-semibold">
              Caution Yellow: <strong>≥ {thresholds?.caution_m ?? 1.5} m</strong>
            </span>
          </div>
        </div>
      </div>

      {/* 4 Summary Metric Cards (White Surfaces with subtle semantic borders) */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-white border border-[#DCE5EF] shadow-xs space-y-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-[#64758A] font-semibold">ACTIVE ALERTS</span>
          <div className="text-3xl font-extrabold font-mono text-[#15304B]">
            {summary?.total_active ?? 0}
          </div>
          <p className="text-[11px] text-[#64758A]">Currently open hazard events</p>
        </div>

        <div className="p-4 rounded-xl bg-white border border-[#FECDD3] shadow-xs space-y-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-[#E11D48] font-semibold">CRITICAL RED</span>
          <div className="text-3xl font-extrabold font-mono text-[#E11D48]">
            {summary?.critical_red ?? 0}
          </div>
          <p className="text-[11px] text-[#64758A]">Exceeding 4.0m danger swell</p>
        </div>

        <div className="p-4 rounded-xl bg-white border border-[#FDE68A] shadow-xs space-y-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-[#D97706] font-semibold">HIGH ORANGE</span>
          <div className="text-3xl font-extrabold font-mono text-[#D97706]">
            {summary?.high_orange ?? 0}
          </div>
          <p className="text-[11px] text-[#64758A]">Elevated swell (≥2.5m)</p>
        </div>

        <div className="p-4 rounded-xl bg-white border border-[#FEF08A] shadow-xs space-y-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-[#CA8A04] font-semibold">CAUTION YELLOW</span>
          <div className="text-3xl font-extrabold font-mono text-[#CA8A04]">
            {summary?.caution_yellow ?? 0}
          </div>
          <p className="text-[11px] text-[#64758A]">Moderate wave action (≥1.5m)</p>
        </div>
      </div>

      {/* Filter Controls Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 bg-white p-3.5 rounded-xl border border-[#DCE5EF] shadow-xs">
        <div>
          <label className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] font-semibold block mb-1.5">
            Filter Status
          </label>
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="w-full bg-[#F0F6FC] border border-[#DCE5EF] rounded-lg px-3 py-2 text-xs text-[#15304B] font-mono focus:outline-none focus:border-[#20AFA7]"
          >
            <option value="ACTIVE_ACKNOWLEDGED">Active & Acknowledged</option>
            <option value="ACTIVE">Active Only</option>
            <option value="ACKNOWLEDGED">Acknowledged Only</option>
            <option value="RESOLVED">Resolved Only</option>
            <option value="ALL">All Lifecycles</option>
          </select>
        </div>

        <div>
          <label className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] font-semibold block mb-1.5">
            Filter Severity
          </label>
          <select
            value={selectedSev}
            onChange={(e) => setSelectedSev(e.target.value)}
            className="w-full bg-[#F0F6FC] border border-[#DCE5EF] rounded-lg px-3 py-2 text-xs text-[#15304B] font-mono focus:outline-none focus:border-[#20AFA7]"
          >
            <option value="ALL">All Severities</option>
            <option value="RED">Critical Red</option>
            <option value="ORANGE">High Orange</option>
            <option value="YELLOW">Caution Yellow</option>
          </select>
        </div>

        <div>
          <label className="text-[11px] font-mono uppercase tracking-wider text-[#64758A] font-semibold block mb-1.5">
            Evaluation Stream
          </label>
          <select
            value={feedMode}
            onChange={(e) => setFeedMode(e.target.value)}
            className="w-full bg-[#F0F6FC] border border-[#DCE5EF] rounded-lg px-3 py-2 text-xs text-[#15304B] font-mono focus:outline-none focus:border-[#20AFA7]"
          >
            <option value="demo">ERA5 Demo Mode</option>
            <option value="live">Live Open-Meteo Ingestion</option>
          </select>
        </div>
      </div>

      {/* Simulation Active Callout Banner */}
      {isSimulationActive && (
        <div className="p-4 rounded-xl bg-linear-to-r from-[#FFFBEB] via-[#FEF3C7] to-[#FFFBEB] border border-[#FCD34D] text-[#92400E] flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-xs animate-in fade-in duration-150">
          <div className="flex items-center gap-2.5">
            <Flame className="h-5 w-5 text-[#D97706] shrink-0" />
            <div>
              <div className="font-bold text-sm text-[#78350F]">
                SIMULATED HAZARD ALERTS: {activeScenario?.name}
              </div>
              <p className="text-xs text-[#92400E]">
                Hypothetical emergency alerts generated from the active simulation model. Live database records remain completely unmutated.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 self-end sm:self-auto">
            <Button size="sm" onClick={openModal} className="bg-white border-[#FCD34D] text-[#92400E] hover:bg-[#FDE68A] text-xs">
              Switch Scenario
            </Button>
            <Button size="sm" variant="secondary" onClick={reset} className="bg-[#245B8F] text-white hover:bg-[#1D4A75] text-xs">
              Exit Simulation
            </Button>
          </div>
        </div>
      )}

      {/* Alerts Timeline / Event Feed */}
      <div className="space-y-4">
        {alerts.length === 0 ? (
          <div className="ocean-panel bg-white border border-[#DCE5EF] rounded-2xl p-8 text-center space-y-4 shadow-xs">
            <div className="h-12 w-12 rounded-full bg-[#ECFDF5] border border-[#A7F3D0] flex items-center justify-center text-[#10B981] mx-auto">
              <CheckCircle2 className="h-6 w-6" />
            </div>
            <div>
              <h4 className="font-['Outfit',sans-serif] text-base font-bold text-[#15304B]">
                No Hazard Alerts Recorded (Safe Conditions)
              </h4>
              <p className="text-xs text-[#64758A] max-w-md mx-auto mt-1 leading-relaxed">
                All registered Arabian Sea coastal stations are currently operating within safe physical thresholds.
                To test how OceanCast AI responds during severe marine weather or cyclonic conditions, run a simulation.
              </p>
            </div>
            <div className="pt-2">
              <Button
                size="sm"
                onClick={openModal}
                className="bg-linear-to-r from-[#245B8F] to-[#1D4A75] hover:from-[#1D4A75] text-white text-xs font-semibold px-4 py-2 rounded-xl inline-flex items-center gap-2 shadow-xs"
              >
                <FlaskConical className="h-4 w-4" />
                <span>Run Hazard Simulation</span>
              </Button>
            </div>
          </div>
        ) : (
          alerts.map((alt) => {
            const isRed = alt.severity === 'RED'
            const isOrange = alt.severity === 'ORANGE'
            const isAcknowledged = alt.status === 'ACKNOWLEDGED'
            const isResolved = alt.status === 'RESOLVED'

            return (
              <div
                key={alt.alert_id}
                className={`ocean-panel rounded-2xl p-5 border transition-all shadow-xs ${
                  isRed
                    ? 'bg-white border-[#FECDD3] border-l-4 border-l-[#E11D48]'
                    : isOrange
                    ? 'bg-white border-[#FDE68A] border-l-4 border-l-[#D97706]'
                    : 'bg-white border-[#DCE5EF] border-l-4 border-l-[#EAB308]'
                }`}
              >
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-[#DCE5EF] pb-3">
                  <div className="flex items-center gap-2.5">
                    <span
                      className={`h-2.5 w-2.5 rounded-full ${
                        isRed ? 'bg-[#E11D48] animate-pulse' : isOrange ? 'bg-[#D97706]' : 'bg-[#EAB308]'
                      }`}
                    />
                    <span className="font-['Outfit',sans-serif] text-base font-bold text-[#15304B]">
                      {alt.location_name}
                    </span>
                    <Badge
                      variant={isRed ? 'destructive' : isOrange ? 'warning' : 'default'}
                      className="font-mono text-[10px] font-bold"
                    >
                      {alt.severity} RISK
                    </Badge>
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-[#F0F6FC] text-[#64758A] border border-[#DCE5EF]">
                      {alt.status}
                    </span>
                  </div>

                  <span className="text-xs font-mono text-[#64758A]">
                    Event Time: {alt.expected_event_time_utc ? new Date(alt.expected_event_time_utc).toUTCString() : 'Immediate'}
                  </span>
                </div>

                {/* Trigger reason text (Tested by vitest) */}
                <div className="my-3 text-xs leading-relaxed text-[#192B3C]">
                  <p className="font-medium">{alt.trigger_reason}</p>
                </div>

                {/* Metrics Readout Strip */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono py-2 bg-[#F0F6FC] rounded-lg p-3 border border-[#DCE5EF]">
                  <div>
                    <span className="text-[10px] text-[#64758A] font-semibold block">PREDICTED $H_s$</span>
                    <span className="text-sm font-bold text-[#15304B]">{alt.predicted_swh_m.toFixed(2)} m</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-[#64758A] font-semibold block">THRESHOLD</span>
                    <span className="text-sm font-bold text-[#64758A]">{alt.threshold_m.toFixed(2)} m</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-[#64758A] font-semibold block">LEAD HORIZON</span>
                    <span className="text-sm font-bold text-[#245B8F]">+{alt.forecast_horizon_hours} Hours</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-[#64758A] font-semibold block">DATA INGESTION</span>
                    <span className="text-sm font-bold text-[#20AFA7]">{alt.data_status}</span>
                  </div>
                </div>

                {/* Acknowledgement & Notes Section */}
                {!isResolved && (
                  <div className="mt-4 pt-3 border-t border-[#DCE5EF] flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
                    <input
                      type="text"
                      placeholder="Operator notes (e.g. Harbor Master notified)..."
                      value={operatorNotes[alt.alert_id] || ''}
                      onChange={(e) =>
                        setOperatorNotes((prev) => ({ ...prev, [alt.alert_id]: e.target.value }))
                      }
                      className="flex-1 bg-[#F0F6FC] border border-[#DCE5EF] rounded-lg px-3 py-1.5 text-xs text-[#15304B] placeholder:text-[#8291A2] focus:outline-none focus:border-[#20AFA7]"
                    />
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={() => handleAcknowledge(alt.alert_id)}
                      isLoading={acknowledgingId === alt.alert_id}
                      className="text-xs font-semibold whitespace-nowrap"
                    >
                      <span>✍️ Acknowledge</span>
                    </Button>
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}

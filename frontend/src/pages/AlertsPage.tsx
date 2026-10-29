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
} from 'lucide-react'
import { api } from '../services/api'
import type { AlertHistoryItem, AlertItem, AlertSummaryResponse, ThresholdConfigResponse } from '../types'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { StatusIndicator } from '../components/ui/StatusIndicator'

export function AlertsPage() {
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
    fetchAlertsData()
  }, [selectedLoc, selectedSev, selectedStatus])

  const handleAcknowledge = async (alertId: string) => {
    try {
      setAcknowledgingId(alertId)
      const notes = operatorNotes[alertId] || 'Acknowledged by operator in web interface.'
      await api.acknowledgeAlert(alertId, notes)
      // Refresh alert list and summary
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
    <div className="space-y-6">
      {/* Top Banner & Prototype Notice */}
      <div className="rounded-xl bg-gradient-to-r from-[#1E1B4B] via-[#1E3A8A] to-[#0369A1] p-5 border border-[#3B82F6] shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-500/20 text-red-400 border border-red-500/40">
                <AlertTriangle className="h-5 w-5" />
              </span>
              <h1 className="text-xl font-bold tracking-tight text-[#F1F5F9] font-['Outfit',sans-serif]">
                Coastal Early-Warning Alert Centre
              </h1>
            </div>
            <p className="mt-1 text-xs text-[#CBD5E1] max-w-3xl leading-relaxed">
              Explainable marine wave hazard detection, persistent alert lifecycle management, and configurable engineering risk thresholds.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleEvaluateNow}
              isLoading={isEvaluating}
              className="text-xs bg-[#0F172A] border-[#38BDF8] text-[#38BDF8] hover:bg-[#1E293B]"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isEvaluating ? 'animate-spin' : ''}`} />
              <span>🔄 Re-Evaluate Now</span>
            </Button>
          </div>
        </div>

        {/* Prototype Warning Notice */}
        <div className="mt-4 rounded-lg bg-[#07111F]/80 p-3.5 border border-[#38BDF8]/40 text-xs text-[#CBD5E1] flex items-start gap-3">
          <Info className="h-4 w-4 text-[#38BDF8] shrink-0 mt-0.5" />
          <div className="leading-relaxed">
            <span className="font-semibold text-[#F1F5F9]">Decision-Support Academic Prototype Notice:</span> This early-warning alert system evaluates marine wave forecasts against configurable engineering thresholds (Demonstration v1.0). <b>It does NOT replace official maritime safety warnings issued by national authorities (e.g. INCOIS, IMD).</b> A <span className="text-emerald-400 font-bold">GREEN</span> assessment indicates absence of threshold crossing; it is <b>never a guarantee of maritime safety</b>.
          </div>
        </div>
      </div>

      {/* KPI Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="bg-[#101D2D] p-3.5 rounded-xl border border-[#1E3A5F]">
          <div className="text-[11px] text-[#94A3B8] uppercase font-semibold">Active Hazard Alerts</div>
          <div className={`text-2xl font-bold font-mono mt-1 ${summary?.total_active && summary.total_active > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
            {summary?.total_active ?? 0}
          </div>
          <div className="text-[10px] text-[#64748B] mt-0.5">Active & Acknowledged</div>
        </div>

        <div className="bg-[#101D2D] p-3.5 rounded-xl border border-[#1E3A5F]">
          <div className="text-[11px] text-[#94A3B8] uppercase font-semibold">Critical Risk (RED)</div>
          <div className="text-2xl font-bold font-mono mt-1 text-[#EF4444]">
            {summary?.critical_red ?? 0}
          </div>
          <div className="text-[10px] text-[#64748B] mt-0.5">Threshold: ≥ {thresholds?.critical_m ?? 4.0} m</div>
        </div>

        <div className="bg-[#101D2D] p-3.5 rounded-xl border border-[#1E3A5F]">
          <div className="text-[11px] text-[#94A3B8] uppercase font-semibold">High Risk (ORANGE)</div>
          <div className="text-2xl font-bold font-mono mt-1 text-[#F97316]">
            {summary?.high_orange ?? 0}
          </div>
          <div className="text-[10px] text-[#64748B] mt-0.5">Threshold: ≥ {thresholds?.high_risk_m ?? 2.5} m</div>
        </div>

        <div className="bg-[#101D2D] p-3.5 rounded-xl border border-[#1E3A5F]">
          <div className="text-[11px] text-[#94A3B8] uppercase font-semibold">Caution (YELLOW)</div>
          <div className="text-2xl font-bold font-mono mt-1 text-[#F59E0B]">
            {summary?.caution_yellow ?? 0}
          </div>
          <div className="text-[10px] text-[#64748B] mt-0.5">Threshold: ≥ {thresholds?.caution_m ?? 1.5} m</div>
        </div>

        <div className="bg-[#101D2D] p-3.5 rounded-xl border border-[#1E3A5F] col-span-2 md:col-span-1">
          <div className="text-[11px] text-[#94A3B8] uppercase font-semibold">Resolved History</div>
          <div className="text-2xl font-bold font-mono mt-1 text-emerald-400">
            {summary?.resolved_count ?? 0}
          </div>
          <div className="text-[10px] text-[#64748B] mt-0.5">De-escalated via Hysteresis</div>
        </div>
      </div>

      {/* Filters Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 bg-[#101D2D] p-3.5 rounded-xl border border-[#1E3A5F]">
        {/* Location Filter */}
        <div>
          <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider block mb-1.5">
            Filter Location
          </label>
          <select
            value={selectedLoc}
            onChange={(e) => setSelectedLoc(e.target.value)}
            className="w-full bg-[#07111F] border border-[#1E3A5F] rounded-lg px-3 py-2 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#35C6E8]"
          >
            <option value="ALL">🌐 All Coastal Locations</option>
            <option value="mumbai_offshore">Mumbai High Offshore (Reference)</option>
            <option value="goa_mormugao">Mormugao / Goa Coastal Node</option>
            <option value="mangalore_port">Mangalore Coastal Node</option>
            <option value="kochi_coast">Kochi Coastal Node</option>
            <option value="veraval_gujarat">Veraval / Gujarat Coast</option>
            <option value="ratnagiri_coast">Ratnagiri Coastal Node</option>
            <option value="kanyakumari_point">Kanyakumari Point</option>
            <option value="lakshadweep_kavaratti">Lakshadweep / Kavaratti</option>
          </select>
        </div>

        {/* Severity Filter */}
        <div>
          <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider block mb-1.5">
            Filter Severity
          </label>
          <select
            value={selectedSev}
            onChange={(e) => setSelectedSev(e.target.value)}
            className="w-full bg-[#07111F] border border-[#1E3A5F] rounded-lg px-3 py-2 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#35C6E8]"
          >
            <option value="ALL">All Severities</option>
            <option value="RED">🔴 RED (Critical Risk)</option>
            <option value="ORANGE">🟠 ORANGE (High Risk)</option>
            <option value="YELLOW">🟡 YELLOW (Caution)</option>
          </select>
        </div>

        {/* Lifecycle Status Filter */}
        <div>
          <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider block mb-1.5">
            Lifecycle Status
          </label>
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="w-full bg-[#07111F] border border-[#1E3A5F] rounded-lg px-3 py-2 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#35C6E8]"
          >
            <option value="ACTIVE_ACKNOWLEDGED">Active & Acknowledged</option>
            <option value="ALL">All Lifecycle States</option>
            <option value="ACTIVE">🔴 Active Only</option>
            <option value="ACKNOWLEDGED">🟡 Acknowledged Only</option>
            <option value="RESOLVED">🟢 Resolved</option>
            <option value="EXPIRED">⚪ Expired</option>
          </select>
        </div>

        {/* Telemetry Stream */}
        <div>
          <label className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider block mb-1.5">
            Evaluation Source
          </label>
          <select
            value={feedMode}
            onChange={(e) => setFeedMode(e.target.value)}
            className="w-full bg-[#07111F] border border-[#1E3A5F] rounded-lg px-3 py-2 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#35C6E8]"
          >
            <option value="demo">📂 Offline Historical Demo (ERA5)</option>
            <option value="live">🌐 Live Open-Meteo REST API</option>
          </select>
        </div>
      </div>

      {/* Hazard Alerts List */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-[#F1F5F9] flex items-center justify-between">
          <span>Active Hazard Alerts Feed ({alerts.length})</span>
          <span className="text-xs text-[#94A3B8] font-normal">
            Persistent SQLite Store • Auto-Deduplicated
          </span>
        </h3>

        {loading ? (
          <div className="rounded-xl bg-[#101D2D] p-12 text-center border border-[#1E3A5F] text-xs text-[#35C6E8]">
            <RefreshCw className="h-6 w-6 animate-spin mx-auto mb-2" />
            <span>Loading active coastal alerts...</span>
          </div>
        ) : alerts.length === 0 ? (
          <div className="rounded-xl bg-[#101D2D] border border-dashed border-[#1E3A5F] p-10 text-center text-xs text-[#64748B]">
            <CheckCircle2 className="h-8 w-8 text-emerald-400 mx-auto mb-2" />
            <h4 className="text-sm font-bold text-[#94A3B8]">No Coastal Hazard Alerts Recorded</h4>
            <p className="mt-1 max-w-md mx-auto">
              No wave height crossings matched the selected filter criteria. Click <b>'Re-Evaluate Now'</b> above to run multi-location hazard assessment.
            </p>
          </div>
        ) : (
          alerts.map((alertItem) => {
            const sevColor =
              alertItem.severity === 'RED'
                ? '#EF4444'
                : alertItem.severity === 'ORANGE'
                ? '#F97316'
                : '#F59E0B'

            const isHistoryExpanded = expandedHistoryId === alertItem.alert_id

            return (
              <div
                key={alertItem.alert_id}
                className="rounded-xl bg-[#0F172A] border border-[#1E293B] overflow-hidden shadow-md"
                style={{ borderLeftWidth: '5px', borderLeftColor: sevColor }}
              >
                <div className="p-4 md:p-5 space-y-3">
                  {/* Card Header */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div className="flex items-center gap-2.5 flex-wrap">
                      <span className="text-base font-bold text-[#F1F5F9]">
                        {alertItem.location_name}
                      </span>
                      {alertItem.is_reference ? (
                        <span className="bg-emerald-950/70 text-emerald-300 border border-emerald-500/30 text-[10px] font-bold px-2 py-0.5 rounded">
                          REFERENCE VALIDATED
                        </span>
                      ) : (
                        <span className="bg-amber-950/70 text-amber-300 border border-amber-500/30 text-[10px] font-bold px-2 py-0.5 rounded">
                          EXPERIMENTAL REGIONAL TRANSFER
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-2">
                      <span
                        className="text-xs font-bold px-2.5 py-1 rounded-md border"
                        style={{
                          color: sevColor,
                          borderColor: `${sevColor}66`,
                          backgroundColor: `${sevColor}15`,
                        }}
                      >
                        {alertItem.severity} RISK
                      </span>
                      <span className="text-xs font-semibold px-2.5 py-1 rounded-md bg-[#1E293B] text-[#CBD5E1]">
                        {alertItem.status}
                      </span>
                    </div>
                  </div>

                  {/* Conditions Ribbon */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 bg-[#111E38] p-3 rounded-lg border border-[#1E3A5F]/80 text-xs">
                    <div>
                      <div className="text-[10px] text-[#94A3B8] uppercase">Predicted Wave (Hs)</div>
                      <div className="text-lg font-bold font-mono mt-0.5" style={{ color: sevColor }}>
                        {alertItem.predicted_swh_m.toFixed(2)} m
                      </div>
                      <div className="text-[10px] text-[#64748B]">Threshold: ≥ {alertItem.threshold_m.toFixed(2)} m</div>
                    </div>

                    <div>
                      <div className="text-[10px] text-[#94A3B8] uppercase">Forecast Horizon</div>
                      <div className="text-lg font-bold font-mono text-[#38BDF8] mt-0.5">
                        +{alertItem.forecast_horizon_hours}h Ahead
                      </div>
                      <div className="text-[10px] text-[#64748B]">
                        Event: {new Date(alertItem.expected_event_time_utc).toUTCString().slice(5, 22)}
                      </div>
                    </div>

                    <div>
                      <div className="text-[10px] text-[#94A3B8] uppercase">Hazard Type</div>
                      <div className="text-sm font-bold text-[#F1F5F9] mt-0.5">
                        {alertItem.event_type.replace('_', ' ')}
                      </div>
                      <div className="text-[10px] text-[#64748B]">Status: {alertItem.data_status.toUpperCase()}</div>
                    </div>

                    <div>
                      <div className="text-[10px] text-[#94A3B8] uppercase">Inference Model</div>
                      <div className="text-sm font-bold text-[#F1F5F9] mt-0.5">
                        {alertItem.model_name || 'Deep Learning'}
                      </div>
                      <div className="text-[10px] text-[#64748B]">
                        Updated: {new Date(alertItem.updated_at_utc).toUTCString().slice(17, 22)} UTC
                      </div>
                    </div>
                  </div>

                  {/* Trigger Explanation */}
                  <div className="text-xs text-[#CBD5E1] bg-[#07111F] p-2.5 rounded-lg border border-[#1E3A5F]/60">
                    <span className="font-semibold text-[#F1F5F9]">Trigger: </span>
                    {alertItem.trigger_reason}
                  </div>

                  {/* Actions & Lifecycle Details */}
                  <div className="pt-2 border-t border-[#1E293B] flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      {alertItem.status === 'ACTIVE' ? (
                        <div className="flex items-center gap-2 flex-1">
                          <input
                            type="text"
                            placeholder="Operator notes (e.g. port notified)"
                            value={operatorNotes[alertItem.alert_id] || ''}
                            onChange={(e) =>
                              setOperatorNotes((prev) => ({
                                ...prev,
                                [alertItem.alert_id]: e.target.value,
                              }))
                            }
                            className="bg-[#07111F] border border-[#1E3A5F] rounded-lg px-3 py-1.5 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#35C6E8] min-w-[220px]"
                          />
                          <Button
                            size="sm"
                            variant="primary"
                            onClick={() => handleAcknowledge(alertItem.alert_id)}
                            isLoading={acknowledgingId === alertItem.alert_id}
                            className="text-xs whitespace-nowrap"
                          >
                            ✍️ Acknowledge
                          </Button>
                        </div>
                      ) : alertItem.status === 'ACKNOWLEDGED' ? (
                        <div className="text-xs text-amber-300 flex items-center gap-1.5">
                          <CheckCircle2 className="h-3.5 w-3.5" />
                          <span>
                            Acknowledged at{' '}
                            {alertItem.acknowledged_at_utc
                              ? new Date(alertItem.acknowledged_at_utc).toUTCString().slice(5, 22)
                              : 'recorded'}
                            . Notes: "{alertItem.operator_notes || 'None'}"
                          </span>
                        </div>
                      ) : (
                        <div className="text-xs text-emerald-400">
                          Resolved at {alertItem.resolved_at_utc ? new Date(alertItem.resolved_at_utc).toUTCString().slice(5, 22) : 'N/A'}
                        </div>
                      )}
                    </div>

                    <button
                      onClick={() => toggleHistory(alertItem.alert_id)}
                      className="text-xs text-[#35C6E8] hover:underline flex items-center gap-1 self-end md:self-auto cursor-pointer"
                    >
                      <span>Lifecycle Audit History</span>
                      {isHistoryExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                    </button>
                  </div>

                  {/* History Expansion Table */}
                  {isHistoryExpanded && (
                    <div className="mt-3 rounded-lg border border-[#1E3A5F] bg-[#07111F] p-3 text-xs">
                      <div className="font-semibold text-xs text-[#94A3B8] mb-2">
                        State Transition Log (Fingerprint: {alertItem.fingerprint.slice(0, 32)}...)
                      </div>
                      {historyItems[alertItem.alert_id] && historyItems[alertItem.alert_id].length > 0 ? (
                        <table className="w-full text-left">
                          <thead className="text-[11px] text-[#64748B] border-b border-[#1E3A5F]">
                            <tr>
                              <th className="py-1">From</th>
                              <th className="py-1">To</th>
                              <th className="py-1">Reason</th>
                              <th className="py-1">Timestamp</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-[#1E3A5F]/40 text-[#CBD5E1]">
                            {historyItems[alertItem.alert_id].map((h) => (
                              <tr key={h.history_id}>
                                <td className="py-1 font-mono">{h.from_status || 'INITIAL'}</td>
                                <td className="py-1 font-mono font-bold text-[#35C6E8]">{h.to_status}</td>
                                <td className="py-1">{h.reason}</td>
                                <td className="py-1 text-[11px] text-[#64748B]">
                                  {new Date(h.changed_at_utc).toUTCString().slice(5, 22)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <div className="text-xs text-[#64748B]">Loading history records...</div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )
          })
        )}
      </div>

      {/* Shared Threshold Reference Box */}
      <div className="rounded-xl bg-[#101D2D] p-5 border border-[#1E3A5F] shadow-lg space-y-3">
        <div className="flex items-center justify-between border-b border-[#1E3A5F] pb-2">
          <h4 className="text-sm font-bold text-[#F1F5F9] flex items-center gap-2">
            <span>⚙️ Shared Engineering Risk Thresholds & Hysteresis Reference</span>
          </h4>
          <span className="text-xs font-mono text-[#35C6E8]">Demonstration v1.0</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
          <div className="bg-[#07111F] p-3 rounded-lg border border-[#1E3A5F]">
            <div className="text-[11px] text-[#F59E0B] font-bold">Caution (YELLOW)</div>
            <div className="text-lg font-bold text-[#F1F5F9] font-mono mt-0.5">
              ≥ {thresholds?.caution_m ?? 1.50} m
            </div>
            <div className="text-[10px] text-[#64748B]">Elevated swell hazard / small craft caution</div>
          </div>

          <div className="bg-[#07111F] p-3 rounded-lg border border-[#1E3A5F]">
            <div className="text-[11px] text-[#F97316] font-bold">High Risk (ORANGE)</div>
            <div className="text-lg font-bold text-[#F1F5F9] font-mono mt-0.5">
              ≥ {thresholds?.high_risk_m ?? 2.50} m
            </div>
            <div className="text-[10px] text-[#64748B]">Rough seas / craft warning</div>
          </div>

          <div className="bg-[#07111F] p-3 rounded-lg border border-[#1E3A5F]">
            <div className="text-[11px] text-[#EF4444] font-bold">Critical Risk (RED)</div>
            <div className="text-lg font-bold text-[#F1F5F9] font-mono mt-0.5">
              ≥ {thresholds?.critical_m ?? 4.00} m
            </div>
            <div className="text-[10px] text-[#64748B]">Severe sea state / high wave alert</div>
          </div>

          <div className="bg-[#07111F] p-3 rounded-lg border border-[#1E3A5F]">
            <div className="text-[11px] text-[#35C6E8] font-bold">Hysteresis Buffer</div>
            <div className="text-lg font-bold text-[#F1F5F9] font-mono mt-0.5">
              {thresholds?.hysteresis_buffer_m ?? 0.20} m
            </div>
            <div className="text-[10px] text-[#64748B]">
              Resolves below {(thresholds?.caution_m ?? 1.5) - (thresholds?.hysteresis_buffer_m ?? 0.2)} m
            </div>
          </div>
        </div>

        <div className="text-[11px] text-[#64748B] pt-1">
          <b>Separation of Concerns:</b> ML operations indicators (feature drift, model degradation) are stored in dedicated logs and <b>never generate maritime coastal safety warnings</b>.
        </div>
      </div>
    </div>
  )
}

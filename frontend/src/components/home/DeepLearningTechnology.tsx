import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Cpu, CheckCircle2, ArrowRight, ShieldCheck } from 'lucide-react'
import { api } from '../../services/api'
import type { MultiHorizonResponse } from '../../types'

interface BenchmarkRow {
  horizon: number
  model: string
  model_type: string
  rmse: number
  mae: number
  r2: number
  skill: number
  is_best: boolean
}

const STATIC_BENCHMARKS: Record<1 | 6 | 24, BenchmarkRow[]> = {
  1: [
    {
      horizon: 1,
      model: 'Naive Persistence',
      model_type: 'Baseline',
      rmse: 0.0213,
      mae: 0.0131,
      r2: 0.9991,
      skill: 0.0,
      is_best: true,
    },
    {
      horizon: 1,
      model: 'WaveLSTM',
      model_type: 'Recurrent LSTM',
      rmse: 0.0271,
      mae: 0.0182,
      r2: 0.9986,
      skill: -27.2,
      is_best: false,
    },
    {
      horizon: 1,
      model: 'WaveANN',
      model_type: 'Dense MLP',
      rmse: 0.0465,
      mae: 0.0349,
      r2: 0.9958,
      skill: -118.81,
      is_best: false,
    },
    {
      horizon: 1,
      model: 'WaveCNN1D',
      model_type: 'Temporal Conv',
      rmse: 0.0587,
      mae: 0.0426,
      r2: 0.9934,
      skill: -176.08,
      is_best: false,
    },
  ],
  6: [
    {
      horizon: 6,
      model: 'WaveLSTM',
      model_type: 'Recurrent LSTM',
      rmse: 0.0769,
      mae: 0.0533,
      r2: 0.9886,
      skill: 13.88,
      is_best: true,
    },
    {
      horizon: 6,
      model: 'WaveANN',
      model_type: 'Dense MLP',
      rmse: 0.0815,
      mae: 0.0574,
      r2: 0.9872,
      skill: 8.77,
      is_best: false,
    },
    {
      horizon: 6,
      model: 'WaveCNN1D',
      model_type: 'Temporal Conv',
      rmse: 0.0818,
      mae: 0.0575,
      r2: 0.9871,
      skill: 8.41,
      is_best: false,
    },
    {
      horizon: 6,
      model: 'Naive Persistence',
      model_type: 'Baseline',
      rmse: 0.0893,
      mae: 0.0624,
      r2: 0.9846,
      skill: 0.0,
      is_best: false,
    },
  ],
  24: [
    {
      horizon: 24,
      model: 'Naive Persistence',
      model_type: 'Baseline',
      rmse: 0.1866,
      mae: 0.1286,
      r2: 0.933,
      skill: 0.0,
      is_best: false,
    },
    {
      horizon: 24,
      model: 'WaveANN',
      model_type: 'Dense MLP',
      rmse: 0.192,
      mae: 0.1225,
      r2: 0.9291,
      skill: -2.89,
      is_best: true, // Best on MAE (+4.71% skill)
    },
    {
      horizon: 24,
      model: 'WaveCNN1D',
      model_type: 'Temporal Conv',
      rmse: 0.1901,
      mae: 0.1362,
      r2: 0.9305,
      skill: -1.9,
      is_best: false,
    },
    {
      horizon: 24,
      model: 'WaveLSTM',
      model_type: 'Recurrent LSTM',
      rmse: 0.1979,
      mae: 0.1229,
      r2: 0.9247,
      skill: -6.06,
      is_best: false,
    },
  ],
}

export function DeepLearningTechnology() {
  const [selectedHorizon, setSelectedHorizon] = useState<1 | 6 | 24>(6)
  const [metricsRows, setMetricsRows] = useState<BenchmarkRow[]>(STATIC_BENCHMARKS[6])

  useEffect(() => {
    let isMounted = true
    const fetchMetrics = async () => {
      try {
        const res = await api.getMultiHorizonMetrics()
        if (isMounted && res) {
          // If API returns detailed items, parse them
          const items = (res as any).items
          if (Array.isArray(items) && items.length > 0) {
            const parsed: Record<1 | 6 | 24, BenchmarkRow[]> = { 1: [], 6: [], 24: [] }
            items.forEach((it: any) => {
              const h = (it.horizon_hours ?? it.horizon) as 1 | 6 | 24
              if (h === 1 || h === 6 || h === 24) {
                parsed[h].push({
                  horizon: h,
                  model: it.model || it.model_name || 'Model',
                  model_type: it.model_type || 'Neural Net',
                  rmse: Number(it.rmse_meters ?? it.rmse ?? 0),
                  mae: Number(it.mae_meters ?? it.mae ?? 0),
                  r2: Number(it.r2_score ?? it.r2 ?? 0),
                  skill: Number(it.rmse_skill_pct ?? it.skill_score ?? 0),
                  is_best: Boolean(it.is_best ?? it.is_best_in_horizon),
                })
              }
            })
            if (parsed[selectedHorizon].length > 0) {
              setMetricsRows(parsed[selectedHorizon])
            }
          }
        }
      } catch (err) {
        // Fall back gracefully to static benchmarks
        setMetricsRows(STATIC_BENCHMARKS[selectedHorizon])
      }
    }
    fetchMetrics()
    return () => {
      isMounted = false
    }
  }, [selectedHorizon])

  const architectures = [
    {
      id: 'lstm',
      name: 'WaveLSTM',
      type: 'Recurrent Neural Network',
      badge: 'Best at +6h Prognostic Lead',
      description:
        'Stacked 2-layer LSTM architecture (64 hidden units) with recurrent cell gates specifically designed to model swell propagation delay and temporal wave-wind inertia across the Arabian Sea.',
      hyperparams: '2 Layers &bull; 64 Hidden Dim &bull; Dropout 0.20 &bull; Input: [Batch, 24, 13]',
      bestHorizon: '+6h (+13.88% Skill Score over Persistence)',
      accentColor: '#246B9E',
    },
    {
      id: 'ann',
      name: 'WaveANN',
      type: 'Dense Feedforward MLP',
      badge: 'Best at +24h Diurnal Horizon',
      description:
        'Multi-layer perceptron (128 &rarr; 64 &rarr; 1) with Batch Normalization. Maps diurnal sea breeze transitions and seasonal monsoonal cycles without recurrent vanishing gradient degradation.',
      hyperparams: '3 Dense Layers &bull; BatchNorm1D &bull; ReLU &bull; Input: [Batch, 24 x 13]',
      bestHorizon: '+24h (+4.71% MAE Calibration Skill)',
      accentColor: '#20AFA7',
    },
    {
      id: 'cnn1d',
      name: 'WaveCNN1D',
      type: 'Temporal 1D Convolutional Network',
      badge: 'Sub-Millisecond Inference',
      description:
        'Hierarchical 1D convolutional filter banks (kernel size 3) extracting localized wave crest acceleration and rapid squall onset directly from temporal sequence slices.',
      hyperparams: 'Conv1D (32, 64) &bull; MaxPool1D &bull; Flatten &bull; Linear Head',
      bestHorizon: 'Rapid Squall & Sudden Crest Shock Detection',
      accentColor: '#7C3AED',
    },
  ]

  return (
    <section id="technology" className="py-24 bg-[#F4F7FB] border-b border-[#DCE5EF]">
      <div className="max-w-7xl mx-auto px-4 md:px-8">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#EAF2F9] border border-[#246B9E]/20 text-xs font-mono font-semibold uppercase text-[#246B9E] mb-3">
            <Cpu className="h-3.5 w-3.5" />
            <span>Deep Learning Architecture</span>
          </div>
          <h2 className="text-3xl md:text-5xl font-bold font-['Outfit',sans-serif] text-[#102A43] tracking-tight">
            Trained on 14.8 Years of ERA5 Reanalysis
          </h2>
          <p className="text-[#64758A] text-base md:text-lg mt-4 font-normal leading-relaxed">
            Rigorous temporal validation across 129,696 chronological hours and 19,455 untouched test samples (2024–2026), evaluating each architecture against naive persistence.
          </p>
        </div>

        {/* 3 Architecture Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
          {architectures.map((arch) => (
            <div
              key={arch.id}
              className="bg-white rounded-2xl p-7 border border-[#DCE5EF] shadow-xs hover:border-[#8CC8E8] transition-all flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-[11px] font-mono font-semibold uppercase text-[#64758A] tracking-wider">
                    {arch.type}
                  </span>
                  <div
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: arch.accentColor }}
                  />
                </div>

                <h3 className="text-2xl font-bold font-['Outfit',sans-serif] text-[#102A43] mb-2 tracking-tight">
                  {arch.name}
                </h3>

                <span className="inline-block text-[10px] font-mono font-semibold uppercase px-2.5 py-1 rounded-md bg-[#F4F7FB] border border-[#DCE5EF] text-[#246B9E] mb-4">
                  {arch.badge}
                </span>

                <p className="text-xs text-[#64758A] leading-relaxed mb-6 font-normal">
                  {arch.description}
                </p>
              </div>

              <div className="pt-4 border-t border-[#DCE5EF] space-y-2 text-xs">
                <div
                  className="text-[11px] font-mono text-[#192B3C] font-medium"
                  dangerouslySetInnerHTML={{ __html: arch.hyperparams }}
                />
                <div className="text-[11px] text-[#0D837C] font-semibold flex items-center gap-1.5">
                  <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                  <span>{arch.bestHorizon}</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Interactive Benchmark Metrics Table */}
        <div className="bg-white rounded-3xl p-6 lg:p-8 border border-[#DCE5EF] shadow-xs">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
            <div>
              <h4 className="text-xl font-bold font-['Outfit',sans-serif] text-[#102A43]">
                Empirical Test Set Performance (19,455 Hourly Samples)
              </h4>
              <p className="text-xs text-[#64758A] mt-1">
                Comparing deep learning models against Naive Persistence baseline ($H_s(t+\Delta) = H_s(t)$)
              </p>
            </div>

            {/* Horizon Selector Tabs */}
            <div className="flex items-center p-1 rounded-xl bg-[#F4F7FB] border border-[#DCE5EF] self-start sm:self-auto">
              {([1, 6, 24] as const).map((h) => (
                <button
                  key={h}
                  onClick={() => {
                    setSelectedHorizon(h)
                    setMetricsRows(STATIC_BENCHMARKS[h])
                  }}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all ${
                    selectedHorizon === h
                      ? 'bg-[#246B9E] text-white shadow-xs'
                      : 'text-[#64758A] hover:text-[#102A43]'
                  }`}
                >
                  +{h}h Lead
                </button>
              ))}
            </div>
          </div>

          {/* Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-[#DCE5EF] text-[#64758A]">
                  <th className="py-3 px-4 font-semibold uppercase tracking-wider">Model</th>
                  <th className="py-3 px-4 font-semibold uppercase tracking-wider">Type</th>
                  <th className="py-3 px-4 font-semibold uppercase tracking-wider">RMSE (m)</th>
                  <th className="py-3 px-4 font-semibold uppercase tracking-wider">MAE (m)</th>
                  <th className="py-3 px-4 font-semibold uppercase tracking-wider">R² Score</th>
                  <th className="py-3 px-4 font-semibold uppercase tracking-wider">RMSE Skill vs Persistence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#DCE5EF]/60">
                {metricsRows.map((item, idx) => {
                  const isBest = item.is_best
                  const isPersistence = item.model.includes('Persistence')

                  return (
                    <tr
                      key={idx}
                      className={`hover:bg-[#F4F7FB]/60 transition-colors ${
                        isBest ? 'bg-[#E1F5F2]/30 font-semibold' : ''
                      }`}
                    >
                      <td className="py-3 px-4 font-bold text-[#102A43] flex items-center gap-2">
                        <span>{item.model}</span>
                        {isBest && (
                          <span className="px-1.5 py-0.5 rounded text-[9px] bg-[#10B981] text-white uppercase font-bold tracking-wider">
                            Best
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-[#64758A]">{item.model_type}</td>
                      <td className="py-3 px-4 text-[#102A43]">{item.rmse.toFixed(4)}</td>
                      <td className="py-3 px-4 text-[#102A43]">{item.mae.toFixed(4)}</td>
                      <td className="py-3 px-4 text-[#102A43]">{item.r2.toFixed(4)}</td>
                      <td className="py-3 px-4">
                        {isPersistence ? (
                          <span className="text-[#64758A]">0.00% (Baseline)</span>
                        ) : item.skill > 0 ? (
                          <span className="text-[#0D837C] font-bold">+{item.skill.toFixed(2)}%</span>
                        ) : (
                          <span className="text-[#E11D48]">{item.skill.toFixed(2)}%</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Scientific Nuance Takeaway Callout */}
          <div className="mt-6 p-4 rounded-xl bg-[#EAF2F9] border border-[#DCE5EF] text-xs text-[#192B3C] flex items-start gap-3">
            <ShieldCheck className="h-4 w-4 text-[#246B9E] shrink-0 mt-0.5" />
            <div className="space-y-1">
              <strong>Scientific Validation Insight:</strong>
              <p className="text-[#64758A] leading-relaxed">
                {selectedHorizon === 1 &&
                  'At +1 hour, ocean wave swell exhibits immense physical inertia (autocorrelation r₁ = 0.9995). Naive Persistence achieves the lowest RMSE (0.0213m), while WaveLSTM is the best neural net (0.0271m).'}
                {selectedHorizon === 6 &&
                  'At +6 hours, WaveLSTM decisively outperforms Naive Persistence on both RMSE (+13.88% Skill) and MAE (+14.51% Skill). Recurrent temporal memory captures swell propagation dynamics across the basin.'}
                {selectedHorizon === 24 &&
                  'At +24 hours, Naive Persistence achieves lowest quadratic peak error (RMSE 0.1866m), while WaveANN achieves superior absolute calibration on MAE (+4.71% Skill).'}
              </p>
            </div>
          </div>

          <div className="mt-6 flex justify-end">
            <Link
              to="/comparison"
              className="inline-flex items-center gap-2 text-xs font-semibold text-[#246B9E] hover:text-[#102A43] transition-colors"
            >
              <span>Explore Detailed Model Benchmarks</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </div>
    </section>
  )
}

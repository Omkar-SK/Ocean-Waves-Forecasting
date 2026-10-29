import React, { useState, useEffect } from 'react'
import {
  BookOpen,
  Compass,
  Cpu,
  Database,
  Layers,
  MapPin,
  ShieldAlert,
  Zap,
  ArrowRight,
  GitBranch,
  CheckCircle2,
  AlertTriangle,
  Scale,
  Activity,
  Server,
  Radio,
  Wifi,
  WifiOff,
} from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { Skeleton } from '../components/ui/Skeleton'
import { api } from '../services/api'
import type { HealthResponse, MetadataResponse, ForecastStatusResponse } from '../types'

export function AboutPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null)
  const [forecastStatus, setForecastStatus] = useState<ForecastStatusResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(true)

  useEffect(() => {
    async function loadStatus() {
      setLoading(true)
      try {
        const [h, m, f] = await Promise.all([
          api.getHealth(),
          api.getMetadata(),
          api.getLiveStatus(),
        ])
        setHealth(h)
        setMetadata(m)
        setForecastStatus(f)
      } catch (err) {
        console.error('Failed to load system status:', err)
      } finally {
        setLoading(false)
      }
    }
    loadStatus()
  }, [])

  const featureList = [
    { name: 'swh', desc: 'Significant wave height (m) — target & primary input', category: 'Ocean Waves' },
    { name: 'mwp', desc: 'Mean wave period (s)', category: 'Ocean Waves' },
    { name: 'mwd_sin / mwd_cos', desc: 'Circular trigonometric encoding of mean wave direction (θ)', category: 'Direction' },
    { name: 'u10 / v10', desc: '10-meter eastward and northward wind velocity components (m/s)', category: 'Atmospheric' },
    { name: 'wind_speed', desc: 'Derived resultant 10m wind speed: √(u10² + v10²)', category: 'Atmospheric' },
    { name: 'wind_dir_sin / cos', desc: 'Circular trigonometric encoding of meteorological wind direction', category: 'Direction' },
    { name: 'hour_sin / hour_cos', desc: '24-hour diurnal cyclical encoding: sin/cos(2π·hour/24)', category: 'Temporal' },
    { name: 'month_sin / month_cos', desc: '12-month seasonal annual encoding: sin/cos(2π·month/12)', category: 'Temporal' },
  ]

  const pipelineSteps = [
    {
      num: '01',
      title: 'ERA5 Reanalysis Extraction',
      desc: 'Copernicus Climate Data Store extraction at offshore grid 18.5°N, 72.5°E across multiple years.',
      badge: 'Data Provenance',
    },
    {
      num: '02',
      title: 'Feature Engineering & Scaling',
      desc: 'Derived wind speed, meteorological inversion, circular trigonometric encodings, and train-only MinMaxScaler.',
      badge: '13 Features',
    },
    {
      num: '03',
      title: 'Sliding Window Sequence Gen',
      desc: 'Formulation of 24 consecutive hourly timesteps (t-23 to t) sliding input tensor X ∈ ℝ^(B × 24 × 13).',
      badge: '24h Window',
    },
    {
      num: '04',
      title: 'Deep Learning Architectures',
      desc: 'Parallel benchmarking of WaveANN (Dense), WaveLSTM (Recurrent), and WaveCNN1D (Temporal Convolution).',
      badge: '3 Architectures',
    },
    {
      num: '05',
      title: 'Multi-Horizon Inference',
      desc: 'Independent checkpoint heads predicting future significant wave height at +1h, +6h, and +24h targets.',
      badge: '1h, 6h, 24h',
    },
    {
      num: '06',
      title: 'Rigorous Verification',
      desc: 'Evaluation against Persistence baseline on 19,455 untouched test samples using RMSE, MAE, and R².',
      badge: 'N = 19,455',
    },
  ]

  return (
    <div className="space-y-8">
      {/* Hero Header */}
      <div className="rounded-2xl bg-gradient-to-br from-[#101D2D] via-[#101D2D] to-[#16263A] border border-[#1E3A5F] p-6 md:p-8">
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <Badge variant="cyan">Academic Capstone Project</Badge>
          <Badge variant="teal">Semester 5 Deep Learning</Badge>
          <Badge variant="outline">VIT Pune</Badge>
        </div>
        <h1 className="font-['Outfit',sans-serif] text-2xl md:text-3xl font-extrabold text-[#F1F5F9] tracking-tight">
          Ocean Wave Forecasting Using Deep Learning: ANN vs LSTM vs CNN
        </h1>
        <p className="text-sm md:text-base text-[#94A3B8] max-w-4xl mt-3 leading-relaxed">
          OceanCast AI is a full-stack intelligence and benchmarking interface investigating whether deep neural
          architectures (Artificial Neural Networks, Long Short-Term Memory networks, and 1D Convolutional Neural Networks)
          can outperform simple persistence baselines when predicting offshore Arabian Sea significant wave height ($H_s$)
          at 1-hour, 6-hour, and 24-hour horizons.
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6 pt-6 border-t border-[#1E3A5F]/80">
          <div>
            <span className="text-[11px] text-[#94A3B8] uppercase tracking-wider block">Target Coordinates</span>
            <span className="text-sm font-semibold font-mono text-[#F1F5F9]">18.50° N, 72.50° E</span>
          </div>
          <div>
            <span className="text-[11px] text-[#94A3B8] uppercase tracking-wider block">Primary Data Source</span>
            <span className="text-sm font-semibold text-[#35C6E8]">ECMWF ERA5 Reanalysis</span>
          </div>
          <div>
            <span className="text-[11px] text-[#94A3B8] uppercase tracking-wider block">Chronological Test Set</span>
            <span className="text-sm font-semibold font-mono text-[#35D0B0]">19,455 Hourly Samples</span>
          </div>
          <div>
            <span className="text-[11px] text-[#94A3B8] uppercase tracking-wider block">Input Window</span>
            <span className="text-sm font-semibold font-mono text-[#F1F5F9]">24 Hours (t-23 to t)</span>
          </div>
        </div>
      </div>

      {/* End-to-End Pipeline Diagram */}
      <div className="space-y-4">
        <div>
          <h2 className="text-lg font-bold text-[#F1F5F9] font-['Outfit',sans-serif]">
            End-to-End Deep Learning Architecture Pipeline
          </h2>
          <p className="text-xs text-[#94A3B8]">
            Complete data flow from reanalysis ingestion through preprocessing, sequence formation, and multi-horizon inference
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {pipelineSteps.map((step) => (
            <div
              key={step.num}
              className="p-5 rounded-xl bg-[#101D2D] border border-[#1E3A5F] relative hover:border-[#35C6E8]/40 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xl font-bold font-mono text-[#35C6E8]/60">{step.num}</span>
                <Badge variant="surface" className="text-[10px]">
                  {step.badge}
                </Badge>
              </div>
              <h3 className="text-sm font-semibold text-[#F1F5F9] mb-1.5">{step.title}</h3>
              <p className="text-xs text-[#94A3B8] leading-relaxed">{step.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Grid: Feature Engineering & Preprocessing Integrity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Feature Specification */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Database className="h-4 w-4 text-[#35C6E8]" />
                <span>13-Feature Ingestion Specification</span>
              </CardTitle>
              <Badge variant="cyan" className="font-mono text-[10px]">
                dim = 13
              </Badge>
            </div>
            <CardDescription>
              Exact canonical feature ordering expected by all 9 PyTorch model checkpoints
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="divide-y divide-[#1E3A5F]/40">
              {featureList.map((f, i) => (
                <div key={i} className="py-2.5 flex items-start justify-between gap-4">
                  <div>
                    <span className="text-xs font-mono font-semibold text-[#35C6E8]">{f.name}</span>
                    <p className="text-[11px] text-[#94A3B8] mt-0.5">{f.desc}</p>
                  </div>
                  <Badge variant="outline" className="text-[9px] shrink-0">
                    {f.category}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Methodology & Data Leakage Prevention */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Scale className="h-4 w-4 text-[#35D0B0]" />
                <span>Strict Chronological Split & Leak-Free Scaling</span>
              </CardTitle>
              <CardDescription>
                Guarantees zero future lookahead contamination across temporal partitions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-xs text-[#94A3B8] leading-relaxed">
              <div className="p-3 rounded-lg bg-[#16263A] border border-[#1E3A5F] space-y-1">
                <span className="font-semibold text-[#F1F5F9] block">Chronological 70 / 15 / 15 Partitioning</span>
                <p>
                  Random K-Fold cross validation is mathematically invalid for auto-correlated geophysical time series.
                  The dataset is split strictly by date: first 70% for training, subsequent 15% for validation and hyperparameter
                  tuning, and the final 15% (19,455 hours from mid-2024 to late-2026) retained untouched for unbiased test evaluation.
                </p>
              </div>

              <div className="p-3 rounded-lg bg-[#16263A] border border-[#1E3A5F] space-y-1">
                <span className="font-semibold text-[#F1F5F9] block">Fit-Only-On-Train MinMaxScaler</span>
                <p>
                  The feature scaler (`scaler.pkl`) was fitted solely on training split rows. Minimums and maximums from
                  validation and test horizons were never exposed to the preprocessing pipeline, preserving realistic out-of-distribution
                  generalization.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Model Architecture Details */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-[#F59E0B]" />
                <span>Deep Learning Architectures</span>
              </CardTitle>
              <CardDescription>Implemented in PyTorch 2.2 with Adam optimizer & MSE loss</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-xs text-[#94A3B8]">
              <div className="p-2.5 rounded-lg bg-[#16263A] border border-[#1E3A5F]">
                <span className="font-semibold text-[#F1F5F9]">WaveLSTM (Recurrent):</span> Two-layer stacked LSTM with hidden
                size 64 and dropout (0.2), capturing sequential temporal memory across the 24-hour wind-sea growth cycle.
              </div>
              <div className="p-2.5 rounded-lg bg-[#16263A] border border-[#1E3A5F]">
                <span className="font-semibold text-[#F1F5F9]">WaveANN (Multi-Layer Perceptron):</span> Flattened (24 × 13 = 312)
                input fed through 3 fully-connected layers (128 → 64 → 1) with ReLU activations and Batch Normalization.
              </div>
              <div className="p-2.5 rounded-lg bg-[#16263A] border border-[#1E3A5F]">
                <span className="font-semibold text-[#F1F5F9]">WaveCNN1D (Temporal Convolution):</span> 1D Conv layers (filters 32, 64)
                with kernel size 3 and MaxPool, extracting local wave-train patterns across consecutive hours.
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* System Status & Data Feed Telemetry */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-4 w-4 text-[#35D0B0]" />
                <span>System Status & Service Health</span>
              </CardTitle>
              <CardDescription>
                Real-time operational status of backend services, PyTorch checkpoints, and data ingestion feeds
              </CardDescription>
            </div>
            <Badge variant={health?.backend_online ? 'teal' : 'destructive'}>
              {health?.backend_online ? 'Backend Connected' : 'Disconnected'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-3.5 rounded-xl bg-[#16263A] border border-[#1E3A5F]">
              <div className="flex items-center justify-between text-xs text-[#94A3B8] mb-1">
                <span>FastAPI Service</span>
                <Server className="h-3.5 w-3.5 text-[#35C6E8]" />
              </div>
              <div className="font-mono text-sm font-bold text-[#F1F5F9]">
                {health?.backend_online ? 'Healthy (Port 8000)' : 'Unhealthy'}
              </div>
              <p className="text-[10px] text-[#35D0B0] mt-1">Uvicorn ASGI Runtime</p>
            </div>

            <div className="p-3.5 rounded-xl bg-[#16263A] border border-[#1E3A5F]">
              <div className="flex items-center justify-between text-xs text-[#94A3B8] mb-1">
                <span>Open-Meteo NWP Feed</span>
                {forecastStatus?.live_api_online ? (
                  <Wifi className="h-3.5 w-3.5 text-[#35D0B0]" />
                ) : (
                  <WifiOff className="h-3.5 w-3.5 text-[#F59E0B]" />
                )}
              </div>
              <div className="font-mono text-sm font-bold text-[#F1F5F9]">
                {forecastStatus?.live_api_online ? 'Live Connected' : 'Demo Fallback'}
              </div>
              <p className="text-[10px] text-[#94A3B8] mt-1">ECMWF Atmospheric Ingestion</p>
            </div>

            <div className="p-3.5 rounded-xl bg-[#16263A] border border-[#1E3A5F]">
              <div className="flex items-center justify-between text-xs text-[#94A3B8] mb-1">
                <span>PyTorch Checkpoints</span>
                <Cpu className="h-3.5 w-3.5 text-[#F59E0B]" />
              </div>
              <div className="font-mono text-sm font-bold text-[#F1F5F9]">
                {health?.checkpoints_available ?? 9} / 9 Loaded
              </div>
              <p className="text-[10px] text-[#35D0B0] mt-1">ANN • LSTM • 1D-CNN</p>
            </div>

            <div className="p-3.5 rounded-xl bg-[#16263A] border border-[#1E3A5F]">
              <div className="flex items-center justify-between text-xs text-[#94A3B8] mb-1">
                <span>Live Buoy Telemetry</span>
                <Radio className="h-3.5 w-3.5 text-[#94A3B8]" />
              </div>
              <div className="font-mono text-sm font-bold text-[#94A3B8]">
                Not Configured
              </div>
              <p className="text-[10px] text-[#94A3B8] mt-1">INCOIS Feed Unconfigured</p>
            </div>
          </div>

          {/* Component Health Items */}
          {health?.components && health.components.length > 0 && (
            <div className="pt-2 border-t border-[#1E3A5F]/60">
              <span className="text-xs font-semibold text-[#94A3B8] block mb-2">Subsystem Component Status</span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {health.components.map((comp, idx) => (
                  <div key={idx} className="flex items-center justify-between p-2 rounded-lg bg-[#07111F] border border-[#1E3A5F] text-xs">
                    <span className="text-[#F1F5F9] font-medium">{comp.name}</span>
                    <Badge variant={comp.status === 'healthy' ? 'teal' : comp.status === 'degraded' ? 'warning' : 'destructive'} className="text-[10px]">
                      {comp.status}
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Critical Scientific Disclaimer */}
      <Card className="border-l-4 border-l-[#F59E0B]">
        <CardHeader>
          <div className="flex items-center gap-2">
            <ShieldAlert className="h-5 w-5 text-[#F59E0B]" />
            <CardTitle className="text-base text-[#F59E0B]">
              Scientific Limitations & Operational Distribution Shift Notice
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-2 text-xs text-[#94A3B8] leading-relaxed">
          <p>
            <strong>Reanalysis vs Live Observations:</strong> ERA5 is an atmospheric reanalysis produced by ECMWF combining
            historical observations with numerical weather prediction models. While highly accurate for climatological studies,
            ERA5 wave heights are smoothed grid-box estimates, not direct in-situ waverider buoy accelerometer measurements.
          </p>
          <p>
            <strong>Operational Distribution Shift:</strong> Real-time forecasts ingested from operational NWP feeds (e.g., Open-Meteo
            ECMWF WAM) may exhibit systematic statistical shifts compared to retrospective ERA5 training distributions due to model
            physics revisions, spatial resolution differences, and atmospheric data assimilation timing. OceanCast AI openly highlights
            these distinctions to ensure research honesty.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}

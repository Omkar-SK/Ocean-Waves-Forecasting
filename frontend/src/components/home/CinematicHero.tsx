import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Compass, ShieldCheck, Waves, Activity, Radio, ChevronDown, Sparkles } from 'lucide-react'
import { OceanScene3D } from './OceanScene3D'
import { ParticleWave } from '../ui/ParticleWave'
import { api } from '../../services/api'
import type { HealthResponse } from '../../types'

export function CinematicHero() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loadingHealth, setLoadingHealth] = useState(true)

  useEffect(() => {
    let isMounted = true
    const checkLiveHealth = async () => {
      try {
        const res = await api.getHealth()
        if (isMounted) setHealth(res)
      } catch (err) {
        if (isMounted) setHealth(null)
      } finally {
        if (isMounted) setLoadingHealth(false)
      }
    }
    checkLiveHealth()
    return () => {
      isMounted = false
    }
  }, [])

  const isLiveConnected = health?.status === 'healthy' && health?.live_api_available

  return (
    <section id="hero" className="relative w-full min-h-screen flex items-center justify-center overflow-hidden bg-[#061322]">
      {/* 3D WebGL Ocean Canvas */}
      <OceanScene3D />

      {/* Atmospheric Vignette & 21st.dev Cinematic Landing Hero Radial Gradients */}
      <div className="absolute inset-0 bg-gradient-to-t from-[#061322] via-transparent to-[#061322]/70 pointer-events-none" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_40%,transparent_20%,#061322_85%)] pointer-events-none" />
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[350px] bg-[#35B8AD]/12 rounded-full blur-[140px] pointer-events-none" />
      <div className="absolute bottom-1/4 left-1/3 w-[500px] h-[300px] bg-[#246B9E]/20 rounded-full blur-[120px] pointer-events-none" />

      {/* Foreground Content */}
      <div className="relative z-10 max-w-5xl mx-auto px-4 md:px-8 py-32 text-center flex flex-col items-center">
        {/* Upper Label & Live Telemetry Indicator */}
        <div className="inline-flex items-center gap-2.5 px-3.5 py-1.5 rounded-full bg-white/10 backdrop-blur-md border border-white/15 text-xs font-mono tracking-wider uppercase text-[#8CC8E8] mb-6 shadow-lg animate-ocean-fade">
          <div className="flex items-center gap-1.5">
            <span className="relative flex h-2 w-2">
              {isLiveConnected && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#10B981] opacity-75" />
              )}
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  loadingHealth
                    ? 'bg-[#F59E0B]'
                    : isLiveConnected
                    ? 'bg-[#10B981]'
                    : 'bg-[#94A3B8]'
                }`}
              />
            </span>
            <span className="font-semibold text-white">
              {loadingHealth ? 'VERIFYING TELEMETRY...' : isLiveConnected ? 'LIVE FEED ACTIVE' : 'HISTORICAL DEMO'}
            </span>
          </div>

          <span className="text-white/30">&bull;</span>
          <span>OCEAN INTELLIGENCE PLATFORM</span>
        </div>

        {/* Main Heading */}
        <h1 className="font-['Outfit',sans-serif] text-4xl sm:text-6xl md:text-7xl font-extrabold text-white tracking-tight leading-[1.08] max-w-4xl mb-6 drop-shadow-md">
          Understand the Ocean.{' '}
          <span className="block bg-gradient-to-r from-white via-[#8CC8E8] to-[#35B8AD] bg-clip-text text-transparent">
            Anticipate What&apos;s Next.
          </span>
        </h1>

        {/* Supporting Text */}
        <p className="text-base sm:text-lg md:text-xl text-[#8CC8E8]/90 max-w-2xl font-normal leading-relaxed mb-8 drop-shadow">
          Advanced deep learning and live marine intelligence for wave forecasting, coastal risk assessment and early warning.
        </p>

        {/* Primary & Secondary Action CTAs */}
        <div className="flex flex-col sm:flex-row items-center gap-4 w-full sm:w-auto mb-6">
          <a
            href="#intelligence"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-7 py-3.5 rounded-xl font-semibold text-sm bg-gradient-to-r from-[#35B8AD] to-[#20AFA7] hover:from-[#2eb0a5] hover:to-[#1a9f97] text-[#061322] shadow-xl hover:shadow-[#35B8AD]/20 transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            <Compass className="h-4 w-4" />
            <span>Explore Ocean Intelligence</span>
          </a>

          <Link
            to="/"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-7 py-3.5 rounded-xl font-semibold text-sm bg-white/10 hover:bg-white/15 text-white border border-white/20 backdrop-blur-md shadow-lg transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            <span>Enter Dashboard</span>
            <ArrowRight className="h-4 w-4 text-[#35B8AD]" />
          </Link>
        </div>

        {/* 21st.dev Particle Wave Dynamic Fluid Layer */}
        <div className="w-full max-w-3xl h-28 relative my-2 pointer-events-auto opacity-75">
          <ParticleWave
            className="w-full h-full"
            particleCountX={34}
            particleCountY={12}
            separation={28}
            waveHeight={18}
            speed={0.02}
            primaryColor="#246B9E"
            accentColor="#35B8AD"
          />
        </div>

        {/* Micro-Features Metadata Ribbon */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-6 w-full max-w-3xl pt-5 border-t border-white/10 text-xs font-mono text-[#8CC8E8]/80">
          <div className="flex flex-col items-center">
            <span className="text-white font-bold text-sm">3 Architectures</span>
            <span className="text-[10px] text-[#8CC8E8]/70">LSTM &bull; ANN &bull; 1D-CNN</span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-white font-bold text-sm">3 Lead Horizons</span>
            <span className="text-[10px] text-[#8CC8E8]/70">+1h &bull; +6h &bull; +24h</span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-white font-bold text-sm">8 Coastal Nodes</span>
            <span className="text-[10px] text-[#8CC8E8]/70">Arabian Sea & Bay of Bengal</span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-white font-bold text-sm">129,696 Hours</span>
            <span className="text-[10px] text-[#8CC8E8]/70">14.8-Yr ERA5 Training Baseline</span>
          </div>
        </div>
      </div>

      {/* Down Scroll Indicator */}
      <a
        href="#intelligence"
        className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center text-white/40 hover:text-white transition-colors"
        aria-label="Scroll to Live Intelligence"
      >
        <span className="text-[10px] font-mono tracking-widest uppercase mb-1">Scroll</span>
        <ChevronDown className="h-4 w-4 animate-bounce" />
      </a>
    </section>
  )
}

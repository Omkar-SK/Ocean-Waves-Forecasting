import React from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Compass, Waves, ShieldCheck } from 'lucide-react'

export function CinematicCTA() {
  return (
    <section className="relative py-28 bg-[#0a1829] overflow-hidden text-center text-white border-b border-[#246B9E]/20">
      {/* Background glow & subtle wave contours */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[350px] bg-gradient-to-r from-[#246B9E]/25 via-[#35B8AD]/20 to-[#246B9E]/25 blur-3xl rounded-full pointer-events-none" />

      <div className="relative z-10 max-w-4xl mx-auto px-4 md:px-8">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/10 backdrop-blur-md border border-white/15 text-xs font-mono font-semibold uppercase text-[#35B8AD] mb-6">
          <Waves className="h-3.5 w-3.5" />
          <span>Operational Marine Platform</span>
        </div>

        <h2 className="text-4xl sm:text-5xl md:text-6xl font-extrabold font-['Outfit',sans-serif] tracking-tight leading-tight mb-6">
          The Ocean Is Constantly Changing.{' '}
          <span className="block bg-gradient-to-r from-white via-[#8CC8E8] to-[#35B8AD] bg-clip-text text-transparent">
            Stay One Step Ahead.
          </span>
        </h2>

        <p className="text-base sm:text-lg text-[#8CC8E8]/90 max-w-2xl mx-auto font-normal leading-relaxed mb-10">
          Access multi-horizon neural forecasts, evaluate localized risk thresholds across 8 coastal promontories, and stress-test dangerous cyclone scenarios.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            to="/"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-8 py-4 rounded-xl font-bold text-sm bg-gradient-to-r from-[#35B8AD] to-[#20AFA7] hover:from-[#2eb0a5] hover:to-[#1a9f97] text-[#061322] shadow-2xl hover:shadow-[#35B8AD]/30 transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            <span>Enter OceanCast Dashboard</span>
            <ArrowRight className="h-4 w-4" />
          </Link>

          <Link
            to="/about"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-8 py-4 rounded-xl font-semibold text-sm bg-white/10 hover:bg-white/15 text-white border border-white/20 backdrop-blur-md transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            <Compass className="h-4 w-4 text-[#35B8AD]" />
            <span>Research Methodology</span>
          </Link>
        </div>

        {/* Bottom micro-indicator */}
        <div className="mt-12 flex items-center justify-center gap-6 text-xs text-[#8CC8E8]/60 font-mono">
          <span className="flex items-center gap-1.5">
            <ShieldCheck className="h-3.5 w-3.5 text-[#35B8AD]" />
            <span>Open Access Prototype</span>
          </span>
          <span>&bull;</span>
          <span>Zero Secret Tokens Required</span>
          <span>&bull;</span>
          <span>FastAPI + PyTorch Inference</span>
        </div>
      </div>
    </section>
  )
}

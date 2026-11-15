import React from 'react'
import { Link } from 'react-router-dom'
import {
  Waves,
  Compass,
  AlertTriangle,
  FlaskConical,
  ArrowRight,
  TrendingUp,
  ShieldCheck,
  Flame,
  Layers,
  Cpu,
} from 'lucide-react'

export function CoreCapabilities() {
  const capabilities = [
    {
      id: 'forecasting',
      title: 'Wave Forecasting',
      badge: 'Multi-Horizon Neural Engine',
      icon: Waves,
      accentColor: '#246B9E',
      iconBg: 'bg-[#EAF2F9]',
      description:
        'Continuous deep learning synthesis predicting significant wave heights (Hs), peak wave periods (Tp), and directional swell across +1h, +6h, and +24h lead horizons.',
      highlights: [
        '+13.88% RMSE skill score over persistence at +6h',
        '13 canonical ocean-atmospheric input features',
        'Fitted zero-lookahead MinMax normalization',
      ],
      link: '/live',
      linkLabel: 'Explore Live Forecasts',
    },
    {
      id: 'coastal-risk',
      title: 'Coastal Risk Intelligence',
      badge: 'Geographic Assessment',
      icon: Compass,
      accentColor: '#20AFA7',
      iconBg: 'bg-[#E1F5F2]',
      description:
        'Spatially evaluated risk categorization across 8 Indian coastal grid stations, mapping localized bathymetry and swell exposure into actionable engineering risk tiers.',
      highlights: [
        'Dynamic caution, high-risk, and critical thresholds',
        'Calibrated reference at Mumbai Offshore (18.50°N)',
        'Zero-watermark OpenStreetMap raster cartography',
      ],
      link: '/map',
      linkLabel: 'View Coastal Risk Map',
    },
    {
      id: 'early-warning',
      title: 'Early Warning System',
      badge: 'Lifecycle Management',
      icon: AlertTriangle,
      accentColor: '#D97706',
      iconBg: 'bg-[#FEF3C7]',
      description:
        'Automated hazard detection engine with persistent lifecycle states (ACTIVE, ACKNOWLEDGED, RESOLVED, EXPIRED) and hysteresis buffers to eliminate alert flickering.',
      highlights: [
        'Deterministic fingerprint duplicate suppression',
        'Audit-logged operator acknowledgement',
        'Decoupled from model drift monitoring events',
      ],
      link: '/alerts',
      linkLabel: 'Inspect Alert Centre',
    },
    {
      id: 'simulation',
      title: 'Ocean Hazard Simulation',
      badge: 'Stress-Testing & Drills',
      icon: FlaskConical,
      accentColor: '#E11D48',
      iconBg: 'bg-[#FFE4E6]',
      description:
        'Interactive disaster-response simulator generating realistic Category 3/4 cyclones, destructive long-period swell, and spring-tide coastal surge without persistent DB mutation.',
      highlights: [
        '4 predefined stress scenarios (Cyclone, Swell, Surge)',
        '24-hour interactive storm progression scrubber',
        'Instant live data restoration with zero contamination',
      ],
      link: '/',
      linkLabel: 'Launch Simulator in Dashboard',
    },
  ]

  return (
    <section className="py-24 bg-white border-b border-[#DCE5EF]">
      <div className="max-w-7xl mx-auto px-4 md:px-8">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#EAF2F9] border border-[#246B9E]/20 text-xs font-mono font-semibold uppercase text-[#246B9E] mb-3">
            <Cpu className="h-3.5 w-3.5" />
            <span>Platform Capabilities</span>
          </div>
          <h2 className="text-3xl md:text-5xl font-bold font-['Outfit',sans-serif] text-[#102A43] tracking-tight">
            Scientific Marine Intelligence, Engineered for Decision Support
          </h2>
          <p className="text-[#64758A] text-base md:text-lg mt-4 font-normal leading-relaxed">
            Bridging raw numerical weather predictions with temporal deep learning architectures to provide reliable wave forecasting and coastal early warnings.
          </p>
        </div>

        {/* 4 Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {capabilities.map((cap) => {
            const Icon = cap.icon
            return (
              <div
                key={cap.id}
                className="group relative rounded-3xl p-8 bg-[#F4F7FB] border border-[#DCE5EF] hover:border-[#8CC8E8] transition-all duration-300 hover:shadow-xl hover:-translate-y-1 flex flex-col justify-between"
              >
                <div>
                  {/* Top Row: Icon & Badge */}
                  <div className="flex items-center justify-between mb-6">
                    <div
                      className={`h-14 w-14 rounded-2xl ${cap.iconBg} flex items-center justify-center text-[#102A43] shadow-xs group-hover:scale-110 transition-transform`}
                    >
                      <Icon className="h-7 w-7" style={{ color: cap.accentColor }} />
                    </div>

                    <span className="text-[11px] font-mono font-semibold uppercase tracking-wider px-3 py-1 rounded-full bg-white border border-[#DCE5EF] text-[#102A43] shadow-2xs">
                      {cap.badge}
                    </span>
                  </div>

                  {/* Title & Description */}
                  <h3 className="text-2xl font-bold font-['Outfit',sans-serif] text-[#102A43] mb-3 tracking-tight">
                    {cap.title}
                  </h3>
                  <p className="text-[#64758A] text-sm leading-relaxed mb-6 font-normal">
                    {cap.description}
                  </p>

                  {/* Highlights List */}
                  <ul className="space-y-2 mb-8 text-xs font-medium text-[#192B3C]">
                    {cap.highlights.map((item, idx) => (
                      <li key={idx} className="flex items-center gap-2">
                        <span
                          className="h-1.5 w-1.5 rounded-full"
                          style={{ backgroundColor: cap.accentColor }}
                        />
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Footer Link */}
                <div className="pt-4 border-t border-[#DCE5EF]">
                  <Link
                    to={cap.link}
                    className="inline-flex items-center gap-2 text-sm font-semibold transition-colors hover:translate-x-1 duration-200"
                    style={{ color: cap.accentColor }}
                  >
                    <span>{cap.linkLabel}</span>
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}

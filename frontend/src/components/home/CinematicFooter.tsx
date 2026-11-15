import React from 'react'
import { Link } from 'react-router-dom'
import { Waves, ShieldAlert, Cpu, ExternalLink, ArrowRight } from 'lucide-react'

export function CinematicFooter() {
  const dashboardLinks = [
    { label: 'Mission Overview', path: '/' },
    { label: 'Live Wave Forecast', path: '/live' },
    { label: 'Coastal Risk Map', path: '/map' },
    { label: 'Early-Warning Alerts', path: '/alerts' },
    { label: 'Coastal Node Registry', path: '/locations' },
    { label: 'Accuracy & Drift Monitoring', path: '/monitoring' },
    { label: 'Model Benchmarking', path: '/comparison' },
    { label: 'Historical Test Analysis', path: '/historical' },
    { label: 'System & Methodology', path: '/about' },
  ]

  const techStack = [
    'PyTorch 2.6 Neural Inference',
    'WaveLSTM (Recurrent Network)',
    'WaveANN (Dense MLP)',
    'WaveCNN1D (Temporal Conv)',
    'FastAPI & Uvicorn Async Server',
    'React 19 & TypeScript',
    'Three.js & React Three Fiber',
    'Leaflet & OpenStreetMap Raster',
    '14.8-Yr ERA5 Reanalysis (129,696 hrs)',
  ]

  return (
    <footer id="about" className="bg-[#061322] text-[#8CC8E8] pt-20 pb-12 border-t border-white/10">
      <div className="max-w-7xl mx-auto px-4 md:px-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-12 gap-10 pb-16 border-b border-white/10">
          {/* Column 1: Brand & Project Overview (4 Cols) */}
          <div className="lg:col-span-4 space-y-4">
            <Link to="/home" className="flex items-center gap-2.5">
              <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-[#246B9E] to-[#102A43] flex items-center justify-center text-[#35B8AD] border border-[#35B8AD]/30 shadow-md">
                <Waves className="h-5 w-5" />
              </div>
              <span className="font-['Outfit',sans-serif] text-xl font-bold text-white tracking-tight">
                OceanCast <span className="text-[#35B8AD]">AI</span>
              </span>
            </Link>

            <p className="text-xs text-[#8CC8E8]/80 leading-relaxed font-normal">
              A real-time ocean wave forecasting and coastal early-warning platform. Benchmarking deep learning sequence models against naive persistence and numerical weather prediction across Arabian Sea coastal promontories.
            </p>

            <div className="pt-2 text-xs font-mono text-[#8CC8E8]/60 space-y-1">
              <div>Domain: Deep Learning & Marine Intelligence</div>
              <div>Scope: Arabian Sea & Indian Coastline</div>
              <div>Lead Horizons: +1h &bull; +6h &bull; +24h</div>
            </div>
          </div>

          {/* Column 2: Dashboard Navigation (3 Cols) */}
          <div className="lg:col-span-3 space-y-3">
            <h4 className="text-xs font-mono uppercase tracking-wider text-white font-bold">
              Dashboard Navigation
            </h4>
            <ul className="space-y-2 text-xs">
              {dashboardLinks.map((link) => (
                <li key={link.path}>
                  <Link
                    to={link.path}
                    className="hover:text-[#35B8AD] transition-colors flex items-center gap-1.5"
                  >
                    <span>&bull;</span>
                    <span>{link.label}</span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Column 3: Technology Stack (2 Cols) */}
          <div className="lg:col-span-2 space-y-3">
            <h4 className="text-xs font-mono uppercase tracking-wider text-white font-bold">
              Technology Stack
            </h4>
            <ul className="space-y-1.5 text-xs text-[#8CC8E8]/75 font-mono">
              {techStack.map((tech, i) => (
                <li key={i} className="truncate">
                  {tech}
                </li>
              ))}
            </ul>
          </div>

          {/* Column 4: Scientific Limitations & Provenance (3 Cols) */}
          <div className="lg:col-span-3 space-y-3">
            <div className="flex items-center gap-2 text-white font-bold text-xs font-mono uppercase tracking-wider">
              <ShieldAlert className="h-4 w-4 text-[#F59E0B]" />
              <span>Scientific Notices</span>
            </div>

            <div className="text-[11px] text-[#8CC8E8]/75 leading-relaxed space-y-2">
              <p>
                <strong>Operational Data:</strong> Real-time feeds query Open-Meteo ECMWF WAM wave model and IFS atmospheric predictions on an ~0.08&deg; grid. Moored in-situ waverider buoys are unconfigured.
              </p>
              <p>
                <strong>Accuracy Reality:</strong> WaveLSTM outperforms persistence at +6h (+13.88% Skill Score). At +1h, naive persistence dominates due to physical wave inertia (r₁ = 0.9995).
              </p>
              <p>
                <strong>Advisory Disclaimer:</strong> Internal thresholds (Green, Yellow, Orange, Red) are engineering metrics, not official maritime warnings from IMD or INCOIS.
              </p>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 flex flex-col sm:flex-row items-center justify-between text-xs text-[#8CC8E8]/50 font-mono gap-4">
          <div>
            &copy; 2026 OceanCast AI &bull; Deep Learning Ocean Wave Radar
          </div>

          <div className="flex items-center gap-4">
            <Link to="/" className="text-[#35B8AD] hover:underline flex items-center gap-1">
              <span>Enter Operational Dashboard</span>
              <ArrowRight className="h-3 w-3" />
            </Link>
          </div>
        </div>
      </div>
    </footer>
  )
}

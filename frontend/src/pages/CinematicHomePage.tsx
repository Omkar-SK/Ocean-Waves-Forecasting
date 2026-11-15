import React, { useEffect } from 'react'
import { CinematicNavbar } from '../components/home/CinematicNavbar'
import { CinematicHero } from '../components/home/CinematicHero'
import { LiveOceanIntelligence } from '../components/home/LiveOceanIntelligence'
import { CoreCapabilities } from '../components/home/CoreCapabilities'
import { InteractiveOceanGlobe } from '../components/home/InteractiveOceanGlobe'
import { DeepLearningTechnology } from '../components/home/DeepLearningTechnology'
import { CinematicCTA } from '../components/home/CinematicCTA'
import { CinematicFooter } from '../components/home/CinematicFooter'
import { Globe, Compass } from 'lucide-react'

export function CinematicHomePage() {
  useEffect(() => {
    window.scrollTo(0, 0)
    document.title = 'OceanCast AI — Deep Learning Wave Radar & Marine Intelligence'
  }, [])

  return (
    <div className="min-h-screen bg-[#F4F7FB] text-[#192B3C] selection:bg-[#35B8AD]/30 selection:text-[#102A43] overflow-x-hidden">
      {/* 1. Transparent Floating Navigation Bar */}
      <CinematicNavbar />

      {/* 2. Hero Section: Full-Screen 3D Procedural Ocean */}
      <CinematicHero />

      {/* 3. Section 2: Live Ocean Intelligence */}
      <LiveOceanIntelligence />

      {/* 4. Section 3: Core Platform Capabilities */}
      <CoreCapabilities />

      {/* 5. Section 4: The Ocean in 3D (Interactive Globe with 8 Coastal Nodes) */}
      <section id="nodes-3d" className="py-24 bg-[#0a1829] text-white border-b border-[#246B9E]/20 relative">
        <div className="max-w-7xl mx-auto px-4 md:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/10 backdrop-blur-md border border-white/15 text-xs font-mono font-semibold uppercase text-[#35B8AD] mb-3">
              <Globe className="h-3.5 w-3.5" />
              <span>3D Geospatial Radar</span>
            </div>
            <h2 className="text-3xl md:text-5xl font-bold font-['Outfit',sans-serif] tracking-tight">
              The Ocean in 3D
            </h2>
            <p className="text-[#8CC8E8]/90 text-base md:text-lg mt-4 font-normal leading-relaxed">
              Explore the Indian coastline and Arabian Sea basin in an interactive 3D globe. Inspect coordinates, calibration status, and localized telemetry across 8 registered candidate stations.
            </p>
          </div>

          {/* Interactive 3D Canvas Globe */}
          <InteractiveOceanGlobe />
        </div>
      </section>

      {/* 6. Section 5: Deep Learning Technology (ANN, LSTM, 1D-CNN) */}
      <DeepLearningTechnology />

      {/* 7. Section 6: Call To Action */}
      <CinematicCTA />

      {/* 8. Section 7: Footer */}
      <CinematicFooter />
    </div>
  )
}

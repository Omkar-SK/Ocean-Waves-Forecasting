import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  Waves,
  ArrowRight,
  Menu,
  X,
  Compass,
  Activity,
  Globe,
  Cpu,
  Info,
} from 'lucide-react'
import { TubelightNavbar, type NavItem } from '../ui/TubelightNavbar'

export function CinematicNavbar() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('Home')

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 40)
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const navItems: NavItem[] = [
    { name: 'Home', url: '#hero', icon: Waves },
    { name: 'Intelligence', url: '#intelligence', icon: Activity },
    { name: '3D Ocean', url: '#nodes-3d', icon: Globe },
    { name: 'Technology', url: '#technology', icon: Cpu },
    { name: 'About', url: '#about', icon: Info },
  ]

  return (
    <header className="fixed top-0 left-0 right-0 z-50 px-4 md:px-8 py-3 transition-all duration-300">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Brand Logo */}
        <Link to="/home" className="flex items-center gap-2.5 group">
          <div className="relative flex items-center justify-center h-10 w-10 rounded-xl bg-gradient-to-br from-[#246B9E] to-[#102A43] text-[#35B8AD] shadow-lg group-hover:scale-105 transition-transform border border-[#35B8AD]/40">
            <Waves className="h-5 w-5 animate-pulse" />
          </div>
          <div className="flex flex-col">
            <span className="font-['Outfit',sans-serif] text-base md:text-lg font-bold tracking-tight text-white leading-none">
              OceanCast <span className="text-[#35B8AD]">AI</span>
            </span>
            <span className="text-[9px] font-mono tracking-widest uppercase mt-0.5 text-[#8CC8E8]">
              Wave Intelligence
            </span>
          </div>
        </Link>

        {/* 21st.dev Tubelight Floating Navbar */}
        <div className="hidden md:flex items-center justify-center">
          <TubelightNavbar
            items={navItems}
            activeTab={activeTab}
            onTabChange={(name) => setActiveTab(name)}
          />
        </div>

        {/* Action Button: Enter Dashboard */}
        <div className="hidden md:flex items-center gap-3">
          <Link
            to="/"
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold tracking-wide bg-gradient-to-r from-[#246B9E] to-[#102A43] hover:from-[#1d557e] hover:to-[#0c1f32] text-white shadow-md border border-[#35B8AD]/40 transition-all hover:scale-[1.03] active:scale-[0.98]"
          >
            <span>Enter Dashboard</span>
            <ArrowRight className="h-3.5 w-3.5 text-[#35B8AD]" />
          </Link>
        </div>

        {/* Mobile Hamburger Toggle */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="md:hidden p-2 rounded-xl bg-[#102A43]/80 border border-[#35B8AD]/30 text-white hover:text-[#35B8AD] transition-colors"
          aria-label="Toggle navigation menu"
        >
          {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden mt-3 rounded-2xl bg-[#102A43]/95 backdrop-blur-xl border border-[#35B8AD]/30 p-5 shadow-2xl animate-ocean-fade">
          <div className="flex flex-col gap-3 font-medium text-sm">
            {navItems.map((item) => (
              <a
                key={item.name}
                href={item.url}
                onClick={() => {
                  setActiveTab(item.name)
                  setMobileMenuOpen(false)
                }}
                className={`py-2 px-3 rounded-lg flex items-center gap-2.5 transition-colors ${
                  activeTab === item.name
                    ? 'bg-[#246B9E]/30 text-white font-bold border-l-2 border-[#35B8AD]'
                    : 'text-[#8CC8E8] hover:text-white hover:bg-white/5'
                }`}
              >
                {item.icon && <item.icon className="h-4 w-4 text-[#35B8AD]" />}
                <span>{item.name}</span>
              </a>
            ))}

            <div className="pt-3 border-t border-[#35B8AD]/20 mt-2">
              <Link
                to="/"
                onClick={() => setMobileMenuOpen(false)}
                className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-[#35B8AD] to-[#20AFA7] text-[#102A43] shadow-md"
              >
                <span>Enter Operational Dashboard</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>
        </div>
      )}
    </header>
  )
}

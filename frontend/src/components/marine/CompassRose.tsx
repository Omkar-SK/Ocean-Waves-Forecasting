import React from 'react'

interface CompassRoseProps {
  degrees: number | null
  size?: number
  label?: string
  className?: string
}

export function CompassRose({
  degrees,
  size = 56,
  label = 'Wind Dir',
  className = '',
}: CompassRoseProps) {
  const rotation = degrees !== null && !isNaN(degrees) ? degrees : 0
  const cardinal =
    degrees === null
      ? '—'
      : degrees >= 337.5 || degrees < 22.5
      ? 'N'
      : degrees < 67.5
      ? 'NE'
      : degrees < 112.5
      ? 'E'
      : degrees < 157.5
      ? 'SE'
      : degrees < 202.5
      ? 'S'
      : degrees < 247.5
      ? 'SW'
      : degrees < 292.5
      ? 'W'
      : 'NW'

  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <div
        className="relative flex items-center justify-center rounded-full border border-[#DCE5EF] bg-[#F0F6FC]"
        style={{ width: `${size}px`, height: `${size}px` }}
      >
        {/* Cardinal marks */}
        <span className="absolute top-0.5 text-[8px] font-mono font-bold text-[#15304B]">N</span>
        <span className="absolute right-1 text-[8px] font-mono text-[#8291A2]">E</span>
        <span className="absolute bottom-0.5 text-[8px] font-mono text-[#8291A2]">S</span>
        <span className="absolute left-1 text-[8px] font-mono text-[#8291A2]">W</span>

        {/* Needle indicator */}
        <svg
          viewBox="0 0 40 40"
          className="h-full w-full transition-transform duration-500 ease-out"
          style={{ transform: `rotate(${rotation}deg)` }}
          aria-hidden="true"
        >
          {/* North pointing arrow (Teal) */}
          <polygon points="20,7 23,20 20,18 17,20" fill="#20AFA7" />
          {/* South pointing tail (Muted Slate) */}
          <polygon points="20,33 23,20 20,22 17,20" fill="#8291A2" opacity="0.7" />
          {/* Center pivot */}
          <circle cx="20" cy="20" r="2.5" fill="#15304B" />
        </svg>
      </div>

      <div className="flex flex-col">
        <span className="text-[10px] uppercase tracking-wider text-[#64758A] font-semibold">{label}</span>
        <div className="flex items-baseline gap-1 font-mono text-sm font-bold text-[#192B3C]">
          <span>{degrees !== null ? `${Math.round(degrees)}°` : '—'}</span>
          <span className="text-xs text-[#20AFA7] font-semibold">{cardinal}</span>
        </div>
      </div>
    </div>
  )
}

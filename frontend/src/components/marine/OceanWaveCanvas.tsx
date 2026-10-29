import React, { useEffect, useRef } from 'react'

interface OceanWaveCanvasProps {
  className?: string
  height?: number
  speedMultiplier?: number
  waveHeight?: number
  variant?: 'light' | 'navy'
}

export function OceanWaveCanvas({
  className = '',
  height = 140,
  speedMultiplier = 1,
  waveHeight = 24,
  variant = 'light',
}: OceanWaveCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animationFrameId: number
    let step = 0

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    const resize = () => {
      const dpr = window.devicePixelRatio || 1
      const rect = canvas.getBoundingClientRect()
      canvas.width = rect.width * dpr
      canvas.height = (height || rect.height) * dpr
      ctx.scale(dpr, dpr)
    }

    resize()
    window.addEventListener('resize', resize)

    // Wave parameters: [speed, freq, amp, y, stroke, fill]
    const waves =
      variant === 'navy'
        ? [
            {
              speed: 0.012 * speedMultiplier,
              freq: 0.007,
              amp: waveHeight * 0.9,
              y: height * 0.55,
              stroke: 'rgba(32, 175, 167, 0.4)',
              fill: 'rgba(32, 175, 167, 0.08)',
            },
            {
              speed: -0.009 * speedMultiplier,
              freq: 0.005,
              amp: waveHeight * 1.2,
              y: height * 0.65,
              stroke: 'rgba(121, 183, 229, 0.35)',
              fill: 'rgba(121, 183, 229, 0.06)',
            },
            {
              speed: 0.006 * speedMultiplier,
              freq: 0.009,
              amp: waveHeight * 0.6,
              y: height * 0.75,
              stroke: 'rgba(57, 127, 184, 0.25)',
              fill: 'rgba(21, 48, 75, 0.3)',
            },
          ]
        : [
            {
              speed: 0.012 * speedMultiplier,
              freq: 0.006,
              amp: waveHeight * 0.9,
              y: height * 0.55,
              stroke: 'rgba(36, 91, 143, 0.25)',
              fill: 'rgba(220, 236, 249, 0.4)',
            },
            {
              speed: -0.008 * speedMultiplier,
              freq: 0.005,
              amp: waveHeight * 1.1,
              y: height * 0.65,
              stroke: 'rgba(32, 175, 167, 0.3)',
              fill: 'rgba(225, 245, 242, 0.45)',
            },
            {
              speed: 0.005 * speedMultiplier,
              freq: 0.008,
              amp: waveHeight * 0.6,
              y: height * 0.75,
              stroke: 'rgba(121, 183, 229, 0.2)',
              fill: 'rgba(240, 246, 252, 0.5)',
            },
          ]

    const render = () => {
      if (!ctx || !canvas) return
      const w = canvas.getBoundingClientRect().width
      const h = height

      ctx.clearRect(0, 0, w, h)

      waves.forEach((wave) => {
        ctx.beginPath()
        ctx.moveTo(0, h)

        for (let x = 0; x <= w; x += 3) {
          const y =
            wave.y +
            Math.sin(x * wave.freq + step * wave.speed) * wave.amp +
            Math.cos(x * wave.freq * 0.5 + step * wave.speed * 0.7) * (wave.amp * 0.4)
          ctx.lineTo(x, y)
        }

        ctx.lineTo(w, h)
        ctx.closePath()

        ctx.fillStyle = wave.fill
        ctx.fill()

        ctx.lineWidth = 1.2
        ctx.strokeStyle = wave.stroke
        ctx.stroke()
      })

      if (!prefersReducedMotion) {
        step += 1
        animationFrameId = requestAnimationFrame(render)
      }
    }

    render()

    return () => {
      cancelAnimationFrame(animationFrameId)
      window.removeEventListener('resize', resize)
    }
  }, [height, speedMultiplier, waveHeight, variant])

  return (
    <canvas
      ref={canvasRef}
      className={`pointer-events-none w-full ${className}`}
      style={{ height: `${height}px` }}
      aria-hidden="true"
    />
  )
}

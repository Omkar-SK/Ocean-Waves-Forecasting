import React, { useRef, useEffect } from 'react'

interface ParticleWaveProps {
  className?: string
  particleCountX?: number
  particleCountY?: number
  separation?: number
  speed?: number
  waveHeight?: number
  interactive?: boolean
  primaryColor?: string
  accentColor?: string
}

export function ParticleWave({
  className = '',
  particleCountX = 40,
  particleCountY = 24,
  separation = 32,
  speed = 0.025,
  waveHeight = 35,
  interactive = true,
  primaryColor = '#246B9E',
  accentColor = '#35B8AD',
}: ParticleWaveProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animationFrameId: number
    let count = 0
    let mouseX = 0
    let mouseY = 0
    let targetMouseX = 0
    let targetMouseY = 0

    const resize = () => {
      canvas.width = canvas.parentElement?.clientWidth || window.innerWidth
      canvas.height = canvas.parentElement?.clientHeight || 450
    }
    resize()
    window.addEventListener('resize', resize)

    const handleMouseMove = (e: MouseEvent) => {
      if (!interactive) return
      const rect = canvas.getBoundingClientRect()
      targetMouseX = (e.clientX - rect.left - canvas.width / 2) * 0.4
      targetMouseY = (e.clientY - rect.top - canvas.height / 2) * 0.4
    }

    if (interactive) {
      window.addEventListener('mousemove', handleMouseMove)
    }

    // Precalculate particle grid
    const numParticles = particleCountX * particleCountY
    const positions = new Float32Array(numParticles * 3)

    let i = 0
    for (let ix = 0; ix < particleCountX; ix++) {
      for (let iy = 0; iy < particleCountY; iy++) {
        positions[i] = ix * separation - (particleCountX * separation) / 2
        positions[i + 1] = 0
        positions[i + 2] = iy * separation - (particleCountY * separation) / 2
        i += 3
      }
    }

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      mouseX += (targetMouseX - mouseX) * 0.05
      mouseY += (targetMouseY - mouseY) * 0.05

      const centerX = canvas.width / 2 + mouseX * 0.2
      const centerY = canvas.height / 2 + 30 + mouseY * 0.2

      let i = 0
      for (let ix = 0; ix < particleCountX; ix++) {
        for (let iy = 0; iy < particleCountY; iy++) {
          // Dynamic wave harmonic displacement
          const wave1 = Math.sin((ix + count) * 0.3) * waveHeight
          const wave2 = Math.cos((iy + count) * 0.4) * (waveHeight * 0.6)
          const wave3 = Math.sin((ix + iy + count * 1.5) * 0.2) * (waveHeight * 0.35)
          const yPos = wave1 + wave2 + wave3

          const x = positions[i]
          const y = positions[i + 1] + yPos
          const z = positions[i + 2] + 450 // Depth offset

          // Perspective projection
          const fov = 400
          const scale = fov / (fov + z)
          const projX = centerX + x * scale
          const projY = centerY + y * scale

          // Depth-based radius and opacity
          const radius = Math.max(0.6, (1.8 + (yPos / waveHeight) * 0.8) * scale * 1.8)
          const alpha = Math.min(0.9, Math.max(0.12, (z / 800) * scale * 1.4))

          // Color blending between Teal (#35B8AD) at crests and Ocean Blue (#246B9E) at troughs
          const crestRatio = Math.max(0, Math.min(1, (yPos + waveHeight) / (waveHeight * 2)))

          ctx.beginPath()
          ctx.arc(projX, projY, radius, 0, Math.PI * 2)

          if (crestRatio > 0.65) {
            ctx.fillStyle = accentColor
            ctx.shadowColor = accentColor
            ctx.shadowBlur = 6
          } else {
            ctx.fillStyle = primaryColor
            ctx.shadowBlur = 0
          }

          ctx.globalAlpha = alpha
          ctx.fill()

          i += 3
        }
      }

      ctx.globalAlpha = 1.0
      ctx.shadowBlur = 0
      count += speed
      animationFrameId = requestAnimationFrame(render)
    }

    render()

    return () => {
      window.removeEventListener('resize', resize)
      if (interactive) window.removeEventListener('mousemove', handleMouseMove)
      cancelAnimationFrame(animationFrameId)
    }
  }, [particleCountX, particleCountY, separation, speed, waveHeight, interactive, primaryColor, accentColor])

  return (
    <canvas
      ref={canvasRef}
      className={`w-full h-full block pointer-events-none ${className}`}
      style={{ background: 'transparent' }}
    />
  )
}

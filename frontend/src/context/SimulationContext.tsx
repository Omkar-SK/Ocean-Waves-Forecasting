import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { api } from '../services/api'
import type {
  ScenarioMetadataItem,
  SimulationResultsResponse,
  SimulationStatusResponse,
} from '../types'

interface SimulationContextValue {
  isSimulationActive: boolean
  simulationState: 'IDLE' | 'RUNNING' | 'PAUSED'
  simulationStatus: SimulationStatusResponse | null
  activeScenario: ScenarioMetadataItem | null
  simulationData: SimulationResultsResponse | null
  scenarios: ScenarioMetadataItem[]
  currentStep: number
  totalSteps: number
  isModalOpen: boolean
  isLoading: boolean
  error: string | null
  openModal: () => void
  closeModal: () => void
  startSimulation: (scenarioId: string, initialStep?: number) => Promise<void>
  pauseSimulation: () => Promise<void>
  resumeSimulation: () => Promise<void>
  stepSimulation: (delta?: number) => Promise<void>
  resetSimulation: () => Promise<void>
  reset: () => Promise<void> // Convenient alias for resetSimulation
}

const defaultSimulationContextValue: SimulationContextValue = {
  isSimulationActive: false,
  simulationState: 'IDLE',
  simulationStatus: null,
  activeScenario: null,
  simulationData: null,
  scenarios: [],
  currentStep: 6,
  totalSteps: 24,
  isModalOpen: false,
  isLoading: false,
  error: null,
  openModal: () => {},
  closeModal: () => {},
  startSimulation: async () => {},
  pauseSimulation: async () => {},
  resumeSimulation: async () => {},
  stepSimulation: async () => {},
  resetSimulation: async () => {},
  reset: async () => {},
}

const SimulationContext = createContext<SimulationContextValue | undefined>(undefined)

export function SimulationProvider({ children }: { children: React.ReactNode }) {
  const [isSimulationActive, setIsSimulationActive] = useState<boolean>(false)
  const [simulationState, setSimulationState] = useState<'IDLE' | 'RUNNING' | 'PAUSED'>('IDLE')
  const [simulationStatus, setSimulationStatus] = useState<SimulationStatusResponse | null>(null)
  const [activeScenario, setActiveScenario] = useState<ScenarioMetadataItem | null>(null)
  const [simulationData, setSimulationData] = useState<SimulationResultsResponse | null>(null)
  const [scenarios, setScenarios] = useState<ScenarioMetadataItem[]>([])
  const [currentStep, setCurrentStep] = useState<number>(6)
  const [totalSteps] = useState<number>(24)
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  // 1. Fetch available scenarios catalog on mount
  useEffect(() => {
    let isMounted = true
    const loadScenarios = async () => {
      try {
        if (typeof api.getSimulationScenarios === 'function') {
          const list = await api.getSimulationScenarios()
          if (isMounted && list) {
            setScenarios(list)
          }
        }
      } catch (err) {
        console.warn('Could not fetch simulation scenarios catalog:', err)
      }
    }
    loadScenarios()
    return () => {
      isMounted = false
    }
  }, [])

  // 2. Sync initial status from backend
  useEffect(() => {
    let isMounted = true
    const checkStatus = async () => {
      try {
        if (typeof api.getSimulationStatus === 'function') {
          const st = await api.getSimulationStatus()
          if (!isMounted) return
          setSimulationStatus(st)
          setIsSimulationActive(st.is_active)
          setSimulationState((st.state as 'IDLE' | 'RUNNING' | 'PAUSED') || 'IDLE')
          setCurrentStep(st.current_step)

          if (st.is_active && typeof api.getSimulationResults === 'function') {
            const res = await api.getSimulationResults()
            if (!isMounted) return
            setSimulationData(res)
            if (res.scenario) setActiveScenario(res.scenario)
          }
        }
      } catch (err) {
        // Backend might be offline or starting up
        console.debug('Initial simulation status check:', err)
      }
    }
    checkStatus()
    return () => {
      isMounted = false
    }
  }, [])

  const openModal = useCallback(() => setIsModalOpen(true), [])
  const closeModal = useCallback(() => setIsModalOpen(false), [])

  const startSimulation = useCallback(
    async (scenarioId: string, initialStep: number = 6) => {
      setIsLoading(true)
      setError(null)
      try {
        const data = await api.startSimulation(scenarioId, initialStep)
        setSimulationData(data)
        setSimulationStatus(data.status)
        setIsSimulationActive(data.status.is_active)
        setSimulationState((data.status.state as 'IDLE' | 'RUNNING' | 'PAUSED') || 'RUNNING')
        setCurrentStep(data.status.current_step)
        setActiveScenario(data.scenario ?? null)
        setIsModalOpen(false)
        window.dispatchEvent(new CustomEvent('oceancast:simulation-started', { detail: { scenarioId } }))
      } catch (err: any) {
        const msg = err?.message || 'Failed to start simulation scenario'
        setError(msg)
        console.error('Error starting simulation:', err)
      } finally {
        setIsLoading(false)
      }
    },
    []
  )

  const pauseSimulation = useCallback(async () => {
    try {
      const st = await api.pauseSimulation()
      setSimulationStatus(st)
      setSimulationState('PAUSED')
    } catch (err: any) {
      console.error('Error pausing simulation:', err)
    }
  }, [])

  const resumeSimulation = useCallback(async () => {
    try {
      const st = await api.resumeSimulation()
      setSimulationStatus(st)
      setSimulationState('RUNNING')
    } catch (err: any) {
      console.error('Error resuming simulation:', err)
    }
  }, [])

  const stepSimulation = useCallback(async (delta: number = 1) => {
    setIsLoading(true)
    try {
      const data = await api.stepSimulation(delta)
      setSimulationData(data)
      setSimulationStatus(data.status)
      setCurrentStep(data.status.current_step)
      if (data.scenario) setActiveScenario(data.scenario)
    } catch (err: any) {
      console.error('Error stepping simulation:', err)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const resetSimulation = useCallback(async () => {
    setIsLoading(true)
    try {
      const st = await api.resetSimulation()
      setSimulationStatus(st)
      setIsSimulationActive(false)
      setSimulationState('IDLE')
      setSimulationData(null)
      setActiveScenario(null)
      setCurrentStep(6)
      window.dispatchEvent(new CustomEvent('oceancast:refresh'))
    } catch (err: any) {
      console.error('Error resetting simulation:', err)
    } finally {
      setIsLoading(false)
    }
  }, [])

  return (
    <SimulationContext.Provider
      value={{
        isSimulationActive,
        simulationState,
        simulationStatus,
        activeScenario,
        simulationData,
        scenarios,
        currentStep,
        totalSteps,
        isModalOpen,
        isLoading,
        error,
        openModal,
        closeModal,
        startSimulation,
        pauseSimulation,
        resumeSimulation,
        stepSimulation,
        resetSimulation,
        reset: resetSimulation,
      }}
    >
      {children}
    </SimulationContext.Provider>
  )
}

export function useSimulation(): SimulationContextValue {
  const context = useContext(SimulationContext)
  if (!context) {
    return defaultSimulationContextValue
  }
  return context
}

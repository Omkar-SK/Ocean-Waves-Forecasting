import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Layout } from './components/layout/Layout'
import { OverviewPage } from './pages/OverviewPage'
import { LiveForecastPage } from './pages/LiveForecastPage'
import { CoastalRiskMapPage } from './pages/CoastalRiskMapPage'
import { AlertsPage } from './pages/AlertsPage'
import { CoastalNodesPage } from './pages/CoastalNodesPage'
import { ForecastAccuracyDriftPage } from './pages/ForecastAccuracyDriftPage'
import { HistoricalAnalysisPage } from './pages/HistoricalAnalysisPage'
import { ModelComparisonPage } from './pages/ModelComparisonPage'
import { AboutPage } from './pages/AboutPage'

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<OverviewPage />} />
            <Route path="live" element={<LiveForecastPage />} />
            <Route path="map" element={<CoastalRiskMapPage />} />
            <Route path="alerts" element={<AlertsPage />} />
            <Route path="locations" element={<CoastalNodesPage />} />
            <Route path="monitoring" element={<ForecastAccuracyDriftPage />} />
            <Route path="historical" element={<HistoricalAnalysisPage />} />
            <Route path="comparison" element={<ModelComparisonPage />} />
            <Route path="about" element={<AboutPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  )
}


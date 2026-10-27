import React, { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'
import { Button } from './ui/Button'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error caught by ErrorBoundary:', error, errorInfo)
  }

  public handleReload = () => {
    window.location.reload()
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#07111F] text-[#F1F5F9] flex items-center justify-center p-6">
          <div className="max-w-md w-full p-6 rounded-2xl bg-[#101D2D] border border-[#EF4444]/40 shadow-xl text-center space-y-4">
            <div className="w-12 h-12 mx-auto rounded-full bg-[#EF4444]/10 border border-[#EF4444]/30 flex items-center justify-center">
              <AlertTriangle className="h-6 w-6 text-[#EF4444]" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-[#F1F5F9] font-['Outfit',sans-serif]">
                Dashboard Rendering Error
              </h2>
              <p className="text-xs text-[#94A3B8] mt-1">
                {this.state.error?.message || 'An unexpected rendering error occurred'}
              </p>
            </div>
            <div className="pt-2">
              <Button variant="primary" size="sm" onClick={this.handleReload} className="gap-2">
                <RefreshCw className="h-4 w-4" />
                <span>Reload Dashboard</span>
              </Button>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

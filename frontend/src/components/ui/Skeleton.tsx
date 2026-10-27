import React from 'react'
import { cn } from '../../lib/utils'

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {}

export function Skeleton({ className, ...props }: SkeletonProps) {
  return (
    <div
      className={cn('animate-pulse rounded-md bg-[#16263A] border border-[#1E3A5F]/40', className)}
      {...props}
    />
  )
}

import React, { useEffect, useRef } from 'react'

export function AIAvatar({ isSpeeching, className = '' }: { isSpeeching: boolean; className?: string }) {
  const videoRef = useRef<HTMLVideoElement | null>(null)

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    if (isSpeeching) {
      v.play().catch(() => {})
    } else {
      v.pause()
      try {
        v.currentTime = 0
      } catch {}
    }
  }, [isSpeeching])

  return (
    <div className={`relative w-full aspect-square rounded-xl overflow-hidden bg-gray-100 ring-1 ring-cizgi ${className}`}>
      <video ref={videoRef} className="w-full h-full object-center object-[50%_35%]" loop muted playsInline>
        <source src="/ai-avatar.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>
      {isSpeeching && (
        <div className="absolute bottom-2 left-2 bg-black/70 backdrop-blur-sm text-white text-xs px-2 py-1 rounded">
          AI Speaking
        </div>
      )}
    </div>
  )
}



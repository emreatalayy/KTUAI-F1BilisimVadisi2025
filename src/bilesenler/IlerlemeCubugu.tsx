export function IlerlemeCubugu({ oran, etiketi }: { oran: number; etiketi?: string }) {
  const yuzde = Math.max(0, Math.min(100, oran))
  return (
    <div>
      {etiketi && <div className="mb-2 text-sm text-metinIkincil">{etiketi}</div>}
      <div className="w-full h-3 rounded-full bg-arka border border-cizgi overflow-hidden" role="progressbar" aria-valuenow={yuzde} aria-valuemin={0} aria-valuemax={100} aria-label={etiketi}>
        <div className="h-full bg-metin" style={{ width: `${yuzde}%` }} />
      </div>
    </div>
  )
}



type Nokta = { x: number; y: number }

export function CizgiGrafik({ veri, genislik = 560, yukseklik = 200, ariaLabel, renk = '#2563EB' }: { veri: Nokta[]; genislik?: number; yukseklik?: number; ariaLabel?: string; renk?: string }) {
  if (veri.length === 0) return <div className="text-sm text-metinIkincil">Veri yok</div>
  const minX = Math.min(...veri.map((d) => d.x))
  const maxX = Math.max(...veri.map((d) => d.x))
  const minY = Math.min(...veri.map((d) => d.y))
  const maxY = Math.max(...veri.map((d) => d.y))
  const p = 28
  const icW = genislik - p * 2
  const icH = yukseklik - p * 2
  const sx = (x: number) => p + ((x - minX) / (maxX - minX || 1)) * icW
  const sy = (y: number) => p + icH - ((y - minY) / (maxY - minY || 1)) * icH
  // Basit bir smoothing (quadratic bezier) uygulayalım
  const pts = veri.map(v => ({ X: sx(v.x), Y: sy(v.y) }))
  function buildPath() {
    if (pts.length === 0) return ''
  const firstPt = pts[0]!
  if (pts.length === 1) return `M ${firstPt.X} ${firstPt.Y}`
  let d = `M ${firstPt.X} ${firstPt.Y}`
    for (let i = 1; i < pts.length; i++) {
      const p0 = pts[i - 1]!
      const p1 = pts[i]!
      const cx = (p0.X + p1.X) / 2
      const cy = (p0.Y + p1.Y) / 2
      d += ` Q ${p0.X} ${p0.Y}, ${cx} ${cy}`
    }
    const last = pts[pts.length - 1]!
    d += ` L ${last.X} ${last.Y}`
    return d
  }
  const lineD = buildPath()
  const first = pts[0]
  const last = pts[pts.length - 1]
  const areaD = lineD && first && last ? lineD + ` L ${last.X} ${p + icH} L ${first.X} ${p + icH} Z` : ''
  // Grid yatay çizgiler
  const tickCount = 4
  const ticks: number[] = []
  for (let i = 0; i <= tickCount; i++) ticks.push(minY + (i / tickCount) * (maxY - minY))

  return (
    <figure className="w-full overflow-hidden">
      <svg width={genislik} height={yukseklik} role="img" aria-label={ariaLabel} className="select-none">
        <defs>
          <linearGradient id="gradFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={renk} stopOpacity={0.25} />
            <stop offset="100%" stopColor={renk} stopOpacity={0} />
          </linearGradient>
        </defs>
        <rect x={0} y={0} width={genislik} height={yukseklik} fill="url(#gradBg)" />
        {ticks.map(t => {
          const y = sy(t)
          return <line key={t} x1={p} y1={y} x2={genislik - p} y2={y} stroke="#E5E7EB" strokeWidth={1} strokeDasharray="4 6" />
        })}
        <line x1={p} y1={p} x2={p} y2={yukseklik - p} stroke="#E5E7EB" strokeWidth={1.2} />
        <line x1={p} y1={yukseklik - p} x2={genislik - p} y2={yukseklik - p} stroke="#E5E7EB" strokeWidth={1.2} />
        <path d={areaD} fill="url(#gradFill)" stroke="none" />
        <path d={lineD} fill="none" stroke={renk} strokeWidth={2.2} />
        {pts.map((pt,i) => (
          <circle key={i} cx={pt.X} cy={pt.Y} r={4} fill="#fff" stroke={renk} strokeWidth={1.5} />
        ))}
      </svg>
    </figure>
  )
}



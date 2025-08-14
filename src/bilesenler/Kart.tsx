import { ReactNode } from 'react'

export function Kart({ baslik, ek, children, className = '' }: { baslik?: string; ek?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={['bg-kart rounded-kart shadow-hafif border border-cizgi', className].join(' ')}>
      {(baslik || ek) && (
        <div className="flex items-center justify-between px-5 pt-5 pb-3 border-b border-cizgi">
          {baslik && <h2 className="text-heading font-semibold">{baslik}</h2>}
          {ek}
        </div>
      )}
      <div className="p-5">{children}</div>
    </section>
  )
}



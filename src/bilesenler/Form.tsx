import { TextareaHTMLAttributes, InputHTMLAttributes, ReactNode } from 'react'

export function Etiket({ children, htmlFor }: { children: ReactNode; htmlFor?: string }) {
  return (
    <label htmlFor={htmlFor} className="block text-label text-metinIkincil mb-2">
      {children}
    </label>
  )
}

type GirdiOzellikleri = InputHTMLAttributes<HTMLInputElement> & { hata?: string; aciklama?: string }
export function Girdi({ hata, aciklama, className = '', ...kalan }: GirdiOzellikleri) {
  return (
    <div>
      <input
        className={
          'h-11 w-full rounded-input border border-cizgi px-3 text-sm bg-white placeholder:text-metinIkincil focus-visible:outline focus-visible:outline-1 focus-visible:outline-metin ' +
          className
        }
        {...kalan}
      />
      {aciklama && <p className="mt-1 text-xs text-metinIkincil">{aciklama}</p>}
      {hata && <p className="mt-1 text-xs text-hata">{hata}</p>}
    </div>
  )
}

type MetinAlaniOzellikleri = TextareaHTMLAttributes<HTMLTextAreaElement> & { hata?: string; aciklama?: string }
export function MetinAlani({ hata, aciklama, className = '', rows = 6, ...kalan }: MetinAlaniOzellikleri) {
  return (
    <div>
      <textarea
        rows={rows}
        className={
          'w-full rounded-input border border-cizgi p-3 text-sm bg-white placeholder:text-metinIkincil focus-visible:outline focus-visible:outline-1 focus-visible:outline-metin ' +
          className
        }
        {...kalan}
      />
      {aciklama && <p className="mt-1 text-xs text-metinIkincil">{aciklama}</p>}
      {hata && <p className="mt-1 text-xs text-hata">{hata}</p>}
    </div>
  )
}

export function RadioGrup({ secenekler, deger, setDeger, name }: { secenekler: { etiket: string; deger: string }[]; deger: string; setDeger: (v: string) => void; name: string }) {
  return (
    <div className="flex flex-wrap gap-3">
      {secenekler.map((s) => (
        <label key={s.deger} className={'flex items-center gap-2 rounded-full border px-4 h-11 cursor-pointer ' + (deger === s.deger ? 'border-metin' : 'border-cizgi')}>
          <input
            type="radio"
            className="sr-only"
            name={name}
            checked={deger === s.deger}
            onChange={() => setDeger(s.deger)}
            aria-checked={deger === s.deger}
          />
          <span className={'h-2.5 w-2.5 rounded-full border ' + (deger === s.deger ? 'bg-metin border-metin' : 'border-cizgi')}></span>
          <span className="text-sm">{s.etiket}</span>
        </label>
      ))}
    </div>
  )
}

export function Kaydirici({ min = 0, max = 100, deger, setDeger, ariaLabel }: { min?: number; max?: number; deger: number; setDeger: (v: number) => void; ariaLabel: string }) {
  return (
    <div className="flex items-center gap-3">
      <input
        type="range"
        min={min}
        max={max}
        value={deger}
        aria-label={ariaLabel}
        onChange={(e) => setDeger(Number(e.target.value))}
        className="w-full accent-metin"
      />
      <span className="text-sm w-10 text-right">{deger}</span>
    </div>
  )
}



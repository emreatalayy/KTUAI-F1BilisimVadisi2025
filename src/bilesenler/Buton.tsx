import { ButtonHTMLAttributes, ReactNode } from 'react'

type Ortak = {
  children?: ReactNode
  solIkon?: ReactNode
  sagIkon?: ReactNode
}

type ButonOzellikleri = ButtonHTMLAttributes<HTMLButtonElement> &
  Ortak & {
    tur?: 'birincil' | 'ikincil' | 'metin'
    genis?: boolean
  }

export function Buton({ children, solIkon, sagIkon, tur = 'birincil', genis, className = '', ...kalan }: ButonOzellikleri) {
  const taban = 'inline-flex items-center justify-center gap-2 h-11 px-4 text-sm font-medium rounded-buton transition focus-visible:outline-1 focus-visible:outline-metin'
  const stil =
    tur === 'birincil'
      ? 'bg-metin text-white hover:opacity-90 disabled:opacity-50'
      : tur === 'ikincil'
      ? 'bg-white text-metin border border-metin hover:bg-[#F0F0F1] disabled:opacity-50'
      : 'bg-transparent text-metin hover:bg-[#F0F0F1]'

  return (
    <button className={[taban, stil, genis ? 'w-full' : '', className].join(' ')} {...kalan}>
      {solIkon && <span aria-hidden>{solIkon}</span>}
      {children}
      {sagIkon && <span aria-hidden>{sagIkon}</span>}
    </button>
  )
}



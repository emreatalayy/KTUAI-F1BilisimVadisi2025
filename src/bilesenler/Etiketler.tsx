import { X } from 'lucide-react'

export function Etiket({ metin, renk = 'arka' }: { metin: string; renk?: 'arka' | 'basari' | 'uyarı' | 'hata' }) {
  const sinif =
    renk === 'basari'
      ? 'bg-[#E8F7ED] text-[#166534]'
      : renk === 'uyarı'
      ? 'bg-[#FEF3C7] text-[#92400E]'
      : renk === 'hata'
      ? 'bg-[#FEE2E2] text-[#991B1B]'
      : 'bg-arka text-metin'
  return <span className={`inline-flex items-center h-6 px-2 rounded-full text-xs ${sinif}`}>{metin}</span>
}

export function EtiketSililebilir({ metin, onSil }: { metin: string; onSil: () => void }) {
  return (
    <span className="inline-flex items-center gap-1 h-7 px-3 rounded-full text-xs bg-arka">
      {metin}
      <button aria-label={`${metin} sil`} className="p-0.5 rounded-full hover:bg-[#E5E7EB]" onClick={onSil}>
        <X size={14} />
      </button>
    </span>
  )
}



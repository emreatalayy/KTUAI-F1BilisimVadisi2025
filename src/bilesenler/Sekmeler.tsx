type Sekme = { anahtar: string; baslik: string }

export function Sekmeler({ sekmeler, aktif, sec }: { sekmeler: Sekme[]; aktif: string; sec: (k: string) => void }) {
  return (
    <div className="border-b border-cizgi flex gap-6">
      {sekmeler.map((s) => (
        <button
          key={s.anahtar}
          onClick={() => sec(s.anahtar)}
          className={
            'py-3 -mb-px text-sm font-medium transition' +
            (aktif === s.anahtar ? ' border-b-2 border-metin text-metin font-semibold' : ' text-metinIkincil hover:text-metin')
          }
          aria-current={aktif === s.anahtar ? 'page' : undefined}
        >
          {s.baslik}
        </button>
      ))}
    </div>
  )
}



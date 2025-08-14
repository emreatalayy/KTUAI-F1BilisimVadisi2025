import { useEffect, useState } from 'react'

export function useToast() {
  const [mesaj, setMesaj] = useState<string | null>(null)
  function goster(m: string) {
    setMesaj(m)
    setTimeout(() => setMesaj(null), 3000)
  }
  const bilesen = <Toast mesaj={mesaj} />
  return { goster, bileşen: bilesen }
}

function Toast({ mesaj }: { mesaj: string | null }) {
  const [goster, setGoster] = useState(false)
  useEffect(() => {
    setGoster(!!mesaj)
  }, [mesaj])
  if (!mesaj || !goster) return null
  return (
    <div className="fixed bottom-6 right-6 z-50">
      <div role="status" className="bg-metin text-white rounded-toast shadow-hafif px-4 py-3 text-sm">
        {mesaj}
      </div>
    </div>
  )
}



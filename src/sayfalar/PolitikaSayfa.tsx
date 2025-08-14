import { useState } from 'react'
import { Kart } from '../bilesenler/Kart'
import { EtiketSililebilir } from '../bilesenler/Etiketler'
import { MetinAlani } from '../bilesenler/Form'

export function PolitikaSayfa() {
  const [etiketler, setEtiketler] = useState<string[]>(['Kişisel veri', 'Finansal bilgi', 'Sağlık verisi'])
  return (
    <div>
      <h1 className="text-heading font-semibold mb-6">Hassas ve Kişisel Veri Koruma</h1>

      <div className="grid grid-cols-1 lg2:grid-cols-2 gap-6">
        <div className="space-y-6">
          <Kart baslik="Politika Aykırı Durum Mesajı">
            <MetinAlani aria-label="Politika aykırı mesaj" placeholder="Aykırı durumda gösterilecek mesaj..." rows={8} />
          </Kart>
          <Kart baslik="Politikalar">
            <div className="flex flex-wrap gap-2">
              {etiketler.map((e) => (
                <EtiketSililebilir key={e} metin={e} onSil={() => setEtiketler((d) => d.filter((x) => x !== e))} />
              ))}
              <button className="text-sm text-metin underline underline-offset-4">+ Yeni Ekle</button>
            </div>
          </Kart>
        </div>
      </div>
    </div>
  )
}



import { useState } from 'react'
import { Layout } from '../yapi/Layout'
import { AnasayfaSayfa } from '../sayfalar/AnasayfaSayfa'
import { AgentSayfa } from '../sayfalar/AgentSayfa'
import { SohbetGecmisiSayfa } from '../sayfalar/SohbetGecmisiSayfa'
import { PolitikaSayfa } from '../sayfalar/PolitikaSayfa'
import { RaporlarSayfa } from '../sayfalar/RaporlarSayfa'

export type Rota = 'anasayfa' | 'agent' | 'sohbet' | 'politika' | 'raporlar'

export function Uygulama() {
  const [rota, setRota] = useState<Rota>('anasayfa')

  return (
    <Layout rota={rota} setRota={setRota}>
      {rota === 'anasayfa' && <AnasayfaSayfa />}
      {rota === 'agent' && <AgentSayfa />}
      {rota === 'sohbet' && <SohbetGecmisiSayfa />}
      {rota === 'politika' && <PolitikaSayfa />}
      {rota === 'raporlar' && <RaporlarSayfa />}
    </Layout>
  )
}



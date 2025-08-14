import { ReactNode, useState } from 'react'
import { Home, Bot, MessageSquare, Shield, BarChart3, Settings, LogOut, Menu, Moon, Sun } from 'lucide-react'
import { Rota } from '../uygulama/Uygulama'

type Ozellikler = {
  rota: Rota
  setRota: (r: Rota) => void
  children: ReactNode
}

export function Layout({ rota, setRota, children }: Ozellikler) {
  const [karanlik, setKaranlik] = useState(false)
  const [mobilAcik, setMobilAcik] = useState(false)
  return (
    <div className={(karanlik ? 'dark ' : '') + 'h-full bg-arka text-metin'}>
      <div className="mx-auto max-w-[1440px] h-full flex">
        <KenarCubuğu rota={rota} setRota={setRota} mobilAcik={mobilAcik} setMobilAcik={setMobilAcik} />
        <main className="flex-1 p-6">
          <div className="mx-auto max-w-1200">
            <div className="md:hidden flex items-center justify-between mb-4">
              <button aria-label="Menüyü aç" className="p-2 rounded-full border border-cizgi" onClick={() => setMobilAcik(true)}>
                <Menu />
              </button>
              <button aria-label="Tema" className="p-2 rounded-full border border-cizgi" onClick={() => setKaranlik((v) => !v)}>
                {karanlik ? <Sun /> : <Moon />}
              </button>
            </div>
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}

function KenarCubuğu({ rota, setRota, mobilAcik, setMobilAcik }: { rota: Rota; setRota: (r: Rota) => void; mobilAcik: boolean; setMobilAcik: (v: boolean) => void }) {
  return (
    <>
      {/* Masaüstü */}
      <aside className="hidden md:flex flex-col justify-between w-[240px] p-6 border-r border-cizgi bg-kart">
        <div>
          <div className="h-8 w-32 mb-8 bg-arka rounded-md" aria-label="Logo alanı"></div>
          <nav className="flex flex-col gap-1" aria-label="Ana menü">
            <MenuOg ad="Anasayfa" aktif={rota === 'anasayfa'} ikon={<Home size={20} />} onClick={() => setRota('anasayfa')} />
            <MenuOg ad="Agent" aktif={rota === 'agent'} ikon={<Bot size={20} />} onClick={() => setRota('agent')} />
            <MenuOg ad="Sohbet Geçmişi" aktif={rota === 'sohbet'} ikon={<MessageSquare size={20} />} onClick={() => setRota('sohbet')} />
            <MenuOg ad="Politika" aktif={rota === 'politika'} ikon={<Shield size={20} />} onClick={() => setRota('politika')} />
            <MenuOg ad="Raporlar" aktif={rota === 'raporlar'} ikon={<BarChart3 size={20} />} onClick={() => setRota('raporlar')} />
          </nav>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 bg-arka rounded-full" aria-hidden="true"></div>
            <span className="text-sm text-metinIkincil">Kullanıcı</span>
          </div>
          <div className="flex items-center gap-3 text-metinIkincil">
            <button aria-label="Ayarlar" className="p-2 rounded-full hover:bg-[#F0F0F1] focus-visible:outline-2 focus-visible:outline-metin"><Settings size={20} /></button>
            <button aria-label="Çıkış" className="p-2 rounded-full hover:bg-[#F0F0F1] focus-visible:outline-2 focus-visible:outline-metin"><LogOut size={20} /></button>
          </div>
        </div>
      </aside>
      {/* Mobil kapalı/çekmece */}
      {mobilAcik && (
        <div className="fixed inset-0 z-50 md:hidden" role="dialog" aria-modal="true">
          <div className="absolute inset-0 bg-black/30" onClick={() => setMobilAcik(false)}></div>
          <aside className="absolute left-0 top-0 h-full w-[240px] p-6 border-r border-cizgi bg-kart flex flex-col justify-between">
            <div>
              <div className="h-8 w-32 mb-8 bg-arka rounded-md" aria-label="Logo alanı"></div>
              <nav className="flex flex-col gap-1" aria-label="Ana menü">
                <MenuOg ad="Anasayfa" aktif={rota === 'anasayfa'} ikon={<Home size={20} />} onClick={() => { setRota('anasayfa'); setMobilAcik(false) }} />
                <MenuOg ad="Agent" aktif={rota === 'agent'} ikon={<Bot size={20} />} onClick={() => { setRota('agent'); setMobilAcik(false) }} />
                <MenuOg ad="Sohbet Geçmişi" aktif={rota === 'sohbet'} ikon={<MessageSquare size={20} />} onClick={() => { setRota('sohbet'); setMobilAcik(false) }} />
                <MenuOg ad="Politika" aktif={rota === 'politika'} ikon={<Shield size={20} />} onClick={() => { setRota('politika'); setMobilAcik(false) }} />
                <MenuOg ad="Raporlar" aktif={rota === 'raporlar'} ikon={<BarChart3 size={20} />} onClick={() => { setRota('raporlar'); setMobilAcik(false) }} />
              </nav>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-8 w-8 bg-arka rounded-full" aria-hidden="true"></div>
                <span className="text-sm text-metinIkincil">Kullanıcı</span>
              </div>
              <div className="flex items-center gap-3 text-metinIkincil">
                <button aria-label="Ayarlar" className="p-2 rounded-full hover:bg-[#F0F0F1] focus-visible:outline-2 focus-visible:outline-metin"><Settings size={20} /></button>
                <button aria-label="Çıkış" className="p-2 rounded-full hover:bg-[#F0F0F1] focus-visible:outline-2 focus-visible:outline-metin"><LogOut size={20} /></button>
              </div>
            </div>
          </aside>
        </div>
      )}
    </>
  )
}

function MenuOg({ ad, aktif, ikon, onClick }: { ad: string; aktif: boolean; ikon: ReactNode; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={
        'flex items-center gap-3 px-3 h-11 rounded-[12px] text-left transition ' +
        (aktif
          ? 'bg-metin text-white'
          : 'text-metinIkincil hover:bg-[#F0F0F1]')
      }
      aria-current={aktif ? 'page' : undefined}
    >
      <span className={aktif ? 'text-white' : 'text-metinIkincil'} aria-hidden="true">{ikon}</span>
      <span className={aktif ? 'font-medium text-white' : 'font-medium'}>{ad}</span>
    </button>
  )
}



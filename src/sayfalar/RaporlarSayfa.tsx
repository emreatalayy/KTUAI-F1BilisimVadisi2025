import { Kart } from '../bilesenler/Kart'
import { CizgiGrafik } from '../bilesenler/CizgiGrafik'

export function RaporlarSayfa() {
  const veri = Array.from({ length: 10 }).map((_, i) => ({ x: i, y: 10 + Math.round(Math.sin(i / 2) * 6 + i) }))
  return (
    <div>
      <h1 className="text-heading font-semibold mb-6">Raporlar</h1>
      <div className="grid grid-cols-1 lg2:grid-cols-2 gap-6">
        <Kart baslik="Genel İstatistikler">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <KPI baslik="Toplam Sohbet" deger="1.245" />
            <KPI baslik="Aktif Kullanıcı" deger="312" />
            <KPI baslik="Ortalama Süre" deger="6.2dk" />
            <KPI baslik="Başarı Oranı" deger="84%" />
          </div>
        </Kart>
        <Kart baslik="Trend">
          <CizgiGrafik veri={veri} ariaLabel="Günlük sohbet sayısı trendi" />
          <div className="mt-4 text-sm">
            <table>
              <thead className="text-metinIkincil">
                <tr><th className="text-left pr-4">Gün</th><th className="text-left">Değer</th></tr>
              </thead>
              <tbody>
                {veri.map((v) => (
                  <tr key={v.x}><td className="pr-4">{v.x}</td><td>{v.y}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </Kart>
      </div>
    </div>
  )
}

function KPI({ baslik, deger }: { baslik: string; deger: string }) {
  return (
    <div className="border border-cizgi rounded-input px-4 py-3">
      <div className="text-sm text-metinIkincil">{baslik}</div>
      <div className="text-2xl font-semibold mt-1">{deger}</div>
    </div>
  )
}



import { Kart } from '../bilesenler/Kart'
import { Etiket } from '../bilesenler/Etiketler'
import { Tablo } from '../bilesenler/Tablo'

export function AnasayfaSayfa() {
  const sonSohbetler = Array.from({ length: 5 }).map((_, i) => [
    new Date(Date.now() - i * 3600_000).toLocaleString('tr-TR'),
    `kullanici_${i + 1}`,
    'Web',
    'Kısa özet metni',
    `${(5 + i)}dk`,
  ])
  return (
    <div>
      <h1 className="text-heading font-semibold mb-6">Hoş geldiniz</h1>

      <div className="grid grid-cols-1 lg2:grid-cols-2 gap-6">
        <Kart baslik="Hızlı Başlangıç">
          <ol className="list-decimal pl-5 text-sm text-metinIkincil space-y-2">
            <li>Agent ayarlarını yapılandırın</li>
            <li>Widget'ı sitenize gömün</li>
            <li>Kalite ve politika kurallarını belirleyin</li>
          </ol>
        </Kart>
        <Kart baslik="Sistem Durumu">
          <div className="flex flex-wrap gap-4">
            <DurumKart baslik="API" durum="Çevrimiçi" renk="basari" />
            <DurumKart baslik="İşleyiciler" durum="Uyarı" renk="uyarı" />
            <DurumKart baslik="Veritabanı" durum="Hata" renk="hata" />
          </div>
        </Kart>

        <Kart baslik="Son Sohbetler">
          <Tablo basliklar={["Tarih-Saat", "Kullanıcı", "Kanal", "Özet", "Süre"]} satirlar={sonSohbetler.map((s) => s.map((v, i) => i === 3 ? <span className="truncate inline-block max-w-[280px]">{v}</span> : <span>{v}</span>))} />
        </Kart>
        <Kart baslik="Duyurular">
          <div className="space-y-3 text-sm text-metinIkincil">
            <p>Yeni sürüm ile performans iyileştirmeleri yapıldı.</p>
            <p>Gizlilik politikası güncellendi.</p>
          </div>
        </Kart>
      </div>
    </div>
  )
}

function DurumKart({ baslik, durum, renk }: { baslik: string; durum: string; renk: 'basari' | 'uyarı' | 'hata' }) {
  return (
    <div className="border border-cizgi rounded-input px-4 py-3 min-w-[160px]">
      <div className="text-sm text-metinIkincil">{baslik}</div>
      <div className="mt-1 font-medium flex items-center gap-2">
        {durum}
        <Etiket metin={renk === 'basari' ? 'başarılı' : renk === 'uyarı' ? 'uyarı' : 'hata'} renk={renk} />
      </div>
    </div>
  )
}



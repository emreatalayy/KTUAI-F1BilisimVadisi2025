import { useEffect, useMemo, useState } from 'react'
import { Kart } from '../bilesenler/Kart'
import { Tablo } from '../bilesenler/Tablo'
import { Etiket as FormEtiket, Girdi } from '../bilesenler/Form'
import { Buton } from '../bilesenler/Buton'
import { sohbetGecmisi, type GecmisSatiri, chatDosyasiIcerik, sohbetGecmisiSil, tumSohbetGecmisiTemizle } from '../agent/api'

type Kayit = { tarih: string; kullanici: string; kanal: string; ozet: string; sure: string }

export function SohbetGecmisiSayfa() {
  const [arama, setArama] = useState('')
  const [tarihBaslangic, setTarihBaslangic] = useState('')
  const [tarihBitis, setTarihBitis] = useState('')
  const [sayfa, setSayfa] = useState(1)
  const [seciliKayitlar, setSeciliKayitlar] = useState<Set<string>>(new Set())
  const [kanalFiltre, setKanalFiltre] = useState('')
  const sayfaBoyut = 10

  const apiUrl = (import.meta as any).env?.VITE_CHAT_API_URL as string | undefined
  const [uzak, setUzak] = useState<GecmisSatiri[] | null>(null)
  const [dosya, setDosya] = useState<{ filename: string; content: string } | null>(null)
  const [yukleniyor, setYukleniyor] = useState(false)
  
  const gecmisYenile = async () => {
    if (!apiUrl) return
    try {
      setYukleniyor(true)
      const veriler = await sohbetGecmisi(apiUrl)
      setUzak(veriler)
    } catch (error) {
      console.error('Sohbet geçmişi yüklenemedi:', error)
      setUzak([])
    } finally {
      setYukleniyor(false)
    }
  }

  useEffect(() => {
    gecmisYenile()
  }, [apiUrl])

  const kayitlar = useMemo(() => {
    if (!uzak) return []
    return uzak.map((r) => ({
      ...r,
      tarih: new Date(r.timestamp).toLocaleString('tr-TR'),
    }))
  }, [uzak])

  const filtreli = useMemo(() => {
    return kayitlar.filter((k) => {
      const a = arama.trim().toLowerCase()
      const eslesme = !a || [k.user, k.channel, k.summary].some((v) => v.toLowerCase().includes(a))
      
      const bas = tarihBaslangic ? new Date(tarihBaslangic).getTime() : -Infinity
      const bit = tarihBitis ? new Date(tarihBitis).getTime() : Infinity
      const t = new Date(k.timestamp).getTime()
      const tarihEslesme = t >= bas && t <= bit
      
      const kanalEslesme = !kanalFiltre || k.channel === kanalFiltre
      
      return eslesme && tarihEslesme && kanalEslesme
    })
  }, [kayitlar, arama, tarihBaslangic, tarihBitis, kanalFiltre])

  const toplamSayfa = Math.max(1, Math.ceil(filtreli.length / sayfaBoyut))
  const gosterilen = filtreli.slice((sayfa - 1) * sayfaBoyut, sayfa * sayfaBoyut)

  const tumunuSec = () => {
    setSeciliKayitlar(new Set(gosterilen.map(k => k.id)))
  }

  const hicbiriniSecme = () => {
    setSeciliKayitlar(new Set())
  }

  const seciliKayitlariSil = async () => {
    if (seciliKayitlar.size === 0) return
    if (!confirm(`${seciliKayitlar.size} kayıt silinecek. Emin misiniz?`)) return
    
    try {
      setYukleniyor(true)
      for (const id of seciliKayitlar) {
        await sohbetGecmisiSil(apiUrl!, id)
      }
      setSeciliKayitlar(new Set())
      await gecmisYenile()
    } catch (error) {
      console.error('Silme hatası:', error)
      alert('Silme işlemi sırasında hata oluştu')
    } finally {
      setYukleniyor(false)
    }
  }

  const tumGecmisiTemizle = async () => {
    if (!confirm('TÜM sohbet geçmişi silinecek. Bu işlem geri alınamaz. Emin misiniz?')) return
    
    try {
      setYukleniyor(true)
      await tumSohbetGecmisiTemizle(apiUrl)
      setSeciliKayitlar(new Set())
      await gecmisYenile()
    } catch (error) {
      console.error('Temizleme hatası:', error)
      alert('Temizleme işlemi sırasında hata oluştu')
    } finally {
      setYukleniyor(false)
    }
  }

  const benzersizKanallar = useMemo(() => {
    const kanallar = Array.from(new Set(kayitlar.map(k => k.channel)))
    return kanallar.sort()
  }, [kayitlar])

  return (
    <div>
      <h1 className="text-heading font-semibold mb-6">Sohbet Geçmişi</h1>
      
      {/* Filtreler */}
      <Kart className="mb-4">
        <div className="flex flex-wrap items-end gap-4 mb-4">
          <div className="min-w-[240px]">
            <FormEtiket>Arama</FormEtiket>
            <Girdi 
              aria-label="Sohbet arama" 
              placeholder="Kullanıcı, kanal veya özet ara..." 
              value={arama} 
              onChange={(e) => setArama(e.target.value)} 
            />
          </div>
          <div>
            <FormEtiket>Kanal</FormEtiket>
            <select 
              value={kanalFiltre} 
              onChange={(e) => setKanalFiltre(e.target.value)}
              className="px-3 py-2 border border-kenar rounded-md bg-beyaz text-metinBirincil focus:outline-none focus:border-vurgu"
            >
              <option value="">Tüm Kanallar</option>
              {benzersizKanallar.map(kanal => (
                <option key={kanal} value={kanal}>{kanal}</option>
              ))}
            </select>
          </div>
          <div>
            <FormEtiket>Başlangıç</FormEtiket>
            <Girdi 
              aria-label="Başlangıç tarihi" 
              type="date" 
              value={tarihBaslangic} 
              onChange={(e) => setTarihBaslangic(e.target.value)} 
            />
          </div>
          <div>
            <FormEtiket>Bitiş</FormEtiket>
            <Girdi 
              aria-label="Bitiş tarihi" 
              type="date" 
              value={tarihBitis} 
              onChange={(e) => setTarihBitis(e.target.value)} 
            />
          </div>
        </div>
        
        {/* Toplu İşlemler */}
        <div className="flex flex-wrap items-center gap-2 border-t border-kenar pt-4">
          <div className="flex items-center gap-2">
            <Buton
              tur="ikincil"
              onClick={tumunuSec}
              disabled={gosterilen.length === 0}
            >
              Tümünü Seç
            </Buton>
            <Buton
              tur="ikincil"
              onClick={hicbiriniSecme}
              disabled={seciliKayitlar.size === 0}
            >
              Seçimi Temizle
            </Buton>
            <Buton
              tur="ikincil"
              onClick={seciliKayitlariSil}
              disabled={seciliKayitlar.size === 0 || yukleniyor}
            >
              Seçilenleri Sil ({seciliKayitlar.size})
            </Buton>
          </div>
          
          <div className="ml-auto flex items-center gap-2">
            <Buton
              tur="ikincil"
              onClick={tumGecmisiTemizle}
              disabled={yukleniyor}
            >
              Tümünü Temizle
            </Buton>
            <Buton
              tur="ikincil"
              onClick={gecmisYenile}
              disabled={yukleniyor}
            >
              {yukleniyor ? 'Yükleniyor...' : 'Yenile'}
            </Buton>
          </div>
        </div>
      </Kart>

      {/* Tablo */}
      <Kart>
        {yukleniyor ? (
          <div className="text-center py-8 text-metinIkincil">Yükleniyor...</div>
        ) : (
          <>
            <Tablo
              basliklar={["", "Tarih-Saat", "Kullanıcı", "Kanal", "Özet", "Süre", "İşlem"]}
              satirlar={gosterilen.map((k) => [
                <input
                  type="checkbox"
                  checked={seciliKayitlar.has(k.id)}
                  onChange={(e) => {
                    const yeniSet = new Set(seciliKayitlar)
                    if (e.target.checked) {
                      yeniSet.add(k.id)
                    } else {
                      yeniSet.delete(k.id)
                    }
                    setSeciliKayitlar(yeniSet)
                  }}
                  className="w-4 h-4"
                />,
                k.tarih,
                k.user,
                <span className={`px-2 py-1 rounded-full text-xs ${
                  k.channel === 'Web' ? 'bg-mavi-100 text-mavi-800' : 
                  k.channel === 'Arşiv' ? 'bg-gri-100 text-gri-800' : 
                  'bg-yesil-100 text-yesil-800'
                }`}>
                  {k.channel}
                </span>,
                <span className="truncate inline-block max-w-[420px]" title={k.summary}>
                  {k.summary}
                </span>,
                k.duration,
                <div className="flex gap-1">
                  <Buton 
                    tur="ikincil" 
                    onClick={async () => {
                      try {
                        const j = await chatDosyasiIcerik(apiUrl!, k.id)
                        setDosya(j)
                      } catch (error) {
                        console.error('Dosya açma hatası:', error)
                        alert('Dosya açılamadı')
                      }
                    }}
                  >
                    Aç
                  </Buton>
                  <Buton 
                    tur="ikincil" 
                    onClick={async () => {
                      if (!confirm('Bu kayıt silinecek. Emin misiniz?')) return
                      try {
                        await sohbetGecmisiSil(apiUrl!, k.id)
                        await gecmisYenile()
                      } catch (error) {
                        console.error('Silme hatası:', error)
                        alert('Silme işlemi başarısız')
                      }
                    }}
                  >
                    Sil
                  </Buton>
                </div>,
              ])}
            />
            
            {/* Sayfalama */}
            <div className="flex items-center justify-between mt-4 text-sm text-metinIkincil">
              <span>
                Toplam {filtreli.length} kayıt 
                {seciliKayitlar.size > 0 && (
                  <span className="ml-2 font-medium">
                    • {seciliKayitlar.size} seçili
                  </span>
                )}
              </span>
              <div className="flex items-center gap-2">
                <Buton 
                  tur="ikincil" 
                  onClick={() => setSayfa((s) => Math.max(1, s - 1))} 
                  disabled={sayfa === 1}
                >
                  Önceki
                </Buton>
                <span>Sayfa {sayfa}/{toplamSayfa}</span>
                <Buton 
                  tur="ikincil" 
                  onClick={() => setSayfa((s) => Math.min(toplamSayfa, s + 1))} 
                  disabled={sayfa === toplamSayfa}
                >
                  Sonraki
                </Buton>
              </div>
            </div>
          </>
        )}
      </Kart>

      {/* Dosya Görüntüleme Modal */}
      {dosya && (
        <Kart className="mt-6">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-base font-semibold">{dosya.filename}</h2>
            <Buton tur="ikincil" onClick={() => setDosya(null)}>Kapat</Buton>
          </div>
          <pre className="text-sm whitespace-pre-wrap break-words bg-arka p-3 rounded-md max-h-[360px] overflow-auto">
            {dosya.content}
          </pre>
        </Kart>
      )}
    </div>
  )
}



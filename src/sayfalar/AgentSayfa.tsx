import { useMemo, useState, useEffect } from 'react'
import { Buton } from '../bilesenler/Buton'
import { Sekmeler } from '../bilesenler/Sekmeler'
import { Kart } from '../bilesenler/Kart'
import { EtiketSililebilir, Etiket } from '../bilesenler/Etiketler'
import { IlerlemeCubugu } from '../bilesenler/IlerlemeCubugu'
import { Girdi, Etiket as FormEtiket, MetinAlani, RadioGrup, Kaydirici } from '../bilesenler/Form'
import { Clipboard } from 'lucide-react'
import { useToast } from '../bilesenler/Toast'
import { SohbetKutusu } from '../bilesenler/SohbetKutusu'
import { AIAvatar } from '../bilesenler/AIAvatar'
import { sohbetGonder } from '../agent/api'
// Kullanılmayan avatar bileşen importları kaldırıldı
import { CizgiGrafik } from '../bilesenler/CizgiGrafik'
import { Tablo } from '../bilesenler/Tablo'

type SekmeAnahtari = 'prompter' | 'ses' | 'entegrasyon' | 'kalite' | 'gelismis' | 'widget' | 'test'

export function AgentSayfa() {
  const [sekme, setSekme] = useState<SekmeAnahtari>('test')
  const { goster, bileşen } = useToast()

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-heading font-semibold">Agent</h1>
        <Buton onClick={() => goster('Test çalıştırıldı')}>
          Test
        </Buton>
      </div>

      <Sekmeler
        sekmeler={[
          { anahtar: 'prompter', baslik: 'Prompter' },
          { anahtar: 'ses', baslik: 'Ses' },
          { anahtar: 'entegrasyon', baslik: 'Entegrasyon' },
          { anahtar: 'kalite', baslik: 'Kalite' },
          { anahtar: 'gelismis', baslik: 'Gelişmiş' },
          { anahtar: 'widget', baslik: 'Widget' },
          { anahtar: 'test', baslik: 'Test' },
        ]}
        aktif={sekme}
        sec={(k) => setSekme(k as SekmeAnahtari)}
      />

      <div className="mt-6">
        {sekme === 'prompter' && <PrompterIcerik />}
        {sekme === 'widget' && <WidgetIcerik onKopyala={(t) => { navigator.clipboard.writeText(t); goster('Kod kopyalandı') }} />}
        {sekme === 'ses' && <SesIcerik />}
        {sekme === 'test' && <TestIcerik />}
  {sekme === 'entegrasyon' && <EntegrasyonIcerik />}
  {sekme === 'kalite' && <KaliteIcerik />}
  {sekme === 'gelismis' && <GelismisIcerik />}
      </div>
      {bileşen}
    </div>
  )
}

function PrompterIcerik() {
  const [etiketler, setEtiketler] = useState<string[]>(['Doğruluk', 'Kısa cevap', 'Kaynak ver'])
  return (
    <div className="grid grid-cols-1 lg2:grid-cols-2 gap-6">
      <div className="lg2:col-span-2">
        <IlerlemeCubugu oran={68} etiketi="Prompt kullanımı doluluk oranı" />
      </div>
      <Kart baslik="Görev Tanımı">
        <MetinAlani aria-label="Görev tanımı" placeholder="Agent görevi..." rows={8} />
      </Kart>
      <Kart baslik="Yanıt Verme Şartları">
        <MetinAlani aria-label="Yanıt verme şartları" placeholder="Şartlar..." rows={8} />
      </Kart>
      <Kart baslik="Genel Kısıtlamalar" ek={<Buton tur="ikincil">+ Ekle</Buton>}>
        <div className="flex flex-wrap gap-2">
          {etiketler.map((e, i) => (
            <EtiketSililebilir key={i} metin={e} onSil={() => setEtiketler((d) => d.filter((x) => x !== e))} />
          ))}
        </div>
      </Kart>
    </div>
  )
}

function WidgetIcerik({ onKopyala }: { onKopyala: (t: string) => void }) {
  const [varyant, setVaryant] = useState('compact')
  const [kartRadius, setKartRadius] = useState(24)
  const [butonRadius, setButonRadius] = useState(32)
  const kod = `<script src=\"https://cdn.ornek.com/widget.js\" data-variant=\"${varyant}\" data-kart-radius=\"${kartRadius}\" data-buton-radius=\"${butonRadius}\"></script>`

  return (
    <div className="grid grid-cols-1 lg2:grid-cols-2 gap-6">
      <Kart baslik="Gömülü Kod" ek={<Buton tur="ikincil" onClick={() => onKopyala(kod)}><span className="inline-flex items-center gap-2"><Clipboard size={18} /> Kopyala</span></Buton>}>
        <Girdi aria-label="Gömülü kod" value={kod} readOnly className="font-mono" />
      </Kart>
      <Kart baslik="Dış Görünüş">
        <div className="space-y-5">
          <div>
            <FormEtiket>Varyant</FormEtiket>
            <RadioGrup
              name="varyant"
              deger={varyant}
              setDeger={setVaryant}
              secenekler={[
                { etiket: 'compact', deger: 'compact' },
                { etiket: 'full', deger: 'full' },
                { etiket: 'expand', deger: 'expand' },
                { etiket: 'expand-chat', deger: 'expand-chat' },
              ]}
            />
          </div>
          <div>
            <FormEtiket>Kart radius</FormEtiket>
            <Kaydirici ariaLabel="Kart radius" min={8} max={40} deger={kartRadius} setDeger={setKartRadius} />
          </div>
          <div>
            <FormEtiket>Buton radius</FormEtiket>
            <Kaydirici ariaLabel="Buton radius" min={12} max={48} deger={butonRadius} setDeger={setButonRadius} />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <FormEtiket>Birincil renk</FormEtiket>
              <Girdi aria-label="Birincil renk" type="color" defaultValue="#111111" />
            </div>
            <div>
              <FormEtiket>Arka plan</FormEtiket>
              <Girdi aria-label="Arka plan" type="color" defaultValue="#F7F7F8" />
            </div>
          </div>
        </div>
      </Kart>
    </div>
  )
}

function TestIcerik() {
  const apiUrl = (import.meta as any).env?.VITE_CHAT_API_URL as string | undefined
  const ttsBase = apiUrl // aynı backend
  const sessionId = useMemo(() => `sess-${Date.now()}-${Math.random().toString(36).slice(2,7)}` , [])
  const [konusuyor, setKonusuyor] = useState(false)
  // SohbetKutusu içinde konuşma başladığında pencere olayı ile haber ver (basit entegrasyon)
  // Bu demo: speechSynthesis 'start','end' olaylarını dinleyemiyoruz burada; periodic check
  useEffect(() => {
    const id = setInterval(() => {
      const speaking = typeof window !== 'undefined' && 'speechSynthesis' in window && window.speechSynthesis.speaking
      setKonusuyor(speaking)
    }, 400)
    return () => clearInterval(id)
  }, [])
  return (
    <div className="grid grid-cols-1 lg2:grid-cols-3 gap-6">
      <Kart baslik="Agent Test Sohbeti" className="lg2:col-span-2">
        {apiUrl ? (
          <SohbetKutusu
            baslik="TürkTel Asistan"
            placeholder="Örn: 5550001, mevcut paketim?"
            baslangicMesaji="Merhaba, ben TürkTel asistanınız. İşleme başlamadan önce lütfen müşteri numaranızı paylaşır mısınız? (Örn: 5550001)"
            onGonder={async (m) => await sohbetGonder(apiUrl, sessionId, m)}
            ttsApiBase={ttsBase}
            onSpeakingChange={(v)=>setKonusuyor(v)}
          />
        ) : (
          <div className="text-sm text-metinIkincil">LLM API yapılandırılmadı. Lütfen `.env` içinde VITE_CHAT_API_URL ayarlayın ve backend'i başlatın.</div>
        )}
      </Kart>
      <div className="space-y-6">
        <Kart baslik="Avatar">
          <div className="p-4 flex items-center justify-center">
            <AIAvatar isSpeeching={konusuyor} className="w-48" />
          </div>
        </Kart>
        <Kart baslik="Notlar">
          <div className="p-4 text-xs leading-relaxed text-metinIkincil space-y-2">
            <p>Bu test ekranında kullanıcı asistan yanıtlarını dinleyip dinlememe seçimini yapabilir. Tarayıcı içi TTS veya sunucu tabanlı TTS arasında geçiş yapılabilir.</p>
            <p>Agent dinamik araç (tool) seçimi yapar; if/else akışıyla sabit scriptlenmemiştir. Konu değişimi komutları: <code>konu değiştir</code>, <code>yeni konu</code>.</p>
          </div>
        </Kart>
      </div>
    </div>
  )
}

function SesIcerik() {
  const [aktif, setAktif] = useState<string>('Tomris')
  // Kürşat -> Oğuz ve yerel dosyalar
  const assetler = import.meta.glob('../*.mp3', { eager: true, as: 'url' }) as Record<string, string>
  // Dosya isimlerinden eşle
  function bul(ad: string) {
    const lower = ad.normalize('NFC').toLowerCase()
    for (const [yol, url] of Object.entries(assetler)) {
      const dosya = yol.split('/').pop() || ''
      if (dosya.replace('.mp3','').toLowerCase() === lower) return url
    }
    return ''
  }
  const sesHaritasi: Record<string, string> = {
    Tomris: bul('Tomris') || bul('Tomris'),
    Mete: bul('mete'),
    Gökçe: bul('gökçe'),
    Oğuz: bul('oguz'), // eski kürşat dosya adı
  }
  const sesler = Object.keys(sesHaritasi)
  function dinlet(s: string) {
    const yol = sesHaritasi[s]
    if (!yol) return
    const audio = new Audio(yol)
    audio.play().catch(() => {})
  }
  return (
    <div className="grid grid-cols-1 lg2:grid-cols-2 gap-6">
      <Kart baslik="Varsayılan Ses">
        <RadioGrup
          name="ses"
          deger={aktif}
          setDeger={setAktif}
          secenekler={sesler.map((s) => ({ etiket: s, deger: s }))}
        />
      </Kart>
      <Kart baslik="Hazır Sesler (Yerel)">
        <div className="space-y-4">
          {sesler.map((s) => (
            <div key={s} className="flex items-center justify-between gap-4">
              <div>
                <div className="text-sm font-medium">{s}</div>
                {/* Yerel dosya yolu gizlendi */}
              </div>
              <Buton tur="ikincil" onClick={() => dinlet(s)}>Dinle</Buton>
            </div>
          ))}
        </div>
      </Kart>
    </div>
  )
}

// --- Yeni Sekmeler: Entegrasyon, Kalite, Gelişmiş (sahte veri ile) ---

function EntegrasyonIcerik() {
  // Sahte veri
  const apiKey = 'sk-live-1234...abcd'
  const saatVeri = useMemo(() => Array.from({ length: 12 }).map((_, i) => ({ x: i, y: Math.round(40 + Math.random() * 60) })), [])
  const eventLog = useMemo(() => Array.from({ length: 6 }).map((_, i) => {
    const turler = ['chat.completion', 'tts.request', 'webhook.delivery', 'auth.failed'] as const
    const t = turler[Math.floor(Math.random() * turler.length)]
    return {
      time: new Date(Date.now() - i * 1000 * 60 * 7).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      type: t,
      status: t === 'auth.failed' ? '401' : '200',
      ms: Math.round(120 + Math.random() * 400)
    }
  }), [])
  const usagePct = 62
  const ratePct = 40
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      <div className="xl:col-span-2 space-y-6">
        <Kart baslik="API Erişimi" ek={<Buton tur="ikincil" onClick={() => navigator.clipboard.writeText(apiKey)}>Kopyala</Buton>}>
          <div className="space-y-3">
            <div>
              <FormEtiket>Base URL</FormEtiket>
              <Girdi readOnly value={location.origin + '/api'} aria-label="Base URL" className="font-mono" />
            </div>
            <div>
              <FormEtiket>API Key</FormEtiket>
              <Girdi readOnly value={apiKey} aria-label="Api Key" className="font-mono" />
            </div>
            <Buton tur="ikincil" onClick={() => alert('Yeni key üretildi (sahte)')}>Key Yenile</Buton>
          </div>
        </Kart>
        <Kart baslik="Saatlik İstek Grafiği">
          <CizgiGrafik veri={saatVeri} ariaLabel="Saatlik istek sayısı" />
        </Kart>
        <Kart baslik="Olay Kaydı">
          <Tablo
            basliklar={['Zaman', 'Tür', 'Durum', 'Süre (ms)']}
            satirlar={eventLog.map(e => [e.time, e.type, <Etiket key={e.time} metin={e.status} renk={e.status === '200' ? 'basari' : 'hata'} />, e.ms])}
          />
        </Kart>
      </div>
      <div className="space-y-6">
        <Kart baslik="Kullanım">
          <IlerlemeCubugu oran={usagePct} etiketi={`Aylık kota kullanımı %${usagePct}`} />
        </Kart>
        <Kart baslik="Rate Limit">
          <IlerlemeCubugu oran={ratePct} etiketi={`Dakikalık limit %${ratePct}`} />
        </Kart>
        <Kart baslik="Webhook">
          <div className="space-y-3">
            <div>
              <FormEtiket>Webhook URL</FormEtiket>
              <Girdi placeholder="https://ornek.com/webhook" aria-label="Webhook" />
            </div>
            <Buton tur="ikincil" onClick={() => alert('Test webhook gönderildi (sahte)')}>Test Gönder</Buton>
          </div>
        </Kart>
      </div>
    </div>
  )
}

function KaliteIcerik() {
  const latencyVeri = useMemo(() => Array.from({ length: 14 }).map((_, i) => ({ x: i, y: Math.round(500 + Math.random() * 400) })), [])
  const tokenVeri = useMemo(() => Array.from({ length: 14 }).map((_, i) => ({ x: i, y: Math.round(800 + Math.random() * 600) })), [])
  const testler = useMemo(() => [
    { ad: 'Selamlama', beklenen: 'Hoş geldiniz', gercek: 'Hoş geldiniz', durum: 'pass' },
    { ad: 'Abone Paket', beklenen: 'Paket detayları döner', gercek: 'Paket detayları döner', durum: 'pass' },
    { ad: 'Fatura Tarihi', beklenen: 'Tarih formatı dd.mm.yyyy', gercek: 'yyyy-mm-dd', durum: 'fail' },
  ], [])
  const success = 92
  const guardrail = 3
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      <div className="xl:col-span-2 space-y-6">
        <Kart baslik="Yanıt Süresi (ms)">
          <CizgiGrafik veri={latencyVeri} ariaLabel="Latency" />
        </Kart>
        <Kart baslik="Token Kullanımı">
          <CizgiGrafik veri={tokenVeri} ariaLabel="Token" />
        </Kart>
        <Kart baslik="Test Senaryoları">
          <Tablo
            basliklar={['Senaryo', 'Beklenen', 'Gerçek', 'Durum']}
            satirlar={testler.map(t => [t.ad, t.beklenen, t.gercek, <Etiket key={t.ad} metin={t.durum} renk={t.durum === 'pass' ? 'basari' : 'hata'} />])}
          />
        </Kart>
      </div>
      <div className="space-y-6">
        <Kart baslik="Başarı Oranı">
          <IlerlemeCubugu oran={success} etiketi={`Başarılı yanıt %${success}`} />
        </Kart>
        <Kart baslik="Guardrail İhlalleri">
          <IlerlemeCubugu oran={guardrail} etiketi={`İhlal oranı %${guardrail}`} />
        </Kart>
        <Kart baslik="Memnuniyet Etiketleri">
          <div className="flex flex-wrap gap-2">
            <Etiket metin="Olumlu" renk="basari" />
            <Etiket metin="Belirsiz" renk="uyarı" />
            <Etiket metin="Olumsuz" renk="hata" />
          </div>
        </Kart>
      </div>
    </div>
  )
}

function GelismisIcerik() {
  const [model, setModel] = useState('gpt-4o-mini')
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState(1)
  const [presence, setPresence] = useState(0)
  const [maxTokens, setMaxTokens] = useState(512)
  const [memory, setMemory] = useState(12)
  const [pdfs, setPdfs] = useState<File[]>([])
  const kaydet = () => { alert('Ayarlar kaydedildi (sahte)') }
  function onPdfChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files || []).filter(f => f.type === 'application/pdf')
    if (files.length) setPdfs(prev => [...prev, ...files])
    e.target.value = ''
  }
  function removePdf(i: number) { setPdfs(p => p.filter((_,x) => x!==i)) }
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      <div className="xl:col-span-2 space-y-6">
        <Kart baslik="Model">
          <RadioGrup
            name="model"
            deger={model}
            setDeger={setModel}
            secenekler={[{ etiket: 'gpt-4o-mini', deger: 'gpt-4o-mini' }, { etiket: 'gpt-4o', deger: 'gpt-4o' }, { etiket: 'sonar-lite', deger: 'sonar-lite' }]}
          />
        </Kart>
        <Kart baslik="Parametreler" ek={<Buton tur="ikincil" onClick={kaydet}>Kaydet</Buton>}>
          <div className="space-y-5">
            <div>
              <FormEtiket>Temperature: {temperature.toFixed(2)}</FormEtiket>
              <Kaydirici ariaLabel="Temperature" min={0} max={1} deger={temperature} setDeger={setTemperature} />
            </div>
            <div>
              <FormEtiket>Top P: {topP.toFixed(2)}</FormEtiket>
              <Kaydirici ariaLabel="Top P" min={0} max={1} deger={topP} setDeger={setTopP} />
            </div>
            <div>
              <FormEtiket>Presence Penalty: {presence.toFixed(2)}</FormEtiket>
              <Kaydirici ariaLabel="Presence" min={0} max={2} deger={presence} setDeger={setPresence} />
            </div>
            <div>
              <FormEtiket>Max Tokens: {maxTokens}</FormEtiket>
              <Kaydirici ariaLabel="Max Tokens" min={64} max={2048} deger={maxTokens} setDeger={setMaxTokens} />
            </div>
          </div>
        </Kart>
        <Kart baslik="Bellek / Özetleme">
          <div className="space-y-4">
            <div>
              <FormEtiket>Konuşma geçmişi saklama (mesaj)</FormEtiket>
              <Kaydirici ariaLabel="Bellek" min={2} max={50} deger={memory} setDeger={setMemory} />
            </div>
            <MetinAlani rows={5} placeholder="Prompt şablon (örnek)" defaultValue={`KULLANICI MESAJLARI:\n{history}\n---\nSon kullanıcı mesajına yardımcı ve kurumsal bir tonla cevap ver.`} />
          </div>
        </Kart>
      </div>
      <div className="space-y-6">
        <Kart baslik="Belgeler (PDF)" ek={<div className="text-xs text-metinIkincil">RAG için yüklenecek</div>}>
          <div className="space-y-4">
            <div>
              <Girdi type="file" accept="application/pdf" multiple aria-label="PDF yükle" onChange={onPdfChange} />
              <div className="mt-2 text-xs text-metinIkincil">Sadece PDF. Henüz işlenmiyor.</div>
            </div>
            {pdfs.length > 0 && (
              <ul className="space-y-2 max-h-48 overflow-auto text-sm">
                {pdfs.map((f,i) => (
                  <li key={i} className="flex items-center justify-between gap-2 bg-arka rounded px-2 py-1">
                    <span className="truncate" title={f.name}>{f.name}</span>
                    <button onClick={() => removePdf(i)} className="text-xs text-hata hover:underline">Sil</button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </Kart>
      </div>
    </div>
  )
}



import { useEffect, useRef, useState } from 'react'
import { Buton } from './Buton'
import { Girdi } from './Form'
import { Volume2, VolumeX, Mic, MicOff, Loader2 } from 'lucide-react'

// Basit tarayıcı TTS + harici mp3 TTS (sunucu) seçeneği.
type TTSTur = 'browser' | 'server'

export type MesajUI = { rol: 'kullanici' | 'asistan'; icerik: string }

export function SohbetKutusu({ baslik, onGonder, placeholder = 'Mesaj yazın...', baslangicMesaji, ttsApiBase, onSpeakingChange }: { baslik: string; onGonder: (m: string) => Promise<string> | string; placeholder?: string; baslangicMesaji?: string; ttsApiBase?: string; onSpeakingChange?: (v:boolean)=>void }) {
  const [girdi, setGirdi] = useState('')
  const [mesajlar, setMesajlar] = useState<MesajUI[]>(baslangicMesaji ? [{ rol: 'asistan', icerik: baslangicMesaji }] : [])
  const [yukleniyor, setYukleniyor] = useState(false)
  const kaydirRef = useRef<HTMLDivElement>(null)
  const [sesAcik, setSesAcik] = useState(true)
  const [ttsTur, setTtsTur] = useState<TTSTur>('browser')
  const [spk, setSpk] = useState<{konusuyor: boolean; tur: TTSTur | null}>({konusuyor:false, tur: null})
  const audioRef = useRef<HTMLAudioElement | null>(null)
  
  // Ses kaydı ile ilgili state'ler
  const [kayitYapiliyor, setKayitYapiliyor] = useState(false)
  const [sesCeviriYapiliyor, setSesCeviriYapiliyor] = useState(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  

  useEffect(() => {
    kaydirRef.current?.scrollTo({ top: kaydirRef.current.scrollHeight })
  }, [mesajlar])

  async function gonder(metinOverride?: string) {
    const t = metinOverride || girdi.trim()
    if (!t) return
    setMesajlar((d) => [...d, { rol: 'kullanici', icerik: t }])
    setGirdi('')
    setYukleniyor(true)
    try {
      // Eğer ses özelliği açıksa ve sunucu TTS modunu kullanıyorsak doğrudan ses ile yanıt alıyoruz
      if (sesAcik && ttsTur === 'server' && ttsApiBase) {
        try {
          // Chat API'sine ses üretme talebiyle istek gönder
          const r = await fetch(`${ttsApiBase}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              session_id: Date.now().toString(),
              user_input: t,
              generate_audio: true,
              voice: seciliSes
            })
          })
          
          if (!r.ok) throw new Error('API hatası')
          
          const data = await r.json()
          setMesajlar((d) => [...d, { rol: 'asistan', icerik: data.text }])
          
          // Eğer ses URL'i döndüyse çal
          if (data.audio_url) {
            setSpeaking(true, 'server')
            if (!audioRef.current) audioRef.current = new Audio()
            audioRef.current.src = `${ttsApiBase}${data.audio_url}`
            audioRef.current.onended = () => setSpeaking(false, null)
            await audioRef.current.play().catch(console.error)
          } else {
            // Ses URL'i dönmediyse normal TTS kullan
            otomatikKonus(data.text)
          }
        } catch (err) {
          console.error("Ses üretme hatası:", err)
          // Hata durumunda standart işlemi gerçekleştir
          const yanit = await onGonder(t)
          setMesajlar((d) => [...d, { rol: 'asistan', icerik: yanit }])
          if (sesAcik) otomatikKonus(yanit)
        }
      } else {
        // Standart yanıt işlemi
        const yanit = await onGonder(t)
        setMesajlar((d) => [...d, { rol: 'asistan', icerik: yanit }])
        if (sesAcik) otomatikKonus(yanit)
      }
    } finally {
      setYukleniyor(false)
    }
  }

  function setSpeaking(v:boolean, tur: TTSTur | null) {
    setSpk({konusuyor:v, tur})
    onSpeakingChange?.(v)
  }

  function browserSpeak(text: string) {
    if (!('speechSynthesis' in window)) return
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text)
    u.lang = 'tr-TR'
    u.onstart = () => setSpeaking(true,'browser')
    u.onend = () => setSpeaking(false,null)
    window.speechSynthesis.speak(u)
  }

  // Kullanılabilir sesleri tutmak için state
  const [kullanilabilirSesler, setKullanilabilirSesler] = useState<string[]>(['mete', 'gökçe', 'oguz', 'Tomris'])
  const [seciliSes, setSeciliSes] = useState<string>('mete')
  
  // Sesleri yükle
  useEffect(() => {
    if (!ttsApiBase) return
    fetch(`${ttsApiBase}/tts/voices`)
      .then(r => r.json())
      .then(data => {
        if (data.voices && data.voices.length > 0) {
          setKullanilabilirSesler(data.voices)
        }
      })
      .catch(console.error)
  }, [ttsApiBase])

  async function serverSpeak(text: string) {
    if (!ttsApiBase) { browserSpeak(text); return }
    try {
      setSpeaking(true,'server')
      
      // XTTS v2 modelini kullanarak TTS isteği yap
      const r = await fetch(`${ttsApiBase}/tts/generate`, {
        method: 'POST', 
        headers: {'Content-Type': 'application/json'}, 
        body: JSON.stringify({
          text, 
          voice: seciliSes
        })
      })
      
      if (!r.ok) throw new Error('TTS sunucu hatası')
      const blob = await r.blob()
      const url = URL.createObjectURL(blob)
      if (!audioRef.current) audioRef.current = new Audio()
      audioRef.current.src = url
      audioRef.current.onended = () => { setSpeaking(false,null); URL.revokeObjectURL(url) }
      await audioRef.current.play().catch(()=>{})
    } catch (err) {
      console.error("TTS hatası:", err)
      setSpeaking(false,null)
    }
  }

  function otomatikKonus(text: string) {
    if (!sesAcik) return
    if (ttsTur === 'server') serverSpeak(text); else browserSpeak(text)
  }

  function manuelKonus(text: string) { otomatikKonus(text) }
  
  // Manuel ses kaydı başlatma fonksiyonu
  async function sesKaydiBaslat() {
    try {
      // Tarayıcıdan mikrofon erişimi iste
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // MediaRecorder nesnesini oluştur
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      // Veri kaydedildiğinde
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      // Kayıt tamamlandığında
      mediaRecorder.onstop = async () => {
        // Kaydedilen ses verisini bir blob olarak birleştir
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        
        // Whisper API'ye gönder
        await sesCevir(audioBlob);
        
        // Stream'i kapat
        stream.getTracks().forEach(track => track.stop());
      };
      
      // Kaydı başlat
      mediaRecorder.start();
      setKayitYapiliyor(true);
    } catch (error) {
      console.error("Ses kaydı başlatılırken hata:", error);
      alert("Mikrofon erişimi sağlanamadı. Lütfen mikrofon izinlerini kontrol edin.");
    }
  }  // Ses kaydını durdurma fonksiyonu
  function sesKaydiDurdur() {
    if (mediaRecorderRef.current && kayitYapiliyor) {
      mediaRecorderRef.current.stop();
      setKayitYapiliyor(false);
    }
  }
  
  // Whisper API kullanarak ses-metin dönüşümü
  async function sesCevir(audioBlob: Blob) {
    if (!ttsApiBase) {
      console.error("API base URL tanımlanmamış");
      return;
    }
    
    try {
      setSesCeviriYapiliyor(true);
      
      console.log("📣 Ses çevirme işlemi başlatılıyor...");
      console.log(`📊 Ses dosyası boyutu: ${(audioBlob.size / 1024).toFixed(2)} KB`);
      
      // Form data oluştur
      const formData = new FormData();
      formData.append('audio', audioBlob, 'audio.wav');
      
      // API'ye gönder
      console.log("🔄 Sunucuya ses dosyası gönderiliyor...");
      const response = await fetch(`${ttsApiBase}/speech-to-text`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`❌ Sunucu hatası (${response.status}):`, errorText);
        
        // Daha detaylı hata mesajı göster
        let errorMessage = "Ses çevirisinde bir hata oluştu. ";
        try {
          // JSON olarak parse etmeye çalış
          const errorData = JSON.parse(errorText);
          if (errorData.detail) {
            errorMessage += errorData.detail;
          }
        } catch {
          // JSON parse edilemezse düz metin olarak kullan
          errorMessage += errorText || "Lütfen tekrar deneyin.";
        }
        
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      console.log("✅ Ses metne çevrildi:", data);
      
      // Eğer dönüştürülmüş metin varsa input'a yerleştir ve gönder
      if (data.text) {
        setGirdi(data.text);
        console.log("⌨️ Metin girdi alanına eklendi, otomatik gönderiliyor...");
        // Otomatik olarak gönder
        setTimeout(() => {
          gonder(data.text);
        }, 500);
      } else {
        console.warn("⚠️ Çevrilen metin boş");
        alert("Ses algılanamadı veya metin çevirisinde bir sorun oluştu.");
      }
    } catch (error) {
      console.error("❌ Ses çevirisi sırasında hata:", error);
      
      // Daha bilgilendirici hata mesajları
      const errorMessage = error instanceof Error ? error.message : "Bilinmeyen bir hata oluştu";
      alert(errorMessage.includes("Ses çevirisinde bir hata") 
        ? errorMessage 
        : "Ses çevirisinde bir hata oluştu. Lütfen tekrar deneyin.");
    } finally {
      setSesCeviriYapiliyor(false);
    }
  }

  return (
    <div className="flex flex-col border border-cizgi rounded-kart bg-kart h-[520px]">
      <div className="px-5 py-3 border-b border-cizgi text-sm text-metinIkincil flex items-center justify-between gap-4">
        <span>{baslik}</span>
        <div className="flex items-center gap-3 text-xs">
          <button onClick={() => setSesAcik(s => !s)} className="inline-flex items-center gap-1 text-metin hover:text-metinIkincil">
            {sesAcik ? <Volume2 size={16}/> : <VolumeX size={16}/>} {sesAcik ? 'Ses Açık' : 'Sessiz'}
          </button>
          <select value={ttsTur} onChange={e=>setTtsTur(e.target.value as TTSTur)} className="border border-cizgi rounded px-1 py-0.5 bg-white">
            <option value="browser">Tarayıcı TTS</option>
            <option value="server">XTTS v2</option>
          </select>
          
          {ttsTur === 'server' && (
            <select 
              value={seciliSes} 
              onChange={e => setSeciliSes(e.target.value)} 
              className="border border-cizgi rounded px-1 py-0.5 bg-white"
            >
              {kullanilabilirSesler.map(ses => (
                <option key={ses} value={ses}>{ses}</option>
              ))}
            </select>
          )}
          
          {spk.konusuyor && <span className="animate-pulse text-[10px] text-basari">Konuşuyor...</span>}
        </div>
      </div>
      <div ref={kaydirRef} className="flex-1 overflow-auto p-5 space-y-4">
        {mesajlar.map((m, i) => (
          <div key={i} className={'max-w-[70%] px-4 py-2 rounded-2xl ' + (m.rol === 'kullanici' ? 'ml-auto bg-arka' : 'bg-white border border-cizgi')}>
            <div className="text-sm whitespace-pre-wrap">{m.icerik}</div>
            {m.rol === 'asistan' && sesAcik && (
              <div className="mt-1">
                <button onClick={() => manuelKonus(m.icerik)} className="text-[10px] text-metinIkincil hover:underline">Dinle</button>
              </div>) }
          </div>
        ))}
        {yukleniyor && <div className="text-sm text-metinIkincil">Yazıyor...</div>}
      </div>
      <div className="p-4 border-t border-cizgi flex items-center gap-2 flex-wrap">
        {/* Girdi */}
        <Girdi
          aria-label="Mesaj"
          placeholder={placeholder}
          value={girdi}
          onChange={(e) => setGirdi(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              gonder()
            }
          }}
          className="flex-1"
        />
        {/* Mikrofon butonu */}
        <button 
          className={`p-2 rounded-full ${kayitYapiliyor ? 'bg-red-500 text-white animate-pulse' : 'border border-cizgi hover:bg-arka'}`}
          onClick={kayitYapiliyor ? sesKaydiDurdur : sesKaydiBaslat}
          disabled={sesCeviriYapiliyor || yukleniyor}
          title={kayitYapiliyor ? 'Ses kaydını durdur' : 'Sesli mesaj gönder'}
        >
          {kayitYapiliyor ? (
            <MicOff size={20} />
          ) : sesCeviriYapiliyor ? (
            <Loader2 size={20} className="animate-spin" />
          ) : (
            <Mic size={20} />
          )}
        </button>
        {/* Gönder butonu */}
        <Buton onClick={() => gonder()} disabled={yukleniyor}>Gönder</Buton>
      </div>
    </div>
  )
}



import { useEffect, useMemo, useRef, useState } from 'react'

const VOWELS = 'aeıioöuüAEIİOÖUÜ'

function wordToSyllables(w: string): string[] {
  const s = w.toLowerCase().replace(/[^a-zçğıöşü]/g, '')
  if (!s) return []
  const parts: string[] = []
  let cur = ''
  for (let i = 0; i < s.length; i++) {
    const ch = s[i]!
    cur += ch
    if (VOWELS.includes(ch)) {
      const next = i + 1 < s.length ? s[i + 1]! : undefined
      if (next && !VOWELS.includes(next)) {
        cur += next
        i++
      }
      parts.push(cur)
      cur = ''
    }
  }
  if (cur) parts.push(cur)
  return parts
}

function syllableToViseme(s: string): 'CLOSED' | 'A' | 'E' | 'O' | 'OO' | 'FV' {
  if (/[mbp]$/.test(s)) return 'CLOSED'
  if (/[fv]/.test(s)) return 'FV'
  if (/[öü]/.test(s)) return 'OO'
  if (/[oouu]/.test(s) || /[oa]$/.test(s)) return 'O'
  if (/[oau]|\ba\b/.test(s)) return 'A'
  return 'E'
}

export function KonusanAvatar() {
  const [viseme, setViseme] = useState<'CLOSED' | 'A' | 'E' | 'O' | 'OO' | 'FV'>('CLOSED')
  const timers = useRef<number[]>([])
  useEffect(() => () => { timers.current.forEach((t) => clearTimeout(t)); timers.current = [] }, [])

  function show(v: typeof viseme) {
    setViseme(v)
  }

  function speak(text: string) {
    const win = window as any
    if (!('speechSynthesis' in window)) { alert('Tarayıcı TTS desteği yok.'); return }
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text)
    const pickVoice = () => {
      const voices = window.speechSynthesis.getVoices()
      u.voice = (voices.find((v) => /tr-|Turkish/i.test(v.lang)) || voices[0] || null) as SpeechSynthesisVoice | null
    }
    pickVoice(); window.speechSynthesis.onvoiceschanged = pickVoice
    u.rate = 1

    u.onstart = () => show('CLOSED')
    u.onend = () => { timers.current.forEach((t) => clearTimeout(t)); timers.current = []; show('CLOSED') }
    u.onboundary = (e: any) => {
      const word = (e?.charLength && typeof e.charIndex === 'number') ? text.slice(e.charIndex, e.charIndex + e.charLength) : ''
      const sylls = wordToSyllables(word)
      if (sylls.length) {
        const base = Math.max(120, 260 - (u.rate - 1) * 60)
        let t = 0
        sylls.forEach((sy) => {
          const v = syllableToViseme(sy)
          const dur = base * (sy.length >= 3 ? 1.2 : 1.0)
          timers.current.push(window.setTimeout(() => show(v), t))
          t += dur
          timers.current.push(window.setTimeout(() => show('CLOSED'), t - 40))
        })
      } else {
        show('CLOSED')
      }
    }
    window.speechSynthesis.speak(u)
  }

  const mouth = useMemo(() => {
    return (
      <div className="relative w-[240px] aspect-square bg-white rounded-[32px] shadow-hafif">
        <div className="absolute top-[35%] left-[34%] w-[18px] h-[18px] bg-[#111] rounded-full" />
        <div className="absolute top-[35%] right-[34%] w-[18px] h-[18px] bg-[#111] rounded-full" />
        {/* Ağız şekilleri */}
        {/* CLOSED */}
        {viseme === 'CLOSED' && (
          <svg className="absolute left-1/2 top-[64%] -translate-x-1/2 w-[120px] h-[70px]" viewBox="0 0 300 180"><path fill="#111" d="M30 92 Q150 120 270 92 Q150 140 30 92 Z"/></svg>
        )}
        {/* A */}
        {viseme === 'A' && (
          <svg className="absolute left-1/2 top-[64%] -translate-x-1/2 w-[120px] h-[70px]" viewBox="0 0 300 180"><path fill="#111" d="M30 70 Q150 40 270 70 Q250 165 50 165 Q30 120 30 70 Z"/><ellipse fill="#e11d48" cx="150" cy="122" rx="85" ry="36"/></svg>
        )}
        {/* E */}
        {viseme === 'E' && (
          <svg className="absolute left-1/2 top-[64%] -translate-x-1/2 w-[120px] h-[70px]" viewBox="0 0 300 180"><path fill="#111" d="M40 95 Q150 70 260 95 Q255 125 45 125 Q40 115 40 95 Z"/><rect fill="#e11d48" x="85" y="105" width="130" height="18" rx="9"/></svg>
        )}
        {/* O */}
        {viseme === 'O' && (
          <svg className="absolute left-1/2 top-[64%] -translate-x-1/2 w-[120px] h-[70px]" viewBox="0 0 300 180"><ellipse fill="#111" cx="150" cy="110" rx="56" ry="34"/><ellipse fill="#e11d48" cx="150" cy="115" rx="36" ry="20"/></svg>
        )}
        {/* OO */}
        {viseme === 'OO' && (
          <svg className="absolute left-1/2 top-[64%] -translate-x-1/2 w-[120px] h-[70px]" viewBox="0 0 300 180"><ellipse fill="#111" cx="150" cy="112" rx="40" ry="26"/><ellipse fill="#e11d48" cx="150" cy="116" rx="24" ry="14"/></svg>
        )}
        {/* FV */}
        {viseme === 'FV' && (
          <svg className="absolute left-1/2 top-[64%] -translate-x-1/2 w-[120px] h-[70px]" viewBox="0 0 300 180"><rect fill="#111" x="80" y="98" width="140" height="12" rx="6"/><rect fill="#e11d48" x="100" y="116" width="100" height="20" rx="10"/></svg>
        )}
      </div>
    )
  }, [viseme])

  const [metin, setMetin] = useState('Merhaba! Ben ücretsiz ve senkron konuşuyorum.')

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-center">{mouth}</div>
      <div className="flex gap-2">
        <input className="flex-1 h-11 rounded-input border border-cizgi px-3 text-sm" value={metin} onChange={(e) => setMetin(e.target.value)} />
        <button className="h-11 px-4 rounded-buton bg-metin text-white" onClick={() => speak(metin)}>Konuş</button>
      </div>
    </div>
  )
}



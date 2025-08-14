// Basit API URL çözümleyici: .env yoksa localhost'a düş
function resolveApiUrl(apiUrl?: string) {
  return apiUrl ?? (import.meta as any).env?.VITE_CHAT_API_URL ?? 'http://127.0.0.1:8000'
}

// API çağrıları (yalnızca sohbet) - TTS bağlantıları kaldırıldı
export async function sohbetGonder(apiUrl: string, sessionId: string, userInput: string): Promise<string> {
  const base = resolveApiUrl(apiUrl)
  const r = await fetch(`${base}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, user_input: userInput }),
  })
  if (!r.ok) throw new Error('Sunucu hatası')
  const j = await r.json()
  return j.text as string
}

export type GecmisSatiri = { id: string; timestamp: string; user: string; channel: string; summary: string; duration: string }
export async function sohbetGecmisi(apiUrl: string): Promise<GecmisSatiri[]> {
  const base = resolveApiUrl(apiUrl)
  // Güncellenmiş endpoint: artık hem session'lar hem dosyalar gelir
  const r = await fetch(`${base}/sohbet_gecmisi`)
  if (!r.ok) throw new Error('Sunucu hatası')
  return (await r.json()) as GecmisSatiri[]
}

// Bir chat dosyasını getir
export async function chatDosyasiIcerik(apiUrl: string, filename: string): Promise<{ filename: string; content: string }> {
  const base = resolveApiUrl(apiUrl)
  const r = await fetch(`${base}/sohbet_gecmisi/${encodeURIComponent(filename)}`)
  if (!r.ok) throw new Error('Sunucu hatası')
  return (await r.json()) as { filename: string; content: string }
}

// Sohbet geçmişi öğesi sil
export async function sohbetGecmisiSil(apiUrl: string, itemId: string): Promise<{ status: string; message: string }> {
  const base = resolveApiUrl(apiUrl)
  const r = await fetch(`${base}/sohbet_gecmisi/${encodeURIComponent(itemId)}`, { method: 'DELETE' })
  if (!r.ok) throw new Error('Sunucu hatası')
  return (await r.json()) as { status: string; message: string }
}

// Tüm sohbet geçmişini temizle
export async function tumSohbetGecmisiTemizle(apiUrl?: string): Promise<{ status: string; message: string }> {
  const base = resolveApiUrl(apiUrl)
  const r = await fetch(`${base}/sohbet_gecmisi/clear_all`, { method: 'POST' })
  if (!r.ok) throw new Error('Sunucu hatası')
  return (await r.json()) as { status: string; message: string }
}




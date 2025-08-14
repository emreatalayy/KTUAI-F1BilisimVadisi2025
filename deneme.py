import requests
import time

API_KEY = "tts-c436fd4702f7aa6537abf1ce455d3d4e"  # kendi key'inle değiştir
BASE = "https://api.ttsopenai.com/uapi/v1"
CREATE_URL = f"{BASE}/text-to-speech"
HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}
PAYLOAD = {
    "model": "tts-1",
    "voice_id": "PE0150",  # Türkçe normal ses
    "speed": 1.0,
    "input": "Merhaba dünya! Bugün nasılsın?"
}

# 1) İş oluştur
r = requests.post(CREATE_URL, headers=HEADERS, json=PAYLOAD, timeout=30)
print("Create status:", r.status_code)
print("Create response:", r.text)
r.raise_for_status()
job = r.json()
uuid = job.get("uuid") or job.get("id")
if not uuid:
    raise SystemExit("UUID bulunamadı")

# 2) Status kontrolü
status_urls = [
    f"{BASE}/text-to-speech/{uuid}",
    f"{BASE}/text-to-speech/status/{uuid}",
]
def get_status():
    for u in status_urls:
        s = requests.get(u, headers=HEADERS, timeout=15)
        if s.status_code < 500:
            return s.json()
    raise RuntimeError("Status alınamadı")

start = time.time()
while True:
    info = get_status()
    st = str(info.get("status") or info.get("status_code") or info.get("state"))
    print("Status:", st)
    if st in ("2", "completed", "done", "success"):
        break
    if st in ("3", "error", "failed"):
        raise RuntimeError(f"Hata: {info}")
    if time.time() - start > 30:
        raise TimeoutError("30sn zaman aşımı")
    time.sleep(1.5)

# 3) İndir
audio_urls = [
    f"{BASE}/text-to-speech/{uuid}/download",
    f"{BASE}/text-to-speech/download/{uuid}",
]
audio_data = None
for au in audio_urls:
    a = requests.get(au, headers=HEADERS, timeout=30)
    if a.status_code < 400 and a.content:
        audio_data = a.content
        break

if not audio_data:
    raise RuntimeError("Ses indirilemedi")

with open("output.mp3", "wb") as f:
    f.write(audio_data)

print("Kaydedildi: output.mp3")
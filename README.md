
# KTU AI-F1 Chatbot Projesi

KTU AI-F1, Türkcell müşteri hizmetleri simülasyonu yapan gelişmiş bir AI chatbot uygulamasıdır. Bu proje, doğal dil işleme teknolojileri kullanarak kullanıcılarla etkileşime geçen, paket değişiklikleri yapabilen ve sesli yanıt verebilen modern bir müşteri hizmetleri platformu sunar.

![Demo](media/demo.gif)

# Özellikler

Bu proje aşağıdaki ana özellikleri içermektedir:

## Doğal Dil İşleme
- Google Gemini AI entegrasyonu ile gelişmiş dil anlama
- Çok adımlı konuşma yönetimi
- Otomatik konu değişimi algılama
- Bağlamsal bellek yönetimi

## Müşteri Hizmetleri Simülasyonu
- Kullanıcı kimlik doğrulama
- Paket bilgilerini sorgulama
- Paket değişiklik işlemleri
- Fatura sorgulama ve ödeme
- Özel teklifler ve kampanyalar

## Sesli Etkileşim
- Text-to-Speech (TTS) desteği
- Çoklu ses profili (mete, oguz, tomris, gokce)
- Real-time audio generation
- XTTS v2 model entegrasyonu

## Modern Web Arayüzü
- React + TypeScript + Tailwind CSS
- Responsive tasarım
- Real-time chat interface
- Audio playback controls

# Teknoloji Stack'i

## Frontend
- **React 18** - Modern UI framework
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **Vite** - Fast build tool
- **Lucide React** - Modern icon library

## Backend
- **FastAPI** - Modern Python web framework
- **Python 3.10+** - Programming language
- **Google Gemini AI** - Natural language processing
- **TTS (Coqui)** - Text-to-speech synthesis
- **PyTorch** - Machine learning framework

## Veri Yönetimi
- **JSON** - Session data storage
- **SQLite** - Future database integration
- **File-based** - Chat history management

# Kurulum

## Gereksinimler

- **Node.js 18+** - Frontend development
- **Python 3.10+** - Backend development
- **Git** - Version control
- **CUDA** (opsiyonel) - GPU acceleration for TTS

## Frontend Kurulumu

Proje kök dizininde aşağıdaki komutları çalıştırın:

```bash
# Bağımlılıkları yükle
npm install

# Geliştirme sunucusunu başlat
npm run dev

# Production build
npm run build
```

## Backend Kurulumu

Server dizinine geçin ve Python ortamını hazırlayın:

```bash
# Server dizinine geç
cd server

# Sanal ortam oluştur (önerilen)
python -m venv .venv310

# Sanal ortamı aktifleştir (Windows)
.\.venv310\Scripts\activate

# Sanal ortamı aktifleştir (Linux/Mac)
source .venv310/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

## Çevre Değişkenleri

### Frontend (.env dosyası)
```env
VITE_CHAT_API_URL=http://127.0.0.1:8000
```

### Backend (server/.env dosyası)
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

# Çalıştırma

## Development Mode

Frontend ve backend'i ayrı terminallerde çalıştırın:

**Frontend Terminal:**
```bash
npm run dev
```

**Backend Terminal:**
```bash
cd server
.\.venv310\Scripts\activate  # Windows
uvicorn app:app --reload --host 127.0.0.1 --port 8000
#çalışmassa
uvicorn app:app --reload --host 127.0.0.1 --port 8001
```

## Production Mode

```bash
# Frontend build
npm run build

# Backend production
cd server
uvicorn app:app --host 0.0.0.0 --port 8000
```

# API Endpoints

## Chat Endpoints
- `POST /chat` - Ana chat endpoint'i
- `GET /sessions` - Aktif oturumları listele
- `GET /session/{session_id}/messages` - Oturum mesajlarını getir

## TTS Endpoints
- `POST /tts/generate` - Metni sese dönüştür
- `GET /tts/voices` - Mevcut sesleri listele
- `POST /tts/status` - TTS servis durumu

## Utility Endpoints
- `GET /diag` - Sistem diagnostics
- `GET /audio/{filename}` - Audio dosyalarını serve et

# Proje Yapısı

```
KTU-AI-F1/
├── src/                    # Frontend source code
│   ├── bilesenler/        # React components
│   ├── sayfalar/          # Pages
│   ├── uygulama/          # Application logic
│   └── yapi/              # Project structure
├── server/                 # Backend source code
│   ├── app.py             # Main FastAPI application
│   ├── requirements.txt   # Python dependencies
│   ├── audio/             # Generated audio files
│   └── sohbet_gecmisi/    # Chat history files
├── public/                # Static assets
├── dist/                  # Built frontend files
└── package.json           # Frontend dependencies
```

# Kullanım

## Temel Kullanım

1. **Uygulama Başlatma**: Frontend ve backend servislerini çalıştırın
2. **Chat Başlatma**: Ana sayfada chat arayüzünü kullanın
3. **Kimlik Doğrulama**: Telefon numarası ile giriş yapın (5550001 veya 5550002)
4. **İşlem Yapma**: Paket sorgulama, değişiklik, fatura işlemleri

## Örnek Senaryolar

### Paket Bilgisi Sorgulama
```
Kullanıcı: "Merhaba, 5550001 numaralı hattımın paket bilgilerini öğrenebilir miyim?"
AI: "Tabii, paket bilgilerinizi kontrol ediyorum..."
```

### Paket Değişikliği
```
Kullanıcı: "Paketimi değiştirmek istiyorum"
AI: "Mevcut paketler: MegaPaket 100, Ekonomik Paket..."
```

### Sesli Yanıt
```
Kullanıcı: Ses aktifleştir butonuna tıklar
AI: Metinsel yanıtın yanında audio dosyası da oluşturulur
```


# Prompt Injector Nedir?

Prompt Injector, yapay zeka tabanlı sohbet sistemlerinde modelin davranışını yönlendirmek veya test etmek amacıyla kullanılan özel bir girdi (prompt) ekleme tekniğidir. Bu yöntemle, modele belirli bir görev, rol veya yanıt biçimi dayatılabilir ya da modelin sınırları ve güvenliği test edilebilir. Projede, modelin farklı senaryolara verdiği yanıtları gözlemlemek ve güvenlik açıklarını analiz etmek için prompt injection tekniklerinden yararlanılmıştır.

<p align="center">
	<img src="media/2.png" width="400" />
</p>

# bert_train_predict_data Klasörü Hakkında

`bert_train_predict_data` klasörü, BERT tabanlı modelin eğitimi, veri hazırlığı ve tahmin (predict) işlemlerine dair script ve dosyaları içerir. Buradaki içerikler, modelin geliştirilmesi ve test edilmesi amacıyla kullanılmıştır. Projenin ana çalışma akışı ve chatbot uygulamasının çalıştırılması için bu klasöre ihtiyaç yoktur. Sadece model geliştirme ve analiz süreçleriyle ilgilenenler için ek bilgi sağlar.


# Veri Seti Hakkında

Bu projede kullanılan veri seti, müşteri hizmetleri senaryolarına uygun olarak hazırlanmış, çoklu etiketli ve Türkçe metinlerden oluşmaktadır. Veri setinde her bir örnek için kategori, duygu türü, konu ve öncelik gibi etiketler bulunmaktadır. Bu etiketler, BERT tabanlı modelin çoklu görevli (multi-task) olarak eğitilmesini sağlamıştır. Veri seti, gerçek müşteri taleplerini ve çeşitli senaryoları kapsayacak şekilde özenle oluşturulmuştur.

Veri seti örnekleri ve etiket yapısı aşağıdaki görsellerde özetlenmiştir:

<p align="center">
	<img src="media/1 (1).jpeg" width="350" />
	<img src="media/1 (2).jpeg" width="350" />
	<img src="media/1 (3).jpeg" width="350" />
</p>

# BERT Modeli ve Analiz

BERT tabanlı modelin eğitimi grafiği aşağıdadır.

<p align="center">
	<img src="media/1 (4).jpeg" width="500" />
</p>


# Sorun Giderme

## Yaygın Sorunlar

### TTS Çalışmıyor
- CUDA driver'ları güncel olduğundan emin olun
- PyTorch ve TTS paketlerini yeniden yükleyin
- XTTS model dosyalarının indirildiğini kontrol edin

### Frontend Backend Bağlantısı
- Port ayarlarını kontrol edin (.env dosyaları)
- CORS ayarlarını kontrol edin
- Firewall/antivirus yazılımlarını kontrol edin

### Gemini API Hatası
- API key'in doğru olduğundan emin olun
- API kullanım limitlerini kontrol edin
- İnternet bağlantısını kontrol edin

# Geliştirme

## Development Guidelines

- **Code Style**: TypeScript/Python best practices
- **Git Flow**: Feature branches kullanın
- **Testing**: Unit testler ekleyin
- **Documentation**: Kod içi documentation yapın

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Create pull request

# Lisans

Bu proje açık kaynak olarak geliştirilmiştir. Detaylar için LICENSE dosyasına bakın.

# İletişim

Proje ile ilgili sorularınız için GitHub Issues kullanabilirsiniz.

# Teşekkürler

Bu proje, modern AI teknolojilerini kullanarak müşteri hizmetleri deneyimini geliştirmeyi amaçlamaktadır. Değerli mentörlerimiz Hakan Kendirci, Merve Ayyüce Kızrak, Oğuzhan Yenen ve Gökhan Üstünkar ve bilişim vadisi ekibi başta olmak üzere katkıda bulunan herkese teşekkürler.


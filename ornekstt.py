# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime
import google.generativeai as genai
from typing import Dict, Any, List
import time

# BERT ve Analiz için Gerekli Kütüphaneler
import torch
from transformers import BertTokenizer, BertModel
import pickle

# Whisper için gerekli kütüphaneler
import whisper

# Ses kaydı için gerekli kütüphaneler
import pyaudio
import wave
import numpy as np

# TTS için gerekli kütüphaneler
from TTS.api import TTS

# ==============================================================================
# BÖLÜM 1: BERT TABANLI ANALİZ MODELİ VE FONKSİYONLARI
# ==============================================================================

# Gerekli dosya yollarını burada belirtin.
MODEL_PATH = 'turkish_bert_three_task_model.pth'
ENCODERS_PATH = 'three_task_label_encoders.pkl'

# BERT Model Sınıfı
class ThreeTaskBERT(torch.nn.Module):
    def __init__(self, n_kategori, n_duygu_turu, n_konu):
        super(ThreeTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained("dbmdz/bert-base-turkish-uncased")
        self.dropout = torch.nn.Dropout(0.3)
        self.kategori_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_kategori)
        self.duygu_turu_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_duygu_turu)
        self.konu_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_konu)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return {
            'kategori': self.kategori_classifier(pooled_output),
            'duygu_turu': self.duygu_turu_classifier(pooled_output),
            'konu': self.konu_classifier(pooled_output)
        }

# Modeli ve bileşenleri yüklemek için fonksiyon
def load_analysis_model_and_components(model_path, encoders_path):
    """
    BERT modelini, tokenizer'ı ve label encoder'ları yükler.
    """
    try:
        print("Metin Analiz Modeli yükleniyor...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        
        n_kategori = len(encoders['kategori'].classes_)
        n_duygu_turu = len(encoders['duygu_turu'].classes_)
        n_konu = len(encoders['konu'].classes_)
        
        tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
        model = ThreeTaskBERT(n_kategori, n_duygu_turu, n_konu)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        print("✅ Metin Analiz Modeli başarıyla yüklendi.")
        return model, tokenizer, encoders, device
    except FileNotFoundError:
        print(f"❌ Analiz modeli için gerekli dosyalar bulunamadı! Lütfen '{model_path}' ve '{encoders_path}' dosyalarının mevcut olduğundan emin olun.")
        return None, None, None, None
    except Exception as e:
        print(f"❌ Analiz modeli yüklenirken bir hata oluştu: {e}")
        return None, None, None, None

# Tahmin Fonksiyonu
def predict_issue_details(text, model, tokenizer, encoders, device):
    """Verilen metni analiz edip tahminleri döndürür."""
    try:
        encoding = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=128,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        _, kategori_preds = torch.max(outputs['kategori'], dim=1)
        _, duygu_turu_preds = torch.max(outputs['duygu_turu'], dim=1)
        _, konu_preds = torch.max(outputs['konu'], dim=1)
        
        kategori = encoders['kategori'].inverse_transform(kategori_preds.cpu().numpy())[0]
        duygu_turu = encoders['duygu_turu'].inverse_transform(duygu_turu_preds.cpu().numpy())[0]
        konu = encoders['konu'].inverse_transform(konu_preds.cpu().numpy())[0]
        
        return {'kategori': kategori, 'duygu_turu': duygu_turu, 'konu': konu}
    except Exception as e:
        print(f"Metin analizi sırasında hata: {e}")
        return None

# ==============================================================================
# BÖLÜM 2: SES KAYIT VE TRANSCRIBE FONKSİYONLARI
# ==============================================================================

def dynamic_audio_record(
    threshold=15, duration_after_silence=2, rate=44100, chunk=1024, output_file="output.wav", show_level=False
):
    """
    Mikrofonu sürekli dinleyerek belirli bir ses seviyesi (threshold) aşıldığında kayda başlar.
    Args:
        threshold (int): Mikrofonun algılayacağı minimum ses seviyesi (0-100 arası).
        duration_after_silence (float): Sessizlikten sonra ne kadar süre daha kaydedileceği (saniye).
        rate (int): Örnekleme frekansı (Hz).
        chunk (int): Her okuma sırasında alınacak örnek boyutu.
        output_file (str): Kaydedilecek ses dosyasının adı.
        show_level (bool): Ses seviyesini sürekli göster.
    """
    # PyAudio nesnesi oluştur
    audio_interface = pyaudio.PyAudio()

    # Ses akışını başlat
    stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )

    print("🎤 Mikrofon dinleniyor... (Başlamak için konuşun)")

    frames = []
    recording = False
    silence_start_time = None

    try:
        while True:
            # Mikrofondan veri oku ve numpy array'e dönüştür
            data = stream.read(chunk, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Ses seviyesini hesapla (RMS)
            if len(audio_data) > 0:
                rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
                normalized_volume = min(100, int((rms / 5000) * 100)) if rms > 0 else 0
            else:
                normalized_volume = 0

            # Debug modunda ses seviyesini göster
            if show_level:
                print(f"Ses seviyesi: {normalized_volume}")

            if normalized_volume > threshold:
                if not recording:
                    print(f"🎤 Ses algılandı! Kayıt başlatılıyor... (Ses seviyesi: {normalized_volume})")
                    recording = True
                frames.append(data)
                silence_start_time = None
            elif recording:
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > duration_after_silence:
                    print("🎤 Sessizlik algılandı. Kayıt durduruluyor...")
                    break
                frames.append(data)
    except KeyboardInterrupt:
        print("🎤 Kayıt manuel olarak durduruldu.")
    finally:
        # Kaynakları temizle
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

        # Ses dosyasını kaydet
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(audio_interface.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(rate)
            wav_file.writeframes(b''.join(frames))

        print(f"✅ Kayıt tamamlandı. '{output_file}' dosyasına kaydedildi.")

def load_whisper_model():
    """Whisper modelini yükler."""
    print("🔄 Whisper ses tanıma modeli yükleniyor...")
    try:
        if not torch.cuda.is_available():
            model = whisper.load_model("turbo", device="cpu")
            print("✅ Whisper modeli CPU üzerinde yüklendi.")
        else:
            model = whisper.load_model("turbo", device="cuda")
            print("✅ Whisper modeli GPU üzerinde yüklendi.")
        return model
    except Exception as e:
        print(f"❌ Whisper modeli yüklenirken bir hata oluştu: {e}")
        return None

def transcribe_audio(whisper_model, audio_file):
    """Ses dosyasını metne çevirir."""
    try:
        print(f"🔄 Ses dosyası metne çevriliyor: {audio_file}")
        result = whisper_model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        print(f"❌ Ses dosyası metne çevrilirken bir hata oluştu: {e}")
        return None

# ==============================================================================
# BÖLÜM 3: TTS (TEXT-TO-SPEECH) FONKSİYONLARI
# ==============================================================================

def load_tts_model():
    """TTS modelini yükler."""
    try:
        print("🔄 TTS ses sentez modeli yükleniyor...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("✅ TTS modeli başarıyla yüklendi.")
        return tts
    except Exception as e:
        print(f"❌ TTS modeli yüklenirken bir hata oluştu: {e}")
        return None

def text_to_speech(tts_model, text, speaker_wav="voice_reference.wav", language="tr", output_file="response.wav"):
    """Metni ses dosyasına dönüştürür."""
    try:
        print("🔄 Asistan yanıtı seslendiriliyor...")
        
        # Ses referans dosyası kontrolü
        if not os.path.exists(speaker_wav):
            print(f"⚠️ Referans ses dosyası bulunamadı: {speaker_wav}. Varsayılan ses kullanılacak.")
            # Bu durumda varsayılan referans ses kullanmak gerekebilir
        
        # Metni ses dosyasına dönüştür
        tts_model.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_file
        )
        print(f"✅ Ses dosyası oluşturuldu: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Metin sese dönüştürülürken bir hata oluştu: {e}")
        return False

def play_audio_file(file_path):
    """Ses dosyasını çalar."""
    try:
        import os
        import platform
        
        # İşletim sistemine göre çalma komutu
        system = platform.system()
        print(f"🔊 Ses dosyası çalınıyor: {file_path}")
        
        if system == "Windows":
            os.system(f"start {file_path}")
        elif system == "Darwin":  # macOS
            os.system(f"afplay {file_path}")
        else:  # Linux ve diğerleri
            os.system(f"aplay {file_path}")
            
        return True
    except Exception as e:
        print(f"❌ Ses dosyası çalınırken bir hata oluştu: {e}")
        return False

# ==============================================================================
# BÖLÜM 4: MÜŞTERİ HİZMETLERİ AGENT'I
# ==============================================================================

# --- Statik Veriler ---
CUSTOMERS = {
    "5550001": {"name": "Ali", "surname": "Can", "current_package_id": "PN2", "contract_end_date": "2025-08-01", "payment_status": "Paid"},
    "5550002": {"name": "Ayşe", "surname": "Demir", "current_package_id": "PN1", "contract_end_date": "2024-12-15", "payment_status": "Overdue"},
}
PACKAGE_CATALOG = [
    {"id": "PN1", "name": "MegaPaket 100", "price": "150 TL", "details": "100 Mbps fiber internet, sınırsız yurt içi konuşma, TV+ full paket."},
    {"id": "PN2", "name": "Ekonomik Paket", "price": "80 TL", "details": "25 Mbps internet, 500 dakika konuşma, 10 GB mobil data."},
]

# --- GEMINI İÇİN ARAÇ (TOOL) FONKSİYONLARI ---

# Analiz Aracı
def analyze_customer_issue(issue_text: str) -> Dict[str, Any]:
    """
    Kullanıcının yazdığı şikayet, sorun veya talebi analiz ederek kategorisini,
    duygusunu ve konusunu belirler.
    """
    print("⚙️ Analiz Aracı Çağrıldı")
    if analysis_model is None:
        return {"error": "Analiz modeli şu an kullanılamıyor."}
    
    results = predict_issue_details(issue_text, analysis_model, analysis_tokenizer, analysis_encoders, analysis_device)
    if results:
        return {
            "kategori": results['kategori'],
            "duygu": results['duygu_turu'],
            "konu": results['konu']
        }
    return {"error": "Metin analizi başarısız oldu."}

# Diğer Araçlar
def get_user_info(user_id: str) -> Dict[str, Any]:
    """Kullanıcının profil bilgilerini ve mevcut paket ID'sini getirir."""
    print(f"⚙️ Araç Çağrıldı: get_user_info(user_id={user_id})")
    if user_id in CUSTOMERS:
        return CUSTOMERS[user_id]
    raise ValueError("Kullanıcı bulunamadı.")

def get_package_details_by_id(package_id: str) -> Dict[str, Any]:
    """Paket ID'sine göre paket detaylarını getirir."""
    print(f"⚙️ Araç Çağrıldı: get_package_details_by_id(package_id={package_id})")
    for package in PACKAGE_CATALOG:
        if package['id'] == package_id:
            return package
    raise ValueError(f"'{package_id}' ID'li bir paket bulunamadı.")

def get_available_packages() -> List[Dict[str, Any]]:
    """Mevcut tüm paketleri listeler."""
    print(f"⚙️ Araç Çağrıldı: get_available_packages()")
    return PACKAGE_CATALOG

def initiate_package_change(user_id: str, new_package_id: str) -> Dict[str, Any]:
    """Kullanıcının paketini değiştirmek için işlem başlatır."""
    print(f"⚙️ Araç Çağrıldı: initiate_package_change({user_id}, {new_package_id})")
    user = get_user_info(user_id)
    if not any(pkg['id'] == new_package_id for pkg in PACKAGE_CATALOG):
        raise ValueError(f"'{new_package_id}' ID'li bir paket bulunamadı.")
    today_str = datetime.now().strftime("%Y-%m-%d")
    if user["contract_end_date"] > today_str:
        return {"success": False, "error": f"Taahhüdünüz {user['contract_end_date']} tarihinde dolacağı için şu an değişiklik yapılamamaktadır. Cayma bedeli hakkında bilgi almak için 'cayma bedeli' yazabilirsiniz."}
    CUSTOMERS[user_id]["current_package_id"] = new_package_id
    new_package_name = next(pkg["name"] for pkg in PACKAGE_CATALOG if pkg["id"] == new_package_id)
    return {"success": True, "message": f"Harika! {user['name']} Hanım/Bey, yeni paketiniz olan '{new_package_name}' başarıyla tanımlanmıştır."}

# ==============================================================================
# BÖLÜM 5: KONSOL ARAYÜZÜ VE ANA UYGULAMA MANTIĞI
# ==============================================================================

def print_header():
    """Konsol uygulamasının başlık bölümünü yazdırır."""
    print("\n" + "="*80)
    print("🤖  TürkTel Akıllı Müşteri Hizmetleri Asistanı")
    print("Gemini 1.5 Flash, BERT Analizi, Whisper Ses Tanıma ve XTTS Ses Sentezi ile Güçlendirilmiştir")
    print("="*80 + "\n")

def get_api_key():
    """Kullanıcıdan Gemini API anahtarını alır."""
    api_key = input("Gemini API Anahtarınızı Girin (gizli tutulacaktır): ")
    return api_key

def initialize_chat(api_key):
    """Gemini sohbet modelini başlatır ve oturumu başlatır."""
    try:
        genai.configure(api_key=api_key)
        
        # Sistem talimatı
        system_instruction = """
        Sen TürkTel müşteri hizmetleri asistanısın. Görevin, kullanıcının sorunlarını çözmek ve taleplerini yerine getirmektir.
        - Daima yardımsever, empatik ve profesyonel ol.
        - BİRİNCİL GÖREV: Kullanıcı bir sorun, şikayet veya belirsiz bir talep ile geldiğinde, durumu daha iyi anlamak için HER ZAMAN ÖNCE `analyze_customer_issue` aracını kullan.
          Bu analiz (kategori, duygu, konu) sana daha empatik ve doğru yanıt vermende yardımcı olacaktır.
        - Analiz yaptıktan veya bir işlem yapmadan önce, eğer gerekliyse, kullanıcı ID'sini (telefon numarasını) öğren. Geçerli ID'ler: '5550001', '5550002'.
        - Araçlarını (`tools`) akıllıca kullan. Bazen bir hedefe ulaşmak için birden fazla aracı sırayla çağırman gerekebilir.
        - Bir aracı kullandıktan sonra, elde ettiğin teknik bilgiyi kullanıcıya daima net, tam cümlelerle ve anlaşılır bir dille özetle.
        - Kullanıcıya asla fonksiyon adı veya teknik terim söyleme. "Sistemden bilgilerinizi kontrol ediyorum", "Talebinizi daha iyi anlamak için analiz ediyorum" gibi ifadeler kullan.
        """
        
        model = genai.GenerativeModel(
            'gemini-1.5-flash-latest',
            system_instruction=system_instruction,
            tools=[
                analyze_customer_issue,
                get_user_info,
                get_package_details_by_id,
                get_available_packages,
                initiate_package_change
            ]
        )
        chat = model.start_chat(enable_automatic_function_calling=True)
        print("\n✅ Sohbet başarıyla başlatıldı.\n")
        return chat
    except Exception as e:
        print(f"\n❌ Model başlatılırken bir hata oluştu: {e}")
        return None

def get_voice_reference():
    """Kullanıcıdan ses referans dosyasını alır."""
    default_path = "voice_reference.wav"
    
    while True:
        print("\nSes klonlama için referans ses dosyası gereklidir.")
        print("1. Varsayılan ses referansı kullan")
        print("2. Özel ses referans dosyası belirt")
        choice = input("Seçiminiz (1/2): ")
        
        if choice == "1":
            if not os.path.exists(default_path):
                print("⚠️ Varsayılan ses dosyası bulunamadı. 5 saniyelik bir referans ses kaydı oluşturulacak.")
                print("Lütfen mikrofona 5 saniye boyunca konuşun...")
                
                # PyAudio nesnesi oluştur
                audio_interface = pyaudio.PyAudio()
                
                # Kayıt parametreleri
                seconds = 5
                fs = 44100  # Örnek oran
                
                # Kayıt akışını başlat
                stream = audio_interface.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=fs,
                    input=True,
                    frames_per_buffer=1024
                )
                
                print("🎤 Kayıt başlıyor...")
                frames = []
                
                # 5 saniye kaydet
                for i in range(0, int(fs / 1024 * seconds)):
                    data = stream.read(1024)
                    frames.append(data)
                    
                print("✅ Kayıt tamamlandı.")
                
                # Kaynakları temizle
                stream.stop_stream()
                stream.close()
                audio_interface.terminate()
                
                # Ses dosyasını kaydet
                with wave.open(default_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(audio_interface.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(fs)
                    wf.writeframes(b''.join(frames))
                
                print(f"✅ Referans ses dosyası kaydedildi: {default_path}")
            
            return default_path
        
        elif choice == "2":
            file_path = input("Referans ses dosyasının tam yolunu girin: ")
            if os.path.exists(file_path):
                return file_path
            else:
                print(f"❌ Belirtilen dosya bulunamadı: {file_path}")
        
        else:
            print("❌ Geçersiz seçim. Lütfen tekrar deneyin.")

def get_interaction_mode():
    """Kullanıcıdan etkileşim modunu seçmesini ister."""
    while True:
        print("\nEtkileşim modunu seçin:")
        print("1. Yazılı konuşma")
        print("2. Sesli konuşma")
        choice = input("Seçiminiz (1/2): ")
        if choice in ["1", "2"]:
            return "text" if choice == "1" else "voice"
        print("❌ Geçersiz seçim. Lütfen 1 veya 2 girin.")

def main():
    """Ana uygulama mantığını yürüten fonksiyon."""
    print_header()
    
    # Analiz modelini yükle
    global analysis_model, analysis_tokenizer, analysis_encoders, analysis_device
    analysis_model, analysis_tokenizer, analysis_encoders, analysis_device = load_analysis_model_and_components(MODEL_PATH, ENCODERS_PATH)
    
    # API anahtarını al
    api_key = get_api_key()
    if not api_key:
        print("❌ API anahtarı sağlanmadı. Program sonlandırılıyor.")
        return
    
    # Sohbet oturumunu başlat
    chat = initialize_chat(api_key)
    if not chat:
        return
    
    # Etkileşim modunu seç
    interaction_mode = get_interaction_mode()
    
    # Model yükleme
    whisper_model = None
    tts_model = None
    voice_reference = None
    
    if interaction_mode == "voice":
        # Whisper modelini yükle (konuşma tanıma)
        whisper_model = load_whisper_model()
        if not whisper_model:
            print("❌ Ses tanıma modeli yüklenemedi. Yazılı konuşma moduna geçiliyor.")
            interaction_mode = "text"
        
        # TTS modelini yükle (sesli yanıt)
        tts_model = load_tts_model()
        if not tts_model:
            print("❌ TTS modeli yüklenemedi. Sesli yanıtlar devre dışı bırakılacak.")
        else:
            # Ses referans dosyasını al
            voice_reference = get_voice_reference()
    
    # Sohbet mesajlarını başlat
    messages = [{"role": "assistant", "content": "Merhaba, ben TürkTel akıllı asistanı. Size nasıl yardımcı olabilirim?"}]
    
    # Karşılama mesajını göster
    print(f"🤖 Asistan: {messages[0]['content']}")
    
    # Sesli yanıt ver
    if interaction_mode == "voice" and tts_model:
        text_to_speech(tts_model, messages[0]["content"], speaker_wav=voice_reference)
        play_audio_file("response.wav")
    
    # Sohbet döngüsü
    try:
        while True:
            user_input = ""
            
            # Kullanıcı girdisi
            if interaction_mode == "text":
                user_input = input("\n👤 Siz: ")
            else:  # Sesli konuşma modu
                print("\n🎤 Konuşmak için 'k' tuşuna basın, çıkış için 'q' tuşuna basın:")
                key = input().strip().lower()
                if key == 'q':
                    print("\n🤖 Asistan: Görüşmek üzere! İyi günler dilerim.")
                    if tts_model:
                        text_to_speech(tts_model, "Görüşmek üzere! İyi günler dilerim.", speaker_wav=voice_reference)
                        play_audio_file("response.wav")
                    break
                elif key == 'k':
                    audio_file = "user_input.wav"
                    dynamic_audio_record(threshold=10, duration_after_silence=2, output_file=audio_file, show_level=False)
                    user_input = transcribe_audio(whisper_model, audio_file)
                    if not user_input:
                        print("❌ Ses tanıma başarısız oldu. Lütfen tekrar deneyin veya yazılı giriş yapın.")
                        continue
                    print(f"\n👤 Siz (ses tanıma): {user_input}")
                else:
                    continue
            
            if user_input.lower() in ['çıkış', 'exit', 'quit', 'q']:
                print("\n🤖 Asistan: Görüşmek üzere! İyi günler dilerim.")
                if interaction_mode == "voice" and tts_model:
                    text_to_speech(tts_model, "Görüşmek üzere! İyi günler dilerim.", speaker_wav=voice_reference)
                    play_audio_file("response.wav")
                break
            
            # Kullanıcı mesajını kaydet
            messages.append({"role": "user", "content": user_input})
            
            print("\n🔄 Düşünüyorum ve analiz ediyorum...")
            try:
                # Gemini'ye mesaj gönder
                response = chat.send_message(user_input)
                assistant_response = response.text
                
                # Asistan yanıtını kaydet ve göster
                messages.append({"role": "assistant", "content": assistant_response})
                print(f"\n🤖 Asistan: {assistant_response}")
                
                # Sesli yanıt ver
                if interaction_mode == "voice" and tts_model:
                    text_to_speech(tts_model, assistant_response, speaker_wav=voice_reference)
                    play_audio_file("response.wav")
                
            except Exception as e:
                error_message = f"Bir hata oluştu: {e}"
                print(f"\n❌ {error_message}")
                messages.append({"role": "assistant", "content": error_message})
                if interaction_mode == "voice" and tts_model:
                    text_to_speech(tts_model, error_message, speaker_wav=voice_reference)
                    play_audio_file("response.wav")
                
    except KeyboardInterrupt:
        print("\n\n👋 Program sonlandırıldı. İyi günler!")

if __name__ == "__main__":
    main()
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import pickle
import os

# Define the FourTaskBERT model class (needs to be the same as your training script)
class FourTaskBERT(torch.nn.Module):
    def __init__(self, n_category, n_topic, n_emotion, n_urgency):
        super(FourTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained("dbmdz/bert-base-turkish-uncased")
        self.dropout = torch.nn.Dropout(0.3)

        # Separate classifier heads for each task
        self.category_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_category)
        self.topic_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_topic)
        self.emotion_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_emotion)
        self.urgency_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_urgency)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Separate outputs for each task
        category_output = self.category_classifier(pooled_output)
        topic_output = self.topic_classifier(pooled_output)
        emotion_output = self.emotion_classifier(pooled_output)
        urgency_output = self.urgency_classifier(pooled_output)

        return {
            'category': category_output,
            'topic': topic_output,
            'emotion': emotion_output,
            'urgency': urgency_output
        }

# Function to predict using the model
def predict_text(text, model, tokenizer, label_encoders, device, max_len=128):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Get prediction probabilities
    category_probs = torch.nn.functional.softmax(outputs['category'], dim=1)
    topic_probs = torch.nn.functional.softmax(outputs['topic'], dim=1)
    emotion_probs = torch.nn.functional.softmax(outputs['emotion'], dim=1)
    urgency_probs = torch.nn.functional.softmax(outputs['urgency'], dim=1)
    
    # Get the index of the highest probability
    _, category_preds = torch.max(outputs['category'], dim=1)
    _, topic_preds = torch.max(outputs['topic'], dim=1)
    _, emotion_preds = torch.max(outputs['emotion'], dim=1)
    _, urgency_preds = torch.max(outputs['urgency'], dim=1)

    # Convert predictions to labels
    category = label_encoders['category'].inverse_transform(category_preds.cpu().numpy())[0]
    topic = label_encoders['topic'].inverse_transform(topic_preds.cpu().numpy())[0]
    emotion = label_encoders['emotion'].inverse_transform(emotion_preds.cpu().numpy())[0]
    urgency = label_encoders['urgency'].inverse_transform(urgency_preds.cpu().numpy())[0]
    
    # Get prediction confidences
    category_conf = category_probs[0][category_preds].item()
    topic_conf = topic_probs[0][topic_preds].item()
    emotion_conf = emotion_probs[0][emotion_preds].item()
    urgency_conf = urgency_probs[0][urgency_preds].item()

    return {
        'category': category,
        'topic': topic,
        'emotion': emotion,
        'urgency': urgency,
        'confidences': {
            'category': category_conf,
            'topic': topic_conf,
            'emotion': emotion_conf,
            'urgency': urgency_conf
        }
    }

def main():
    print("="*70)
    print("Telekomünikasyon Müşteri Hizmetleri Modeli Test Programı")
    print("="*70)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Çalışma cihazı: {device}")
    
    # Load label encoders
    try:
        print("\nLabel encoder'lar yükleniyor...")
        with open('telekom_label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        print("Label encoder'lar başarıyla yüklendi.")
    except FileNotFoundError:
        print("Hata: telekom_label_encoders.pkl dosyası bulunamadı!")
        return
    
    # Initialize tokenizer
    print("BERT tokenizer yükleniyor...")
    tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
    
    # Get the number of classes
    n_category = len(label_encoders['category'].classes_)
    n_topic = len(label_encoders['topic'].classes_)
    n_emotion = len(label_encoders['emotion'].classes_)
    n_urgency = len(label_encoders['urgency'].classes_)
    
    print(f"Sınıf sayıları: Kategori={n_category}, Konu={n_topic}, Duygu={n_emotion}, Aciliyet={n_urgency}")
    
    # Initialize model
    print("Model oluşturuluyor...")
    model = FourTaskBERT(n_category, n_topic, n_emotion, n_urgency)
    
    # List available model files
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    if not model_files:
        print("Hata: Hiçbir model dosyası (.pth) bulunamadı!")
        return
    
    print("\nKullanılabilir model dosyaları:")
    for i, file in enumerate(model_files):
        print(f"{i+1} - {file}")
    
    try:
        model_choice = int(input("\nHangi model dosyasını yüklemek istersiniz? (Numara girin): "))
        if 1 <= model_choice <= len(model_files):
            model_path = model_files[model_choice-1]
        else:
            print("Geçersiz seçim! İlk model dosyası kullanılacak.")
            model_path = model_files[0]
    except ValueError:
        print("Geçersiz giriş! İlk model dosyası kullanılacak.")
        model_path = model_files[0]
    
    # Load model
    print(f"\n{model_path} model dosyası yükleniyor...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model başarıyla yüklendi!")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return
    
    print("\n" + "="*70)
    print("TEST MODU BAŞLATILDI")
    print("="*70)
    print("Tahmin yapmak için cümleler girin (çıkmak için 'q' yazın)")
    
    # Provide some example test sentences
    example_texts = [
        "internetim çok yavaş",
        "faturamı ödemek istiyorum",
        "telefonum çalındı acil yardım edin",
        "teknik servis ekibinize teşekkür ederim, sorunumu hızlıca çözdünüz",
        "tv kutum bozuldu, kanal geçişlerinde donmalar oluyor",
        "adsl hattımı fibere yükseltmek istiyorum"
    ]
    
    print("\nÖrnek cümleler:")
    for i, text in enumerate(example_texts):
        print(f"{i+1}. {text}")
    
    # Interactive testing loop
    while True:
        print("\n" + "-"*70)
        user_input = input("Test cümlesi (veya örnek numara 1-6) girin (çıkmak için 'q'): ")
        
        if user_input.lower() == 'q':
            print("Test programı sonlandırılıyor...")
            break
        
        # Check if user entered an example number
        try:
            example_num = int(user_input)
            if 1 <= example_num <= len(example_texts):
                user_input = example_texts[example_num-1]
                print(f"Seçilen örnek: \"{user_input}\"")
            else:
                print("Geçersiz örnek numarası. Kendi cümleniz olarak işleniyor.")
        except ValueError:
            pass  # User entered a custom sentence
        
        # Make prediction
        try:
            predictions = predict_text(user_input, model, tokenizer, label_encoders, device)
            
            print("\nTAHMİN SONUÇLARI:")
            print("-"*50)
            print(f"Metin: \"{user_input}\"")
            print("-"*50)
            print(f"Kategori: {predictions['category']} (Güven: {predictions['confidences']['category']:.2%})")
            print(f"Konu: {predictions['topic']} (Güven: {predictions['confidences']['topic']:.2%})")
            print(f"Duygu: {predictions['emotion']} (Güven: {predictions['confidences']['emotion']:.2%})")
            print(f"Aciliyet: {predictions['urgency']} (Güven: {predictions['confidences']['urgency']:.2%})")
        except Exception as e:
            print(f"Tahmin hatası: {e}")

if __name__ == "__main__":
    main()
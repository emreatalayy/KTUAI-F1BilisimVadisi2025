# continue_training.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle
import os
import random

# Reproducibility için seed ayarı
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)

set_seed()

# Veri seti sınıfını yeniden tanımla
class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.iloc[index]['metin'])

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Hedef değişkenleri alın
        kategori = self.data.iloc[index]['kategori_encoded']
        duygu_turu = self.data.iloc[index]['duygu_turu_encoded']
        duygu_yogunlugu = self.data.iloc[index]['duygu_yogunlugu_encoded']
        konu = self.data.iloc[index]['konu_encoded']

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'kategori': torch.tensor(kategori, dtype=torch.long),
            'duygu_turu': torch.tensor(duygu_turu, dtype=torch.long),
            'duygu_yogunlugu': torch.tensor(duygu_yogunlugu, dtype=torch.long),
            'konu': torch.tensor(konu, dtype=torch.long)
        }

# Model sınıfını yeniden tanımla
class MultitaskBERT(torch.nn.Module):
    def __init__(self, n_kategori, n_duygu_turu, n_duygu_yogunlugu, n_konu):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained("dbmdz/bert-base-turkish-uncased")
        self.dropout = torch.nn.Dropout(0.3)

        # Her görev için ayrı sınıflandırıcı başları
        self.kategori_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_kategori)
        self.duygu_turu_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_duygu_turu)
        self.duygu_yogunlugu_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_duygu_yogunlugu)
        self.konu_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_konu)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Her görev için ayrı çıktılar
        kategori_output = self.kategori_classifier(pooled_output)
        duygu_turu_output = self.duygu_turu_classifier(pooled_output)
        duygu_yogunlugu_output = self.duygu_yogunlugu_classifier(pooled_output)
        konu_output = self.konu_classifier(pooled_output)

        return {
            'kategori': kategori_output,
            'duygu_turu': duygu_turu_output,
            'duygu_yogunlugu': duygu_yogunlugu_output,
            'konu': konu_output
        }

# Eğitim fonksiyonu
def train_epoch(model, data_loader, optimizer, device, scheduler, criterion):
    model.train()
    losses = []

    for batch in tqdm(data_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        kategori = batch['kategori'].to(device)
        duygu_turu = batch['duygu_turu'].to(device)
        duygu_yogunlugu = batch['duygu_yogunlugu'].to(device)
        konu = batch['konu'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Her görev için kayıp hesaplayın
        kategori_loss = criterion(outputs['kategori'], kategori)
        duygu_turu_loss = criterion(outputs['duygu_turu'], duygu_turu)
        duygu_yogunlugu_loss = criterion(outputs['duygu_yogunlugu'], duygu_yogunlugu)
        konu_loss = criterion(outputs['konu'], konu)

        # Toplam kayıp
        loss = kategori_loss + duygu_turu_loss + duygu_yogunlugu_loss + konu_loss
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return np.mean(losses)

# Değerlendirme fonksiyonu
def eval_model(model, data_loader, device, criterion):
    model.eval()
    losses = []
    kategori_correct = 0
    duygu_turu_correct = 0
    duygu_yogunlugu_correct = 0
    konu_correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            kategori = batch['kategori'].to(device)
            duygu_turu = batch['duygu_turu'].to(device)
            duygu_yogunlugu = batch['duygu_yogunlugu'].to(device)
            konu = batch['konu'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Her görev için kayıp hesaplayın
            kategori_loss = criterion(outputs['kategori'], kategori)
            duygu_turu_loss = criterion(outputs['duygu_turu'], duygu_turu)
            duygu_yogunlugu_loss = criterion(outputs['duygu_yogunlugu'], duygu_yogunlugu)
            konu_loss = criterion(outputs['konu'], konu)

            # Toplam kayıp
            loss = kategori_loss + duygu_turu_loss + duygu_yogunlugu_loss + konu_loss
            losses.append(loss.item())

            # Tahminleri hesaplayın
            _, kategori_preds = torch.max(outputs['kategori'], dim=1)
            _, duygu_turu_preds = torch.max(outputs['duygu_turu'], dim=1)
            _, duygu_yogunlugu_preds = torch.max(outputs['duygu_yogunlugu'], dim=1)
            _, konu_preds = torch.max(outputs['konu'], dim=1)

            # Doğruluk hesaplayın
            kategori_correct += (kategori_preds == kategori).sum().item()
            duygu_turu_correct += (duygu_turu_preds == duygu_turu).sum().item()
            duygu_yogunlugu_correct += (duygu_yogunlugu_preds == duygu_yogunlugu).sum().item()
            konu_correct += (konu_preds == konu).sum().item()
            total += kategori.size(0)

    # Ortalama kayıp ve doğruluk değerlerini döndürün
    return {
        'loss': np.mean(losses),
        'kategori_acc': kategori_correct / total,
        'duygu_turu_acc': duygu_turu_correct / total,
        'duygu_yogunlugu_acc': duygu_yogunlugu_correct / total,
        'konu_acc': konu_correct / total
    }

def main():
    print("8. Epoch'tan eğitime devam ediliyor...")
    
    # Veri setini yükle
    print("Veri seti yükleniyor...")
    df = pd.read_csv('karistirilmis_veri_seti.csv')
    
    # Label encoder'ları yükle
    print("Label encoder'lar yükleniyor...")
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
        kategori_encoder = encoders['kategori']
        duygu_turu_encoder = encoders['duygu_turu']
        duygu_yogunlugu_encoder = encoders['duygu_yogunlugu']
        konu_encoder = encoders['konu']
    
    # Hedef değişkenleri encode et
    df['kategori_encoded'] = kategori_encoder.transform(df['kategori'])
    df['duygu_turu_encoded'] = duygu_turu_encoder.transform(df['duygu_turu'])
    df['duygu_yogunlugu_encoded'] = duygu_yogunlugu_encoder.transform(df['duygu_yogunlugu'])
    df['konu_encoded'] = konu_encoder.transform(df['konu'])
    
    # Veriyi eğitim ve test olarak böl
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Tokenizer'ı yükle
    print("Tokenizer yükleniyor...")
    tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
    
    # Veri yükleyicileri oluştur
    train_dataset = MultiLabelDataset(train_df, tokenizer)
    test_dataset = MultiLabelDataset(test_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Cihazı ayarla
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    # Sınıf sayılarını al
    n_kategori = len(kategori_encoder.classes_)
    n_duygu_turu = len(duygu_turu_encoder.classes_)
    n_duygu_yogunlugu = len(duygu_yogunlugu_encoder.classes_)
    n_konu = len(konu_encoder.classes_)
    
    # Modeli oluştur
    print("Model oluşturuluyor...")
    model = MultitaskBERT(n_kategori, n_duygu_turu, n_duygu_yogunlugu, n_konu)
    
    # Kaydedilmiş modeli yükle
    model_path = 'best_model_epoch_8.pth'
    print(f"Model yükleniyor: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Optimizer ve criterion
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Eğitim parametreleri
    additional_epochs = 7  # 7 epoch daha ekleyerek toplam 15'e tamamla
    total_steps = len(train_loader) * additional_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Early stopping parametreleri
    patience = 3
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    # Eğitime devam et
    print("Eğitime devam ediliyor...")
    for epoch in range(additional_epochs):
        current_epoch = epoch + 9  # Gerçek epoch numarası (9, 10, 11, ...)
        print(f'Epoch {current_epoch}/15')  # 15 = 8 (önceki) + 7 (yeni)
        
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler, criterion)
        print(f'Train loss: {train_loss}')
        
        # Test seti ile değerlendirme
        eval_results = eval_model(model, test_loader, device, criterion)
        print(f'Validation loss: {eval_results["loss"]}')
        print(f'Kategori accuracy: {eval_results["kategori_acc"]}')
        print(f'Duygu türü accuracy: {eval_results["duygu_turu_acc"]}')
        print(f'Duygu yoğunluğu accuracy: {eval_results["duygu_yogunlugu_acc"]}')
        print(f'Konu accuracy: {eval_results["konu_acc"]}')
        
        # Early stopping kontrolü
        if eval_results['loss'] < best_val_loss:
            best_val_loss = eval_results['loss']
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
            # En iyi modeli kaydet
            torch.save(model.state_dict(), f'best_model_epoch_{current_epoch}.pth')
            print(f"En iyi model kaydedildi! (Epoch {current_epoch})")
        else:
            early_stop_counter += 1
            print(f"Validation loss iyileşmedi. Counter: {early_stop_counter}/{patience}")
        
        if early_stop_counter >= patience:
            print(f"Early stopping! Epoch {current_epoch}")
            break
        
        print('-' * 50)
    
    # En iyi modeli geri yükle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("En iyi model geri yüklendi!")
    
    # Final modeli kaydet
    torch.save(model.state_dict(), 'turkish_bert_multihead_final_model.pth')
    print("Final model kaydedildi: turkish_bert_multihead_final_model.pth")
    
    # Test seti üzerinde son değerlendirme
    final_results = eval_model(model, test_loader, device, criterion)
    print("\nFİNAL TEST SONUÇLARI:")
    print(f'Test loss: {final_results["loss"]}')
    print(f'Kategori accuracy: {final_results["kategori_acc"]}')
    print(f'Duygu türü accuracy: {final_results["duygu_turu_acc"]}')
    print(f'Duygu yoğunluğu accuracy: {final_results["duygu_yogunlugu_acc"]}')
    print(f'Konu accuracy: {final_results["konu_acc"]}')

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Telekom BERT Model Analyzer
Kullanıcı mesajlarını analiz eder ve sonuçları kaydeder
"""

import os
import json
import torch
from transformers import BertTokenizer, BertModel
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelekomBERTAnalyzer:
    """Telekom BERT modelini kullanarak mesaj analizi yapar"""
    
    def __init__(self, model_path: str = None, encoders_path: str = None):
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "telekom_bert_four_task_model.pth")
        self.encoders_path = encoders_path or os.path.join(os.path.dirname(__file__), "telekom_label_encoders.pkl")
        self.analysis_results_path = os.path.join(os.path.dirname(__file__), "analysis_results.json")
        
        self.model = None
        self.tokenizer = None
        self.label_encoders = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model yüklemeyi dene
        self._load_model()
        
    def _load_model(self):
        """BERT modelini ve label encoder'ları yükler"""
        try:
            # Label encoders yükle
            if os.path.exists(self.encoders_path):
                with open(self.encoders_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
                logger.info("Label encoders başarıyla yüklendi")
            else:
                logger.warning(f"Label encoders dosyası bulunamadı: {self.encoders_path}")
                return
            
            # Tokenizer yükle
            self.tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
            logger.info("BERT tokenizer yüklendi")
            
            # Model sınıfını tanımla
            class FourTaskBERT(torch.nn.Module):
                def __init__(self, n_kategori, n_duygu_turu, n_konu, n_oncelik):
                    super(FourTaskBERT, self).__init__()
                    self.bert = BertModel.from_pretrained("dbmdz/bert-base-turkish-uncased")
                    self.dropout = torch.nn.Dropout(0.3)
                    self.kategori_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_kategori)
                    self.duygu_turu_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_duygu_turu)
                    self.konu_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_konu)
                    self.oncelik_classifier = torch.nn.Linear(self.bert.config.hidden_size, n_oncelik)

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
                        'konu': self.konu_classifier(pooled_output),
                        'oncelik': self.oncelik_classifier(pooled_output)
                    }
            
            # Model boyutlarını label encoder'lardan al
            n_kategori = len(self.label_encoders.get('kategori', {}).classes_) if 'kategori' in self.label_encoders else 5
            n_duygu_turu = len(self.label_encoders.get('duygu_turu', {}).classes_) if 'duygu_turu' in self.label_encoders else 3
            n_konu = len(self.label_encoders.get('konu', {}).classes_) if 'konu' in self.label_encoders else 10
            n_oncelik = len(self.label_encoders.get('oncelik', {}).classes_) if 'oncelik' in self.label_encoders else 3
            
            # Model oluştur
            self.model = FourTaskBERT(n_kategori, n_duygu_turu, n_konu, n_oncelik)
            
            # Model ağırlıklarını yükle
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Telekom BERT modeli başarıyla yüklendi: {self.model_path}")
            else:
                logger.warning(f"Model dosyası bulunamadı: {self.model_path}")
                self.model = None
                
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            self.model = None
            self.tokenizer = None
            self.label_encoders = None
    
    def is_available(self) -> bool:
        """Model hazır olup olmadığını kontrol eder"""
        return self.model is not None and self.tokenizer is not None and self.label_encoders is not None
    
    def analyze_message(self, message: str, session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """Kullanıcı mesajını analiz eder"""
        if not self.is_available():
            logger.warning("BERT modeli kullanılamıyor, analiz atlanıyor")
            return {
                "error": "Model kullanılamıyor",
                "timestamp": datetime.now().isoformat(),
                "message": message[:100] + "..." if len(message) > 100 else message
            }
        
        try:
            # Mesajı tokenize et
            encoding = self.tokenizer(
                message,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Model tahminini yap
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
            
            # Sonuçları işle
            results = {}
            for task_name, logits in outputs.items():
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class_idx].item()
                
                # Label encoder'dan sınıf adını al
                if task_name in self.label_encoders:
                    predicted_class = self.label_encoders[task_name].classes_[predicted_class_idx]
                else:
                    predicted_class = f"class_{predicted_class_idx}"
                
                results[task_name] = {
                    "predicted_class": predicted_class,
                    "confidence": round(confidence, 4),
                    "class_index": predicted_class_idx
                }
            
            # Analiz sonucunu hazırla
            analysis_result = {
                "message": message,
                "message_length": len(message),
                "analysis": results,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "user_id": user_id,
                "model_info": {
                    "model_path": os.path.basename(self.model_path),
                    "device": str(self.device)
                }
            }
            
            # Sonucu kaydet
            self._save_analysis_result(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Mesaj analizi hatası: {str(e)}")
            error_result = {
                "error": str(e),
                "message": message[:100] + "..." if len(message) > 100 else message,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "user_id": user_id
            }
            self._save_analysis_result(error_result)
            return error_result
    
    def _save_analysis_result(self, result: Dict[str, Any]):
        """Analiz sonucunu JSON dosyasına kaydeder"""
        try:
            # Mevcut sonuçları yükle
            if os.path.exists(self.analysis_results_path):
                with open(self.analysis_results_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            else:
                existing_results = []
            
            # Yeni sonucu ekle
            existing_results.append(result)
            
            # Sonuçları 1000 ile sınırla (dosya boyutunu kontrol et)
            if len(existing_results) > 1000:
                existing_results = existing_results[-1000:]
            
            # Dosyaya kaydet
            with open(self.analysis_results_path, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Analiz sonucu kaydedildi: {os.path.basename(self.analysis_results_path)}")
            
        except Exception as e:
            logger.error(f"Analiz sonucu kaydetme hatası: {str(e)}")
    
    def get_analysis_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Kayıtlı analiz geçmişini getirir"""
        try:
            if not os.path.exists(self.analysis_results_path):
                return []
            
            with open(self.analysis_results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Son N sonucu döndür
            return results[-limit:] if len(results) > limit else results
            
        except Exception as e:
            logger.error(f"Analiz geçmişi okuma hatası: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Analiz istatistiklerini hesaplar"""
        try:
            results = self.get_analysis_history(limit=1000)
            
            if not results:
                return {"message": "Henüz analiz verisi yok"}
            
            # Başarılı analizleri filtrele
            successful_results = [r for r in results if "analysis" in r]
            
            stats = {
                "total_analyses": len(results),
                "successful_analyses": len(successful_results),
                "error_count": len(results) - len(successful_results),
                "date_range": {
                    "first": results[0]["timestamp"] if results else None,
                    "last": results[-1]["timestamp"] if results else None
                }
            }
            
            if successful_results:
                # Task bazında istatistikler
                for task_name in ["kategori", "duygu_turu", "konu", "oncelik"]:
                    task_results = []
                    for result in successful_results:
                        if "analysis" in result and task_name in result["analysis"]:
                            task_results.append(result["analysis"][task_name])
                    
                    if task_results:
                        # En sık görülen sınıflar
                        class_counts = {}
                        confidence_scores = []
                        
                        for tr in task_results:
                            predicted_class = tr.get("predicted_class", "unknown")
                            confidence = tr.get("confidence", 0)
                            
                            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
                            confidence_scores.append(confidence)
                        
                        # Sıralı sınıf sayıları
                        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                        
                        stats[f"{task_name}_stats"] = {
                            "total_predictions": len(task_results),
                            "top_classes": sorted_classes[:5],
                            "avg_confidence": round(sum(confidence_scores) / len(confidence_scores), 4) if confidence_scores else 0,
                            "min_confidence": round(min(confidence_scores), 4) if confidence_scores else 0,
                            "max_confidence": round(max(confidence_scores), 4) if confidence_scores else 0
                        }
            
            return stats
            
        except Exception as e:
            logger.error(f"İstatistik hesaplama hatası: {str(e)}")
            return {"error": str(e)}

# Global analyzer instance
_analyzer_instance = None

def get_analyzer() -> TelekomBERTAnalyzer:
    """Global analyzer instance'ını döndürür"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = TelekomBERTAnalyzer()
    return _analyzer_instance

def analyze_user_message(message: str, session_id: str = None, user_id: str = None) -> Optional[Dict[str, Any]]:
    """Kullanıcı mesajını analiz eder (convenience function)"""
    analyzer = get_analyzer()
    if analyzer.is_available():
        return analyzer.analyze_message(message, session_id, user_id)
    return None

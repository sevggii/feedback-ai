"""
Veri ön-işleme modülü.
Sosyal medya yorumlarını temizler, tokenize eder ve model için hazırlar.
"""

import json
import re
import random
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from collections import Counter

from ..utils.config import config
from ..utils.logging import get_logger

logger = get_logger("preprocess")


class TextPreprocessor:
    """Metin ön-işleme sınıfı."""
    
    def __init__(self, model_name: str = "xlm-roberta-base"):
        """
        Preprocessor'ı başlat.
        
        Args:
            model_name: Tokenizer model adı
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Dil tespiti için basit heuristikler
        self.turkish_words = {
            'çok', 'çok', 'güzel', 'harika', 'mükemmel', 'süper', 'iyi', 'kötü',
            'fena', 'berbat', 'harika', 'muhteşem', 'olağanüstü', 'müthiş',
            'teşekkür', 'sağol', 'merhaba', 'selam', 'nasıl', 'ne', 'kim',
            'nerede', 'ne zaman', 'neden', 'niçin', 'nasılsın', 'iyi misin'
        }
        
        self.english_words = {
            'great', 'awesome', 'amazing', 'wonderful', 'fantastic', 'excellent',
            'good', 'bad', 'terrible', 'awful', 'horrible', 'perfect', 'super',
            'thanks', 'thank you', 'hello', 'hi', 'how', 'what', 'who', 'where',
            'when', 'why', 'how are you', 'are you ok'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Metni temizle.
        
        Args:
            text: Ham metin
            
        Returns:
            Temizlenmiş metin
        """
        # URL'leri normalize et
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                     '[URL]', text)
        
        # Mention'ları normalize et
        text = re.sub(r'@\w+', '[MENTION]', text)
        
        # Hashtag'leri koru ama normalize et
        text = re.sub(r'#(\w+)', r'#\1', text)
        
        # Tekrar karakterleri sıkıştır (cooool -> cool)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        
        # Başta ve sonda boşlukları temizle
        text = text.strip()
        
        return text
    
    def detect_language(self, text: str) -> str:
        """
        Dil tespiti yap (basit heuristik).
        
        Args:
            text: Metin
            
        Returns:
            Dil kodu ('tr', 'en', 'mixed')
        """
        text_lower = text.lower()
        words = set(text_lower.split())
        
        turkish_count = len(words.intersection(self.turkish_words))
        english_count = len(words.intersection(self.english_words))
        
        if turkish_count > english_count and turkish_count > 0:
            return 'tr'
        elif english_count > turkish_count and english_count > 0:
            return 'en'
        else:
            return 'mixed'
    
    def tokenize_text(self, text: str, max_length: int = 160) -> Dict[str, Any]:
        """
        Metni tokenize et.
        
        Args:
            text: Metin
            max_length: Maksimum uzunluk
            
        Returns:
            Tokenize edilmiş veri
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
    
    def preprocess_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tek bir öğeyi ön-işle.
        
        Args:
            item: Ham veri öğesi
            
        Returns:
            Ön-işlenmiş veri öğesi
        """
        # Metni temizle
        cleaned_text = self.clean_text(item['text'])
        
        # Dil tespiti
        detected_lang = self.detect_language(cleaned_text)
        
        # Tokenize et
        tokenized = self.tokenize_text(cleaned_text, config.MAX_LENGTH)
        
        return {
            'text': cleaned_text,
            'original_text': item['text'],
            'lang': detected_lang,
            'meta': item.get('meta', {}),
            'labels': item['labels'],
            'input_ids': tokenized['input_ids'].tolist(),
            'attention_mask': tokenized['attention_mask'].tolist()
        }


def load_data(file_path: Path) -> List[Dict[str, Any]]:
    """
    JSONL dosyasından veri yükle.
    
    Args:
        file_path: Dosya yolu
        
    Returns:
        Veri listesi
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    logger.info(f"✅ {len(data)} örnek yüklendi: {file_path}")
    return data


def calculate_class_weights(data: List[Dict[str, Any]], task: str) -> List[float]:
    """
    Sınıf ağırlıklarını hesapla.
    
    Args:
        data: Veri listesi
        task: Görev ('sentiment' veya 'authenticity')
        
    Returns:
        Sınıf ağırlıkları
    """
    labels = [item['labels'][task] for item in data]
    label_counts = Counter(labels)
    
    total_samples = len(data)
    num_classes = len(label_counts)
    
    weights = []
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    logger.info(f"{task} sınıf ağırlıkları: {weights}")
    return weights


def split_data(
    data: List[Dict[str, Any]], 
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Veriyi train/val/test olarak böl.
    
    Args:
        data: Veri listesi
        train_split: Train oranı
        val_split: Validation oranı
        test_split: Test oranı
        random_seed: Rastgele seed
        
    Returns:
        (train_data, val_data, test_data)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Split toplamı 1.0 olmalı"
    
    # İlk olarak train ve geçici olarak ayır
    train_data, temp_data = train_test_split(
        data, 
        test_size=(val_split + test_split),
        random_state=random_seed,
        stratify=[item['labels']['authenticity'] for item in data]
    )
    
    # Geçici veriyi val ve test olarak ayır
    val_size = val_split / (val_split + test_split)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_size),
        random_state=random_seed,
        stratify=[item['labels']['authenticity'] for item in temp_data]
    )
    
    logger.info(f"📊 Veri bölünmesi:")
    logger.info(f"  Train: {len(train_data)} (%{len(train_data)/len(data)*100:.1f})")
    logger.info(f"  Val: {len(val_data)} (%{len(val_data)/len(data)*100:.1f})")
    logger.info(f"  Test: {len(test_data)} (%{len(test_data)/len(data)*100:.1f})")
    
    return train_data, val_data, test_data


def preprocess_data(
    input_file: Path,
    output_dir: Path,
    model_name: str = "xlm-roberta-base"
) -> None:
    """
    Veriyi ön-işle ve kaydet.
    
    Args:
        input_file: Giriş dosyası
        output_dir: Çıkış dizini
        model_name: Model adı
    """
    logger.info("🚀 Veri ön-işleme başlıyor...")
    
    # Veriyi yükle
    raw_data = load_data(input_file)
    
    # Preprocessor'ı başlat
    preprocessor = TextPreprocessor(model_name)
    
    # Veriyi ön-işle
    processed_data = []
    for item in raw_data:
        try:
            processed_item = preprocessor.preprocess_single(item)
            processed_data.append(processed_item)
        except Exception as e:
            logger.warning(f"Öğe işlenemedi: {item.get('text', '')[:50]}... - {e}")
    
    logger.info(f"✅ {len(processed_data)} öğe ön-işlendi")
    
    # Veriyi böl
    train_data, val_data, test_data = split_data(
        processed_data,
        config.TRAIN_SPLIT,
        config.VAL_SPLIT,
        config.TEST_SPLIT,
        config.RANDOM_SEED
    )
    
    # Çıkış dizinini oluştur
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Veriyi kaydet
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_file = output_dir / f"{split_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"💾 {split_name} verisi kaydedildi: {output_file}")
    
    # Sınıf ağırlıklarını hesapla ve kaydet
    class_weights = {
        'sentiment': calculate_class_weights(processed_data, 'sentiment'),
        'authenticity': calculate_class_weights(processed_data, 'authenticity')
    }
    
    weights_file = output_dir / "class_weights.json"
    with open(weights_file, 'w', encoding='utf-8') as f:
        json.dump(class_weights, f, ensure_ascii=False, indent=2)
    
    logger.info(f"⚖️ Sınıf ağırlıkları kaydedildi: {weights_file}")
    
    # İstatistikleri kaydet
    stats = {
        'total_samples': len(processed_data),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'model_name': model_name,
        'max_length': config.MAX_LENGTH,
        'class_weights': class_weights
    }
    
    stats_file = output_dir / "preprocessing_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📈 İstatistikler kaydedildi: {stats_file}")
    logger.info("✅ Veri ön-işleme tamamlandı!")


if __name__ == "__main__":
    # Örnek kullanım
    input_file = Path("src/data/samples/comments_demo.jsonl")
    output_dir = Path("src/data/processed")
    
    preprocess_data(input_file, output_dir, config.MODEL_NAME)


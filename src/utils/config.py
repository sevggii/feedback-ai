"""
Konfigürasyon yönetimi modülü.
Environment değişkenlerini yükler ve uygulama ayarlarını sağlar.
"""

import os
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()


class Config:
    """Uygulama konfigürasyonu."""
    
    # Model ayarları
    MODEL_NAME: str = os.getenv("MODEL_NAME", "xlm-roberta-base")
    MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "160"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "2e-5"))
    EPOCHS: int = int(os.getenv("EPOCHS", "5"))
    WARMUP_STEPS: int = int(os.getenv("WARMUP_STEPS", "100"))
    
    # Veri yolları
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "src/data"))
    RAW_DATA_DIR: Path = Path(os.getenv("RAW_DATA_DIR", "src/data/raw"))
    PROCESSED_DATA_DIR: Path = Path(os.getenv("PROCESSED_DATA_DIR", "src/data/processed"))
    SAMPLES_DIR: Path = Path(os.getenv("SAMPLES_DIR", "src/data/samples"))
    
    # Eğitim ayarları
    TRAIN_SPLIT: float = float(os.getenv("TRAIN_SPLIT", "0.8"))
    VAL_SPLIT: float = float(os.getenv("VAL_SPLIT", "0.1"))
    TEST_SPLIT: float = float(os.getenv("TEST_SPLIT", "0.1"))
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    
    # Kayıp ağırlıkları
    SENTIMENT_WEIGHT: float = float(os.getenv("SENTIMENT_WEIGHT", "0.4"))
    AUTHENTICITY_WEIGHT: float = float(os.getenv("AUTHENTICITY_WEIGHT", "0.6"))
    
    # API ayarları
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # Firebase ayarları
    FIREBASE_CREDENTIALS_PATH: Optional[str] = os.getenv("FIREBASE_CREDENTIALS_PATH")
    FIREBASE_PROJECT_ID: Optional[str] = os.getenv("FIREBASE_PROJECT_ID")
    FIREBASE_COLLECTION: str = os.getenv("FIREBASE_COLLECTION", "feedback_logs")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/social_feedback_ai.log")
    
    # Güvenlik
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
    CORS_ORIGINS: List[str] = eval(os.getenv("CORS_ORIGINS", '["http://localhost:3000", "http://localhost:5173"]'))
    
    # Model export
    EXPORT_FORMAT: str = os.getenv("EXPORT_FORMAT", "onnx")
    MODEL_EXPORT_DIR: Path = Path(os.getenv("MODEL_EXPORT_DIR", "models/exported"))
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Gerekli dizinleri oluştur."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.SAMPLES_DIR,
            cls.MODEL_EXPORT_DIR,
            Path("logs"),
            Path("models"),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls) -> None:
        """Konfigürasyonu doğrula."""
        assert cls.TRAIN_SPLIT + cls.VAL_SPLIT + cls.TEST_SPLIT == 1.0, "Split toplamı 1.0 olmalı"
        assert cls.SENTIMENT_WEIGHT + cls.AUTHENTICITY_WEIGHT == 1.0, "Ağırlık toplamı 1.0 olmalı"
        assert cls.MAX_LENGTH > 0, "MAX_LENGTH pozitif olmalı"
        assert cls.BATCH_SIZE > 0, "BATCH_SIZE pozitif olmalı"
        assert cls.LEARNING_RATE > 0, "LEARNING_RATE pozitif olmalı"


# Global config instance
config = Config()
config.ensure_directories()
config.validate()

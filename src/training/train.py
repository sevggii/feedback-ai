"""
Model eğitimi modülü.
Multi-task learning ile sentiment ve authenticity sınıflandırması.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    get_linear_schedule_with_warmup
)
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import config
from ..utils.logging import get_logger
from ..utils.metrics import calculate_metrics, plot_confusion_matrix

logger = get_logger("training")


class SocialFeedbackDataset(Dataset):
    """Sosyal medya yorumları için dataset."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        """
        Dataset'i başlat.
        
        Args:
            data: Ön-işlenmiş veri listesi
        """
        self.data = data
        
        # Label mapping
        self.sentiment_labels = ['neg', 'neu', 'pos']
        self.authenticity_labels = ['spam', 'support', 'genuine']
        
        self.sentiment_to_id = {label: i for i, label in enumerate(self.sentiment_labels)}
        self.authenticity_to_id = {label: i for i, label in enumerate(self.authenticity_labels)}
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'sentiment_label': torch.tensor(
                self.sentiment_to_id[item['labels']['sentiment']], 
                dtype=torch.long
            ),
            'authenticity_label': torch.tensor(
                self.authenticity_to_id[item['labels']['authenticity']], 
                dtype=torch.long
            )
        }


class MultiTaskModel(nn.Module):
    """Multi-task model: sentiment + authenticity."""
    
    def __init__(
        self, 
        model_name: str = "xlm-roberta-base",
        num_sentiment_classes: int = 3,
        num_authenticity_classes: int = 3,
        dropout_rate: float = 0.1
    ):
        """
        Model'i başlat.
        
        Args:
            model_name: Base model adı
            num_sentiment_classes: Sentiment sınıf sayısı
            num_authenticity_classes: Authenticity sınıf sayısı
            dropout_rate: Dropout oranı
        """
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification heads
        self.sentiment_classifier = nn.Linear(
            self.config.hidden_size, 
            num_sentiment_classes
        )
        self.authenticity_classifier = nn.Linear(
            self.config.hidden_size, 
            num_authenticity_classes
        )
        
        # Loss functions
        self.sentiment_criterion = nn.CrossEntropyLoss()
        self.authenticity_criterion = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        sentiment_labels: Optional[torch.Tensor] = None,
        authenticity_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            sentiment_labels: Sentiment etiketleri (opsiyonel)
            authenticity_labels: Authenticity etiketleri (opsiyonel)
            
        Returns:
            Model çıktıları
        """
        # Base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pooled output (CLS token)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Classification heads
        sentiment_logits = self.sentiment_classifier(pooled_output)
        authenticity_logits = self.authenticity_classifier(pooled_output)
        
        result = {
            'sentiment_logits': sentiment_logits,
            'authenticity_logits': authenticity_logits
        }
        
        # Loss hesapla (training sırasında)
        if sentiment_labels is not None and authenticity_labels is not None:
            sentiment_loss = self.sentiment_criterion(sentiment_logits, sentiment_labels)
            authenticity_loss = self.authenticity_criterion(authenticity_logits, authenticity_labels)
            
            # Weighted loss
            total_loss = (
                config.SENTIMENT_WEIGHT * sentiment_loss + 
                config.AUTHENTICITY_WEIGHT * authenticity_loss
            )
            
            result.update({
                'loss': total_loss,
                'sentiment_loss': sentiment_loss,
                'authenticity_loss': authenticity_loss
            })
        
        return result


class Trainer:
    """Model eğitici sınıfı."""
    
    def __init__(
        self,
        model: MultiTaskModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ):
        """
        Trainer'ı başlat.
        
        Args:
            model: Eğitilecek model
            train_loader: Train data loader
            val_loader: Validation data loader
            device: Cihaz (CPU/GPU)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.01
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_sentiment_f1': [],
            'val_authenticity_f1': []
        }
        
        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_model_state = None
    
    def train_epoch(self) -> float:
        """Tek epoch eğitimi."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            # Batch'i device'a taşı
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                sentiment_labels=batch['sentiment_label'],
                authenticity_labels=batch['authenticity_label']
            )
            
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Progress bar güncelle
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Dict[str, float]:
        """Validation."""
        self.model.eval()
        total_loss = 0.0
        
        all_sentiment_preds = []
        all_sentiment_labels = []
        all_authenticity_preds = []
        all_authenticity_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    sentiment_labels=batch['sentiment_label'],
                    authenticity_labels=batch['authenticity_label']
                )
                
                total_loss += outputs['loss'].item()
                
                # Predictions
                sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=1)
                authenticity_preds = torch.argmax(outputs['authenticity_logits'], dim=1)
                
                all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
                all_sentiment_labels.extend(batch['sentiment_label'].cpu().numpy())
                all_authenticity_preds.extend(authenticity_preds.cpu().numpy())
                all_authenticity_labels.extend(batch['authenticity_label'].cpu().numpy())
        
        # Metrikleri hesapla
        avg_loss = total_loss / len(self.val_loader)
        sentiment_f1 = f1_score(all_sentiment_labels, all_sentiment_preds, average='macro')
        authenticity_f1 = f1_score(all_authenticity_labels, all_authenticity_preds, average='macro')
        
        return {
            'loss': avg_loss,
            'sentiment_f1': sentiment_f1,
            'authenticity_f1': authenticity_f1
        }
    
    def train(self, epochs: int = 5) -> None:
        """Model eğitimi."""
        logger.info(f"🚀 Model eğitimi başlıyor ({epochs} epoch)...")
        
        for epoch in range(epochs):
            logger.info(f"\n📅 Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # History'yi güncelle
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_sentiment_f1'].append(val_metrics['sentiment_f1'])
            self.history['val_authenticity_f1'].append(val_metrics['authenticity_f1'])
            
            # Log
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Sentiment F1: {val_metrics['sentiment_f1']:.4f}")
            logger.info(f"Val Authenticity F1: {val_metrics['authenticity_f1']:.4f}")
            
            # Best model'i kaydet
            current_f1 = val_metrics['authenticity_f1']  # Authenticity F1'e göre kaydet
            if current_f1 > self.best_val_f1:
                self.best_val_f1 = current_f1
                self.best_model_state = self.model.state_dict().copy()
                logger.info(f"🎯 Yeni en iyi model! Authenticity F1: {current_f1:.4f}")
        
        # En iyi model'i yükle
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"✅ En iyi model yüklendi (Authenticity F1: {self.best_val_f1:.4f})")
    
    def save_model(self, save_path: Path) -> None:
        """Model'i kaydet."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Model state dict
        torch.save(self.model.state_dict(), save_path / "model.pt")
        
        # Model config
        config_dict = {
            'model_name': config.MODEL_NAME,
            'num_sentiment_classes': 3,
            'num_authenticity_classes': 3,
            'dropout_rate': 0.1,
            'best_val_f1': self.best_val_f1
        }
        
        with open(save_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Training history
        with open(save_path / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"💾 Model kaydedildi: {save_path}")
    
    def plot_training_history(self, save_path: Path) -> None:
        """Eğitim geçmişini çiz."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Sentiment F1
        axes[0, 1].plot(self.history['val_sentiment_f1'], label='Validation')
        axes[0, 1].set_title('Sentiment F1')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Authenticity F1
        axes[1, 0].plot(self.history['val_authenticity_f1'], label='Validation')
        axes[1, 0].set_title('Authenticity F1')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined F1
        combined_f1 = [
            (s + a) / 2 for s, a in zip(
                self.history['val_sentiment_f1'],
                self.history['val_authenticity_f1']
            )
        ]
        axes[1, 1].plot(combined_f1, label='Combined')
        axes[1, 1].set_title('Combined F1')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Training history plot kaydedildi: {save_path / 'training_history.png'}")


def load_processed_data(data_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Ön-işlenmiş veriyi yükle."""
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    test_file = data_dir / "test.jsonl"
    
    def load_jsonl(file_path: Path) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)
    test_data = load_jsonl(test_file)
    
    logger.info(f"📊 Veri yüklendi:")
    logger.info(f"  Train: {len(train_data)}")
    logger.info(f"  Val: {len(val_data)}")
    logger.info(f"  Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def train_model(
    data_dir: Path = Path("src/data/processed"),
    model_save_dir: Path = Path("models/trained"),
    epochs: int = 5
) -> None:
    """Model eğitimi ana fonksiyonu."""
    logger.info("🚀 Model eğitimi başlıyor...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ Device: {device}")
    
    # Veriyi yükle
    train_data, val_data, test_data = load_processed_data(data_dir)
    
    # Datasets
    train_dataset = SocialFeedbackDataset(train_data)
    val_dataset = SocialFeedbackDataset(val_data)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0  # macOS'ta 0 kullan
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Model
    model = MultiTaskModel(
        model_name=config.MODEL_NAME,
        num_sentiment_classes=3,
        num_authenticity_classes=3
    )
    
    logger.info(f"🤖 Model oluşturuldu: {config.MODEL_NAME}")
    logger.info(f"📊 Parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # Eğitim
    trainer.train(epochs)
    
    # Model'i kaydet
    trainer.save_model(model_save_dir)
    
    # Training history plot
    trainer.plot_training_history(model_save_dir)
    
    logger.info("✅ Model eğitimi tamamlandı!")


if __name__ == "__main__":
    # Model eğitimi
    train_model(
        data_dir=Path("src/data/processed"),
        model_save_dir=Path("models/trained"),
        epochs=config.EPOCHS
    )


"""
Metrik hesaplama ve değerlendirme yardımcıları.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .logging import get_logger

logger = get_logger("metrics")


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str],
    task_name: str = "classification"
) -> Dict[str, Any]:
    """
    Sınıflandırma metriklerini hesapla.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        labels: Sınıf isimleri
        task_name: Görev adı
        
    Returns:
        Metrikler sözlüğü
    """
    # Temel metrikler
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(labels))
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Karışıklık matrisi
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    
    metrics = {
        "task": task_name,
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }
    
    logger.info(f"{task_name} - Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    save_path: Optional[Path] = None
) -> None:
    """
    Karışıklık matrisini çiz.
    
    Args:
        cm: Karışıklık matrisi
        labels: Sınıf isimleri
        title: Grafik başlığı
        save_path: Kaydetme yolu (opsiyonel)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def calculate_class_weights(y: List[int], num_classes: int) -> List[float]:
    """
    Sınıf ağırlıklarını hesapla.
    
    Args:
        y: Etiketler
        num_classes: Sınıf sayısı
        
    Returns:
        Sınıf ağırlıkları
    """
    from collections import Counter
    
    class_counts = Counter(y)
    total_samples = len(y)
    
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)  # En az 1 örnek varsay
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    logger.info(f"Class weights: {weights}")
    return weights


def calculate_confidence_intervals(
    scores: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Güven aralıklarını hesapla.
    
    Args:
        scores: Skorlar
        confidence: Güven seviyesi
        
    Returns:
        (mean, lower_bound, upper_bound)
    """
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    n = len(scores)
    
    # t-distribution için kritik değer (basitleştirilmiş)
    from scipy import stats
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    
    margin_error = t_critical * (std_score / np.sqrt(n))
    
    lower_bound = mean_score - margin_error
    upper_bound = mean_score + margin_error
    
    return mean_score, lower_bound, upper_bound


def format_classification_report(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str]
) -> str:
    """
    Sınıflandırma raporunu formatla.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        labels: Sınıf isimleri
        
    Returns:
        Formatlanmış rapor
    """
    report = classification_report(
        y_true, y_pred,
        target_names=labels,
        digits=4
    )
    return report

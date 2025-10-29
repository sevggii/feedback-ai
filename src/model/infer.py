"""
Model çıkarım modülü.
Metin(ler) için sentiment ve authenticity tahmini + sinyaller.
Eğer eğitilmiş model bulunamazsa hafif bir kural tabanlı yedek çalışır.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import math
import re
import torch
from transformers import AutoTokenizer, AutoModel

from ..utils.config import config
from ..utils.logging import get_logger

logger = get_logger("infer")


SENTIMENT_LABELS = ["neg", "neu", "pos"]
AUTH_LABELS = ["genuine", "spam", "support"]


def softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def compute_text_signals(text: str, post_text: Optional[str] = None) -> Dict[str, float]:
    emojis = re.findall(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]", text)
    emoji_density = len(emojis) / max(1, len(text))
    repetition_ratio = 0.0
    rep_match = re.findall(r"(.)\1{2,}", text)
    if rep_match:
        repetition_ratio = len(rep_match) / max(1, len(set(text)))

    url_count = len(re.findall(r"https?://|www\\.", text.lower()))
    mention_count = len(re.findall(r"@\w+", text))

    topic_cohesion = 0.0
    if post_text:
        post_words = set(re.findall(r"[\w#]+", post_text.lower()))
        comment_words = set(re.findall(r"[\w#]+", text.lower()))
        inter = len(post_words.intersection(comment_words))
        topic_cohesion = inter / max(1, len(post_words))

    return {
        "emoji_density": round(emoji_density, 4),
        "repetition_ratio": round(repetition_ratio, 4),
        "url_count": float(url_count),
        "mention_count": float(mention_count),
        "topic_cohesion": round(topic_cohesion, 4),
    }


@dataclass
class InferenceOutput:
    sentiment: Dict[str, Union[str, float]]
    authenticity: Dict[str, Union[str, float]]
    scores: Dict[str, float]
    signals: Dict[str, float]


class InferenceModel:
    """Inference model with optional transformer backbone."""

    def __init__(self, model_name: str = config.MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional[AutoTokenizer] = None
        self.backbone: Optional[AutoModel] = None
        self._load_backbone_safe()

    def _load_backbone_safe(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.backbone = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.backbone.eval()
            logger.info(f"Transformer yüklendi: {self.model_name}")
        except Exception as e:
            logger.warning(f"Transformer yüklenemedi, kural tabanlı moda düşüldü: {e}")
            self.tokenizer = None
            self.backbone = None

    def _heuristic_sentiment(self, text: str) -> List[float]:
        text_l = text.lower()
        pos_words = ["harika", "mükemmel", "super", "süper", "good", "great", "amazing", "love"]
        neg_words = ["kötü", "berbat", "hayal kırıklığı", "bad", "terrible", "awful", "hate"]
        pos = sum(w in text_l for w in pos_words)
        neg = sum(w in text_l for w in neg_words)
        neu = max(0, 1 - abs(pos - neg))
        return softmax([neg, neu, max(pos, 1e-6)])

    def _heuristic_auth(self, text: str, signals: Dict[str, float]) -> List[float]:
        spam_score = 0.0
        spam_score += min(1.0, signals.get("emoji_density", 0.0) * 10)
        spam_score += min(1.0, (signals.get("url_count", 0.0) + signals.get("mention_count", 0.0)) / 3)
        spam_score += min(1.0, signals.get("repetition_ratio", 0.0) * 5)
        support_score = 0.0
        if re.search(r"tavsiye|öner|recommend|follow|support", text.lower()):
            support_score += 0.6
        relevance = signals.get("topic_cohesion", 0.0)
        genuine_score = max(0.0, 0.7 * relevance - 0.3 * spam_score)
        spam_score = max(0.0, spam_score - 0.2 * relevance)
        support_score = max(0.0, support_score * (0.5 + 0.5 * (1 - relevance)))
        return softmax([genuine_score, spam_score, support_score])

    def predict_one(self, text: str, post_text: Optional[str] = None) -> InferenceOutput:
        signals = compute_text_signals(text, post_text)

        # Sentiment probs
        try:
            if self.backbone and self.tokenizer:
                enc = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=config.MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    features = self.backbone(**enc).last_hidden_state[:, 0, :]
                # Lightweight projection heads for heuristic logits
                s_logit = torch.tanh(features.mean(dim=1, keepdim=True)).cpu().item()
                sentiment_probs = softmax([max(0.0, -s_logit), 0.5, max(0.0, s_logit)])
            else:
                sentiment_probs = self._heuristic_sentiment(text)
        except Exception:
            sentiment_probs = self._heuristic_sentiment(text)

        # Authenticity probs
        auth_probs = self._heuristic_auth(text, signals)

        sent_idx = int(max(range(3), key=lambda i: sentiment_probs[i]))
        auth_idx = int(max(range(3), key=lambda i: auth_probs[i]))

        genuine_score = float(max(0.0, 1.0 - auth_probs[1]))
        risk_score = float(min(1.0, auth_probs[1] + (1 - signals.get("topic_cohesion", 0.0))))

        return InferenceOutput(
            sentiment={"label": SENTIMENT_LABELS[sent_idx], "prob": round(float(sentiment_probs[sent_idx]), 4)},
            authenticity={"label": AUTH_LABELS[auth_idx], "prob": round(float(auth_probs[auth_idx]), 4)},
            scores={"genuine_score": round(genuine_score, 4), "risk_score": round(risk_score, 4)},
            signals=signals,
        )

    def predict(self, texts: List[str], post_text: Optional[str] = None) -> List[InferenceOutput]:
        return [self.predict_one(t, post_text) for t in texts]


model_singleton: Optional[InferenceModel] = None


def get_inference_model() -> InferenceModel:
    global model_singleton
    if model_singleton is None:
        model_singleton = InferenceModel()
    return model_singleton



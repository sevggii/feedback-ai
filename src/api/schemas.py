from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl


class AnalyzeOneRequest(BaseModel):
    text: str = Field(..., description="Tek yorum/metin")
    post_text: Optional[str] = Field(None, description="Yorumun ait olduğu gönderi metni")


class AnalyzeBatchRequest(BaseModel):
    comments: List[str] = Field(..., description="Yorum listesi")
    post_text: Optional[str] = None


class AnalyzeUrlRequest(BaseModel):
    url: HttpUrl
    sample_comments: Optional[List[str]] = Field(None, description="Varsa örnek yorumlar")


class Prediction(BaseModel):
    label: str
    prob: float


class AnalyzeResult(BaseModel):
    sentiment: Prediction
    authenticity: Prediction
    scores: Dict[str, float]
    signals: Dict[str, float]


class AnalyzeResponse(BaseModel):
    results: List[AnalyzeResult]
    post_text: Optional[str] = None
    meta: Dict[str, Any] = {}



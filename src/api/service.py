from __future__ import annotations

from typing import List, Dict, Any

from ..model.infer import get_inference_model


def analyze_texts(comments: List[str], post_text: str | None = None) -> List[Dict[str, Any]]:
    model = get_inference_model()
    outputs = model.predict(comments, post_text)
    results: List[Dict[str, Any]] = []
    for out in outputs:
        results.append(
            {
                "sentiment": out.sentiment,
                "authenticity": out.authenticity,
                "scores": out.scores,
                "signals": out.signals,
            }
        )
    return results



from __future__ import annotations

import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    AnalyzeOneRequest, AnalyzeBatchRequest, AnalyzeUrlRequest,
    AnalyzeResponse, AnalyzeResult
)
from .utils import rate_limit
from ..utils.config import config
from ..utils.logging import get_logger
from .service import analyze_texts
from ..integrations.scraping import fetch_post_context, fetch_instagram_comments

logger = get_logger("api")

app = FastAPI(title="social-feedback-ai", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/analyze_one", response_model=AnalyzeResult)
async def analyze_one(req: AnalyzeOneRequest, request: Request):
    rate_limit(request)
    results = analyze_texts([req.text], req.post_text)
    return AnalyzeResult(**results[0])


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeBatchRequest, request: Request):
    rate_limit(request)
    results = analyze_texts(req.comments, req.post_text)
    return AnalyzeResponse(results=results, post_text=req.post_text, meta={})


@app.post("/analyze_url", response_model=AnalyzeResponse)
async def analyze_url(req: AnalyzeUrlRequest, request: Request):
    rate_limit(request)
    ctx = await fetch_post_context(str(req.url))
    post_text = ctx.get("post_text")
    comments = req.sample_comments or []
    if not comments:
        # Otomatik yorum çek (best-effort)
        comments = await fetch_instagram_comments(str(req.url), max_comments=50)
    results = analyze_texts(comments if comments else [post_text or ""], post_text)
    return AnalyzeResponse(results=results, post_text=post_text, meta={"platform": ctx.get("platform"), **ctx.get("meta", {})})



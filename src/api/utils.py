from __future__ import annotations

import time
from typing import Dict
from fastapi import Request, HTTPException

from ..utils.config import config

_rate_store: Dict[str, list[float]] = {}


def get_client_ip(request: Request) -> str:
    ip = request.headers.get("x-forwarded-for")
    if ip:
        return ip.split(",")[0].strip()
    client_host = request.client.host if request.client else "unknown"
    return client_host


def rate_limit(request: Request) -> None:
    max_per_min = config.RATE_LIMIT_PER_MINUTE
    ip = get_client_ip(request)
    now = time.time()
    window_start = now - 60
    history = _rate_store.setdefault(ip, [])
    # Eski kayıtları temizle
    _rate_store[ip] = [t for t in history if t >= window_start]
    if len(_rate_store[ip]) >= max_per_min:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    _rate_store[ip].append(now)



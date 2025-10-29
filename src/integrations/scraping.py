"""
Basit scraping/çekme yardımcıları.
Instagram URL'leri için oEmbed üzerinden başlık (caption) denemesi yapar.
Not: Üretimde hız limitleri ve engeller için proxy/başlık ayarları gerekebilir.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import re
import httpx
import json
from bs4 import BeautifulSoup

from ..utils.logging import get_logger

logger = get_logger("scraping")


def is_instagram_url(url: str) -> bool:
    return bool(re.search(r"instagram\.com/p/|instagram\.com/reel/|instagram\.com/tv/", url))


async def fetch_instagram_oembed(url: str) -> Optional[Dict[str, Any]]:
    endpoint = "https://www.instagram.com/oembed/"
    params = {"url": url}
    timeout = httpx.Timeout(10.0, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}) as client:
            resp = await client.get(endpoint, params=params)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"oEmbed yanıtı {resp.status_code}")
    except Exception as e:
        logger.warning(f"oEmbed hata: {e}")
    return None


async def fetch_post_context(url: str) -> Dict[str, Any]:
    """
    Verilen URL için post metni (caption) ve temel meta bilgisini döndür.
    Şu an Instagram için oEmbed kullanır; diğer kaynaklar genişletilebilir.
    """
    result: Dict[str, Any] = {"url": url, "platform": None, "post_text": None, "meta": {}}

    if is_instagram_url(url):
        result["platform"] = "instagram"
        data = await fetch_instagram_oembed(url)
        if data:
            # 'title' çoğunlukla caption içerir
            result["post_text"] = data.get("title") or data.get("author_name")
            result["meta"] = {
                "author_name": data.get("author_name"),
                "provider": data.get("provider_name"),
                "width": data.get("width"),
                "height": data.get("height"),
            }
    else:
        result["platform"] = "generic"

    return result


def _extract_shortcode(url: str) -> Optional[str]:
    m = re.search(r"instagram\.com/(?:p|reel|tv)/([A-Za-z0-9_-]+)/?", url)
    return m.group(1) if m else None


def _parse_comments_from_scripts(soup: BeautifulSoup, max_comments: int = 50) -> List[str]:
    texts: List[str] = []
    scripts = soup.find_all("script")
    for sc in scripts:
        content = sc.string or sc.text or ""
        if not content:
            continue
        # En kapalı yaklaşım: JSON içindeki \"text\" alanlarını çek
        for m in re.finditer(r'"text"\s*:\s*"(.*?)"', content):
            t = m.group(1)
            t = t.encode("utf-8").decode("unicode_escape")
            t = re.sub(r"\\/", "/", t)
            # Basit filtreler
            if 1 <= len(t) <= 400:
                texts.append(t)
            if len(texts) >= max_comments:
                break
        if len(texts) >= max_comments:
            break
    # Tekilleştir ve sadeleştir
    uniq = []
    seen = set()
    for t in texts:
        key = t.strip()
        if key and key not in seen:
            seen.add(key)
            uniq.append(key)
    return uniq[:max_comments]


async def fetch_instagram_comments(url: str, max_comments: int = 50) -> List[str]:
    """
    Herkese açık postlar için sayfa HTML'inden yorumları çıkarmaya çalışır.
    Bu yöntem Instagram değişikliklerinden etkilenebilir (best-effort).
    """
    if not is_instagram_url(url):
        return []
    timeout = httpx.Timeout(10.0, connect=5.0)
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning(f"Instagram sayfa durumu {resp.status_code}")
                return []
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")
            comments = _parse_comments_from_scripts(soup, max_comments=max_comments)
            return comments
    except Exception as e:
        logger.warning(f"Instagram yorum çekme hata: {e}")
        return []



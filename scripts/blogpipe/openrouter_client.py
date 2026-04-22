"""OpenAI-compatible client for OpenRouter and embeddings."""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

import httpx

from . import config

LOG = logging.getLogger(__name__)
_TIMEOUT = httpx.Timeout(120.0, connect=15.0)


def _client() -> httpx.Client:
    return httpx.Client(verify=True, timeout=_TIMEOUT)


def _headers() -> dict[str, str]:
    h = {
        "HTTP-Referer": "https://anshu92.github.io",
        "X-Title": "blogpipe",
        "User-Agent": "blogpipe/0.1",
    }
    if config.openrouter_key():
        h["Authorization"] = f"Bearer {config.openrouter_key()}"
    return h


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.4,
    max_tokens: int = 8192,
) -> str:
    if config.dry_run() or not config.openrouter_key():
        return ""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = _client().post(
        f"{config.openrouter_base()}/chat/completions",
        headers={**_headers(), "Content-Type": "application/json"},
        content=json.dumps(body).encode("utf-8"),
    )
    r.raise_for_status()
    data = r.json()
    ch = (data or {}).get("choices") or []
    if not ch:
        return ""
    return str((ch[0].get("message") or {}).get("content") or "")


def llm_text(system: str, user: str, model: Optional[str] = None) -> str:
    m = model or config.blogpipe_model()
    return chat_completion(
        m,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )


def embed_text(text: str) -> Optional[List[float]]:
    key = config.openai_key() or config.openrouter_key()
    if not key or not text.strip():
        return None
    base = "https://api.openai.com/v1" if config.openai_key() else f"{config.openrouter_base()}"
    if not config.openai_key() and "openrouter" in base:
        # OpenRouter embeddings: same base as chat
        pass
    body = {
        "model": config.embed_model(),
        "input": text[:8000],
    }
    h = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    r = _client().post(
        f"{base}/embeddings" if "openai.com" in base else f"{config.openrouter_base()}/embeddings",
        headers=h,
        content=json.dumps(body).encode("utf-8"),
    )
    if r.status_code != 200:
        LOG.debug("embed status %s: %s", r.status_code, r.text[:200])
        return None
    try:
        d = r.json()
        v = d["data"][0]["embedding"]
        return [float(x) for x in v]
    except Exception as e:
        LOG.debug("embed parse: %s", e)
        return None

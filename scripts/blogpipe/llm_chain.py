"""Multi-provider model chains — keep in sync with evr `stock_screener/agents/config.py`."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import httpx

from . import config

LOG = logging.getLogger(__name__)
_TIMEOUT = httpx.Timeout(120.0, connect=15.0)

# Same tuples as evr (model_id, provider)
_FAST_MODEL_CHAIN: list[tuple[str, str]] = [
    ("llama-3.3-70b-versatile", "groq"),
    ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
    ("nousresearch/hermes-3-llama-3.1-405b:free", "openrouter"),
    ("nvidia/nemotron-3-super-120b-a12b:free", "openrouter"),
    ("openai/gpt-oss-120b:free", "openrouter"),
    ("qwen/qwen3-32b", "groq"),
    ("qwen/qwen3-next-80b-a3b-instruct:free", "openrouter"),
    ("qwen/qwen3.6-plus-preview:free", "openrouter"),
    ("meta-llama/llama-4-scout-17b-16e-instruct", "groq"),
    ("stepfun/step-3.5-flash:free", "openrouter"),
    ("z-ai/glm-4.5-air:free", "openrouter"),
    ("arcee-ai/trinity-large-preview:free", "openrouter"),
    ("moonshotai/kimi-k2-instruct", "groq"),
    ("google/gemma-3-27b-it:free", "openrouter"),
    ("nvidia/nemotron-3-nano-30b-a3b:free", "openrouter"),
    ("minimax/minimax-m2.5:free", "openrouter"),
    ("openai/gpt-oss-20b:free", "openrouter"),
    ("google/gemma-3-12b-it:free", "openrouter"),
    ("llama-3.1-8b-instant", "groq"),
    ("nvidia/nemotron-nano-9b-v2:free", "openrouter"),
    ("arcee-ai/trinity-mini:free", "openrouter"),
]

_SMART_MODEL_CHAIN: list[tuple[str, str]] = [
    ("gemini-2.5-flash", "gemini"),
    ("llama-3.3-70b-versatile", "groq"),
    ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
    ("nousresearch/hermes-3-llama-3.1-405b:free", "openrouter"),
    ("nvidia/nemotron-3-super-120b-a12b:free", "openrouter"),
    ("openai/gpt-oss-120b:free", "openrouter"),
    ("qwen/qwen3-32b", "groq"),
    ("qwen/qwen3.6-plus-preview:free", "openrouter"),
]

_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
    },
}

# Same set as evr: disable <think> noise where supported
REASONING_MODELS: set[str] = {
    "qwen/qwen3-32b",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "qwen/qwen3.6-plus-preview:free",
    "qwen/qwen3-coder:free",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-20b:free",
}


def _available() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, preset in _PROVIDER_PRESETS.items():
        key = os.getenv(preset["api_key_env"], "").strip()
        if key:
            out[name] = key
    return out


def _build_model_chain(
    chain: list[tuple[str, str]],
    available: dict[str, str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model_id, provider in chain:
        if provider not in available:
            continue
        preset = _PROVIDER_PRESETS[provider]
        result.append(
            {
                "model": model_id,
                "provider": provider,
                "api_key": available[provider],
                "base_url": preset["base_url"],
            }
        )
    return result


def resolved_chains() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    a = _available()
    return (
        _build_model_chain(_FAST_MODEL_CHAIN, a),
        _build_model_chain(_SMART_MODEL_CHAIN, a),
    )


def _openrouter_headers() -> dict[str, str]:
    return {
        "HTTP-Referer": "https://anshu92.github.io",
        "X-Title": "blogpipe",
        "User-Agent": "blogpipe/0.1",
    }


def _one_chat(
    entry: dict[str, Any],
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    base = (entry.get("base_url") or "").rstrip("/")
    url = f"{base}/chat/completions"
    prov = entry.get("provider", "")
    body: dict[str, Any] = {
        "model": entry["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    m = str(entry.get("model", ""))
    if m in REASONING_MODELS:
        body["reasoning_effort"] = "none"
    h: dict[str, str] = {
        "Authorization": f"Bearer {entry['api_key']}",
        "Content-Type": "application/json",
    }
    if prov == "openrouter":
        h.update(_openrouter_headers())
    with httpx.Client(verify=True, timeout=_TIMEOUT) as cl:
        r = cl.post(url, headers=h, content=json.dumps(body).encode("utf-8"))
    if r.is_error:
        LOG.warning(
            "llm %s (model=%s, provider=%s): %s",
            r.status_code,
            m,
            prov,
            (r.text or "")[:500],
        )
    r.raise_for_status()
    data = r.json()
    ch = (data or {}).get("choices") or []
    if not ch:
        return ""
    return str((ch[0].get("message") or {}).get("content") or "")


def chat_with_chain(
    messages: list[dict[str, str]],
    mode: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 8192,
) -> str:
    """Try the evr fast chain, or smart then fast (like evr ``_smart_call``)."""
    fast, smart = resolved_chains()
    if mode == "smart":
        pools: list[list[dict[str, Any]]] = [smart, fast]
    else:
        pools = [fast]
    last: Optional[Exception] = None
    for pool in pools:
        for entry in pool:
            try:
                return _one_chat(entry, messages, temperature, max_tokens)
            except Exception as e:  # noqa: BLE001 — try next, like evr
                last = e
                continue
    if last is not None:
        LOG.warning("llm chain exhausted: %s", last)
    return ""

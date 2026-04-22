"""Multi-provider model chains — keep in sync with evr `stock_screener/agents/config.py`."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import httpx

from . import config

LOG = logging.getLogger(__name__)
_TIMEOUT = httpx.Timeout(120.0, connect=15.0)

# (model_id, provider). OpenRouter slugs are from https://openrouter.ai/api/v1/models — use :free
# only where the API lists that id (no paid-only ids with a fake :free suffix).
_FAST_MODEL_CHAIN: list[tuple[str, str]] = [
    # Groq
    ("llama-3.3-70b-versatile", "groq"),
    ("qwen/qwen3-32b", "groq"),
    ("meta-llama/llama-4-scout-17b-16e-instruct", "groq"),
    ("moonshotai/kimi-k2-instruct", "groq"),
    ("llama-3.1-8b-instant", "groq"),
    # OpenRouter free pool (order: strong general + router + breadth + small fallbacks)
    ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
    ("nousresearch/hermes-3-llama-3.1-405b:free", "openrouter"),
    ("nvidia/nemotron-3-super-120b-a12b:free", "openrouter"),
    ("openai/gpt-oss-120b:free", "openrouter"),
    ("qwen/qwen3-next-80b-a3b-instruct:free", "openrouter"),
    ("openrouter/free", "openrouter"),
    ("google/gemma-4-31b-it:free", "openrouter"),
    ("google/gemma-4-26b-a4b-it:free", "openrouter"),
    ("minimax/minimax-m2.5:free", "openrouter"),
    ("z-ai/glm-4.5-air:free", "openrouter"),
    ("arcee-ai/trinity-large-preview:free", "openrouter"),
    ("qwen/qwen3-coder:free", "openrouter"),
    ("inclusionai/ling-2.6-flash:free", "openrouter"),
    ("google/gemma-3-27b-it:free", "openrouter"),
    ("google/gemma-3-12b-it:free", "openrouter"),
    ("google/gemma-3-4b-it:free", "openrouter"),
    ("google/gemma-3n-e4b-it:free", "openrouter"),
    ("google/gemma-3n-e2b-it:free", "openrouter"),
    ("nvidia/nemotron-3-nano-30b-a3b:free", "openrouter"),
    ("nvidia/nemotron-nano-9b-v2:free", "openrouter"),
    ("nvidia/nemotron-nano-12b-v2-vl:free", "openrouter"),
    ("openai/gpt-oss-20b:free", "openrouter"),
    ("meta-llama/llama-3.2-3b-instruct:free", "openrouter"),
    ("liquid/lfm-2.5-1.2b-instruct:free", "openrouter"),
    ("cognitivecomputations/dolphin-mistral-24b-venice-edition:free", "openrouter"),
]

_SMART_MODEL_CHAIN: list[tuple[str, str]] = [
    ("gemini-2.5-flash", "gemini"),
    ("llama-3.3-70b-versatile", "groq"),
    ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
    ("nousresearch/hermes-3-llama-3.1-405b:free", "openrouter"),
    ("openrouter/free", "openrouter"),
    ("google/gemma-4-31b-it:free", "openrouter"),
    ("google/gemma-4-26b-a4b-it:free", "openrouter"),
    ("nvidia/nemotron-3-super-120b-a12b:free", "openrouter"),
    ("openai/gpt-oss-120b:free", "openrouter"),
    ("qwen/qwen3-next-80b-a3b-instruct:free", "openrouter"),
    ("qwen/qwen3-32b", "groq"),
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

# Per-model max completion tokens (stay under low TPM/TPD providers; caller default is capped by this).
_MODEL_MAX_TOKENS: dict[str, int] = {
    "qwen/qwen3-32b": 3000,
    "llama-3.3-70b-versatile": 6000,
}

# Disable <think> where the API allows; omit gpt-oss (OpenRouter requires reasoning on).
REASONING_MODELS: set[str] = {
    "qwen/qwen3-32b",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "qwen/qwen3-coder:free",
}

_BLACKLIST: dict[str, float] = {}
_BLACKLIST_TTL_DAILY = 86400.0
_USAGE: dict[str, Any] = {"ok": 0, "fail": 0, "blacklisted_skips": 0, "blacklist_added": []}


def reset_llm_usage() -> None:
    """Call once per pipeline run (e.g. start of research) to aggregate usage across stages."""
    global _USAGE
    _USAGE = {"ok": 0, "fail": 0, "blacklisted_skips": 0, "blacklist_added": []}


def get_llm_usage() -> dict[str, Any]:
    return {
        "ok": int(_USAGE.get("ok", 0)),
        "fail": int(_USAGE.get("fail", 0)),
        "blacklisted_skips": int(_USAGE.get("blacklisted_skips", 0)),
        "blacklist_added": list(_USAGE.get("blacklist_added") or []),
    }


def bump_llm_ok() -> None:
    _USAGE["ok"] = int(_USAGE.get("ok", 0)) + 1


def bump_llm_fail() -> None:
    _USAGE["fail"] = int(_USAGE.get("fail", 0)) + 1


def _root() -> str:
    return os.environ.get("BLOGPIPE_REPO_ROOT", os.getcwd()).strip() or os.getcwd()


def _blacklist_path() -> str:
    return os.path.join(_root(), "cache", "llm_blacklist.json")


def _prune_expired(m: dict[str, float], now: float) -> None:
    dead = [k for k, exp in m.items() if exp <= now]
    for k in dead:
        del m[k]


def _load_disk_blacklist() -> dict[str, float]:
    p = _blacklist_path()
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(d, dict):
        return {}
    now = time.time()
    out: dict[str, float] = {}
    for k, v in d.items():
        if not isinstance(k, str):
            continue
        try:
            exp = float(v)
        except (TypeError, ValueError):
            continue
        if exp > now:
            out[k] = exp
    return out


def _merge_blacklist_from_disk() -> None:
    global _BLACKLIST
    disk = _load_disk_blacklist()
    now = time.time()
    for k, exp in disk.items():
        if exp > now and exp > _BLACKLIST.get(k, 0):
            _BLACKLIST[k] = exp
    _prune_expired(_BLACKLIST, now)


def _persist_blacklist() -> None:
    p = _blacklist_path()
    try:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    except OSError as e:
        LOG.debug("blacklist cache dir: %s", e)
        return
    now = time.time()
    _prune_expired(_BLACKLIST, now)
    to_save = {k: v for k, v in _BLACKLIST.items() if v > now}
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=0)
    except OSError as e:
        LOG.debug("blacklist write: %s", e)


def _entry_key(entry: dict[str, Any]) -> str:
    return f"{entry.get('provider', '')}:{entry.get('model', '')}"


def _blacklist_ttl_for_body(status: int, body: str) -> Optional[float]:
    low = (body or "").lower()
    if status == 429 and any(
        s in low
        for s in (
            "tokens per day",
            "free-models-per-day",
            "resource_exhausted",
            "resource exhausted",
            "rate limit",
            "too many requests",
        )
    ):
        return _BLACKLIST_TTL_DAILY
    if status in (400, 404) and any(
        s in low
        for s in (
            "not a valid model",
            "invalid model",
            "unknown model",
            "not a valid model id",
        )
    ):
        return _BLACKLIST_TTL_DAILY
    return None


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


_GROQ_CONTENT_CAP = 5500  # stay under Groq TPM on long prompts (API returns 413 when over budget)


def _messages_for_provider(messages: list[dict[str, str]], prov: str) -> list[dict[str, str]]:
    if prov != "groq":
        return messages
    out: list[dict[str, str]] = []
    for m in messages:
        c = m.get("content", "")
        if len(c) > _GROQ_CONTENT_CAP:
            c = c[:_GROQ_CONTENT_CAP] + "\n\n[truncated for Groq request size limits]"
        out.append({**m, "content": c})
    return out


def _effective_max_tokens(model: str, requested: int) -> int:
    cap = _MODEL_MAX_TOKENS.get(model)
    if cap is not None:
        return min(requested, cap)
    return requested


def _one_chat(
    entry: dict[str, Any],
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    base = (entry.get("base_url") or "").rstrip("/")
    url = f"{base}/chat/completions"
    prov = entry.get("provider", "")
    m = str(entry.get("model", ""))
    cap = _effective_max_tokens(m, max_tokens)
    to_send = _messages_for_provider(messages, prov)
    body: dict[str, Any] = {
        "model": entry["model"],
        "messages": to_send,
        "temperature": temperature,
        "max_tokens": cap,
    }
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
        err_body = (r.text or "")[:2000]
        LOG.warning(
            "llm %s (model=%s, provider=%s): %s",
            r.status_code,
            m,
            prov,
            err_body[:500],
        )
        ttl = _blacklist_ttl_for_body(int(r.status_code), err_body)
        if ttl is not None:
            k = _entry_key(entry)
            _BLACKLIST[k] = time.time() + ttl
            if k not in (_USAGE.get("blacklist_added") or []):
                _USAGE["blacklist_added"].append(k)
            _persist_blacklist()
        r.raise_for_status()
    data = r.json()
    ch = (data or {}).get("choices") or []
    if not ch:
        _USAGE["fail"] = int(_USAGE.get("fail", 0)) + 1
        return ""
    _USAGE["ok"] = int(_USAGE.get("ok", 0)) + 1
    return str((ch[0].get("message") or {}).get("content") or "")


def _is_active_blacklist(k: str) -> bool:
    return _BLACKLIST.get(k, 0) > time.time()


def chat_with_chain(
    messages: list[dict[str, str]],
    mode: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 1536,
) -> str:
    """Try the evr fast chain, or smart then fast (like evr ``_smart_call``)."""
    _merge_blacklist_from_disk()
    now = time.time()
    _prune_expired(_BLACKLIST, now)
    fast, smart = resolved_chains()
    if mode == "smart":
        pools: list[list[dict[str, Any]]] = [smart, fast]
    else:
        pools = [fast]
    last: Optional[Exception] = None
    for pool in pools:
        for entry in pool:
            k = _entry_key(entry)
            if _is_active_blacklist(k):
                _USAGE["blacklisted_skips"] = int(_USAGE.get("blacklisted_skips", 0)) + 1
                continue
            try:
                return _one_chat(entry, messages, temperature, max_tokens)
            except Exception as e:  # noqa: BLE001 — try next, like evr
                _USAGE["fail"] = int(_USAGE.get("fail", 0)) + 1
                last = e
                continue
    if last is not None:
        LOG.warning("llm chain exhausted: %s", last)
    return ""

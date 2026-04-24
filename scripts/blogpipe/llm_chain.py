"""Multi-provider model chains — keep in sync with evr `stock_screener/agents/config.py`."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, Generator, Optional

import httpx

from . import config

LOG = logging.getLogger(__name__)
_TIMEOUT = httpx.Timeout(120.0, connect=15.0)

# (model_id, provider). OpenRouter slugs are from https://openrouter.ai/api/v1/models — use :free
# only where the API lists that id (no paid-only ids with a fake :free suffix).
_FAST_MODEL_CHAIN: list[tuple[str, str]] = [
    # Native Gemini first when GEMINI_API_KEY is set (task-agnostic fast fallback).
    ("gemini-2.5-flash", "gemini"),
    # Groq (verified live ids; see https://console.groq.com/docs/models)
    ("llama-3.3-70b-versatile", "groq"),
    ("qwen/qwen3-32b", "groq"),
    ("meta-llama/llama-4-scout-17b-16e-instruct", "groq"),
    ("llama-3.1-8b-instant", "groq"),
    # OpenRouter free pool (order: strong general + router + breadth + small fallbacks)
    ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
    ("nvidia/nemotron-3-super-120b-a12b:free", "openrouter"),
    ("openai/gpt-oss-120b:free", "openrouter"),
    ("qwen/qwen3-next-80b-a3b-instruct:free", "openrouter"),
    ("openrouter/free", "openrouter"),
    ("z-ai/glm-4.5-air:free", "openrouter"),
    ("qwen/qwen3-coder:free", "openrouter"),
    ("google/gemma-3-27b-it:free", "openrouter"),
    ("google/gemma-3-12b-it:free", "openrouter"),
    ("nvidia/nemotron-nano-9b-v2:free", "openrouter"),
    ("openai/gpt-oss-20b:free", "openrouter"),
    ("meta-llama/llama-3.2-3b-instruct:free", "openrouter"),
]

_SMART_MODEL_CHAIN: list[tuple[str, str]] = [
    ("gemini-2.5-flash", "gemini"),
    ("llama-3.3-70b-versatile", "groq"),
    ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
    ("openrouter/free", "openrouter"),
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
_BLACKLIST_TTL_SHORT = 600.0  # 10 minutes — for transient TPM/503 spikes
_BLACKLIST_TTL_MED = 3600.0  # 1 hour — for upstream "temporarily rate-limited" pools
def _empty_usage() -> dict[str, Any]:
    return {
        "ok": 0,
        "fail": 0,
        "blacklisted_skips": 0,
        "blacklist_added": [],
        "tokens_in": 0,
        "tokens_out": 0,
        "usd_spent": 0.0,
        "by_task": {},
        "by_model": {},
    }


_USAGE: dict[str, Any] = _empty_usage()
_LLM_CALL_CAP: int = 0  # 0 = disabled
_STAGE_QUOTAS: dict[str, int] = {}
_STAGE_COUNTS: dict[str, int] = {}
_CURRENT_STAGE: ContextVar[Optional[str]] = ContextVar("blogpipe_llm_stage", default=None)


def set_llm_call_cap(n: int) -> None:
    """0 disables; when set, chat_with_chain stops when ok+fail >= n (after each attempt)."""
    global _LLM_CALL_CAP
    _LLM_CALL_CAP = max(0, int(n))


def set_stage_quotas(quotas: dict[str, int] | None) -> None:
    """Per-stage call caps; empty/None disables per-stage (global cap only). Counts are not reset here."""
    global _STAGE_QUOTAS
    if not quotas:
        _STAGE_QUOTAS = {}
        return
    _STAGE_QUOTAS = {str(k).strip(): max(0, int(v)) for k, v in quotas.items() if str(k).strip()}


def current_stage() -> str | None:
    return _CURRENT_STAGE.get()


def is_stage_full(name: str | None = None) -> bool:
    """True when this stage (or the context stage if name is None) has hit its per-stage cap."""
    s = (name or "").strip() or current_stage() or ""
    if not s or s not in _STAGE_QUOTAS:
        return False
    q = int(_STAGE_QUOTAS.get(s, 0))
    if q <= 0:
        return False
    return int(_STAGE_COUNTS.get(s, 0)) >= q


@contextmanager
def stage(name: str) -> Generator[None, None, None]:
    """Bind LLM usage accounting to a pipeline stage (propagates in async/threads via copy_context)."""
    t = _CURRENT_STAGE.set((name or "").strip())
    try:
        yield
    finally:
        _CURRENT_STAGE.reset(t)


budget = SimpleNamespace(stage=stage)


def _bump_stage_on_llm_outcome() -> None:
    s = (current_stage() or "").strip()
    if not s or s not in _STAGE_QUOTAS:
        return
    _STAGE_COUNTS[s] = int(_STAGE_COUNTS.get(s, 0)) + 1


def _at_call_cap() -> bool:
    if _LLM_CALL_CAP > 0:
        if int(_USAGE.get("ok", 0)) + int(_USAGE.get("fail", 0)) >= _LLM_CALL_CAP:
            return True
    if is_stage_full():
        return True
    return False


def is_llm_call_cap_reached() -> bool:
    """True when global or per-stage LLM cap would block further chat/completion calls."""
    return _at_call_cap()


def reset_llm_usage() -> None:
    """Call once per pipeline run (e.g. start of research) to aggregate usage across stages."""
    global _USAGE, _STAGE_COUNTS
    _USAGE = _empty_usage()
    _STAGE_COUNTS = {}


def get_llm_usage() -> dict[str, Any]:
    return {
        "ok": int(_USAGE.get("ok", 0)),
        "fail": int(_USAGE.get("fail", 0)),
        "blacklisted_skips": int(_USAGE.get("blacklisted_skips", 0)),
        "blacklist_added": list(_USAGE.get("blacklist_added") or []),
        "tokens_in": int(_USAGE.get("tokens_in", 0)),
        "tokens_out": int(_USAGE.get("tokens_out", 0)),
        "usd_spent": float(_USAGE.get("usd_spent", 0) or 0.0),
        "by_task": dict(_USAGE.get("by_task") or {}),
        "by_model": dict(_USAGE.get("by_model") or {}),
        "by_stage": dict(_STAGE_COUNTS),
    }


def budget_remaining_usd() -> float:
    from . import config  # local: avoid import cycle

    return max(0.0, float(config.usd_budget()) - float(_USAGE.get("usd_spent", 0) or 0.0))


def record_completion_tokens(
    model_slug: str, in_t: int, out_t: int, task: str | None = None
) -> None:
    """Update usage counters after a direct OpenRouter completion (non-chain path)."""
    if in_t or out_t:
        _record_token_usage(model_slug, in_t, out_t, task)


def bump_llm_ok() -> None:
    _USAGE["ok"] = int(_USAGE.get("ok", 0)) + 1
    _bump_stage_on_llm_outcome()


def bump_llm_fail() -> None:
    _USAGE["fail"] = int(_USAGE.get("fail", 0)) + 1
    _bump_stage_on_llm_outcome()


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


def register_blacklist_key(key: str, ttl: float) -> None:
    """Block a logical key (e.g. image:pollinations) for ttl seconds, persisted to disk."""
    if not (key or "").strip():
        return
    _merge_blacklist_from_disk()
    _BLACKLIST[key.strip()] = time.time() + max(1.0, float(ttl))
    _persist_blacklist()


def is_blacklist_key_active(key: str) -> bool:
    if not (key or "").strip():
        return False
    _merge_blacklist_from_disk()
    return _BLACKLIST.get(key.strip(), 0) > time.time()


def _entry_key(entry: dict[str, Any]) -> str:
    return f"{entry.get('provider', '')}:{entry.get('model', '')}"


def _blacklist_ttl_for_body(status: int, body: str) -> Optional[float]:
    low = (body or "").lower()
    # Permanent-for-the-day: model is gone, daily quota exhausted, or hard not-found.
    if status in (400, 404) and any(
        s in low
        for s in (
            "not a valid model",
            "invalid model",
            "unknown model",
            "not a valid model id",
            "model_not_found",
            "does not exist",
            "no endpoints found",
            "no allowed providers",
        )
    ):
        return _BLACKLIST_TTL_DAILY
    if status == 429 and any(
        s in low
        for s in (
            "tokens per day",
            "requests per day",
            "free-models-per-day",
            "resource_exhausted",
            "resource exhausted",
        )
    ):
        return _BLACKLIST_TTL_DAILY
    # Medium TTL: provider-side pool rate-limit ("temporarily rate-limited upstream").
    if status == 429 and "temporarily rate-limited" in low:
        return _BLACKLIST_TTL_MED
    # Short TTL: per-minute TPM/RPM spikes and transient upstream errors.
    if status == 429 and any(
        s in low
        for s in (
            "tokens per minute",
            "requests per minute",
            "rate limit",
            "too many requests",
        )
    ):
        return _BLACKLIST_TTL_SHORT
    if status in (502, 503, 504):
        return _BLACKLIST_TTL_SHORT
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
# Pre-trim long user payloads in critics so Groq never word-cuts; leave margin below _GROQ_CONTENT_CAP
GROQ_USER_CONTENT_BUDGET = _GROQ_CONTENT_CAP - 500


def _messages_for_provider(messages: list[dict[str, str]], prov: str) -> list[dict[str, str]]:
    if prov != "groq":
        return messages
    out: list[dict[str, str]] = []
    for m in messages:
        c = m.get("content", "")
        if len(c) > _GROQ_CONTENT_CAP:
            # Cut at a clean word boundary; do NOT append a textual marker — past attempts
            # ("[truncated for Groq request size limits]") were echoed verbatim into drafts.
            cut = c[:_GROQ_CONTENT_CAP].rsplit(" ", 1)[0]
            c = cut.rstrip() + "\n"
        out.append({**m, "content": c})
    return out


def _effective_max_tokens(model: str, requested: int) -> int:
    cap = _MODEL_MAX_TOKENS.get(model)
    if cap is not None:
        return min(requested, cap)
    return requested


def _record_token_usage(
    model_slug: str,
    in_t: int,
    out_t: int,
    task: str | None,
) -> None:
    from . import model_registry  # local

    u = model_registry.usd_for_tokens(model_slug, in_t, out_t)
    _USAGE["tokens_in"] = int(_USAGE.get("tokens_in", 0)) + in_t
    _USAGE["tokens_out"] = int(_USAGE.get("tokens_out", 0)) + out_t
    _USAGE["usd_spent"] = float(_USAGE.get("usd_spent", 0) or 0.0) + u
    bm = _USAGE.get("by_model")
    if not isinstance(bm, dict):
        bm = {}
    mk = f"{model_slug}"
    cur = bm.get(mk) or {}
    if not isinstance(cur, dict):
        cur = {}
    cur["calls"] = int(cur.get("calls", 0) or 0) + 1
    cur["tokens_in"] = int(cur.get("tokens_in", 0) or 0) + in_t
    cur["tokens_out"] = int(cur.get("tokens_out", 0) or 0) + out_t
    cur["usd"] = float(cur.get("usd", 0) or 0.0) + u
    bm[mk] = cur
    _USAGE["by_model"] = bm
    if task and task.strip():
        bt = _USAGE.get("by_task")
        if not isinstance(bt, dict):
            bt = {}
        tcur = bt.get(task) or {}
        if not isinstance(tcur, dict):
            tcur = {}
        tcur["calls"] = int(tcur.get("calls", 0) or 0) + 1
        tcur["tokens_in"] = int(tcur.get("tokens_in", 0) or 0) + in_t
        tcur["tokens_out"] = int(tcur.get("tokens_out", 0) or 0) + out_t
        tcur["usd"] = float(tcur.get("usd", 0) or 0.0) + u
        bt[task] = tcur
        _USAGE["by_task"] = bt


def _retry_delay_seconds_429(response: httpx.Response) -> float:
    """Bounded wait before retrying a rate-limited chat request."""
    try:
        ra = (response.headers.get("retry-after") or "").strip()
        if ra:
            return min(90.0, max(2.0, float(ra)))
    except ValueError:
        pass
    txt = (response.text or "")[:4000]
    m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", txt, re.I)
    if m:
        return min(90.0, max(2.0, float(m.group(1))))
    return min(45.0, 6.0 + random.random() * 6.0)


def _one_chat(
    entry: dict[str, Any],
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    *,
    task: str | None = None,
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
    r: httpx.Response | None = None
    for attempt in range(3):
        with httpx.Client(verify=True, timeout=_TIMEOUT) as cl:
            r = cl.post(url, headers=h, content=json.dumps(body).encode("utf-8"))
        if r.status_code != 429:
            break
        if attempt >= 2:
            break
        delay = _retry_delay_seconds_429(r)
        LOG.warning(
            "llm 429, sleeping %.1fs then retry (%d/2) model=%s provider=%s",
            delay,
            attempt + 1,
            m,
            prov,
        )
        time.sleep(delay)
    assert r is not None
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
    data = r.json() or {}
    ch = data.get("choices") or []
    if not ch:
        _USAGE["fail"] = int(_USAGE.get("fail", 0)) + 1
        _bump_stage_on_llm_outcome()
        return ""
    _USAGE["ok"] = int(_USAGE.get("ok", 0)) + 1
    _bump_stage_on_llm_outcome()
    out_t = int(
        (data.get("usage") or {}).get("completion_tokens", 0)
        or (data.get("usage") or {}).get("completion", 0)
    )
    in_t = int(
        (data.get("usage") or {}).get("prompt_tokens", 0)
        or (data.get("usage") or {}).get("prompt", 0)
    )
    text = str((ch[0].get("message") or {}).get("content") or "")
    if in_t == 0 and out_t == 0:
        in_t = max(1, sum(len(str(mm.get("content", "")) or "") for mm in to_send) // 4)
        out_t = max(1, len(text) // 4)
    _record_token_usage(m, in_t, out_t, task)
    return text


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
    if _at_call_cap():
        LOG.warning("llm: call cap reached, skipping provider chain")
        return ""
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
            if _at_call_cap():
                return ""
            k = _entry_key(entry)
            if _is_active_blacklist(k):
                _USAGE["blacklisted_skips"] = int(_USAGE.get("blacklisted_skips", 0)) + 1
                continue
            try:
                return _one_chat(entry, messages, temperature, max_tokens, task=None)
            except Exception as e:  # noqa: BLE001 — try next, like evr
                _USAGE["fail"] = int(_USAGE.get("fail", 0)) + 1
                _bump_stage_on_llm_outcome()
                last = e
                continue
    if last is not None:
        LOG.warning("llm chain exhausted: %s", last)
    return ""


def chat_with_task_chain(
    messages: list[dict[str, str]],
    task: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 1536,
) -> str:
    """Run task-specific model chain from model_registry; fallback to mode pool if empty."""
    if _at_call_cap():
        LOG.warning("llm: call cap reached, skipping task chain")
        return ""
    _merge_blacklist_from_disk()
    from . import model_registry  # local

    est = max(
        1,
        sum(len(str(m.get("content", "")) or "") for m in messages) // 4,
    )
    raw_chain = model_registry.select_chain(
        task,
        est,
        prefer_free=config.prefer_free_models(),
        usd_budget_remaining=budget_remaining_usd(),
    )
    a = _available()
    built = _build_model_chain(raw_chain, a)
    if not built:
        return chat_with_chain(
            messages, "fast", temperature=temperature, max_tokens=max_tokens
        )
    last: Optional[Exception] = None
    for entry in built:
        if _at_call_cap():
            return ""
        k = _entry_key(entry)
        if _is_active_blacklist(k):
            _USAGE["blacklisted_skips"] = int(_USAGE.get("blacklisted_skips", 0)) + 1
            continue
        try:
            return _one_chat(
                entry, messages, temperature, max_tokens, task=task
            )
        except Exception as e:  # noqa: BLE001
            _USAGE["fail"] = int(_USAGE.get("fail", 0)) + 1
            _bump_stage_on_llm_outcome()
            last = e
            continue
    if last is not None:
        LOG.warning("llm task chain exhausted: %s", last)
    return chat_with_chain(
        messages, "fast", temperature=temperature, max_tokens=max_tokens
    )

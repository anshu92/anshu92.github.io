"""OpenAI-compatible LLM and embeddings (OpenRouter, Groq, Gemini via evr-ordered chain)."""

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
    max_tokens: Optional[int] = None,
) -> str:
    """Single OpenRouter chat. Returns "" on error or if no key."""
    from . import llm_chain  # lazy: import order

    if max_tokens is None:
        max_tokens = config.max_tokens_fast()
    if llm_chain.is_llm_call_cap_reached():
        return ""
    if config.dry_run() or not config.openrouter_key():
        return ""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = _client().post(
            f"{config.openrouter_base()}/chat/completions",
            headers={**_headers(), "Content-Type": "application/json"},
            content=json.dumps(body).encode("utf-8"),
        )
        if r.is_error:
            LOG.warning("openrouter chat %s (model=%s): %s", r.status_code, model, (r.text or "")[:800])
        r.raise_for_status()
        data = r.json() or {}
        ch = data.get("choices") or []
        if not ch:
            llm_chain.bump_llm_fail()
            return ""
        out = str((ch[0].get("message") or {}).get("content") or "")
        u = (data.get("usage") or {})
        pin = int(u.get("prompt_tokens") or u.get("prompt", 0) or 0)
        cout = int(u.get("completion_tokens") or u.get("completion", 0) or 0)
        if out.strip() and (pin or cout):
            llm_chain.record_completion_tokens(model, pin, cout, None)
        if out.strip():
            llm_chain.bump_llm_ok()
        else:
            llm_chain.bump_llm_fail()
        return out
    except httpx.HTTPError as e:
        LOG.warning("openrouter request failed: %s", e)
        llm_chain.bump_llm_fail()
        return ""


def llm_text(
    system: str,
    user: str,
    model: Optional[str] = None,
    *,
    mode: str = "fast",
    max_tokens: Optional[int] = None,
    task: Optional[str] = None,
    temperature: float = 0.4,
) -> str:
    """If ``BLOGPIPE_MODEL`` / ``BLOGPIPE_EDITOR_MODEL`` or ``model=`` is set, try that on OpenRouter first, then task registry (``task=``) or the evr chain."""
    from . import llm_chain

    if config.dry_run():
        return ""
    if llm_chain.is_llm_call_cap_reached():
        return ""
    ovr = int(config.max_tokens_for_task(task) or 0) if (task or "").strip() else 0
    if max_tokens is None:
        if ovr > 0:
            max_tokens = ovr
        else:
            max_tokens = (
                config.max_tokens_smart() if mode == "smart" else config.max_tokens_fast()
            )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    m_arg = (model or "").strip()
    if mode not in ("fast", "smart"):
        mode = "fast"
    if not m_arg and mode == "fast" and config.blogpipe_model().strip():
        m_arg = config.blogpipe_model().strip()
    if not m_arg and mode == "smart" and config.editor_model().strip():
        m_arg = config.editor_model().strip()

    if m_arg and config.openrouter_key():
        out = chat_completion(m_arg, messages, temperature=temperature, max_tokens=max_tokens)
        if out.strip():
            return out
        LOG.warning("explicit model %s returned empty; trying provider chain", m_arg)
    elif m_arg and not config.openrouter_key():
        LOG.warning("model override set but no OPENROUTER_API_KEY; using provider chain")

    if (task or "").strip() and config.llm_configured():
        from .llm_chain import chat_with_task_chain

        return chat_with_task_chain(
            messages,
            task or "",
            temperature=temperature,
            max_tokens=max_tokens or 1536,
        )

    if not config.llm_configured():
        return ""

    from .llm_chain import chat_with_chain

    return chat_with_chain(
        messages,
        "smart" if mode == "smart" else "fast",
        temperature=temperature,
        max_tokens=max_tokens,
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
    except Exception as e:  # noqa: BLE001
        LOG.debug("embed parse: %s", e)
        return None

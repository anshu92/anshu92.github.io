from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import httpx

LOG = logging.getLogger(__name__)

LLM_TASK_ENV_VARS: dict[str, str] = {
    "selector": "BLOGPIPE_LLM_MODEL_SELECTOR",
    "outline": "BLOGPIPE_LLM_MODEL_OUTLINE",
    "outline_repair": "BLOGPIPE_LLM_MODEL_OUTLINE_REPAIR",
    "draft": "BLOGPIPE_LLM_MODEL_DRAFT",
    "draft_section": "BLOGPIPE_LLM_MODEL_DRAFT_SECTION",
    "editor": "BLOGPIPE_LLM_MODEL_EDITOR",
    "quality_review": "BLOGPIPE_LLM_MODEL_QUALITY_REVIEW",
    "repair": "BLOGPIPE_LLM_MODEL_REPAIR",
}

LLM_TASK_CHAIN_ENV_VARS: dict[str, str] = {
    "selector": "BLOGPIPE_LLM_CHAIN_SELECTOR",
    "outline": "BLOGPIPE_LLM_CHAIN_OUTLINE",
    "outline_repair": "BLOGPIPE_LLM_CHAIN_OUTLINE_REPAIR",
    "draft": "BLOGPIPE_LLM_CHAIN_DRAFT",
    "draft_section": "BLOGPIPE_LLM_CHAIN_DRAFT_SECTION",
    "editor": "BLOGPIPE_LLM_CHAIN_EDITOR",
    "quality_review": "BLOGPIPE_LLM_CHAIN_QUALITY_REVIEW",
    "repair": "BLOGPIPE_LLM_CHAIN_REPAIR",
}


def _get(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(_get(name, str(default)))
    except ValueError:
        value = default
    return max(low, min(high, value))


def _float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(_get(name, str(default)))
    except ValueError:
        value = default
    return max(low, min(high, value))


def _csv(name: str) -> list[str]:
    raw = _get(name)
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


# Recommended Gemini roster (June 2026). See ai.google.dev/gemini-api/docs/deprecations.
# gemini-2.0-flash* shut down 2026-06-01; gemini-2.5-* remain until 2026-10-16.
# gemini-3.1-flash-lite-preview shut down 2026-05-25; use gemini-3.1-flash-lite.
DEFAULT_GEMINI_MODEL_FAST = "gemini-3.5-flash"
DEFAULT_GEMINI_MODEL_SMART = "gemini-3.1-pro-preview"
DEFAULT_GEMINI_CHAIN_FAST = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
]
DEFAULT_GEMINI_CHAIN_SMART = [
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-2.5-pro",
]

# OpenRouter free roster (June 2026). Verified via GET https://openrouter.ai/api/v1/models.
DEFAULT_OPENROUTER_FREE_MODELS = [
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "nex-agi/nex-n2-pro:free",
    "qwen/qwen3-coder:free",
    "poolside/laguna-m.1:free",
    "poolside/laguna-xs.2:free",
    "openai/gpt-oss-120b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-20b:free",
    "openrouter/free",
]

# OpenRouter free models that routinely exceed fast-task budgets under load.
SLOW_OPENROUTER_MODELS = frozenset(
    {
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
    }
)

# Fast OpenRouter free models to try first on native Gemini 429 (latency over peak quality).
DEFAULT_OPENROUTER_RATE_LIMIT_FALLBACK = [
    "openrouter/free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "poolside/laguna-xs.2:free",
    "openai/gpt-oss-20b:free",
    "qwen/qwen3-coder:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]

# Emergency roster after the native chain is exhausted under rate limits.
DEFAULT_OPENROUTER_SMART_EMERGENCY = list(DEFAULT_OPENROUTER_RATE_LIMIT_FALLBACK)
_OPENROUTER_DYNAMIC_FREE_MODELS_CACHE: list[str] | None = None


def openrouter_free_models_after_rate_limit(
    roster: list[str],
    *,
    tried: list[str],
    limit: int = 2,
) -> list[str]:
    seen = {model for model in tried if model}
    out: list[str] = []
    for model in roster:
        if model not in seen:
            out.append(model)
            if len(out) >= limit:
                break
    return out


def openrouter_chain_models(*, custom: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for model in [*DEFAULT_OPENROUTER_RATE_LIMIT_FALLBACK, *custom]:
        normalized = (model or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def openrouter_dynamic_free_models_enabled() -> bool:
    raw = os.environ.get("BLOGPIPE_OPENROUTER_DYNAMIC_FREE_MODELS", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return False


def openrouter_models_timeout_seconds() -> float:
    return _float("BLOGPIPE_OPENROUTER_MODELS_TIMEOUT_SECONDS", 5.0, 1.0, 30.0)


def openrouter_dynamic_free_model_limit() -> int:
    return _int("BLOGPIPE_OPENROUTER_DYNAMIC_FREE_MODEL_LIMIT", 16, 4, 40)


def discover_openrouter_free_models(*, base_url: str, api_key: str = "") -> list[str]:
    global _OPENROUTER_DYNAMIC_FREE_MODELS_CACHE
    if _OPENROUTER_DYNAMIC_FREE_MODELS_CACHE is not None:
        return list(_OPENROUTER_DYNAMIC_FREE_MODELS_CACHE)
    if not openrouter_dynamic_free_models_enabled():
        return []
    timeout = openrouter_models_timeout_seconds()
    try:
        headers = {"User-Agent": user_agent()}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)),
        )
        response.raise_for_status()
        payload = response.json()
        models = _rank_openrouter_free_models(payload.get("data", []), limit=openrouter_dynamic_free_model_limit())
    except Exception as exc:  # noqa: BLE001
        LOG.warning("openrouter dynamic free-model discovery failed; using static fallback: %s", exc)
        _OPENROUTER_DYNAMIC_FREE_MODELS_CACHE = []
        return []
    if models:
        _OPENROUTER_DYNAMIC_FREE_MODELS_CACHE = list(models)
    return models


def _rank_openrouter_free_models(data: object, *, limit: int) -> list[str]:
    if not isinstance(data, list):
        return []
    candidates: list[tuple[float, str]] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        model_id = str(raw.get("id") or "").strip()
        if not model_id or not _is_free_openrouter_model(raw):
            continue
        if not _is_text_output_model(raw) or _is_specialized_non_writer_model(raw):
            continue
        if _expires_soon(raw.get("expiration_date")):
            continue
        candidates.append((_openrouter_model_score(raw), model_id))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    out: list[str] = []
    seen: set[str] = set()
    for _score, model_id in candidates:
        if model_id in seen:
            continue
        seen.add(model_id)
        out.append(model_id)
        if len(out) >= limit:
            break
    if "openrouter/free" not in seen:
        out.append("openrouter/free")
    return out


def _is_free_openrouter_model(model: dict[str, object]) -> bool:
    model_id = str(model.get("id") or "")
    pricing = model.get("pricing")
    if model_id.endswith(":free"):
        return True
    if not isinstance(pricing, dict):
        return False
    return str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0"


def _is_text_output_model(model: dict[str, object]) -> bool:
    architecture = model.get("architecture")
    if not isinstance(architecture, dict):
        return False
    outputs = architecture.get("output_modalities")
    inputs = architecture.get("input_modalities")
    return isinstance(outputs, list) and "text" in outputs and isinstance(inputs, list) and "text" in inputs


def _is_specialized_non_writer_model(model: dict[str, object]) -> bool:
    blob = " ".join(str(model.get(key) or "").lower() for key in ("id", "name", "description"))
    blocked = ("content safety", "guardrail", "moderation", "image generation", "tts", "embedding")
    if any(term in blob for term in blocked):
        return True
    architecture = model.get("architecture")
    modality = str(architecture.get("modality") if isinstance(architecture, dict) else "").lower()
    return "->text" not in modality


def _expires_soon(value: object) -> bool:
    if not value:
        return False
    try:
        expires = date.fromisoformat(str(value)[:10])
    except ValueError:
        return False
    today = datetime.now(timezone.utc).date()
    return expires <= today + timedelta(days=2)


def _openrouter_model_score(model: dict[str, object]) -> float:
    benchmarks = model.get("benchmarks")
    artificial = benchmarks.get("artificial_analysis") if isinstance(benchmarks, dict) else {}
    if not isinstance(artificial, dict):
        artificial = {}
    agentic = _floatish(artificial.get("agentic_index"))
    coding = _floatish(artificial.get("coding_index"))
    intelligence = _floatish(artificial.get("intelligence_index"))
    arena = _design_arena_score(benchmarks.get("design_arena") if isinstance(benchmarks, dict) else None)
    context = _floatish(model.get("context_length"))
    created = _floatish(model.get("created"))
    supported = model.get("supported_parameters")
    structured_bonus = 4.0 if isinstance(supported, list) and "structured_outputs" in supported else 0.0
    tool_bonus = 2.0 if isinstance(supported, list) and "tools" in supported else 0.0
    return (
        1.8 * agentic
        + 1.2 * coding
        + intelligence
        + 0.04 * arena
        + min(context / 8192.0, 64.0)
        + min(max(created - 1_700_000_000.0, 0.0) / 10_000_000.0, 10.0)
        + structured_bonus
        + tool_bonus
    )


def _design_arena_score(value: object) -> float:
    if not isinstance(value, list):
        return 0.0
    scores = [_floatish(item.get("elo")) for item in value if isinstance(item, dict)]
    return max(scores) if scores else 0.0


def _floatish(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def openrouter_rate_limit_fallback_limit() -> int:
    return _int("BLOGPIPE_OPENROUTER_RATE_LIMIT_FALLBACK_LIMIT", 4, 1, 12)


def openrouter_rate_limit_circuit_breaker_hits() -> int:
    return _int("BLOGPIPE_OPENROUTER_RATE_LIMIT_CIRCUIT_BREAKER_HITS", 3, 1, 24)


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    openrouter_base_url: str
    openrouter_api_key: str
    model: str
    model_primary: str
    model_legacy: str
    model_fast: str
    model_smart: str
    openrouter_free_models: list[str]
    model_by_task: dict[str, str]
    model_chain_default: list[str]
    model_chain_by_task: dict[str, list[str]]
    max_calls: int
    max_tokens: int
    temperature: float
    max_runtime_seconds: float
    fast_timeout_seconds: float
    smart_timeout_seconds: float
    openrouter_timeout_seconds: float


def contact_email() -> str:
    return _get("CONTACT_EMAIL", "anshuman264@gmail.com")


def user_agent() -> str:
    return f"blogpipe/0.2 (+https://anshu92.github.io; mailto:{contact_email()})"


def llm_config() -> LLMConfig:
    openrouter_base = _get("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
    openrouter_key = _get("OPENROUTER_API_KEY")
    gemini_key = _get("GEMINI_API_KEY")
    default_base = "https://generativelanguage.googleapis.com/v1beta/openai" if gemini_key else openrouter_base
    base = _get("BLOGPIPE_LLM_BASE_URL", default_base)
    key = _get("BLOGPIPE_LLM_API_KEY", gemini_key or openrouter_key)
    model_primary = _get("BLOGPIPE_LLM_MODEL")
    model_legacy = _get("BLOGPIPE_MODEL")
    default_model = DEFAULT_GEMINI_MODEL_FAST if gemini_key else "openrouter/free"
    model = model_primary or model_legacy or default_model
    model_fast = _get("BLOGPIPE_LLM_MODEL_FAST")
    model_smart = _get("BLOGPIPE_LLM_MODEL_SMART")
    openrouter_free_models = _csv("BLOGPIPE_OPENROUTER_FREE_MODELS")
    if not openrouter_free_models and openrouter_dynamic_free_models_enabled():
        openrouter_free_models = discover_openrouter_free_models(base_url=openrouter_base, api_key=openrouter_key)
    if not openrouter_free_models:
        openrouter_free_models = list(DEFAULT_OPENROUTER_FREE_MODELS)
    model_by_task = {
        task: override
        for task, env_name in LLM_TASK_ENV_VARS.items()
        if (override := _get(env_name))
    }
    model_chain_default = _csv("BLOGPIPE_LLM_CHAIN")
    model_chain_by_task = {
        task: chain
        for task, env_name in LLM_TASK_CHAIN_ENV_VARS.items()
        if (chain := _csv(env_name))
    }
    return LLMConfig(
        base_url=base.rstrip("/"),
        api_key=key,
        openrouter_base_url=openrouter_base.rstrip("/"),
        openrouter_api_key=openrouter_key,
        model=model,
        model_primary=model_primary,
        model_legacy=model_legacy,
        model_fast=model_fast,
        model_smart=model_smart,
        openrouter_free_models=openrouter_free_models,
        model_by_task=model_by_task,
        model_chain_default=model_chain_default,
        model_chain_by_task=model_chain_by_task,
        max_calls=_int("BLOGPIPE_LLM_MAX_CALLS", 6, 0, 20),
        max_tokens=_int("BLOGPIPE_LLM_MAX_TOKENS", 4096, 512, 12000),
        temperature=_float("BLOGPIPE_LLM_TEMPERATURE", 0.25, 0.0, 1.2),
        max_runtime_seconds=_float("BLOGPIPE_LLM_MAX_RUNTIME_SECONDS", 1500.0, 60.0, 1800.0),
        fast_timeout_seconds=_float("BLOGPIPE_LLM_FAST_TIMEOUT_SECONDS", 45.0, 10.0, 180.0),
        smart_timeout_seconds=_float("BLOGPIPE_LLM_SMART_TIMEOUT_SECONDS", 90.0, 15.0, 240.0),
        openrouter_timeout_seconds=_float("BLOGPIPE_LLM_OPENROUTER_TIMEOUT_SECONDS", 60.0, 30.0, 300.0),
    )


def dry_run_env() -> bool:
    return _get("BLOGPIPE_DRY_RUN", "0").lower() in {"1", "true", "yes", "on"}


def openalex_key() -> str:
    return _get("OPENALEX_API_KEY")


def semantic_scholar_key() -> str:
    return _get("SEMANTIC_SCHOLAR_API_KEY")


def aec_scholar_rss() -> str:
    return _get("AEC_SCHOLAR_RSS")


def min_papers() -> int:
    return _int("BLOGPIPE_MIN_PAPERS", 4, 0, 8)


def max_blogs() -> int:
    return _int("BLOGPIPE_MAX_BLOGS", 2, 0, 8)


def daily_primary_papers() -> int:
    return _int("BLOGPIPE_DAILY_PRIMARY_PAPERS", 4, 2, 5)


def daily_supporting_items() -> int:
    return _int("BLOGPIPE_DAILY_SUPPORTING_ITEMS", 2, 0, 4)


def min_signal_score() -> float:
    return _float("BLOGPIPE_MIN_SIGNAL_SCORE", 0.75, 0.0, 1.0)


def generic_phrase_max_density() -> float:
    return _float("BLOGPIPE_GENERIC_PHRASE_MAX_DENSITY", 0.015, 0.0, 0.1)


def profile_results() -> int:
    return _int("BLOGPIPE_PROFILE_RESULTS", 40, 5, 100)


def arxiv_max_retries() -> int:
    return _int("BLOGPIPE_ARXIV_MAX_RETRIES", 3, 0, 6)


def arxiv_retry_backoff_seconds() -> float:
    return _float("BLOGPIPE_ARXIV_RETRY_BACKOFF_SECONDS", 2.0, 0.5, 30.0)


def selector_candidates() -> int:
    return _int("BLOGPIPE_SELECTOR_CANDIDATES", 24, 8, 60)


def selector_max_tokens() -> int:
    return _int("BLOGPIPE_SELECTOR_MAX_TOKENS", 3200, 512, 12000)


def outline_max_tokens() -> int:
    return _int("BLOGPIPE_OUTLINE_MAX_TOKENS", 3200, 512, 12000)


def daily_min_words() -> int:
    return _int("BLOGPIPE_DAILY_MIN_WORDS", 1200, 300, 4000)


def sectionwise_drafting_enabled() -> bool:
    return _get("BLOGPIPE_SECTIONWISE_DRAFTING", "0").lower() in {"1", "true", "yes", "on"}


def openrouter_smart_fallback_enabled() -> bool:
    raw = os.environ.get("BLOGPIPE_OPENROUTER_SMART_FALLBACK", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(_get("OPENROUTER_API_KEY"))


def llm_rate_limit_cooldown_seconds() -> float:
    return _float("BLOGPIPE_LLM_RATE_LIMIT_COOLDOWN_SECONDS", 5.0, 0.0, 120.0)


def llm_retry_attempts() -> int:
    return _int("BLOGPIPE_LLM_RETRY_ATTEMPTS", 2, 1, 3)


def llm_slow_openrouter_min_budget_seconds() -> float:
    return _float("BLOGPIPE_LLM_SLOW_OPENROUTER_MIN_BUDGET_SECONDS", 180.0, 60.0, 600.0)


def outline_repair_error_threshold() -> int:
    return _int("BLOGPIPE_OUTLINE_REPAIR_ERROR_THRESHOLD", 4, 1, 12)


def agent_deep_dive_min_budget_seconds() -> float:
    return _float("BLOGPIPE_AGENT_DEEP_DIVE_MIN_BUDGET_SECONDS", 420.0, 0.0, 1800.0)


def openreview_venues() -> tuple[str, ...]:
    raw = _get("BLOGPIPE_OPENREVIEW_VENUES")
    if raw:
        venues = tuple(part.strip() for part in raw.split(",") if part.strip())
        if venues:
            return venues
    return (
        "ICLR.cc/2026/Conference",
        "ICLR.cc/2025/Conference",
        "NeurIPS.cc/2025/Conference",
        "ICML.cc/2026/Conference",
    )

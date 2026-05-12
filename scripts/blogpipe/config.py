from __future__ import annotations

import os
from dataclasses import dataclass

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


DEFAULT_OPENROUTER_FREE_MODELS = [
    "minimax/minimax-m2.5:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "arcee-ai/trinity-large-thinking:free",
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-coder:free",
    "inclusionai/ring-2.6-1t:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/free",
]


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


def contact_email() -> str:
    return _get("CONTACT_EMAIL", "anshuman264@gmail.com")


def user_agent() -> str:
    return f"blogpipe/0.2 (+https://anshu92.github.io; mailto:{contact_email()})"


def llm_config() -> LLMConfig:
    openrouter_base = _get("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
    openrouter_key = _get("OPENROUTER_API_KEY")
    base = _get("BLOGPIPE_LLM_BASE_URL", openrouter_base)
    key = _get("BLOGPIPE_LLM_API_KEY", _get("OPENROUTER_API_KEY"))
    model_primary = _get("BLOGPIPE_LLM_MODEL")
    model_legacy = _get("BLOGPIPE_MODEL")
    model = model_primary or model_legacy or "openrouter/free"
    model_fast = _get("BLOGPIPE_LLM_MODEL_FAST")
    model_smart = _get("BLOGPIPE_LLM_MODEL_SMART")
    openrouter_free_models = _csv("BLOGPIPE_OPENROUTER_FREE_MODELS") or list(DEFAULT_OPENROUTER_FREE_MODELS)
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

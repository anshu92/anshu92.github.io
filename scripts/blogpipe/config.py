"""Environment-driven configuration. No hardcoded secrets."""

from __future__ import annotations

import os


def _get(name: str, default: str = "") -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


def openrouter_base() -> str:
    return _get("OPENROUTER_BASE", "https://openrouter.ai/api/v1").rstrip("/")


def openrouter_key() -> str:
    return _get("OPENROUTER_API_KEY")


def openai_key() -> str:
    return _get("OPENAI_API_KEY")


def groq_key() -> str:
    return _get("GROQ_API_KEY")


def gemini_key() -> str:
    return _get("GEMINI_API_KEY")


def llm_configured() -> bool:
    return bool(groq_key() or gemini_key() or openrouter_key())


def semantic_scholar_key() -> str:
    return _get("SEMANTIC_SCHOLAR_API_KEY")


def fal_key() -> str:
    return _get("FAL_API_KEY")


def aec_scholar_rss() -> str:
    return _get("AEC_SCHOLAR_RSS", "")


def blogpipe_model() -> str:
    # Empty = use evr fast model chain; set to pin one OpenRouter model.
    return _get("BLOGPIPE_MODEL", "")


def editor_model() -> str:
    # Empty = use evr smart model chain; set to pin one OpenRouter model.
    return _get("BLOGPIPE_EDITOR_MODEL", "")


def embed_model() -> str:
    return _get("BLOGPIPE_EMBED_MODEL", "text-embedding-3-small")


def kroki_url() -> str:
    return _get("KROKI_URL", "http://127.0.0.1:8000").rstrip("/")


def dry_run() -> bool:
    return _get("BLOGPIPE_DRY_RUN", "0") in ("1", "true", "yes", "on")


def research_max_calls() -> int:
    try:
        return int(_get("BLOGPIPE_RESEARCH_MAX_CALLS", "15"))
    except ValueError:
        return 15


def editor_min_score() -> int:
    try:
        return int(_get("BLOGPIPE_EDITOR_MIN_SCORE", "8"))
    except ValueError:
        return 8


def editor_max_loops() -> int:
    try:
        return int(_get("BLOGPIPE_EDITOR_MAX_LOOPS", "2"))
    except ValueError:
        return 2


def pillar_floor() -> float:
    try:
        return float(_get("BLOGPIPE_PILLAR_FLOOR", "0.10"))
    except ValueError:
        return 0.1


def format_cooldown() -> int:
    try:
        return int(_get("BLOGPIPE_FORMAT_COOLDOWN", "5"))
    except ValueError:
        return 5


def art_cooldown() -> int:
    try:
        return int(_get("BLOGPIPE_ART_COOLDOWN", "4"))
    except ValueError:
        return 4


def force_format() -> str:
    return _get("BLOGPIPE_FORCE_FORMAT", "")


def mcp_enrichment_enabled() -> bool:
    return _get("BLOGPIPE_MCP_ENRICHMENT", "0") in ("1", "true", "yes", "on")


def brave_api_key() -> str:
    return _get("BRAVE_API_KEY") or _get("BRAVE_SEARCH_API_KEY")


def tavily_api_key() -> str:
    """Tavily offers a free developer tier; use for web search when Brave is not configured."""
    return _get("TAVILY_API_KEY")


def github_token() -> str:
    return _get("GITHUB_TOKEN")


def context7_api_key() -> str:
    return _get("CONTEXT7_API_KEY")

"""Environment-driven configuration. No hardcoded secrets."""

from __future__ import annotations

import json
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


def hf_token() -> str:
    """Hugging Face token for optional Inference API image fallbacks (FLUX, etc.)."""
    return _get("HF_TOKEN") or _get("HUGGING_FACE_HUB_TOKEN", "")


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


def rubric_floor_no_cites_max() -> int:
    """Max rubric score allowed when the body has no [cite: ] markers / inline citation links."""
    try:
        return int(_get("BLOGPIPE_RUBRIC_FLOOR_NO_CITES_MAX", "5"))
    except ValueError:
        return 5


def rubric_floor_no_numbers_max() -> int:
    """Max rubric score allowed when the body has no numeric claim with units/percentages."""
    try:
        return int(_get("BLOGPIPE_RUBRIC_FLOOR_NO_NUMBERS_MAX", "5"))
    except ValueError:
        return 5


def rubric_floor_low_util_max() -> int:
    """Max rubric score allowed when the evidence-utilization score is below threshold."""
    try:
        return int(_get("BLOGPIPE_RUBRIC_FLOOR_LOW_UTIL_MAX", "6"))
    except ValueError:
        return 6


def rubric_floor_low_util_threshold() -> float:
    """Evidence utilization score below which the rubric is capped."""
    try:
        return float(_get("BLOGPIPE_RUBRIC_FLOOR_LOW_UTIL_THRESHOLD", "0.20"))
    except ValueError:
        return 0.20


def rubric_floor_lint_max() -> int:
    """Max rubric score allowed when key structural lints fire (fake table, taxonomy, no tradeoffs, redundancy)."""
    try:
        return int(_get("BLOGPIPE_RUBRIC_FLOOR_LINT_MAX", "6"))
    except ValueError:
        return 6


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


def max_tokens_fast() -> int:
    try:
        v = int(_get("BLOGPIPE_MAX_TOKENS_FAST", "1536"))
    except ValueError:
        v = 1536
    return max(256, min(v, 8192))


def max_tokens_smart() -> int:
    try:
        v = int(_get("BLOGPIPE_MAX_TOKENS_SMART", "3072"))
    except ValueError:
        v = 3072
    return max(256, min(v, 8192))


def llm_call_cap() -> int:
    """Max successful+failed model completions per run (graph + chain hard stop)."""
    try:
        v = int(_get("BLOGPIPE_LLM_CALL_CAP", "90"))
    except ValueError:
        v = 90
    return max(1, min(v, 200))


_DEFAULT_STAGE_QUOTAS: dict[str, int] = {
    "committee": 15,
    "paper_reader": 8,
    "draft": 18,
    "editor": 4,
    "polish": 4,
    "extras": 4,
}


def stage_quotas() -> dict[str, int]:
    """Per-stage LLM call budgets (ok+fail); JSON env overrides defaults. Invalid JSON or keys fall back."""
    out = dict(_DEFAULT_STAGE_QUOTAS)
    raw = _get("BLOGPIPE_STAGE_QUOTAS", "")
    if not raw.strip():
        return out
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return out
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if not k:
            continue
        key = str(k).strip()
        if not key:
            continue
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n < 0:
            continue
        out[key] = n
    return out


def brave_api_key() -> str:
    return _get("BRAVE_API_KEY") or _get("BRAVE_SEARCH_API_KEY")


def tavily_api_key() -> str:
    """Tavily offers a free developer tier; use for web search when Brave is not configured."""
    return _get("TAVILY_API_KEY")


def github_token() -> str:
    return _get("GITHUB_TOKEN")


def context7_api_key() -> str:
    return _get("CONTEXT7_API_KEY")


def committee_enabled() -> bool:
    return _get("BLOGPIPE_COMMITTEE_DISABLED", "0") not in (
        "1",
        "true",
        "yes",
        "on",
    )


def committee_analysts() -> list[str]:
    raw = _get("BLOGPIPE_COMMITTEE_ANALYSTS", "")
    if not raw.strip():
        return [
            "methods",
            "empirical",
            "adversarial",
            "related",
            "practitioner",
            "code",
            "web",
            "glossary",
            "visual_planner",
        ]
    return [p.strip() for p in raw.split(",") if p.strip()]


def committee_per_analyst_max_tokens() -> int:
    try:
        v = int(_get("BLOGPIPE_COMMITTEE_PER_ANALYST_MAX_TOKENS", "800"))
    except ValueError:
        v = 800
    return max(256, min(v, 8192))


def supervisor_enabled() -> bool:
    """If false, run all config.committee_analysts (legacy); if true, LLM picks 2-4 optionals on top of core."""
    return _get("BLOGPIPE_SUPERVISOR", "1") in ("1", "true", "yes", "on")


def checkpointer_path() -> str:
    """SQLite file for LangGraph checkpointer. Use :memory: to disable on-disk (tests)."""
    p = _get("BLOGPIPE_CHECKPOINT_PATH", "")
    if p:
        return p
    from . import memory as _m

    return str(_m._ROOT / "cache" / "graph.sqlite")


def auto_approve_editor_gate() -> bool:
    """If true, do not call interrupt() when pass_gate is false (default for CI/PR)."""
    return _get("BLOGPIPE_AUTO_APPROVE", "1") in ("1", "true", "yes", "on")


def graph_stream_enabled() -> bool:
    """Stream node updates to logs; set 0 to use invoke()."""
    return _get("BLOGPIPE_STREAM", "1") in ("1", "true", "yes", "on")


def graph_thread_id_override() -> str:
    """If set, used as LangGraph thread_id (resume same run)."""
    return _get("BLOGPIPE_THREAD_ID", "")


def require_topic_match() -> bool:
    """If true (default), drop harvested items that match none of the configured topic themes."""
    return _get("BLOGPIPE_REQUIRE_TOPIC_MATCH", "1") in ("1", "true", "yes", "on")


def topic_relevance_weight() -> float:
    """Heuristic boost per matched theme in the ranker."""
    try:
        return max(0.0, float(_get("BLOGPIPE_TOPIC_RELEVANCE_WEIGHT", "0.6")))
    except ValueError:
        return 0.6


def prefer_free_models() -> bool:
    return _get("BLOGPIPE_PREFER_FREE", "1") in ("1", "true", "yes", "on")


def usd_budget() -> float:
    try:
        return max(0.0, float(_get("BLOGPIPE_USD_BUDGET", "0")))
    except ValueError:
        return 0.0


def model_overrides() -> dict[str, str]:
    raw = _get("BLOGPIPE_MODEL_OVERRIDES", "")
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(d, dict):
        return {}
    return {str(k): str(v) for k, v in d.items()}


def max_tokens_for_task(task: str) -> int:
    """Override max completion tokens; falls back to fast/smart by env."""
    o = _get(f"BLOGPIPE_MAX_TOKENS_{(task or '').upper()}", "")
    if o:
        try:
            return max(64, min(int(o), 8192))
        except ValueError:
            pass
    return 0

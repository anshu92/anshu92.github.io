"""Task → model selection for OpenRouter / Groq / Gemini. Pure dispatch, no I/O."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

from . import config


@dataclass(frozen=True)
class ModelProfile:
    """Catalog entry for a chat model (slug + provider in llm_chain)."""

    slug: str
    provider: str
    cost_in_per_mtok: float
    cost_out_per_mtok: float
    context_window: int
    capabilities: FrozenSet[str]


@dataclass(frozen=True)
class TaskProfile:
    name: str
    max_output: int
    required_caps: FrozenSet[str]
    free_chain: tuple[tuple[str, str], ...]
    paid_chain: tuple[tuple[str, str], ...]


_CAP_FAST = frozenset({"fast"})
_CAP_LC = frozenset({"long_context"})
_CAP_RE = frozenset({"reasoning"})
_CAP_ST = frozenset({"structured"})
_CAP_CD = frozenset({"code"})
_CAP_LC_RE = frozenset({"long_context", "reasoning"})

# Slug, provider, $/1M in, $/1M out, ctx max, caps. Free OpenRouter: cost 0.
_MODELS: dict[str, ModelProfile] = {
    "llama-3.1-8b-instant": ModelProfile(
        "llama-3.1-8b-instant", "groq", 0, 0, 128000, _CAP_FAST
    ),
    "llama-3.3-70b-versatile": ModelProfile(
        "llama-3.3-70b-versatile", "groq", 0, 0, 128000, _CAP_FAST
    ),
    "qwen/qwen3-32b": ModelProfile("qwen/qwen3-32b", "groq", 0, 0, 131072, _CAP_RE | _CAP_FAST),
    "meta-llama/llama-3.3-70b-instruct:free": ModelProfile(
        "meta-llama/llama-3.3-70b-instruct:free", "openrouter", 0, 0, 131072, _CAP_LC
    ),
    "deepseek/deepseek-r1-0528:free": ModelProfile(
        "deepseek/deepseek-r1-0528:free", "openrouter", 0, 0, 163840, _CAP_RE
    ),
    "qwen/qwen3-235b-thinking:free": ModelProfile(
        "qwen/qwen3-235b-thinking:free", "openrouter", 0, 0, 256000, _CAP_LC_RE
    ),
    "qwen/qwen3-next-80b-a3b-instruct:free": ModelProfile(
        "qwen/qwen3-next-80b-a3b-instruct:free", "openrouter", 0, 0, 262144, _CAP_LC
    ),
    "nvidia/nemotron-3-super-120b-a12b:free": ModelProfile(
        "nvidia/nemotron-3-super-120b-a12b:free", "openrouter", 0, 0, 1000000, _CAP_LC
    ),
    "qwen/qwen3-coder:free": ModelProfile(
        "qwen/qwen3-coder:free", "openrouter", 0, 0, 131072, _CAP_CD
    ),
    "openrouter/free": ModelProfile(
        "openrouter/free",
        "openrouter",
        0,
        0,
        200000,
        _CAP_FAST | _CAP_LC | _CAP_RE | _CAP_ST | _CAP_CD,
    ),
    "openai/gpt-5-nano": ModelProfile(
        "openai/gpt-5-nano", "openrouter", 0.05, 0.40, 400000, _CAP_FAST
    ),
    "google/gemini-2.0-flash-001": ModelProfile(
        "google/gemini-2.0-flash-001", "openrouter", 0.10, 0.40, 1000000, _CAP_LC
    ),
    "openai/gpt-4.1-mini": ModelProfile(
        "openai/gpt-4.1-mini", "openrouter", 0.40, 1.60, 1000000, _CAP_ST
    ),
    "google/gemini-2.5-flash": ModelProfile(
        "google/gemini-2.5-flash", "openrouter", 0.30, 2.50, 1000000, _CAP_LC
    ),
    "anthropic/claude-3-5-haiku-20241022": ModelProfile(
        "anthropic/claude-3-5-haiku-20241022", "openrouter", 1.00, 5.00, 200000, _CAP_RE | _CAP_ST
    ),
}


def _fc(*t: tuple[str, str]) -> tuple[tuple[str, str], ...]:
    return t


# Task names match call sites: query_gen, paper_chunk_summary, …
_TASKS: dict[str, TaskProfile] = {
    "query_gen": TaskProfile(
        "query_gen",
        400,
        _CAP_FAST,
        _fc(
            ("llama-3.1-8b-instant", "groq"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(("openai/gpt-5-nano", "openrouter"), ("llama-3.1-8b-instant", "groq")),
    ),
    "paper_chunk_summary": TaskProfile(
        "paper_chunk_summary",
        600,
        _CAP_LC,
        _fc(
            ("qwen/qwen3-next-80b-a3b-instruct:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("google/gemini-2.0-flash-001", "openrouter"),
            ("qwen/qwen3-next-80b-a3b-instruct:free", "openrouter"),
        ),
    ),
    "mcp_planner": TaskProfile(
        "mcp_planner",
        500,
        _CAP_RE,
        _fc(
            ("deepseek/deepseek-r1-0528:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("google/gemini-2.0-flash-001", "openrouter"),
            ("deepseek/deepseek-r1-0528:free", "openrouter"),
        ),
    ),
    "mcp_adversarial": TaskProfile(
        "mcp_adversarial",
        500,
        _CAP_RE,
        _fc(
            ("deepseek/deepseek-r1-0528:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("google/gemini-2.0-flash-001", "openrouter"),
            ("deepseek/deepseek-r1-0528:free", "openrouter"),
        ),
    ),
    "rank_reasoning": TaskProfile(
        "rank_reasoning",
        1500,
        _CAP_RE | _CAP_ST,
        _fc(
            ("llama-3.3-70b-versatile", "groq"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("openai/gpt-4.1-mini", "openrouter"),
            ("llama-3.3-70b-versatile", "groq"),
        ),
    ),
    "analyst_methods": TaskProfile(
        "analyst_methods",
        1200,
        _CAP_LC | _CAP_RE,
        _fc(
            ("qwen/qwen3-235b-thinking:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("google/gemini-2.5-flash", "openrouter"),
            ("qwen/qwen3-235b-thinking:free", "openrouter"),
        ),
    ),
    "analyst_empirical": TaskProfile(
        "analyst_empirical",
        900,
        _CAP_LC | _CAP_ST,
        _fc(
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("openai/gpt-4.1-mini", "openrouter"),
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
        ),
    ),
    "analyst_adversarial": TaskProfile(
        "analyst_adversarial",
        800,
        _CAP_RE,
        _fc(
            ("deepseek/deepseek-r1-0528:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("anthropic/claude-3-5-haiku-20241022", "openrouter"),
            ("deepseek/deepseek-r1-0528:free", "openrouter"),
        ),
    ),
    "analyst_related": TaskProfile(
        "analyst_related",
        900,
        _CAP_LC,
        _fc(
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("google/gemini-2.0-flash-001", "openrouter"),
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
        ),
    ),
    "analyst_practitioner": TaskProfile(
        "analyst_practitioner",
        600,
        _CAP_FAST,
        _fc(
            ("llama-3.1-8b-instant", "groq"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("google/gemini-2.0-flash-001", "openrouter"),
            ("llama-3.1-8b-instant", "groq"),
        ),
    ),
    "analyst_code": TaskProfile(
        "analyst_code",
        700,
        _CAP_CD,
        _fc(
            ("qwen/qwen3-coder:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("openai/gpt-4.1-mini", "openrouter"),
            ("qwen/qwen3-coder:free", "openrouter"),
        ),
    ),
    "analyst_web": TaskProfile(
        "analyst_web",
        600,
        _CAP_FAST,
        _fc(
            ("llama-3.1-8b-instant", "groq"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("openai/gpt-5-nano", "openrouter"),
            ("llama-3.1-8b-instant", "groq"),
        ),
    ),
    "analyst_glossary": TaskProfile(
        "analyst_glossary",
        1000,
        _CAP_LC | _CAP_ST,
        _fc(
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("openai/gpt-4.1-mini", "openrouter"),
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
        ),
    ),
    "visual_planner": TaskProfile(
        "visual_planner",
        1500,
        _CAP_LC | _CAP_ST,
        _fc(
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("openai/gpt-4.1-mini", "openrouter"),
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
        ),
    ),
    "explainer_rewrite": TaskProfile(
        "explainer_rewrite",
        2000,
        _CAP_LC,
        _fc(
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
            ("llama-3.3-70b-versatile", "groq"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("google/gemini-2.0-flash-001", "openrouter"),
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
        ),
    ),
    "committee_synthesis": TaskProfile(
        "committee_synthesis",
        1800,
        _CAP_LC | _CAP_RE,
        _fc(
            ("qwen/qwen3-235b-thinking:free", "openrouter"),
            ("nvidia/nemotron-3-super-120b-a12b:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("anthropic/claude-3-5-haiku-20241022", "openrouter"),
            ("qwen/qwen3-235b-thinking:free", "openrouter"),
        ),
    ),
    "draft_full": TaskProfile(
        "draft_full",
        4096,
        _CAP_LC,
        _fc(
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("google/gemini-2.5-flash", "openrouter"),
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
        ),
    ),
    "draft_section_critic": TaskProfile(
        "draft_section_critic",
        600,
        _CAP_RE,
        _fc(
            ("qwen/qwen3-32b", "groq"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("google/gemini-2.0-flash-001", "openrouter"),
            ("qwen/qwen3-32b", "groq"),
        ),
    ),
    "draft_rewrite_section": TaskProfile(
        "draft_rewrite_section",
        1200,
        _CAP_FAST,
        _fc(
            ("llama-3.3-70b-versatile", "groq"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("google/gemini-2.0-flash-001", "openrouter"),
            ("llama-3.3-70b-versatile", "groq"),
        ),
    ),
    "draft_cite_repair": TaskProfile(
        "draft_cite_repair",
        1500,
        _CAP_ST,
        _fc(
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("openai/gpt-4.1-mini", "openrouter"),
            ("meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
        ),
    ),
    "editor_rubric": TaskProfile(
        "editor_rubric",
        900,
        _CAP_RE,
        _fc(
            ("deepseek/deepseek-r1-0528:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("google/gemini-2.5-flash", "openrouter"),
            ("deepseek/deepseek-r1-0528:free", "openrouter"),
        ),
    ),
    "editor_grounding": TaskProfile(
        "editor_grounding",
        900,
        _CAP_RE | _CAP_ST,
        _fc(
            ("deepseek/deepseek-r1-0528:free", "openrouter"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("anthropic/claude-3-5-haiku-20241022", "openrouter"),
            ("deepseek/deepseek-r1-0528:free", "openrouter"),
        ),
    ),
    "supervisor_route": TaskProfile(
        "supervisor_route",
        600,
        _CAP_FAST | _CAP_ST,
        _fc(
            ("llama-3.1-8b-instant", "groq"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("openai/gpt-4.1-mini", "openrouter"),
            ("llama-3.1-8b-instant", "groq"),
        ),
    ),
    "keyword_extract": TaskProfile(
        "keyword_extract",
        400,
        _CAP_FAST | _CAP_ST,
        _fc(
            ("llama-3.1-8b-instant", "groq"),
            ("openrouter/free", "openrouter"),
        ),
        _fc(
            ("openai/gpt-4.1-mini", "openrouter"),
            ("llama-3.1-8b-instant", "groq"),
        ),
    ),
}

_DEFAULT = _TASKS["draft_full"]


def get_task_profile(task: str) -> TaskProfile:
    return _TASKS.get(task) or _DEFAULT


def _ctx_ok(
    m: ModelProfile, est_in: int
) -> bool:
    return int(est_in) < int(m.context_window * 0.85)


def _profile_for_slug(slug: str, provider: str) -> ModelProfile:
    p = _MODELS.get(slug)
    if p is not None:
        return p
    return ModelProfile(
        slug, provider, 0, 0, 256_000, frozenset(["fast", "long_context", "reasoning", "code", "structured"])
    )


def select_chain(
    task: str,
    est_input_tokens: int,
    *,
    prefer_free: bool,
    usd_budget_remaining: float,
) -> list[tuple[str, str]]:
    """Return ordered (model, provider) pairs: OpenRouter and Groq slugs."""
    tp = get_task_profile(task)
    ovr = config.model_overrides().get(task)
    est_in = max(0, int(est_input_tokens))
    out = tp.max_output

    base: list[tuple[str, str]] = []
    if ovr and ovr.strip():
        base.append((ovr.strip(), "openrouter"))

    if prefer_free and usd_budget_remaining <= 0.0:
        base.extend(list(tp.free_chain))
    else:
        est = estimate_cost_for_chain_first_paid(
            [tp.paid_chain[0]], est_in, out, task=task
        )
        if usd_budget_remaining > 0.0 and est <= usd_budget_remaining + 1e-9:
            base.extend(list(tp.paid_chain))
        base.extend(list(tp.free_chain))

    # Dedupe, filter
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for slug, prov in base:
        key = (slug, prov)
        if key in seen:
            continue
        prof = _profile_for_slug(slug, prov)
        if not _ctx_ok(prof, est_in):
            continue
        seen.add(key)
        result.append(key)

    if not result and base:
        result = [("openrouter/free", "openrouter")]
    return result


def estimate_cost_for_chain_first_paid(
    chain: list[tuple[str, str]], in_tok: int, out_tok: int, task: str = "draft_full"
) -> float:
    _ = task
    for pair in chain:
        s, p = pair
        m = _MODELS.get(s)
        if p == "openrouter" and m and (m.cost_in_per_mtok > 0 or m.cost_out_per_mtok > 0):
            return (
                (in_tok / 1_000_000.0) * m.cost_in_per_mtok
                + (out_tok / 1_000_000.0) * m.cost_out_per_mtok
            )
    return 0.0


def usd_for_tokens(slug: str, in_tok: int, out_tok: int) -> float:
    m = _MODELS.get(slug) or _profile_for_slug(
        slug, "openrouter" if "/" in slug else "groq"
    )
    return (in_tok / 1_000_000.0) * m.cost_in_per_mtok + (
        out_tok / 1_000_000.0
    ) * m.cost_out_per_mtok


def estimate_cost_usd(task: str, in_tokens: int, out_tokens: int) -> float:
    """Rough USD using first paid model in the task’s paid chain, else 0."""
    tp = get_task_profile(task)
    for pair in list(tp.paid_chain) + list(tp.free_chain)[:1]:
        r = estimate_cost_for_chain_first_paid([pair], in_tokens, out_tokens, task=task)
        if r > 0 or _MODELS.get(pair[0], ModelProfile("x", "openrouter", 0, 0, 0, frozenset())).cost_in_per_mtok:
            s, p = pair
            m = _MODELS.get(s)
            if m:
                return (
                    (in_tokens / 1_000_000.0) * m.cost_in_per_mtok
                    + (out_tokens / 1_000_000.0) * m.cost_out_per_mtok
                )
    return 0.0

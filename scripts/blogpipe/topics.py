"""Topic themes the blog focuses on.

Single source of truth for what blogpipe considers an in-scope topic. Used by:
- curator (`_classify_pillar` falls back to themed keywords)
- rank (heuristic boost + on/off-topic filter + LLM ranker system prompt)

Each theme is a short id + display label + keyword list (lowercased substrings)
and a default Pillar hint. Themes can overlap; an item is "in-scope" if it
matches at least one theme keyword.

Learned keywords (extracted from accepted drafts) are stored in
``cache/topic_keywords.json`` and merged in at lookup time, capped per theme
so the registry cannot grow unbounded.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .models import Pillar

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Theme:
    id: str
    label: str
    pillar: Pillar
    keywords: tuple[str, ...]


THEMES: tuple[Theme, ...] = (
    Theme(
        id="llm_scaling",
        label="Scaling LLM training and inference (architecture, optimization, infra)",
        pillar=Pillar.systems,
        keywords=(
            "llm", "language model", "transformer", "attention",
            "kv cache", "kv-cache", "speculative decoding", "paged attention",
            "throughput", "latency", "tokens per second", "tokens/s",
            "tensor parallel", "pipeline parallel", "sequence parallel",
            "expert parallel", "model parallel", "data parallel",
            "gpu", "tpu", "hopper", "h100", "h200", "blackwell", "mi300", "mi325",
            "vllm", "tensorrt", "triton", "cuda", "rocm", "fp8", "fp4", "int4",
            "deepspeed", "megatron", "fsdp", "zero", "nccl", "rdma", "infiniband",
            "moe", "mixture of experts", "router", "load balanc",
            "ring attention", "flashattention", "flash-attention",
            "inference engine", "kernel", "cutlass",
        ),
    ),
    Theme(
        id="training_recipes",
        label="Training techniques, recipes, frameworks, pretraining, post-training, RL, datasets",
        pillar=Pillar.research,
        keywords=(
            "pretrain", "pre-train", "post-train", "post training",
            "instruction tun", "sft ", "supervised fine", "fine-tun", "fine tun",
            "lora", "qlora", "dora", "peft", "adapter",
            "rlhf", "dpo", "ipo", "kto", "rloo", "grpo", "ppo", "rpo",
            "preference data", "reward model",
            "data mix", "dataset curation", "data curation", "data pipeline",
            "tokenizer", "byte pair", "bpe", "deduplication", "decontamination",
            "quality filter", "synthetic data", "self-play", "self play",
            "constitutional", "rejection sampling",
            "distillation", "knowledge distill", "annealing", "cooldown phase",
            "training framework", "torchtitan", "axolotl", "trl", "verl",
        ),
    ),
    Theme(
        id="aec_ml",
        label="ML in Architecture, Engineering and Construction (spatial/semantic reasoning, BIM)",
        pillar=Pillar.aec,
        keywords=(
            "aec", "bim", "ifc", "construction", "civil engineering",
            "structural engineering", "mep ", "facility",
            "autodesk", "revit", "rhino", "grasshopper", "navisworks",
            "point cloud", "lidar", "scan-to-bim", "scan to bim", "scan-to-mesh",
            "cad", "geometry", "mesh", "voxel", "nerf", "gaussian splat",
            "spatial reasoning", "spatial graph", "scene graph",
            "generative design", "topology optimization",
            "building", "floorplan", "floor plan", "site", "infrastructure",
            "digital twin", "urban",
        ),
    ),
    Theme(
        id="new_models",
        label="New models, architectures, benchmarks, novel approaches",
        pillar=Pillar.research,
        keywords=(
            "we propose", "we introduce", "novel architecture", "new architecture",
            "state-of-the-art", "state of the art", "sota",
            "outperforms", "ablation",
            "benchmark", "leaderboard", "eval suite", "evaluation suite",
            "mmlu", "gpqa", "humaneval", "swe-bench", "math500", "aime",
            "long context", "128k", "1m token", "million token",
            "diffusion", "flow matching", "rectified flow", "dit",
            "ssm ", "mamba", "state space model", "rwkv", "retnet",
            "vision-language", "vlm", "multimodal", "speech model",
            "world model",
        ),
    ),
    Theme(
        id="problem_oriented_ml",
        label="Problem-oriented ML: unique uses of ML to solve real problems",
        pillar=Pillar.applied,
        keywords=(
            "agent", "agentic", "tool use", "tool-use", "function call",
            "code generation", "code agent", "code assistant",
            "rag ", "retrieval-augmented", "retrieval augmented",
            "robotics", "manipulation", "vla ", "vision-language-action",
            "biology", "protein", "drug discovery", "materials",
            "weather", "climate", "earth", "remote sensing", "satellite",
            "anomaly detection", "fraud", "recommend",
            "search ranking", "personalization",
            "real-world deployment", "production", "case study",
        ),
    ),
    Theme(
        id="ml_engineering_deep_dive",
        label="Deeply technical ML engineering discussions",
        pillar=Pillar.systems,
        keywords=(
            "profiling", "profiler", "nsight", "ncu", "perf trace",
            "memory layout", "tiling", "fusion", "kernel fusion",
            "sharding", "checkpoint", "gradient accumulation",
            "mixed precision", "bf16", "fp16",
            "compile", "torch.compile", "jit", "xla",
            "engineering", "infra", "infrastructure",
            "observability", "telemetry",
            "internals", "deep dive", "from scratch", "from-scratch",
        ),
    ),
)


_LABELS = {t.id: t.label for t in THEMES}
_THEME_IDS = {t.id for t in THEMES}

# --- Learned-keyword store -------------------------------------------------

# Cap learned keywords per theme so the file never grows unbounded.
MAX_LEARNED_PER_THEME = 200
# Cap how many keywords a single draft can contribute across themes.
MAX_LEARNED_PER_DRAFT = 12
# Length bounds for an acceptable keyword.
_KW_MIN_LEN = 3
_KW_MAX_LEN = 40

_KW_OK = re.compile(r"^[a-z0-9][a-z0-9 \-/.+]{1,38}[a-z0-9.+]$")
_KW_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "into", "over", "post",
    "blog", "paper", "model", "models", "method", "methods", "approach", "approaches",
    "result", "results", "section", "figure", "figures", "table", "tables",
    "example", "examples", "open", "closed", "general", "specific", "system",
    "neural", "network", "learning", "machine", "training", "inference",
    # built-ins like "llm" stay as built-ins; nothing to learn from these
}


def _learned_path() -> Path:
    """`cache/topic_keywords.json` (lazy import avoids a circular config dep)."""
    from . import memory

    return memory.CACHE / "topic_keywords.json"


def _load_learned() -> dict[str, list[str]]:
    p = _learned_path()
    if not p.is_file():
        return {tid: [] for tid in _THEME_IDS}
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        LOG.warning("topics: unreadable %s: %s", p, e)
        return {tid: [] for tid in _THEME_IDS}
    out: dict[str, list[str]] = {tid: [] for tid in _THEME_IDS}
    if not isinstance(raw, dict):
        return out
    for tid, entry in raw.items():
        if tid not in _THEME_IDS or not isinstance(entry, dict):
            continue
        kws = entry.get("keywords") or []
        if isinstance(kws, list):
            out[tid] = [str(k) for k in kws if isinstance(k, str)]
    return out


def _save_learned(data: dict[str, list[str]]) -> None:
    p = _learned_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        tid: {
            "keywords": data.get(tid, [])[:MAX_LEARNED_PER_THEME],
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        for tid in _THEME_IDS
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def effective_keywords(theme_id: str) -> list[str]:
    """Built-in keywords + learned keywords for a theme (deduped, lowercased)."""
    by_id = {t.id: t for t in THEMES}
    if theme_id not in by_id:
        return []
    builtin = list(by_id[theme_id].keywords)
    learned = _load_learned().get(theme_id, [])
    seen: set[str] = set()
    out: list[str] = []
    for kw in builtin + learned:
        k = kw.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def all_known_keywords() -> set[str]:
    """Union of every effective keyword across all themes (for dedupe in extraction)."""
    s: set[str] = set()
    for t in THEMES:
        for k in effective_keywords(t.id):
            s.add(k)
    return s


# --- Item classification ---------------------------------------------------


def _blob(item) -> str:
    """Lowercased searchable text for an `Item`-shaped object."""
    title = getattr(item, "title", "") or ""
    abstract = getattr(item, "abstract", "") or ""
    tags = getattr(item, "tags", None) or []
    if not isinstance(tags, list):
        tags = []
    return (
        f"{title}\n{abstract[:2000]}\n{' '.join(str(x) for x in tags)}"
    ).lower()


def matched_themes(item) -> list[str]:
    """Theme ids whose keywords appear in the item's title/abstract/tags."""
    text = _blob(item)
    out: list[str] = []
    for t in THEMES:
        for kw in effective_keywords(t.id):
            if kw and kw in text:
                out.append(t.id)
                break
    return out


def is_in_scope(item) -> bool:
    """True iff the item matches at least one theme keyword."""
    return bool(matched_themes(item))


def relevance_score(item) -> float:
    """Coverage score in [0,1]: fraction of THEMES the item touches."""
    return len(matched_themes(item)) / float(len(THEMES))


def primary_pillar_hint(item) -> Pillar | None:
    """First matched theme's pillar, useful as a default classification."""
    matches = matched_themes(item)
    if not matches:
        return None
    by_id = {t.id: t for t in THEMES}
    return by_id[matches[0]].pillar


def themes_prompt_block() -> str:
    """Human-readable list for LLM system prompts."""
    lines = ["Allowed topic themes (the post MUST fit at least one):"]
    for t in THEMES:
        lines.append(f"- {t.id}: {t.label}")
    return "\n".join(lines)


def labels() -> dict[str, str]:
    """`theme_id -> human label`."""
    return dict(_LABELS)


# --- Learning from drafts --------------------------------------------------


def _normalize_kw(s: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    s = s.strip(" .,;:!?\"'`()[]{}")
    return s


def _is_acceptable_kw(s: str) -> bool:
    if not (_KW_MIN_LEN <= len(s) <= _KW_MAX_LEN):
        return False
    if s in _KW_STOPWORDS:
        return False
    if s.isnumeric():
        return False
    return bool(_KW_OK.match(s))


def _strip_markdown(body: str) -> str:
    """Best-effort plaintext for keyword extraction (drop fences and images)."""
    body = re.sub(r"```[\s\S]*?```", " ", body)
    body = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", body)
    body = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)
    body = re.sub(r"^---[\s\S]*?---\s*\n", "", body, count=1)
    return body


def _extract_with_llm(plain: str, primary_title: str) -> list[str]:
    """Ask a cheap LLM for technical keyword phrases from the draft body."""
    from . import config, openrouter_client

    if config.dry_run() or not config.llm_configured():
        return []
    system = (
        "Extract the technical keywords most distinctive of this ML/engineering blog post. "
        "Return JSON only: {\"keywords\": [\"...\"]}. "
        "Each keyword: 1-4 words, lowercase, technical (e.g. 'paged attention', 'fp8', "
        "'rope scaling', 'scan-to-bim'). No generic words like 'model' or 'training'. "
        "Max 15 keywords."
    )
    user = f"TITLE: {primary_title[:200]}\n\nBODY:\n{plain[:6000]}"
    try:
        raw = openrouter_client.llm_text(
            system, user, max_tokens=400, task="keyword_extract"
        )
    except (RuntimeError, ValueError, OSError) as e:
        LOG.info("topics: llm extract failed: %s", e)
        return []
    m = re.search(r"\{[\s\S]*\}", raw or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    kws = data.get("keywords") if isinstance(data, dict) else None
    if not isinstance(kws, list):
        return []
    return [str(k) for k in kws if isinstance(k, str)]


def add_learned_keywords(theme_id: str, kws: list[str]) -> list[str]:
    """Persist new keywords for a theme; returns the list actually added."""
    if theme_id not in _THEME_IDS or not kws:
        return []
    known = all_known_keywords()
    store = _load_learned()
    existing = list(store.get(theme_id, []))
    added: list[str] = []
    for raw in kws:
        kw = _normalize_kw(raw)
        if not _is_acceptable_kw(kw):
            continue
        if kw in known or kw in existing:
            continue
        existing.append(kw)
        added.append(kw)
        known.add(kw)
        if len(existing) >= MAX_LEARNED_PER_THEME:
            break
    if added:
        store[theme_id] = existing[:MAX_LEARNED_PER_THEME]
        _save_learned(store)
    return added


def update_themes_from_draft(body: str, primary_item) -> dict[str, list[str]]:
    """Extract keywords from a finished draft and append to the related themes.

    "Related themes" = themes the primary item already matches. If none match,
    we fall back to the themes the body itself triggers, so the registry can
    still grow when the source item was tagged loosely.

    Returns a dict of `theme_id -> keywords appended` for telemetry/logging.
    """
    if not (body or "").strip():
        return {}
    target = matched_themes(primary_item) or matched_themes(
        type("X", (), {"title": "", "abstract": body[:4000], "tags": []})()
    )
    if not target:
        LOG.info("topics: no related themes for draft; skipping keyword learning")
        return {}
    plain = _strip_markdown(body)
    primary_title = getattr(primary_item, "title", "") or ""
    raw_kws = _extract_with_llm(plain, primary_title)
    if not raw_kws:
        return {}
    raw_kws = raw_kws[:MAX_LEARNED_PER_DRAFT]
    out: dict[str, list[str]] = {}
    for tid in target:
        added = add_learned_keywords(tid, raw_kws)
        if added:
            out[tid] = added
    if out:
        LOG.info(
            "topics: learned %d keywords across %d themes from draft",
            sum(len(v) for v in out.values()),
            len(out),
        )
    return out

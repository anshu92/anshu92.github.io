from __future__ import annotations

from pathlib import Path

from . import extract, memory
from .models import EvidenceChunk, EvidencePack, RankedItem


def build_daily_pack(ranked: list[RankedItem], *, prior_limit: int = 5) -> EvidencePack:
    chunks: list[EvidenceChunk] = []
    for item in ranked:
        chunks.extend(_chunks_for_item(item, len(chunks), max_chunks=3))
    return EvidencePack(kind="daily", ranked_items=ranked, chunks=chunks, prior_posts=_prior_posts(prior_limit))


def build_deep_dive_pack(ranked: RankedItem) -> EvidencePack:
    return EvidencePack(
        kind="deep_dive",
        ranked_items=[ranked],
        chunks=_chunks_for_item(ranked, 0, max_chunks=10),
        prior_posts=_prior_posts(5),
    )


def _chunks_for_item(ranked: RankedItem, offset: int, *, max_chunks: int) -> list[EvidenceChunk]:
    item = ranked.item
    text = "\n".join(
        part for part in (item.abstract_or_excerpt, item.body_text) if part
    )
    sentences = extract.sentence_split(text)
    if not sentences and text:
        sentences = [text[:900]]
    scored = sorted(sentences, key=_sentence_value, reverse=True)
    selected = _select_evidence_sentences(scored, max_chunks=max_chunks)
    out: list[EvidenceChunk] = []
    for idx, (sentence, evidence_type) in enumerate(selected, start=1):
        out.append(
            EvidenceChunk(
                evidence_id=f"E{offset + idx}",
                item_id=item.item_id,
                title=item.title,
                url=item.canonical_url,
                text=sentence[:1100],
                section=f"abstract_or_body:{evidence_type}",
                evidence_type=evidence_type,
            )
        )
    return out


def _sentence_value(sentence: str) -> tuple[int, int]:
    lower = sentence.lower()
    cues = sum(1 for cue in _all_cues() if cue in lower)
    return len(_evidence_types(sentence)), cues, len(sentence)


def _select_evidence_sentences(sentences: list[str], *, max_chunks: int) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for evidence_type in ("mechanism", "math_or_objective", "experiment", "limitation", "impact"):
        for sentence in sentences:
            if sentence in seen:
                continue
            if evidence_type in _evidence_types(sentence):
                selected.append((sentence, evidence_type))
                seen.add(sentence)
                break
        if len(selected) >= max_chunks:
            return selected
    for sentence in sentences:
        if sentence in seen:
            continue
        selected.append((sentence, _primary_evidence_type(sentence)))
        seen.add(sentence)
        if len(selected) >= max_chunks:
            break
    return selected


def _primary_evidence_type(sentence: str) -> str:
    types = _evidence_types(sentence)
    return types[0] if types else "context"


def _evidence_types(sentence: str) -> list[str]:
    lower = sentence.lower()
    out: list[str] = []
    for evidence_type, cues in _CUES_BY_TYPE.items():
        if any(cue in lower for cue in cues):
            out.append(evidence_type)
    return out


_CUES_BY_TYPE: dict[str, tuple[str, ...]] = {
    "mechanism": (
        "algorithm", "architecture", "method", "pipeline", "framework", "training",
        "inference", "retrieval", "cache", "graph", "model", "we propose", "we introduce",
    ),
    "math_or_objective": (
        "objective", "loss", "equation", "theorem", "bound", "gradient", "complexity",
        "o(", "optimization", "regularization", "likelihood", "posterior", "reward",
    ),
    "experiment": (
        "benchmark", "ablation", "result", "accuracy", "dataset", "evaluation",
        "experiment", "tasks", "throughput", "latency", "%", "table", "compare",
    ),
    "limitation": (
        "limitation", "caveat", "failure", "error", "tradeoff", "robustness",
        "sensitivity", "future work", "fails",
    ),
    "impact": (
        "deployment", "production", "cost", "practical", "facility", "building",
        "operations", "reproducibility", "monitoring", "serving", "open source", "code",
    ),
}


def _all_cues() -> tuple[str, ...]:
    return tuple(cue for cues in _CUES_BY_TYPE.values() for cue in cues)


def _prior_posts(limit: int) -> list[dict[str, str]]:
    posts: list[dict[str, str]] = []
    for path in sorted(memory.CONTENT_POST.glob("*.md"), reverse=True)[:limit]:
        posts.append({"path": str(path.relative_to(memory.ROOT)), "title": _title(path)})
    return posts


def _title(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return path.stem
    for line in text.splitlines()[:20]:
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip().strip('"')
    return path.stem

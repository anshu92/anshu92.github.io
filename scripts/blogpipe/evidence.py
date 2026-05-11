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
    scored = sorted(sentences, key=_sentence_value, reverse=True)[:max_chunks]
    out: list[EvidenceChunk] = []
    for idx, sentence in enumerate(scored, start=1):
        out.append(
            EvidenceChunk(
                evidence_id=f"E{offset + idx}",
                item_id=item.item_id,
                title=item.title,
                url=item.canonical_url,
                text=sentence[:1100],
                section="abstract_or_body",
            )
        )
    return out


def _sentence_value(sentence: str) -> tuple[int, int]:
    lower = sentence.lower()
    cues = sum(
        1
        for cue in (
            "benchmark", "ablation", "latency", "throughput", "dataset", "method",
            "we propose", "we introduce", "code", "result", "accuracy", "%",
        )
        if cue in lower
    )
    return cues, len(sentence)


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

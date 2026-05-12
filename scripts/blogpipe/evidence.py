from __future__ import annotations

from pathlib import Path

from . import extract, memory
from .models import EvidenceCard, EvidenceChunk, EvidencePack, RankedItem


def build_daily_pack(ranked: list[RankedItem], *, prior_limit: int = 5) -> EvidencePack:
    chunks: list[EvidenceChunk] = []
    for item in ranked:
        max_chunks = 5 if _selector_role(item) == "primary" else 2
        chunks.extend(_chunks_for_item(item, len(chunks), max_chunks=max_chunks))
    return EvidencePack(
        kind="daily",
        ranked_items=ranked,
        chunks=chunks,
        evidence_cards=_evidence_cards(ranked, chunks),
        prior_posts=_prior_posts(prior_limit),
    )


def build_deep_dive_pack(ranked: RankedItem) -> EvidencePack:
    return EvidencePack(
        kind="deep_dive",
        ranked_items=[ranked],
        chunks=_chunks_for_item(ranked, 0, max_chunks=10),
        evidence_cards=[],
        prior_posts=_prior_posts(5),
    )


def _evidence_cards(ranked: list[RankedItem], chunks: list[EvidenceChunk]) -> list[EvidenceCard]:
    chunks_by_item: dict[str, list[EvidenceChunk]] = {}
    for chunk in chunks:
        chunks_by_item.setdefault(chunk.item_id, []).append(chunk)
    cards: list[EvidenceCard] = []
    for ranked_item in ranked:
        item = ranked_item.item
        item_chunks = chunks_by_item.get(item.item_id, [])
        by_type: dict[str, list[EvidenceChunk]] = {}
        for chunk in item_chunks:
            by_type.setdefault(chunk.evidence_type or "context", []).append(chunk)
        problem = _best_sentence(item_chunks, ("problem", "challenge", "gap", "lack", "bottleneck")) or item.abstract_or_excerpt[:500]
        mechanism = _chunk_text(by_type, "mechanism")
        math_or_objective = _chunk_text(by_type, "math_or_objective")
        experiment = _chunk_text(by_type, "experiment")
        limitation = _chunk_text(by_type, "limitation")
        impact = _chunk_text(by_type, "impact")
        role = _selector_role(ranked_item)
        cards.append(
            EvidenceCard(
                item_id=item.item_id,
                title=item.title,
                url=item.canonical_url,
                role=role,
                problem=problem or "not found in evidence",
                mechanism=mechanism,
                math_or_objective=math_or_objective,
                experiment=experiment,
                limitation=limitation,
                impact=impact,
                paper_supported_claim=_paper_supported_claim(problem, mechanism, experiment, impact),
                paper_supported_limit=_paper_supported_limit(limitation),
                transfer_hypothesis=_transfer_hypothesis(item),
                open_research_question=_open_research_question(item, mechanism, experiment, limitation),
                evidence_ids={
                    evidence_type: [chunk.evidence_id for chunk in typed_chunks]
                    for evidence_type, typed_chunks in sorted(by_type.items())
                },
            )
        )
    return cards


def _paper_supported_claim(problem: str, mechanism: str, experiment: str, impact: str) -> str:
    for candidate in (mechanism, experiment, impact, problem):
        if candidate and candidate != "not found in evidence":
            return candidate
    return "not found in evidence"


def _paper_supported_limit(limitation: str) -> str:
    return limitation if limitation and limitation != "not found in evidence" else "not found in evidence"


def _selector_role(ranked: RankedItem) -> str:
    role = str(ranked.item.extra.get("selector_role", "") or "").lower()
    return "supporting" if role == "supporting" else "primary"


def _chunk_text(by_type: dict[str, list[EvidenceChunk]], evidence_type: str) -> str:
    chunks = by_type.get(evidence_type, [])
    return chunks[0].text if chunks else "not found in evidence"


def _best_sentence(chunks: list[EvidenceChunk], cues: tuple[str, ...]) -> str:
    for chunk in chunks:
        lower = chunk.text.lower()
        if any(cue in lower for cue in cues):
            return chunk.text
    return chunks[0].text if chunks else ""


def _transfer_hypothesis(item) -> str:
    blob = f"{item.title} {item.abstract_or_excerpt} {' '.join(item.tags)}".lower()
    if any(cue in blob for cue in ("drawing", "sheet", "pdf", "ocr", "layout", "document")):
        return "Directly relevant to 2D document intelligence workflows."
    if any(cue in blob for cue in ("cad", "bim", "ifc", "revit", "construction", "building")):
        return "Directly relevant to AEC model and building-data workflows."
    if any(cue in blob for cue in ("evaluation", "benchmark", "retrieval", "rag", "agent")):
        return "Transferable as an evaluation or agent-system pattern for AEC document products."
    if any(cue in blob for cue in ("inference", "latency", "throughput", "serving", "quantization")):
        return "Transferable as production infrastructure for document-scale model serving."
    return "Potentially relevant as adjacent ML research; validate transfer before adoption."


def _open_research_question(item, mechanism: str, experiment: str, limitation: str) -> str:
    if limitation and limitation != "not found in evidence":
        return f"Open question: validate whether the reported limitation transfers to AEC or 2D-document workflows: {limitation}"
    blob = f"{item.title} {mechanism} {experiment}".strip()
    if blob:
        return "Open question: validate whether this paper's mechanism remains useful under AEC document distributions and evaluation constraints."
    return "Open question: more source-grounded evidence is needed before deriving an AEC transfer claim."


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

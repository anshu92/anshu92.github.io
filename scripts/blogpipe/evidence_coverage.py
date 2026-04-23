"""Measure how much of an EvidenceBundle the draft body actually uses.

The pipeline collects rich research context (benchmarks, related papers, analyst
claims, paper quotes) but the writer often skims it. This module produces a
deterministic, no-LLM coverage report that:

- powers the rubric floor (low utilization caps `rubric_score`),
- feeds the specificity-rewriter so it can pull in the unused facts,
- surfaces the unused items in `editor_report.json` for visibility.

The four buckets are weighted to penalise shallow surfacing of evidence:
    score = 0.4*numbers + 0.3*entities + 0.2*analyst_claims + 0.1*quotes
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .models import EvidenceBundle


# Generic words we should not credit as "named entities" even if capitalised.
_ENTITY_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with",
        "by", "from", "as", "at", "is", "are", "was", "were", "be", "been",
        "this", "that", "these", "those", "it", "its", "their", "our", "we",
        "i", "you", "they", "he", "she",
        "method", "model", "models", "results", "paper", "authors", "section",
        "figure", "table", "approach", "system", "framework", "technique",
        "techniques", "experiment", "experiments", "training", "evaluation",
        "performance", "accuracy", "introduction", "conclusion", "abstract",
    }
)

_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|x|ms|s|gb|mb|kb|tb|m|b|k)?\b", re.I)
_QUOTE_FRAGMENT_LEN = 24  # min chars of a quote fragment we'll search for in body


def _norm_token(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def _norm_number(s: str) -> str:
    """Canonical key for a number+unit (handles spacing and `B` vs `M` etc)."""
    m = _NUM_RE.match(s.strip())
    if not m:
        return ""
    raw = m.group(0)
    return re.sub(r"\s+", "", raw).lower()


def _capitalised_phrases(text: str) -> list[str]:
    """Capitalised words / hyphenated proper nouns ignoring sentence-start noise."""
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(
        r"\b([A-Z][A-Za-z0-9]*(?:[-\s][A-Z][A-Za-z0-9]*){0,3})\b", text or ""
    ):
        raw = m.group(1).strip()
        key = _norm_token(raw)
        if not key or len(key) < 3:
            continue
        if key in _ENTITY_STOPWORDS:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


def _claim_first_sentence(claim: str) -> str:
    s = (claim or "").strip()
    if not s:
        return ""
    end = re.search(r"[.!?]", s)
    head = s[: end.start() + 1] if end else s
    return head.strip()[:160]


def _extract_numbers(text: str) -> set[str]:
    """Canonicalised numeric tokens in `text`."""
    out: set[str] = set()
    for m in _NUM_RE.finditer(text or ""):
        k = _norm_number(m.group(0))
        if k:
            out.add(k)
    return out


def _bench_numbers(b) -> list[str]:
    out: list[str] = []
    for fld in ("value", "baseline"):
        v = (getattr(b, fld, "") or "").strip()
        for k in _extract_numbers(v):
            out.append(k)
    return out


def _related_items(bundle: EvidenceBundle) -> list[Any]:
    """`related-work` proxies on EvidenceBundle: ancestors + competitors + followups + enrichment."""
    out: list[Any] = []
    for grp in (
        getattr(bundle, "ancestors", None) or [],
        getattr(bundle, "competitors", None) or [],
        getattr(bundle, "followups", None) or [],
        getattr(bundle, "enrichment_items", None) or [],
    ):
        out.extend(grp)
    return out


def harvested_facts(bundle: EvidenceBundle) -> dict[str, list[Any]]:
    """Return four buckets of facts harvested from the bundle, deduped per-bucket."""
    numbers: list[str] = []
    seen_n: set[str] = set()
    for b in getattr(bundle, "benchmarks", None) or []:
        for k in _bench_numbers(b):
            if k not in seen_n:
                seen_n.add(k)
                numbers.append(k)
    for note in getattr(bundle, "analyst_notes", None) or []:
        for c in (note.claims or []):
            for k in _extract_numbers(c):
                if k not in seen_n:
                    seen_n.add(k)
                    numbers.append(k)

    entities: list[str] = []
    seen_e: set[str] = set()

    def _push_entity(s: str) -> None:
        key = _norm_token(s)
        if not key or len(key) < 3 or key in _ENTITY_STOPWORDS or key in seen_e:
            return
        seen_e.add(key)
        entities.append(s.strip())

    for it in _related_items(bundle):
        title = (getattr(it, "title", "") or "").strip()
        if title:
            _push_entity(title[:120])
    for b in getattr(bundle, "benchmarks", None) or []:
        nm = (getattr(b, "name", "") or "").strip()
        if nm:
            _push_entity(nm[:80])
    for note in getattr(bundle, "analyst_notes", None) or []:
        for c in (note.claims or []):
            for cap in _capitalised_phrases(c):
                _push_entity(cap)

    quotes: list[str] = []
    for q in getattr(bundle, "quotes", None) or []:
        text = (getattr(q, "text", "") or "").strip()
        if len(text) >= _QUOTE_FRAGMENT_LEN:
            quotes.append(text[:240])
    for q in getattr(bundle, "paper_quotes", None) or []:
        text = (getattr(q, "text", "") or "").strip()
        if len(text) >= _QUOTE_FRAGMENT_LEN:
            quotes.append(text[:240])

    analyst_claims: list[str] = []
    seen_c: set[str] = set()
    for note in getattr(bundle, "analyst_notes", None) or []:
        if getattr(note, "skipped", False):
            continue
        for c in (note.claims or []):
            head = _claim_first_sentence(c)
            key = _norm_token(head)
            if not head or not key or key in seen_c:
                continue
            seen_c.add(key)
            analyst_claims.append(head)

    return {
        "numbers": numbers,
        "entities": entities,
        "quotes": quotes,
        "analyst_claims": analyst_claims,
    }


def _body_contains_number(body_numbers: set[str], k: str) -> bool:
    if k in body_numbers:
        return True
    # Allow `81.7` to match `81.67` (writer rounding) - prefix match on the canonical form.
    for bn in body_numbers:
        if not k or not bn:
            continue
        a, b = (k, bn) if len(k) <= len(bn) else (bn, k)
        if len(a) >= 3 and b.startswith(a):
            return True
    return False


def _body_contains_entity(body_lower: str, body_norm_tokens: set[str], e: str) -> bool:
    s = (e or "").strip()
    if not s:
        return False
    if s.lower() in body_lower:
        return True
    # Fall back to the normalised form to absorb punctuation/whitespace differences.
    nk = _norm_token(s)
    return bool(nk) and nk in body_norm_tokens


def _body_contains_quote(body_lower: str, q: str) -> bool:
    head = (q or "").strip()[:80].lower()
    if len(head) < _QUOTE_FRAGMENT_LEN:
        return False
    # Strip non-alpha noise so quotes with stray punctuation still match.
    norm_head = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", head)).strip()
    if not norm_head:
        return False
    body_alpha = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", body_lower)).strip()
    return norm_head[:_QUOTE_FRAGMENT_LEN] in body_alpha


def _body_contains_claim(body_lower: str, c: str) -> bool:
    head = (c or "").strip()[:80].lower()
    if len(head) < 18:
        return False
    body_alpha = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", body_lower)).strip()
    head_alpha = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", head)).strip()
    return bool(head_alpha) and head_alpha[:24] in body_alpha


def coverage_report(body: str, bundle: EvidenceBundle) -> dict[str, Any]:
    """Return per-bucket hit counts, ratios, weighted score, and unused items."""
    facts = harvested_facts(bundle)
    text = body or ""
    body_lower = text.lower()
    body_numbers = _extract_numbers(text)
    body_norm_tokens: set[str] = set()
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", text):
        nk = _norm_token(tok)
        if nk:
            body_norm_tokens.add(nk)

    used: dict[str, list[Any]] = {k: [] for k in facts}
    unused: dict[str, list[Any]] = {k: [] for k in facts}
    for k in facts["numbers"]:
        (used if _body_contains_number(body_numbers, k) else unused)["numbers"].append(k)
    for e in facts["entities"]:
        (used if _body_contains_entity(body_lower, body_norm_tokens, e) else unused)[
            "entities"
        ].append(e)
    for q in facts["quotes"]:
        (used if _body_contains_quote(body_lower, q) else unused)["quotes"].append(q)
    for c in facts["analyst_claims"]:
        (used if _body_contains_claim(body_lower, c) else unused)[
            "analyst_claims"
        ].append(c)

    def _ratio(bucket: str) -> float:
        n = len(facts[bucket]) or 0
        if n == 0:
            return 1.0
        return float(len(used[bucket])) / float(n)

    ratios = {k: _ratio(k) for k in facts}
    score = (
        0.4 * ratios["numbers"]
        + 0.3 * ratios["entities"]
        + 0.2 * ratios["analyst_claims"]
        + 0.1 * ratios["quotes"]
    )
    return {
        "score": round(score, 4),
        "ratios": {k: round(v, 4) for k, v in ratios.items()},
        "totals": {k: len(facts[k]) for k in facts},
        "used_counts": {k: len(used[k]) for k in facts},
        "unused": {k: list(unused[k][:8]) for k in facts},
    }


def unused_facts_for_section(
    section_title: str,
    body: str,
    bundle: EvidenceBundle,
    *,
    max_per_bucket: int = 4,
) -> dict[str, list[Any]]:
    """Top unused items per bucket, suitable to feed the specificity rewriter."""
    rep = coverage_report(body, bundle)
    out: dict[str, list[Any]] = {}
    for k, items in rep["unused"].items():
        out[k] = list(items[:max_per_bucket])
    return out


def render_unused_facts(unused: dict[str, Iterable[Any]]) -> str:
    """Compact, prompt-ready rendering of unused facts for the rewriter user message."""
    parts: list[str] = []
    nums = list(unused.get("numbers") or [])
    if nums:
        parts.append("UNUSED NUMBERS: " + ", ".join(str(x) for x in nums))
    ents = list(unused.get("entities") or [])
    if ents:
        parts.append(
            "UNUSED NAMED ENTITIES (cite or compare against by name):\n"
            + "\n".join(f"- {e}" for e in ents)
        )
    claims = list(unused.get("analyst_claims") or [])
    if claims:
        parts.append(
            "UNUSED ANALYST CLAIMS (use the literal phrasing if it fits):\n"
            + "\n".join(f"- {c}" for c in claims)
        )
    quotes = list(unused.get("quotes") or [])
    if quotes:
        parts.append(
            "UNUSED PAPER QUOTES (anchor an opinion in one of these):\n"
            + "\n".join(f"- \"{q[:160]}\"" for q in quotes)
        )
    return "\n\n".join(parts)

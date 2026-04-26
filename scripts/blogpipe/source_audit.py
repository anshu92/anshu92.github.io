"""Deterministic source registry and citation audit helpers."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from .models import CitationAuditReport, EvidenceBundle, SourceRegistryEntry


def _normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    p = urlparse(raw)
    host = (p.netloc or "").lower().removeprefix("www.")
    path = re.sub(r"/+", "/", p.path or "/").rstrip("/")
    return f"{host}{path}"


def build_source_registry(bundle: EvidenceBundle) -> list[SourceRegistryEntry]:
    bundle.register_ids()
    out: list[SourceRegistryEntry] = []
    seen: set[tuple[str, str, str]] = set()

    def add(entry: SourceRegistryEntry) -> None:
        key = (entry.kind, entry.key, _normalize_url(entry.url))
        if key in seen:
            return
        seen.add(key)
        out.append(entry)

    for source_id, it in bundle.by_id.items():
        add(
            SourceRegistryEntry(
                kind="item",
                source_id=source_id,
                key=source_id,
                url=it.url,
                text=it.title,
                metadata={"title": it.title, "source": it.source},
            )
        )
    for idx, row in enumerate(bundle.benchmarks or []):
        add(
            SourceRegistryEntry(
                kind="benchmark",
                source_id=bundle.primary.id,
                key=f"benchmark:{idx}",
                url=bundle.primary.url,
                text=f"{row.name}: {row.value} {row.unit}".strip(),
                metadata={"baseline": row.baseline, "notes": row.notes},
            )
        )
    for idx, quote in enumerate(bundle.quotes or []):
        add(
            SourceRegistryEntry(
                kind="quote",
                source_id=quote.source_id,
                key=f"quote:{idx}",
                url=quote.url or bundle.by_id.get(quote.source_id, bundle.primary).url,
                text=quote.text[:500],
            )
        )
    for key, value in (bundle.section_evidence or {}).items():
        add(
            SourceRegistryEntry(
                kind="section_evidence",
                source_id=bundle.primary.id,
                key=key,
                url=bundle.primary.url,
                text=(value or "")[:2000],
            )
        )
    return out


_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


def audit_citations(body: str, registry: list[SourceRegistryEntry]) -> CitationAuditReport:
    allowed = {_normalize_url(x.url) for x in registry if x.url}
    verified: list[str] = []
    invalid: list[str] = []
    for _label, url in _MD_LINK.findall(body or ""):
        norm = _normalize_url(url)
        if not norm:
            continue
        if norm in allowed:
            verified.append(url)
        else:
            invalid.append(url)
    return CitationAuditReport(
        ok=not bool(invalid),
        verified_links=list(dict.fromkeys(verified)),
        invalid_links=list(dict.fromkeys(invalid)),
        warnings=(["citation_links_outside_registry"] if invalid else []),
    )


def strip_unregistered_links(body: str, registry: list[SourceRegistryEntry]) -> tuple[str, CitationAuditReport]:
    report = audit_citations(body, registry)
    if report.ok:
        return body, report
    allowed = {_normalize_url(x.url) for x in registry if x.url}
    removed: list[str] = []

    def repl(m: re.Match[str]) -> str:
        label, url = m.group(1), m.group(2)
        if _normalize_url(url) in allowed:
            return m.group(0)
        removed.append(url)
        return label

    revised = _MD_LINK.sub(repl, body or "")
    report.removed_links = list(dict.fromkeys(removed))
    report.ok = False
    return revised, report

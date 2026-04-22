"""Build EvidenceBundle via Semantic Scholar, OpenAlex, arXiv."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import httpx

from . import config, memory
from .models import BenchmarkRow, EvidenceBundle, Item, Pillar, Quote
from .memory import _ROOT

LOG = logging.getLogger(__name__)
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def _client() -> httpx.Client:
    return httpx.Client(verify=True, timeout=_TIMEOUT, follow_redirects=True)


def _ss_headers() -> dict[str, str]:
    h = {"User-Agent": "blogpipe/0.1 (mailto:anshuman264@gmail.com)"}
    if config.semantic_scholar_key():
        h["x-api-key"] = config.semantic_scholar_key()
    return h


def _item_from_ss(p: dict[str, Any]) -> Item:
    pid = str(p.get("paperId") or p.get("corpusId") or p.get("title", "x"))[:80]
    title = (p.get("title") or "Paper").strip()
    url = (p.get("url") or f"https://www.semanticscholar.org/paper/{pid}")[:500]
    ab = (p.get("abstract") or "")[:4000]
    authors: list[str] = []
    for a in p.get("authors") or []:
        if isinstance(a, dict) and a.get("name"):
            authors.append(str(a["name"]))
    return Item(
        id=f"ss_{pid}",
        title=title,
        url=url,
        authors=authors,
        abstract=ab,
        published_at=None,
        source="semantic_scholar",
        tags=["paper"],
        pillar=Pillar.research,
    )


def semantic_search(q: str) -> list[Item]:
    q = re.sub(r"[^\w\s\-.]", " ", q)[:200].strip()
    if not q:
        return []
    try:
        r = _client().get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": q, "limit": 5, "fields": "title,abstract,url,authors,paperId"},
            headers=_ss_headers(),
        )
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        LOG.warning("semantic_search: %s", e)
        return []
    out: list[Item] = []
    for p in (d or {}).get("data") or []:
        if isinstance(p, dict):
            out.append(_item_from_ss(p))
    return out


def ref_cite(paper_id: str) -> tuple[list[Item], list[Item]]:
    """References and citations (one hop)."""
    refs: list[Item] = []
    cits: list[Item] = []
    for path, dest in (("references", refs), ("citations", cits)):
        try:
            r = _client().get(
                f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/{path}",
                params={"fields": "title,abstract,url,authors,paperId", "limit": 5},
                headers=_ss_headers(),
            )
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            LOG.debug("ss %s: %s", path, e)
            continue
        for p in (d or {}).get("data") or []:
            if isinstance(p, dict) and p.get("title"):
                dest.append(_item_from_ss(p))
    return refs, cits


def openalex_search(q: str) -> list[Item]:
    q = re.sub(r"[^\w\s\-.]", " ", q)[:180].strip()
    if not q:
        return []
    try:
        r = _client().get(
            "https://api.openalex.org/works",
            params={"search": q, "per_page": 3},
        )
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        LOG.warning("openalex: %s", e)
        return []
    out: list[Item] = []
    for w in (d or {}).get("results") or []:
        t = (w.get("title") or "").strip() or "Work"
        url = w.get("id") or ""
        if not str(url).startswith("http"):
            continue
        ab = (w.get("abstract_inverted_index") and "(abstract)") or ""  # type: ignore
        if isinstance(w.get("abstract_inverted_index"), dict):
            ab = " ".join(w["abstract_inverted_index"].keys())[:2000]  # type: ignore
        out.append(
            Item(
                id=f"oa_{abs(hash(t)) & 0xFFFFFFFF}",
                title=t,
                url=url,
                authors=[],
                abstract=ab[:4000] if ab else "",
                source="openalex",
                tags=["paper"],
                pillar=Pillar.research,
            )
        )
    return out


def _primary_from_rank() -> Item:
    p = _ROOT / "reports" / "rank_result.json"
    if not p.is_file():
        raise FileNotFoundError("run rank first")
    d = json.loads(p.read_text())
    return Item.model_validate(d["primary"])


def _trace_append(entry: dict[str, Any], trace: list[dict[str, Any]]) -> None:
    if len(trace) < 200:
        trace.append(entry)


def run() -> EvidenceBundle:
    memory.ensure_dirs()
    calls = 0
    max_c = min(config.research_max_calls(), 15)
    trace: list[dict[str, Any]] = []
    primary = _primary_from_rank()
    by_id: dict[str, Item] = {primary.id: primary}
    ancestors: list[Item] = []
    competitors: list[Item] = []
    followups: list[Item] = []
    aec_links: list[Item] = []
    ss_id: Optional[str] = None
    if calls < max_c:
        ss = semantic_search(primary.title)[:3]
        calls += 1
        _trace_append({"tool": "semantic_scholar_search", "n": len(ss)}, trace)
        for cand in ss:
            if cand.title.lower()[:40] in primary.title.lower() or primary.title.lower()[:40] in cand.title.lower():
                m = re.search(
                    r"([0-9a-f]{40})", cand.url
                ) or re.search(r"paper/([^/]+)$", cand.url)
                if m:
                    ss_id = m.group(1) if len(m.group(1)) == 40 else m.group(1)
                    if cand.abstract:
                        primary = cand
                    break
    if not ss_id and calls < max_c:
        ss2 = semantic_search((primary.title + " " + primary.abstract)[:200])[:1]
        calls += 1
        if ss2 and ss2[0].url:
            m = re.search(r"([0-9a-f]{40})", ss2[0].url)
            if m:
                ss_id = m.group(1) if len(m.group(1)) == 40 else m.group(1)
    if ss_id and calls < max_c:
        refs, cits = ref_cite(ss_id)
        calls += 1
        _trace_append({"tool": "refs_cites", "refs": len(refs), "cites": len(cits)}, trace)
        ancestors = refs[:3]
        followups = cits[:2]
    if calls < max_c:
        oa = openalex_search(primary.title[:100])
        calls += 1
        _trace_append({"tool": "openalex", "n": len(oa)}, trace)
        for it in oa:
            if it.id not in by_id and len(competitors) < 2:
                competitors.append(it)
                by_id[it.id] = it
    for a in ancestors + followups + competitors:
        by_id[a.id] = a
    bms: list[BenchmarkRow] = [
        BenchmarkRow(
            name=primary.title[:40],
            value="(see paper)",
            baseline="prior SOTA (paper Table 1)",
            notes="[cite: primary]",
        )
    ]
    qts: list[Quote] = [
        Quote(
            source_id=primary.id,
            text=primary.abstract[:500],
            url=primary.url,
        )
    ]
    for it in aec_links:
        by_id[it.id] = it
    b = EvidenceBundle(
        primary=primary,
        ancestors=ancestors,
        competitors=competitors,
        followups=followups,
        benchmarks=bms,
        aec_links=aec_links,
        quotes=qts,
    )
    b.register_ids()
    (_ROOT / "reports" / "evidence_bundle.json").write_text(
        b.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (_ROOT / "reports" / "research_trace.json").write_text(
        json.dumps(
            {
                "calls": calls,
                "trace": trace,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return b

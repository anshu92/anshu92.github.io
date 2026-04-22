"""Build EvidenceBundle via Semantic Scholar, OpenAlex, arXiv."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import httpx

from . import config, mcp_enrichment, memory, openrouter_client
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


def _competitor_queries(title: str, abstract: str) -> list[str]:
    """LLM: short OpenAlex search phrases for related methods. Caller must gate dry_run / no key."""
    raw = openrouter_client.llm_text(
        'Output JSON only: {"queries": ["<q1>", "<q2>"]}. '
        "Each query: 4-8 words naming a competing, adjacent, or ancestor method/line of work. "
        "No generic words only.",
        f"Title: {title}\nAbstract: {abstract[:600]}",
    )
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            qlist = json.loads(m.group(0)).get("queries") or []
            out = [str(x).strip() for x in qlist if str(x).strip()][:2]
            if out:
                return out
        except (json.JSONDecodeError, TypeError) as e:
            LOG.debug("competitor_queries json: %s", e)
    return [title[:80].strip() or "machine learning"]


def _extract_benchmarks_from_abstract(title: str, abstract: str) -> list[BenchmarkRow]:
    """LLM: pull explicit numbers from abstract into BenchmarkRow (no invention)."""
    if not abstract.strip():
        return []
    raw = openrouter_client.llm_text(
        'Extract benchmark or result numbers from the abstract only as JSON: '
        '{"rows": [{"method": "...", "metric": "...", "value": "...", "baseline": "..."}]}. '
        "Use only values stated in the text. If none, return {\"rows\": []}.",
        f"Title: {title}\nAbstract: {abstract[:2000]}",
    )
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return []
    try:
        rows = (json.loads(m.group(0)) or {}).get("rows") or []
    except (json.JSONDecodeError, TypeError) as e:
        LOG.debug("extract_benchmarks json: %s", e)
        return []
    out: list[BenchmarkRow] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        method = (r.get("method") or r.get("name") or "").strip()
        value = (r.get("value") or "").strip()
        if not method or not value:
            continue
        metric = (r.get("metric") or "").strip()
        baseline = (r.get("baseline") or "").strip()
        out.append(
            BenchmarkRow(
                name=method[:200],
                value=value[:200],
                unit=metric[:200],
                baseline=baseline[:200],
                notes="from abstract (LLM extract)",
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
    competitor_queries: list[str] = []
    if calls < max_c:
        if not config.dry_run() and config.llm_configured():
            competitor_queries = _competitor_queries(primary.title, primary.abstract)
            calls += 1
        else:
            competitor_queries = [primary.title[:80].strip() or "machine learning paper"]
        _trace_append({"tool": "competitor_queries", "n": len(competitor_queries)}, trace)
    for q in competitor_queries or [primary.title[:80]]:
        if len(competitors) >= 3 or calls >= max_c:
            break
        oa = openalex_search(q)
        calls += 1
        _trace_append({"tool": "openalex", "q": q[:100], "n": len(oa)}, trace)
        for it in oa:
            if it.id not in by_id and len(competitors) < 3:
                competitors.append(it)
                by_id[it.id] = it
    for a in ancestors + followups + competitors:
        by_id[a.id] = a
    bms: list[BenchmarkRow] = []
    if (
        calls < max_c
        and not config.dry_run()
        and config.llm_configured()
        and primary.abstract.strip()
    ):
        bms = _extract_benchmarks_from_abstract(primary.title, primary.abstract)
        calls += 1
        _trace_append({"tool": "extract_benchmarks", "n": len(bms)}, trace)
    if not bms:
        bms = [
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
    mres = mcp_enrichment.run(primary, calls, max_c, trace)
    calls = mres.calls_used
    b = EvidenceBundle(
        primary=primary,
        ancestors=ancestors,
        competitors=competitors,
        followups=followups,
        benchmarks=bms,
        aec_links=aec_links,
        quotes=qts,
        enrichment_items=mres.enrichment_items,
        planner_buckets=mres.planner_buckets,
        planner_questions=mres.planner_questions,
        section_evidence=mres.section_evidence,
        contradiction_notes=mres.contradiction_notes,
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

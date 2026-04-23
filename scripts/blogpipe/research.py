"""Build EvidenceBundle via Semantic Scholar, OpenAlex, arXiv.

For the LangGraph end-to-end pipeline (rank → research → draft → editor), use
``python -m blogpipe graph`` or ``python -m blogpipe run`` instead of this stage alone.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any, Optional

import httpx

from . import config, mcp_enrichment, memory, openrouter_client, paper_reader
from .llm_chain import get_llm_usage
from .models import (
    AnalystNote,
    BenchmarkRow,
    EvidenceBundle,
    EvidencePack,
    Item,
    Pillar,
    Quote,
)
from .memory import _ROOT

LOG = logging.getLogger(__name__)
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# Per-run stats (SS cache, rate limits) — read in run() for traces / stdout.
_RESEARCH_SINK: dict[str, int] = {
    "ss_cache_hits": 0,
    "ss_429": 0,
    "ss_search_calls": 0,
    "ss_ref_calls": 0,
}
_SS_TTL = 86400.0


def _client() -> httpx.Client:
    return httpx.Client(verify=True, timeout=_TIMEOUT, follow_redirects=True)


def _ss_headers() -> dict[str, str]:
    h = {"User-Agent": "blogpipe/0.1 (mailto:anshuman264@gmail.com)"}
    if config.semantic_scholar_key():
        h["x-api-key"] = config.semantic_scholar_key()
    return h


def _ss_qhash(q: str) -> str:
    return hashlib.sha1(q.encode("utf-8")).hexdigest()


def _ss_cache_get_search(q: str) -> Optional[list[Item]]:
    p = _ROOT / "cache" / "ss_cache" / f"{_ss_qhash(q)}.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if time.time() > float(d.get("exp", 0)):
        return None
    if d.get("kind") != "search":
        return None
    raw = d.get("items")
    if not isinstance(raw, list):
        return None
    out: list[Item] = []
    for x in raw:
        if isinstance(x, dict):
            try:
                out.append(Item.model_validate(x))
            except Exception:  # noqa: BLE001
                pass
    return out


def _ss_cache_put_search(q: str, items: list[Item]) -> None:
    p = _ROOT / "cache" / "ss_cache" / f"{_ss_qhash(q)}.json"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        LOG.debug("ss cache mkdir: %s", e)
        return
    payload = {
        "kind": "search",
        "exp": time.time() + _SS_TTL,
        "items": [x.model_dump(mode="json") for x in items],
    }
    try:
        p.write_text(json.dumps(payload, indent=0), encoding="utf-8")
    except OSError as e:
        LOG.debug("ss cache write: %s", e)


def _ss_cache_get_refs(paper_id: str) -> Optional[tuple[list[Item], list[Item]]]:
    p = _ROOT / "cache" / "ss_cache" / f"ref_{_ss_qhash(paper_id)}.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if time.time() > float(d.get("exp", 0)) or d.get("kind") != "refs":
        return None
    ra, ca = d.get("refs") or [], d.get("cits") or []
    r_out: list[Item] = []
    c_out: list[Item] = []
    for x in ra if isinstance(ra, list) else []:
        if isinstance(x, dict):
            try:
                r_out.append(Item.model_validate(x))
            except Exception:  # noqa: BLE001
                pass
    for x in ca if isinstance(ca, list) else []:
        if isinstance(x, dict):
            try:
                c_out.append(Item.model_validate(x))
            except Exception:  # noqa: BLE001
                pass
    return r_out, c_out


def _ss_cache_put_refs(paper_id: str, refs: list[Item], cits: list[Item]) -> None:
    p = _ROOT / "cache" / "ss_cache" / f"ref_{_ss_qhash(paper_id)}.json"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        LOG.debug("ss ref cache mkdir: %s", e)
        return
    payload = {
        "kind": "refs",
        "exp": time.time() + _SS_TTL,
        "refs": [x.model_dump(mode="json") for x in refs],
        "cits": [x.model_dump(mode="json") for x in cits],
    }
    try:
        p.write_text(json.dumps(payload, indent=0), encoding="utf-8")
    except OSError as e:
        LOG.debug("ss ref cache write: %s", e)


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
    hit = _ss_cache_get_search(q)
    if hit is not None:
        _RESEARCH_SINK["ss_cache_hits"] = int(_RESEARCH_SINK.get("ss_cache_hits", 0)) + 1
        return hit
    _RESEARCH_SINK["ss_search_calls"] = int(_RESEARCH_SINK.get("ss_search_calls", 0)) + 1
    d: dict[str, Any] = {}
    for attempt in range(3):
        try:
            r = _client().get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": q,
                    "limit": 5,
                    "fields": "title,abstract,url,authors,paperId",
                },
                headers=_ss_headers(),
            )
        except httpx.HTTPError as e:
            LOG.warning("semantic_search: %s", e)
            return []
        if r.status_code == 429:
            _RESEARCH_SINK["ss_429"] = int(_RESEARCH_SINK.get("ss_429", 0)) + 1
            ra = (r.headers.get("Retry-After") or "").strip()
            try:
                wait = int(ra) if ra else 2**attempt
            except ValueError:
                wait = 2**attempt
            if attempt < 2:
                time.sleep(float(max(1, min(wait, 60))))
                continue
            LOG.warning("semantic_search: 429 after retries")
            return []
        try:
            r.raise_for_status()
            d = r.json() or {}
        except Exception as e:
            LOG.warning("semantic_search: %s", e)
            return []
        break
    out: list[Item] = []
    for p in d.get("data") or []:
        if isinstance(p, dict):
            out.append(_item_from_ss(p))
    if out:
        _ss_cache_put_search(q, out)
    return out


def ref_cite(paper_id: str) -> tuple[list[Item], list[Item]]:
    """References and citations (one hop)."""
    if not (paper_id or "").strip():
        return [], []
    cached = _ss_cache_get_refs(paper_id)
    if cached is not None:
        _RESEARCH_SINK["ss_cache_hits"] = int(_RESEARCH_SINK.get("ss_cache_hits", 0)) + 1
        return cached
    _RESEARCH_SINK["ss_ref_calls"] = int(_RESEARCH_SINK.get("ss_ref_calls", 0)) + 1
    refs: list[Item] = []
    cits: list[Item] = []
    for path, dest in (("references", refs), ("citations", cits)):
        d: dict[str, Any] = {}
        for attempt in range(3):
            try:
                r = _client().get(
                    f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/{path}",
                    params={"fields": "title,abstract,url,authors,paperId", "limit": 5},
                    headers=_ss_headers(),
                )
            except httpx.HTTPError as e:
                LOG.debug("ss %s: %s", path, e)
                break
            if r.status_code == 429:
                _RESEARCH_SINK["ss_429"] = int(_RESEARCH_SINK.get("ss_429", 0)) + 1
                ra = (r.headers.get("Retry-After") or "").strip()
                try:
                    wait = int(ra) if ra else 2**attempt
                except ValueError:
                    wait = 2**attempt
                if attempt < 2:
                    time.sleep(float(max(1, min(wait, 60))))
                    continue
                break
            try:
                r.raise_for_status()
                d = r.json() or {}
            except Exception as e:
                LOG.debug("ss %s: %s", path, e)
                break
            break
        for p in d.get("data") or []:
            if isinstance(p, dict) and p.get("title"):
                dest.append(_item_from_ss(p))
    if refs or cits:
        _ss_cache_put_refs(paper_id, refs, cits)
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
        'JSON only: {"queries": ["4-8 word OpenAlex query", "second query"]}. '
        "Name competing, adjacent, or ancestor methods—not generic filler.",
        f"Title: {title}\nAbstract: {abstract[:600]}",
        max_tokens=1536,
        task="query_gen",
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
        max_tokens=1536,
        task="paper_chunk_summary",
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


def _clean_benchmark_name(text: str) -> str:
    t = re.sub(r"^[,;:\s]+|[,;:\s]+$", "", text or "")
    if "," in t:
        t = t.split(",")[-1].strip()
    t = re.sub(r"^(using|with|both|the|and|as well as)\s+", "", t, flags=re.I)
    t = re.sub(r"\bwe achieve superior performance on\b", "", t, flags=re.I)
    t = re.sub(r"\bwe achieve\b", "", t, flags=re.I)
    t = re.sub(r"^significantly outperforming both\s+", "", t, flags=re.I)
    t = re.sub(r"^significantly outperforming\s+", "", t, flags=re.I)
    if "using only" in t.lower():
        t = t.split("using only", 1)[-1].strip()
    t = re.sub(r"^the baseline\s+", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" ,.;:")
    return t[:80]


def _fallback_benchmarks_from_abstract(title: str, abstract: str) -> list[BenchmarkRow]:
    """Regex fallback: prefer factual rows over `(see paper)` placeholders."""
    txt = (abstract or "").strip()
    if not txt:
        return []
    rows: list[BenchmarkRow] = []

    pct_hits = re.findall(r"([^()]{3,120}?)\s*\((\d+(?:\.\d+)?)%\)", txt)
    if pct_hits:
        cleaned = [(_clean_benchmark_name(name), f"{value}%") for name, value in pct_hits]
        baseline_label = ""
        baseline_value = ""
        for name, value in cleaned:
            if "baseline" in name.lower():
                baseline_label = name
                baseline_value = value
                break
        if not baseline_label and cleaned:
            baseline_label, baseline_value = cleaned[-1]
        for name, value in cleaned[:4]:
            if not name:
                continue
            baseline = ""
            if baseline_label and name != baseline_label:
                baseline = f"{baseline_label} {baseline_value}".strip()
            rows.append(
                BenchmarkRow(
                    name=name,
                    value=value,
                    unit="accuracy",
                    baseline=baseline[:200],
                    notes="from abstract (regex fallback)",
                )
            )
        if rows:
            return rows[:4]

    size_match = re.search(
        r"(\d+(?:\.\d+)?M\s*-\s*\d+(?:\.\d+)?M)\s+parameters", txt, re.I
    )
    edge_match = re.search(
        r"(\d+(?:\.\d+)?M\s*-\s*\d+(?:\.\d+)?B)\s+parameter", txt, re.I
    )
    span_match = re.search(r"first\s+(\d+\s*-\s*\d+)\s+words", txt, re.I)
    comp_match = re.search(
        r"matching several\s+(\d+(?:\.\d+)?M\s*-\s*\d+(?:\.\d+)?M-class)",
        txt,
        re.I,
    )
    if size_match:
        rows.append(
            BenchmarkRow(
                name="micro language models",
                value=size_match.group(1).replace(" ", ""),
                unit="parameters",
                baseline=edge_match.group(1).replace(" ", "") if edge_match else "",
                notes="from abstract (regex fallback)",
            )
        )
    if span_match:
        rows.append(
            BenchmarkRow(
                name="on-device opener",
                value=span_match.group(1).replace(" ", "") + " words",
                unit="generated prefix",
                baseline="cloud continuation",
                notes="from abstract (regex fallback)",
            )
        )
    if comp_match:
        rows.append(
            BenchmarkRow(
                name="comparable existing models",
                value=comp_match.group(1).replace(" ", ""),
                unit="matched model class",
                baseline="existing 70M-256M-class models",
                notes="from abstract (regex fallback)",
            )
        )
    return rows[:4]


def _fallback_benchmarks_from_result_notes(result_notes: list[str]) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    seen: set[str] = set()
    for sent in result_notes:
        text = re.sub(r"\s+", " ", sent or "").strip()
        if not text:
            continue
        pieces = re.split(r"\bwhile\b|;", text, flags=re.I)
        for piece in pieces:
            for pat in (
                r"([A-Za-z0-9][A-Za-z0-9+/\-() ]{2,90}?)\s+(?:achieves?|reaches?|yields?|scores?)\s+(\d+(?:\.\d+)?%)",
                r"([A-Za-z0-9][A-Za-z0-9+/\-() ]{2,90}?)\s+at\s+(\d+(?:\.\d+)?%)",
            ):
                for m in re.finditer(pat, piece, re.I):
                    name = _clean_benchmark_name(m.group(1))
                    value = m.group(2)
                    key = f"{name.lower()}::{value}"
                    if not name or key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        BenchmarkRow(
                            name=name,
                            value=value,
                            unit="",
                            baseline="same paper setup",
                            notes="from full paper (regex fallback)",
                        )
                    )
                    if len(rows) >= 4:
                        return rows
    return rows[:4]


def _primary_from_rank() -> Item:
    p = _ROOT / "reports" / "rank_result.json"
    if not p.is_file():
        raise FileNotFoundError("run rank first")
    d = json.loads(p.read_text())
    return Item.model_validate(d["primary"])


def gather_evidence_pack(primary: Item) -> tuple[EvidencePack, int, list[dict[str, Any]]]:
    """Retrieval: Semantic Scholar, OpenAlex (no LLM queries), ref/cite, paper PDF."""
    global _RESEARCH_SINK
    _RESEARCH_SINK = {
        "ss_cache_hits": 0,
        "ss_429": 0,
        "ss_search_calls": 0,
        "ss_ref_calls": 0,
    }
    memory.ensure_dirs()
    trace: list[dict[str, Any]] = []
    calls = 0
    by_id: dict[str, Item] = {primary.id: primary}
    ancestors: list[Item] = []
    followups: list[Item] = []
    competitors: list[Item] = []
    ss_id: Optional[str] = None
    refs: list[Item] = []
    cits: list[Item] = []
    ss: list[Item] = []
    if calls < 99:
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
                        extra = dict(primary.extra)
                        if cand.url and cand.url != primary.url:
                            extra["semantic_scholar_url"] = cand.url
                        primary = primary.model_copy(
                            update={
                                "title": cand.title or primary.title,
                                "abstract": cand.abstract,
                                "authors": cand.authors or primary.authors,
                                "tags": list(dict.fromkeys(primary.tags + cand.tags)),
                                "extra": extra,
                            }
                        )
                    break
    if not ss and calls < 99:
        ss2 = semantic_search((primary.title + " " + primary.abstract)[:200])[:1]
        calls += 1
        if ss2 and ss2[0].url:
            m = re.search(r"([0-9a-f]{40})", ss2[0].url)
            if m:
                ss_id = m.group(1) if len(m.group(1)) == 40 else m.group(1)
    if ss_id and calls < 99:
        refs, cits = ref_cite(ss_id)
        calls += 1
        ancestors = refs[:3]
        followups = cits[:2]
        _trace_append({"tool": "refs_cites", "refs": len(refs), "cites": len(cits)}, trace)
    competitor_queries = [primary.title[:80].strip() or "machine learning paper"]
    for q in competitor_queries:
        if len(competitors) >= 3 or calls >= 99:
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
    aec_links: list[Item] = []
    paper_section_evidence: dict[str, str] = {}
    paper_limitations: list[str] = []
    paper_result_notes: list[str] = []
    paper_quotes: list[Quote] = []
    outline: list[str] = []
    paper_ev = paper_reader.fetch_primary_paper_evidence(primary)
    if paper_ev.cleaned_text:
        primary.extra["paper_pdf_url"] = paper_ev.pdf_url
        primary.extra["paper_digest"] = "\n\n".join(
            [
                paper_ev.section_evidence.get("paper_problem", ""),
                paper_ev.section_evidence.get("paper_method", ""),
                paper_ev.section_evidence.get("paper_experiments", ""),
                paper_ev.section_evidence.get("paper_limitations", ""),
            ]
        )[:6000]
        paper_section_evidence = dict(paper_ev.section_evidence)
        paper_limitations = list(paper_ev.limitation_notes)
        paper_result_notes = list(paper_ev.result_notes)
        paper_quotes = list(paper_ev.quotes)
        outline = list(paper_ev.outline)
        _trace_append(
            {
                "tool": "paper_reader",
                "outline_sections": len(paper_ev.outline),
                "results": len(paper_ev.result_notes),
                "limitations": len(paper_ev.limitation_notes),
                "llm_chunks": int(paper_ev.llm_chunks_used),
            },
            trace,
        )
    for it in aec_links:
        by_id[it.id] = it
    pack = EvidencePack(
        primary=primary,
        ss_id=ss_id,
        ancestors=ancestors,
        competitors=competitors,
        followups=followups,
        aec_links=aec_links,
        section_evidence=paper_section_evidence,
        paper_limitations=paper_limitations,
        paper_result_notes=paper_result_notes,
        paper_quotes=paper_quotes,
        outline=outline,
        by_id=by_id,
        trace=trace,
        calls_used=calls,
        refs=refs,
        cits=cits,
    )
    return pack, calls, trace


def _trace_append(entry: dict[str, Any], trace: list[dict[str, Any]]) -> None:
    if len(trace) < 200:
        trace.append(entry)


def finalize_evidence_from_pack(
    pack: EvidencePack,
    calls: int,
    trace: list[dict[str, Any]],
    *,
    analyst_notes: list[AnalystNote] | None = None,
    committee_synthesis: str = "",
) -> tuple[EvidenceBundle, int, list[dict[str, Any]]]:
    """LLM OpenAlex, benchmarks, MCP; merge with committee outputs."""
    max_c = min(config.research_max_calls(), 15)
    primary = pack.primary
    by_id: dict[str, Item] = dict(pack.by_id)
    ancestors = list(pack.ancestors)
    competitors = list(pack.competitors)
    followups = list(pack.followups)
    aec_links = list(pack.aec_links)
    paper_section_evidence = dict(pack.section_evidence)
    paper_limitations = list(pack.paper_limitations)
    paper_result_notes = list(pack.paper_result_notes)
    paper_quotes = list(pack.paper_quotes)
    competitive_queries: list[str] = []
    if calls < max_c:
        if not config.dry_run() and config.llm_configured():
            competitive_queries = _competitor_queries(primary.title, primary.abstract)
            calls += 1
        else:
            competitive_queries = [primary.title[:80].strip() or "machine learning paper"]
        _trace_append(
            {"tool": "competitor_queries", "n": len(competitive_queries)}, trace
        )
    for q in competitive_queries or [primary.title[:80]]:
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
        bms = _fallback_benchmarks_from_abstract(primary.title, primary.abstract)
        if bms:
            _trace_append({"tool": "extract_benchmarks_fallback", "n": len(bms)}, trace)
    if not bms and paper_result_notes:
        bms = _fallback_benchmarks_from_result_notes(paper_result_notes)
        if bms:
            _trace_append(
                {"tool": "extract_benchmarks_paper_fallback", "n": len(bms)},
                trace,
            )
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
    ] + paper_quotes[:6]
    for it in aec_links:
        by_id[it.id] = it
    mres = mcp_enrichment.run(primary, calls, max_c, trace)
    calls = mres.calls_used
    merged_section_evidence = dict(paper_section_evidence)
    for key, value in (mres.section_evidence or {}).items():
        if not value:
            continue
        if key not in merged_section_evidence:
            merged_section_evidence[key] = value
        else:
            merged_section_evidence[key] = (
                merged_section_evidence[key].rstrip() + "\n\n" + value.lstrip()
            )[:4000]
    nlist = list(analyst_notes) if analyst_notes else []
    from .analysts.visual_planner import parse_visual_plan_from_analysts

    visual_plan = parse_visual_plan_from_analysts(nlist)
    extra_contra = list(
        dict.fromkeys(
            [c for n in nlist for c in (n.contradictions or [])] + paper_limitations
        )
    )
    contradiction_notes = list(
        dict.fromkeys(extra_contra + (mres.contradiction_notes or []))
    )[:10]
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
        section_evidence=merged_section_evidence,
        contradiction_notes=contradiction_notes,
        analyst_notes=nlist,
        committee_synthesis=committee_synthesis,
        visual_plan=visual_plan,
    )
    b.register_ids()
    return b, calls, trace


def run() -> EvidenceBundle:
    global _RESEARCH_SINK
    _RESEARCH_SINK = {
        "ss_cache_hits": 0,
        "ss_429": 0,
        "ss_search_calls": 0,
        "ss_ref_calls": 0,
    }
    memory.ensure_dirs()
    u_research0 = get_llm_usage()
    primary0 = _primary_from_rank()
    pack, calls, trace = gather_evidence_pack(primary0)
    b, calls, trace = finalize_evidence_from_pack(
        pack, calls, trace, analyst_notes=None, committee_synthesis=""
    )
    (_ROOT / "reports" / "evidence_bundle.json").write_text(
        b.model_dump_json(indent=2),
        encoding="utf-8",
    )
    u = get_llm_usage()
    research_llm_ok = u["ok"] - int(u_research0.get("ok", 0) or 0)
    research_llm_fail = u["fail"] - int(u_research0.get("fail", 0) or 0)
    trace_out = {
        "calls": calls,
        "trace": trace,
        "ss_cache_hits": int(_RESEARCH_SINK.get("ss_cache_hits", 0)),
        "ss_429": int(_RESEARCH_SINK.get("ss_429", 0)),
        "ss_search_calls": int(_RESEARCH_SINK.get("ss_search_calls", 0)),
        "ss_ref_calls": int(_RESEARCH_SINK.get("ss_ref_calls", 0)),
        "llm_ok": research_llm_ok,
        "llm_fail": research_llm_fail,
    }
    (_ROOT / "reports" / "research_trace.json").write_text(
        json.dumps(trace_out, indent=2),
        encoding="utf-8",
    )
    max_c = min(config.research_max_calls(), 15)
    n_bm = len(b.benchmarks) if b.benchmarks else 0
    n_comp = len(b.competitors) if b.competitors else 0
    print(
        f"stage=research calls={calls}/{max_c} competitors={n_comp} "
        f"benchmarks={n_bm} ss_cache_hits={trace_out['ss_cache_hits']} "
        f"ss_429={trace_out['ss_429']}",
        flush=True,
    )
    return b

"""MCP-aligned research enrichment: single adapter boundary for blogpipe.

Implements Phase 1 capabilities via direct HTTPS APIs (same trust model as other
blogpipe HTTP clients). MCP server names below map 1:1 to common Cursor MCP configs.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from . import config, openrouter_client
from .models import Item, Pillar

LOG = logging.getLogger(__name__)
_TIMEOUT = httpx.Timeout(25.0, connect=10.0)

# Curated shortlist (tiers) — keep in sync with project docs; do not duplicate in plan files.
MCP_SHORTLIST: dict[str, list[dict[str, str]]] = {
    "phase_1": [
        {"id": "github_mcp", "role": "Repos, issues, PRs, releases for verifiable technical claims"},
        {"id": "context7_mcp", "role": "Version-specific library docs (Context7 HTTP API)"},
        {
            "id": "web_search",
            "role": "Web discovery: set TAVILY_API_KEY (free dev tier) or BRAVE_API_KEY (paid)",
        },
        {"id": "fetch_mcp", "role": "Simple HTTP fetch — use httpx + allowlists in pipeline instead"},
    ],
    "phase_2": [
        {"id": "semantic_scholar_mcp", "role": "Optional; blogpipe already uses Semantic Scholar REST"},
        {"id": "arxiv_mcp", "role": "Preprint-heavy niches — optional alongside arXiv harvest"},
        {"id": "markitdown_mcp", "role": "PDF/slides → markdown when needed"},
    ],
    "phase_3": [
        {"id": "playwright_mcp", "role": "UI walkthroughs / screenshots — not default for paper posts"},
        {"id": "zotero_mcp", "role": "Managed bibliography — only if Zotero-first workflow"},
    ],
}


def _client() -> httpx.Client:
    return httpx.Client(verify=True, timeout=_TIMEOUT, follow_redirects=True)


def _trace_append(trace: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    if len(trace) < 250:
        trace.append(entry)


def _planner_llm(title: str, abstract: str) -> tuple[list[str], list[str]]:
    """Returns (buckets, questions). One LLM call."""
    raw = openrouter_client.llm_text(
        "Output JSON only: "
        '{"buckets": ["definition", "mechanism", "tradeoffs", "limitations", "competing_methods"], '
        '"questions": ["<=8 short research questions for a technical blog"]}. '
        "Questions must be answerable with web or paper evidence.",
        f"Title: {title}\nAbstract: {abstract[:1500]}",
    )
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return [], []
    try:
        d = json.loads(m.group(0))
        buckets = [str(x).strip() for x in (d.get("buckets") or []) if str(x).strip()][:8]
        questions = [str(x).strip() for x in (d.get("questions") or []) if str(x).strip()][:8]
        return buckets, questions
    except (json.JSONDecodeError, TypeError) as e:
        LOG.debug("mcp planner json: %s", e)
        return [], []


def _adversarial_llm(title: str, abstract: str) -> list[str]:
    """Limitations / counterpoints to research in the draft planning layer."""
    raw = openrouter_client.llm_text(
        "Output JSON only: {\"notes\": [\"<=5 short counterpoints, limitations, or falsification angles "
        'for a technical blog — not paper ablation results]}.',
        f"Title: {title}\nAbstract: {abstract[:1500]}",
    )
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return []
    try:
        notes = (json.loads(m.group(0)) or {}).get("notes") or []
        return [str(x).strip() for x in notes if str(x).strip()][:5]
    except (json.JSONDecodeError, TypeError) as e:
        LOG.debug("mcp adversarial json: %s", e)
        return []


def _tavily_search(q: str) -> list[Item]:
    key = config.tavily_api_key()
    if not q.strip() or not key:
        return []
    try:
        r = _client().post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "query": q[:200],
                "max_results": 5,
                "search_depth": "basic",
            },
        )
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        LOG.warning("mcp tavily: %s", e)
        return []
    results = (d or {}).get("results") or []
    out: list[Item] = []
    for i, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        title = str(row.get("title") or "Result").strip()[:300]
        desc = str(row.get("content") or row.get("snippet") or "")[:800]
        if not url.startswith("https://"):
            continue
        out.append(
            Item(
                id=f"tavily_{i}_{abs(hash(url)) & 0xFFFFFFFF:x}",
                title=title,
                url=url[:500],
                abstract=desc,
                source="tavily",
                tags=["mcp", "web"],
                pillar=Pillar.research,
                extra={"tool": "tavily_search"},
            )
        )
    return out


def _brave_search(q: str) -> list[Item]:
    key = config.brave_api_key()
    if not q.strip() or not key:
        return []
    try:
        r = _client().get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": q[:200], "count": 5},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": key,
            },
        )
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        LOG.warning("mcp brave: %s", e)
        return []
    results = ((d or {}).get("web") or {}).get("results") or []
    out: list[Item] = []
    for i, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        title = str(row.get("title") or "Result").strip()[:300]
        desc = str(row.get("description") or row.get("extra_snippets") or "")[:800]
        if not url.startswith("https://"):
            continue
        out.append(
            Item(
                id=f"brave_{i}_{abs(hash(url)) & 0xFFFFFFFF:x}",
                title=title,
                url=url[:500],
                abstract=desc,
                source="brave_search",
                tags=["mcp", "web"],
                pillar=Pillar.research,
                extra={"tool": "brave_search_mcp"},
            )
        )
    return out


def _web_search(q: str) -> tuple[list[Item], str]:
    """Prefer Tavily (typical free tier); fall back to Brave if only that key is set."""
    q = q.strip()
    if not q:
        return [], "none"
    if config.tavily_api_key():
        t = _tavily_search(q)
        if t:
            return t, "tavily"
    if config.brave_api_key():
        b = _brave_search(q)
        if b:
            return b, "brave"
    return [], "none"


def _github_search_repos(q: str) -> list[Item]:
    token = config.github_token()
    if not q.strip():
        return []
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "blogpipe-mcp-enrichment",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = _client().get(
            "https://api.github.com/search/repositories",
            params={"q": q[:180], "per_page": 3},
            headers=headers,
        )
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        LOG.warning("mcp github search: %s", e)
        return []
    items_raw = (d or {}).get("items") or []
    out: list[Item] = []
    for i, row in enumerate(items_raw):
        if not isinstance(row, dict):
            continue
        url = str(row.get("html_url") or "").strip()
        name = str(row.get("full_name") or row.get("name") or "repo").strip()[:200]
        desc = str(row.get("description") or "")[:800]
        if not url.startswith("https://"):
            continue
        out.append(
            Item(
                id=f"gh_{i}_{abs(hash(url)) & 0xFFFFFFFF:x}",
                title=name,
                url=url[:500],
                abstract=desc,
                source="github",
                tags=["mcp", "repo"],
                pillar=Pillar.systems,
                extra={"tool": "github_mcp"},
            )
        )
    return out


def _guess_library_name(title: str, abstract: str) -> Optional[str]:
    """Heuristic library/framework guess for Context7 (no extra LLM call)."""
    blob = f"{title}\n{abstract}"
    for pat in (
        r"\b(PyTorch|TensorFlow|JAX|Keras|transformers|Hugging Face|LoRA|PEFT|vLLM|Ray|LangChain)\b",
        r"\b(PyTorch|Tensorflow)\b",
    ):
        m = re.search(pat, blob, re.I)
        if m:
            return m.group(1)
    return None


def _context7_snippet(library_guess: str, title: str, abstract: str) -> tuple[str, list[Item]]:
    """Returns (markdown section, optional doc items). Uses Context7 REST API."""
    key = config.context7_api_key()
    if not key or not library_guess:
        return "", []
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    try:
        r = _client().get(
            "https://context7.com/api/v2/libs/search",
            headers=headers,
            params={
                "libraryName": library_guess[:80],
                "query": f"{title[:200]} {abstract[:300]}",
            },
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        LOG.warning("mcp context7 search: %s", e)
        return "", []
    res = (data or {}).get("results") or []
    if not res or not isinstance(res[0], dict):
        return "", []
    best = res[0]
    lib_id = str(best.get("id") or "").strip()
    if not lib_id:
        return "", []
    try:
        r2 = _client().get(
            "https://context7.com/api/v2/context",
            headers=headers,
            params={
                "libraryId": lib_id,
                "query": f"Key APIs and usage relevant to: {title[:200]}",
                "type": "json",
            },
        )
        r2.raise_for_status()
        ctx = r2.json()
    except Exception as e:
        LOG.warning("mcp context7 context: %s", e)
        return "", []
    parts: list[str] = [f"Library match: {best.get('title') or library_guess} (`{lib_id}`)"]
    for sn in (ctx or {}).get("infoSnippets") or []:
        if isinstance(sn, dict) and sn.get("content"):
            parts.append(str(sn["content"])[:1200])
    for sn in (ctx or {}).get("codeSnippets") or []:
        if isinstance(sn, dict) and sn.get("codeList"):
            for c in (sn.get("codeList") or [])[:2]:
                if isinstance(c, dict) and c.get("code"):
                    parts.append("```\n" + str(c["code"])[:500] + "\n```")
    text = "\n\n".join(parts)[:8000]
    item = Item(
        id="ctx7_doc",
        title=f"Context7 docs: {best.get('title') or library_guess}"[:200],
        url="https://context7.com",
        abstract=text[:2000],
        source="context7",
        tags=["mcp", "docs"],
        pillar=Pillar.systems,
        extra={"tool": "context7_mcp", "libraryId": lib_id},
    )
    return text, [item]


@dataclass
class McpEnrichmentResult:
    calls_used: int
    planner_buckets: list[str] = field(default_factory=list)
    planner_questions: list[str] = field(default_factory=list)
    section_evidence: dict[str, str] = field(default_factory=dict)
    contradiction_notes: list[str] = field(default_factory=list)
    enrichment_items: list[Item] = field(default_factory=list)


def run(
    primary: Item,
    calls: int,
    max_c: int,
    trace: list[dict[str, Any]],
) -> McpEnrichmentResult:
    """Planned research enrichment within the shared research call budget."""
    if not config.mcp_enrichment_enabled() or config.dry_run():
        return McpEnrichmentResult(calls_used=calls)
    rem = max_c - calls
    if rem <= 0:
        return McpEnrichmentResult(calls_used=calls)

    title = primary.title
    abstract = primary.abstract
    out = McpEnrichmentResult(calls_used=calls)
    c = calls

    def use(n: int = 1) -> bool:
        nonlocal c, rem
        if rem < n:
            return False
        c += n
        rem -= n
        return True

    if not config.llm_configured():
        return McpEnrichmentResult(calls_used=calls)
    if not use(1):
        return McpEnrichmentResult(calls_used=calls)

    bks, qns = _planner_llm(title, abstract)
    out.planner_buckets = bks
    out.planner_questions = qns
    _trace_append(trace, {"tool": "mcp_planner", "n_questions": len(qns), "n_buckets": len(bks)})

    if not use(1):
        out.calls_used = c
        return out
    adv = _adversarial_llm(title, abstract)
    out.contradiction_notes = adv
    _trace_append(trace, {"tool": "mcp_adversarial", "n": len(adv)})

    section: dict[str, str] = {}
    if qns:
        section["planning"] = "\n".join(f"- {x}" for x in qns)
    if bks:
        section["buckets"] = ", ".join(bks)
    if adv:
        section["counterpoints"] = "\n".join(f"- {x}" for x in adv)

    items: list[Item] = []

    if use(1):
        wq = f"{title[:120]} {abstract[:200]}"
        w_items, w_provider = _web_search(wq)
        items.extend(w_items)
        if w_items:
            section["web_research"] = "\n".join(
                f"- {it.title} — {it.url}\n  {it.abstract[:280]}" for it in w_items[:5]
            )
        _trace_append(
            trace,
            {"tool": "web_search", "provider": w_provider, "n": len(w_items)},
        )

    if use(1):
        g_items = _github_search_repos(
            re.sub(r"[^\w\s-]", " ", title)[:120].strip() or "machine learning"
        )
        items.extend(g_items)
        if g_items:
            section["repositories"] = "\n".join(
                f"- {it.title}: {it.url}\n  {it.abstract[:240]}" for it in g_items[:3]
            )
        _trace_append(trace, {"tool": "github_mcp", "n": len(g_items)})

    if use(1):
        lib = _guess_library_name(title, abstract)
        doc_md = ""
        c_items: list[Item] = []
        if lib:
            doc_md, c_items = _context7_snippet(lib, title, abstract)
        if lib and c_items:
            items.extend(c_items)
        if lib and doc_md:
            section["library_documentation"] = doc_md
        _trace_append(
            trace,
            {"tool": "context7_mcp", "library": lib or "", "n_items": len(c_items)},
        )
    out.section_evidence = section
    out.enrichment_items = items
    out.calls_used = c
    return out

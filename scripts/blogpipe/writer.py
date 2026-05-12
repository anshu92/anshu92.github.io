from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

from . import memory
from .llm import LLMClient
from .models import EvidencePack, RankedItem, WriteResult

EVIDENCE_REF_RE = re.compile(r"\[E(\d+)\]")
NUMBER_RE = re.compile(r"(?<![\w-])\d+(?:\.\d+)?%?(?![\w-])")
HEADING_RE = re.compile(r"^#{2,3}\s+(.+?)\s*$", re.M)
URL_RE = re.compile(r"https?://[^\s<>\])\"]+")
NUMERIC_CLAIM_CUES = (
    "%",
    "percent",
    "x ",
    "ms",
    "s latency",
    "seconds",
    "tokens",
    "tasks",
    "examples",
    "samples",
    "parameters",
    "billion",
    "million",
    "accuracy",
    "score",
    "benchmark",
    "throughput",
    "latency",
    "faster",
    "slower",
    "lower",
    "higher",
    "reduced",
    "improved",
)
DAILY_REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    "technical_thesis": ("technical thesis",),
    "paper_mechanisms": ("paper mechanisms", "mechanisms"),
    "math_or_objective": ("math", "objective"),
    "experiments_and_limits": ("experiments", "evaluation", "evidence", "benchmark", "limits", "caveats"),
    "why_it_matters": ("why it matters", "impact", "implications", "why this matters"),
}


def write_daily(pack: EvidencePack, *, llm: LLMClient | None = None, dry_run: bool = False) -> WriteResult:
    client = llm or LLMClient()
    today = datetime.now(timezone.utc).date().isoformat()
    title = f"Research Radar: Paper Mechanisms and Impact - {today}"
    body = _call_writer(client, _daily_system(), _daily_user(pack, title))
    return _validate_repair_and_publish(
        client=client,
        pack=pack,
        title=title,
        body=body,
        slug=f"{today}-research-radar",
        post_type="daily",
        dry_run=dry_run,
    )


def write_deep_dive(pack: EvidencePack, *, llm: LLMClient | None = None, dry_run: bool = False) -> WriteResult:
    client = llm or LLMClient()
    primary = pack.ranked_items[0].item
    today = datetime.now(timezone.utc).date().isoformat()
    title = f"{primary.title} - guided learning deep dive"
    slug = f"{today}-{memory.slugify(primary.title)}-guided-deep-dive"
    body = _call_writer(client, _deep_system(), _deep_user(pack, title))
    return _validate_repair_and_publish(
        client=client,
        pack=pack,
        title=title,
        body=body,
        slug=slug,
        post_type="deep_dive",
        dry_run=dry_run,
    )


def validate_body(body: str, pack: EvidencePack) -> list[str]:
    errors: list[str] = []
    chunks_by_id = {chunk.evidence_id: chunk for chunk in pack.chunks}
    evidence_ids = set(chunks_by_id)
    refs = {f"E{m.group(1)}" for m in EVIDENCE_REF_RE.finditer(body or "")}
    unknown = sorted(refs - evidence_ids)
    if unknown:
        errors.append("unknown_evidence_ids:" + ",".join(unknown))
    if not refs:
        errors.append("no_evidence_ids")
    body_url_keys = _body_url_keys(body)
    for url in _required_urls_for_refs(refs, chunks_by_id):
        if url and not (_url_keys(url) & body_url_keys):
            errors.append("missing_source_link:" + url)
    evidence_blob = pack.evidence_blob()
    for number in sorted(set(_meaningful_numbers(body))):
        if number not in evidence_blob:
            errors.append("unsupported_number:" + number)
    if _copies_large_evidence_span(body, pack):
        errors.append("copied_large_evidence_span")
    if pack.kind == "daily":
        errors.extend(_validate_daily_technical_focus(body, pack))
    try:
        MarkdownIt().parse(body)
    except Exception as exc:
        errors.append(f"markdown_parse_error:{exc}")
    return errors


def _call_writer(client: LLMClient, system: str, user: str) -> str:
    return _strip_fence(client.complete(system=system, user=user)).strip()


def _validate_repair_and_publish(
    *,
    client: LLMClient,
    pack: EvidencePack,
    title: str,
    body: str,
    slug: str,
    post_type: str,
    dry_run: bool,
) -> WriteResult:
    errors = validate_body(body, pack)
    repair_attempted = False
    if errors:
        repair_attempted = True
        repair_user = _repair_user(pack, body, errors)
        try:
            body = _call_writer(client, _repair_system(), repair_user)
            errors = validate_body(body, pack)
        except Exception as exc:
            errors.append(f"repair_failed:{exc}")
    if errors:
        report = memory.REPORTS / f"{slug}.blocked.json"
        memory.ensure_dirs()
        report.write_text(
            json.dumps(
                {"title": title, "slug": slug, "errors": errors, "body": body},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return WriteResult(ok=False, title=title, body=body, errors=errors, repair_attempted=repair_attempted)
    post = _frontmatter(title, post_type, pack) + "\n" + body.strip() + "\n"
    if dry_run:
        path = memory.REPORTS / f"{slug}.preview.md"
    else:
        path = memory.CONTENT_POST / f"{slug}.md"
    memory.ensure_dirs()
    path.write_text(post, encoding="utf-8")
    return WriteResult(ok=True, path=str(path.relative_to(memory.ROOT)), title=title, body=body, repair_attempted=repair_attempted)


def _frontmatter(title: str, post_type: str, pack: EvidencePack) -> str:
    tracks = sorted({track for ranked in pack.ranked_items for track in ranked.topic_scores.tracks})
    source_count = len(pack.ranked_items)
    paper_count = sum(1 for r in pack.ranked_items if r.item.source_kind == "paper")
    blog_count = sum(1 for r in pack.ranked_items if r.item.source_kind == "blog")
    return "\n".join(
        [
            "---",
            f'date: "{datetime.now(timezone.utc).date().isoformat()}"',
            "draft: true",
            f'title: "{_yaml_escape(title)}"',
            f"post_type: {post_type}",
            "categories: [\"Machine Learning\"]",
            "tags: [\"research-radar\", \"llm\", \"mle\", \"aec\"]",
            f"tracks: {json.dumps(tracks)}",
            f"source_count: {source_count}",
            f"paper_count: {paper_count}",
            f"blog_count: {blog_count}",
            "math: true",
            "mermaid: true",
            "---",
        ]
    )


def _daily_system() -> str:
    return (
        "You are writing Synaptic Radio, a technical ML research blog. Write polished Markdown, not JSON. "
        "The post is paper-centered: explain mechanisms, math/objectives when evidence supports them, experiments, limits, and impact. "
        "Use the evidence pack only. The prose must be original and evidence-grounded. "
        "Every substantive item paragraph must include one or more evidence markers like [E1] and a source link. "
        "Do not invent numbers, benchmarks, authors, or claims. Target 1200-2000 words when evidence supports it."
    )


def _daily_user(pack: EvidencePack, title: str) -> str:
    return (
        f"TITLE: {title}\n\n"
        "Write a paper-first technical blog post. Use these Markdown sections exactly: "
        "Technical thesis, Paper mechanisms, Math or objective details, Experiments and limits, Why it matters, "
        "Supporting engineering context. Do not create a Top engineering blogs section; source blogs are supporting context only. "
        "Cover 5-8 strongest items when evidence supports them; do not force coverage of every input item. "
        "For each item you discuss, answer what problem it attacks, what mechanism or objective it uses, what evidence supports it, "
        "what limitation or caveat is visible, and why it matters. Include source URLs inline for every cited evidence ID. "
        "Avoid exact numeric claims unless the number appears verbatim in the evidence text.\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}"
    )


def _deep_system() -> str:
    return (
        "You write guided technical deep dives for ML engineers. Use only the evidence pack. "
        "Center the method, math/objective interpretation, experiment analysis, reproduction notes, limits, and impact. "
        "Write original Markdown. Include evidence markers [E1] and source links for substantive claims. "
        "Include a Mermaid diagram and one code or pseudocode block only if supported by evidence. "
        "Do not invent numbers or implementation details. Target 2500-4500 words when evidence supports it."
    )


def _deep_user(pack: EvidencePack, title: str) -> str:
    return (
        f"TITLE: {title}\n\n"
        "Write sections for technical thesis, prerequisites, problem framing, method walkthrough, "
        "math or objective interpretation, experiments, reproduction notes, limits, impact, and study questions.\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}"
    )


def _repair_system() -> str:
    return (
        "Repair Markdown for factuality and source grounding. Return only Markdown body. "
        "Keep the required daily technical sections when repairing daily posts. "
        "Every substantive paragraph must include valid evidence IDs such as [E1] and source URLs from the evidence pack. "
        "Do not add unsupported facts. Remove claims that cannot be tied to evidence."
    )


def _repair_user(pack: EvidencePack, body: str, errors: list[str]) -> str:
    chunks_by_id = {chunk.evidence_id: chunk for chunk in pack.chunks}
    refs = {f"E{m.group(1)}" for m in EVIDENCE_REF_RE.finditer(body or "")}
    cited_source_requirements = [
        {"evidence_id": evidence_id, "title": chunks_by_id[evidence_id].title, "url": chunks_by_id[evidence_id].url}
        for evidence_id in sorted(refs)
        if evidence_id in chunks_by_id
    ]
    section_contract = (
        "For daily posts, include these exact level-2 headings: Technical thesis, "
        "Paper mechanisms, Math or objective details, Experiments and limits, Why it matters, "
        "Supporting engineering context."
        if pack.kind == "daily"
        else "Preserve the requested deep-dive technical structure."
    )
    return (
        "Fix this Markdown draft. Return only the repaired Markdown body.\n\n"
        f"{section_contract}\n"
        "Hard requirements:\n"
        "- Use only evidence IDs present in EVIDENCE_PACK, formatted exactly like [E1].\n"
        "- For every evidence ID you cite, include the matching source URL inline in that paragraph.\n"
        "- You do not need to cover every item in EVIDENCE_PACK; omit weak items rather than inventing details.\n"
        "- Remove unsupported numbers and unsupported claims; prefer qualitative phrasing when uncertain.\n"
        "- Do not output JSON, explanations, or a validation report.\n\n"
        f"VALIDATOR_ERRORS:\n{json.dumps(_repair_safe_errors(errors), indent=2)}\n\n"
        f"CITED_SOURCE_URLS:\n{json.dumps(cited_source_requirements, indent=2, ensure_ascii=False)}\n\n"
        f"EVIDENCE_PACK:\n{pack.as_prompt_json()}\n\n"
        f"DRAFT:\n{_repair_safe_draft(body, errors)}"
    )


def _repair_safe_errors(errors: list[str]) -> list[str]:
    safe: list[str] = []
    for error in errors:
        if error.startswith("unsupported_number:"):
            if "unsupported_number" not in safe:
                safe.append("unsupported_number")
            continue
        if error.startswith("missing_source_link:"):
            if "missing_source_link_for_cited_evidence" not in safe:
                safe.append("missing_source_link_for_cited_evidence")
            continue
        safe.append(error)
    return safe


def _repair_safe_draft(body: str, errors: list[str]) -> str:
    safe = body
    for error in errors:
        if error.startswith("unsupported_number:"):
            token = error.split(":", 1)[1]
            safe = safe.replace(token, "[unsupported number]")
    return safe


def _strip_fence(text: str) -> str:
    stripped = (text or "").strip()
    m = re.match(r"^```(?:markdown|md)?\n([\s\S]*?)\n```$", stripped, re.I)
    return m.group(1).strip() if m else stripped


def _yaml_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _required_urls_for_refs(refs: set[str], chunks_by_id: dict[str, object]) -> list[str]:
    urls = {
        chunk.url
        for evidence_id in refs
        if (chunk := chunks_by_id.get(evidence_id)) is not None and getattr(chunk, "url", "")
    }
    return sorted(urls)


def _body_url_keys(body: str) -> set[str]:
    return {key for url in URL_RE.findall(body or "") for key in _url_keys(_strip_url_punctuation(url))}


def _strip_url_punctuation(url: str) -> str:
    return (url or "").rstrip(".,;:")


def _url_keys(url: str) -> set[str]:
    raw = (url or "").strip()
    if not raw:
        return set()
    parts = urlsplit(raw)
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parts.path.rstrip("/") or "/"
    keys = {f"{netloc}{path}".lower()}
    if netloc == "arxiv.org" and path.startswith("/abs/"):
        arxiv_id = path.rsplit("/", 1)[-1].lower()
        unversioned = re.sub(r"v\d+$", "", arxiv_id)
        keys.add(f"arxiv:{arxiv_id}")
        keys.add(f"arxiv:{unversioned}")
    return keys


def _meaningful_numbers(body: str) -> list[str]:
    scan = re.sub(r"https?://\S+", " ", body or "")
    out: list[str] = []
    for match in NUMBER_RE.finditer(scan):
        token = match.group(0)
        if len(token) == 4 and token.startswith(("19", "20")):
            continue
        if token in {"1", "2", "3", "4", "5", "6", "7", "8"}:
            continue
        if _looks_like_arxiv_identifier(token):
            continue
        if not _looks_like_numeric_claim(scan, match.start(), match.end(), token):
            continue
        out.append(token)
    return out


def _looks_like_arxiv_identifier(token: str) -> bool:
    return bool(re.fullmatch(r"\d{4}\.\d{4,5}", token or ""))


def _looks_like_numeric_claim(text: str, start: int, end: int, token: str) -> bool:
    if token.endswith("%"):
        return True
    window = text[max(0, start - 48) : min(len(text), end + 64)].lower()
    return any(cue in window for cue in NUMERIC_CLAIM_CUES)


def _copies_large_evidence_span(body: str, pack: EvidencePack) -> bool:
    compact_body = re.sub(r"\s+", " ", body or "")
    for chunk in pack.chunks:
        text = re.sub(r"\s+", " ", chunk.text or "").strip()
        if len(text) >= 300 and text[:300] in compact_body:
            return True
    return False


def _validate_daily_technical_focus(body: str, pack: EvidencePack) -> list[str]:
    errors: list[str] = []
    headings = [h.lower() for h in HEADING_RE.findall(body or "")]
    for name, aliases in DAILY_REQUIRED_SECTIONS.items():
        if not any(any(alias in heading for alias in aliases) for heading in headings):
            errors.append(f"missing_daily_section:{name}")
    if not any(r.item.source_kind == "paper" for r in pack.ranked_items):
        return errors
    if _looks_like_generic_roundup(headings):
        errors.append("generic_roundup_structure")
    lower = (body or "").lower()
    technical_terms = (
        "method", "mechanism", "objective", "loss", "equation", "algorithm",
        "architecture", "experiment", "ablation", "benchmark", "limitation", "impact",
    )
    if sum(1 for term in technical_terms if term in lower) < 4:
        errors.append("insufficient_technical_focus")
    return errors


def _looks_like_generic_roundup(headings: list[str]) -> bool:
    generic = {"top papers", "top engineering blogs", "what mattered today"}
    required = {"technical thesis", "paper mechanisms", "math or objective details"}
    return bool(generic & set(headings)) and not bool(required & set(headings))

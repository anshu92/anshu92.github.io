"""Generate Hug draft markdown from EvidenceBundle + EditorialBrief + PostFormat.

For section-level refinement with the same PR gate, prefer ``python -m blogpipe graph``.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from . import config, formats, lint, memory, openrouter_client
from .llm_chain import get_llm_usage
from .models import EditorialBrief, EvidenceBundle, Item, VisualPlan
from .memory import _ROOT
from .prompting import render_prompt
from .voice import get_anchor

LOG = logging.getLogger(__name__)


def _voice_exemplar_block_for_prompt() -> str:
    """Return the chosen voice anchor's WRITING EXEMPLAR text for the system prompt."""
    try:
        return get_anchor().exemplar_block or ""
    except Exception:
        return ""


_DRAFT_REWRITE_SYSTEM = (
    "Rewrite into a final markdown body about the PRIMARY paper only. No questions, no topic "
    "swaps, no related-post pivots. Use story-specific ## headings, one ```mermaid``` block, and "
    "grounded [cite: id] claims. Output markdown only, no outer frontmatter, no code fences around "
    "the whole post."
)

_CITE_REPAIR_SYSTEM = (
    "Fix [missing cite: ...] by paraphrase without a cite or by removing the sentence. Output the "
    "full revised markdown body only, no commentary, no code fences."
)

_DRAFT_STRUCTURAL_REPAIR_SYSTEM = (
    "Fix the listed issues while keeping the post on the PRIMARY paper. Do not ask questions. "
    "Return the full markdown body with one mermaid block and story-specific ## headings. Ground "
    "numbers in EVIDENCE. Output markdown only, no code fences around the whole post."
)

_DRAFT_GROUNDING_SOFTENER_SYSTEM = (
    "Revise the full markdown body to remove unsupported numeric claims while keeping the same paper, "
    "sections, and main thesis. Rules: (1) do not invent any new numbers, thresholds, seeds, or setup "
    "details, (2) if a sentence uses derived arithmetic like 'up to 6 points' and that exact delta is "
    "not explicitly in EVIDENCE, rewrite it to cite the underlying benchmark numbers instead, "
    "(3) if a recommendation includes exact counts like prompts or seeds that are not in EVIDENCE, "
    "generalize the recommendation rather than fabricating specifics, (4) keep citations or add them "
    "where needed, (5) output markdown only."
)

_DERIVED_ARITHMETIC_HINT = re.compile(
    r"\b(?:up to|about|roughly|nearly|almost|around)\s+\d+(?:\.\d+)?\s+points?\b",
    re.I,
)
_ILLUSTRATIVE_SCORE_NOTE = re.compile(
    r"^\*?Note:\s*Scores are illustrative\b.*$",
    re.I | re.M,
)
_GENERIC_COMPARATIVE_SENTENCE = re.compile(
    r"\b(?:significant advantage|substantial gains|improves?|boosts?|enhances?)\b"
    r"(?![^.]{0,120}\b(?:%|x|ms|s|accuracy|score|metric|benchmark|fid|psnr|ssim|mmlu)\b)",
    re.I,
)
_EVALUATIVE_PARA_WITHOUT_SUPPORT = re.compile(
    r"\b(competitive|impressive|strong|useful|valuable|important|significant(?:ly)?|matters)\b",
    re.I,
)
_CORRUPTED_COMPARATIVE_PHRASES = (
    (
        re.compile(
            r"scaling laws suggest that model quality suggests stronger performance predictably with total parameters",
            re.I,
        ),
        "Scaling laws suggest that model quality scales predictably with total parameters",
    ),
    (
        re.compile(r"\bto suggests stronger performance quality via more experts\b", re.I),
        "to improve model quality via more experts",
    ),
    (
        re.compile(r"\bis a suggests stronger performance\b", re.I),
        "is a meaningful advantage",
    ),
)


def _slugify(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", title.lower())
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    return s[:80] or "post"


def _load_json(name: str) -> dict:
    p = _ROOT / "reports" / name
    return json.loads(p.read_text()) if p.is_file() else {}


_ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,6})", re.I)


def _short_anchor(it, primary_id: str, occurrence: int) -> str:
    """Compact citation anchor: arXiv id when available; 'the paper' for repeats of the primary."""
    if it.id == primary_id and occurrence > 0:
        return "the paper"
    m = _ARXIV_ID.search(it.url or "")
    if m:
        return f"arXiv:{m.group(1)}"
    title = (it.title or "").strip()
    short = title.split(":", 1)[0]
    if len(short) > 48:
        short = short[:45].rsplit(" ", 1)[0] + "…"
    return short or it.id


def _resolve_cites(md: str, bundle: EvidenceBundle) -> str:
    bundle.register_ids()
    primary_id = bundle.primary.id
    seen: dict[str, int] = {}

    def repl(m: re.Match[str]) -> str:
        cid = m.group(1).strip()
        it = bundle.by_id.get(cid)
        if not it:
            return f"[missing cite: {cid}]"
        occurrence = seen.get(cid, 0)
        seen[cid] = occurrence + 1
        return f"[{_short_anchor(it, primary_id, occurrence)}]({it.url})"

    out = re.sub(r"\[cite:\s*([^\]]+)\]", repl, md)
    if not bundle.by_id:
        return out
    bare_ids = "|".join(re.escape(k) for k in sorted(bundle.by_id.keys(), key=len, reverse=True))
    if not bare_ids:
        return out
    return re.sub(rf"\[({bare_ids})\](?!\()", repl, out)


def _strip_unresolved(md: str) -> bool:
    return bool(re.search(r"\[missing cite:", md))


def _cleanup_missing_cites(md: str) -> str:
    md = re.sub(r"\s*\[missing cite:\s*[^\]]+\]", "", md)
    md = re.sub(r"[ \t]+\n", "\n", md)
    return md


def _unwrap_markdown_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:markdown|md)?\n([\s\S]*?)\n```$", t, re.I)
    return m.group(1).strip() if m else t


def _has_result_signal(text: str) -> bool:
    return bool(lint.numeric_claims(text))


_MERMAID_BARE_OPENER = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<kw>graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|"
    r"gantt|pie|journey|gitGraph|mindmap|timeline|requirementDiagram|c4Context)\b"
    r"(?P<rest>[^\n]*)$",
    re.M,
)


def _wrap_unfenced_mermaid_blocks(body: str) -> str:
    """Wrap stray ``graph TD`` / ``flowchart …`` blocks (not inside a code fence) in ``mermaid`` fences."""
    text = body or ""
    if not text:
        return text
    fence_spans: list[tuple[int, int]] = []
    for m in re.finditer(r"```[\s\S]*?```", text):
        fence_spans.append((m.start(), m.end()))

    def in_fence(idx: int) -> bool:
        for s, e in fence_spans:
            if s <= idx < e:
                return True
        return False

    out_parts: list[str] = []
    cur = 0
    for m in _MERMAID_BARE_OPENER.finditer(text):
        if in_fence(m.start()):
            continue
        line_start = m.start()
        # Find end of block: first blank line or next H1-H6 heading or end of text.
        rest = text[m.end():]
        end_in_rest = len(rest)
        for tail in re.finditer(r"\n[ \t]*\n|^#{1,6}\s+", rest, re.M):
            if tail.start() == 0:
                continue
            end_in_rest = tail.start()
            break
        block_end = m.end() + end_in_rest
        # Leave anything after the block alone and never swallow another fence.
        for s, _ in fence_spans:
            if line_start < s < block_end:
                block_end = s
                break
        block = text[line_start:block_end].rstrip()
        out_parts.append(text[cur:line_start])
        out_parts.append("```mermaid\n" + block + "\n```")
        cur = block_end
    if cur == 0:
        return text
    out_parts.append(text[cur:])
    return "".join(out_parts)


def _promote_h3_when_no_h2(body: str) -> str:
    """Writer mistake rescue: if there are zero ``## `` lines but >= 2 ``### `` lines, promote one level."""
    text = body or ""
    if not text:
        return text
    has_h2 = re.search(r"^##\s+\S", text, re.M) is not None
    if has_h2:
        return text
    h3_count = len(re.findall(r"^###\s+\S", text, re.M))
    if h3_count < 2:
        return text
    LOG.info("draft: no H2 sections; promoting %d H3s to H2", h3_count)
    return re.sub(r"^###\s+", "## ", text, flags=re.M)


def _ensure_mermaid(body: str, title: str, brief: EditorialBrief) -> str:
    body = _wrap_unfenced_mermaid_blocks(body)
    if "```mermaid" in body:
        return body
    LOG.info("draft: no mermaid block; injecting diagram")
    if config.dry_run() or not config.llm_configured():
        return (
            body.rstrip()
            + "\n\n```mermaid\nflowchart LR\n  A[Input] --> B[Core method] --> C[Output]\n```\n"
        )
    raw = openrouter_client.llm_text(
        f"Output one ```mermaid``` fenced block only, no prose. Style: {brief.diagram_style}. "
        "Flow: main inputs → method steps → outputs.",
        f"Paper: {title}",
        max_tokens=1024,
        task="draft_rewrite_section",
    )
    if raw.strip():
        body = body.rstrip() + "\n\n" + raw.strip() + "\n"
    return body


def _apply_result_first_takeaway(body: str, title: str) -> str:
    line = body.split("\n", 1)[0].strip() if body.strip() else ""
    if not line or _has_result_signal(line):
        return body
    if config.dry_run() or not config.llm_configured():
        return body
    fixed = openrouter_client.llm_text(
        "One sentence, first line only: name the key result with a number. No quote marks, no preface.",
        f"Title: {title}\nCurrent first line: {line}",
        max_tokens=512,
        task="draft_rewrite_section",
    )
    if not fixed.strip() or not _has_result_signal(fixed):
        return body
    new_line = fixed.strip()[:500]
    if "\n" in body:
        return new_line + "\n" + body.split("\n", 1)[1]
    return new_line


def _looks_like_markdown_body(text: str) -> bool:
    t = _unwrap_markdown_fence(text)
    if "## " not in t:
        return False
    if len(t.split()) < 120:
        return False
    bad_starts = (
        "i've identified",
        "i identified",
        "here's the revised",
        "here is the revised",
        "i replaced",
        "i removed",
    )
    low = t.lower()
    if low.startswith(bad_starts):
        return False
    bad_contains = (
        "i cannot write a coherent technical blog post",
        "to proceed, i need clarification",
        "please confirm which topic",
        "should i write about",
        "these two unrelated topics",
    )
    return not any(x in low for x in bad_contains)


def _plain_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^#{1,6}\s*", "", t)
    t = re.sub(r"^[-*+]\s+", "", t)
    t = re.sub(r"\[(.*?)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"[*_`~]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" '\"")


def _canonical_text(text: str) -> str:
    t = _plain_text(text).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _dedupe_adjacent_lines(body: str) -> str:
    out: list[str] = []
    prev = ""
    in_fence = False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            prev = ""
            continue
        if (
            not in_fence
            and line.strip()
            and prev
            and _canonical_text(line) == _canonical_text(prev)
        ):
            continue
        out.append(line)
        if not in_fence and line.strip():
            prev = line
    return "\n".join(out).strip() + "\n"


def _normalize_takeaway_line(body: str) -> str:
    lines = body.splitlines()
    if not lines:
        return body
    if lines[0].lstrip().startswith("#"):
        lines[0] = _plain_text(lines[0])
    return "\n".join(lines).strip() + "\n"


def _normalize_research_attribution(body: str) -> str:
    text = body or ""
    replacements = (
        (
            r"\bWe (introduce|present|propose|show|find|observe|use|model|evaluate|analyze|"
            r"study|train|fine-tune|adapt|apply|achieve|compare|extend|demonstrate|report|focus|"
            r"design|identify|derive|select|benchmark|measure|test)\b",
            r"The authors \1",
        ),
        (
            r"\bwe (introduce|present|propose|show|find|observe|use|model|evaluate|analyze|"
            r"study|train|fine-tune|adapt|apply|achieve|compare|extend|demonstrate|report|focus|"
            r"design|identify|derive|select|benchmark|measure|test)\b",
            r"the authors \1",
        ),
        (
            r"\bOur ((?:[A-Za-z-]+\s+){0,3}(?:methodology|method|methods|approach|approaches|"
            r"model|models|results|experiments|study|paper|framework|analysis|system|systems|"
            r"technique|techniques|baseline|baselines|llms?|μlms?))\b",
            r"The authors' \1",
        ),
        (
            r"\bour ((?:[A-Za-z-]+\s+){0,3}(?:methodology|method|methods|approach|approaches|"
            r"model|models|results|experiments|study|paper|framework|analysis|system|systems|"
            r"technique|techniques|baseline|baselines|llms?|μlms?))\b",
            r"the authors' \1",
        ),
    )
    for pat, repl in replacements:
        text = re.sub(pat, repl, text)
    text = re.sub(r"\b([Tt]he authors [a-z]+) the authors'\s+", r"\1 their ", text)
    return text


def _section_text(bundle: EvidenceBundle, *keys: str) -> str:
    for key in keys:
        text = str(bundle.section_evidence.get(key) or "").strip()
        if text:
            return text
    return ""


def _pick_section_sentence(bundle: EvidenceBundle, *keys: str, numeric: bool = False) -> str:
    text = _section_text(bundle, *keys)
    for sent in _split_sentences(_normalize_research_attribution(_plain_text(text))):
        if len(sent.split()) < 6 and ":" not in sent:
            continue
        if numeric and not _has_result_signal(sent):
            continue
        return sent[:500]
    return ""


def _paper_link(bundle: EvidenceBundle, label: str = "the paper") -> str:
    url = (bundle.primary.url or "").strip()
    if not url:
        return label
    return f"[{label}]({url})"


def _clean_sentence(sent: str) -> str:
    return _normalize_research_attribution(_plain_text(sent))[:500]


def _ensure_grounded_takeaway(body: str, bundle: EvidenceBundle) -> str:
    lines = body.splitlines()
    if not lines:
        return body
    if not _has_result_signal(lines[0]):
        lines[0] = _takeaway_from_bundle(bundle)
    return "\n".join(lines).strip() + "\n"


def _strip_leading_title_echo_heading(body: str, title: str) -> str:
    lines = body.splitlines()
    if not lines:
        return body
    takeaway_key = _canonical_text(lines[0])
    title_key = _canonical_text(title)
    out = [lines[0]]
    in_fence = False
    seen_content = False
    skipped = False
    for line in lines[1:]:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if not skipped and not seen_content and re.match(r"^##\s+", line):
            heading_key = _canonical_text(re.sub(r"^##\s+", "", line))
            if heading_key and heading_key in {takeaway_key, title_key}:
                skipped = True
                continue
        if line.strip():
            seen_content = True
        out.append(line)
    return "\n".join(out).strip() + "\n"


def _normalize_generic_section_headings(body: str) -> str:
    lines = body.splitlines()
    out: list[str] = []
    for line in lines:
        if re.match(r"^##\s*Conclusion\s*$", line, re.I):
            out.append("## What I would test next")
            continue
        if re.match(r"^##\s*How to (?:apply|use).*$", line, re.I) or re.search(
            r"^##\s*.*real-world scenarios.*$", line,
            re.I,
        ):
            out.append("## What a practitioner should test next")
            continue
        if re.match(r"^##\s*Why .*(game[- ]changer).*$", line, re.I):
            out.append("## Why this matters in practice")
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n"


def _drop_illustrative_score_notes(body: str) -> str:
    text = _ILLUSTRATIVE_SCORE_NOTE.sub("", body or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _soften_unmeasured_comparatives(body: str, bundle: EvidenceBundle) -> str:
    out: list[str] = []
    for block in re.split(r"\n{2,}", (body or "").strip()):
        if block.lstrip().startswith(("```", "|")):
            out.append(block)
            continue
        if _GENERIC_COMPARATIVE_SENTENCE.search(block) and not lint.has_numeric_claim(block):
            block = re.sub(
                _GENERIC_COMPARATIVE_SENTENCE,
                "suggests stronger performance",
                block,
                count=1,
            )
            if "http" not in block and "[cite:" not in block:
                block = block.rstrip() + f" {_paper_link(bundle)}"
        out.append(block)
    return "\n\n".join(x for x in out if x.strip()).strip() + "\n"


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [x.strip() for x in raw if x and x.strip()]


def _clean_fallback_row_name(text: str) -> str:
    t = re.sub(r"\s+", " ", text or "").strip(" ,.;:")
    if "," in t:
        t = t.split(",")[-1].strip()
    t = re.sub(r"^while\s+", "", t, flags=re.I)
    t = re.sub(r"^outperforming\s+", "", t, flags=re.I)
    t = re.sub(r"^(using|with|both|the|and|as well as)\s+", "", t, flags=re.I)
    t = re.sub(r"^we achieve superior performance on\s+", "", t, flags=re.I)
    t = re.sub(r"^we achieve\s+", "", t, flags=re.I)
    t = re.sub(r"^significantly outperforming both\s+", "", t, flags=re.I)
    t = re.sub(r"^significantly outperforming\s+", "", t, flags=re.I)
    if "using only" in t.lower():
        t = t.split("using only", 1)[-1].strip()
    t = re.sub(r"^the baseline\s+", "", t, flags=re.I)
    return t[:80]


def _paper_result_rows(bundle: EvidenceBundle) -> list[tuple[str, str, str]]:
    text = _section_text(bundle, "paper_experiments")
    if not text:
        return []
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for sent in _split_sentences(_plain_text(text)):
        for pat in (
            r"([A-Za-z0-9][A-Za-z0-9+/\-() ]{2,90}?)\s+(?:achieves?|reaches?|yields?|scores?)\s+(\d+(?:\.\d+)?%)",
            r"([A-Za-z0-9][A-Za-z0-9+/\-() ]{2,90}?)\s+at\s+(\d+(?:\.\d+)?%)",
        ):
            for m in re.finditer(pat, sent, re.I):
                name = _clean_fallback_row_name(m.group(1))
                value = m.group(2)
                key = f"{name.lower()}::{value}"
                if not name or key in seen:
                    continue
                seen.add(key)
                rows.append((name[:80], value, "same paper setup"))
                if len(rows) >= 4:
                    return rows
    return rows[:4]


def _fallback_benchmark_rows(bundle: EvidenceBundle) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for r in bundle.benchmarks:
        val = (r.value or "").strip()
        if not val or val == "(see paper)":
            continue
        metric = f"{val}{(' ' + r.unit.strip()) if r.unit else ''}".strip()
        rows.append((r.name.strip()[:80], metric[:120], (r.baseline or "n/a")[:120]))
    if rows:
        return rows[:4]
    paper_rows = _paper_result_rows(bundle)
    if paper_rows:
        return paper_rows[:4]
    return []


def _render_results_table(rows: list[tuple[str, str, str]]) -> str:
    lines = ["| Method | Metric | Baseline |", "| --- | --- | --- |"]
    for method, metric, baseline in rows[:4]:
        lines.append(f"| {method} | {metric} | {baseline or 'n/a'} |")
    return "\n".join(lines)


def _repair_or_inject_results_table(body: str, bundle: EvidenceBundle) -> str:
    rows = _fallback_benchmark_rows(bundle)
    evidence_text = bundle.model_dump_json()
    table_block = re.compile(
        r"(?ms)^\s*\|[^\n]*\bMethod\b[^\n]*\|\n^\s*\|(?:[-: ]+\|)+\n(?:^\s*\|.*\|\n?)+"
    )
    match = table_block.search(body)
    if not rows:
        if match:
            return body[: match.start()] + body[match.end() :]
        return body
    table = _render_results_table(rows)
    if match:
        table_text = match.group(0)
        if lint.unsupported_numeric_claims(table_text, evidence_text):
            return body[: match.start()] + table + body[match.end() :]
        return body
    insert = "\n\n## Numbers the paper actually gives us\n\n" + table + "\n"
    mermaid_match = re.search(r"```mermaid[\s\S]*?```", body, re.I)
    if mermaid_match:
        return body[: mermaid_match.end()] + insert + body[mermaid_match.end() :]
    return body.rstrip() + insert + "\n"


def _drop_unsupported_numeric_paragraphs(body: str, bundle: EvidenceBundle) -> str:
    bad = set(lint.unsupported_numeric_claims(body, bundle.model_dump_json()))
    if not bad:
        return body
    blocks = re.split(r"\n{2,}", body.strip())
    kept: list[str] = []
    for idx, block in enumerate(blocks):
        if "| Method | Metric | Baseline |" in block:
            head, sep, tail = block.partition("| Method | Metric | Baseline |")
            if any(claim in head for claim in bad):
                kept.append((sep + tail).strip())
                continue
        if "```" in block or block.lstrip().startswith("|"):
            kept.append(block)
            continue
        if any(claim in block for claim in bad):
            if idx == 0:
                kept.append(_takeaway_from_bundle(bundle))
            continue
        kept.append(block)
    return "\n\n".join(x for x in kept if x.strip()).strip() + "\n"


def _drop_orphan_headings(body: str) -> str:
    lines = body.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^##\s+", line):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines) or re.match(r"^##\s+", lines[j]):
                i += 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out).strip() + "\n"


def _takeaway_from_bundle(bundle: EvidenceBundle) -> str:
    rows = _fallback_benchmark_rows(bundle)
    pct_rows = [r for r in rows if "%" in r[1]]
    if len(pct_rows) >= 2:
        lead = pct_rows[0]
        alt = pct_rows[1]
        lead_name = lead[0]
        if re.match(r"^\d", lead_name):
            lead_name = f"Using {lead_name}"
        sent = f"{lead_name} reaches {lead[1]}, ahead of {alt[0]} at {alt[1]}."
        if _has_result_signal(sent):
            return _normalize_research_attribution(sent[:220])
    for sent in _split_sentences(bundle.primary.abstract):
        cleaned = re.sub(r"https?://\S+", "", sent).strip()
        if _has_result_signal(cleaned) and re.search(
            r"(introduce|achieve|outperform|beat|match|first|latenc|parameter|layer)",
            cleaned,
            re.I,
        ):
            return _normalize_research_attribution(_plain_text(cleaned)[:240])
    for key in ("paper_experiments", "paper_method", "paper_problem"):
        sent = _pick_section_sentence(bundle, key, numeric=True)
        if sent:
            return sent[:240]
    eval_text = " ".join(
        x
        for x in (
            _section_text(bundle, "paper_method"),
            _section_text(bundle, "paper_experiments"),
            " ".join(q.text for q in (bundle.quotes or [])[:6] if getattr(q, "text", "")),
        )
        if x
    )
    metric_count = re.search(r"\b(four|4)\s+metrics?\b", eval_text, re.I)
    if metric_count:
        return (
            "GSI-Bench scores spatial editing on four metrics, giving the paper a concrete way "
            "to test whether generative training transfers back to spatial reasoning."
        )
    if rows:
        method, metric, baseline = rows[0]
        sent = f"{method} reaches {metric} against {baseline}."
        if _has_result_signal(sent):
            return _normalize_research_attribution(sent[:220])
    title = bundle.primary.title.split(":")[0][:120]
    return f"{title} is interesting only if the concrete tradeoff survives contact with deployment."


_GLOSS_PARENS = re.compile(r"\s*\(\s*[A-Z][^)]*[a-z][^)]*\)\s*[—-]\s*[a-z][^.]*?(?=[.,;:]|$)")


def _strip_explainer_cruft(s: str) -> str:
    """Remove appended glossary expansions like 'LLM (Large Language Model) — a type of...' from short text."""
    return _GLOSS_PARENS.sub("", s)


def _first_sentence(s: str, limit: int) -> str:
    """Return first sentence, hard-capped at limit; never cut mid-word."""
    s = (s or "").strip()
    if not s:
        return ""
    m = re.search(r"[.!?](?:\s|$)", s)
    if m:
        s = s[: m.end()].rstrip()
    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit(" ", 1)[0].rstrip(",;:- ")
    if not cut.endswith((".", "!", "?")):
        cut = cut + "…"
    return cut


def _sanitize_frontmatter_text(text: str, limit: int) -> str:
    s = _plain_text(text).replace('"', "'")
    s = _strip_explainer_cruft(s)
    s = re.sub(r"\bScores are illustrative\b.*", "", s, flags=re.I)
    return _first_sentence(s, limit)


def _frontmatter_takeaway(body: str, bundle: EvidenceBundle) -> str:
    line = (body.split("\n", 1)[0] if body else "").strip()
    if not line:
        return _sanitize_frontmatter_text(_takeaway_from_bundle(bundle), 500)
    unsupported = lint.unsupported_numeric_claims(line, bundle.model_dump_json())
    if unsupported:
        return _sanitize_frontmatter_text(_takeaway_from_bundle(bundle), 500)
    return _sanitize_frontmatter_text(line, 500)


def _ensure_mechanism_section(body: str, bundle: EvidenceBundle) -> str:
    if re.search(r"\b(why this works|how it works|mechanism)\b", body or "", re.I):
        return body
    method = _pick_section_sentence(bundle, "paper_method")
    if not method:
        return body
    mech = (
        "## Why this works\n\n"
        f"{method} {_paper_link(bundle)}\n"
    )
    table_match = re.search(r"(?ms)^##\s+Numbers the paper actually gives us.*?(?=^##\s+|\Z)", body or "")
    if table_match:
        return body[: table_match.start()] + mech + "\n" + body[table_match.start():]
    first_h2 = re.search(r"^##\s+", body or "", re.M)
    if first_h2:
        return body[: first_h2.start()] + mech + "\n" + body[first_h2.start():]
    return mech + "\n" + (body or "")


def _ensure_decision_section(body: str, bundle: EvidenceBundle) -> str:
    if re.search(r"\b(when to use|when not to use|should you use|what i would test next|what a practitioner should test next)\b", body or "", re.I):
        return body
    limits = _pick_section_sentence(bundle, "paper_limitations", "paper_reproducibility")
    if not limits:
        limits = "I would treat this as a benchmark and evaluation contribution first, then test transfer before copying the method into a product setting."
    elif not limits.lower().startswith(("i would", "start with", "before")):
        limits = (
            "I would start by reproducing the smallest reported comparison and only then decide whether the extra complexity is worth adopting. "
            + limits
        )
    section = (
        "## What I would test next\n\n"
        f"{limits} {_paper_link(bundle)}\n"
    )
    return (body.rstrip() + "\n\n" + section).strip() + "\n"


def _add_support_links_to_evaluative_paragraphs(body: str, bundle: EvidenceBundle) -> str:
    out: list[str] = []
    for block in re.split(r"\n{2,}", (body or "").strip()):
        stripped = block.strip()
        if not stripped or stripped.startswith(("```", "|")):
            out.append(block)
            continue
        if _EVALUATIVE_PARA_WITHOUT_SUPPORT.search(stripped) and "http" not in stripped and "[cite:" not in stripped:
            block = block.rstrip() + f" {_paper_link(bundle)}"
        out.append(block)
    return "\n\n".join(x for x in out if x.strip()).strip() + "\n"


def _repair_phrase_corruption(body: str) -> str:
    text = body or ""
    for pat, repl in _CORRUPTED_COMPARATIVE_PHRASES:
        text = pat.sub(repl, text)
    return text.strip() + "\n"


def _glossary_terms_from_bundle(bundle: EvidenceBundle) -> list[str]:
    out: list[str] = []
    for n in bundle.analyst_notes or []:
        if getattr(n, "role", "") == "glossary" and not getattr(n, "skipped", False):
            for c in n.claims or []:
                if c and c.strip():
                    out.append(c.strip())
    return out


def _glossary_block(terms: list[str]) -> str:
    if not terms:
        return ""
    return "\n".join(f"- {t}" for t in terms[:14])


def explain_undefined_terms(body: str, bundle: EvidenceBundle) -> str:
    """Inline-expand acronyms that lack a parenthetical or glossary definition on first use."""
    terms = _glossary_terms_from_bundle(bundle)
    missing = lint.undefined_acronyms(body, terms)
    if not missing:
        return body
    if config.dry_run() or not config.llm_configured():
        return body
    LOG.info("explainer: %s undefined acronyms: %s", len(missing), missing)
    glossary = _glossary_block(terms)
    instruction = (
        "Define the listed acronyms on their first occurrence in prose. GLOSSARY is ground truth. "
        "Format: ACRONYM (expansion) and a short clause, or 'TERM — short definition'.\n"
        "Do not edit (or return the body unchanged): (1) text inside `[text](url)` links, "
        "(2) heading lines, (3) the first body line, (4) numbers, the mermaid block, or [cite: id], "
        "(5) do not add links. Return the full markdown body only, no commentary, no code fences."
    )
    user = (
        f"UNDEFINED_ACRONYMS:\n- " + "\n- ".join(missing[:12]) + "\n\n"
        f"GLOSSARY:\n{glossary or '(empty — infer brief, accurate definitions from context)'}\n\n"
        f"BODY:\n{body[:16000]}\n"
    )
    rewrite = openrouter_client.llm_text(
        instruction, user, max_tokens=2000, task="explainer_rewrite"
    )
    revised = _unwrap_markdown_fence(rewrite)
    if not revised.strip() or not _looks_like_markdown_body(revised):
        return body
    if not _explainer_preserves_structure(body, revised):
        LOG.info("explainer: rejected revision (link/heading structure changed)")
        return body
    after = lint.undefined_acronyms(revised, terms)
    if len(after) >= len(missing):
        return body
    return revised


_LINK_RE = re.compile(r"\[([^\[\]\n]+)\]\(([^)\s]+)\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)


def _explainer_preserves_structure(before: str, after: str) -> bool:
    """Reject explainer rewrites that mutate markdown links, headings, or first line."""
    if before.split("\n", 1)[0].strip() != after.split("\n", 1)[0].strip():
        return False
    b_links = [(t.strip(), u) for t, u in _LINK_RE.findall(before)]
    a_links = [(t.strip(), u) for t, u in _LINK_RE.findall(after)]
    if len(b_links) != len(a_links):
        return False
    for (bt, bu), (at, au) in zip(b_links, a_links):
        if bu != au:
            return False
        if bt != at:
            return False
    b_heads = [(lvl, ttl.strip()) for lvl, ttl in _HEADING_RE.findall(before)]
    a_heads = [(lvl, ttl.strip()) for lvl, ttl in _HEADING_RE.findall(after)]
    if b_heads != a_heads:
        return False
    return True


def _h2_titles_from_body(body: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"^##\s+(.+?)\s*$", body, re.M)]


def _match_h2_for_hint(headings: list[str], hint: str) -> str | None:
    """Pick the H2 that best matches a free-text placement hint."""
    h_l = (hint or "").lower().strip()
    if not h_l:
        return headings[0] if headings else None
    for h in headings:
        if h.lower() in h_l or h_l in h.lower():
            return h
    for h in headings:
        words = re.findall(r"[\w][\w'-]{2,40}", h_l)
        for w in words:
            if len(w) > 3 and w in h.lower():
                return h
    return headings[0] if headings else None


def _end_of_h2_section(body: str, h2_title: str) -> int | None:
    m = re.search(
        r"^##\s+" + re.escape(h2_title.strip()) + r"\s*$\n",
        body,
        re.M,
    )
    if not m:
        return None
    rest = body[m.end() :]
    m2 = re.search(r"^##\s+", rest, re.M)
    if m2:
        return m.end() + m2.start()
    return len(body)


def _insert_before_index(body: str, idx: int, block: str) -> str:
    return body[:idx].rstrip() + "\n\n" + block.strip() + "\n" + body[idx:].lstrip()


def _figure_md(spec, slug: str) -> str:
    u = f"/img/posts/{slug}/figures/{spec.id}.png"
    lines = [f"![{getattr(spec, 'alt', None) or spec.id}]({u})"]
    cap = (getattr(spec, "caption", None) or "").strip()
    if cap:
        lines.append(f"*{cap}*")
    return "\n".join(lines)


_EQ_WRAP_PAIRS = (
    (r"^\$\$\s*", r"\s*\$\$$"),
    (r"^\$\s*", r"\s*\$$"),
    (r"^\\\(\s*", r"\s*\\\)$"),
    (r"^\\\[\s*", r"\s*\\\]$"),
)


def _normalize_latex(lx: str) -> str:
    """Strip outer math delimiters so embedding can apply $$...$$ exactly once."""
    s = (lx or "").strip()
    changed = True
    while changed and s:
        changed = False
        for left, right in _EQ_WRAP_PAIRS:
            if re.match(left, s) and re.search(right, s):
                s = re.sub(right, "", re.sub(left, "", s, count=1), count=1).strip()
                changed = True
                break
    return s


def _equation_md(spec) -> str:
    cap = (getattr(spec, "caption", None) or "").strip()
    lx = _normalize_latex(getattr(spec, "latex", None) or "")
    if not lx:
        return ""
    out = f"$$\n{lx}\n$$"
    if cap:
        return f"*{cap}*\n\n{out}\n"
    return f"{out}\n"


def embed_planned_visuals(body: str, plan: VisualPlan | None, slug: str) -> str:
    """Insert planned figure and equation blocks where still missing and placement matches."""
    if not plan or (not plan.figures and not plan.equations):
        return body
    b = body
    heads = _h2_titles_from_body(b)
    for spec in plan.figures:
        needle = f"/figures/{spec.id}.png"
        if needle in b:
            continue
        h2 = _match_h2_for_hint(heads, spec.placement_hint)
        block = _figure_md(spec, slug)
        end = _end_of_h2_section(b, h2) if h2 else None
        if end is None:
            b = b.rstrip() + "\n\n" + block + "\n"
        else:
            b = _insert_before_index(b, end, block)
    for spec in plan.equations:
        lx = _normalize_latex(spec.latex or "")
        if not lx:
            continue
        compact = lx.replace(" ", "")
        bcompact = b.replace(" ", "")
        if lx in b or (len(compact) > 2 and compact in bcompact):
            continue
        h2 = _match_h2_for_hint(heads, spec.placement_hint)
        block = _equation_md(spec)
        if not block:
            continue
        end = _end_of_h2_section(b, h2) if h2 else None
        if end is None:
            b = b.rstrip() + "\n\n" + block + "\n"
        else:
            b = _insert_before_index(b, end, block)
    return b


def _polish_body(body: str, bundle: EvidenceBundle, brief: EditorialBrief) -> str:
    body = _unwrap_markdown_fence(body)
    # Structural / cite-repair LLM steps can emit [cite: …] ids that are not bundle items;
    # resolve + strip here so every path through polish leaves valid links or no marker.
    bundle.register_ids()
    body = _resolve_cites(body, bundle)
    body = _cleanup_missing_cites(body)
    body = _normalize_research_attribution(body)
    body = _normalize_takeaway_line(body)
    body = _ensure_grounded_takeaway(body, bundle)
    body = _drop_illustrative_score_notes(body)
    body = _soften_unmeasured_comparatives(body, bundle)
    body = _normalize_generic_section_headings(body)
    body = _dedupe_adjacent_lines(body)
    body = _strip_leading_title_echo_heading(body, bundle.primary.title)
    body = _promote_h3_when_no_h2(body)
    body = _repair_or_inject_results_table(body, bundle)
    body = _ensure_mechanism_section(body, bundle)
    body = _ensure_decision_section(body, bundle)
    body = _add_support_links_to_evaluative_paragraphs(body, bundle)
    body = _repair_phrase_corruption(body)
    body = _drop_unsupported_numeric_paragraphs(body, bundle)
    body = _drop_orphan_headings(body)
    body = _ensure_mermaid(body, bundle.primary.title, brief)
    body = _apply_result_first_takeaway(body, bundle.primary.title)
    body = _dedupe_adjacent_lines(body)
    body = _strip_leading_title_echo_heading(body, bundle.primary.title)
    return body.strip() + "\n"


def build_prompt(
    bundle: EvidenceBundle,
    brief: EditorialBrief,
    fmt: formats.PostFormat,
) -> tuple[str, str]:
    bundle.register_ids()
    post_slug = _slugify(bundle.primary.title)
    goals = fmt.goals_markdown()
    ev = []
    ev.append(f"PRIMARY: {bundle.primary.title}\n{bundle.primary.abstract[:900]}")
    b_rows = []
    for r in bundle.benchmarks:
        u = f" {r.unit}" if r.unit else ""
        b_rows.append(
            f"- {r.name}: {r.value}{u} (baseline: {r.baseline or 'n/a'}) {r.notes or ''}"
        )
    if b_rows:
        ev.append(
            "BENCHMARKS_TABLE — use ONLY these numbers in the prose table and inline. "
            "COVERAGE FLOOR: at least 4 distinct numbers from this list must appear in "
            "the body, each paired with a named baseline or comparison. Do not invent "
            "other metrics.\n" + "\n".join(b_rows)
        )
    related_lines: list[str] = []
    for lab, items in (
        ("ANCESTORS", bundle.ancestors),
        ("COMPETITORS", bundle.competitors),
        ("FOLLOWUPS", bundle.followups),
    ):
        for it in items[:4]:
            head = (it.abstract or "").strip().split(". ", 1)[0][:200]
            related_lines.append(f"- [{lab}] {it.title} — {head}")
    if related_lines:
        ev.append(
            "RELATED_WORK_LIST — name >=2 of these entries by their literal title in "
            "the tradeoffs section as 'the road not taken'. Comparing the primary "
            "method to these is what makes the post useful:\n"
            + "\n".join(related_lines[:10])
        )
    if bundle.quotes:
        quote_lines = []
        for i, q in enumerate(bundle.quotes[:6], start=1):
            txt = (q.text or "").strip()
            if not txt:
                continue
            src = getattr(q, "source_id", "") or "primary"
            quote_lines.append(f"{i}. \"{txt[:260]}\" [cite: {src}]")
        if quote_lines:
            ev.append(
                "KEY_QUOTES_NUMBERED — anchor at least one first-person opinion "
                "sentence in one of these quotes (paraphrase, then cite):\n"
                + "\n".join(quote_lines)
            )
    if bundle.planner_questions:
        ev.append(
            "RESEARCH_QUESTIONS (cover across sections where relevant):\n"
            + "\n".join(f"- {q}" for q in bundle.planner_questions[:12])
        )
    if bundle.section_evidence:
        for k, v in bundle.section_evidence.items():
            if (v or "").strip():
                ev.append(f"SECTION_EVIDENCE[{k}]:\n{v.strip()[:1200]}")
    if bundle.contradiction_notes:
        ev.append(
            "CONTRADICTIONS_AND_LIMITS (address honestly; do not bury):\n"
            + "\n".join(f"- {x}" for x in bundle.contradiction_notes[:8])
        )
    if (bundle.committee_synthesis or "").strip():
        ev.append(
            "EDITORIAL_ANGLE (committee): "
            + (bundle.committee_synthesis or "")[:2000]
        )
    glossary_terms: list[str] = []
    if bundle.analyst_notes:
        claim_lines: list[str] = []
        skill_idx = 0
        for n in bundle.analyst_notes:
            if getattr(n, "skipped", False):
                continue
            if n.role == "visual_planner":
                continue
            role = n.role
            for c in (n.claims or [])[:6]:
                head = (c or "").strip().split("\n", 1)[0]
                if not head:
                    continue
                skill_idx += 1
                claim_lines.append(f"{skill_idx}. [{role}] {head[:240]}")
            if role == "glossary":
                glossary_terms.extend(n.claims or [])
        if claim_lines:
            ev.append(
                "ANALYST_CLAIMS_NUMBERED — these are the committee's load-bearing "
                "observations. COVERAGE FLOOR: surface at least 4 of these by name, "
                "literally quoting the operative noun-phrase or number. Do NOT compress "
                "them into one generic sentence:\n"
                + "\n".join(claim_lines[:18])
            )
    if glossary_terms:
        ev.append(
            "GLOSSARY (define each term on its FIRST mention in the body, inline and concise — "
            "either 'TERM (expansion): definition' or 'TERM — definition'. Do not skip any term "
            "the post actually uses, especially acronyms like RDP):\n"
            + "\n".join(f"- {t}" for t in glossary_terms[:14])
        )
    if bundle.visual_plan and (
        bundle.visual_plan.figures or bundle.visual_plan.equations
    ):
        vlines: list[str] = [
            f"VISUAL_PLAN (use exactly; do not add extra images or LaTeX; asset path prefix: "
            f"/img/posts/{post_slug}/figures/<id>.png):"
        ]
        for f in bundle.visual_plan.figures:
            u = f"/img/posts/{post_slug}/figures/{f.id}.png"
            vlines.append(
                f"- FIGURE id={f.id} kind={f.kind} alt={f.alt!r} caption={f.caption!r} "
                f"placement={f.placement_hint!r} insert_markdown: ![{f.alt or f.id}]({u})"
            )
            vlines.append(f"  image_prompt: {f.prompt[:1200]}")
        for e in bundle.visual_plan.equations:
            vlines.append(
                f"- EQUATION id={e.id} placement={e.placement_hint!r} "
                f"caption={e.caption!r} latex_block: $$ ... {e.latex[:800]} ... $$"
            )
        ev.append("\n".join(vlines)[:5000])
    for it in bundle.enrichment_items[:5]:
        ev.append(
            f"ENRICHMENT {it.source} {it.title} | {it.url}\n{it.abstract[:260]}"
        )
    ctx = "\n\n".join(ev)
    if len(ctx) > 11000:
        ctx = ctx[:11000] + "\n\n[EVIDENCE truncated to budget]\n"
    brief_json = brief.model_dump_json()
    system = render_prompt(
        "draft_writer",
        citation_ids=list(bundle.by_id.keys()),
        opener_hook=brief.opener_hook,
        voice_guide=brief.voice_guide,
        voice_override=fmt.voice_override,
        voice_exemplar=_voice_exemplar_block_for_prompt(),
    )
    user = (
        f"CONTENT GOALS (cover across the post, in your own words and your own headings):\n{goals}\n\n"
        f"EDITORIAL BRIEF:\n{brief_json}\n\n"
        f"EVIDENCE:\n{ctx}\n\n"
        f"RELATED POSTS (link with [title](/post/slug/) path only, no ref shortcode in output): "
        f"{brief.recent_topics[:5]}\n"
    )
    return system, user


def soften_unsupported_numeric_claims(
    body: str,
    evidence_text: str,
) -> str:
    unsupported = lint.unsupported_numeric_claims(body, evidence_text)
    derived_hints = list(dict.fromkeys(m.group(0).strip() for m in _DERIVED_ARITHMETIC_HINT.finditer(body or "")))
    issues = list(dict.fromkeys(unsupported + derived_hints))
    if not issues:
        return body
    if config.dry_run() or not config.llm_configured():
        return body
    issue_lines = "\n".join(f"- {x}" for x in issues[:8])
    revised = openrouter_client.llm_text(
        _DRAFT_GROUNDING_SOFTENER_SYSTEM,
        f"UNSUPPORTED_NUMERIC_CLAIMS:\n{issue_lines}\n\nDRAFT_MD:\n{body[:16000]}\n\nEVIDENCE:\n{evidence_text[:16000]}",
        mode="smart",
        max_tokens=config.max_tokens_smart(),
        task="draft_full",
        temperature=config.verifier_temperature(),
    )
    if not revised.strip():
        return body
    out = _unwrap_markdown_fence(revised)
    low = out.lower().strip()
    if "## " not in out or low.startswith(
        (
            "i've identified",
            "i identified",
            "here's the revised",
            "here is the revised",
            "i replaced",
            "i removed",
        )
    ):
        return body
    try:
        bundle = EvidenceBundle.model_validate_json(evidence_text)
        out = _resolve_cites(out, bundle)
    except Exception:
        pass
    return out if out.strip() else body


def run() -> Path:
    memory.ensure_dirs()
    u0 = get_llm_usage()
    bundle = EvidenceBundle.model_validate_json(
        (_ROOT / "reports" / "evidence_bundle.json").read_text()
    )
    brief = EditorialBrief.model_validate_json(
        (_ROOT / "reports" / "editorial_brief.json").read_text()
    )
    fmt = formats.FORMATS.get(brief.format_name) or formats.FORMATS["deep_dive"]
    system, user = build_prompt(bundle, brief, fmt)
    body = ""
    if config.dry_run() or not config.llm_configured():
        body = _stub_body(bundle, brief, fmt)
    else:
        body = openrouter_client.llm_text(
            system, user, max_tokens=config.max_tokens_smart(), task="draft_full"
        )
        if not body.strip():
            body = _stub_body(bundle, brief, fmt)
        elif not _looks_like_markdown_body(body):
            LOG.warning("draft: initial generation not usable; requesting grounded rewrite")
            rewrite = openrouter_client.llm_text(
                _DRAFT_REWRITE_SYSTEM,
                f"PRIMARY_TITLE: {bundle.primary.title}\n\nBAD_DRAFT:\n{body[:12000]}\n\nEVIDENCE:\n{user[:14000]}",
                mode="smart",
                max_tokens=config.max_tokens_smart(),
                task="draft_full",
            )
            if rewrite.strip() and _looks_like_markdown_body(rewrite):
                body = _unwrap_markdown_fence(rewrite)
            else:
                body = _stub_body(bundle, brief, fmt)
    body = _resolve_cites(_unwrap_markdown_fence(body), bundle)
    if _strip_unresolved(body):
        LOG.warning("draft: unresolved cites; attempting one repair")
        repair = openrouter_client.llm_text(
            _CITE_REPAIR_SYSTEM,
            body[:12000],
            max_tokens=2048,
            task="draft_cite_repair",
        )
        if repair.strip() and _looks_like_markdown_body(repair):
            body = _unwrap_markdown_fence(repair)
        else:
            body = _cleanup_missing_cites(body)
    else:
        body = _cleanup_missing_cites(body)
    body = _polish_body(body, bundle, brief)
    struct = lint.structural_issues(body)
    unsupported = lint.unsupported_numeric_claims(body, bundle.model_dump_json())
    if (
        (struct or unsupported)
        and not config.dry_run()
        and config.llm_configured()
    ):
        LOG.warning("draft: attempting rewrite for structural/grounding issues: %s / %s", struct, unsupported)
        rewrite = openrouter_client.llm_text(
            _DRAFT_STRUCTURAL_REPAIR_SYSTEM,
            f"ISSUES:\n- " + "\n- ".join(struct + unsupported[:8]) + f"\n\nBODY:\n{body[:16000]}\n\nEVIDENCE:\n{user[:14000]}",
            mode="smart",
            max_tokens=config.max_tokens_smart(),
            task="draft_full",
        )
        if rewrite.strip() and _looks_like_markdown_body(rewrite):
            body = _polish_body(_unwrap_markdown_fence(rewrite), bundle, brief)
            struct = lint.structural_issues(body)
            unsupported = lint.unsupported_numeric_claims(body, bundle.model_dump_json())
    if unsupported:
        softened = soften_unsupported_numeric_claims(body, bundle.model_dump_json())
        if softened != body:
            body = _polish_body(softened, bundle, brief)
            struct = lint.structural_issues(body)
            unsupported = lint.unsupported_numeric_claims(body, bundle.model_dump_json())
    post_slug = _slugify(bundle.primary.title)
    body = embed_planned_visuals(body, bundle.visual_plan, post_slug)
    undefined = lint.undefined_acronyms(body, _glossary_terms_from_bundle(bundle))
    miss_vis = lint.missing_planned_visuals(body, bundle.visual_plan)
    if struct:
        LOG.warning("draft: structural lint: %s", struct)
    if unsupported:
        LOG.warning("draft: unsupported numeric claims: %s", unsupported)
    if undefined:
        LOG.warning("draft: still-undefined acronyms after explainer: %s", undefined)
    u = get_llm_usage()
    draft_calls_ok = u["ok"] - int(u0.get("ok", 0) or 0)
    draft_calls_fail = u["fail"] - int(u0.get("fail", 0) or 0)
    rep_draft = {
        "structural": struct,
        "unsupported_numeric_claims": unsupported,
        "undefined_acronyms": undefined,
        "missing_planned_visuals": miss_vis,
        "stage": "draft",
        "stage_llm_ok": draft_calls_ok,
        "stage_llm_fail": draft_calls_fail,
    }
    (_ROOT / "reports" / "draft_lint.json").write_text(
        json.dumps(rep_draft, indent=2),
        encoding="utf-8",
    )
    print(
        f"stage=draft calls={draft_calls_ok}/{draft_calls_ok + max(0, draft_calls_fail)} "
        f"structural={len(struct)}",
        flush=True,
    )
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = post_slug
    title = _sanitize_frontmatter_text(bundle.primary.title, 200)
    takeaway = _frontmatter_takeaway(body, bundle)
    fm = f"""---
date: "{day}"
draft: true
title: "{title}"
description: "{takeaway[:160]}"
categories: ["Machine Learning"]
tags: {json.dumps(bundle.primary.tags[:8] + ["blogpipe"])}
math: true
mermaid: true
one_sentence_takeaway: "{takeaway[:300]}"
image: /img/posts/{slug}/hero.png
rubric_score: 0
---

"""
    full = fm + body.lstrip()
    if config.dry_run():
        out = _ROOT / "reports" / f"draft_preview_{day}-{slug}.md"
    else:
        out = _ROOT / "content" / "post" / f"{day}-{slug}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(full, encoding="utf-8")
    (_ROOT / "reports" / "draft_path.txt").write_text(str(out.relative_to(_ROOT)), encoding="utf-8")
    LOG.info("draft: wrote %s", out)
    return out


def _named_alternative_for_stub(bundle: EvidenceBundle) -> str:
    """Best-effort named alternative method for the stub tradeoff paragraph."""
    for grp in ("competitors", "ancestors", "followups", "enrichment_items"):
        items = getattr(bundle, grp, None) or []
        for it in items[:4]:
            t = (getattr(it, "title", "") or "").strip()
            if t:
                return t.split(":", 1)[0].strip()[:80]
    primary_title = (bundle.primary.title or "").lower()
    haystack = " ".join(
        [
            _section_text(bundle, "paper_experiments", "paper_method"),
            bundle.primary.abstract or "",
        ]
    )
    cap_re = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:[- ][A-Z][A-Za-z0-9]+)+\b")
    for m in cap_re.finditer(haystack):
        cand = m.group(0)
        if cand.lower() in primary_title:
            continue
        return cand
    return "Full Fine-Tuning"


def _stub_body(
    bundle: EvidenceBundle, brief: EditorialBrief, fmt: formats.PostFormat
) -> str:
    """Deterministic placeholder for dry runs / no API key."""
    takeaway = _takeaway_from_bundle(bundle)
    primary_id = bundle.primary.id
    abstract_sents = [
        _clean_sentence(x) for x in _split_sentences(bundle.primary.abstract) if x.strip()
    ]
    problem = next((s for s in abstract_sents if not _has_result_signal(s)), "")
    if not problem:
        problem = _pick_section_sentence(bundle, "paper_problem")
    if not problem and abstract_sents:
        problem = abstract_sents[0]
    method = _pick_section_sentence(bundle, "paper_method")
    if not method:
        method = abstract_sents[1] if len(abstract_sents) > 1 else problem
    setup = _clean_sentence(
        _section_text(
            bundle,
            "paper_setup",
            "paper_reproducibility",
            "paper_experiments",
        )
    )
    if not setup:
        setup = _pick_section_sentence(
            bundle,
            "paper_setup",
            "paper_reproducibility",
            "paper_experiments",
        )
    result = _pick_section_sentence(bundle, "paper_experiments", numeric=True)
    if not result:
        result = next((s for s in abstract_sents if _has_result_signal(s)), method)
    limitation = _pick_section_sentence(bundle, "paper_limitations")
    if not limitation and bundle.contradiction_notes:
        limitation = _clean_sentence(bundle.contradiction_notes[0])
    if not limitation:
        limitation = (
            "The evidence still leaves open how far the result generalizes across model families, "
            "benchmarks, and compute budgets."
        )
    table = _render_results_table(_fallback_benchmark_rows(bundle))
    mermaid = (
        "```mermaid\nflowchart LR\n  A[Hidden-state trajectory] --> B[RDP breakpoints]\n"
        "  B --> C[Selected layers]\n  C --> D[LoRA fine-tune]\n```\n"
        if re.search(
            r"\bRDP\b|layer selection|trajectory|breakpoint|geometry",
            bundle.primary.abstract + "\n" + _section_text(bundle, "paper_method"),
            re.I,
        )
        else "```mermaid\nflowchart LR\n  A[Constraint] --> B[Local opener]\n"
        "  B --> C[Cloud continuation]\n  C --> D[User-visible response]\n```\n"
    )
    setup_line = (
        f"{setup} [cite: {primary_id}] "
        "That setup matters because the headline only means something once the model, benchmark, "
        "and baselines are pinned down."
        if setup
        else "The paper reports its result against named baselines on a concrete evaluation rig, "
        f"which is the minimum context I want before taking the headline seriously. [cite: {primary_id}]"
    )
    result_line = (
        f"{_clean_sentence(result)} [cite: {primary_id}] "
        "The comparison I care about is whether the structured selection rule beats both the full "
        "budget and an unprincipled sparse baseline."
        if result
        else f"{takeaway} [cite: {primary_id}]"
    )
    limit_line = (
        f"{_clean_sentence(limitation)} [cite: {primary_id}] "
        "In my view, that keeps the result in the 'update your priors' bucket rather than the "
        "'copy this blindly' bucket."
    )
    alt_name = _named_alternative_for_stub(bundle)
    tradeoff_line = (
        f"The road not taken here is {alt_name}, which spends the full adaptation budget without "
        "a structural rule. Choosing the structured variant over "
        f"{alt_name} is the actual trade-off: you give up uniform coverage in exchange for a "
        f"layer-selection rule that can be ablated against {alt_name} on the same benchmark. "
        f"[cite: {primary_id}]"
    )
    return (
        f"{takeaway}\n\n"
        "## The operating constraint the paper is actually fighting\n"
        f"{_clean_sentence(problem)} [cite: {primary_id}] "
        "The useful reframing here is that the paper spends its complexity budget on a concrete "
        "engineering bottleneck rather than on a vague promise of general improvement.\n\n"
        "## The mechanism is simpler than the headline suggests\n"
        f"{_clean_sentence(method)} [cite: {primary_id}] "
        "What I find convincing is that this matters because the method tries to allocate a fixed "
        "adaptation budget with a structural rule, not just throw more trainable parameters at the "
        "problem.\n\n"
        f"{mermaid}\n"
        "## What the paper actually measures\n"
        f"{setup_line}\n\n"
        f"{result_line}\n\n"
        f"{table}\n\n"
        "## The road not taken\n"
        f"{tradeoff_line}\n\n"
        "## Why I still treat the win as local\n"
        f"{limit_line}\n\n"
        "## The next experiment worth stealing\n"
        "Before I would copy this into a real workflow, I would rerun the selection rule on my own "
        "model family, task mix, and failure budget. The reusable idea is not the headline number "
        "by itself; it is the habit of making the layer or component choice explainable enough to "
        "test against a strong baseline. "
        f"[cite: {primary_id}]\n"
    )

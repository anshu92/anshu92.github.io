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
from .models import EditorialBrief, EvidenceBundle, Item
from .memory import _ROOT

LOG = logging.getLogger(__name__)


def _slugify(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", title.lower())
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    return s[:80] or "post"


def _load_json(name: str) -> dict:
    p = _ROOT / "reports" / name
    return json.loads(p.read_text()) if p.is_file() else {}


def _resolve_cites(md: str, bundle: EvidenceBundle) -> str:
    bundle.register_ids()

    def repl(m: re.Match[str]) -> str:
        cid = m.group(1).strip()
        it = bundle.by_id.get(cid)
        if not it:
            return f"[missing cite: {cid}]"
        return f"[{it.title[:60]}]({it.url})"

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


def _ensure_mermaid(body: str, title: str, brief: EditorialBrief) -> str:
    if "```mermaid" in body:
        return body
    LOG.info("draft: no mermaid block; injecting diagram")
    if config.dry_run() or not config.llm_configured():
        return (
            body.rstrip()
            + "\n\n```mermaid\nflowchart LR\n  A[Input] --> B[Core method] --> C[Output]\n```\n"
        )
    raw = openrouter_client.llm_text(
        "Output only a mermaid code block (```mermaid ... ```) — no explanation. "
        f"Style: {brief.diagram_style}. Show the key method or process flow.",
        f"Paper: {title}",
        max_tokens=1024,
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
        "Rewrite as ONE sentence takeaway that names the key result with a number. "
        "Output only the sentence, no quotes.",
        f"Title: {title}\nCurrent first line: {line}",
        max_tokens=512,
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
    abstract = bundle.primary.abstract
    pct_hits = re.findall(r"([^()]{3,120}?)\s*\((\d+(?:\.\d+)?)%\)", abstract)
    if pct_hits:
        cleaned = []
        for name, value in pct_hits[:4]:
            method = _clean_fallback_row_name(name)
            cleaned.append((method[:80], f"{value}%", "from abstract"))
        return cleaned[:4]
    size_match = re.search(
        r"(\d+(?:\.\d+)?M\s*-\s*\d+(?:\.\d+)?M)\s+parameters", abstract, re.I
    )
    edge_match = re.search(
        r"(\d+(?:\.\d+)?M\s*-\s*\d+(?:\.\d+)?B)\s+parameter", abstract, re.I
    )
    words_match = re.search(r"first\s+(\d+\s*-\s*\d+)\s+words", abstract, re.I)
    comp_match = re.search(
        r"matching several\s+(\d+(?:\.\d+)?M\s*-\s*\d+(?:\.\d+)?M-class)",
        abstract,
        re.I,
    )
    if size_match:
        rows.append(
            (
                "micro language models",
                f"{size_match.group(1).replace(' ', '')} parameters",
                edge_match.group(1).replace(" ", "") if edge_match else "larger edge models",
            )
        )
    if words_match:
        rows.append(
            (
                "on-device opener",
                f"{words_match.group(1).replace(' ', '')} words",
                "cloud continuation",
            )
        )
    if comp_match:
        rows.append(
            (
                "comparable existing models",
                comp_match.group(1).replace(" ", ""),
                "existing models",
            )
        )
    return rows[:4]


def _render_results_table(rows: list[tuple[str, str, str]]) -> str:
    lines = ["| Method | Metric | Baseline |", "| --- | --- | --- |"]
    for method, metric, baseline in rows[:4]:
        lines.append(f"| {method} | {metric} | {baseline or 'n/a'} |")
    return "\n".join(lines)


def _repair_or_inject_results_table(body: str, bundle: EvidenceBundle) -> str:
    rows = _fallback_benchmark_rows(bundle)
    if not rows:
        return body
    table = _render_results_table(rows)
    evidence_text = bundle.model_dump_json()
    table_block = re.compile(
        r"(?ms)^\s*\|[^\n]*\bMethod\b[^\n]*\|\n^\s*\|(?:[-: ]+\|)+\n(?:^\s*\|.*\|\n?)+"
    )
    match = table_block.search(body)
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
    if rows:
        method, metric, baseline = rows[0]
        sent = f"{method} reaches {metric} against {baseline}."
        if _has_result_signal(sent):
            return _normalize_research_attribution(sent[:220])
    title = bundle.primary.title.split(":")[0][:120]
    return f"{title} is interesting only if the concrete tradeoff survives contact with deployment."


def _sanitize_frontmatter_text(text: str, limit: int) -> str:
    return _plain_text(text).replace('"', "'")[:limit]


def _polish_body(body: str, bundle: EvidenceBundle, brief: EditorialBrief) -> str:
    body = _unwrap_markdown_fence(body)
    body = _cleanup_missing_cites(body)
    body = _normalize_research_attribution(body)
    body = _normalize_takeaway_line(body)
    body = _ensure_grounded_takeaway(body, bundle)
    body = _dedupe_adjacent_lines(body)
    body = _strip_leading_title_echo_heading(body, bundle.primary.title)
    body = _repair_or_inject_results_table(body, bundle)
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
            "BENCHMARK rows from evidence — use ONLY these numbers in the table and prose; "
            "do not invent other metrics:\n" + "\n".join(b_rows)
        )
    for lab, items in (
        ("ANCESTORS", bundle.ancestors),
        ("COMPETITORS", bundle.competitors),
        ("FOLLOWUPS", bundle.followups),
    ):
        for it in items[:4]:
            ev.append(f"{lab} {it.title} | {it.url}\n{it.abstract[:260]}")
    if bundle.quotes:
        ev.append(
            "PAPER_EXCERPTS (use for grounded paraphrase; do not overquote):\n"
            + "\n".join(
                f"- {q.text[:260]} [{q.source_id}]"
                for q in bundle.quotes[:6]
                if (q.text or "").strip()
            )
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
    for it in bundle.enrichment_items[:5]:
        ev.append(
            f"ENRICHMENT {it.source} {it.title} | {it.url}\n{it.abstract[:260]}"
        )
    ctx = "\n\n".join(ev)
    if len(ctx) > 9000:
        ctx = ctx[:9000] + "\n\n[EVIDENCE truncated to budget]\n"
    brief_json = brief.model_dump_json()
    system = (
        "You write Hugo markdown blog posts for senior engineers and researchers. "
        "You are a technical blogger dissecting someone else's paper after reading it closely. "
        "Output ONLY the markdown body (no outer frontmatter — the tool will add it). "
        "Do not emit YAML, a title line, or any heading that simply repeats the paper title. "
        "A great technical blog does three things at once: it teaches something real, earns trust, "
        "and respects the reader's time.\n"
        "\nVOICE AND ATTRIBUTION:\n"
        "- This is not written from the paper authors' point of view.\n"
        "- For paper claims, use 'the authors', 'the paper', or the method name. "
        "Do NOT write 'we' or 'our' for experiments, models, baselines, or results unless "
        "EVIDENCE explicitly says we ran our own reproduction.\n"
        "- Use first-person singular only for your own judgment: 'I buy this because...', "
        "'I would test...', 'What I find convincing is...'.\n"
        "- High-signal ML technical blogs usually open with a crisp thesis or operational pain point, "
        "then explain why the experiment matters, walk through the mechanism, name the evaluation "
        "setup, and close with lessons, caveats, and reproducibility status. Follow that standard "
        "without turning it into a rigid template.\n"
        "\nTHE 10 PRINCIPLES — follow all:\n"
        "1. Start with a real problem the reader cares about. Not history. Not abstract theory.\n"
        "2. Be concrete. Replace vague wording with specific systems, numbers, constraints, and failure "
        "modes. 'We reduced p95 from 420 ms to 180 ms by batching requests and eliminating redundant "
        "serialization' beats 'we improved performance'.\n"
        "3. Clarity over cleverness. Flow should feel inevitable: context -> problem -> approach -> "
        "implementation -> results -> limits. Do not hide the problem.\n"
        "4. Teach through examples. Show before/after, inputs/outputs, edge cases, code that earns its "
        "place, diagrams that reduce cognitive load.\n"
        "5. Make tradeoffs explicit. Name the alternatives considered and what was given up "
        "(latency, cost, simplicity, accuracy, maintainability, data availability, team maturity).\n"
        "6. Show failures and limits. Name where the method breaks, the assumptions required, and when "
        "NOT to use it. Credibility lives here. Do NOT present only the success path.\n"
        "7. Back every claim (faster / cheaper / safer / more accurate / more robust) with a number, "
        "benchmark, cited source, or explicit comparison against a named baseline. No unsupported "
        "confidence.\n"
        "7b. Do not smuggle in uncited ML folklore. If you mention layer roles, transfer behavior, "
        "or architecture intuitions, they must be stated in EVIDENCE or omitted.\n"
        "8. Tight narrative. Each section answers one question that leads to the next. Kill "
        "throat-clearing, repeated caveats, and long background sections that delay the payoff.\n"
        "9. Write for the reader's next action: a technique to try, a pitfall to avoid, a pattern to "
        "steal, or a different way to think about the system.\n"
        "10. Have a point of view. Argue why a design worked, why a common assumption is wrong, why a "
        "tradeoff is underrated, or why a result will not generalise.\n"
        "\nTHE FIVE-QUESTION TEST — a smart reader must be able to answer these after one skim:\n"
        "(a) What problem is being solved?\n"
        "(b) Why is it hard?\n"
        "(c) What was tried?\n"
        "(d) What worked and what did NOT?\n"
        "(e) What should I do with this?\n"
        "\nSTRUCTURE — no rigid template:\n"
        "- Invent your own ## H2 headings that fit THIS story. Headings must be descriptive and "
        "specific: '30% lower p95 at the cost of 2x memory', 'Where batched KV cache breaks'. "
        "Generic headings are banned: Introduction, Background, Overview, Results, Summary, Conclusion.\n"
        "- Use as many sections as the story needs. No fixed count, no fixed order. Each section "
        "should earn its place by answering one clear question.\n"
        "- Do NOT emit the CONTENT GOALS text as literal headings. They are objectives to cover, "
        "not titles.\n"
        "- First line of the body: a one-sentence takeaway with a concrete number (no heading on "
        "that line).\n"
        "- Never repeat the takeaway or the paper title as an immediate ## heading.\n"
        "- Short paragraphs. Descriptive sub-heads. Highlight key takeaways with a leading bold "
        "phrase or a one-sentence summary at the end of a section so the post is skimmable.\n"
        "\nHARD REQUIREMENTS:\n"
        "- Every external claim uses [cite: ID] where ID is 'primary' or one of these exact ids: "
        + ", ".join(bundle.by_id.keys())
        + ".\n"
        "- Exactly one ```mermaid``` code block that clarifies a mechanism, data flow, or comparison. "
        "A diagram must reduce cognitive load; if it needs a paragraph to decode, rewrite it.\n"
        "- One markdown table with | Method | Metric | Baseline | where numbers are compared. "
        "Numbers in the table MUST match the prose. If BENCHMARK rows are given in EVIDENCE, use "
        "ONLY those values — do not invent metrics. If the evidence only gives model-size or "
        "latency ranges rather than benchmark percentages, build the table from those grounded "
        "numbers instead of fabricating scores.\n"
        "- If ENRICHMENT or SECTION_EVIDENCE blocks are present, use those URLs/snippets for external "
        "claims; do not invent other sources.\n"
        "- Read the whole paper evidence, not just the abstract. If PAPER_EXCERPTS or paper_* "
        "sections are present, use them to ground the mechanism, setup, results, and limitations.\n"
        "- Do not paraphrase the abstract sentence by sentence. Synthesize the paper's argument "
        "across setup, comparison, and failure modes.\n"
        "- The topic is always the PRIMARY paper. Competitors, enrichment items, and related posts "
        "are supporting context only. Never switch topics and never ask the reader for clarification.\n"
        "- If the setup or reproduction details are incomplete in EVIDENCE, say they are incomplete. "
        "Do not fill in missing hardware, code availability, dataset splits, or calibration steps.\n"
        "- Never invent people, internal teams, or anecdotes not present in EVIDENCE.\n"
        "- Name the evaluation setup when the paper gives it: model, dataset, benchmark, hardware, "
        "or configuration. A strong technical blog tells the reader what was actually measured.\n"
        "- State explicit tool / model / library versions wherever relevant (e.g. 'PyTorch 2.4', "
        "'Transformers 4.40') so the post does not rot silently.\n"
        "- At least one sentence of honest limitation or failure mode: a reproduction barrier "
        "(hardware, missing code, compute), a broken assumption (model family, task type, scale), "
        "or a failure the authors themselves mention. Ablations that validate the method do NOT "
        "count as 'what did not work'.\n"
        "- At least one first-person sentence of opinion ('What I find is...', 'In my view...', "
        "'I think...', 'The remarkable thing here is...'). Follow the phrase with a real opinion, "
        "not a hedge. First-person opinion may interpret significance, but it must not add new "
        "transfer hypotheses or mechanism claims that are absent from EVIDENCE.\n"
        "- Be honest about scope: if a result holds at one scale, one dataset, one team, say so. "
        "Do not oversell a local finding as a general law.\n"
        "- Close with a concrete next action the reader can take, but keep it grounded in EVIDENCE. "
        "Do not introduce new tools, plots, datasets, prompt counts, thresholds, or procedures that "
        "the paper does not mention. Prefer validation advice already implied by the paper, such as "
        "replicating the reported setup or adding multi-seed evaluation.\n"
        "- If there is a concrete construction / BIM / design-tool angle, name it in one sentence. "
        "Otherwise omit — do NOT pad with generic 'implications for NLP / various applications'.\n"
        "\nRED FLAGS — avoid these; they mark weak posts:\n"
        "- Opening with a history lesson or 'In recent years...'.\n"
        "- Marketing vocabulary: 'leveraged', 'seamless', 'powerful', 'game-changing', 'unlock', "
        "'next-generation', 'cutting-edge', 'revolutionary', 'holistic', 'synergy'.\n"
        "- Hiding the actual problem past the midpoint.\n"
        "- Code with no explanation, or code pasted 'to look technical'.\n"
        "- Only the success path — no limitations.\n"
        "- Numbers in the prose that contradict the table.\n"
        "- Writing as if we conducted the paper's experiments.\n"
        "- Meta headings like 'Author Takeaway', 'Changed Minds', 'Next Steps', 'Performance "
        "Comparison', or 'Empirical Results'. Rename them into story-specific headings or fold the "
        "content into adjacent sections.\n"
        "- Slang, snark, or casual put-downs such as 'slap adapters on every layer' or 'flops hard'.\n"
        "- Impressive-sounding prose that teaches nothing reusable.\n"
        "- Reading like an internal status update dressed up as an article.\n"
        "\nOpener style: "
        + brief.opener_hook
        + ". Voice: "
        + brief.voice_guide
        + " "
        + fmt.voice_override
    )
    user = (
        f"CONTENT GOALS (cover across the post, in your own words and your own headings):\n{goals}\n\n"
        f"EDITORIAL BRIEF:\n{brief_json}\n\n"
        f"EVIDENCE:\n{ctx}\n\n"
        f"RELATED POSTS (link with [title](/post/slug/) path only, no ref shortcode in output): "
        f"{brief.recent_topics[:5]}\n"
    )
    return system, user


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
            system, user, max_tokens=config.max_tokens_smart()
        )
        if not body.strip():
            body = _stub_body(bundle, brief, fmt)
        elif not _looks_like_markdown_body(body):
            LOG.warning("draft: initial generation not usable; requesting grounded rewrite")
            rewrite = openrouter_client.llm_text(
                "Rewrite the draft into a final markdown blog post about the PRIMARY paper only. "
                "Do not ask questions, do not request clarification, do not mention topic confusion, "
                "and do not switch to related posts or supporting context. Output only the final "
                "markdown body with story-specific ## headings, one mermaid block, and grounded [cite: id] claims.",
                f"PRIMARY_TITLE: {bundle.primary.title}\n\nBAD_DRAFT:\n{body[:12000]}\n\nEVIDENCE:\n{user[:14000]}",
                mode="smart",
                max_tokens=config.max_tokens_smart(),
            )
            if rewrite.strip() and _looks_like_markdown_body(rewrite):
                body = _unwrap_markdown_fence(rewrite)
            else:
                body = _stub_body(bundle, brief, fmt)
    body = _resolve_cites(_unwrap_markdown_fence(body), bundle)
    if _strip_unresolved(body):
        LOG.warning("draft: unresolved cites; attempting one repair")
        repair = openrouter_client.llm_text(
            "Fix [missing cite: ...] markers by replacing them with paraphrase without external cite, "
            "or remove the affected sentence. Output only the full revised markdown body, no commentary "
            "and no code fences.",
            body[:12000],
            max_tokens=2048,
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
            "Revise the markdown body to fix the listed issues while keeping the topic fixed to the PRIMARY paper. "
            "Do not ask questions. Do not switch topics. Keep all numbers grounded in EVIDENCE. "
            "Return the full markdown body only, with one mermaid block and story-specific headings.",
            f"ISSUES:\n- " + "\n- ".join(struct + unsupported[:8]) + f"\n\nBODY:\n{body[:16000]}\n\nEVIDENCE:\n{user[:14000]}",
            mode="smart",
            max_tokens=config.max_tokens_smart(),
        )
        if rewrite.strip() and _looks_like_markdown_body(rewrite):
            body = _polish_body(_unwrap_markdown_fence(rewrite), bundle, brief)
            struct = lint.structural_issues(body)
            unsupported = lint.unsupported_numeric_claims(body, bundle.model_dump_json())
    if struct:
        LOG.warning("draft: structural lint: %s", struct)
    if unsupported:
        LOG.warning("draft: unsupported numeric claims: %s", unsupported)
    u = get_llm_usage()
    draft_calls_ok = u["ok"] - int(u0.get("ok", 0) or 0)
    draft_calls_fail = u["fail"] - int(u0.get("fail", 0) or 0)
    rep_draft = {
        "structural": struct,
        "unsupported_numeric_claims": unsupported,
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
    slug = _slugify(bundle.primary.title)
    title = _sanitize_frontmatter_text(bundle.primary.title, 200)
    takeaway = _sanitize_frontmatter_text(body.split("\n", 1)[0].strip(), 500)
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
        "## Why I still treat the win as local\n"
        f"{limit_line}\n\n"
        "## The next experiment worth stealing\n"
        "Before I would copy this into a real workflow, I would rerun the selection rule on my own "
        "model family, task mix, and failure budget. The reusable idea is not the headline number "
        "by itself; it is the habit of making the layer or component choice explainable enough to "
        "test against a strong baseline. "
        f"[cite: {primary_id}]\n"
    )

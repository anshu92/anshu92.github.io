"""Structural and red-flag lints for drafted markdown.

These lints intentionally do NOT require specific section titles or a fixed
outline. The only structural expectations are: a numeric takeaway line, a
mermaid diagram, a results table whose numbers match the prose, and the absence
of empty placeholder sections. Everything else is content/quality.
"""

from __future__ import annotations

import re

_BANNED = re.compile(
    r"\b(seamless(ly)?|powerful|revolutionary|leverage[sd]?|"
    r"game[- ]chang(er|ing)|unlocks?|next[- ]generation|cutting[- ]edge|"
    r"empower(s|ed|ing)?|synerg(y|ies|istic)|holistic)\b",
    re.I,
)
_ROBUST_UNQUAL = re.compile(r"\brobust\b(?![^\n]{0,80}\d)", re.I)
_SOTA_UNQUAL = re.compile(
    r"\bstate[- ]of[- ]the[- ]art\b(?![^\n]{0,120}(table|benchmark|BLEU|score))", re.I
)
_THROAT = re.compile(
    r"^(In this (blog|post|article)|In recent years|With the rise of|As we all know|"
    r"It goes without saying|In today's world)\b",
    re.I | re.M,
)
_GENERIC_H2 = re.compile(
    r"^##\s*(Introduction|Background|Overview|Results|Conclusion|Summary)\s*$",
    re.I | re.M,
)
_TEMPLATED_H2 = re.compile(
    r"^##\s*(Changed Minds|Author Takeaway|Next Steps|Performance Comparison|"
    r"Empirical Results|How (?:.+ )?Works|Why It Works|The Problem(?:\s*:.*)?|"
    r"Limitations and Failure Modes)\s*$",
    re.I | re.M,
)
_CODE_FENCE = re.compile(r"```[\s\S]*?```", re.M)
_DIGITS = re.compile(r"\d")
_NUMERIC_CLAIM = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*(?:%|x|×|ms|s|gb|mb|kb|"
    r"tokens?|words?|layers?|params?|parameters?|hours?|days?|"
    r"[kmb](?:\s*-\s*\d+(?:\.\d+)?[kmb])?(?:-class)?)(?=\b|[^a-z0-9])",
    re.I,
)


def count_digits(s: str) -> int:
    return len(_DIGITS.findall(s))


def has_numeric_claim(s: str) -> bool:
    """True if `s` contains at least one number with a unit / percent / size suffix."""
    return bool(_NUMERIC_CLAIM.search(s or ""))


def lint_forbidden_phrases(body: str) -> list[str]:
    issues: list[str] = []
    if _BANNED.search(body):
        issues.append("banned_marketing_phrase")
    for _ in _ROBUST_UNQUAL.finditer(body):
        issues.append("robust_without_metric")
        break
    for _ in _SOTA_UNQUAL.finditer(body):
        issues.append("sota_without_benchmark")
        break
    first_para = body.split("\n\n", 1)[0][:800]
    if _THROAT.search(first_para):
        issues.append("throat_clearing_opener")
    return issues


def lint_generic_headings(body: str) -> list[str]:
    """Flag generic one-word-ish headings. Specific, story-named H2s are required."""
    bad: list[str] = []
    for m in _GENERIC_H2.finditer(body):
        line = m.group(0).strip()
        if ":" not in line:
            bad.append(line)
    return ["generic_heading_used"] if bad else []


def lint_templated_headings(body: str) -> list[str]:
    """Catch blog-factory headings that usually signal weak, generic structure."""
    return ["templated_heading_used"] if _TEMPLATED_H2.search(body or "") else []


def _normalize_heading(text: str) -> str:
    t = re.sub(r"[`*_#>\[\]\(\):]", "", text or "")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def lint_duplicate_headings(body: str) -> list[str]:
    seen: set[str] = set()
    dup = False
    for m in re.finditer(r"^##\s+(.+?)\s*$", body or "", re.M):
        key = _normalize_heading(m.group(1))
        if not key:
            continue
        if key in seen:
            dup = True
            break
        seen.add(key)
    return ["duplicate_heading"] if dup else []


def lint_takeaway_repeated_as_heading(body: str) -> list[str]:
    first = (body or "").split("\n", 1)[0].strip()
    if not first:
        return []
    key = _normalize_heading(first)
    if not key:
        return []
    for m in re.finditer(r"^##\s+(.+?)\s*$", body or "", re.M):
        if m.start() == 0:
            continue
        if _normalize_heading(m.group(1)) == key:
            return ["takeaway_repeated_as_heading"]
    return []


def lint_structure(body: str) -> list[str]:
    """Format-independent structural checks. Does not care about section titles."""
    issues: list[str] = []
    if "```mermaid" not in body:
        issues.append("no_mermaid_block")
    first = body.split("\n", 1)[0].strip() if body.strip() else ""
    if first.startswith("#"):
        issues.append("takeaway_is_heading")
    if first and not re.search(r"\d", first):
        issues.append("takeaway_lacks_number")
    table_match = re.search(r"^\s*\|[^\n]*\bMethod\b[^\n]*\|", body, re.M)
    if not table_match:
        issues.append("no_results_table")
    if table_match:
        prose = body[: table_match.start()]
        table_part = body[table_match.start() :]
    else:
        prose = body
        table_part = ""
    table_nums = set(re.findall(r"[\d.]+\s*%", table_part))
    prose_nums = set(re.findall(r"[\d.]+\s*%", prose))
    if table_nums and prose_nums and not table_nums.intersection(prose_nums):
        issues.append("table_numbers_diverge_from_prose")
    return issues


_H2_LINE = re.compile(r"^##\s+\S", re.M)
_MERMAID_OPENER = re.compile(
    r"^(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|"
    r"gantt|pie|journey|gitGraph|mindmap|timeline|requirementDiagram|c4Context)\b",
    re.M,
)
_CITE_TOKEN = re.compile(r"\[cite:\s*[^\]]+\]", re.I)
_UNRESOLVED_SOURCE_ALIAS = re.compile(r"\[(?:hf|arxiv|pwc)_[^\]]+\]")
_COMPARATIVE_CLAIM = re.compile(
    r"\b(outperform(?:s|ed|ing)?|beat(?:s|en)?|improv(?:e|es|ed|ement)|"
    r"reduce(?:s|d)?|lower(?:s|ed)?|higher|better|worse)\b",
    re.I,
)
_DECISION_HINT = re.compile(
    r"\b(when to use|when not to use|should you use|decision|adopt|deploy|production|steal this|next steps|engineering habit|next experiment|before i would copy)\b",
    re.I,
)
_MECHANISM_HINT = re.compile(
    r"\b(why this works|how it works|mechanism|causal|decision signal|trajectory|conditioning)\b",
    re.I,
)
_ADVICE_HINT = re.compile(
    r"^##\s+(.+)$|(?:^|\n)\s*(?:I'd|I would|Try|Use|Start with|Steal This|Engineering habit)",
    re.I | re.M,
)


def lint_h2_minimum(body: str, *, minimum: int = 2) -> list[str]:
    """Body must contain at least ``minimum`` H2 sections (writer prompt mandates ##)."""
    n = len(_H2_LINE.findall(body or ""))
    return ["fewer_than_required_h2_sections"] if n < minimum else []


def _outside_fence_spans(body: str) -> list[tuple[int, int]]:
    """Index ranges that are NOT inside a fenced code block."""
    out: list[tuple[int, int]] = []
    cur = 0
    for m in _CODE_FENCE.finditer(body or ""):
        out.append((cur, m.start()))
        cur = m.end()
    out.append((cur, len(body or "")))
    return out


def lint_unfenced_mermaid(body: str) -> list[str]:
    """Find bare mermaid diagram openers that are not inside a fenced code block."""
    text = body or ""
    spans = _outside_fence_spans(text)
    for start, end in spans:
        chunk = text[start:end]
        if _MERMAID_OPENER.search(chunk):
            return ["unfenced_mermaid_block"]
    return []


def lint_citation_minimum(body: str, *, minimum: int = 2) -> list[str]:
    """Drafts should carry at least ``minimum`` ``[cite: id]`` markers (post-resolve, the markers stay as links)."""
    text = body or ""
    n = len(_CITE_TOKEN.findall(text))
    if n >= minimum:
        return []
    # After _resolve_cites the [cite: id] markers become inline markdown links — count those too.
    md_links = len(re.findall(r"\]\(https?://[^)]+\)", text))
    if md_links >= minimum:
        return []
    return ["citation_count_below_min"]


def lint_unresolved_source_aliases(body: str) -> list[str]:
    return ["unresolved_source_alias"] if _UNRESOLVED_SOURCE_ALIAS.search(body or "") else []


def lint_comparative_claims(body: str) -> list[str]:
    issues: list[str] = []
    for para in re.split(r"\n\s*\n", body or ""):
        if not _COMPARATIVE_CLAIM.search(para):
            continue
        if has_numeric_claim(para):
            continue
        if re.search(r"\b(than|over|versus|vs\.?|baseline|random|dense|full)\b", para, re.I):
            continue
        if re.search(r"\b(fid|psnr|ssim|accuracy|latency|benchmark|mmlu|bleu|loss)\b", para, re.I):
            continue
        issues.append("comparative_claim_missing_metric")
        break
    return issues


def lint_required_sections(body: str) -> list[str]:
    text = body or ""
    issues: list[str] = []
    if not _MECHANISM_HINT.search(text):
        issues.append("missing_mechanism_section")
    if not _DECISION_HINT.search(text):
        issues.append("missing_decision_section")
    return issues


def lint_advice_traceability(body: str) -> list[str]:
    text = body or ""
    if not _ADVICE_HINT.search(text):
        return []
    if re.search(r"\b(In my view|I think|I would|I buy this|author synthesis)\b", text, re.I):
        return []
    if _CITE_TOKEN.search(text) or re.search(r"\]\(https?://[^)]+\)", text):
        return []
    return ["advice_without_traceability"]


# --- Section redundancy ----------------------------------------------------

_REDUNDANCY_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "these", "those",
        "their", "they", "them", "have", "has", "had", "are", "was", "were",
        "been", "but", "not", "into", "such", "also", "more", "most", "than",
        "then", "there", "here", "when", "where", "which", "while", "what",
        "who", "whom", "all", "any", "some", "one", "two", "can", "will",
        "would", "should", "could", "may", "might", "must", "about", "over",
        "between", "across", "within", "many", "much", "very", "just", "only",
        "even", "ever", "still", "yet", "so", "as", "to", "of", "in", "on",
        "by", "or", "an", "a", "is", "be", "it", "its", "his", "her", "our",
        "we", "you", "i", "do", "does", "did", "no", "nor", "if", "because",
    }
)


def _bigram_set(text: str) -> set[tuple[str, str]]:
    """Adjacent lowercase non-stopword bigrams in `text`, fence-stripped."""
    src = _strip_code_fences(text or "")
    src = re.sub(r"\[cite:[^\]]+\]", " ", src)
    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z\-]+", src)
        if w.lower() not in _REDUNDANCY_STOPWORDS and len(w) > 2
    ]
    return {(words[i], words[i + 1]) for i in range(len(words) - 1)}


def _h2_section_bodies(body: str) -> list[tuple[str, str]]:
    """Return [(title, body_text), ...] split on `## ` boundaries."""
    text = body or ""
    out: list[tuple[str, str]] = []
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, re.M))
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((title, text[start:end].strip()))
    return out


def lint_section_redundancy(body: str, *, threshold: float = 0.55) -> list[str]:
    """Adjacent H2 sections whose noun-bigram Jaccard >= threshold are flagged."""
    sections = _h2_section_bodies(body)
    if len(sections) < 2:
        return []
    sets = [_bigram_set(b) for _, b in sections]
    for i in range(len(sets) - 1):
        a, b = sets[i], sets[i + 1]
        if not a or not b:
            continue
        inter = len(a & b)
        union = len(a | b)
        if union >= 8 and (inter / union) >= threshold:
            return ["redundant_adjacent_sections"]
    return []


# --- Fake / prose-disguised results table ----------------------------------

_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$", re.M)


def _markdown_tables(body: str) -> list[str]:
    """Return raw markdown table blocks (header + rows joined)."""
    text = body or ""
    if not text:
        return []
    row_re = re.compile(r"^\s*\|.+\|\s*$")
    blocks: list[list[str]] = []
    cur: list[str] = []
    for line in text.splitlines():
        if row_re.match(line):
            cur.append(line.strip())
            continue
        if len(cur) >= 2:
            blocks.append(cur)
        cur = []
    if len(cur) >= 2:
        blocks.append(cur)
    return ["\n".join(b) for b in blocks]


def _table_has_numeric_rows(table_md: str, *, min_numeric_cells: int = 1) -> bool:
    rows = table_md.splitlines()
    for row in rows[2:]:  # skip header + separator
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        nums = sum(1 for c in cells if re.search(r"\d", c))
        if nums >= min_numeric_cells:
            return True
    return False


def lint_fake_results_table(body: str) -> list[str]:
    """Body looks like it tries to claim a results table but no real numeric row exists."""
    text = body or ""
    has_method_metric = bool(
        re.search(
            r"\bmethod\b[\s\S]{0,200}?\bmetric\b[\s\S]{0,200}?\bbaseline\b",
            text,
            re.I,
        )
    )
    if not has_method_metric:
        return []
    for tbl in _markdown_tables(text):
        if (
            re.search(r"\bmethod\b", tbl, re.I)
            and re.search(r"\bmetric|baseline\b", tbl, re.I)
            and _table_has_numeric_rows(tbl)
        ):
            return []
    return ["fake_results_table"]


# --- Mermaid diagram is a taxonomy, not a method flow ----------------------

_MERMAID_FENCE = re.compile(r"```mermaid\s*\n([\s\S]*?)\n```", re.M)
_EDGE_LINE = re.compile(
    r"""
    ^\s*
    (?P<src>[A-Za-z0-9_]+)
    (?:\s*\[[^\]]*\])?           # optional [label] on src
    \s*-->                        # edge
    (?:\s*\|(?P<lbl>[^|]+)\|)?   # optional |edge label|
    \s*
    (?P<dst>[A-Za-z0-9_]+)
    """,
    re.M | re.X,
)
_MEASURABLE_VERB = re.compile(
    r"\b(reduce|cut|drop|shrink|lower|raise|increase|boost|grow|"
    r"replace|swap|prune|select|train|fine[- ]tune|encode|decode|"
    r"compute|measure|score|sample|batch|cache|quantize|distil|"
    r"compress|"
    r"\d+\s*(%|x|ms|s|gb|mb|tokens|layers|params))",
    re.I,
)


def lint_mermaid_taxonomy(body: str) -> list[str]:
    """Inside ```mermaid``` blocks, flag a single-sink diagram with no measurable edge labels."""
    for m in _MERMAID_FENCE.finditer(body or ""):
        diagram = m.group(1)
        edges = list(_EDGE_LINE.finditer(diagram))
        if len(edges) < 5:
            continue
        in_deg: dict[str, int] = {}
        labels: list[str] = []
        for e in edges:
            dst = (e.group("dst") or "").strip()
            in_deg[dst] = in_deg.get(dst, 0) + 1
            lbl = (e.group("lbl") or "").strip()
            if lbl:
                labels.append(lbl)
        if not in_deg:
            continue
        max_in = max(in_deg.values())
        if max_in < 5:
            continue
        if any(_MEASURABLE_VERB.search(lbl) for lbl in labels):
            continue
        return ["mermaid_is_taxonomy"]
    return []


# --- Tradeoffs (named alternatives + contrastive markers) ------------------

_CONTRAST_MARKERS = re.compile(
    r"\b("
    r"vs\.?|versus|"
    r"compared (?:to|with)|in contrast (?:to|with)|"
    r"as opposed to|"
    r"instead of|rather than|in place of|"
    r"in exchange for|at the cost of|in return for|"
    r"alternative(?:ly|s)?|the alternative is|"
    r"trade[- ]?off|"
    r"unlike "
    r")\b",
    re.I,
)


def _expected_alternative_names(bundle) -> list[str]:
    """Names the writer should be using as 'the road not taken'."""
    out: list[str] = []
    if bundle is None:
        return out
    for grp_name in ("competitors", "ancestors", "followups", "enrichment_items"):
        for it in (getattr(bundle, grp_name, None) or [])[:6]:
            t = (getattr(it, "title", "") or "").strip()
            if t:
                out.append(t)
    for b in (getattr(bundle, "benchmarks", None) or [])[:8]:
        nm = (getattr(b, "name", "") or "").strip()
        if nm:
            out.append(nm)
    seen: set[str] = set()
    deduped: list[str] = []
    for n in out:
        k = re.sub(r"[^a-z0-9]+", "", n.lower())
        if not k or k in seen:
            continue
        seen.add(k)
        deduped.append(n)
    return deduped


def _body_has_named_alternative(body: str, candidates: list[str]) -> bool:
    if not candidates:
        return False
    body_lower = (body or "").lower()
    for n in candidates:
        head = n.lower().split(":", 1)[0].strip()
        head = re.sub(r"\s+", " ", head)
        if not head:
            continue
        if head[:60] in body_lower:
            return True
    return False


def lint_tradeoffs_present(body: str, bundle=None) -> list[str]:
    """Body must contain a paragraph with both a contrast marker AND a named alternative.

    Bundle is optional; without it we accept any paragraph that has both a contrast
    marker AND something that looks like a proper-noun alternative (capitalised
    multi-word phrase or known method-style token).
    """
    text = body or ""
    if not text.strip():
        return []
    paragraphs = re.split(r"\n{2,}", text)
    candidates = _expected_alternative_names(bundle)
    for p in paragraphs:
        # A "## Heading\nbody..." block ends up in one paragraph; only score the body.
        body_only = re.sub(r"^\s*##\s+[^\n]+\n?", "", p)
        if not _CONTRAST_MARKERS.search(body_only):
            continue
        if candidates:
            if _body_has_named_alternative(body_only, candidates):
                return []
        else:
            # Fallback: at least one capitalised multi-word noun (e.g. "Full Fine-Tuning").
            # Require space (not arbitrary whitespace) so we don't accept heading\nWord.
            if re.search(
                r"\b[A-Z][A-Za-z0-9]+(?:[- ][A-Z][A-Za-z0-9]+)+\b", body_only
            ):
                return []
    return ["no_tradeoffs_paragraph"]


# --- Improvement claims must sit next to a named baseline -------------------

_IMPROVEMENT = re.compile(
    r"\b(?:improve|reduc|cut|drop|lower|raise|boost|increase|outperform|"
    r"beat|beats|surpass|achiev|reach|gain)\w*\s*[^.\n]{0,80}?"
    r"\b\d+(?:\.\d+)?\s*(?:%|x|×|ms|s|gb|mb|tokens?|layers?|points?)?",
    re.I,
)
_BASELINE_NEAR = re.compile(
    r"\b(baseline[s]?|sota|state[- ]of[- ]the[- ]art|"
    r"vs\.?|versus|compared (?:to|with)|in contrast (?:to|with)|"
    r"over (?:the )?(?:prior|previous|baseline|sota)|"
    r"against (?:the )?(?:prior|previous|baseline|sota))\b",
    re.I,
)


def lint_baseline_pairing(
    body: str, bundle=None, *, min_paired_ratio: float = 0.5
) -> list[str]:
    """Numeric improvement claims must sit next to a named baseline or named alternative."""
    text = body or ""
    if not text.strip():
        return []
    matches = list(_IMPROVEMENT.finditer(text))
    if not matches:
        return []
    candidates = _expected_alternative_names(bundle)
    paired = 0
    for m in matches:
        s = max(0, m.start() - 200)
        e = min(len(text), m.end() + 200)
        window = text[s:e]
        if _BASELINE_NEAR.search(window):
            paired += 1
            continue
        if candidates and _body_has_named_alternative(window, candidates):
            paired += 1
            continue
    ratio = paired / float(len(matches))
    if ratio < min_paired_ratio:
        return ["unpaired_improvement_claims"]
    return []


_H2_BLOCK = re.compile(
    r"^##\s*[^\n]+\n([\s\S]*?)(?=^##\s|\Z)", re.M
)
_PLACEHOLDER_SEC = re.compile(
    r"^(no [a-z]+ (information|reference|equation|code|content)\b"
    r"|prose placeholder\b"
    r"|(tbd|tba|todo)\b)",
    re.I | re.M,
)


def lint_empty_placeholders(body: str) -> list[str]:
    """Flag sections whose first line is an explicit empty/placeholder marker."""
    issues: list[str] = []
    for m in _H2_BLOCK.finditer(body or ""):
        block = (m.group(1) or "").strip()
        first = block.split("\n", 1)[0].strip() if block else ""
        if _PLACEHOLDER_SEC.search(first):
            issues.append("empty_placeholder_section")
            break
    return issues


_POV_INTRO = re.compile(
    r"\b(what I find|what i find|the remarkable thing here is)\b",
    re.I,
)
_FIRST_PERSON = re.compile(
    r"\b(I'|I\s|I’m|I'm|I think|I find|my view|in my view)\b", re.I
)
_COLLECTIVE_RESEARCH = re.compile(
    r"\bwe\s+(introduce|present|propose|show|find|observe|use|model|evaluate|"
    r"analyze|study|train|fine[- ]tune|adapt|apply|achieve|compare|extend|demonstrate|"
    r"report|focus|design|identify|derive|select|benchmark|measure|test)\b"
    r"|\bour\s+(method|methods|approach|approaches|model|models|results|experiments|study|"
    r"paper|framework|analysis|system|systems|technique|techniques|baseline|baselines|"
    r"llms?|μlms?)\b",
    re.I,
)


def lint_pov_after_phrase(body: str) -> list[str]:
    """If a POV lead-in phrase appears, it must be followed by an actual opinion."""
    m = _POV_INTRO.search(body or "")
    if not m:
        return []
    after = (body or "")[m.end() : m.end() + 400]
    if not _FIRST_PERSON.search(after) and not re.search(
        r"\b(clear|striking|tells|matters|worth|surprising|valuable|weak|strong|concern|like|dislike)\b",
        after,
        re.I,
    ):
        return ["pov_phrase_without_opinion"]
    return []


def lint_collective_research_voice(body: str) -> list[str]:
    """Paper summaries should not claim the authors' work in first-person plural."""
    return ["collective_research_voice"] if _COLLECTIVE_RESEARCH.search(body or "") else []


def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE.sub("", text or "")


def _normalize_claim(text: str) -> str:
    t = (text or "").lower().replace("×", "x")
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\bparameters?\b", "param", t)
    t = re.sub(r"\bparams?\b", "param", t)
    t = re.sub(r"\bwords?\b", "word", t)
    t = re.sub(r"\blayers?\b", "layer", t)
    t = re.sub(r"\btokens?\b", "token", t)
    t = re.sub(r"\bhours?\b", "hour", t)
    t = re.sub(r"\bdays?\b", "day", t)
    t = re.sub(r"\s+", "", t)
    return t


def numeric_claims(text: str) -> list[str]:
    """Extract normalized numeric phrases that should be grounded in the evidence."""
    src = _strip_code_fences(text)
    found: dict[str, str] = {}
    for m in _NUMERIC_CLAIM.finditer(src):
        raw = m.group(0).strip()
        key = _normalize_claim(raw)
        if key and key not in found:
            found[key] = raw
    return list(found.values())


def unsupported_numeric_claims(body: str, evidence_text: str) -> list[str]:
    """Claims with numbers/units that do not appear in the evidence text."""
    evidence_found = {_normalize_claim(x) for x in numeric_claims(evidence_text)}
    unsupported: list[str] = []
    for claim in numeric_claims(body):
        key = _normalize_claim(claim)
        if key and key not in evidence_found:
            unsupported.append(claim)
    return unsupported


_ACRONYM = re.compile(r"\b([A-Z]{2,6})(?:s)?\b")
_COMMON_OK = frozenset(
    {
        "AI",
        "ML",
        "API",
        "URL",
        "GPU",
        "CPU",
        "RAM",
        "OS",
        "JSON",
        "XML",
        "HTML",
        "CSS",
        "HTTP",
        "HTTPS",
        "PDF",
        "RFC",
        "CLI",
        "REST",
        "OK",
        "TODO",
        "TBD",
        "FAQ",
        "RTX",
        "EU",
        "US",
        "USA",
        "USD",
        "I",
        "A",
        "BLEU",
    }
)


def _strip_code_and_links(text: str) -> str:
    t = _CODE_FENCE.sub(" ", text or "")
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub(r"\[cite:[^\]]+\]", " ", t)
    t = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", t)
    return t


def missing_planned_visuals(body: str, plan) -> dict[str, list[str]]:
    """Report planned figure or equation items not yet present in the body (soft signal)."""
    if plan is None:
        return {"missing_figures": [], "missing_equations": []}
    missing_f: list[str] = []
    missing_e: list[str] = []
    for f in plan.figures or []:
        fid = str(getattr(f, "id", "") or "")
        if fid and f"/figures/{fid}.png" not in body:
            missing_f.append(fid)
    for e in plan.equations or []:
        lx = str(getattr(e, "latex", "") or "").strip()
        eid = str(getattr(e, "id", "") or "")
        if not lx:
            continue
        compact = lx.replace(" ", "")
        bcompact = body.replace(" ", "")
        if lx in body or (len(compact) >= 3 and compact in bcompact):
            continue
        missing_e.append(eid or "eq")
    return {"missing_figures": missing_f, "missing_equations": missing_e}


def undefined_acronyms(body: str, glossary_terms: list[str] | None = None) -> list[str]:
    """Acronyms whose first occurrence is not followed by a parenthetical expansion or glossary def."""
    text = _strip_code_and_links(body or "")
    glossary_set = {
        (t or "").strip().split(" ")[0].upper().rstrip(":,.;")
        for t in (glossary_terms or [])
        if t
    }
    seen: dict[str, int] = {}
    for m in _ACRONYM.finditer(text):
        a = m.group(1)
        if a in _COMMON_OK or a in seen:
            continue
        seen[a] = m.start()
    out: list[str] = []
    for a, idx in seen.items():
        window = text[max(0, idx - 200) : idx + len(a) + 200]
        if a in glossary_set:
            continue
        if re.search(rf"\(\s*{re.escape(a)}\s*\)", window):
            continue
        if re.search(
            rf"{re.escape(a)}\s*\([A-Z][^()]{{2,80}}\)", window
        ):
            continue
        if re.search(
            rf"\b([A-Z][a-z]+(?:[-\s][A-Z][a-z]+){{1,4}})\s*\(\s*{re.escape(a)}\s*\)",
            window,
        ):
            continue
        out.append(a)
    return out


def structural_issues(body: str, bundle=None) -> list[str]:
    """All structural lints in one list (used by editor gate).

    `bundle` is optional - lints that benefit from EvidenceBundle context
    (`lint_tradeoffs_present`, `lint_baseline_pairing`) only run when given one.
    """
    out = (
        lint_structure(body)
        + lint_generic_headings(body)
        + lint_templated_headings(body)
        + lint_duplicate_headings(body)
        + lint_takeaway_repeated_as_heading(body)
        + lint_empty_placeholders(body)
        + lint_pov_after_phrase(body)
        + lint_collective_research_voice(body)
        + lint_h2_minimum(body)
        + lint_unfenced_mermaid(body)
        + lint_citation_minimum(body)
        + lint_unresolved_source_aliases(body)
        + lint_comparative_claims(body)
        + lint_required_sections(body)
        + lint_advice_traceability(body)
        + lint_section_redundancy(body)
        + lint_fake_results_table(body)
        + lint_mermaid_taxonomy(body)
        + lint_tradeoffs_present(body, bundle)
        + lint_baseline_pairing(body, bundle)
    )
    return list(dict.fromkeys(out))

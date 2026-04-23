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


def structural_issues(body: str) -> list[str]:
    """All structural lints in one list (used by editor gate)."""
    return (
        lint_structure(body)
        + lint_generic_headings(body)
        + lint_templated_headings(body)
        + lint_duplicate_headings(body)
        + lint_takeaway_repeated_as_heading(body)
        + lint_empty_placeholders(body)
        + lint_pov_after_phrase(body)
        + lint_collective_research_voice(body)
    )

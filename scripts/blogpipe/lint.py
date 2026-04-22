"""Structural and red-flag lints for drafted markdown."""

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
_BLAND_H2 = re.compile(
    r"^##\s*(Introduction|Background|Overview|Results|Conclusion|Summary)\s*$", re.M
)
_DIGITS = re.compile(r"\d")


def count_digits(s: str) -> int:
    return len(_DIGITS.findall(s))


def lint_forbidden_phrases(body: str) -> list[str]:
    issues: list[str] = []
    if _BANNED.search(body):
        issues.append("banned_marketing_phrase")
    for m in _ROBUST_UNQUAL.finditer(body):
        issues.append("robust_without_metric")
        break
    for m in _SOTA_UNQUAL.finditer(body):
        issues.append("sota_without_benchmark")
        break
    first_para = body.split("\n\n", 1)[0][:800]
    if _THROAT.search(first_para):
        issues.append("throat_clearing_opener")
    return issues


def lint_bland_h2(body: str) -> list[str]:
    bad: list[str] = []
    for m in _BLAND_H2.finditer(body):
        line = m.group(0)
        if ":" not in line:
            bad.append(line)
    return bad


def _first_h2_section(body: str, heading_phrase: str) -> str:
    pat = re.compile(
        r"^##\s*" + re.escape(heading_phrase) + r"[^\n]*\n([\s\S]*?)(?=^##\s|\Z)",
        re.M | re.I,
    )
    m = pat.search(body)
    return (m.group(1) or "").strip() if m else ""


def lint_structure(body: str) -> list[str]:
    issues: list[str] = []
    if "```mermaid" not in body:
        issues.append("no_mermaid_block")
    first = body.split("\n", 1)[0].strip() if body.strip() else ""
    if first and not re.search(r"\d", first):
        issues.append("takeaway_lacks_number")
    table_match = re.search(r"^\s*\|[^\n]*\bMethod\b[^\n]*\|", body, re.M)
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
    pov_markers = (
        "what i find",
        "the remarkable thing",
        "i think",
        "in my view",
        "what strikes me",
    )
    b = body.lower()
    if not any(m in b for m in pov_markers):
        issues.append("no_author_pov")
    aec_section = _first_h2_section(body, "Where this shows up in AEC")
    if aec_section:
        if len(aec_section.split()) < 15:
            issues.append("aec_section_too_short")
        if re.search(
            r"implications for.*?"
            r"(natural language|various applications|nlp|computer vision|many applications)\b",
            aec_section,
            re.I,
        ):
            issues.append("aec_section_generic_filler")
    return issues


_H2_BLOCK = re.compile(
    r"^##\s*[^\n]+\n([\s\S]*?)(?=^##\s|\Z)", re.M
)
_PLACEHOLDER_SEC = re.compile(
    r"^no [a-z]+ (information|reference|equation|code|content)\b",
    re.I | re.M,
)


def lint_empty_placeholders(body: str) -> list[str]:
    """Flags sections that are explicit 'no X' empty placeholders under ## bodies."""
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


def lint_pov_after_phrase(body: str) -> list[str]:
    """Require first-person or evaluative follow-up after the POV lead-in phrase."""
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


def structural_issues(body: str) -> list[str]:
    """All structural lints in one list (used by editor gate)."""
    return (
        lint_structure(body)
        + lint_empty_placeholders(body)
        + lint_pov_after_phrase(body)
    )

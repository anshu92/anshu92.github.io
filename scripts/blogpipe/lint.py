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
_DIGITS = re.compile(r"\d")


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


def structural_issues(body: str) -> list[str]:
    """All structural lints in one list (used by editor gate)."""
    return (
        lint_structure(body)
        + lint_generic_headings(body)
        + lint_empty_placeholders(body)
        + lint_pov_after_phrase(body)
    )

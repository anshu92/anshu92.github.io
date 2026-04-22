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

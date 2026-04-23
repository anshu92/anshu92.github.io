"""Committee of analysts: one tool + one LLM call per role."""

from __future__ import annotations

from typing import Any, Callable

from .adversarial import run as run_adversarial
from .code import run as run_code
from .empirical import run as run_empirical
from .methods import run as run_methods
from .practitioner import run as run_practitioner
from .related import run as run_related
from .web import run as run_web

RUNNERS: dict[str, Callable[[Any], Any]] = {
    "methods": run_methods,
    "empirical": run_empirical,
    "adversarial": run_adversarial,
    "related": run_related,
    "practitioner": run_practitioner,
    "code": run_code,
    "web": run_web,
}

__all__ = [
    "RUNNERS",
    "run_adversarial",
    "run_code",
    "run_empirical",
    "run_methods",
    "run_practitioner",
    "run_related",
    "run_web",
]

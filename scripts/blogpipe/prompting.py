"""Load and render file-based prompts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:
    from jinja2 import StrictUndefined, TemplateError
    from jinja2.sandbox import SandboxedEnvironment
except Exception:  # noqa: BLE001
    StrictUndefined = None  # type: ignore
    TemplateError = Exception  # type: ignore
    SandboxedEnvironment = None  # type: ignore


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=32)
def _load_prompt_text(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.j2"
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **context: object) -> str:
    text = _load_prompt_text(name)
    if SandboxedEnvironment is None:
        return text
    env = SandboxedEnvironment(
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    try:
        return env.from_string(text).render(**context).strip()
    except TemplateError:
        return text

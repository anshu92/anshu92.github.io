"""Load/save cache files under cache/ and reports/."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _find_repo() -> Path:
    """Find repo root (directory containing hugo.toml)."""
    if os.environ.get("BLOGPIPE_REPO_ROOT"):
        return Path(os.environ["BLOGPIPE_REPO_ROOT"]).resolve()
    cwd = Path(os.getcwd()).resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "hugo.toml").is_file():
            return p
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "hugo.toml").is_file():
            return p
    return Path(__file__).resolve().parent.parent.parent


_ROOT = _find_repo()

CACHE = _ROOT / "cache"
REPORTS = _ROOT / "reports"
CONTENT_POST = _ROOT / "content" / "post"
STATIC_FONTS = _ROOT / "static" / "fonts"


def ensure_dirs() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPO_STATIC_IMG := _ROOT / "static" / "img" / "posts").mkdir(
        parents=True, exist_ok=True
    )
    _ = REPO_STATIC_IMG  # quiet lint


def load_json(name: str, default: Any) -> Any:
    p = CACHE / name
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(name: str, data: Any) -> None:
    ensure_dirs()
    p = CACHE / name
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def try_restore_from_branch(
    branch: str = "blogpipe-memory", *relative_paths: str
) -> None:
    """Best-effort: git show origin/branch:path from CI checkout."""
    ensure_dirs()
    for rel in relative_paths:
        try:
            out = subprocess.run(
                [
                    "git",
                    "show",
                    f"origin/{branch}:{rel}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                cwd=_ROOT,
            )
            if out.returncode == 0 and out.stdout:
                p = _ROOT / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(out.stdout, encoding="utf-8")
        except Exception:
            pass

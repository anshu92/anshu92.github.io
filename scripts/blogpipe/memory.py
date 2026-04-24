"""Load/save cache files under cache/ and reports/."""

from __future__ import annotations

import json
import os
import re
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
# Hugo: static/img/posts/... → site URL /img/posts/...
STATIC_IMG_POSTS = _ROOT / "static" / "img" / "posts"


def static_img_post_dir(slug: str) -> Path:
    """Directory for one post’s generated images (cover, hero, diagram, etc.)."""
    d = STATIC_IMG_POSTS / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def static_img_figures_dir(slug: str) -> Path:
    """Directory for embedded concept figures: static/img/posts/{slug}/figures/."""
    d = static_img_post_dir(slug) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_dirs() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    STATIC_IMG_POSTS.mkdir(parents=True, exist_ok=True)


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


def append_json_list(name: str, item: Any, *, limit: int = 100) -> list[Any]:
    ensure_dirs()
    data = load_json(name, [])
    if not isinstance(data, list):
        data = []
    data.append(item)
    if limit > 0:
        data = data[-limit:]
    save_json(name, data)
    return data


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.I)


def tokenize_text(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


def _flatten_item_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_flatten_item_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_flatten_item_text(v) for v in value.values())
    return ""


def retrieve_similar_json_items(
    name: str,
    query_text: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    data = load_json(name, [])
    if not isinstance(data, list):
        return []
    items = [x for x in data if isinstance(x, dict)]
    if not items:
        return []
    q_tokens = tokenize_text(query_text)
    if not q_tokens:
        return items[-limit:]
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, item in enumerate(items):
        item_text = _flatten_item_text(item)
        tokens = tokenize_text(item_text)
        if not tokens:
            continue
        overlap = len(q_tokens & tokens)
        if overlap <= 0:
            continue
        recall = overlap / max(1, len(q_tokens))
        precision = overlap / max(1, len(tokens))
        score = (0.7 * recall) + (0.3 * precision)
        scored.append((score, idx, item))
    if not scored:
        return items[-limit:]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [item for _score, _idx, item in scored[:limit]]


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

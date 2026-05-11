from __future__ import annotations

import os
import re
from pathlib import Path


def repo_root() -> Path:
    override = os.environ.get("BLOGPIPE_REPO_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    cwd = Path.cwd().resolve()
    for p in (cwd, *cwd.parents):
        if (p / "hugo.toml").is_file():
            return p
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "hugo.toml").is_file():
            return p
    return cwd


ROOT = repo_root()
DATA = ROOT / "radar-data"
DAILY_DATA = DATA / "daily"
REPORTS = ROOT / "reports"
CONTENT_POST = ROOT / "content" / "post"
STATIC_POSTS = ROOT / "static" / "img" / "posts"


def ensure_dirs() -> None:
    for path in (DATA, DAILY_DATA, REPORTS, CONTENT_POST, STATIC_POSTS):
        path.mkdir(parents=True, exist_ok=True)


def slugify(text: str, *, limit: int = 86) -> str:
    s = re.sub(r"[^\w\s-]", "", (text or "").lower())
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    return (s[:limit].strip("-") or "post")


def post_asset_dir(slug: str) -> Path:
    out = STATIC_POSTS / slug
    out.mkdir(parents=True, exist_ok=True)
    return out

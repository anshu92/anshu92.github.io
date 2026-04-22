"""Generate cover, hero, and diagram assets."""

from __future__ import annotations

import json
import logging
from typing import Optional
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import config, memory
from .memory import _ROOT
from .models import EditorialBrief

LOG = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None  # type: ignore


def _draft_slug() -> tuple[str, Path]:
    rel = (_ROOT / "reports" / "draft_path.txt").read_text().strip()
    path = _ROOT / rel
    return path.stem, path


def _brief() -> EditorialBrief:
    p = _ROOT / "reports" / "editorial_brief.json"
    if p.is_file():
        return EditorialBrief.model_validate_json(p.read_text())
    return EditorialBrief()


def _extract_mermaid(md: str) -> str:
    m = re.search(r"```mermaid\n([\s\S]*?)```", md, re.I)
    return m.group(1).strip() if m else "flowchart LR\n  A[Start] --> B[End]"


def _kroki_svg(diagram: str, variant: str = "mermaid") -> Optional[bytes]:
    if config.dry_run():
        return b"<svg></svg>"
    u = f"{config.kroki_url()}/{variant}"
    try:
        r = httpx.post(
            u,
            content=diagram.encode("utf-8"),
            headers={
                "Content-Type": "text/plain",
                "Accept": "image/svg+xml",
            },
            timeout=60.0,
        )
        if r.status_code == 200:
            return r.content
    except Exception as e:
        LOG.warning("kroki: %s", e)
    return None


def _pillow_card(
    out: Path, title: str, subtitle: str, palette: tuple[int, int, int, int, int, int]
) -> None:
    if not Image:
        out.write_bytes(b"")
        return
    w, h = 1200, 630
    c1, c2, c3 = palette[0:3]
    c4, c5, c6 = palette[3:6]
    im = Image.new("RGB", (w, h), (c1, c2, c3))
    dr = ImageDraw.Draw(im)
    for y in range(h):
        t = y / h
        r = int(c1 + (c4 - c1) * t)
        g = int(c2 + (c5 - c2) * t)
        b_ = int(c3 + (c6 - c3) * t)
        dr.line([(0, y), (w, y)], fill=(r, g, b_))
    font = sm = None
    for fp in (
        _ROOT / "static" / "fonts" / "Inter-VariableFont_opsz,wght.ttf",
        _ROOT / "static" / "fonts" / "Inter-Latin.woff2",
    ):
        try:
            if fp.suffix.lower() in (".ttf", ".otf") and fp.is_file():
                font = ImageFont.truetype(str(fp), 36)
                sm = ImageFont.truetype(str(fp), 24)
                break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()
        sm = font
    dr.text((64, 80), title[:90], fill=(255, 255, 255), font=font)
    dr.text((64, 200), subtitle[:200], fill=(240, 240, 240), font=sm)
    im.save(out, format="PNG")


def _try_node_cover(out: Path, payload: dict) -> bool:
    script = _ROOT / "scripts" / "blogpipe" / "visuals" / "render_cover.mjs"
    if not script.is_file():
        return False
    jin = out.parent / "cover_input.json"
    jin.write_text(json.dumps(payload), encoding="utf-8")
    try:
        subprocess.run(
            ["node", str(script), str(jin), str(out)],
            check=True,
            timeout=60,
            cwd=str(script.parent),
        )
        return out.is_file() and out.stat().st_size > 100
    except Exception as e:
        LOG.debug("node cover: %s", e)
    return False


def run() -> None:
    memory.ensure_dirs()
    _, draft_path = _draft_slug()
    brief = _brief()
    md = draft_path.read_text(encoding="utf-8")
    slug = draft_path.stem
    m = re.search(r'title:\s*"(.*)"', md)
    title = m.group(1) if m else slug
    st = re.search(r'one_sentence_takeaway:\s*"(.*)"', md, re.S)
    sub = (st.group(1) if st else title)[:200]
    ddir = _ROOT / "static" / "img" / "posts" / slug
    ddir.mkdir(parents=True, exist_ok=True)
    # Palettes by art_direction
    pals = {
        "editorial_illustration": (40, 60, 120, 80, 120, 200),
        "isometric_diagram": (30, 80, 50, 60, 140, 90),
        "minimal_geometric": (20, 20, 20, 60, 60, 60),
        "risograph_print": (200, 80, 100, 255, 180, 200),
        "data_viz_portrait": (10, 30, 50, 30, 80, 120),
        "hand_drawn_schematic": (250, 245, 220, 220, 200, 180),
        "cyan_blueprint": (0, 40, 60, 0, 100, 120),
        "dark_technical_photo": (5, 5, 5, 40, 40, 50),
        "collage_cutout": (180, 100, 50, 220, 160, 100),
    }
    pal = pals.get(
        brief.art_direction,
        (45, 55, 70, 90, 110, 140),
    )
    cover = ddir / "cover.png"
    hero = ddir / "hero.png"
    if not _try_node_cover(
        cover, {"format": brief.format_name, "title": title, "takeaway": sub}
    ):
        _pillow_card(cover, title, sub, (*pal, pal[0], pal[1], pal[2]))
    if config.gemini_key() and not config.dry_run():
        # Optional: Imagen via google-genai if installed and API available.
        try:
            from google import genai as _genai  # type: ignore

            client = _genai.Client(api_key=config.gemini_key())
            pr = f"Editorial cover art for ML article, style {brief.art_direction}, no text, abstract, no human faces: {title}"
            r = client.models.generate_images(  # type: ignore[union-attr]
                model="imagen-3.0-generate-002",
                prompt=pr,
            )
            if r and r.generated_images:  # type: ignore[union-attr]
                b = r.generated_images[0].as_png_bytes()  # type: ignore[union-attr]
                if b:
                    hero.write_bytes(b)
        except Exception as e:
            LOG.warning("gemini image (optional): %s", e)
    if not hero.is_file() or hero.stat().st_size < 100:
        if cover.is_file() and cover.stat().st_size > 0:
            hero.write_bytes(cover.read_bytes())
    mer = _extract_mermaid(md)
    svg = _kroki_svg(mer, "mermaid")
    if svg and b"svg" in svg[:200].lower() + svg[-100:].lower():
        (ddir / "diagram.svg").write_bytes(svg)
    # Variety ledger
    leg = _ROOT / "cache" / "variety_ledger.json"
    data = (
        json.loads(leg.read_text()) if leg.is_file() else {"entries": []}
    ) or {"entries": []}
    data["entries"] = (data.get("entries") or []) + [
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "format": brief.format_name,
            "opener": brief.opener_hook,
            "art": brief.art_direction,
            "diagram": "mermaid",
            "slug": slug,
        }
    ]
    data["entries"] = data["entries"][-24:]
    leg.parent.mkdir(parents=True, exist_ok=True)
    leg.write_text(json.dumps(data, indent=2), encoding="utf-8")
    LOG.info("visuals: wrote assets under %s", ddir)

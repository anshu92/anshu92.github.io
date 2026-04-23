"""Generate cover, hero, diagram, and optional planned body figures."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from . import config, memory
from .draft import _slugify
from .llm_chain import (
    _BLACKLIST_TTL_DAILY,
    _BLACKLIST_TTL_SHORT,
    is_blacklist_key_active,
    register_blacklist_key,
)
from .memory import _ROOT
from .models import EditorialBrief, EvidenceBundle, FigureSpec

LOG = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None  # type: ignore


def _draft_paths() -> tuple[Path, str, str]:
    rel = (_ROOT / "reports" / "draft_path.txt").read_text().strip()
    path = _ROOT / rel
    md = path.read_text(encoding="utf-8")
    m = re.search(r'title:\s*"(.*)"', md)
    title = m.group(1) if m else path.stem
    slug = _slugify(title)
    return path, title, slug


def _load_evidence() -> tuple[EvidenceBundle | None, VisualPlan | None]:
    p = _ROOT / "reports" / "evidence_bundle.json"
    if not p.is_file():
        return None, None
    try:
        b = EvidenceBundle.model_validate_json(p.read_text())
        return b, b.visual_plan
    except Exception as e:  # noqa: BLE001
        LOG.debug("evidence for visuals: %s", e)
        return None, None


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
        r_ = int(c1 + (c4 - c1) * t)
        g_ = int(c2 + (c5 - c2) * t)
        b_ = int(c3 + (c6 - c3) * t)
        dr.line([(0, y), (w, y)], fill=(r_, g_, b_))
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


def _imagen_write(prompt: str, out: Path) -> bool:
    if not config.gemini_key() or config.dry_run():
        return False
    try:
        from google import genai as _genai  # type: ignore

        client = _genai.Client(api_key=config.gemini_key())
        for model_id in (
            "imagen-4.0-generate-001",
            "imagen-3.0-generate-002",
            "imagen-3.0-generate-001",
        ):
            try:
                r = client.models.generate_images(  # type: ignore[union-attr]
                    model=model_id,
                    prompt=prompt,
                )
                if r and r.generated_images:  # type: ignore[union-attr]
                    b = r.generated_images[0].as_png_bytes()  # type: ignore[union-attr]
                    if b:
                        out.write_bytes(b)
                        if out.is_file() and out.stat().st_size > 100:
                            return True
            except Exception as inner:
                LOG.debug("imagen %s: %s", model_id, inner)
    except Exception as e:
        LOG.debug("imagen: %s", e)
    return False


def _pollinations_write(prompt: str, out: Path) -> bool:
    if is_blacklist_key_active("image:pollinations"):
        return False
    from urllib.parse import quote

    p = (prompt or "")[:900]
    seed = int(hashlib.sha256(p.encode("utf-8", errors="replace")).hexdigest()[:8], 16) % 10**9
    u = (
        f"https://image.pollinations.ai/prompt/{quote(p)}"
        f"?width=1280&height=720&nologo=true&model=flux&seed={seed}"
    )
    try:
        r = httpx.get(u, timeout=httpx.Timeout(60.0), follow_redirects=True)
        if r.status_code == 200:
            c = r.content
            is_png = c[:8] == b"\x89PNG\r\n\x1a\n" or "png" in (r.headers.get("content-type") or "").lower()
            if c and is_png and len(c) > 200:
                out.write_bytes(c)
                return out.is_file() and out.stat().st_size > 200
        ttl = _BLACKLIST_TTL_DAILY if r.status_code in (400, 404) else _BLACKLIST_TTL_SHORT
        if r.status_code in (400, 404, 429, 500, 502, 503, 504):
            register_blacklist_key("image:pollinations", float(ttl))
    except Exception as e:
        LOG.debug("pollinations: %s", e)
        register_blacklist_key("image:pollinations", float(_BLACKLIST_TTL_SHORT))
    return False


def _huggingface_flux_write(prompt: str, out: Path) -> bool:
    if is_blacklist_key_active("image:huggingface:flux-schnell"):
        return False
    tok = config.hf_token()
    if not tok:
        return False
    url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    try:
        r = httpx.post(
            url,
            headers={"Authorization": f"Bearer {tok}"},
            json={"inputs": (prompt or "")[:900]},
            timeout=httpx.Timeout(120.0),
        )
        if r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n":
            out.write_bytes(r.content)
            return out.is_file() and out.stat().st_size > 200
        ttl = _BLACKLIST_TTL_DAILY if r.status_code in (400, 404) else _BLACKLIST_TTL_SHORT
        if r.status_code in (400, 404, 429, 500, 502, 503, 504):
            register_blacklist_key("image:huggingface:flux-schnell", float(ttl))
    except Exception as e:
        LOG.debug("huggingface flux: %s", e)
        register_blacklist_key("image:huggingface:flux-schnell", float(_BLACKLIST_TTL_SHORT))
    return False


def _pillow_figure(out: Path, spec: FigureSpec, palette: tuple[int, int, int, int, int, int]) -> None:
    if not Image:
        out.write_bytes(b"")
        return
    w, h = 1280, 720
    c1, c2, c3 = palette[0:3]
    c4, c5, c6 = palette[3:6]
    im = Image.new("RGB", (w, h), (c1, c2, c3))
    dr = ImageDraw.Draw(im)
    for y in range(h):
        t = y / h
        r_ = int(c1 + (c4 - c1) * t)
        g_ = int(c2 + (c5 - c2) * t)
        b_ = int(c3 + (c6 - c3) * t)
        dr.line([(0, y), (w, y)], fill=(r_, g_, b_))
    font = sm = None
    for fp in (
        _ROOT / "static" / "fonts" / "Inter-VariableFont_opsz,wght.ttf",
    ):
        try:
            if fp.suffix.lower() in (".ttf", ".otf") and fp.is_file():
                font = ImageFont.truetype(str(fp), 32)
                sm = ImageFont.truetype(str(fp), 22)
                break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()
        sm = font
    label = (spec.caption or spec.alt or spec.id)[:120]
    dr.text((48, 48), label, fill=(255, 255, 255), font=font)
    sub = (spec.prompt or "")[:220].replace("\n", " ")
    dr.text((48, 120), sub, fill=(240, 240, 240), font=sm)
    im.save(out, format="PNG")


def generate_figure(
    spec: FigureSpec,
    slug: str,
    brief: EditorialBrief,
    palette: tuple[int, ...],
) -> str | None:
    """Run free-first image chain; write static/img/posts/{slug}/figures/{id}.png. Return URL or None."""
    ddir = _ROOT / "static" / "img" / "posts" / slug / "figures"
    ddir.mkdir(parents=True, exist_ok=True)
    out = ddir / f"{spec.id}.png"
    if config.dry_run():
        out.write_bytes(b"")
        return f"/img/posts/{slug}/figures/{spec.id}.png"
    pr = (
        f"Editorial technical illustration, {brief.art_direction}, no text in image, "
        f"no human faces, abstract: {spec.prompt}"
    )
    if _imagen_write(pr, out):
        return f"/img/posts/{slug}/figures/{spec.id}.png"
    if _pollinations_write(pr, out):
        return f"/img/posts/{slug}/figures/{spec.id}.png"
    if _huggingface_flux_write(pr, out):
        return f"/img/posts/{slug}/figures/{spec.id}.png"
    px = [int(x) for x in palette[:6]]
    while len(px) < 6:
        px.append(px[-1] if px else 45)
    _pillow_figure(out, spec, tuple(px))
    if out.is_file() and out.stat().st_size > 50:
        return f"/img/posts/{slug}/figures/{spec.id}.png"
    return None


def run() -> None:
    memory.ensure_dirs()
    draft_path, title, slug = _draft_paths()
    brief = (
        EditorialBrief.model_validate_json(
            (_ROOT / "reports" / "editorial_brief.json").read_text()
        )
        if (_ROOT / "reports" / "editorial_brief.json").is_file()
        else EditorialBrief()
    )
    md = draft_path.read_text(encoding="utf-8")
    m = re.search(r'one_sentence_takeaway:\s*"(.*)"', md, re.S)
    sub = (m.group(1) if m else title)[:200]
    ddir = _ROOT / "static" / "img" / "posts" / slug
    ddir.mkdir(parents=True, exist_ok=True)
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
    hero_pr = (
        f"Editorial cover art for ML article, style {brief.art_direction}, no text, "
        f"abstract, no human faces: {title}"
    )
    if not _imagen_write(hero_pr, hero):
        if not _pollinations_write(hero_pr, hero) and not _huggingface_flux_write(hero_pr, hero):
            if cover.is_file() and cover.stat().st_size > 0:
                hero.write_bytes(cover.read_bytes())
    if not hero.is_file() or hero.stat().st_size < 100:
        if cover.is_file() and cover.stat().st_size > 0:
            hero.write_bytes(cover.read_bytes())
    _bundle, vplan = _load_evidence()
    if vplan and vplan.figures:
        for spec in vplan.figures:
            u = generate_figure(spec, slug, brief, pal)
            if u:
                LOG.info("visuals: figure %s -> %s", spec.id, u)
    mer = _extract_mermaid(md)
    svg = _kroki_svg(mer, "mermaid")
    if svg and b"svg" in svg[:200].lower() + svg[-100:].lower():
        (ddir / "diagram.svg").write_bytes(svg)
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
    LOG.info("visuals: wrote assets under %s (slug=%s)", ddir, slug)

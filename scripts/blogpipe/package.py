"""Build reports/daily_email.html and optional draft_post.pdf for notification."""

from __future__ import annotations

import html
import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .memory import _ROOT

LOG = logging.getLogger(__name__)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _split_frontmatter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not m:
        return {}, text
    front: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        front[k.strip()] = v.strip().strip('"')
    return front, text[m.end() :]


def _first_section(body: str, heading: str) -> str:
    m = re.search(
        rf"^##\s+{re.escape(heading)}\s*\n([\s\S]*?)(?=^##\s+|\Z)",
        body,
        re.MULTILINE,
    )
    if not m:
        return ""
    return m.group(1).strip()


def _h2_sections(body: str, limit: int = 2) -> list[tuple[str, str]]:
    """Return the first `limit` (heading, body) pairs in order — heading-agnostic."""
    out: list[tuple[str, str]] = []
    pat = re.compile(r"^##\s+(.+?)\s*\n([\s\S]*?)(?=^##\s+|\Z)", re.M)
    for m in pat.finditer(body or ""):
        out.append((m.group(1).strip(), m.group(2).strip()))
        if len(out) >= limit:
            break
    return out


def _opening_prose(body: str) -> str:
    """Grab everything before the first ## heading (the takeaway + lead paragraph)."""
    m = re.search(r"^##\s", body or "", re.M)
    head = body[: m.start()] if m else (body or "")
    return head.strip()


def _compact(text: str, limit: int = 320) -> str:
    out = " ".join(text.split())
    return out[:limit].rstrip() + ("..." if len(out) > limit else "")


def _rubric_rows(items: list[dict]) -> list[str]:
    rows: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        label = str(it.get("item") or it.get("name") or "").strip()
        if not label:
            continue
        score = int(it.get("score", 0) or 0)
        cls = "ok" if score else "bad"
        rows.append(
            f"<tr class='{cls}'><td>{score}</td><td>{html.escape(label)}</td></tr>"
        )
    return rows


def run() -> Path:
    rep = _ROOT / "reports"
    out = rep / "daily_email.html"
    brief = _read_json(rep / "editorial_brief.json")
    editor = _read_json(rep / "editor_report.json")
    rank = _read_json(rep / "rank_result.json")
    rejected = rank.get("rejected") or []
    dpath = (rep / "draft_path.txt").read_text().strip() if (rep / "draft_path.txt").is_file() else ""
    draft_rel = dpath.strip()
    front: dict = {}
    body = ""
    if draft_rel:
        draft_file = _ROOT / draft_rel
        if draft_file.is_file():
            front, body = _split_frontmatter(draft_file.read_text(encoding="utf-8"))
    title = front.get("title") or Path(draft_rel).stem
    takeaway = front.get("one_sentence_takeaway") or ""
    sections = _h2_sections(body, limit=2)
    opening = _compact(_opening_prose(body), 700)
    lead_heading = sections[0][0] if sections else ""
    lead_body = _compact(sections[0][1], 700) if sections else ""
    second_heading = sections[1][0] if len(sections) > 1 else ""
    second_body = _compact(sections[1][1], 420) if len(sections) > 1 else ""
    pr_url = (rep / "pr_url.txt").read_text().strip() if (rep / "pr_url.txt").is_file() else ""
    branch = (rep / "branch.txt").read_text().strip() if (rep / "branch.txt").is_file() else ""
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Blog draft</title>",
        "<style>body{font-family:system-ui,Segoe UI,sans-serif;max-width:800px;margin:2rem auto;padding:0 1rem;color:#111;}",
        "table{border-collapse:collapse;width:100%} th,td{border:1px solid #ccc;padding:0.4rem;text-align:left;}",
        ".ok{background:#dff6dd}.bad{background:#fde2e1}.muted{color:#555} code{background:#f6f8fa;padding:0.1rem 0.25rem;border-radius:4px}</style></head><body>",
        "<h1>Review blog draft</h1>",
    ]
    if pr_url:
        lines.append(
            f"<p><strong>PR:</strong> <a href='{html.escape(pr_url)}'>{html.escape(pr_url)}</a></p>"
        )
    elif branch:
        lines.append(
            f"<p><strong>Branch pushed:</strong> <code>{html.escape(branch)}</code> "
            "<span class='muted'>(PR was not created automatically)</span></p>"
        )
    else:
        lines.append(
            "<p><strong>PR status:</strong> <span class='muted'>not created</span></p>"
        )
    subj = f"[Blog Draft, rubric {editor.get('rubric_score', 0)}/15]"
    lines.append(f"<p><strong>Subject line hint:</strong> {html.escape(subj)}</p>")
    if title:
        lines.append(f"<h2>{html.escape(str(title))}</h2>")
    if takeaway:
        lines.append(f"<p><strong>Takeaway:</strong> {html.escape(_compact(str(takeaway), 320))}</p>")
    if draft_rel:
        lines.append(f"<p><strong>Draft file:</strong> <code>{html.escape(draft_rel)}</code></p>")
    if opening:
        lines.append(f"<p>{html.escape(opening)}</p>")
    if lead_heading:
        lines.append(
            f"<h3>{html.escape(lead_heading)}</h3><p>{html.escape(lead_body)}</p>"
        )
    if second_heading:
        lines.append(
            f"<h3>{html.escape(second_heading)}</h3><p>{html.escape(second_body)}</p>"
        )
    if brief:
        summary_bits = []
        if brief.get("format_name"):
            summary_bits.append(f"format: {brief['format_name']}")
        if brief.get("opener_hook"):
            summary_bits.append(f"opener: {brief['opener_hook']}")
        if brief.get("art_direction"):
            summary_bits.append(f"art: {brief['art_direction']}")
        if summary_bits:
            lines.append(f"<p><strong>Editorial plan:</strong> {html.escape(', '.join(summary_bits))}</p>")
    if editor:
        sc = int(editor.get("rubric_score", 0))
        rows = _rubric_rows(editor.get("rubric_items") or [])
        lines.append(f"<h2>Rubric: {sc}/15</h2>")
        if rows:
            lines.append("<table><tr><th>Score</th><th>Criterion</th></tr>")
            lines.extend(rows)
            lines.append("</table>")
        if editor.get("five_questions"):
            lines.append("<h3>Five-question check</h3><ul>")
            for key, value in (editor.get("five_questions") or {}).items():
                lines.append(
                    f"<li><strong>{html.escape(str(key))}:</strong> {html.escape(_compact(str(value), 220))}</li>"
                )
            lines.append("</ul>")
    if rejected:
        lines.append("<h2>Not selected</h2><ul>")
        for it in rejected[:5]:
            if isinstance(it, dict):
                title_text = html.escape(str(it.get("title", "")))
                source_text = html.escape(str(it.get("source", "")))
                lines.append(f"<li>{title_text} <span class='muted'>({source_text})</span></li>")
        lines.append("</ul>")
    lines.append("<p><em>Merge the PR to publish, or close it to discard the draft.</em></p>")
    lines.append("</body></html>")
    out.write_text("\n".join(lines), encoding="utf-8")
    try:
        write_draft_print_pdf()
    except Exception as e:  # noqa: BLE001 — never block email HTML on PDF
        LOG.warning("draft_post.pdf not created: %s", e)
    # CI attaches draft_post.pdf; ensure a one-page file exists if wkhtmltopdf is available
    _pdf = rep / "draft_post.pdf"
    if shutil.which("wkhtmltopdf") and not _pdf.is_file():
        _fallback_minimal_pdf(
            _pdf,
            "The draft was not available to render (see daily_email.html and the repository).",
        )
    return out


_PRINT_CSS = """
@page { margin: 1.2cm; }
html { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 11pt; line-height: 1.45; color: #111; }
h1 { font-size: 1.35rem; border-bottom: 1px solid #ccc; padding-bottom: 0.25em; }
h2, h3 { font-size: 1.1rem; margin-top: 1.2em; }
p { margin: 0.5em 0; }
code, pre { font-size: 0.88em; }
pre { background: #f6f8fa; padding: 0.5em; overflow-x: auto; border-radius: 4px; }
table { border-collapse: collapse; width: 100%; font-size: 0.95em; }
th, td { border: 1px solid #ccc; padding: 0.35em 0.5em; text-align: left; }
img { max-width: 100%; height: auto; }
"""


def _pandoc_md_to_html_fragment(md_body: str) -> str | None:
    """Convert markdown (GFM) to an HTML body fragment; requires ``pandoc`` on PATH."""
    pandoc = shutil.which("pandoc")
    if not pandoc:
        return None
    try:
        r = subprocess.run(
            [pandoc, "-f", "gfm", "-t", "html"],
            input=md_body,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        LOG.warning("pandoc failed: %s", e)
        return None
    return r.stdout or ""


def _html_resolve_static_urls(html: str, root: Path) -> str:
    """Map ``/img/...`` to ``file://.../static/img/...`` for local PDF renderers."""

    def repl_img(m: re.Match[str]) -> str:
        url = m.group(1) or ""
        if not url.startswith("/img/"):
            return m.group(0)
        rel = url.removeprefix("/img/")
        p = (root / "static" / "img" / rel).resolve()
        if p.is_file():
            return f'src="file://{p}"'
        return m.group(0)

    return re.sub(r'src="(/img/[^"]+)"', repl_img, html)


def _try_write_draft_pdf(rep: Path, front: dict, body: str) -> Path | None:
    """Build ``rep/draft_post.pdf`` if pandoc and wkhtmltopdf are available."""
    wk = shutil.which("wkhtmltopdf")
    if not wk:
        LOG.info("wkhtmltopdf not on PATH; skip draft_post.pdf")
        return None
    frag = _pandoc_md_to_html_fragment(body)
    if not frag:
        if not shutil.which("pandoc"):
            LOG.info("pandoc not on PATH; using minimal PDF note")
        return _fallback_minimal_pdf(
            pdf_path,
            "Pandoc was not available to render the full draft. Open the PR or use the markdown in this repo.",
        )
    title = str(front.get("title") or "Draft post").strip()
    full_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>{_PRINT_CSS}</style>
</head><body>
<article>
<h1>{html.escape(title)}</h1>
{_html_resolve_static_urls(frag, _ROOT)}
</article>
</body></html>"""
    pdf_path = rep / "draft_post.pdf"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(full_html)
        tmp_path = tmp.name
    r: subprocess.CompletedProcess[str] | None = None
    try:
        r = subprocess.run(
            [
                wk,
                "--quiet",
                "--print-media-type",
                "--enable-local-file-access",
                tmp_path,
                str(pdf_path),
            ],
            capture_output=True,
            text=True,
        )
    except OSError as e:
        LOG.warning("wkhtmltopdf: %s", e)
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)  # type: ignore[call-arg]
        except OSError:
            pass
    if r is None or r.returncode != 0 or not pdf_path.is_file():
        if r is not None:
            LOG.warning(
                "wkhtmltopdf failed rc=%s stderr=%s",
                r.returncode,
                (r.stderr or "")[:1000],
            )
        if pdf_path.is_file():
            try:
                pdf_path.unlink()
            except OSError:
                pass
        return _fallback_minimal_pdf(pdf_path, "Draft PDF render failed; view the markdown in the repo PR.")
    return pdf_path


def _fallback_minimal_pdf(pdf_path: Path, message: str) -> Path | None:
    """One-page note when full render is unavailable; keeps email attachment path valid."""
    wk = shutil.which("wkhtmltopdf")
    if not wk:
        return None
    msg_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><title>Draft</title>
<style>body{{font-family:sans-serif;padding:1rem}}</style></head><body>
<p>{html.escape(message)}</p>
</body></html>"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(msg_html)
        tpath = tmp.name
    try:
        s = subprocess.run(
            [wk, "--quiet", tpath, str(pdf_path)],
            capture_output=True,
            text=True,
        )
        if s.returncode == 0 and pdf_path.is_file():
            return pdf_path
    except OSError:
        pass
    finally:
        try:
            Path(tpath).unlink(missing_ok=True)  # type: ignore[call-arg]
        except OSError:
            pass
    return None


def write_draft_print_pdf() -> Path | None:
    """If ``draft_path.txt`` points at a post, render ``draft_post.pdf`` for email attachment."""
    rep = _ROOT / "reports"
    dpath = (rep / "draft_path.txt").read_text().strip() if (rep / "draft_path.txt").is_file() else ""
    if not dpath:
        return None
    draft_file = _ROOT / dpath
    if not draft_file.is_file():
        return None
    front, body = _split_frontmatter(draft_file.read_text(encoding="utf-8"))
    return _try_write_draft_pdf(rep, front, body)

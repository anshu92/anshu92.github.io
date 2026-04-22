"""Build reports/daily_email.html for notification."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from .memory import _ROOT


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
    tldr = _compact(_first_section(body, "TL;DR"), 700)
    why = _compact(_first_section(body, "Why this matters"), 420)
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
    if tldr:
        lines.append(f"<h3>TL;DR</h3><p>{html.escape(tldr)}</p>")
    if why:
        lines.append(f"<h3>Why this matters</h3><p>{html.escape(why)}</p>")
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
    return out

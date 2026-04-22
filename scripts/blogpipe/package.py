"""Build reports/daily_email.html for notification."""

from __future__ import annotations

import json
import html
from pathlib import Path

from .memory import _ROOT


def run() -> Path:
    rep = _ROOT / "reports"
    out = rep / "daily_email.html"
    brief: dict = {}
    j = rep / "editorial_brief.json"
    if j.is_file():
        brief = json.loads(j.read_text())
    ed = rep / "editor_report.json"
    editor = json.loads(ed.read_text()) if ed.is_file() else {}
    rjson = rep / "rank_result.json"
    rank = json.loads(rjson.read_text()) if rjson.is_file() else {}
    rejected = rank.get("rejected") or []
    dpath = (rep / "draft_path.txt").read_text() if (rep / "draft_path.txt").is_file() else ""
    tldr = ""
    if dpath:
        t = (_ROOT / dpath.strip()).read_text(encoding="utf-8")
        if "## TL;DR" in t or "## TL" in t:
            tldr = t[:1200]
    pr_url = (rep / "pr_url.txt").read_text() if (rep / "pr_url.txt").is_file() else "PR: (create locally)"
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Blog draft</title>",
        "<style>body{font-family:system-ui,Segoe UI,sans-serif;max-width:800px;margin:2rem auto;padding:0 1rem;}",
        "table{border-collapse:collapse;width:100%} th,td{border:1px solid #ccc;padding:0.4rem;}",
        ".ok{background:#d4edda}.bad{background:#f8d7da}</style></head><body>",
        f"<h1>Review blog draft</h1><p><strong>PR / branch:</strong> {html.escape(pr_url)}</p>",
    ]
    subj = f"[Blog Draft, rubric {editor.get('rubric_score', 0)}/15]"
    lines.append(f"<p><strong>Subject line hint:</strong> {html.escape(subj)}</p>")
    if tldr:
        lines.append(f"<h2>Excerpt</h2><pre style='white-space:pre-wrap'>{html.escape(tldr)}</pre>")
    if brief:
        lines.append(
            f"<h2>Editorial context</h2><pre>{html.escape(json.dumps(brief, indent=2)[:5000])}</pre>"
        )
    if editor:
        sc = int(editor.get("rubric_score", 0))
        lines.append(f"<h2>Rubric: {sc}/15</h2><table><tr><th>OK</th><th>Detail</th></tr>")
        for it in editor.get("rubric_items", []) or []:
            if isinstance(it, dict):
                ok = it.get("ok", True)
                lines.append(
                    f"<tr class='{'ok' if ok else 'bad'}'><td>{ok}</td><td>{html.escape(str(it))}</td></tr>"
                )
        lines.append("</table>")
        if editor.get("five_questions"):
            lines.append(
                "<h2>Five-question check</h2><pre style='white-space:pre-wrap'>"
                f"{html.escape(json.dumps(editor['five_questions'], indent=2))}</pre>"
            )
    if rejected:
        lines.append("<h2>Not selected (top alternatives)</h2><ul>")
        for it in rejected[:5]:
            if isinstance(it, dict):
                lines.append(
                    f"<li>{html.escape(it.get('title',''))} &mdash; {html.escape(it.get('source',''))}</li>"
                )
        lines.append("</ul>")
    lines.append("<p><em>Merge the PR to publish (draft=false flip runs in a second workflow) or close to discard.</em></p>")
    lines.append("</body></html>")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

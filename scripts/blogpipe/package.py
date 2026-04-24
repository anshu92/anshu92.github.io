"""Build review artifacts from the canonical quality report."""

from __future__ import annotations

import base64
import html
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

from . import quality, visuals
from .memory import _ROOT
from .models import ArtifactResult, EditorReport, FailureReason, QualityReport, RenderReport

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


def _h2_sections(body: str, limit: int = 2) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pat = re.compile(r"^##\s+(.+?)\s*\n([\s\S]*?)(?=^##\s+|\Z)", re.M)
    for m in pat.finditer(body or ""):
        out.append((m.group(1).strip(), m.group(2).strip()))
        if len(out) >= limit:
            break
    return out


def _opening_prose(body: str) -> str:
    m = re.search(r"^##\s", body or "", re.M)
    head = body[: m.start()] if m else (body or "")
    return head.strip()


def _compact(text: str, limit: int = 320) -> str:
    out = " ".join(text.split())
    return out[:limit].rstrip() + ("..." if len(out) > limit else "")


_REASON_LABELS = {
    "grounding_issue": "Unsupported claim",
    "render_error": "Render blocked",
    "package_error": "Packaging blocked",
    "prepackage_quality_blocked": "Packaging was intentionally skipped because the quality gate failed.",
    "package_skipped_due_to_quality_gate": "Render checks were not attempted because the draft was already blocked.",
    "takeaway_lacks_number": "The opening takeaway does not include a concrete number.",
    "generic_heading_used": "The draft still uses generic section headings.",
    "templated_heading_used": "The draft still uses factory-style template headings.",
    "citation_count_below_min": "The draft does not include enough citations for review.",
    "comparative_claim_missing_metric": "A comparative claim is missing a named metric or comparison target.",
    "missing_mechanism_section": "The draft does not clearly explain how the method works.",
    "missing_decision_section": "The draft does not help the reader decide when to use the method.",
    "advice_without_traceability": "Advice is present without support from evidence or explicit author synthesis.",
}


def _humanize_reason(reason: FailureReason) -> str:
    code = (reason.code or "").strip()
    if code.startswith("lint:"):
        code = code.split(":", 1)[1]
    label = _REASON_LABELS.get(code)
    if label:
        return label
    if reason.message and reason.message != reason.code:
        return reason.message
    return code.replace("_", " ").strip().capitalize() or "Quality issue"


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


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)  # type: ignore[call-arg]
    except OSError:
        pass


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
figure { margin: 1rem 0; }
figcaption { color: #444; font-size: 0.92em; }
"""


def _render_mermaid_blocks(body: str) -> tuple[str, list[str], bool]:
    errors: list[str] = []
    rendered_any = False

    def repl(m: re.Match[str]) -> str:
        nonlocal rendered_any
        diagram = (m.group(1) or "").strip()
        svg = visuals._kroki_svg(diagram, "mermaid")
        if not svg:
            errors.append("mermaid_render_failed")
            return m.group(0)
        rendered_any = True
        data = base64.b64encode(svg).decode("ascii")
        return (
            '<figure class="diagram">'
            '<img alt="Rendered mermaid diagram" '
            f'src="data:image/svg+xml;base64,{data}"/>'
            "<figcaption>Rendered mermaid diagram</figcaption>"
            "</figure>"
        )

    out = re.sub(r"```mermaid\s*\n([\s\S]*?)\n```", repl, body or "", flags=re.I)
    has_mermaid = "```mermaid" in (body or "")
    return out, errors, (rendered_any or not has_mermaid)


def _pandoc_md_to_html_fragment(md_body: str) -> str | None:
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


def _html_resolve_static_urls(text: str, root: Path) -> str:
    def repl_img(m: re.Match[str]) -> str:
        url = m.group(1) or ""
        if not url.startswith("/img/"):
            return m.group(0)
        rel = url.removeprefix("/img/")
        p = (root / "static" / "img" / rel).resolve()
        if p.is_file():
            return f'src="file://{p}"'
        return m.group(0)

    return re.sub(r'src="(/img/[^"]+)"', repl_img, text)


def _validate_rendered_html(body: str, html_text: str) -> tuple[list[str], list[str], dict[str, bool]]:
    errors: list[str] = []
    warnings: list[str] = []
    has_table_in_md = bool(re.search(r"^\|.+\|$", body or "", re.M))
    html_has_table = "<table" in html_text.lower()
    mermaid_raw = bool(
        re.search(r"(language-mermaid|```mermaid|(?:graph|flowchart)\s+[A-Z]{1,3})", html_text, re.I)
    )
    tables_raw = "| Method | Metric | Baseline |" in html_text
    images_resolved = True
    captions_ok = True

    for src in re.findall(r'src="([^"]+)"', html_text):
        if src.startswith("data:image/"):
            continue
        if src.startswith("file://"):
            path = Path(src.removeprefix("file://"))
            if not path.is_file():
                images_resolved = False
                errors.append(f"missing_image:{path}")
        elif src.startswith("/img/"):
            path = (_ROOT / "static" / src.removeprefix("/")).resolve()
            if not path.is_file():
                images_resolved = False
                errors.append(f"missing_image:{src}")

    img_tags = re.findall(r"<img\b[^>]*>", html_text, re.I)
    for tag in img_tags:
        m = re.search(r'alt="([^"]*)"', tag, re.I)
        alt = (m.group(1).strip() if m else "").lower()
        if not alt or alt in {"image", "figure", "diagram", "accessibility", "placeholder"}:
            captions_ok = False
            errors.append("placeholder_image_alt")

    density_ok = len(re.findall(r"\b\w+\b", body or "")) >= 120
    if not density_ok:
        errors.append("page_density_too_low")

    if mermaid_raw:
        errors.append("raw_mermaid_in_render")
    if has_table_in_md and not html_has_table:
        errors.append("results_table_not_rendered")
    if tables_raw:
        errors.append("raw_markdown_table_in_render")
    if not has_table_in_md:
        warnings.append("no_results_table_in_markdown")
    flags = {
        "mermaid_rendered": not mermaid_raw,
        "tables_rendered": (not has_table_in_md) or html_has_table,
        "images_resolved": images_resolved,
        "captions_ok": captions_ok,
        "density_ok": density_ok,
    }
    return errors, warnings, flags


def _render_full_html(front: dict, body: str) -> tuple[str | None, list[str], list[str], dict[str, bool]]:
    body_for_html, mermaid_errors, mermaid_ok = _render_mermaid_blocks(body)
    frag = _pandoc_md_to_html_fragment(body_for_html)
    if not frag:
        flags = {
            "mermaid_rendered": False,
            "tables_rendered": False,
            "images_resolved": False,
            "captions_ok": False,
            "density_ok": False,
        }
        return None, (mermaid_errors + ["pandoc_render_failed"]), [], flags
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
    errors, warnings, flags = _validate_rendered_html(body, full_html)
    flags["mermaid_rendered"] = flags["mermaid_rendered"] and mermaid_ok
    errors = list(dict.fromkeys(mermaid_errors + errors))
    return full_html, errors, warnings, flags


def _render_pdf_from_html(html_path: Path, pdf_path: Path) -> ArtifactResult:
    wk = shutil.which("wkhtmltopdf")
    if not wk:
        return ArtifactResult(
            artifact_type="pdf",
            ok=False,
            artifact_path=str(pdf_path),
            errors=["wkhtmltopdf_missing"],
        )
    try:
        r = subprocess.run(
            [
                wk,
                "--quiet",
                "--print-media-type",
                "--enable-local-file-access",
                str(html_path),
                str(pdf_path),
            ],
            capture_output=True,
            text=True,
        )
    except OSError as e:
        return ArtifactResult(
            artifact_type="pdf",
            ok=False,
            artifact_path=str(pdf_path),
            errors=[f"wkhtmltopdf_error:{e}"],
        )
    if r.returncode != 0 or not pdf_path.is_file():
        _remove_if_exists(pdf_path)
        return ArtifactResult(
            artifact_type="pdf",
            ok=False,
            artifact_path=str(pdf_path),
            errors=[f"wkhtmltopdf_failed:{(r.stderr or '')[:400]}"],
        )
    return ArtifactResult(artifact_type="pdf", ok=True, artifact_path=str(pdf_path))


def _build_blocked_notice_html(
    *,
    message: str,
    quality_report: QualityReport | None = None,
    benchmark_report: dict | None = None,
) -> str:
    parts = [
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/><title>Draft blocked</title>",
        "<style>body{font-family:sans-serif;padding:1rem;max-width:880px;margin:0 auto} "
        "table{border-collapse:collapse;width:100%} th,td{border:1px solid #ccc;padding:0.4rem;text-align:left;} "
        "code{background:#f6f8fa;padding:0.1rem 0.25rem;border-radius:4px}</style></head><body>",
        "<h1>Draft blocked</h1>",
        f"<p>{html.escape(message)}</p>",
    ]
    if quality_report is not None:
        parts.append(f"<h2>Quality Contracts</h2>{_render_contract_summary(quality_report)}")
        if quality_report.blocking_reasons:
            parts.append("<h3>Blocking reasons</h3><ul>")
            for reason in quality_report.blocking_reasons:
                parts.append(
                    f"<li><strong>{html.escape(reason.stage or 'quality')}:</strong> "
                    f"{html.escape(_humanize_reason(reason))}</li>"
                )
            parts.append("</ul>")
    if benchmark_report:
        parts.append(f"<h2>Benchmark Harness</h2>{_render_benchmark_summary(benchmark_report)}")
    parts.append("</body></html>")
    return "".join(parts)


def _write_failure_notice_pdf(
    pdf_path: Path,
    message: str,
    *,
    quality_report: QualityReport | None = None,
    benchmark_report: dict | None = None,
) -> ArtifactResult:
    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(
        _build_blocked_notice_html(
            message=message,
            quality_report=quality_report,
            benchmark_report=benchmark_report,
        ),
        encoding="utf-8",
    )
    wk = shutil.which("wkhtmltopdf")
    if not wk:
        return ArtifactResult(
            artifact_type="blocked_notice_pdf",
            ok=False,
            artifact_path=str(pdf_path),
            errors=["wkhtmltopdf_missing"],
        )
    try:
        s = subprocess.run(
            [wk, "--quiet", str(html_path), str(pdf_path)],
            capture_output=True,
            text=True,
        )
        if s.returncode == 0 and pdf_path.is_file():
            return ArtifactResult(
                artifact_type="blocked_notice_pdf",
                ok=True,
                artifact_path=str(pdf_path),
            )
    except OSError:
        pass
    return ArtifactResult(
        artifact_type="blocked_notice_pdf",
        ok=False,
        artifact_path=str(pdf_path),
        errors=["blocked_notice_pdf_failed"],
    )


def write_draft_print_pdf() -> RenderReport:
    rep = _ROOT / "reports"
    dpath = (rep / "draft_path.txt").read_text().strip() if (rep / "draft_path.txt").is_file() else ""
    if not dpath:
        return RenderReport(errors=["draft_path_missing"])
    draft_file = _ROOT / dpath
    if not draft_file.is_file():
        return RenderReport(errors=["draft_file_missing"])
    front, body = _split_frontmatter(draft_file.read_text(encoding="utf-8"))
    full_html, errors, warnings, flags = _render_full_html(front, body)
    html_artifact = ArtifactResult(
        artifact_type="html",
        ok=False,
        artifact_path=str(rep / "draft_post.rendered.html"),
    )
    pdf_artifact = ArtifactResult(
        artifact_type="pdf",
        ok=False,
        artifact_path=str(rep / "draft_post.pdf"),
    )
    if full_html is None:
        return RenderReport(
            html_valid=False,
            pdf_valid=False,
            errors=errors,
            warnings=warnings,
            artifacts=[html_artifact, pdf_artifact],
            **flags,
        )
    html_path = rep / "draft_post.rendered.html"
    html_path.write_text(full_html, encoding="utf-8")
    html_artifact = ArtifactResult(
        artifact_type="html",
        ok=not errors,
        artifact_path=str(html_path),
        errors=list(errors),
        warnings=list(warnings),
    )
    pdf_path = rep / "draft_post.pdf"
    _remove_if_exists(pdf_path)
    pdf_artifact = _render_pdf_from_html(html_path, pdf_path)
    return RenderReport(
        html_valid=not errors,
        pdf_valid=pdf_artifact.ok,
        errors=list(errors) + list(pdf_artifact.errors),
        warnings=warnings + pdf_artifact.warnings,
        artifacts=[html_artifact, pdf_artifact],
        **flags,
    )


def _render_contract_summary(qrep: QualityReport) -> str:
    rows = [
        ("evidence_valid", qrep.evidence_valid),
        ("draft_valid", qrep.draft_valid),
        ("render_valid", qrep.render_valid if qrep.render_checked else "not attempted"),
        ("package_valid", qrep.package_valid if qrep.package_checked else "not attempted"),
        ("overall_status", qrep.overall_status),
        ("pass_gate", qrep.pass_gate),
    ]
    html_rows = [
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in rows
    ]
    return "<table>" + "".join(html_rows) + "</table>"


def _render_benchmark_summary(report: dict) -> str:
    if not isinstance(report, dict) or not report:
        return "<p class='muted'>Benchmark report not available.</p>"
    total = int(report.get("total_cases", 0) or 0)
    passed = int(report.get("passed_cases", 0) or 0)
    failed = int(report.get("failed_cases", 0) or 0)
    ok = bool(report.get("ok", False))
    rows = [
        ("benchmark_ok", ok),
        ("total_cases", total),
        ("passed_cases", passed),
        ("failed_cases", failed),
    ]
    html_rows = [
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in rows
    ]
    lines = ["<table>" + "".join(html_rows) + "</table>"]
    cases = report.get("cases") or []
    failed_cases = [
        c for c in cases
        if isinstance(c, dict) and not bool(c.get("ok", False))
    ]
    if failed_cases:
        lines.append("<h3>Benchmark failures</h3><ul>")
        for case in failed_cases[:5]:
            name = str(case.get("name") or "unnamed_case")
            actual = case.get("actual") or {}
            status = str(actual.get("overall_status") or "")
            codes = ", ".join(str(x) for x in (actual.get("blocking_codes") or [])[:4])
            lines.append(
                f"<li><strong>{html.escape(name)}</strong>"
                f" <span class='muted'>({html.escape(status)})</span>"
                f"{': ' + html.escape(codes) if codes else ''}</li>"
            )
        lines.append("</ul>")
    return "".join(lines)


def _build_daily_email(
    out: Path,
    *,
    brief: dict,
    editor: dict,
    quality_report: QualityReport,
    benchmark_report: dict,
    rank: dict,
    front: dict,
    body: str,
) -> None:
    rejected = rank.get("rejected") or []
    draft_rel = (
        (_ROOT / "reports" / "draft_path.txt").read_text().strip()
        if (_ROOT / "reports" / "draft_path.txt").is_file()
        else ""
    )
    title = front.get("title") or Path(draft_rel).stem
    takeaway = front.get("one_sentence_takeaway") or ""
    show_draft_excerpt = bool(quality_report.pass_gate)
    sections = _h2_sections(body, limit=2) if show_draft_excerpt else []
    opening = _compact(_opening_prose(body), 700) if show_draft_excerpt else ""
    lead_heading = sections[0][0] if sections else ""
    lead_body = _compact(sections[0][1], 700) if sections else ""
    second_heading = sections[1][0] if len(sections) > 1 else ""
    second_body = _compact(sections[1][1], 420) if len(sections) > 1 else ""
    pr_url = ((_ROOT / "reports" / "pr_url.txt").read_text().strip() if (_ROOT / "reports" / "pr_url.txt").is_file() else "")
    branch = ((_ROOT / "reports" / "branch.txt").read_text().strip() if (_ROOT / "reports" / "branch.txt").is_file() else "")
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Blog draft</title>",
        "<style>body{font-family:system-ui,Segoe UI,sans-serif;max-width:880px;margin:2rem auto;padding:0 1rem;color:#111;}",
        "table{border-collapse:collapse;width:100%} th,td{border:1px solid #ccc;padding:0.4rem;text-align:left;}",
        ".ok{background:#dff6dd}.bad{background:#fde2e1}.muted{color:#555} code{background:#f6f8fa;padding:0.1rem 0.25rem;border-radius:4px}</style></head><body>",
        "<h1>Review blog draft</h1>",
    ]
    if pr_url:
        lines.append(f"<p><strong>PR:</strong> <a href='{html.escape(pr_url)}'>{html.escape(pr_url)}</a></p>")
    elif branch:
        lines.append(f"<p><strong>Branch pushed:</strong> <code>{html.escape(branch)}</code></p>")
    lines.append(f"<h2>Quality Contracts</h2>{_render_contract_summary(quality_report)}")
    lines.append(f"<h2>Benchmark Harness</h2>{_render_benchmark_summary(benchmark_report)}")
    if quality_report.blocking_reasons:
        lines.append("<h3>Blocking reasons</h3><ul>")
        for reason in quality_report.blocking_reasons:
            lines.append(
                f"<li><strong>{html.escape(reason.stage or 'quality')}:</strong> "
                f"{html.escape(_humanize_reason(reason))}</li>"
            )
        lines.append("</ul>")
    if title:
        lines.append(f"<h2>{html.escape(str(title))}</h2>")
    if takeaway:
        lines.append(f"<p><strong>Takeaway:</strong> {html.escape(_compact(str(takeaway), 320))}</p>")
    if draft_rel:
        lines.append(f"<p><strong>Draft file:</strong> <code>{html.escape(draft_rel)}</code></p>")
    if not show_draft_excerpt and quality_report.overall_status == "blocked":
        lines.append(
            "<p><strong>Draft excerpt withheld:</strong> the system blocked this run before packaging, "
            "so the review artifact only includes summary signals and blocking reasons.</p>"
        )
    if opening:
        lines.append(f"<p>{html.escape(opening)}</p>")
    if lead_heading:
        lines.append(f"<h3>{html.escape(lead_heading)}</h3><p>{html.escape(lead_body)}</p>")
    if second_heading:
        lines.append(f"<h3>{html.escape(second_heading)}</h3><p>{html.escape(second_body)}</p>")
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
        if editor.get("grounding_issues"):
            lines.append("<h3>Unsupported claims</h3><ul>")
            for issue in editor.get("grounding_issues")[:8]:
                lines.append(f"<li>{html.escape(str(issue))}</li>")
            lines.append("</ul>")
    if rejected:
        lines.append("<h2>Not selected</h2><ul>")
        for it in rejected[:5]:
            if isinstance(it, dict):
                title_text = html.escape(str(it.get("title", "")))
                source_text = html.escape(str(it.get("source", "")))
                lines.append(f"<li>{title_text} <span class='muted'>({source_text})</span></li>")
        lines.append("</ul>")
    lines.append("</body></html>")
    out.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    rep = _ROOT / "reports"
    out = rep / "daily_email.html"
    brief = _read_json(rep / "editorial_brief.json")
    editor_raw = _read_json(rep / "editor_report.json")
    rank = _read_json(rep / "rank_result.json")
    benchmark_report = _read_json(rep / "benchmark_report.json")
    qrep = quality.load(rep / "quality_report.json")
    if qrep is None:
        editor_obj = EditorReport.model_validate(editor_raw) if editor_raw else EditorReport()
        qrep = quality.from_editor(editor_obj)
    dpath = (rep / "draft_path.txt").read_text().strip() if (rep / "draft_path.txt").is_file() else ""
    front: dict = {}
    body = ""
    if dpath:
        draft_file = _ROOT / dpath
        if draft_file.is_file():
            front, body = _split_frontmatter(draft_file.read_text(encoding="utf-8"))

    render_report = RenderReport(
        html_valid=False,
        pdf_valid=False,
        errors=[],
        warnings=["package_skipped_due_to_quality_gate"],
    )
    package_errors: list[str] = []
    artifact_paths: dict[str, str] = {"daily_email_html": str(out)}
    draft_pdf = rep / "draft_post.pdf"
    _remove_if_exists(draft_pdf)

    if qrep.pass_gate and body:
        render_report = write_draft_print_pdf()
        for artifact in render_report.artifacts:
            if artifact.ok and artifact.artifact_path:
                artifact_paths[artifact.artifact_type] = artifact.artifact_path
    else:
        package_errors.append("prepackage_quality_blocked")
        blocked_notice = _write_failure_notice_pdf(
            rep / "draft_post_blocked_notice.pdf",
            "Draft packaging was blocked because the quality contracts did not pass. See quality_report.json.",
            quality_report=qrep,
            benchmark_report=benchmark_report,
        )
        if blocked_notice.ok and blocked_notice.artifact_path:
            artifact_paths["blocked_notice_pdf"] = blocked_notice.artifact_path
        blocked_notice_html = rep / "draft_post_blocked_notice.html"
        if blocked_notice_html.is_file():
            artifact_paths["blocked_notice_html"] = str(blocked_notice_html)

    package_valid = bool(qrep.pass_gate and render_report.ok and out.parent.exists())
    merged = quality.with_render(
        qrep,
        render_report,
        package_valid=package_valid,
        render_checked=bool(qrep.pass_gate and body),
        package_checked=bool(qrep.pass_gate and body),
        artifact_paths=artifact_paths,
        package_errors=package_errors,
    )
    if not merged.pass_gate:
        _remove_if_exists(draft_pdf)
    _build_daily_email(
        out,
        brief=brief,
        editor=editor_raw,
        quality_report=merged,
        benchmark_report=benchmark_report,
        rank=rank,
        front=front,
        body=body,
    )
    merged = quality.with_render(
        qrep,
        render_report,
        package_valid=package_valid and out.is_file(),
        render_checked=bool(qrep.pass_gate and body),
        package_checked=bool(qrep.pass_gate and body),
        artifact_paths={**artifact_paths, "daily_email_html": str(out)},
        package_errors=package_errors,
    )
    if not merged.pass_gate:
        _remove_if_exists(draft_pdf)
    quality.save(rep / "quality_report.json", merged)
    (rep / "package_result.json").write_text(
        json.dumps(
            {
                "ok": merged.pass_gate,
                "overall_status": merged.overall_status,
                "artifact_paths": merged.artifact_paths,
                "render_errors": merged.render_report.errors if merged.render_report else [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "ok": merged.pass_gate,
        "overall_status": merged.overall_status,
        "artifact_paths": merged.artifact_paths,
    }

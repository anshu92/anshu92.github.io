from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
os.environ.setdefault("BLOGPIPE_REPO_ROOT", os.path.join(os.path.dirname(__file__), ".."))


def test_quality_report_from_editor_blocks_grounding_contradictions() -> None:
    from blogpipe.models import EditorReport
    from blogpipe.quality import from_editor

    rep = EditorReport(
        rubric_score=10,
        five_questions_ok=True,
        llm_ok=True,
        grounding_ok=False,
        grounding_issues=["invented score"],
    )
    qrep = from_editor(rep)
    assert not qrep.evidence_valid
    assert not qrep.pass_gate
    assert qrep.overall_status == "blocked"


def test_quality_report_from_editor_does_not_block_advisory_grounding_details() -> None:
    from blogpipe.models import EditorReport
    from blogpipe.quality import from_editor

    rep = EditorReport(
        rubric_score=10,
        five_questions_ok=True,
        llm_ok=True,
        grounding_ok=False,
        grounding_issues=["seed lists", "epsilon tuning details", "average three seeds"],
    )
    qrep = from_editor(rep)
    assert qrep.evidence_valid
    assert qrep.pass_gate
    assert qrep.overall_status == "passed"


def test_validate_rendered_html_flags_raw_mermaid_and_raw_table() -> None:
    from blogpipe.package import _validate_rendered_html

    body = (
        "Takeaway 1.0%.\n\n"
        "## Why this works\n```mermaid\nflowchart LR\nA --> B\n```\n\n"
        "## Should you use it\n| Method | Metric | Baseline |\n| --- | --- | --- |\n| X | 1% | Y |\n"
    )
    html = "<html><body><pre><code class='language-mermaid'>flowchart LR\nA --> B</code></pre>| Method | Metric | Baseline |</body></html>"
    errors, _, flags = _validate_rendered_html(body, html)
    assert "raw_mermaid_in_render" in errors
    assert "results_table_not_rendered" in errors
    assert not flags["mermaid_rendered"]


def test_package_run_blocks_and_removes_success_pdf_name_when_quality_fails(tmp_path: Path, monkeypatch) -> None:
    from blogpipe import package
    from blogpipe.models import ArtifactResult
    from blogpipe.models import EditorReport
    from blogpipe.quality import from_editor, save

    monkeypatch.setattr(package, "_ROOT", tmp_path)
    monkeypatch.setattr(
        package,
        "_render_full_html",
        lambda front, body: (
            f"<html><body><article><h1>{front.get('title', '')}</h1><p>{body}</p></article></body></html>",
            [],
            [],
            {
                "mermaid_rendered": True,
                "tables_rendered": True,
                "images_resolved": True,
                "captions_ok": True,
                "density_ok": True,
            },
        ),
    )
    monkeypatch.setattr(
        package,
        "_render_pdf_from_html",
        lambda html_path, pdf_path: (
            pdf_path.write_bytes(b"%PDF-1.4 rejected draft"),
            ArtifactResult(
                artifact_type="pdf" if pdf_path.name == "draft_post.pdf" else "rejected_pdf",
                ok=True,
                artifact_path=str(pdf_path),
            ),
        )[1],
    )
    reports = tmp_path / "reports"
    content = tmp_path / "content" / "post"
    reports.mkdir(parents=True)
    content.mkdir(parents=True)
    draft = content / "draft.md"
    draft.write_text(
        "---\n"
        'title: "Test"\n'
        'one_sentence_takeaway: "Takeaway 1.0%."\n'
        "---\n\n"
        "Takeaway 1.0%.\n\n## Why this works\nBody.\n\n## Should you use it\nNo.\n",
        encoding="utf-8",
    )
    (reports / "draft_path.txt").write_text("content/post/draft.md", encoding="utf-8")
    (reports / "editor_report.json").write_text(
        EditorReport(
            rubric_score=10,
            five_questions_ok=True,
            llm_ok=True,
            grounding_ok=False,
            grounding_issues=["unsupported claim"],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    save(
        reports / "quality_report.json",
        from_editor(
            EditorReport(
                rubric_score=10,
                five_questions_ok=True,
                llm_ok=True,
                grounding_ok=False,
                grounding_issues=["unsupported claim"],
            )
        ),
    )
    (reports / "benchmark_report.json").write_text(
        json.dumps(
            {
                "ok": False,
                "total_cases": 4,
                "passed_cases": 3,
                "failed_cases": 1,
                "cases": [
                    {
                        "name": "blocked_render_raw_mermaid",
                        "ok": False,
                        "actual": {
                            "overall_status": "blocked",
                            "blocking_codes": ["render_error"],
                        },
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    stale_pdf = reports / "draft_post.pdf"
    stale_pdf.write_bytes(b"stale")

    result = package.run()

    assert result["ok"] is False
    assert not stale_pdf.exists()
    assert (reports / "quality_report.json").is_file()
    payload = json.loads((reports / "package_result.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert (reports / "daily_email.html").is_file()
    assert (reports / "draft_post_rejected.html").is_file()
    email_html = (reports / "daily_email.html").read_text(encoding="utf-8")
    rejected_html = (reports / "draft_post_rejected.html").read_text(encoding="utf-8")
    assert "Draft excerpt withheld" in email_html
    assert "Body." not in email_html
    assert "render_valid</th><td>not attempted" in email_html
    assert "package_valid</th><td>not attempted" in email_html
    assert "Unsupported claim: unsupported claim" in email_html
    assert "Benchmark Harness" in email_html
    assert "blocked_render_raw_mermaid" in email_html
    assert "Quality Contracts" in rejected_html
    assert "Benchmark Harness" in rejected_html
    assert "blocked_render_raw_mermaid" in rejected_html
    assert "Rejected draft review copy" in rejected_html
    assert "Body." in rejected_html


def test_humanize_reason_shows_specific_grounding_claim() -> None:
    from blogpipe.models import FailureReason
    from blogpipe.package import _humanize_reason

    reason = FailureReason(code="grounding_issue", message="invented 85% benchmark score", stage="editor")
    assert _humanize_reason(reason) == "Unsupported claim: invented 85% benchmark score"


def test_write_draft_print_pdf_requires_rendered_mermaid(monkeypatch, tmp_path: Path) -> None:
    from blogpipe import package

    monkeypatch.setattr(package, "_ROOT", tmp_path)
    reports = tmp_path / "reports"
    content = tmp_path / "content" / "post"
    reports.mkdir(parents=True)
    content.mkdir(parents=True)
    draft = content / "draft.md"
    draft.write_text(
        "---\n"
        'title: "Test"\n'
        "---\n\n"
        "Takeaway 1.0%.\n\n## Why this works\n```mermaid\nflowchart LR\nA --> B\n```\n\n## Should you use it\n"
        "| Method | Metric | Baseline |\n| --- | --- | --- |\n| X | 1% | Y |\n",
        encoding="utf-8",
    )
    (reports / "draft_path.txt").write_text("content/post/draft.md", encoding="utf-8")

    monkeypatch.setattr(package, "_pandoc_md_to_html_fragment", lambda _: "<pre><code class='language-mermaid'>flowchart LR</code></pre>")
    monkeypatch.setattr(package.visuals, "_kroki_svg", lambda *_args, **_kwargs: None)

    render = package.write_draft_print_pdf()

    assert not render.html_valid
    assert not render.pdf_valid
    assert "raw_mermaid_in_render" in render.errors


def test_render_full_html_falls_back_without_pandoc_or_kroki(monkeypatch) -> None:
    from blogpipe import package

    monkeypatch.setattr(package, "_pandoc_md_to_html_fragment", lambda _body: None)
    monkeypatch.setattr(package.visuals, "_kroki_svg", lambda *_args, **_kwargs: None)

    body = (
        "Takeaway 4 metrics.\n\n"
        "## Why this works\n"
        "```mermaid\nflowchart LR\nA --> B\n```\n\n"
        "## Numbers the paper actually gives us\n"
        "| Method | Metric | Baseline |\n| --- | --- | --- |\n| X | 1% | Y |\n"
    )
    html_text, errors, warnings, flags = package._render_full_html({"title": "Test"}, body)

    assert html_text is not None
    assert "raw_mermaid_in_render" not in errors
    assert "results_table_not_rendered" not in errors
    assert "pandoc_render_fallback_used" in warnings
    assert flags["tables_rendered"]

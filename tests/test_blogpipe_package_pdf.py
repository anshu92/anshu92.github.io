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
    from blogpipe.models import EditorReport
    from blogpipe.quality import from_editor, save

    monkeypatch.setattr(package, "_ROOT", tmp_path)
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
    assert (reports / "draft_post_blocked_notice.html").is_file()
    email_html = (reports / "daily_email.html").read_text(encoding="utf-8")
    blocked_html = (reports / "draft_post_blocked_notice.html").read_text(encoding="utf-8")
    assert "Draft excerpt withheld" in email_html
    assert "Body." not in email_html
    assert "render_valid</th><td>not attempted" in email_html
    assert "package_valid</th><td>not attempted" in email_html
    assert "Unsupported claim" in email_html
    assert "Benchmark Harness" in email_html
    assert "blocked_render_raw_mermaid" in email_html
    assert "Quality Contracts" in blocked_html
    assert "Benchmark Harness" in blocked_html
    assert "blocked_render_raw_mermaid" in blocked_html


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
    assert "raw_mermaid_in_render" in render.errors or "mermaid_render_failed" in render.errors

from __future__ import annotations

from pathlib import Path


def test_daily_workflow_attaches_blocked_notice_pdf_when_draft_pdf_missing() -> None:
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "daily-blog-draft.yml"
    ).read_text(encoding="utf-8")

    assert "Compute email attachments" in workflow
    assert "reports/draft_post_blocked_notice.pdf" in workflow
    assert "steps.mail_attachments.outputs.attachments" in workflow

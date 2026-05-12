from __future__ import annotations

from pathlib import Path


def test_research_radar_pr_creation_has_fallback_link():
    workflow = Path(".github/workflows/research-radar.yml").read_text()
    assert "secrets.PR_CREATION_TOKEN || secrets.GITHUB_TOKEN" in workflow
    assert "FALLBACK_PR_URL=\"https://github.com/${GITHUB_REPOSITORY}/pull/new/${BR}\"" in workflow
    assert "GitHub CLI could not create a PR" in workflow
    assert "Open or create the draft PR" in workflow

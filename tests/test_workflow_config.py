from __future__ import annotations

from pathlib import Path


def test_research_radar_pr_creation_has_fallback_link():
    workflow = Path(".github/workflows/research-radar.yml").read_text()
    assert "secrets.PR_CREATION_TOKEN || secrets.GITHUB_TOKEN" in workflow
    assert "FALLBACK_PR_URL=\"https://github.com/${GITHUB_REPOSITORY}/pull/new/${BR}\"" in workflow
    assert "GitHub CLI could not create a PR" in workflow
    assert "Open or create the draft PR" in workflow


def test_research_radar_pr_requires_content_post_change():
    workflow = Path(".github/workflows/research-radar.yml").read_text()
    assert 'POST_CHANGES="$(git status --porcelain content/post)"' in workflow
    assert 'if [[ -n "$POST_CHANGES" ]]; then' in workflow
    assert "Generated radar data/assets changed, but no content/post draft was created. Skipping PR." in workflow


def test_research_radar_workflow_caps_runtime():
    workflow = Path(".github/workflows/research-radar.yml").read_text()
    assert "timeout-minutes: 20" in workflow
    assert "BLOGPIPE_LLM_MAX_RUNTIME_SECONDS: ${{ vars.BLOGPIPE_LLM_MAX_RUNTIME_SECONDS || '1200' }}" in workflow
    assert "BLOGPIPE_SECTIONWISE_DRAFTING: ${{ vars.BLOGPIPE_SECTIONWISE_DRAFTING || '0' }}" in workflow

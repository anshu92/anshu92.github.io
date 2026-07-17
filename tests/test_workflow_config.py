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


def test_research_radar_workflow_uses_swarm_with_openai_defaults():
    workflow = Path(".github/workflows/research-radar.yml").read_text()
    assert "timeout-minutes: 30" in workflow
    assert "python -m blogpipe swarm run --window 14d" in workflow
    assert "BLOGPIPE_LLM_BASE_URL: ${{ vars.BLOGPIPE_LLM_BASE_URL || 'https://api.openai.com/v1' }}" in workflow
    assert "BLOGPIPE_LLM_API_KEY: ${{ secrets.BLOGPIPE_LLM_API_KEY || secrets.OPENAI_API_KEY }}" in workflow
    assert "BLOGPIPE_LLM_MODEL_FAST: ${{ vars.BLOGPIPE_LLM_MODEL_FAST || 'gpt-5.6-luna' }}" in workflow
    assert "BLOGPIPE_LLM_MODEL_SMART: ${{ vars.BLOGPIPE_LLM_MODEL_SMART || 'gpt-5.6-terra' }}" in workflow
    assert "BLOGPIPE_LLM_MODEL_TECHNICAL_EXPLAINER" in workflow
    assert "BLOGPIPE_LLM_MODEL_IMPLEMENTATION_ENGINEER" in workflow
    assert "BLOGPIPE_LLM_MODEL_MANAGING_EDITOR" in workflow
    assert "BLOGPIPE_ROLE_MARKET_LIVE" in workflow
    assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" in workflow


def test_research_radar_workflow_removes_legacy_generation_knobs():
    workflow = Path(".github/workflows/research-radar.yml").read_text()
    assert "python -m blogpipe run" not in workflow
    assert "BLOGPIPE_LLM_MODEL_SELECTOR" not in workflow
    assert "BLOGPIPE_LLM_MODEL_OUTLINE" not in workflow
    assert "BLOGPIPE_AGENT_DEEP_DIVE_MIN_BUDGET_SECONDS" not in workflow
    assert "BLOGPIPE_OPENROUTER_DYNAMIC_FREE_MODELS" not in workflow
    assert "OPENROUTER_API_KEY" not in workflow

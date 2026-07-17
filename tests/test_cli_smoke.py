from __future__ import annotations

import json
from pathlib import Path

import pytest

from blogpipe import memory
from blogpipe.cli import main
from blogpipe.models import ComponentSpec, RoleMarketReport, TableSpec
from blogpipe.swarm import CatalogueEditor, RoleMarketScout, seed_sources_for_lesson
from blogpipe.visuals import validate_component, validate_table


def test_swarm_run_with_fixtures_writes_preview_and_reports(monkeypatch, tmp_path):
    _patch_memory(monkeypatch, tmp_path)
    _disable_live_llm(monkeypatch)
    lesson = CatalogueEditor().choose(RoleMarketReport())
    fixtures = _write_fixture_items(tmp_path, seed_sources_for_lesson(lesson))

    code = main(["swarm", "run", "--fixtures", str(fixtures), "--dry-run", "--db", str(tmp_path / "items.sqlite")])

    assert code == 0
    report_dir = tmp_path / "reports" / "swarm"
    assert (report_dir / "run_report.json").is_file()
    assert (report_dir / "role_market_signal.json").is_file()
    assert (report_dir / "catalogue_decision.json").is_file()
    assert (report_dir / "lesson_brief.json").is_file()
    assert (report_dir / "visual_plan.json").is_file()
    previews = list(report_dir.glob("*.preview.md"))
    assert len(previews) == 1
    preview = previews[0].read_text(encoding="utf-8")
    assert "draft: true" in preview
    assert "mermaid: true" in preview
    assert "```mermaid" in preview
    assert "Which matmul path answers which engineering question?" in preview
    assert "matmul-output-cell.svg" in preview
    assert "blogpipe-callout" in preview
    assert list((tmp_path / "static" / "img" / "posts").glob("**/*.svg"))

    report = json.loads((report_dir / "run_report.json").read_text(encoding="utf-8"))
    assert report["final_article"]["ok"] is True
    assert report["selected_lesson"]["id"] == "matmul-from-scalar-operations"
    assert len(set(report["lesson_brief"]["source_urls"])) >= 2
    assert any("deeplearningbook.org" in url for url in report["lesson_brief"]["source_urls"])
    assert [item["agent_name"] for item in report["agent_reports"]][:4] == [
        "RoleMarketScout",
        "CatalogueEditor",
        "ResearchLead",
        "TechnicalExplainer",
    ]


def test_swarm_blocks_when_evidence_is_insufficient(monkeypatch, tmp_path):
    _patch_memory(monkeypatch, tmp_path)
    _disable_live_llm(monkeypatch)
    monkeypatch.setenv("BLOGPIPE_CATALOGUE_LESSON", "distributed-training-parallelism")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "source_items.json").write_text('{"items": []}', encoding="utf-8")

    code = main(["swarm", "run", "--fixtures", str(fixtures), "--dry-run", "--db", str(tmp_path / "items.sqlite")])

    assert code == 0
    report_dir = tmp_path / "reports" / "swarm"
    blocked = list(report_dir.glob("*.blocked.json"))
    assert blocked
    report = json.loads((report_dir / "run_report.json").read_text(encoding="utf-8"))
    assert report["final_article"]["ok"] is False
    assert "insufficient_evidence:0/2" in report["final_article"]["errors"]
    assert not list((tmp_path / "content" / "post").glob("*.md"))


def test_legacy_run_command_is_removed():
    with pytest.raises(SystemExit):
        main(["run"])


def test_role_market_scout_fixture_filters_staff_principal_engineering_roles(monkeypatch, tmp_path):
    postings = {
        "postings": [
            {
                "company": "OpenAI",
                "title": "Staff Machine Learning Engineer, Inference Systems",
                "url": "https://openai.com/careers/staff-inference",
                "description": "Lead inference serving, latency, throughput, cache, evaluation, and reliability work.",
            },
            {
                "company": "Anthropic",
                "title": "Junior Data Analyst",
                "url": "https://anthropic.com/careers/junior",
                "description": "Dashboard reporting and analytics.",
            },
            {
                "company": "NVIDIA",
                "title": "Principal Product Manager",
                "url": "https://nvidia.com/careers/product",
                "description": "Roadmap planning for developer products.",
            },
            {
                "company": "Google DeepMind",
                "title": "Principal Research Engineer, Distributed Training",
                "url": "https://deepmind.google/careers/principal-training",
                "description": "Distributed training, sharding, FSDP, checkpoint recovery, NCCL, and benchmark design.",
            },
            {
                "company": "Anthropic",
                "title": "Careers at Anthropic",
                "url": "https://anthropic.com/careers",
                "description": "Staff work on agents, tools, AI, infrastructure, and research across the company.",
            },
        ]
    }
    fixture = tmp_path / "roles.json"
    fixture.write_text(json.dumps(postings), encoding="utf-8")
    monkeypatch.setenv("BLOGPIPE_ROLE_MARKET_FIXTURES", str(fixture))

    report = RoleMarketScout().run()

    assert [signal.company for signal in report.signals] == ["OpenAI", "Google DeepMind"]
    assert "inference-serving" in report.signals[0].topic_tags
    assert "distributed-training" in report.signals[1].topic_tags
    assert report.signals[0].source_url


def test_role_market_scout_parses_job_cards_without_landing_page_noise():
    html = """
    <html><body>
      <h1>Careers</h1>
      <p>Our staff build AI systems, agents, infrastructure, and tools.</p>
      <a href="/careers/staff-inference">Staff Machine Learning Engineer, Inference Systems</a>
      <p>Lead inference serving, latency, throughput, cache, evaluation, and reliability work.</p>
    </body></html>
    """

    signals = RoleMarketScout()._signals_from_text("https://openai.com/careers/search", html)

    assert len(signals) == 1
    assert signals[0].role_title == "Staff Machine Learning Engineer, Inference Systems"
    assert signals[0].source_url == "https://openai.com/careers/staff-inference"


def test_catalogue_editor_respects_prerequisites(monkeypatch, tmp_path):
    _patch_memory(monkeypatch, tmp_path)
    memory.ensure_dirs()
    (tmp_path / "radar-data" / "catalogue_state.json").write_text(
        json.dumps({"completed_lesson_ids": ["matmul-from-scalar-operations"]}),
        encoding="utf-8",
    )

    lesson = CatalogueEditor().choose(RoleMarketReport(recurring_topics=["distributed-training"]))

    assert lesson.id == "matrix-shapes-and-layout"
    assert "distributed-training" in lesson.market_rationale


def test_visual_component_sanitizer_rejects_unsafe_html():
    component = ComponentSpec(
        component_id="bad",
        kind="callout",
        title="Bad",
        purpose="Should be rejected",
        html='<div onclick="alert(1)"><script>alert(1)</script></div>',
        evidence_ids=["E1"],
    )

    errors = validate_component(component, evidence_ids={"E1"})

    assert "unsafe_component_html:bad" in errors


def test_table_designer_rejects_numeric_claims_without_evidence():
    table = TableSpec(
        table_id="numeric",
        title="Unsupported numbers",
        purpose="Catch invented metrics",
        headers=["Metric", "Value"],
        rows=[["Latency", "20% faster"]],
        evidence_ids=[],
    )

    errors = validate_table(table, evidence_ids={"E1"})

    assert "table_numeric_claim_without_evidence:numeric" in errors


def _patch_memory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(memory, "ROOT", tmp_path)
    monkeypatch.setattr(memory, "DATA", tmp_path / "radar-data")
    monkeypatch.setattr(memory, "DAILY_DATA", tmp_path / "radar-data" / "daily")
    monkeypatch.setattr(memory, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(memory, "CONTENT_POST", tmp_path / "content" / "post")
    monkeypatch.setattr(memory, "STATIC_POSTS", tmp_path / "static" / "img" / "posts")
    monkeypatch.setenv("BLOGPIPE_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("BLOGPIPE_ROLE_MARKET_LIVE", "0")


def _disable_live_llm(monkeypatch) -> None:
    monkeypatch.setenv("BLOGPIPE_LLM_MAX_CALLS", "0")
    monkeypatch.delenv("BLOGPIPE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSE", raising=False)
    monkeypatch.delenv("BLOGPIPE_FAKE_LLM_RESPONSES", raising=False)


def _write_fixture_items(tmp_path: Path, items) -> Path:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "source_items.json").write_text(
        json.dumps({"items": [item.model_dump(mode="json") for item in items]}, indent=2),
        encoding="utf-8",
    )
    return fixtures

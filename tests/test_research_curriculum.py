from __future__ import annotations

import json
from pathlib import Path

import yaml

from blogpipe import research_curriculum
from blogpipe.cli import main


ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_PATH = ROOT / "data" / "research_curriculum.yaml"


def test_canonical_curriculum_covers_52_weeks_and_14_reports():
    curriculum = research_curriculum.load_curriculum(CURRICULUM_PATH)

    assert curriculum.module_count == 12
    assert curriculum.report_count == 14
    assert [module.number for module in curriculum.modules] == list(range(1, 13))
    assert sum(len(module.reports) for module in curriculum.modules) == 14
    assert len(curriculum.modules[-1].reports) == 3
    assert [module.id for module in curriculum.modules if module.current] == ["module-01-loss-semantics"]
    assert research_curriculum.validate_curriculum(curriculum) == []


def test_curriculum_prerequisites_only_point_backward():
    curriculum = research_curriculum.load_curriculum(CURRICULUM_PATH)
    seen: set[str] = set()

    for module in curriculum.modules:
        assert set(module.prerequisites) <= seen
        seen.add(module.id)


def test_module_one_skeleton_is_valid_and_claims_no_result():
    path = ROOT / "content" / "post" / "the-two-numbers-behind-every-language-model-loss" / "index.md"
    raw = path.read_text(encoding="utf-8")

    assert 'result_status: "planned"' in raw
    assert 'result: "not-run"' in raw
    assert 'evidence_level: "not-started"' in raw
    assert "## Results\n\n<!-- Intentionally empty" in raw
    assert research_curriculum.validate_report("module-01-loss-semantics", root=ROOT) == []


def test_prepare_report_dry_run_does_not_write(monkeypatch, tmp_path):
    _copy_curriculum(tmp_path)
    monkeypatch.setenv("BLOGPIPE_REPO_ROOT", str(tmp_path))

    output = research_curriculum.prepare_report("module-02-tokenizer-stream", root=tmp_path, dry_run=True)

    assert output == tmp_path / "content" / "post" / "a-tokenizer-is-a-compute-allocation-policy" / "index.md"
    assert not output.exists()


def test_prepare_report_writes_once_and_never_overwrites(tmp_path):
    _copy_curriculum(tmp_path)

    output = research_curriculum.prepare_report("module-02-tokenizer-stream", root=tmp_path)

    assert output.is_file()
    original = output.read_text(encoding="utf-8")
    assert "## Directional hypothesis" in original
    assert "not-run" in original
    assert research_curriculum.validate_report("module-02-tokenizer-stream", root=tmp_path) == []

    try:
        research_curriculum.prepare_report("module-02-tokenizer-stream", root=tmp_path)
    except FileExistsError as exc:
        assert "research_report_exists" in str(exc)
    else:
        raise AssertionError("prepare_report overwrote an existing learning artifact")


def test_published_report_requires_evidence_artifacts(tmp_path):
    _copy_curriculum(tmp_path)
    output = research_curriculum.prepare_report("module-02-tokenizer-stream", root=tmp_path)
    curriculum_file = tmp_path / "data" / "research_curriculum.yaml"
    curriculum = yaml.safe_load(curriculum_file.read_text(encoding="utf-8"))
    curriculum["modules"][1]["reports"][0]["status"] = "published"
    curriculum_file.write_text(yaml.safe_dump(curriculum, sort_keys=False), encoding="utf-8")
    raw = output.read_text(encoding="utf-8")
    raw = raw.replace("result_status: planned", "result_status: published")
    raw = raw.replace("draft: true", "draft: false")
    output.write_text(raw, encoding="utf-8")

    errors = research_curriculum.validate_report("module-02-tokenizer-stream", root=tmp_path)

    assert "published_report_without_result:report-02-tokenizer-stream" in errors
    assert "published_report_without_evidence:report-02-tokenizer-stream" in errors
    assert "published_report_missing:report-02-tokenizer-stream:artifact_manifest" in errors
    assert "published_report_contains_placeholders:report-02-tokenizer-stream" in errors


def test_curriculum_status_cli_is_non_generative(monkeypatch, capsys):
    monkeypatch.setenv("BLOGPIPE_REPO_ROOT", str(ROOT))

    code = main(["curriculum", "status"])

    assert code == 0
    output = capsys.readouterr().out
    assert '"current_module": "module-01-loss-semantics"' in output
    assert '"published": 0' in output
    assert '"total": 14' in output


def test_site_navigation_exposes_curriculum_surfaces():
    config = (ROOT / "hugo.toml").read_text(encoding="utf-8")
    for label in ("Start Here", "Curriculum", "Research Reports", "Labs", "Artifacts", "Reading Map", "About"):
        assert f"name = '{label}'" in config


def test_every_report_has_unique_hugo_identity():
    payload = yaml.safe_load(CURRICULUM_PATH.read_text(encoding="utf-8"))
    reports = [report for module in payload["modules"] for report in module["reports"]]

    assert len({report["id"] for report in reports}) == len(reports)
    assert len({report["slug"] for report in reports}) == len(reports)


def test_public_evidence_schemas_are_valid_json_and_require_lineage():
    run_schema = json.loads((ROOT / "static" / "schemas" / "research-run-manifest.schema.json").read_text())
    result_schema = json.loads((ROOT / "static" / "schemas" / "research-result-summary.schema.json").read_text())

    assert {"module_id", "run_id", "code", "configuration", "data", "compute", "outputs"} <= set(run_schema["required"])
    assert {"module_id", "report_id", "result", "evidence_level", "run_manifests", "claim_boundary"} <= set(
        result_schema["required"]
    )


def _copy_curriculum(root: Path) -> None:
    target = root / "data"
    target.mkdir(parents=True)
    (target / "research_curriculum.yaml").write_text(CURRICULUM_PATH.read_text(encoding="utf-8"), encoding="utf-8")

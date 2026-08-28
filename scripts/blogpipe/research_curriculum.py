from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from . import memory


WorkflowStatus = Literal["planned", "preregistered", "running", "published", "deferred"]
EvidenceLevel = Literal["not-started", "derived", "implemented", "validated", "scaled", "transferred"]

REQUIRED_REPORT_HEADINGS = (
    "Research question",
    "Directional hypothesis",
    "Governing contract",
    "Trusted baseline",
    "Intervention and negative control",
    "Run design",
    "Results",
    "Failure analysis",
    "Evidence boundary",
    "Scale bridge",
    "Reproduce the artifact",
)


class ResearchReport(BaseModel):
    id: str
    title: str
    slug: str
    status: WorkflowStatus = "planned"
    evidence_level: EvidenceLevel = "not-started"
    publish_by: str = ""


class ResearchModule(BaseModel):
    id: str
    number: int
    phase: str
    weeks: str
    starts: str
    publish_by: str
    status: WorkflowStatus = "planned"
    current: bool = False
    evidence_level: EvidenceLevel = "not-started"
    title: str
    research_question: str
    prerequisites: list[str] = Field(default_factory=list)
    competencies: list[str] = Field(default_factory=list)
    implementation_target: str
    publication_gate: str
    code_tag: str
    resources: list[str] = Field(default_factory=list)
    reports: list[ResearchReport]


class ResearchCurriculum(BaseModel):
    version: str
    title: str
    subtitle: str
    research_baseline: str
    start_date: str
    target_end_date: str
    duration_weeks: int
    module_count: int
    report_count: int
    weekly_hours: str
    code_repository: dict[str, str]
    workflow: list[dict[str, str]]
    evidence_levels: list[dict[str, str]]
    result_states: list[str]
    reading_stack: list[dict[str, str]]
    modules: list[ResearchModule]


def curriculum_path(root: Path | None = None) -> Path:
    return (root or memory.repo_root()) / "data" / "research_curriculum.yaml"


def load_curriculum(path: str | Path | None = None) -> ResearchCurriculum:
    source = Path(path) if path else curriculum_path()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    curriculum = ResearchCurriculum.model_validate(payload)
    errors = validate_curriculum(curriculum)
    if errors:
        raise ValueError("research_curriculum_invalid:" + ";".join(errors))
    return curriculum


def validate_curriculum(curriculum: ResearchCurriculum) -> list[str]:
    errors: list[str] = []
    modules = curriculum.modules
    if curriculum.duration_weeks != 52:
        errors.append(f"duration_weeks:{curriculum.duration_weeks}/52")
    if curriculum.module_count != len(modules):
        errors.append(f"module_count:{curriculum.module_count}/{len(modules)}")

    module_ids = [module.id for module in modules]
    if len(module_ids) != len(set(module_ids)):
        errors.append("duplicate_module_id")
    expected_numbers = list(range(1, len(modules) + 1))
    if [module.number for module in modules] != expected_numbers:
        errors.append("module_numbers_not_contiguous")

    covered_weeks: list[int] = []
    seen: set[str] = set()
    report_ids: list[str] = []
    report_slugs: list[str] = []
    current_count = 0
    resource_ids = {str(resource.get("id", "")) for resource in curriculum.reading_stack}
    evidence_ids = {str(level.get("id", "")) for level in curriculum.evidence_levels}

    for module in modules:
        current_count += int(module.current)
        covered_weeks.extend(_week_range(module.weeks, errors, module.id))
        missing_prerequisites = sorted(set(module.prerequisites) - seen)
        if missing_prerequisites:
            errors.append(f"unsafe_prerequisites:{module.id}:{','.join(missing_prerequisites)}")
        missing_resources = sorted(set(module.resources) - resource_ids)
        if missing_resources:
            errors.append(f"unknown_resources:{module.id}:{','.join(missing_resources)}")
        if module.evidence_level not in evidence_ids:
            errors.append(f"unknown_evidence_level:{module.id}:{module.evidence_level}")
        if not module.research_question.rstrip().endswith("?"):
            errors.append(f"research_question_not_question:{module.id}")
        if module.code_tag != module.id:
            errors.append(f"code_tag_mismatch:{module.id}:{module.code_tag}")
        if not module.reports:
            errors.append(f"module_without_report:{module.id}")
        for report in module.reports:
            report_ids.append(report.id)
            report_slugs.append(report.slug)
            if report.evidence_level not in evidence_ids:
                errors.append(f"unknown_report_evidence_level:{report.id}:{report.evidence_level}")
        seen.add(module.id)

    if current_count != 1:
        errors.append(f"current_module_count:{current_count}/1")
    if sorted(covered_weeks) != list(range(1, 53)):
        errors.append("weeks_do_not_cover_1_through_52_once")
    if curriculum.report_count != len(report_ids):
        errors.append(f"report_count:{curriculum.report_count}/{len(report_ids)}")
    if len(report_ids) != len(set(report_ids)):
        errors.append("duplicate_report_id")
    if len(report_slugs) != len(set(report_slugs)):
        errors.append("duplicate_report_slug")
    capstone = next((module for module in modules if module.number == 12), None)
    if capstone is None or len(capstone.reports) != 3:
        errors.append("capstone_report_count_not_three")
    return errors


def status_payload(curriculum: ResearchCurriculum | None = None) -> dict[str, object]:
    curriculum = curriculum or load_curriculum()
    reports = [report for module in curriculum.modules for report in module.reports]
    current = next(module for module in curriculum.modules if module.current)
    return {
        "version": curriculum.version,
        "current_module": current.id,
        "modules": {
            "published": sum(module.status == "published" for module in curriculum.modules),
            "total": len(curriculum.modules),
        },
        "reports": {
            "published": sum(report.status == "published" for report in reports),
            "total": len(reports),
        },
        "evidence": {module.id: module.evidence_level for module in curriculum.modules},
    }


def print_status(curriculum: ResearchCurriculum | None = None) -> None:
    print(json.dumps(status_payload(curriculum), indent=2, sort_keys=True))


def prepare_report(
    module_id: str,
    *,
    report_id: str = "",
    root: Path | None = None,
    dry_run: bool = False,
) -> Path:
    root = root or memory.repo_root()
    curriculum = load_curriculum(curriculum_path(root))
    module = _module_by_id(curriculum, module_id)
    if report_id:
        report = next((candidate for candidate in module.reports if candidate.id == report_id), None)
        if report is None:
            raise ValueError(f"unknown_research_report:{module_id}:{report_id}")
    elif len(module.reports) == 1:
        report = module.reports[0]
    else:
        raise ValueError(f"report_id_required:{module_id}")

    output = root / "content" / "post" / report.slug / "index.md"
    if output.exists():
        raise FileExistsError(f"research_report_exists:{output}")
    content = render_report_skeleton(module, report)
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=False)
        output.write_text(content, encoding="utf-8")
    return output


def render_report_skeleton(module: ResearchModule, report: ResearchReport) -> str:
    front_matter = {
        "title": report.title,
        "description": f"A planned evidence-bearing report for {module.title.lower()}.",
        "date": report.publish_by or module.publish_by,
        "draft": True,
        "slug": report.slug,
        "author": "Anshuman Sahoo",
        "content_type": "research-report",
        "module_id": module.id,
        "report_id": report.id,
        "result_status": "planned",
        "result": "not-run",
        "evidence_level": "not-started",
        "research_question": module.research_question,
        "directional_hypothesis": "",
        "competencies": module.competencies,
        "prerequisites": module.prerequisites,
        "code_tag": module.code_tag,
        "artifact_manifest": "",
        "raw_results": "",
        "reproduction_command": "",
        "compute_scale": "Not run",
        "tags": [],
        "categories": [],
        "math": True,
        "mermaid": True,
    }
    rendered_front_matter = yaml.safe_dump(front_matter, sort_keys=False, allow_unicode=True).strip()
    sections = [
        "> **Evidence status:** Planned. This skeleton makes no implementation, measurement, or reproduction claim.",
        "",
    ]
    for heading in REQUIRED_REPORT_HEADINGS:
        sections.extend([f"## {heading}", "", _placeholder_for(heading, module), ""])
    return f"---\n{rendered_front_matter}\n---\n\n" + "\n".join(sections).rstrip() + "\n"


def validate_report(
    module_id: str,
    *,
    report_id: str = "",
    root: Path | None = None,
) -> list[str]:
    root = root or memory.repo_root()
    curriculum = load_curriculum(curriculum_path(root))
    module = _module_by_id(curriculum, module_id)
    reports = [report for report in module.reports if not report_id or report.id == report_id]
    if not reports:
        raise ValueError(f"unknown_research_report:{module_id}:{report_id}")

    errors: list[str] = []
    for report in reports:
        path = root / "content" / "post" / report.slug / "index.md"
        if not path.is_file():
            errors.append(f"missing_report_file:{report.id}:{path}")
            continue
        front_matter, body = _read_markdown(path)
        errors.extend(_validate_report_document(module, report, front_matter, body))
    return errors


def _validate_report_document(
    module: ResearchModule,
    report: ResearchReport,
    front_matter: dict[str, object],
    body: str,
) -> list[str]:
    errors: list[str] = []
    required_values = {
        "title": report.title,
        "content_type": "research-report",
        "module_id": module.id,
        "report_id": report.id,
        "slug": report.slug,
        "code_tag": module.code_tag,
        "research_question": module.research_question,
        "result_status": report.status,
        "evidence_level": report.evidence_level,
    }
    for key, expected in required_values.items():
        if str(front_matter.get(key, "")) != expected:
            errors.append(f"front_matter_mismatch:{report.id}:{key}")

    status = str(front_matter.get("result_status", ""))
    result = str(front_matter.get("result", ""))
    evidence = str(front_matter.get("evidence_level", ""))
    if status not in {"planned", "preregistered", "running", "published", "deferred"}:
        errors.append(f"invalid_result_status:{report.id}:{status}")
    if result not in {"positive", "negative", "mixed", "inconclusive", "not-run"}:
        errors.append(f"invalid_result:{report.id}:{result}")
    if evidence not in {"not-started", "derived", "implemented", "validated", "scaled", "transferred"}:
        errors.append(f"invalid_evidence_level:{report.id}:{evidence}")

    headings = set(re.findall(r"^##\s+(.+?)\s*$", body, flags=re.MULTILINE))
    for heading in REQUIRED_REPORT_HEADINGS:
        if heading not in headings:
            errors.append(f"missing_required_heading:{report.id}:{heading}")

    if status == "published":
        if bool(front_matter.get("draft", True)):
            errors.append(f"published_report_is_draft:{report.id}")
        if result == "not-run":
            errors.append(f"published_report_without_result:{report.id}")
        if evidence == "not-started":
            errors.append(f"published_report_without_evidence:{report.id}")
        for key in ("directional_hypothesis", "artifact_manifest", "raw_results", "reproduction_command"):
            if not str(front_matter.get(key, "")).strip():
                errors.append(f"published_report_missing:{report.id}:{key}")
        if "<!--" in body:
            errors.append(f"published_report_contains_placeholders:{report.id}")
    return errors


def _module_by_id(curriculum: ResearchCurriculum, module_id: str) -> ResearchModule:
    module = next((candidate for candidate in curriculum.modules if candidate.id == module_id), None)
    if module is None:
        raise ValueError(f"unknown_research_module:{module_id}")
    return module


def _week_range(value: str, errors: list[str], module_id: str) -> list[int]:
    match = re.fullmatch(r"(\d+)-(\d+)", value.strip())
    if not match:
        errors.append(f"invalid_week_range:{module_id}:{value}")
        return []
    start, end = (int(part) for part in match.groups())
    if start < 1 or end < start or end > 52:
        errors.append(f"invalid_week_range:{module_id}:{value}")
        return []
    return list(range(start, end + 1))


def _read_markdown(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"missing_yaml_front_matter:{path}")
    front_matter = yaml.safe_load(match.group(1)) or {}
    if not isinstance(front_matter, dict):
        raise ValueError(f"invalid_yaml_front_matter:{path}")
    return front_matter, match.group(2)


def _placeholder_for(heading: str, module: ResearchModule) -> str:
    if heading == "Research question":
        return module.research_question
    if heading == "Run design":
        return "<!-- Record model, data, tokens, seeds, hardware, stopping rule, metrics, config hash, and compute ceiling. -->"
    if heading == "Results":
        return "<!-- Leave empty until raw and aggregate results exist. Do not write expected values as observations. -->"
    if heading == "Reproduce the artifact":
        return "<!-- Link the immutable code tag, run manifest, raw results, environment, and exact command after verification. -->"
    return f"<!-- Complete this section during the {heading.lower()} stage; do not invent missing evidence. -->"

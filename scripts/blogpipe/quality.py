"""Canonical quality-report helpers shared across editor, graph, and packaging."""

from __future__ import annotations

import json
from pathlib import Path

from .models import EditorReport, FailureReason, QualityReport, RenderReport


def _dedupe_reasons(reasons: list[FailureReason]) -> list[FailureReason]:
    seen: set[tuple[str, str, str]] = set()
    out: list[FailureReason] = []
    for reason in reasons:
        key = (reason.stage, reason.code, reason.message)
        if key in seen:
            continue
        seen.add(key)
        out.append(reason)
    return out


def recompute(report: QualityReport) -> QualityReport:
    blocking = [r for r in report.blocking_reasons if r.blocking]
    pass_gate = bool(
        report.evidence_valid
        and report.draft_valid
        and (not report.render_checked or report.render_valid)
        and (not report.package_checked or report.package_valid)
        and not blocking
    )
    status = "passed" if pass_gate else "blocked"
    return report.model_copy(
        update={
            "blocking_reasons": _dedupe_reasons(report.blocking_reasons),
            "pass_gate": pass_gate,
            "overall_status": status,
        }
    )


def from_editor(editor: EditorReport) -> QualityReport:
    reasons: list[FailureReason] = []
    if not editor.grounding_ok:
        for issue in editor.grounding_issues or ["grounding_failed"]:
            reasons.append(
                FailureReason(
                    code="grounding_issue",
                    message=str(issue),
                    stage="editor",
                )
            )
    for issue in editor.lint_issues:
        reasons.append(
            FailureReason(
                code=f"lint:{issue}",
                message=str(issue),
                stage="draft",
            )
        )
    if not editor.five_questions_ok:
        reasons.append(
            FailureReason(
                code="five_questions_incomplete",
                message="The draft does not answer the five skim questions.",
                stage="editor",
            )
        )
    if not editor.llm_ok:
        reasons.append(
            FailureReason(
                code="llm_quality_incomplete",
                message="The required editor/grounding LLM passes did not complete cleanly.",
                stage="editor",
            )
        )
    evidence_valid = bool(editor.grounding_ok and not editor.grounding_issues)
    draft_valid = bool(
        editor.five_questions_ok
        and editor.llm_ok
        and not editor.lint_issues
        and editor.rubric_score > 0
    )
    return recompute(
        QualityReport(
            evidence_valid=evidence_valid,
            draft_valid=draft_valid,
            render_valid=False,
            package_valid=False,
            render_checked=False,
            package_checked=False,
            editor_report=editor,
            llm_ok=editor.llm_ok,
            blocking_reasons=reasons,
        )
    )


def with_render(
    report: QualityReport,
    render_report: RenderReport,
    *,
    package_valid: bool,
    render_checked: bool = True,
    package_checked: bool = True,
    artifact_paths: dict[str, str] | None = None,
    package_errors: list[str] | None = None,
) -> QualityReport:
    reasons = list(report.blocking_reasons)
    for err in render_report.errors:
        reasons.append(
            FailureReason(code="render_error", message=str(err), stage="render")
        )
    for err in package_errors or []:
        reasons.append(
            FailureReason(code="package_error", message=str(err), stage="package")
        )
    merged_paths = dict(report.artifact_paths)
    merged_paths.update(artifact_paths or {})
    return recompute(
        report.model_copy(
            update={
                "render_checked": bool(render_checked),
                "package_checked": bool(package_checked),
                "render_valid": render_report.ok if render_checked else False,
                "package_valid": bool(package_valid) if package_checked else False,
                "render_report": render_report,
                "artifact_paths": merged_paths,
                "blocking_reasons": reasons,
            }
        )
    )


def load(path: Path) -> QualityReport | None:
    if not path.is_file():
        return None
    try:
        return QualityReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save(path: Path, report: QualityReport) -> None:
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def save_dict(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

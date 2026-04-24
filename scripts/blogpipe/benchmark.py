"""Offline benchmark harness for blog quality regressions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config, quality
from .graph.nodes import node_meta_review
from .memory import _ROOT
from .models import EditorReport, FailureReason, QualityReport, RenderReport
from .package import _build_daily_email, _validate_rendered_html


def _default_fixture_path() -> Path:
    override = (config.benchmark_fixture_path() or "").strip()
    if override:
        return Path(override)
    return _ROOT / "tests" / "fixtures" / "blogpipe_benchmark_cases.json"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _quality_from_case(case: dict[str, Any]) -> tuple[QualityReport, dict[str, Any]]:
    editor_report = EditorReport.model_validate(case.get("editor_report") or {})
    qrep = quality.from_editor(editor_report)
    detail: dict[str, Any] = {}
    review_notes = case.get("review_notes") or []
    if review_notes:
        meta = node_meta_review({"review_notes": review_notes}).get("meta_review") or {}
        detail["meta_review"] = meta
        findings = [str(x) for x in (meta.get("findings") or [])]
        if findings:
            reasons = list(qrep.blocking_reasons)
            reasons.extend(
                FailureReason(code=f"meta:{item}", message=item, stage="review")
                for item in findings
            )
            qrep = quality.recompute(qrep.model_copy(update={"blocking_reasons": reasons}))
    html_text = str(case.get("html") or "")
    body = str(case.get("body") or "")
    if html_text:
        errors, warnings, flags = _validate_rendered_html(body, html_text)
        render_report = RenderReport(
            html_valid=not bool(errors),
            pdf_valid=not bool(errors),
            errors=errors,
            warnings=warnings,
            **flags,
        )
        qrep = quality.with_render(
            qrep,
            render_report,
            package_valid=render_report.ok,
            render_checked=True,
            package_checked=True,
        )
        detail["render_report"] = render_report.model_dump()
    return qrep, detail


def _evaluate_review_email(
    name: str,
    case: dict[str, Any],
    qrep: QualityReport,
) -> dict[str, Any]:
    front = {
        "title": str(case.get("title") or name),
        "one_sentence_takeaway": str(case.get("takeaway") or ""),
    }
    body = str(case.get("body") or "")
    benchmark_report = case.get("benchmark_report") or {
        "ok": True,
        "total_cases": 1,
        "passed_cases": 1,
        "failed_cases": 0,
        "cases": [],
    }
    out = _ROOT / "reports" / f"benchmark_email_preview_{name}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    _build_daily_email(
        out,
        brief=case.get("brief") or {},
        editor=case.get("editor_report") or {},
        quality_report=qrep,
        benchmark_report=benchmark_report,
        rank=case.get("rank") or {},
        front=front,
        body=body,
    )
    email_html = out.read_text(encoding="utf-8")
    contains = [str(x) for x in (case.get("expect_email_contains") or [])]
    not_contains = [str(x) for x in (case.get("expect_email_not_contains") or [])]
    return {
        "path": str(out),
        "contains_ok": all(x in email_html for x in contains),
        "not_contains_ok": all(x not in email_html for x in not_contains),
        "failed_contains": [x for x in contains if x not in email_html],
        "failed_not_contains": [x for x in not_contains if x in email_html],
    }


def run(path: Path | None = None) -> dict[str, Any]:
    fixture = path or _default_fixture_path()
    cases = _load_cases(fixture)
    results: list[dict[str, Any]] = []
    passed = 0
    for case in cases:
        name = str(case.get("name") or "unnamed_case")
        qrep, detail = _quality_from_case(case)
        expected_pass_gate = bool(case.get("expect_pass_gate", False))
        expected_status = str(case.get("expect_overall_status") or ("passed" if expected_pass_gate else "blocked"))
        expected_blocks = [str(x) for x in (case.get("expect_blocking_contains") or [])]
        actual_codes = [r.code for r in qrep.blocking_reasons]
        ok = (
            qrep.pass_gate == expected_pass_gate
            and qrep.overall_status == expected_status
            and all(any(expect in code for code in actual_codes) for expect in expected_blocks)
        )
        if case.get("expect_email_contains") or case.get("expect_email_not_contains"):
            email_eval = _evaluate_review_email(name, case, qrep)
            detail["email_review"] = email_eval
            ok = ok and bool(email_eval["contains_ok"]) and bool(email_eval["not_contains_ok"])
        if ok:
            passed += 1
        results.append(
            {
                "name": name,
                "ok": ok,
                "expected": {
                    "pass_gate": expected_pass_gate,
                    "overall_status": expected_status,
                    "blocking_contains": expected_blocks,
                },
                "actual": {
                    "pass_gate": qrep.pass_gate,
                    "overall_status": qrep.overall_status,
                    "blocking_codes": actual_codes,
                },
                "detail": detail,
            }
        )
    report = {
        "fixture_path": str(fixture),
        "total_cases": len(results),
        "passed_cases": passed,
        "failed_cases": len(results) - passed,
        "ok": passed == len(results),
        "cases": results,
    }
    reports_dir = _ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "benchmark_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report

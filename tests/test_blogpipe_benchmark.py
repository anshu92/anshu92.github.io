from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
os.environ.setdefault("BLOGPIPE_REPO_ROOT", os.path.join(os.path.dirname(__file__), ".."))


def test_benchmark_harness_runs_fixture_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    from blogpipe import benchmark

    fixture = Path(__file__).parent / "fixtures" / "blogpipe_benchmark_cases.json"
    monkeypatch.setattr(benchmark, "_ROOT", tmp_path)
    report = benchmark.run(fixture)

    assert report["ok"] is True
    assert report["total_cases"] >= 5
    out = tmp_path / "reports" / "benchmark_report.json"
    assert out.is_file()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["passed_cases"] == saved["total_cases"]

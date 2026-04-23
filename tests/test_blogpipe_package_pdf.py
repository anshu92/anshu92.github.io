"""Optional PDF helpers (requires wkhtmltopdf on PATH to exercise)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
os.environ.setdefault("BLOGPIPE_REPO_ROOT", os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.skipif(not shutil.which("wkhtmltopdf"), reason="wkhtmltopdf not installed")
def test_fallback_minimal_pdf_writes_file(tmp_path: Path) -> None:
    from blogpipe.package import _fallback_minimal_pdf

    out = tmp_path / "note.pdf"
    r = _fallback_minimal_pdf(out, "Test note for PDF attachment.")
    assert r is not None
    assert out.is_file()
    assert out.stat().st_size > 100

from __future__ import annotations

import configparser
import os
from pathlib import Path


def test_setup_cfg_includes_voice_package() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = configparser.ConfigParser()
    cfg.read(root / "scripts" / "setup.cfg", encoding="utf-8")
    packages_raw = cfg.get("options", "packages")
    packages = {line.strip() for line in packages_raw.splitlines() if line.strip()}
    assert "blogpipe.voice" in packages


def test_voice_package_has_init_file() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "blogpipe" / "voice" / "__init__.py").is_file()

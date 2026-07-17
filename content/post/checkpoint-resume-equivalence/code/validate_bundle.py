#!/usr/bin/env python3
"""Validate the Markdown article bundle without requiring Hugo or network access."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

REQUIRED_FRONTMATTER = {
    "title",
    "description",
    "summary",
    "date",
    "lastmod",
    "draft",
    "slug",
    "author",
    "lead_archetype",
    "secondary_archetypes",
    "series",
    "series_order",
    "tags",
    "categories",
    "competencies",
    "current_role_tracks",
    "frontier_tracks",
    "mastery_artifact",
    "claim_boundary",
}
REQUIRED_ASSET_FIELDS = {
    "path",
    "type",
    "asset_type",
    "purpose",
    "alt_text",
    "caption",
    "provenance_or_license",
    "generation_method",
    "source_data_path",
    "article_section",
    "units",
    "assumptions",
    "accessibility",
}
FORBIDDEN_FINAL_PHRASES = {
    "everything else is bookkeeping",
    "that is the complete checkpoint contract",
    "this section proves the production invariant",
    "that is the complete mechanism",
    "the irreducible core",
}
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")
IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")
URL_PATTERN = re.compile(r"https?://[^\s)>}\]]+")


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        raise ValueError("article must start with YAML front matter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("front matter closing delimiter missing")
    raw = text[4:end]
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("front matter must parse to a mapping")
    return data, text[end + 5 :]


def is_external(target: str) -> bool:
    return urlparse(target).scheme in {"http", "https"}


def check_math(body: str, errors: list[str]) -> None:
    if body.count("\\[") != body.count("\\]"):
        fail(errors, "display-math delimiters \\[ and \\] are unbalanced")
    if body.count("\\(") != body.count("\\)"):
        fail(errors, "inline-math delimiters \\( and \\) are unbalanced")
    display_blocks = re.findall(r"\\\[(.*?)\\\]", body, flags=re.S)
    inline_blocks = re.findall(r"\\\((.*?)\\\)", body, flags=re.S)
    for index, block in enumerate(display_blocks + inline_blocks, start=1):
        if block.count("{") != block.count("}"):
            fail(errors, f"math block {index} has unbalanced braces")
        if CONTROL_CHARACTERS.search(block):
            fail(errors, f"math block {index} contains an ASCII control character")
    if "\\theta" in body and "\\[" not in body:
        fail(errors, "LaTeX command found without a display-math block")


def local_target_path(root: Path, target: str) -> Path | None:
    target = target.split("#", 1)[0]
    if not target:
        return None
    if target.startswith("/"):
        return Path(target)
    return (root / target).resolve()


def svg_text_children(svg_root: ET.Element) -> tuple[bool, bool]:
    children = list(svg_root)
    names = [child.tag.split("}")[-1] for child in children]
    return "title" in names, "desc" in names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    checks: dict[str, Any] = {}

    forbidden_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name in {"__pycache__", ".pytest_cache"}]
    if forbidden_dirs:
        fail(errors, f"cache directories must not be packaged: {[str(p.relative_to(root)) for p in forbidden_dirs]}")

    markdown_files = [p for p in root.rglob("*.md") if not any(part.startswith(".") for part in p.relative_to(root).parts)]
    checks["markdown_file_count"] = len(markdown_files)
    if markdown_files != [root / "index.md"]:
        fail(errors, f"exactly one Markdown article named index.md is required; found {[str(p.relative_to(root)) for p in markdown_files]}")
        article = root / "index.md"
    else:
        article = markdown_files[0]

    if not article.exists():
        fail(errors, "index.md is missing")
        text = ""
        frontmatter: dict[str, Any] = {}
        body = ""
    else:
        text = article.read_text(encoding="utf-8")
        if CONTROL_CHARACTERS.search(text):
            fail(errors, "index.md contains a disallowed ASCII control character")
        try:
            frontmatter, body = parse_frontmatter(text)
        except Exception as exc:  # noqa: BLE001
            fail(errors, f"front matter parse failed: {exc}")
            frontmatter, body = {}, text

    missing_frontmatter = sorted(REQUIRED_FRONTMATTER - set(frontmatter))
    if missing_frontmatter:
        fail(errors, f"missing front matter fields: {missing_frontmatter}")
    if frontmatter.get("draft") is not False:
        fail(errors, "the sole article must have draft: false")
    if frontmatter.get("author") != "Anshuman Sahoo":
        fail(errors, "author must be Anshuman Sahoo")
    if frontmatter.get("lead_archetype") != "Experimental Field Reporter":
        fail(errors, "lead_archetype mismatch")
    secondary = frontmatter.get("secondary_archetypes", [])
    if not isinstance(secondary, list) or len(secondary) > 2:
        fail(errors, "secondary_archetypes must be a list with at most two entries")

    words = re.findall(r"\b[\w’'-]+\b", body)
    checks["article_word_count"] = len(words)
    if not 1800 <= len(words) <= 4000:
        fail(errors, f"article word count {len(words)} is outside the expected 1800-4000 range")
    h2 = re.findall(r"^## (.+)$", body, flags=re.M)
    checks["h2_headings"] = h2
    if not 4 <= len(h2) <= 7:
        fail(errors, f"article must have 4-7 H2 sections; found {len(h2)}")
    if body.count("```") % 2:
        fail(errors, "fenced code blocks are unbalanced")
    check_math(body, errors)

    lower = body.lower()
    for phrase in sorted(FORBIDDEN_FINAL_PHRASES):
        if phrase in lower:
            fail(errors, f"forbidden mechanical phrase remains: {phrase!r}")
    invariant_count = len(re.findall(r"\binvariant\b", lower))
    checks["invariant_count"] = invariant_count
    if invariant_count > 1:
        fail(errors, "the word 'invariant' appears more than once")

    local_links: set[str] = set()
    external_links: set[str] = set()
    image_links: set[str] = set()
    for target in LINK_PATTERN.findall(body):
        if is_external(target):
            external_links.add(target.rstrip(".,;"))
        else:
            local_links.add(target)
    for target in IMAGE_PATTERN.findall(body):
        image_links.add(target)
        local_links.add(target)
    frontmatter_image = frontmatter.get("image")
    if isinstance(frontmatter_image, str) and frontmatter_image:
        image_links.add(frontmatter_image)
        local_links.add(frontmatter_image)
    checks["local_link_count"] = len(local_links)
    checks["external_link_count"] = len(external_links)
    for target in sorted(local_links):
        path = local_target_path(root, target)
        if path is None:
            continue
        try:
            path.relative_to(root)
        except ValueError:
            fail(errors, f"local link escapes bundle root: {target}")
            continue
        if not path.exists():
            fail(errors, f"broken local link: {target}")

    link_report_path = root / "data" / "link-check.json"
    if not link_report_path.exists():
        fail(errors, "data/link-check.json is missing")
    else:
        try:
            link_report = json.loads(link_report_path.read_text(encoding="utf-8"))
            records = {record["url"]: record for record in link_report.get("links", [])}
            missing = sorted(external_links - set(records))
            if missing:
                fail(errors, f"external links missing from link-check report: {missing}")
            bad = sorted(url for url in external_links if records.get(url, {}).get("status") != "PASS")
            if bad:
                fail(errors, f"external links did not pass validation: {bad}")
        except Exception as exc:  # noqa: BLE001
            fail(errors, f"link-check report parse failed: {exc}")

    manifest_path = root / "asset-manifest.yaml"
    if not manifest_path.exists():
        fail(errors, "asset-manifest.yaml is missing")
        manifest = {}
    else:
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise TypeError("manifest is not a mapping")
        except Exception as exc:  # noqa: BLE001
            fail(errors, f"asset manifest parse failed: {exc}")
            manifest = {}
    visuals = manifest.get("visuals", []) if isinstance(manifest, dict) else []
    if not isinstance(visuals, list) or len(visuals) < 3:
        fail(errors, "asset manifest must contain one cover and at least two explanatory visuals")
        visuals = []
    manifest_paths: set[str] = set()
    terminal_snapshot_count = 0
    for index, visual in enumerate(visuals, start=1):
        if not isinstance(visual, dict):
            fail(errors, f"visual {index} is not a mapping")
            continue
        missing = sorted(REQUIRED_ASSET_FIELDS - set(visual))
        if missing:
            fail(errors, f"visual {index} missing fields: {missing}")
        rel = str(visual.get("path", ""))
        manifest_paths.add(rel)
        asset = root / rel
        if not asset.exists():
            fail(errors, f"manifest asset missing: {rel}")
            continue
        source = visual.get("source_data_path")
        if source and not (root / str(source)).exists():
            fail(errors, f"source data missing for {rel}: {source}")
        if asset.suffix.lower() == ".svg":
            try:
                svg_root = ET.parse(asset).getroot()
                has_title, has_desc = svg_text_children(svg_root)
                if not has_title or not has_desc:
                    fail(errors, f"SVG lacks direct title/desc accessibility children: {rel}")
                if "viewBox" not in svg_root.attrib:
                    fail(errors, f"SVG lacks viewBox: {rel}")
                if visual.get("type") == "terminal_transcript_svg":
                    terminal_snapshot_count += 1
                    source_path = root / str(source)
                    transcript_lines = source_path.read_text(encoding="utf-8").rstrip("\n").splitlines()
                    svg_text_lines = [
                        "".join(node.itertext())
                        for node in svg_root.iter()
                        if node.tag.split("}")[-1] == "text"
                    ]
                    missing_lines = [line for line in transcript_lines if line not in svg_text_lines]
                    if missing_lines:
                        fail(errors, f"terminal SVG does not preserve transcript lines for {rel}: {missing_lines[:3]}")
                    if not transcript_lines or not transcript_lines[0].startswith("$ "):
                        fail(errors, f"terminal transcript must begin with the executed command: {source}")
            except Exception as exc:  # noqa: BLE001
                fail(errors, f"SVG XML parse failed for {rel}: {exc}")
    checks["terminal_snapshot_count"] = terminal_snapshot_count
    if terminal_snapshot_count < 3:
        fail(errors, "article must contain at least three validated terminal transcript snapshots")
    if image_links != manifest_paths:
        fail(errors, f"article image references and asset manifest differ: article={sorted(image_links)}, manifest={sorted(manifest_paths)}")

    required_files = [
        "references.bib",
        "requirements.txt",
        "code/checkpoint_equivalence.py",
        "code/test_checkpoint_equivalence.py",
        "code/generate_visuals.py",
        "code/generate_terminal_snapshots.py",
        "code/validate_bundle.py",
        "data/terminal-01-fixture.txt",
        "data/terminal-02-seed-11-matrix.txt",
        "data/terminal-03-scheduler-trace.txt",
        "data/terminal-04-tests.txt",
        "data/results.csv",
        "data/run-summary.json",
        "data/raw-output.txt",
        "data/environment.json",
        "data/checkpoint-schema.json",
        "data/step-traces.json",
    ]
    for rel in required_files:
        if not (root / rel).exists():
            fail(errors, f"required artifact missing: {rel}")

    try:
        summary = json.loads((root / "data/run-summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "PASS":
            fail(errors, "run-summary status is not PASS")
        if summary.get("run_count") != 35:
            fail(errors, "run-summary must contain 35 runs")
        acceptance = summary.get("acceptance_checks", {})
        if not acceptance or not all(acceptance.values()):
            fail(errors, "not every experiment acceptance check passed")
        with (root / "data/results.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 35:
            fail(errors, f"results.csv must contain 35 data rows; found {len(rows)}")
        expected_claim_strings = ["5/5", "0/5", "8.740e-2", "7.114e-2", "7.471e-2", "5.941e-3", "3.202e-2"]
        for claim in expected_claim_strings:
            if claim not in body:
                fail(errors, f"expected retained-data claim missing from article: {claim}")
    except Exception as exc:  # noqa: BLE001
        fail(errors, f"experiment data validation failed: {exc}")

    bib = (root / "references.bib").read_text(encoding="utf-8") if (root / "references.bib").exists() else ""
    if CONTROL_CHARACTERS.search(bib):
        fail(errors, "references.bib contains a disallowed control character")
    bib_urls = {url.rstrip(".,;}") for url in URL_PATTERN.findall(bib)}
    if not bib_urls.issubset(external_links):
        fail(errors, f"references.bib contains URLs not cited in the article: {sorted(bib_urls - external_links)}")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "root": str(root),
        "checks": checks,
        "error_count": len(errors),
        "errors": errors,
    }
    report_path = args.report
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

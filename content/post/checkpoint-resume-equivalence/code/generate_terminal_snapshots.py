#!/usr/bin/env python3
"""Render retained command transcripts as accessible local terminal SVGs."""
from __future__ import annotations

import argparse
import html
from pathlib import Path

BG = "#0D1117"
PANEL = "#161B22"
BORDER = "#30363D"
TEXT = "#E6EDF3"
MUTED = "#8B949E"
PROMPT = "#7EE787"
TITLE = "#C9D1D9"
RED = "#FF7B72"
YELLOW = "#D2A8FF"

SNAPSHOTS = (
    (
        "data/terminal-01-fixture.txt",
        "figures/terminal-01-fixture.svg",
        "Inspecting the concrete model and checkpoint payloads",
        "Actual terminal transcript showing Python and PyTorch versions, the TinyDropoutClassifier architecture, parameter shapes and counts, training configuration, and checkpoint payload keys.",
    ),
    (
        "data/terminal-02-seed-11-matrix.txt",
        "figures/terminal-02-seed-11-matrix.svg",
        "Running all seven interventions for seed 11",
        "Actual terminal transcript showing exact resume status, first divergent optimizer step, final parameter error, next-batch match, and learning-rate error for seven checkpoint scenarios.",
    ),
    (
        "data/terminal-03-scheduler-trace.txt",
        "figures/terminal-03-scheduler-trace.svg",
        "Tracing the delayed scheduler failure",
        "Actual terminal transcript showing the control and resumed learning rates. The learning rate differs at step eight while parameters still match; model parameters first differ at step nine.",
    ),
    (
        "data/terminal-04-tests.txt",
        "figures/terminal-04-tests.svg",
        "Running the direct equivalence and counterexample tests",
        "Actual terminal transcript showing twenty-one Pytest checks passing for the checkpoint equivalence harness and its inspectable commands.",
    ),
)


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def line_color(line: str, index: int) -> str:
    if index == 0 and line.startswith("$"):
        return PROMPT
    if " DIFF" in line or "exact=false" in line or "final_exact=false" in line:
        return RED
    if "exact=true" in line or "21 passed" in line:
        return PROMPT
    if line in {"runtime", "model", "parameters", "training", "checkpoint payloads"}:
        return YELLOW
    return TEXT


def render(source: Path, destination: Path, title: str, description: str) -> None:
    lines = source.read_text(encoding="utf-8").rstrip("\n").splitlines()
    max_chars = max((len(line) for line in lines), default=1)
    font_size = 18 if max_chars <= 115 else 15
    char_width = font_size * 0.61
    width = max(1120, min(1880, int(max_chars * char_width + 112)))
    line_height = font_size + 10
    top = 96
    bottom = 44
    height = top + len(lines) * line_height + bottom
    body: list[str] = [
        f'<rect width="{width}" height="{height}" rx="18" fill="{BG}"/>',
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="18" fill="none" stroke="{BORDER}" stroke-width="2"/>',
        f'<rect x="1" y="1" width="{width - 2}" height="58" rx="18" fill="{PANEL}"/>',
        f'<rect x="1" y="42" width="{width - 2}" height="17" fill="{PANEL}"/>',
        '<circle cx="28" cy="29" r="7" fill="#FF5F56"/>',
        '<circle cx="52" cy="29" r="7" fill="#FFBD2E"/>',
        '<circle cx="76" cy="29" r="7" fill="#27C93F"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="16" font-weight="600" fill="{TITLE}">{esc(title)}</text>',
    ]
    y = top
    for index, line in enumerate(lines):
        fill = line_color(line, index)
        body.append(
            f'<text x="36" y="{y}" xml:space="preserve" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="{font_size}" fill="{fill}">{esc(line)}</text>'
        )
        y += line_height
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        f'<title id="title">{esc(title)}</title>\n'
        f'<desc id="desc">{esc(description)}</desc>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    for source_rel, destination_rel, title, description in SNAPSHOTS:
        render(args.root / source_rel, args.root / destination_rel, title, description)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

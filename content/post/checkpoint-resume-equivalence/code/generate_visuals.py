#!/usr/bin/env python3
"""Generate the article's four local SVG visuals from retained experiment data."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable

WIDTH = 1200
BG = "#F8F6F1"
INK = "#17212B"
MUTED = "#5C6873"
ACCENT = "#B34D2E"
ACCENT2 = "#286C72"
GOOD = "#2E7D55"
BAD = "#B23A48"
GRID = "#D9D6CF"
PANEL = "#FFFFFF"


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def text(x: float, y: float, value: str, size: int = 24, weight: int = 400,
         fill: str = INK, anchor: str = "start", family: str = "Inter,Arial,sans-serif") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(value)}</text>'
    )


def multiline(x: float, y: float, lines: Iterable[str], size: int = 22,
              line_height: int = 30, weight: int = 400, fill: str = INK,
              anchor: str = "start") -> str:
    spans = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else line_height
        spans.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    return (
        f'<text x="{x}" y="{y}" font-family="Inter,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
        + "".join(spans) + "</text>"
    )


def rect(x: float, y: float, w: float, h: float, fill: str = PANEL,
         stroke: str = GRID, radius: int = 18, sw: int = 2) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = INK,
         sw: int = 4, dash: str | None = None) -> str:
    attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{attr}/>'


def circle(x: float, y: float, r: float, fill: str, stroke: str = "none", sw: int = 0) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def svg_document(height: int, title_value: str, desc_value: str, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">
<title id="title">{esc(title_value)}</title>
<desc id="desc">{esc(desc_value)}</desc>
<rect width="100%" height="100%" fill="{BG}"/>
{body}
</svg>\n'''


def checkpoint_capsule(x: float, y: float, labels: list[str], complete: bool) -> str:
    parts = [rect(x, y, 290, 210, fill=PANEL, stroke=GOOD if complete else BAD, radius=22, sw=4)]
    parts.append(text(x + 24, y + 36, "checkpoint payload", 22, 700, GOOD if complete else BAD))
    for i, label in enumerate(labels):
        cy = y + 70 + i * 26
        parts.append(circle(x + 28, cy - 6, 7, GOOD if complete else (GOOD if i < 2 else BAD)))
        parts.append(text(x + 45, cy, label, 18, 500))
    return "".join(parts)


def generate_cover(path: Path) -> None:
    h = 675
    b = []
    b.append(text(70, 82, "A restart is not a resume", 48, 750))
    b.append(text(70, 121, "The checkpoint equivalence test", 25, 500, ACCENT))
    # timeline
    y_control, y_resume = 260, 455
    x0, cut, x1 = 90, 525, 1110
    b += [text(90, 220, "uninterrupted control", 22, 700), line(x0, y_control, x1, y_control, ACCENT2, 8)]
    for x in [150, 270, 390, 525, 680, 840, 1010]:
        b.append(circle(x, y_control, 10, ACCENT2))
    b += [text(90, 414, "interrupted run", 22, 700), line(x0, y_resume, cut, y_resume, ACCENT2, 8)]
    for x in [150, 270, 390, 525]:
        b.append(circle(x, y_resume, 10, ACCENT2))
    b.append(line(cut, 180, cut, 535, MUTED, 3, "9 9"))
    b.append(text(cut, 165, "save / load", 19, 700, MUTED, "middle"))
    # good resumed path overlaps control in projected upper lane
    b.append(line(cut, y_resume, 650, 355, GOOD, 7))
    b.append(line(650, 355, x1, 355, GOOD, 7))
    b.append(text(705, 398, "complete payload: same future transition", 20, 700, GOOD))
    # bad path branches down
    b.append(line(cut, y_resume, 660, 520, BAD, 7))
    b.append(line(660, 520, x1, 590, BAD, 7))
    b.append(text(705, 505, "missing state: branch appears later", 20, 700, BAD))
    b.append(checkpoint_capsule(735, 150, ["model + optimizer", "scheduler", "RNG + data cursor", "partial gradients (mid-step)"], True))
    b.append(text(70, 635, "A successful load only proves deserialization. The next update is the test.", 24, 600, INK))
    path.write_text(svg_document(h, "A Restart Is Not a Resume", "Two training timelines meet at a checkpoint. A complete payload continues along the control trajectory while an incomplete payload branches after the next affected transition.", "".join(b)), encoding="utf-8")


def generate_state_surface(path: Path) -> None:
    h = 880
    b = [text(60, 70, "What crosses the interruption boundary?", 40, 750),
         text(60, 108, "The fixture saves the inputs needed to reproduce the next transition—not only the visible weights.", 22, 400, MUTED)]
    groups = [
        ("Update state", ["model parameters + buffers", "optimizer moments", "scheduler clock"], ACCENT2),
        ("Stochastic state", ["Python RNG", "NumPy RNG", "PyTorch CPU RNG"], ACCENT),
        ("Data state", ["epoch + cursor", "current permutation", "permutation-generator state"], GOOD),
        ("Partial-step state", ["accumulation position", "parameter gradients", "global microstep"], BAD),
    ]
    start_x, start_y, w, card_h, gap = 60, 170, 510, 245, 40
    for i, (heading, items, color) in enumerate(groups):
        col, row = i % 2, i // 2
        x = start_x + col * (w + gap)
        y = start_y + row * (card_h + gap)
        b.append(rect(x, y, w, card_h, fill=PANEL, stroke=color, radius=20, sw=3))
        b.append(text(x + 26, y + 42, heading, 25, 750, color))
        for j, item in enumerate(items):
            cy = y + 86 + j * 46
            b.append(circle(x + 30, cy - 7, 7, color))
            b.append(text(x + 50, cy, item, 21, 500))
        if heading == "Partial-step state":
            b.append(text(x + 26, y + 225, "Needed only when the save cuts through accumulation.", 17, 500, MUTED))
    b.append(line(60, 748, 1140, 748, GRID, 2))
    b.append(text(60, 785, "Boundary save", 20, 700, GOOD))
    b.append(multiline(60, 815, [
        "Update + stochastic + data state,",
        "with counters at a clean optimizer-step boundary."
    ], 17, 24, 400))
    b.append(text(620, 785, "Mid-accumulation save", 20, 700, BAD))
    b.append(multiline(620, 815, [
        "The same payload plus partial gradients",
        "and the accumulation position."
    ], 17, 24, 400))
    path.write_text(svg_document(h, "Checkpoint state surface in the experiment", "Four groups of state cross the interruption boundary: update state, stochastic state, data state, and partial-step state. Partial gradients are required only for mid-accumulation saves.", "".join(b)), encoding="utf-8")


def load_step_traces(path: Path, seed: int = 11) -> dict[str, list[dict]]:
    rows = json.loads(path.read_text())
    return {row["scenario"]: row["step_trace"] for row in rows if row["seed"] == seed}


def generate_intervention_traces(path: Path, traces_path: Path) -> None:
    traces = load_step_traces(traces_path)
    h = 860
    b = [text(60, 65, "Where each omission first changed the run", 40, 750),
         text(60, 103, "Representative seed 11. The vertical distance encodes branch visibility, not loss magnitude.", 21, 400, MUTED)]
    left, right, top, bottom = 120, 1120, 170, 680
    b.append(rect(left, top, right-left, bottom-top, fill=PANEL, stroke=GRID, radius=16, sw=2))
    for step in range(5, 13):
        x = left + (step-5) * (right-left) / 7
        b.append(line(x, top+20, x, bottom-45, GRID, 1, "4 8"))
        b.append(text(x, bottom-15, str(step), 18, 600, MUTED, "middle"))
    b.append(text((left+right)/2, 735, "optimizer step", 20, 600, MUTED, "middle"))
    b.append(line(left, 270, right, 270, ACCENT2, 6))
    b.append(text(left+15, 246, "uninterrupted control", 19, 700, ACCENT2))
    b.append(line(left, 150, left, bottom, MUTED, 3, "8 8"))
    b.append(text(left, 145, "checkpoint after step 5", 18, 700, MUTED, "middle"))

    lanes = [
        ("full boundary", "full_boundary", 270, GOOD),
        ("omit RNG", "omit_rng", 360, BAD),
        ("omit data stream", "omit_stream", 445, ACCENT),
        ("omit scheduler", "omit_scheduler", 530, "#7A4EAB"),
        ("mid-save, no gradients", "mid_without_gradients", 615, "#8B5E34"),
    ]
    for label, scenario, y, color in lanes:
        first = None
        if traces.get(scenario):
            first = next((r["logical_step"] for r in traces[scenario] if not r["model_exact"]), None)
        if scenario == "full_boundary":
            b.append(line(left, 270, right, 270, GOOD, 3, "10 8"))
            b.append(text(right-15, 255, "overlaps control through step 12", 18, 700, GOOD, "end"))
            continue
        start_x = left + ((first or 5)-5) * (right-left)/7
        b.append(line(left, 270, start_x, 270, color, 4, "8 6"))
        b.append(line(start_x, 270, min(right, start_x+95), y, color, 5))
        b.append(line(min(right, start_x+95), y, right-15, y, color, 5))
        b.append(circle(start_x, 270, 9, color))
        b.append(text(right-25, y-12, f"{label}: first branch at step {first}", 18, 700, color, "end"))
    b.append(multiline(60, 785, [
        "Observation: stochastic and data omissions branch on the first resumed update.",
        "The scheduler omission waits until its shifted decay boundary at step 9."
    ], 19, 28, 500, INK))
    path.write_text(svg_document(h, "Failure traces for checkpoint-state omissions", "A representative experiment trace shows complete boundary recovery overlapping the control. Missing RNG, data stream, or partial gradients branches at step six, while a missing scheduler branches at step nine.", "".join(b)), encoding="utf-8")


def load_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def generate_results(path: Path, results_path: Path) -> None:
    rows = load_results(results_path)
    order = ["full_boundary", "full_mid_accumulation", "model_optimizer_only", "omit_rng", "omit_stream", "omit_scheduler", "mid_without_gradients"]
    labels = {
        "full_boundary":"Full boundary payload",
        "full_mid_accumulation":"Full mid-accumulation payload",
        "model_optimizer_only":"Model + optimizer only",
        "omit_rng":"Omit RNG",
        "omit_stream":"Omit data stream",
        "omit_scheduler":"Omit scheduler",
        "mid_without_gradients":"Mid-save without partial gradients",
    }
    agg = {}
    for scenario in order:
        subset = [r for r in rows if r["scenario"] == scenario]
        errors = [float(r["max_parameter_abs_error"]) for r in subset]
        exact = sum(r["exact_resume"].lower() == "true" for r in subset)
        agg[scenario] = {"exact": exact, "median": median(errors), "max": max(errors)}
    h = 930
    b = [multiline(60, 62, [
            "Five-seed result: complete payloads matched;",
            "every planted omission failed"
         ], 36, 42, 750),
         multiline(60, 145, [
            "Exact means model, optimizer, scheduler, stream, RNG, gradients,",
            "and the next batch all matched the uninterrupted control."
         ], 19, 27, 400, MUTED)]
    x_label, x_exact, x_bar0, x_bar1 = 65, 570, 720, 1115
    b.append(text(x_exact, 215, "exact runs", 18, 700, MUTED, "middle"))
    b.append(text((x_bar0+x_bar1)/2, 215, "median final max |parameter error| (log scale)", 18, 700, MUTED, "middle"))
    floor = 1e-6
    max_error = max(v["median"] for v in agg.values())
    log_min, log_max = math.log10(floor), math.log10(max_error)
    for i, scenario in enumerate(order):
        y = 270 + i*82
        is_good = agg[scenario]["exact"] == 5
        b.append(rect(45, y-43, 1110, 64, fill=PANEL, stroke=GRID, radius=12, sw=1))
        b.append(text(x_label, y, labels[scenario], 20, 650, INK))
        b.append(text(x_exact, y, f"{agg[scenario]['exact']}/5", 22, 750, GOOD if is_good else BAD, "middle"))
        med = agg[scenario]["median"]
        if med == 0:
            b.append(line(x_bar0, y-6, x_bar1, y-6, GRID, 8))
            b.append(circle(x_bar0, y-6, 9, GOOD))
            b.append(text(x_bar0+18, y, "0 (bitwise match)", 18, 700, GOOD))
        else:
            pos = x_bar0 + (math.log10(max(med, floor))-log_min)/(log_max-log_min)*(x_bar1-x_bar0)
            b.append(line(x_bar0, y-6, x_bar1, y-6, GRID, 8))
            b.append(line(x_bar0, y-6, pos, y-6, BAD, 8))
            b.append(circle(pos, y-6, 9, BAD))
            if pos > x_bar1 - 120:
                b.append(text(pos-14, y-17, f"{med:.3e}", 17, 700, BAD, "end"))
            else:
                b.append(text(pos+14, y-17, f"{med:.3e}", 17, 700, BAD, "start"))
    b.append(text(60, 900, "Source: data/results.csv, 35 runs (7 scenarios × 5 seeds), PyTorch CPU fixture.", 19, 500, MUTED))
    path.write_text(svg_document(h, "Aggregate checkpoint equivalence results", "Seven checkpoint scenarios were run across five seeds. Both complete payload scenarios matched exactly in all five runs. Every incomplete payload scenario failed in all five runs, with nonzero final parameter error.", "".join(b)), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--step-traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_cover(args.output_dir.parent / "cover.svg")
    generate_state_surface(args.output_dir / "figure-01-state-surface.svg")
    generate_intervention_traces(args.output_dir / "figure-02-intervention-traces.svg", args.step_traces)
    generate_results(args.output_dir / "figure-03-results.svg", args.results)
    print("generated cover.svg and 3 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

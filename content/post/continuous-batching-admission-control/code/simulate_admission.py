#!/usr/bin/env python3
"""Deterministic continuous-batching admission-control simulator.

This is a mechanism model, not a GPU performance benchmark. It models:
- token-budgeted scheduler steps,
- block-granular KV-cache allocation,
- FCFS waiting,
- decode-first scheduling,
- optional reserve-full-input admission,
- an optional free-block watermark,
- preemption by recomputation when the running set exceeds capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque
import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Iterable


@dataclass
class Request:
    request_id: str
    arrival_step: int
    prompt_tokens: int
    output_tokens: int
    prefilled: int = 0
    decoded: int = 0
    admitted_step: int | None = None
    first_token_step: int | None = None
    completed_step: int | None = None
    preemptions: int = 0

    @property
    def finished(self) -> bool:
        return self.prefilled >= self.prompt_tokens and self.decoded >= self.output_tokens

    @property
    def total_cached_tokens(self) -> int:
        return self.prefilled + self.decoded

    def blocks(self, block_size: int) -> int:
        if self.total_cached_tokens == 0:
            return 0
        return math.ceil(self.total_cached_tokens / block_size)

    def full_input_blocks(self, block_size: int) -> int:
        return math.ceil(self.prompt_tokens / block_size)


@dataclass
class Config:
    total_blocks: int = 48
    block_size: int = 16
    max_num_batched_tokens: int = 64
    max_num_seqs: int = 12
    reserve_full_input: bool = False
    watermark_fraction: float = 0.0
    max_steps: int = 500


def generate_workload(seed: int = 7, count: int = 80) -> list[Request]:
    rng = random.Random(seed)
    requests = []
    for i in range(count):
        # Bursty arrivals with a long-tail prompt distribution.
        arrival = i // 3 + rng.choice([0, 0, 0, 1, 2])
        if rng.random() < 0.18:
            prompt = rng.randint(180, 420)
        else:
            prompt = rng.randint(16, 96)
        output = rng.randint(16, 96)
        requests.append(Request(f"r{i:03d}", arrival, prompt, output))
    return sorted(requests, key=lambda r: (r.arrival_step, r.request_id))


def simulate(requests: Iterable[Request], cfg: Config) -> tuple[list[dict], dict]:
    waiting = deque()
    future = deque(sorted(requests, key=lambda r: (r.arrival_step, r.request_id)))
    running: list[Request] = []
    finished: list[Request] = []
    timeline: list[dict] = []
    cumulative_preemptions = 0
    rejected_oversized = 0

    for step in range(cfg.max_steps):
        while future and future[0].arrival_step <= step:
            req = future.popleft()
            if req.full_input_blocks(cfg.block_size) > cfg.total_blocks:
                rejected_oversized += 1
            else:
                waiting.append(req)

        used_blocks = sum(r.blocks(cfg.block_size) for r in running)
        free_blocks = cfg.total_blocks - used_blocks
        watermark_blocks = math.floor(cfg.total_blocks * cfg.watermark_fraction)

        # Admission: FCFS, bounded by sequence slots and cache headroom.
        while waiting and len(running) < cfg.max_num_seqs:
            req = waiting[0]
            required = (
                req.full_input_blocks(cfg.block_size)
                if cfg.reserve_full_input
                else 1
            )
            if free_blocks - required < watermark_blocks:
                break
            waiting.popleft()
            req.admitted_step = step if req.admitted_step is None else req.admitted_step
            running.append(req)
            # Reservation is an admission check, not a physical allocation.
            # Actual block use is charged when tokens are processed below.

        token_budget = cfg.max_num_batched_tokens

        # Decode first: one token per active decoded request.
        for req in list(running):
            if token_budget <= 0:
                break
            if req.prefilled >= req.prompt_tokens and req.decoded < req.output_tokens:
                before = req.blocks(cfg.block_size)
                req.decoded += 1
                token_budget -= 1
                if req.first_token_step is None:
                    req.first_token_step = step
                after = req.blocks(cfg.block_size)
                free_blocks -= (after - before)

        # Chunked prefill in FCFS running order.
        for req in list(running):
            if token_budget <= 0:
                break
            if req.prefilled < req.prompt_tokens:
                remaining = req.prompt_tokens - req.prefilled
                chunk = min(remaining, token_budget)
                before = req.blocks(cfg.block_size)
                req.prefilled += chunk
                token_budget -= chunk
                after = req.blocks(cfg.block_size)
                free_blocks -= (after - before)

        # If allocations exceeded physical capacity, preempt newest requests
        # and recompute them later.
        while free_blocks < 0 and running:
            victim = running.pop()
            free_blocks += victim.blocks(cfg.block_size)
            victim.prefilled = 0
            victim.decoded = 0
            victim.first_token_step = None
            victim.preemptions += 1
            cumulative_preemptions += 1
            waiting.appendleft(victim)

        for req in list(running):
            if req.finished:
                req.completed_step = step
                running.remove(req)
                finished.append(req)

        timeline.append({
            "step": step,
            "waiting": len(waiting),
            "running": len(running),
            "finished": len(finished),
            "used_blocks": cfg.total_blocks - free_blocks,
            "free_blocks": free_blocks,
            "preemptions": cumulative_preemptions,
        })

        if not future and not waiting and not running:
            break

    completed = [r for r in finished if r.completed_step is not None]
    def percentile(values: list[float], p: float) -> float | None:
        if not values:
            return None
        values = sorted(values)
        idx = min(len(values) - 1, math.ceil(p * len(values)) - 1)
        return float(values[idx])

    ttft = [
        r.first_token_step - r.arrival_step
        for r in completed if r.first_token_step is not None
    ]
    e2e = [r.completed_step - r.arrival_step + 1 for r in completed]
    metrics = {
        "config": asdict(cfg),
        "completed": len(completed),
        "unfinished": len(requests) - len(completed),
        "rejected_oversized": rejected_oversized,
        "preemptions": cumulative_preemptions,
        "steps": len(timeline),
        "throughput_requests_per_step": len(completed) / max(1, len(timeline)),
        "p50_ttft_steps": percentile(ttft, 0.50),
        "p95_ttft_steps": percentile(ttft, 0.95),
        "p50_e2e_steps": percentile(e2e, 0.50),
        "p95_e2e_steps": percentile(e2e, 0.95),
        "mean_peak_used_blocks": max((x["used_blocks"] for x in timeline), default=0),
    }
    return timeline, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    workload = generate_workload(args.seed)
    configs = {
        "naive": Config(reserve_full_input=False, watermark_fraction=0.0),
        "reserve": Config(reserve_full_input=True, watermark_fraction=0.0),
        "reserve_watermark": Config(reserve_full_input=True, watermark_fraction=0.125),
    }

    all_metrics = {}
    for name, cfg in configs.items():
        cloned = [Request(r.request_id, r.arrival_step, r.prompt_tokens, r.output_tokens)
                  for r in workload]
        timeline, metrics = simulate(cloned, cfg)
        all_metrics[name] = metrics
        with (args.out / f"{name}_timeline.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=timeline[0].keys())
            writer.writeheader()
            writer.writerows(timeline)

    with (args.out / "metrics.json").open("w") as f:
        json.dump(all_metrics, f, indent=2)

    with (args.out / "workload.json").open("w") as f:
        json.dump([asdict(r) for r in workload], f, indent=2)

    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()

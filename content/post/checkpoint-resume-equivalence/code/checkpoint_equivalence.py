#!/usr/bin/env python3
"""Differential checkpoint-resume equivalence experiment.

The experiment compares an uninterrupted PyTorch CPU training run with resumed
runs that selectively omit checkpoint state. It is intentionally small enough
to inspect while retaining the stochastic and stateful mechanisms that make a
real resume non-trivial: optimizer moments, a learning-rate scheduler, Python,
NumPy and PyTorch RNGs, a shuffled data stream, and gradient accumulation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


SEEDS = (11, 17, 29, 47, 83)
SCENARIOS = (
    "full_boundary",
    "model_optimizer_only",
    "omit_rng",
    "omit_stream",
    "omit_scheduler",
    "full_mid_accumulation",
    "mid_without_gradients",
)


@dataclass(frozen=True)
class ExperimentConfig:
    input_dim: int = 8
    hidden_dim: int = 16
    num_classes: int = 3
    num_examples: int = 48
    batch_size: int = 4
    accumulation_steps: int = 3
    total_optimizer_steps: int = 12
    boundary_interrupt_step: int = 5
    mid_interrupt_completed_steps: int = 5
    mid_interrupt_microsteps: int = 1
    learning_rate: float = 0.03
    weight_decay: float = 0.01
    scheduler_step_size: int = 4
    scheduler_gamma: float = 0.7
    numpy_noise_std: float = 0.015
    python_scale_half_width: float = 0.05
    dropout_probability: float = 0.35
    dataset_seed: int = 20260717

    @property
    def total_microsteps(self) -> int:
        return self.total_optimizer_steps * self.accumulation_steps

    @property
    def boundary_interrupt_microsteps(self) -> int:
        return self.boundary_interrupt_step * self.accumulation_steps

    @property
    def mid_interrupt_microstep_index(self) -> int:
        return (
            self.mid_interrupt_completed_steps * self.accumulation_steps
            + self.mid_interrupt_microsteps
        )


class TinyDropoutClassifier(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.input = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout_probability)
        self.output = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(self.input(x))
        x = self.dropout(x)
        return self.output(x)


class StatefulBatchStream:
    """Shuffled, finite-epoch batch stream with an inspectable state_dict."""

    def __init__(self, x: Tensor, y: Tensor, order_seed: int, batch_size: int) -> None:
        if len(x) != len(y):
            raise ValueError("x and y must contain the same number of examples")
        if len(x) % batch_size != 0:
            raise ValueError("num_examples must be divisible by batch_size in this fixture")
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(order_seed)
        self.epoch = 0
        self.cursor = 0
        self.order = torch.randperm(len(x), generator=self.generator)

    def _advance_epoch(self) -> None:
        self.epoch += 1
        self.cursor = 0
        self.order = torch.randperm(len(self.x), generator=self.generator)

    def next_batch(self) -> tuple[Tensor, Tensor, Tensor]:
        if self.cursor == len(self.order):
            self._advance_epoch()
        end = self.cursor + self.batch_size
        if end > len(self.order):
            raise RuntimeError("fixture requires complete fixed-size batches")
        ids = self.order[self.cursor:end].clone()
        self.cursor = end
        return self.x[ids].clone(), self.y[ids].clone(), ids

    def state_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": self.order.clone(),
            "generator_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("batch size mismatch")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self.order = state["order"].clone()
        self.generator.set_state(state["generator_state"].clone())

    def peek_next_ids(self) -> list[int]:
        state = copy.deepcopy(self.state_dict())
        _, _, ids = self.next_batch()
        self.load_state_dict(state)
        return [int(v) for v in ids.tolist()]


@dataclass
class TrainingJob:
    cfg: ExperimentConfig
    seed: int
    model: TinyDropoutClassifier
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    stream: StatefulBatchStream
    optimizer_step: int = 0
    microstep_in_accumulation: int = 0
    global_microstep: int = 0
    last_batch_ids: tuple[int, ...] = ()
    last_loss: float = float("nan")


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    checkpoint_position: str
    include_model: bool = True
    include_optimizer: bool = True
    include_scheduler: bool = True
    include_rng: bool = True
    include_stream: bool = True
    include_counters: bool = True
    include_gradients: bool = False


SPECS = {
    "full_boundary": ScenarioSpec(
        name="full_boundary", checkpoint_position="optimizer_step_boundary"
    ),
    "model_optimizer_only": ScenarioSpec(
        name="model_optimizer_only",
        checkpoint_position="optimizer_step_boundary",
        include_scheduler=False,
        include_rng=False,
        include_stream=False,
        include_counters=False,
    ),
    "omit_rng": ScenarioSpec(
        name="omit_rng",
        checkpoint_position="optimizer_step_boundary",
        include_rng=False,
    ),
    "omit_stream": ScenarioSpec(
        name="omit_stream",
        checkpoint_position="optimizer_step_boundary",
        include_stream=False,
    ),
    "omit_scheduler": ScenarioSpec(
        name="omit_scheduler",
        checkpoint_position="optimizer_step_boundary",
        include_scheduler=False,
    ),
    "full_mid_accumulation": ScenarioSpec(
        name="full_mid_accumulation",
        checkpoint_position="mid_accumulation",
        include_gradients=True,
    ),
    "mid_without_gradients": ScenarioSpec(
        name="mid_without_gradients",
        checkpoint_position="mid_accumulation",
        include_gradients=False,
    ),
}


def configure_deterministic_cpu() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataset(cfg: ExperimentConfig) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.dataset_seed)
    x = torch.randn(cfg.num_examples, cfg.input_dim, generator=generator)
    teacher = torch.randn(cfg.input_dim, cfg.num_classes, generator=generator)
    bias = torch.randn(cfg.num_classes, generator=generator) * 0.2
    y = (x @ teacher + bias).argmax(dim=1)
    return x, y


def make_job(seed: int, cfg: ExperimentConfig | None = None) -> TrainingJob:
    cfg = cfg or ExperimentConfig()
    configure_deterministic_cpu()
    seed_all(seed)
    x, y = make_dataset(cfg)
    model = TinyDropoutClassifier(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma
    )
    stream = StatefulBatchStream(
        x=x,
        y=y,
        order_seed=seed + 10_000,
        batch_size=cfg.batch_size,
    )
    optimizer.zero_grad(set_to_none=True)
    return TrainingJob(
        cfg=cfg,
        seed=seed,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        stream=stream,
    )


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(copy.deepcopy(state["python"]))
    np.random.set_state(copy.deepcopy(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].clone())


def capture_gradients(model: nn.Module) -> dict[str, Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def restore_gradients(model: nn.Module, gradients: dict[str, Tensor | None]) -> None:
    parameters = dict(model.named_parameters())
    if set(parameters) != set(gradients):
        raise ValueError("gradient keys do not match model parameters")
    for name, saved in gradients.items():
        parameters[name].grad = None if saved is None else saved.clone()


def save_checkpoint(job: TrainingJob, spec: ScenarioSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "scenario": spec.name,
        "config": asdict(job.cfg),
    }
    if spec.include_model:
        payload["model"] = copy.deepcopy(job.model.state_dict())
    if spec.include_optimizer:
        payload["optimizer"] = copy.deepcopy(job.optimizer.state_dict())
    if spec.include_scheduler:
        payload["scheduler"] = copy.deepcopy(job.scheduler.state_dict())
    if spec.include_rng:
        payload["rng"] = capture_rng_state()
    if spec.include_stream:
        payload["stream"] = copy.deepcopy(job.stream.state_dict())
    if spec.include_counters:
        payload["counters"] = {
            "optimizer_step": job.optimizer_step,
            "microstep_in_accumulation": job.microstep_in_accumulation,
            "global_microstep": job.global_microstep,
        }
    if spec.include_gradients:
        payload["gradients"] = capture_gradients(job.model)
    return payload


def load_checkpoint(job: TrainingJob, payload: dict[str, Any]) -> None:
    # The scheduler is constructed before optimizer.load_state_dict(), matching
    # the documented PyTorch restore order. Its saved state is loaded afterward.
    if "model" in payload:
        job.model.load_state_dict(payload["model"])
    if "optimizer" in payload:
        job.optimizer.load_state_dict(payload["optimizer"])
    if "scheduler" in payload:
        job.scheduler.load_state_dict(payload["scheduler"])
    if "stream" in payload:
        job.stream.load_state_dict(payload["stream"])
    if "counters" in payload:
        counters = payload["counters"]
        job.optimizer_step = int(counters["optimizer_step"])
        job.microstep_in_accumulation = int(counters["microstep_in_accumulation"])
        job.global_microstep = int(counters["global_microstep"])
    if "gradients" in payload:
        restore_gradients(job.model, payload["gradients"])
    else:
        job.optimizer.zero_grad(set_to_none=True)
    # Restore the process RNG last so object construction and state loading do
    # not consume values from the resumed stochastic stream.
    if "rng" in payload:
        restore_rng_state(payload["rng"])


def run_one_microstep(job: TrainingJob) -> bool:
    cfg = job.cfg
    x, y, ids = job.stream.next_batch()

    # Three deliberately independent stochastic sources. This makes omission
    # failures observable without claiming these exact augmentations are a
    # recommended production recipe.
    python_scale = 1.0 + (2.0 * random.random() - 1.0) * cfg.python_scale_half_width
    numpy_noise = np.random.normal(
        loc=0.0, scale=cfg.numpy_noise_std, size=tuple(x.shape)
    ).astype(np.float32)
    augmented = x * python_scale + torch.from_numpy(numpy_noise)

    logits = job.model(augmented)
    raw_loss = F.cross_entropy(logits, y)
    scaled_loss = raw_loss / cfg.accumulation_steps
    scaled_loss.backward()

    job.last_batch_ids = tuple(int(v) for v in ids.tolist())
    job.last_loss = float(raw_loss.detach().item())
    job.microstep_in_accumulation += 1
    job.global_microstep += 1

    stepped = False
    if job.microstep_in_accumulation == cfg.accumulation_steps:
        job.optimizer.step()
        job.scheduler.step()
        job.optimizer.zero_grad(set_to_none=True)
        job.optimizer_step += 1
        job.microstep_in_accumulation = 0
        stepped = True
    return stepped


def run_until_microstep(
    job: TrainingJob,
    target_global_microstep: int,
    on_optimizer_step: Callable[[TrainingJob], None] | None = None,
) -> None:
    if target_global_microstep < job.global_microstep:
        raise ValueError("target is behind current job state")
    while job.global_microstep < target_global_microstep:
        stepped = run_one_microstep(job)
        if stepped and on_optimizer_step is not None:
            on_optimizer_step(job)


def run_until_optimizer_step(
    job: TrainingJob,
    target_optimizer_step: int,
    on_optimizer_step: Callable[[TrainingJob], None] | None = None,
) -> None:
    if target_optimizer_step < job.optimizer_step:
        raise ValueError("target is behind current job state")
    while job.optimizer_step < target_optimizer_step:
        stepped = run_one_microstep(job)
        if stepped and on_optimizer_step is not None:
            on_optimizer_step(job)


def tensor_sha256(tensor: Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def model_vector(model: nn.Module) -> Tensor:
    return torch.cat(
        [parameter.detach().reshape(-1).cpu() for parameter in model.parameters()]
    )


def model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def nested_tensor_items(value: Any, prefix: str = "") -> list[tuple[str, Tensor]]:
    items: list[tuple[str, Tensor]] = []
    if isinstance(value, Tensor):
        items.append((prefix, value.detach().cpu()))
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda x: str(x)):
            child = f"{prefix}.{key}" if prefix else str(key)
            items.extend(nested_tensor_items(value[key], child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            items.extend(nested_tensor_items(item, child))
    return items


def exact_nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            exact_nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            exact_nested_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    return left == right


def max_nested_tensor_error(left: Any, right: Any) -> float:
    left_items = dict(nested_tensor_items(left))
    right_items = dict(nested_tensor_items(right))
    if set(left_items) != set(right_items):
        return float("inf")
    error = 0.0
    for key in left_items:
        a = left_items[key]
        b = right_items[key]
        if a.shape != b.shape:
            return float("inf")
        if a.numel():
            error = max(error, float((a.to(torch.float64) - b.to(torch.float64)).abs().max().item()))
    return error


def snapshot(job: TrainingJob) -> dict[str, Any]:
    return {
        "optimizer_step": job.optimizer_step,
        "global_microstep": job.global_microstep,
        "microstep_in_accumulation": job.microstep_in_accumulation,
        "model": copy.deepcopy(job.model.state_dict()),
        "optimizer": copy.deepcopy(job.optimizer.state_dict()),
        "scheduler": copy.deepcopy(job.scheduler.state_dict()),
        "stream": copy.deepcopy(job.stream.state_dict()),
        "rng": capture_rng_state(),
        "gradients": capture_gradients(job.model),
        "model_vector": model_vector(job.model),
        "model_hash": model_hash(job.model),
        "next_batch_ids": job.stream.peek_next_ids(),
        "learning_rate": float(job.optimizer.param_groups[0]["lr"]),
        "last_batch_ids": list(job.last_batch_ids),
        "last_loss": job.last_loss,
    }


def compare_snapshots(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    model_error = float(
        (actual["model_vector"].to(torch.float64) - expected["model_vector"].to(torch.float64))
        .abs()
        .max()
        .item()
    )
    return {
        "model_exact": exact_nested_equal(actual["model"], expected["model"]),
        "optimizer_exact": exact_nested_equal(actual["optimizer"], expected["optimizer"]),
        "scheduler_exact": exact_nested_equal(actual["scheduler"], expected["scheduler"]),
        "stream_exact": exact_nested_equal(actual["stream"], expected["stream"]),
        "rng_exact": exact_nested_equal(actual["rng"], expected["rng"]),
        "gradients_exact": exact_nested_equal(actual["gradients"], expected["gradients"]),
        "next_batch_match": actual["next_batch_ids"] == expected["next_batch_ids"],
        "max_parameter_abs_error": model_error,
        "max_optimizer_tensor_abs_error": max_nested_tensor_error(
            actual["optimizer"], expected["optimizer"]
        ),
        "learning_rate_error": abs(actual["learning_rate"] - expected["learning_rate"]),
    }


def make_uninterrupted_reference(seed: int, cfg: ExperimentConfig) -> dict[int, dict[str, Any]]:
    job = make_job(seed, cfg)
    traces: dict[int, dict[str, Any]] = {0: snapshot(job)}

    def record(current: TrainingJob) -> None:
        traces[current.optimizer_step] = snapshot(current)

    run_until_optimizer_step(job, cfg.total_optimizer_steps, on_optimizer_step=record)
    return traces


def interrupt_job(seed: int, cfg: ExperimentConfig, position: str) -> TrainingJob:
    job = make_job(seed, cfg)
    if position == "optimizer_step_boundary":
        run_until_optimizer_step(job, cfg.boundary_interrupt_step)
        if job.microstep_in_accumulation != 0:
            raise AssertionError("boundary checkpoint is not at a clean optimizer-step boundary")
    elif position == "mid_accumulation":
        run_until_microstep(job, cfg.mid_interrupt_microstep_index)
        if job.microstep_in_accumulation != cfg.mid_interrupt_microsteps:
            raise AssertionError("mid checkpoint is not at the configured accumulation position")
    else:
        raise ValueError(f"unknown checkpoint position: {position}")
    return job


def evaluate_scenario(seed: int, scenario: str, cfg: ExperimentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ExperimentConfig()
    spec = SPECS[scenario]
    reference = make_uninterrupted_reference(seed, cfg)
    source = interrupt_job(seed, cfg, spec.checkpoint_position)
    payload = save_checkpoint(source, spec)

    resumed = make_job(seed, cfg)
    load_checkpoint(resumed, payload)

    # The deliberately incomplete model+optimizer payload omits counters. A real
    # orchestration layer still knows the requested target step, so advance it
    # by the remaining number of optimizer updates rather than stopping early.
    if spec.include_counters:
        target_step = cfg.total_optimizer_steps
    else:
        target_step = cfg.total_optimizer_steps - cfg.boundary_interrupt_step

    step_comparisons: list[dict[str, Any]] = []

    def record(current: TrainingJob) -> None:
        logical_step = (
            current.optimizer_step
            if spec.include_counters
            else cfg.boundary_interrupt_step + current.optimizer_step
        )
        expected = reference[logical_step]
        comparison = compare_snapshots(snapshot(current), expected)
        step_comparisons.append(
            {
                "logical_step": logical_step,
                "actual_optimizer_step_counter": current.optimizer_step,
                "model_exact": comparison["model_exact"],
                "max_parameter_abs_error": comparison["max_parameter_abs_error"],
                "actual_learning_rate": float(current.optimizer.param_groups[0]["lr"]),
                "expected_learning_rate": float(expected["learning_rate"]),
                "learning_rate_error": comparison["learning_rate_error"],
                "last_batch_ids": list(current.last_batch_ids),
                "expected_last_batch_ids": expected["last_batch_ids"],
                "batch_match": list(current.last_batch_ids) == expected["last_batch_ids"],
            }
        )

    run_until_optimizer_step(resumed, target_step, on_optimizer_step=record)
    final = compare_snapshots(snapshot(resumed), reference[cfg.total_optimizer_steps])
    first_divergent = next(
        (row["logical_step"] for row in step_comparisons if not row["model_exact"]),
        None,
    )
    exact_resume = all(
        final[key]
        for key in (
            "model_exact",
            "optimizer_exact",
            "scheduler_exact",
            "stream_exact",
            "rng_exact",
            "gradients_exact",
            "next_batch_match",
        )
    )
    return {
        "seed": seed,
        "scenario": scenario,
        "checkpoint_position": spec.checkpoint_position,
        "payload_keys": sorted(payload),
        "interrupt_optimizer_step": source.optimizer_step,
        "interrupt_microstep_in_accumulation": source.microstep_in_accumulation,
        "reference_final_model_hash": reference[cfg.total_optimizer_steps]["model_hash"],
        "resumed_final_model_hash": model_hash(resumed.model),
        "first_divergent_optimizer_step": first_divergent,
        "exact_resume": exact_resume,
        **final,
        "step_trace": step_comparisons,
    }


def run_experiment(
    output_dir: Path,
    seeds: Iterable[int] = SEEDS,
    scenarios: Iterable[str] = SCENARIOS,
    cfg: ExperimentConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or ExperimentConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    raw_lines: list[str] = []
    for seed in seeds:
        for scenario in scenarios:
            result = evaluate_scenario(seed, scenario, cfg)
            rows.append(result)
            raw_lines.append(
                " ".join(
                    [
                        f"seed={seed}",
                        f"scenario={scenario}",
                        f"exact={str(result['exact_resume']).lower()}",
                        f"first_divergent_step={result['first_divergent_optimizer_step']}",
                        f"max_param_error={result['max_parameter_abs_error']:.9e}",
                        f"next_batch_match={str(result['next_batch_match']).lower()}",
                        f"lr_error={result['learning_rate_error']:.9e}",
                    ]
                )
            )

    csv_fields = [
        "seed",
        "scenario",
        "checkpoint_position",
        "exact_resume",
        "model_exact",
        "optimizer_exact",
        "scheduler_exact",
        "stream_exact",
        "rng_exact",
        "gradients_exact",
        "next_batch_match",
        "max_parameter_abs_error",
        "max_optimizer_tensor_abs_error",
        "learning_rate_error",
        "first_divergent_optimizer_step",
        "interrupt_optimizer_step",
        "interrupt_microstep_in_accumulation",
        "reference_final_model_hash",
        "resumed_final_model_hash",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in csv_fields})

    (output_dir / "raw-output.txt").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    (output_dir / "step-traces.json").write_text(
        json.dumps(
            [
                {
                    "seed": row["seed"],
                    "scenario": row["scenario"],
                    "step_trace": row["step_trace"],
                }
                for row in rows
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    scenario_summary: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        subset = [row for row in rows if row["scenario"] == scenario]
        scenario_summary[scenario] = {
            "runs": len(subset),
            "exact_runs": sum(bool(row["exact_resume"]) for row in subset),
            "exact_rate": sum(bool(row["exact_resume"]) for row in subset) / len(subset),
            "max_parameter_abs_error_min": min(row["max_parameter_abs_error"] for row in subset),
            "max_parameter_abs_error_median": float(
                np.median([row["max_parameter_abs_error"] for row in subset])
            ),
            "max_parameter_abs_error_max": max(row["max_parameter_abs_error"] for row in subset),
            "first_divergent_steps": [row["first_divergent_optimizer_step"] for row in subset],
            "all_next_batch_match": all(row["next_batch_match"] for row in subset),
        }

    summary = {
        "status": "PASS",
        "experiment": "checkpoint_resume_equivalence",
        "configuration": asdict(cfg),
        "seeds": list(seeds),
        "scenarios": list(scenarios),
        "run_count": len(rows),
        "scenario_summary": scenario_summary,
        "acceptance_checks": {
            "full_boundary_all_exact": scenario_summary["full_boundary"]["exact_runs"] == len(list(seeds)),
            "full_mid_accumulation_all_exact": scenario_summary["full_mid_accumulation"]["exact_runs"] == len(list(seeds)),
            "model_optimizer_only_all_fail": scenario_summary["model_optimizer_only"]["exact_runs"] == 0,
            "omit_rng_all_fail": scenario_summary["omit_rng"]["exact_runs"] == 0,
            "omit_stream_all_fail": scenario_summary["omit_stream"]["exact_runs"] == 0,
            "omit_scheduler_all_fail": scenario_summary["omit_scheduler"]["exact_runs"] == 0,
            "mid_without_gradients_all_fail": scenario_summary["mid_without_gradients"]["exact_runs"] == 0,
        },
        "claim_boundary": "Exact bitwise equality is asserted only for this deterministic CPU fixture and software environment.",
    }
    if not all(summary["acceptance_checks"].values()):
        summary["status"] = "FAIL"
    (output_dir / "run-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    environment = runtime_environment()
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    schema = {
        "successful_boundary_payload_fields": [
            "model", "optimizer", "scheduler", "rng", "stream", "counters"
        ],
        "independently_ablated_boundary_fields": ["scheduler", "rng", "stream"],
        "combined_anti_control_omissions": [
            "scheduler", "rng", "stream", "counters"
        ],
        "successful_mid_accumulation_additional_fields": ["gradients"],
        "independently_ablated_mid_accumulation_fields": ["gradients"],
        "retained_but_not_independently_ablated": [
            "counters", "microstep_in_accumulation", "global_microstep"
        ],
        "rng_components_in_fixture": ["python", "numpy", "torch_cpu"],
        "stream_components_in_fixture": [
            "epoch", "cursor", "current_permutation", "permutation_generator_state"
        ],
        "comparison_surface": [
            "model", "optimizer", "scheduler", "stream", "rng", "gradients", "next_batch_ids"
        ],
        "not_claimed": [
            "CUDA RNG capture", "per-rank distributed state", "FSDP optimizer re-sharding",
            "DataLoader worker queues", "elastic world-size changes"
        ],
    }
    (output_dir / "checkpoint-schema.json").write_text(
        json.dumps(schema, indent=2) + "\n", encoding="utf-8"
    )
    print("\n".join(raw_lines))
    print(json.dumps(summary["acceptance_checks"], sort_keys=True))
    if summary["status"] != "PASS":
        raise RuntimeError("experiment acceptance checks failed")
    return summary



def runtime_environment() -> dict[str, Any]:
    """Return the runtime facts retained with every experiment run."""
    configure_deterministic_cpu()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "device": "cpu",
    }


def print_fixture_inspection(seed: int) -> None:
    """Print an inspectable model/config/checkpoint snapshot for the article."""
    cfg = ExperimentConfig()
    job = make_job(seed, cfg)
    env = runtime_environment()
    print("runtime")
    print(f"  python={platform.python_version()}")
    print(f"  torch={env['torch']} numpy={env['numpy']} device={env['device']}")
    print(
        "  deterministic_algorithms="
        f"{str(env['deterministic_algorithms_enabled']).lower()} "
        f"torch_threads={env['torch_num_threads']}"
    )
    print("model")
    for line_value in str(job.model).splitlines():
        print(f"  {line_value}")
    print("parameters")
    total = 0
    for name, parameter in job.model.named_parameters():
        count = parameter.numel()
        total += count
        print(f"  {name:<14} shape={str(tuple(parameter.shape)):<10} count={count}")
    print(f"  trainable_total={total}")
    print("training")
    print(
        f"  batch_shape=({cfg.batch_size}, {cfg.input_dim}) "
        f"logits_shape=({cfg.batch_size}, {cfg.num_classes})"
    )
    print(
        f"  optimizer=AdamW(lr={cfg.learning_rate}, weight_decay={cfg.weight_decay})"
    )
    print(
        "  scheduler="
        f"StepLR(step_size={cfg.scheduler_step_size}, gamma={cfg.scheduler_gamma})"
    )
    print(
        f"  accumulation_steps={cfg.accumulation_steps} "
        f"checkpoint_after_optimizer_step={cfg.boundary_interrupt_step}"
    )
    boundary_job = interrupt_job(seed, cfg, "optimizer_step_boundary")
    boundary_payload = save_checkpoint(boundary_job, SPECS["full_boundary"])
    mid_job = interrupt_job(seed, cfg, "mid_accumulation")
    mid_payload = save_checkpoint(mid_job, SPECS["full_mid_accumulation"])
    print("checkpoint payloads")
    print(f"  boundary_keys={sorted(boundary_payload)}")
    print(f"  mid_accumulation_keys={sorted(mid_payload)}")


def print_seed_matrix(seed: int) -> None:
    """Run every intervention for one seed and print a compact terminal matrix."""
    print(f"seed={seed}")
    print("scenario                    exact  first  next_batch  lr_error   max_param_error")
    for scenario in SCENARIOS:
        result = evaluate_scenario(seed, scenario)
        first = result["first_divergent_optimizer_step"]
        print(
            f"{scenario:<27} "
            f"{'yes' if result['exact_resume'] else 'no':<5}  "
            f"{str(first) if first is not None else '--':>5}  "
            f"{'same' if result['next_batch_match'] else 'DIFF':>10}  "
            f"{result['learning_rate_error']:.3e}  "
            f"{result['max_parameter_abs_error']:.3e}"
        )


def print_scenario_trace(seed: int, scenario: str) -> None:
    """Print the per-update differential trace for one intervention."""
    result = evaluate_scenario(seed, scenario)
    print(f"seed={seed} scenario={scenario}")
    print(
        "step  control_lr  resumed_lr  lr_error     batch  model  max_param_error"
    )
    for row in result["step_trace"]:
        print(
            f"{row['logical_step']:>4}  "
            f"{row['expected_learning_rate']:.6f}    "
            f"{row['actual_learning_rate']:.6f}    "
            f"{row['learning_rate_error']:.3e}  "
            f"{'match' if row['batch_match'] else 'DIFF':>5}  "
            f"{'match' if row['model_exact'] else 'DIFF':>5}  "
            f"{row['max_parameter_abs_error']:.3e}"
        )
    print(
        f"final_exact={str(result['exact_resume']).lower()} "
        f"first_divergent_step={result['first_divergent_optimizer_step']}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run all seeds and scenarios")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    one_parser = subparsers.add_parser("one", help="run one seed/scenario")
    one_parser.add_argument("--seed", type=int, required=True)
    one_parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    inspect_parser = subparsers.add_parser(
        "inspect", help="print the concrete model, runtime, and checkpoint payloads"
    )
    inspect_parser.add_argument("--seed", type=int, default=SEEDS[0])
    matrix_parser = subparsers.add_parser(
        "matrix", help="run every intervention for one seed"
    )
    matrix_parser.add_argument("--seed", type=int, required=True)
    trace_parser = subparsers.add_parser(
        "trace", help="print a per-update differential trace for one scenario"
    )
    trace_parser.add_argument("--seed", type=int, required=True)
    trace_parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        run_experiment(args.output_dir)
        return 0
    if args.command == "inspect":
        print_fixture_inspection(args.seed)
        return 0
    if args.command == "matrix":
        print_seed_matrix(args.seed)
        return 0
    if args.command == "trace":
        print_scenario_trace(args.seed, args.scenario)
        return 0
    result = evaluate_scenario(args.seed, args.scenario)
    serializable = {key: value for key, value in result.items() if key != "step_trace"}
    print(json.dumps(serializable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

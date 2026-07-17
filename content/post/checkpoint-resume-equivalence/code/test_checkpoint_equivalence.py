from __future__ import annotations

import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))
import checkpoint_equivalence as module

ExperimentConfig = module.ExperimentConfig
evaluate_scenario = module.evaluate_scenario
SEEDS = module.SEEDS


@pytest.mark.parametrize("seed", SEEDS)
def test_full_boundary_resume_is_bitwise_exact(seed: int) -> None:
    result = evaluate_scenario(seed, "full_boundary")
    assert result["exact_resume"]
    assert result["max_parameter_abs_error"] == 0.0
    assert result["first_divergent_optimizer_step"] is None


@pytest.mark.parametrize("seed", SEEDS)
def test_full_mid_accumulation_resume_is_bitwise_exact(seed: int) -> None:
    result = evaluate_scenario(seed, "full_mid_accumulation")
    assert result["exact_resume"]
    assert result["gradients_exact"]
    assert result["max_parameter_abs_error"] == 0.0


@pytest.mark.parametrize(
    "scenario",
    [
        "model_optimizer_only",
        "omit_rng",
        "omit_stream",
        "omit_scheduler",
        "mid_without_gradients",
    ],
)
def test_planted_omissions_fail_for_every_seed(scenario: str) -> None:
    results = [evaluate_scenario(seed, scenario) for seed in SEEDS]
    assert all(not result["exact_resume"] for result in results)
    assert all(result["max_parameter_abs_error"] > 0.0 for result in results)
    assert all(result["first_divergent_optimizer_step"] is not None for result in results)


def test_scheduler_omission_delays_divergence_until_schedule_boundary() -> None:
    result = evaluate_scenario(SEEDS[0], "omit_scheduler")
    # The optimizer loads the current LR, so the first resumed updates can match.
    # Divergence appears when the unsaved scheduler clock reaches a different decay.
    assert result["first_divergent_optimizer_step"] is not None
    assert result["first_divergent_optimizer_step"] > ExperimentConfig().boundary_interrupt_step + 1


def test_rng_omission_keeps_next_batch_but_changes_first_update() -> None:
    result = evaluate_scenario(SEEDS[0], "omit_rng")
    assert result["next_batch_match"]
    assert result["first_divergent_optimizer_step"] == ExperimentConfig().boundary_interrupt_step + 1


def test_stream_omission_changes_next_batch_and_first_update() -> None:
    result = evaluate_scenario(SEEDS[0], "omit_stream")
    assert not result["next_batch_match"]
    assert result["first_divergent_optimizer_step"] == ExperimentConfig().boundary_interrupt_step + 1


def test_inspect_command_exposes_actual_model_and_payloads(capsys: pytest.CaptureFixture[str]) -> None:
    assert module.main(["inspect", "--seed", "11"]) == 0
    output = capsys.readouterr().out
    assert "TinyDropoutClassifier(" in output
    assert "Linear(in_features=8, out_features=16" in output
    assert "trainable_total=195" in output
    assert "boundary_keys=" in output
    assert "mid_accumulation_keys=" in output


def test_seed_matrix_prints_all_seven_interventions(capsys: pytest.CaptureFixture[str]) -> None:
    assert module.main(["matrix", "--seed", "11"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == len(module.SCENARIOS) + 2
    assert any(line.startswith("full_boundary") and "yes" in line for line in lines)
    assert any(line.startswith("omit_scheduler") and "no" in line for line in lines)


def test_scheduler_trace_shows_lr_branch_before_parameter_branch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert module.main(["trace", "--seed", "11", "--scenario", "omit_scheduler"]) == 0
    output = capsys.readouterr().out
    step_rows = {
        int(parts[0]): parts
        for line in output.splitlines()
        if (parts := line.split()) and parts[0].isdigit()
    }
    assert step_rows[8][3] != "0.000e+00"
    assert step_rows[8][5] == "match"
    assert step_rows[9][5] == "DIFF"

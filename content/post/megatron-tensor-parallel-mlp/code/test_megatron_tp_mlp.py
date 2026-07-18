from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "megatron_tp_mlp.py"


def run(mode: str):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    p = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", str(SCRIPT), "--mode", mode],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    start = p.stdout.find("{")
    return json.loads(p.stdout[start:])


def test_equivalence_forward_backward_and_update():
    r = run("equivalence")
    assert r["passed"]
    for key, value in r.items():
        if key.endswith("diff"):
            assert value < 1e-6


def test_missing_forward_reduce_is_detected_immediately():
    r = run("missing_forward_reduce")
    assert not r["passed"]
    assert r["forward_max_abs_diff"] > 1e-2
    assert r["post_step_weight_max_abs_diff"] > 1e-4


def test_missing_backward_reduce_preserves_forward_but_breaks_upstream_gradients():
    r = run("missing_backward_reduce")
    assert not r["passed"]
    assert r["forward_max_abs_diff"] < 1e-6
    assert r["loss_abs_diff"] < 1e-6
    assert r["input_grad_max_abs_diff"] > 1e-2
    assert r["norm_grad_max_abs_diff"] > 1e-2


def test_inspect_reports_transformer_shapes_and_collectives():
    r = run("inspect")
    assert r["world_size"] == 2
    assert r["dense_shapes"]["x"] == [3, 2, 8]
    assert r["rank_local_shapes"]["swiglu"] == [3, 2, 6]
    assert len(r["collectives"]) == 2

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "simulate_admission.py"
SPEC = importlib.util.spec_from_file_location("simulate_admission", MODULE_PATH)
simulate_admission = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = simulate_admission
SPEC.loader.exec_module(simulate_admission)

Config = simulate_admission.Config
Request = simulate_admission.Request
simulate = simulate_admission.simulate

def clone(reqs):
    return [Request(r.request_id, r.arrival_step, r.prompt_tokens, r.output_tokens) for r in reqs]

def test_full_input_reservation_avoids_over_admission():
    reqs = [
        Request("a", 0, 48, 8),
        Request("b", 0, 48, 8),
        Request("c", 0, 48, 8),
    ]
    naive = Config(total_blocks=6, block_size=16, max_num_batched_tokens=64,
                   max_num_seqs=3, reserve_full_input=False, max_steps=100)
    guarded = Config(total_blocks=6, block_size=16, max_num_batched_tokens=64,
                     max_num_seqs=3, reserve_full_input=True, max_steps=100)
    _, naive_metrics = simulate(clone(reqs), naive)
    _, guarded_metrics = simulate(clone(reqs), guarded)
    assert guarded_metrics["preemptions"] <= naive_metrics["preemptions"]

def test_oversized_prompt_is_rejected():
    reqs = [Request("too-large", 0, 200, 8)]
    cfg = Config(total_blocks=4, block_size=16, max_steps=10)
    _, metrics = simulate(reqs, cfg)
    assert metrics["rejected_oversized"] == 1

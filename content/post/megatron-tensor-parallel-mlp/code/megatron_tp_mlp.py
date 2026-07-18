from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

SEQ, BATCH, HIDDEN, FFN = 3, 2, 8, 12
LR = 0.05
EPS = 1e-6


class CopyToTensorParallelRegion(torch.autograd.Function):
    """Identity in forward; sum replicated-input gradients in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        grad = grad_output.clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return (grad,)


class CopyWithoutBackwardReduce(torch.autograd.Function):
    """Deliberately wrong: identity in both directions."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (grad_output,)


class ReduceFromTensorParallelRegion(torch.autograd.Function):
    """Sum partial outputs in forward; identity in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (grad_output,)


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return x * scale * weight


@dataclass
class RunResult:
    mode: str
    world_size: int
    forward_max_abs_diff: float
    loss_abs_diff: float
    input_grad_max_abs_diff: float
    norm_grad_max_abs_diff: float
    gate_up_grad_max_abs_diff: float
    down_grad_max_abs_diff: float
    down_bias_grad_max_abs_diff: float
    post_step_weight_max_abs_diff: float
    passed: bool


def init_dist() -> tuple[int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return dist.get_rank(), dist.get_world_size()


def make_fixture() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(31)
    return {
        "x": torch.randn(SEQ, BATCH, HIDDEN, generator=g),
        "target": torch.randn(SEQ, BATCH, HIDDEN, generator=g),
        "norm_weight": 1.0 + 0.05 * torch.randn(HIDDEN, generator=g),
        # [gate_or_up, ffn_feature, hidden]
        "w_gate_up": torch.randn(2, FFN, HIDDEN, generator=g) / HIDDEN**0.5,
        "w_down": torch.randn(HIDDEN, FFN, generator=g) / FFN**0.5,
        "b_down": 0.01 * torch.randn(HIDDEN, generator=g),
    }


def dense_forward(x, norm_weight, w_gate_up, w_down, b_down):
    residual = x
    normed = rms_norm(x, norm_weight)
    gate = F.linear(normed, w_gate_up[0])
    up = F.linear(normed, w_gate_up[1])
    hidden = F.silu(gate) * up
    return residual + F.linear(hidden, w_down, b_down)


def dense_step(fx: dict[str, torch.Tensor]):
    params = {k: fx[k].clone().requires_grad_(True) for k in ["x", "norm_weight", "w_gate_up", "w_down", "b_down"]}
    out = dense_forward(**params)
    loss = F.mse_loss(out, fx["target"])
    loss.backward()
    grads = {k: v.grad.detach().clone() for k, v in params.items()}
    updated = {k: v.detach() - LR * v.grad for k, v in params.items() if k != "x"}
    return out.detach(), loss.detach(), grads, updated


def tp_step(fx: dict[str, torch.Tensor], rank: int, world_size: int, mode: str):
    assert FFN % world_size == 0
    local_ffn = FFN // world_size
    lo, hi = rank * local_ffn, (rank + 1) * local_ffn

    x = fx["x"].clone().requires_grad_(True)
    norm_weight = fx["norm_weight"].clone().requires_grad_(True)
    w_gate_up = fx["w_gate_up"][:, lo:hi, :].clone().requires_grad_(True)
    w_down = fx["w_down"][:, lo:hi].clone().requires_grad_(True)
    b_down = fx["b_down"].clone().requires_grad_(True)

    residual = x
    normed = rms_norm(x, norm_weight)
    copier = CopyWithoutBackwardReduce if mode == "missing_backward_reduce" else CopyToTensorParallelRegion
    normed_parallel = copier.apply(normed)

    gate_local = F.linear(normed_parallel, w_gate_up[0])
    up_local = F.linear(normed_parallel, w_gate_up[1])
    hidden_local = F.silu(gate_local) * up_local
    partial = F.linear(hidden_local, w_down, None)

    if mode == "missing_forward_reduce":
        mlp_out = partial
    else:
        mlp_out = ReduceFromTensorParallelRegion.apply(partial)

    out = residual + mlp_out + b_down
    loss = F.mse_loss(out, fx["target"])
    loss.backward()

    grads = {
        "x": x.grad.detach(),
        "norm_weight": norm_weight.grad.detach(),
        "w_gate_up": w_gate_up.grad.detach(),
        "w_down": w_down.grad.detach(),
        "b_down": b_down.grad.detach(),
    }
    updated = {
        "norm_weight": norm_weight.detach() - LR * norm_weight.grad,
        "w_gate_up": w_gate_up.detach() - LR * w_gate_up.grad,
        "w_down": w_down.detach() - LR * w_down.grad,
        "b_down": b_down.detach() - LR * b_down.grad,
    }
    return out.detach(), loss.detach(), grads, updated


def gather_cat(t: torch.Tensor, dim: int) -> torch.Tensor:
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return torch.cat(gathered, dim=dim)


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def run(mode: str) -> RunResult | None:
    rank, world_size = init_dist()
    if world_size != 2:
        raise RuntimeError(f"This fixture expects exactly 2 ranks, got {world_size}")

    fx = make_fixture()
    dense_out, dense_loss, dense_grads, dense_updated = dense_step(fx)
    tp_out, tp_loss, tp_grads, tp_updated = tp_step(fx, rank, world_size, mode)

    gate_up_grad = gather_cat(tp_grads["w_gate_up"], dim=1)
    down_grad = gather_cat(tp_grads["w_down"], dim=1)
    gate_up_new = gather_cat(tp_updated["w_gate_up"], dim=1)
    down_new = gather_cat(tp_updated["w_down"], dim=1)

    if rank != 0:
        return None

    diffs = {
        "forward": maxdiff(tp_out, dense_out),
        "loss": float((tp_loss - dense_loss).abs().item()),
        "input": maxdiff(tp_grads["x"], dense_grads["x"]),
        "norm": maxdiff(tp_grads["norm_weight"], dense_grads["norm_weight"]),
        "gate_up": maxdiff(gate_up_grad, dense_grads["w_gate_up"]),
        "down": maxdiff(down_grad, dense_grads["w_down"]),
        "down_bias": maxdiff(tp_grads["b_down"], dense_grads["b_down"]),
    }
    post = max(
        maxdiff(tp_updated["norm_weight"], dense_updated["norm_weight"]),
        maxdiff(gate_up_new, dense_updated["w_gate_up"]),
        maxdiff(down_new, dense_updated["w_down"]),
        maxdiff(tp_updated["b_down"], dense_updated["b_down"]),
    )
    correct = mode == "equivalence" and max(diffs.values()) < 1e-6 and post < 1e-6
    return RunResult(
        mode=mode,
        world_size=world_size,
        forward_max_abs_diff=diffs["forward"],
        loss_abs_diff=diffs["loss"],
        input_grad_max_abs_diff=diffs["input"],
        norm_grad_max_abs_diff=diffs["norm"],
        gate_up_grad_max_abs_diff=diffs["gate_up"],
        down_grad_max_abs_diff=diffs["down"],
        down_bias_grad_max_abs_diff=diffs["down_bias"],
        post_step_weight_max_abs_diff=post,
        passed=correct,
    )


def inspect() -> None:
    rank, world_size = init_dist()
    if rank != 0:
        return
    payload = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "backend": dist.get_backend(),
        "world_size": world_size,
        "block_path": "pre-RMSNorm -> fused gate/up -> SwiGLU -> down projection -> residual add",
        "dense_shapes": {
            "x": [SEQ, BATCH, HIDDEN],
            "norm_weight": [HIDDEN],
            "w_gate_up": [2, FFN, HIDDEN],
            "w_down": [HIDDEN, FFN],
            "output": [SEQ, BATCH, HIDDEN],
        },
        "rank_local_shapes": {
            "gate_up_weight": [2, FFN // world_size, HIDDEN],
            "gate": [SEQ, BATCH, FFN // world_size],
            "up": [SEQ, BATCH, FFN // world_size],
            "swiglu": [SEQ, BATCH, FFN // world_size],
            "down_weight": [HIDDEN, FFN // world_size],
            "partial_output": [SEQ, BATCH, HIDDEN],
        },
        "collectives": [
            "row-parallel forward: all-reduce partial down-projection outputs",
            "column-parallel backward: all-reduce hidden-state gradient before RMSNorm",
        ],
        "not_included": [
            "self-attention",
            "sequence parallelism",
            "CUDA/NCCL",
            "mixed precision",
            "fused kernels",
            "Megatron-Core classes",
        ],
    }
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["inspect", "equivalence", "missing_forward_reduce", "missing_backward_reduce"],
        required=True,
    )
    args = parser.parse_args()
    try:
        if args.mode == "inspect":
            inspect()
        else:
            result = run(args.mode)
            if result is not None:
                print(json.dumps(asdict(result), indent=2))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Minimal exact loss reduction for variable token counts.

The core contract is two sufficient statistics:
  numerator   = sum of unreduced valid-token losses
  denominator = number (or total weight) of valid tokens
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

def local_stats(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    losses = F.cross_entropy(logits, targets, reduction="none", ignore_index=IGNORE_INDEX)
    valid = targets.ne(IGNORE_INDEX)
    return losses[valid].sum(), valid.sum().to(dtype=losses.dtype)

def exact_global_loss(shards: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    numerators, denominators = zip(*(local_stats(logits, targets) for logits, targets in shards))
    numerator = torch.stack(list(numerators)).sum()
    denominator = torch.stack(list(denominators)).sum()
    if denominator.item() == 0:
        return numerator * 0.0
    return numerator / denominator

def wrong_mean_of_rank_means(shards: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    means=[]
    for logits, targets in shards:
        n,d=local_stats(logits,targets)
        if d.item()>0:
            means.append(n/d)
    return torch.stack(means).mean() if means else sum(x[0].sum()*0 for x in shards)

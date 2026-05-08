"""Group-relative advantage from the GRPO surrogate (DeepSeekMath).

For each group of G rollouts on the same prompt, advantage hat A_i is the
z-scored reward within the group. Only the per-rollout scalar reward is
replaced by AtmoZero — the GRPO objective itself is unchanged.
"""
from __future__ import annotations

import torch


def group_relative_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """rewards: (B, G) tensor; returns same-shape z-scored advantages."""
    if rewards.ndim != 2:
        raise ValueError("rewards must be (B, G)")
    mu = rewards.mean(dim=-1, keepdim=True)
    sigma = rewards.std(dim=-1, keepdim=True).clamp_min(eps)
    return (rewards - mu) / sigma

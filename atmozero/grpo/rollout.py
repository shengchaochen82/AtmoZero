"""Group sampling helper for GRPO rollouts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Union

import torch
import torch.nn.functional as F


@dataclass
class RolloutBatch:
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    response_log_probs: torch.Tensor
    response_mask: torch.Tensor
    rewards: torch.Tensor
    precision: torch.Tensor
    infeasibility: torch.Tensor
    captions: List[List[str]]


@dataclass
class RewardOut:
    """Bundle returned by reward_fn so the trainer can apply the cold-start
    gate and Proposer veto with the canonical quantities."""
    reward: float
    precision: float
    infeasible: bool = False


def sample_group(
    solver,
    tokenizer,
    prompt_ids: torch.Tensor,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    reward_fn: Callable[[List[str], int], Union["RewardOut", float]],
) -> RolloutBatch:
    B, L = prompt_ids.shape
    expanded_prompt = prompt_ids.repeat_interleave(group_size, dim=0)
    output = solver.generate(
        input_ids=expanded_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequences = output.sequences
    response_ids = sequences[:, L:]
    captions_flat = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    with torch.no_grad():
        logits = solver(sequences).logits[:, L - 1:-1]
        log_probs = F.log_softmax(logits, dim=-1)
        gather = response_ids.unsqueeze(-1)
        chosen_lp = log_probs.gather(-1, gather).squeeze(-1)
    response_lp = chosen_lp.view(B, group_size, -1)

    rewards = torch.zeros(B, group_size)
    precision = torch.zeros(B, group_size)
    infeasibility = torch.zeros(B, group_size)
    captions: List[List[str]] = [[None] * group_size for _ in range(B)]
    for b in range(B):
        for g in range(group_size):
            idx = b * group_size + g
            text = captions_flat[idx]
            captions[b][g] = text
            out = reward_fn([text], b)
            if hasattr(out, "reward"):
                rewards[b, g] = float(out.reward)
                precision[b, g] = float(getattr(out, "precision", 0.0) or 0.0)
                infeasibility[b, g] = float(bool(getattr(out, "infeasible", False)))
            else:
                rewards[b, g] = float(out)
                precision[b, g] = float(out > 0.0)
                infeasibility[b, g] = 0.0

    response_mask = (response_ids != tokenizer.pad_token_id).view(B, group_size, -1).float()
    return RolloutBatch(
        prompt_ids=prompt_ids,
        response_ids=response_ids.view(B, group_size, -1),
        response_log_probs=response_lp,
        response_mask=response_mask,
        rewards=rewards,
        precision=precision,
        infeasibility=infeasibility,
        captions=captions,
    )

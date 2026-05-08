"""AtmoZero GRPO trainer with bi-level Proposer-Solver objective."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from atmozero.grpo.advantage import group_relative_advantage
from atmozero.grpo.rollout import RolloutBatch, sample_group
from atmozero.policies.coldstart import ColdStartController, COLD_START_THRESHOLD
from atmozero.policies.proposer import Proposer
from atmozero.reward.shaping import ShapingSchedule


@dataclass
class GRPOConfig:
    total_steps: int = 2450
    group_size: int = 8
    batch_size: int = 32
    learning_rate: float = 4.8e-6
    proposer_lr: float = 1.0e-4
    weight_decay: float = 0.08
    grad_clip: float = 1.0
    kl_coef: float = 0.025
    clip_epsilon: float = 0.20
    entropy_bonus: float = 0.05
    max_new_tokens: int = 384
    temperature: float = 1.0
    top_p: float = 0.95
    log_every: int = 10
    save_every: int = 250
    output_dir: str = "runs/atmozero/default"
    cold_start_threshold: float = COLD_START_THRESHOLD


class AtmoZeroGRPOTrainer:
    def __init__(
        self,
        solver: nn.Module,
        ref_solver: nn.Module,
        tokenizer,
        proposer: Proposer,
        reward_fn: Callable,
        cfg: GRPOConfig,
        shaping: Optional[ShapingSchedule] = None,
    ):
        self.solver = solver
        self.ref_solver = ref_solver
        for p in self.ref_solver.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        self.proposer = proposer
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.shaping = shaping or ShapingSchedule()
        self.cold = ColdStartController(threshold=cfg.cold_start_threshold)
        self.opt_solver = AdamW(
            [p for p in self.solver.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.opt_proposer = AdamW(
            [p for p in self.proposer.parameters() if p.requires_grad],
            lr=cfg.proposer_lr,
        )
        self._step = 0

    def step(self, prompt_ids: torch.Tensor, x_batch, regimes) -> Dict[str, float]:
        batch: RolloutBatch = sample_group(
            self.solver,
            self.tokenizer,
            prompt_ids,
            group_size=self.cfg.group_size,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            reward_fn=lambda txt, b: self.reward_fn(txt, x_batch[b], regimes[b], self._step),
        )

        advantage = group_relative_advantage(batch.rewards).to(batch.response_log_probs.device)

        with torch.no_grad():
            ref_lp = _ref_log_probs(self.ref_solver, batch)
        ratio = torch.exp(batch.response_log_probs - ref_lp)
        eps = self.cfg.clip_epsilon
        clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        per_token_obj = torch.minimum(
            ratio * advantage.unsqueeze(-1), clipped * advantage.unsqueeze(-1)
        )
        per_token_obj = (per_token_obj * batch.response_mask).sum(-1) / batch.response_mask.sum(-1).clamp_min(1)
        actor_loss = -per_token_obj.mean()

        kl = ((batch.response_log_probs - ref_lp).exp() - (batch.response_log_probs - ref_lp) - 1)
        kl = (kl * batch.response_mask).sum(-1) / batch.response_mask.sum(-1).clamp_min(1)
        kl_loss = self.cfg.kl_coef * kl.mean()

        loss = actor_loss + kl_loss
        self.opt_solver.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.solver.parameters(), self.cfg.grad_clip)
        self.opt_solver.step()

        verifier_precision = float(batch.precision.mean().item())
        if self.cold.update(self._step, verifier_precision):
            self.proposer.engage()

        prop_loss = torch.zeros((), device=batch.rewards.device)
        if self.cold.engaged_at_step is not None:
            window_idx, regime_idx = regimes
            log_probs = self.proposer()
            flat_idx = window_idx * self.proposer.cfg.n_regimes + regime_idx
            chosen_lp = log_probs[flat_idx]
            mean_reward = batch.rewards.mean(dim=-1).detach()
            prop_loss = (chosen_lp * mean_reward).mean()
            prop_loss = prop_loss - self.cfg.entropy_bonus * self.proposer.entropy()
            gamma = float(self.proposer.cfg.infeasibility_penalty)
            if gamma > 0.0:
                infeas_per_window = batch.infeasibility.mean(dim=-1).detach().to(chosen_lp.device)
                prop_loss = prop_loss + gamma * (chosen_lp * infeas_per_window).mean()
            self.opt_proposer.zero_grad(set_to_none=True)
            prop_loss.backward()
            self.opt_proposer.step()

        self._step += 1
        return {
            "step": self._step,
            "actor_loss": float(actor_loss.detach()),
            "kl": float(kl.mean().detach()),
            "reward_mean": float(batch.rewards.mean()),
            "verifier_precision": verifier_precision,
            "proposer_engaged": self.cold.engaged_at_step is not None,
            "proposer_loss": float(prop_loss.detach()),
        }

    def fit(self, prompt_loader) -> None:
        for step in range(self.cfg.total_steps):
            prompt_ids, x_batch, regimes = next(prompt_loader)
            stats = self.step(prompt_ids, x_batch, regimes)
            if step % self.cfg.log_every == 0:
                print(stats)
            if step % self.cfg.save_every == 0 and step > 0:
                self.save(Path(self.cfg.output_dir) / f"checkpoint-{step}")

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.solver.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.proposer.state_dict(), path / "proposer.pt")


def _ref_log_probs(ref_solver, batch: RolloutBatch) -> torch.Tensor:
    B, G, T = batch.response_ids.shape
    full = torch.cat([batch.prompt_ids.unsqueeze(1).expand(-1, G, -1), batch.response_ids], dim=-1)
    full = full.reshape(B * G, -1)
    with torch.no_grad():
        logits = ref_solver(full).logits[:, batch.prompt_ids.shape[-1] - 1:-1]
        lp = F.log_softmax(logits, dim=-1)
        gather = batch.response_ids.reshape(B * G, -1).unsqueeze(-1)
        ref_lp = lp.gather(-1, gather).squeeze(-1).view(B, G, -1)
    return ref_lp

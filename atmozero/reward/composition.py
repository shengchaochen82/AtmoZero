"""Reward composition R(c, x) = sum_k w_k S_k - lambda U + mu C + nu D."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from atmozero.verifier.grader import VerifierSignals


@dataclass
class RewardConfig:
    """Defaults: lambda=2.0, mu=0.22, nu=0.06, w_k=1/7."""
    w: Dict[int, float] = field(default_factory=lambda: {k: 1.0 / 7.0 for k in range(1, 8)})
    lam: float = 2.0
    mu: float = 0.22
    nu: float = 0.06

    def per_family_weight(self, fid: int) -> float:
        return self.w.get(fid, 1.0 / 7.0)


def specificity_bonus(caption: str) -> float:
    """Type-token ratio over content tokens; bounded to [0, 1]."""
    tokens = [t for t in caption.lower().split() if t.isalpha()]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compose_reward(
    signals: VerifierSignals,
    caption: str,
    cfg: RewardConfig,
) -> float:
    """R(c, x). Only the -lambda U term can drive R strictly negative."""
    R = sum(cfg.per_family_weight(fid) * s for fid, s in signals.S.items())
    R -= cfg.lam * signals.U
    R += cfg.mu * signals.C
    R += cfg.nu * specificity_bonus(caption)
    return float(R)

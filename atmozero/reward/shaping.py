"""Shaped reward R_tilde_t(c, x) = R + alpha_t R_cycle + beta_t R_uplift.

Both schedules decay as the verifier becomes informative; clamp to zero after
``clamp_step`` so the final objective is governed by the verifier signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ShapingSchedule:
    alpha0: float = 0.42
    beta0: float = 0.31
    tau_alpha: float = 610.0
    tau_beta: float = 740.0
    clamp_step: int = 1960

    def alpha(self, step: int) -> float:
        if step >= self.clamp_step:
            return 0.0
        return self.alpha0 * math.exp(-step / self.tau_alpha)

    def beta(self, step: int) -> float:
        if step >= self.clamp_step:
            return 0.0
        return self.beta0 * math.exp(-step / self.tau_beta)


def shaped_reward(
    R: float,
    R_cycle: float,
    R_uplift: float,
    step: int,
    schedule: ShapingSchedule,
) -> float:
    return R + schedule.alpha(step) * R_cycle + schedule.beta(step) * R_uplift

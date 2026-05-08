"""Verifier grader: dispatches per-family s_k(v_q, x) and emits S_k, U, C."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from atmozero.verifier.base import RULE_REGISTRY, RuleFamily, WindowContext


@dataclass
class VerifierSignals:
    S: Dict[int, float]
    U: int
    C: float
    g: Dict[int, int]


class VerifierGrader:
    def __init__(self, thresholds: Optional[Dict[int, Tuple[float, float]]] = None):
        self.rules: Dict[int, RuleFamily] = {}
        for fid, cls in sorted(RULE_REGISTRY.items()):
            tau_m, tau_p = (thresholds or {}).get(fid, (cls.tau_minus, cls.tau_plus))
            self.rules[fid] = cls(tau_minus=tau_m, tau_plus=tau_p)

    @property
    def family_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.rules.keys()))

    def grade_window(
        self,
        claims: List[Tuple[int, str]],
        ctx: WindowContext,
    ) -> VerifierSignals:
        """Compute (S_k, U, C, g_k) for one caption's typed-claim list on a window.

        Claims whose family lacks required channels at this station are
        ignored, consistent with the variable-subset-aware design.
        """
        available = set(k for k, vals in ctx.x.items() if vals is not None)
        triggers: Dict[int, int] = {}
        S: Dict[int, float] = {}
        U_total = 0

        for fid, rule in self.rules.items():
            if not rule.has_required_channels(available):
                triggers[fid] = 0
                S[fid] = 0.0
                continue
            triggers[fid] = rule.trigger(ctx)

            family_claims = [v for (t, v) in claims if t == fid]
            if not family_claims:
                S[fid] = 0.0
                continue
            scores = [rule.grade(v, ctx) for v in family_claims]
            supported = [s for s in scores if s >= rule.tau_plus]
            S[fid] = float(np.mean(supported)) if supported else 0.0
            U_total += sum(1 for s in scores if s <= rule.tau_minus)

        active = [fid for fid, t in triggers.items() if t == 1]
        if not active:
            coverage = 0.0
        else:
            covered = sum(1 for fid in active if S[fid] > 0.0)
            coverage = covered / len(active)

        return VerifierSignals(S=S, U=U_total, C=coverage, g=triggers)

    def per_claim_grades(
        self,
        claims: List[Tuple[int, str]],
        ctx: WindowContext,
    ) -> List[float]:
        out = []
        available = set(k for k, vals in ctx.x.items() if vals is not None)
        for (t, v) in claims:
            rule = self.rules.get(t)
            if rule is None or not rule.has_required_channels(available):
                out.append(0.0)
            else:
                out.append(rule.grade(v, ctx))
        return out

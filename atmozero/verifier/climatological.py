"""k=6 Climatological rule family. Channels = X_stn, plus Köppen zone kappa."""
from __future__ import annotations

import numpy as np

from atmozero.verifier.base import RuleFamily, WindowContext, register_rule
from atmozero.verifier.lexicons import LEXICON_CLIMATOLOGICAL


@register_rule
class ClimatologicalRule(RuleFamily):
    family_id = 6
    family_name = "climatological"
    channels = ("T", "P", "q", "u", "v", "r")
    lexicon = LEXICON_CLIMATOLOGICAL
    tau_minus = 0.20
    tau_plus = 0.68

    Z_TRIGGER = 1.5
    Z_REFUTE = 0.5
    Z_LIGHT_ANOMALY = 1.0
    Z_STRONG_ANOMALY = 2.5
    Z_NEAR_NORMAL_BAND = 0.8

    _MEAN_KEYS = {"T": "T_mean", "P": "P_mean", "q": "q_mean", "r": "r_mean"}
    _STD_KEYS = {"T": "T_std", "P": "P_std", "q": "q_std", "r": "r_std"}

    def trigger(self, ctx: WindowContext) -> int:
        if ctx.climatology is None:
            return 0
        for v in ("T", "P", "q", "r"):
            z = self._z_score(ctx, v)
            if z is not None and abs(z) >= self.Z_TRIGGER:
                return 1
        return 0

    def grade(self, v_q: str, ctx: WindowContext) -> float:
        if ctx.climatology is None:
            return 0.0
        z_T = self._z_score(ctx, "T")
        z_P = self._z_score(ctx, "P")
        z_q = self._z_score(ctx, "q")

        if v_q == "climatological warm anomaly" and z_T is not None:
            return float(np.clip(_ramp(z_T, self.Z_LIGHT_ANOMALY, self.Z_STRONG_ANOMALY), 0.0, 1.0))
        if v_q == "climatological cold anomaly" and z_T is not None:
            return float(np.clip(_ramp(-z_T, self.Z_LIGHT_ANOMALY, self.Z_STRONG_ANOMALY), 0.0, 1.0))
        if v_q == "climatological wet anomaly" and z_q is not None:
            return float(np.clip(_ramp(z_q, self.Z_LIGHT_ANOMALY, self.Z_STRONG_ANOMALY), 0.0, 1.0))
        if v_q == "climatological dry anomaly" and z_q is not None:
            return float(np.clip(_ramp(-z_q, self.Z_LIGHT_ANOMALY, self.Z_STRONG_ANOMALY), 0.0, 1.0))
        if v_q == "climatological pressure anomaly" and z_P is not None:
            return float(np.clip(_ramp(abs(z_P), self.Z_LIGHT_ANOMALY, self.Z_STRONG_ANOMALY), 0.0, 1.0))
        if v_q == "near-normal regime":
            zs = [z for z in (z_T, z_P, z_q) if z is not None]
            if not zs:
                return 0.0
            return float(np.clip(1.0 - _ramp(max(abs(z) for z in zs), 0.0, self.Z_NEAR_NORMAL_BAND), 0.0, 1.0))
        return 0.0

    def _z_score(self, ctx: WindowContext, var: str):
        x = ctx.x.get(var)
        if x is None:
            return None
        m = ctx.climatology.get(self._MEAN_KEYS[var])
        s = ctx.climatology.get(self._STD_KEYS[var], 1.0)
        if m is None:
            return None
        return float((float(np.mean(x)) - m) / max(s, 1e-3))


def _ramp(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)

"""k=1 Thermodynamic rule family. Channels T, q."""
from __future__ import annotations

import numpy as np

from atmozero.verifier.base import RuleFamily, WindowContext, register_rule
from atmozero.verifier.lexicons import LEXICON_THERMODYNAMIC


@register_rule
class ThermodynamicRule(RuleFamily):
    family_id = 1
    family_name = "thermodynamic"
    channels = ("T", "q")
    lexicon = LEXICON_THERMODYNAMIC
    tau_minus = 0.18
    tau_plus = 0.72

    SUSTAINED_SLOPE_K_PER_HOUR = 0.20
    SUSTAINED_HOURS = 12
    REVERSAL_SLOPE_K_PER_HOUR = 0.15
    DEWPOINT_DEP_THRESHOLD_K = 1.5
    DEWPOINT_DEP_PERSIST_HOURS = 6

    def trigger(self, ctx: WindowContext) -> int:
        T = ctx.x.get("T")
        q = ctx.x.get("q")
        if T is None or q is None or T.size < 24:
            return 0
        if _has_sustained_slope(T, self.SUSTAINED_SLOPE_K_PER_HOUR, self.SUSTAINED_HOURS):
            return 1
        Td = _dewpoint_from_q(q)
        dep = T - Td
        if _has_persistent_run(dep < self.DEWPOINT_DEP_THRESHOLD_K, self.DEWPOINT_DEP_PERSIST_HOURS):
            return 1
        if _has_reversal(T, self.REVERSAL_SLOPE_K_PER_HOUR):
            return 1
        return 0

    def grade(self, v_q: str, ctx: WindowContext) -> float:
        T = ctx.x.get("T")
        q = ctx.x.get("q")
        if T is None or q is None or T.size < 6:
            return 0.0
        slope = float(np.polyfit(np.arange(T.size), T, 1)[0])
        Td = _dewpoint_from_q(q)
        dep_mean = float(np.mean(T - Td))

        # Window-level trend ramps are wider than the 12-h trigger threshold:
        # over a 192-h window even a steady 0.02 K/h drift means ~4 K total.
        if v_q == "warming trend":
            score = _ramp(slope, 0.01, 0.10)
        elif v_q == "cooling trend":
            score = _ramp(-slope, 0.01, 0.10)
        elif v_q == "thermal stability":
            score = 1.0 - _ramp(abs(slope), 0.01, 0.10)
        elif v_q == "thermal reversal":
            score = _reversal_score(T, self.REVERSAL_SLOPE_K_PER_HOUR)
        elif v_q == "dry air mass":
            score = _ramp(dep_mean, 4.0, 12.0)
        elif v_q == "humid air mass":
            score = 1.0 - _ramp(dep_mean, self.DEWPOINT_DEP_THRESHOLD_K, 6.0)
        else:
            return 0.0

        return float(np.clip(score, 0.0, 1.0))


def _ramp(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def _has_sustained_slope(T: np.ndarray, threshold: float, hours: int) -> bool:
    if T.size < hours + 1:
        return False
    for start in range(T.size - hours):
        seg = T[start:start + hours + 1]
        slope = float(np.polyfit(np.arange(seg.size), seg, 1)[0])
        if abs(slope) >= threshold:
            return True
    return False


def _has_persistent_run(mask: np.ndarray, run_length: int) -> bool:
    run = 0
    for v in mask:
        run = run + 1 if bool(v) else 0
        if run >= run_length:
            return True
    return False


def _has_reversal(T: np.ndarray, slope_threshold: float) -> bool:
    if T.size < 25:
        return False
    for t in range(12, T.size - 12):
        pre = float(np.polyfit(np.arange(12), T[t - 12:t], 1)[0])
        post = float(np.polyfit(np.arange(12), T[t:t + 12], 1)[0])
        if abs(pre) >= slope_threshold and abs(post) >= slope_threshold and (pre * post < 0):
            return True
    return False


def _reversal_score(T: np.ndarray, slope_threshold: float) -> float:
    if T.size < 25:
        return 0.0
    best = 0.0
    for t in range(12, T.size - 12):
        pre = float(np.polyfit(np.arange(12), T[t - 12:t], 1)[0])
        post = float(np.polyfit(np.arange(12), T[t:t + 12], 1)[0])
        if pre * post >= 0:
            continue
        score = _ramp(min(abs(pre), abs(post)), slope_threshold, slope_threshold * 3)
        if score > best:
            best = score
    return best


def _dewpoint_from_q(q: np.ndarray) -> np.ndarray:
    """Approximate dewpoint (deg C) from specific humidity q (kg/kg) at p ~ 1000 hPa."""
    e = q * 1000.0 / 0.622
    e = np.clip(e, 1e-3, 80.0)
    return 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))

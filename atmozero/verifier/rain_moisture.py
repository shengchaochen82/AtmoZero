"""k=2 Rain-Moisture rule family. Channels r, q, P, T."""
from __future__ import annotations

import numpy as np

from atmozero.verifier.base import RuleFamily, WindowContext, register_rule
from atmozero.verifier.lexicons import LEXICON_RAIN_MOISTURE


@register_rule
class RainMoistureRule(RuleFamily):
    family_id = 2
    family_name = "rain_moisture"
    channels = ("r", "q", "P")
    lexicon = LEXICON_RAIN_MOISTURE
    tau_minus = 0.20
    tau_plus = 0.70

    LIGHT_RAIN_MM_PER_HOUR = 0.5
    MODERATE_RAIN_MM_PER_HOUR = 2.5
    HEAVY_RAIN_MM_PER_HOUR = 7.5
    TORRENTIAL_RAIN_MM_PER_HOUR = 15.0
    TRIGGER_TOTAL_MM = 5.0

    def trigger(self, ctx: WindowContext) -> int:
        r = ctx.x.get("r")
        q = ctx.x.get("q")
        P = ctx.x.get("P")
        if r is None or q is None or P is None:
            return 0
        if float(np.sum(r)) < self.TRIGGER_TOTAL_MM:
            return 0
        rain_mask = r > 0.1
        if rain_mask.sum() < 2:
            return 0
        if float(np.mean(q[rain_mask])) < float(np.mean(q)) - 1e-4:
            return 0
        return int(float(np.polyfit(np.arange(P.size), P, 1)[0]) <= 0)

    def grade(self, v_q: str, ctx: WindowContext) -> float:
        r = ctx.x.get("r")
        q = ctx.x.get("q")
        P = ctx.x.get("P")
        if r is None or q is None or P is None:
            return 0.0

        rain_total = float(np.sum(r))
        peak_rate = float(np.max(r))
        q_trend = float(np.polyfit(np.arange(q.size), q, 1)[0])
        p_slope = float(np.polyfit(np.arange(P.size), P, 1)[0])

        if v_q == "light rain":
            score = (
                _ramp(peak_rate, 0.05, self.LIGHT_RAIN_MM_PER_HOUR)
                * (1.0 - _ramp(peak_rate, self.MODERATE_RAIN_MM_PER_HOUR, self.HEAVY_RAIN_MM_PER_HOUR))
            )
            return float(np.clip(score, 0.0, 1.0))
        if v_q == "moderate rain":
            score = (
                _ramp(peak_rate, self.LIGHT_RAIN_MM_PER_HOUR, self.MODERATE_RAIN_MM_PER_HOUR)
                * (1.0 - _ramp(peak_rate, self.HEAVY_RAIN_MM_PER_HOUR, self.TORRENTIAL_RAIN_MM_PER_HOUR))
            )
            return float(np.clip(score, 0.0, 1.0))
        if v_q == "heavy rain":
            score = (
                _ramp(peak_rate, self.MODERATE_RAIN_MM_PER_HOUR, self.HEAVY_RAIN_MM_PER_HOUR)
                * (1.0 - _ramp(peak_rate, self.TORRENTIAL_RAIN_MM_PER_HOUR, self.TORRENTIAL_RAIN_MM_PER_HOUR * 2))
            )
            return float(np.clip(score, 0.0, 1.0))
        if v_q == "torrential rain":
            return float(np.clip(_ramp(peak_rate, self.HEAVY_RAIN_MM_PER_HOUR, self.TORRENTIAL_RAIN_MM_PER_HOUR), 0.0, 1.0))
        if v_q == "moist advection":
            return float(np.clip(_ramp(q_trend, 0.0, 1e-5) * _ramp(-p_slope, 0.05, 0.5), 0.0, 1.0))
        if v_q == "dry advection":
            return float(np.clip(_ramp(-q_trend, 0.0, 1e-5) * _ramp(p_slope, 0.05, 0.5), 0.0, 1.0))
        if v_q == "post-frontal drying":
            return float(np.clip(_ramp(rain_total, 1.0, 5.0) * _ramp(-q_trend, 0.0, 1e-5) * _ramp(p_slope, 0.0, 0.3), 0.0, 1.0))
        return 0.0


def _ramp(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)

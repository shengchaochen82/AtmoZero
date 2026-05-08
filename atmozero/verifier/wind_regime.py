"""k=4 Wind Regime rule family. Channels u, v."""
from __future__ import annotations

import numpy as np

from atmozero.verifier.base import RuleFamily, WindowContext, register_rule
from atmozero.verifier.lexicons import LEXICON_WIND_REGIME


@register_rule
class WindRegimeRule(RuleFamily):
    family_id = 4
    family_name = "wind_regime"
    channels = ("u", "v")
    lexicon = LEXICON_WIND_REGIME
    tau_minus = 0.22
    tau_plus = 0.70

    SUSTAINED_THRESHOLD_MPS = 5.0
    SUSTAINED_HOURS = 36
    GUST_FACTOR = 2.2
    GUST_TRANSIENT_HOURS = 6
    CALM_THRESHOLD_MPS = 1.5

    # Canonical bearings: (u, v) point in the direction the wind is going.
    DIRECTION_BEARING_DEG = {
        "monsoonal southerly":     0.0,
        "trade easterly":        270.0,
        "westerly jet entrance":  90.0,
        "post-frontal northerly":180.0,
    }

    def trigger(self, ctx: WindowContext) -> int:
        u = ctx.x.get("u"); v = ctx.x.get("v")
        if u is None or v is None or u.size < 2:
            return 0
        if _sustained_speed(u, v, self.SUSTAINED_THRESHOLD_MPS, self.SUSTAINED_HOURS):
            return 1
        return int(_gust_factor(u, v, self.GUST_TRANSIENT_HOURS) >= self.GUST_FACTOR)

    def grade(self, v_q: str, ctx: WindowContext) -> float:
        u = ctx.x.get("u"); v = ctx.x.get("v")
        if u is None or v is None or u.size < 2:
            return 0.0
        speed = np.hypot(u, v)
        mean_speed = float(np.mean(speed))
        bearing_mean = _circular_mean(np.degrees(np.arctan2(u, v)) % 360)
        bearing_consistency = _circular_consistency(np.degrees(np.arctan2(u, v)) % 360)
        gust = _gust_factor(u, v, self.GUST_TRANSIENT_HOURS)

        if v_q in self.DIRECTION_BEARING_DEG:
            target = self.DIRECTION_BEARING_DEG[v_q]
            cos_align = (np.cos(np.radians(bearing_mean - target)) + 1) / 2
            score = (
                _ramp(mean_speed, self.CALM_THRESHOLD_MPS, self.SUSTAINED_THRESHOLD_MPS)
                * bearing_consistency * cos_align
            )
            if v_q == "westerly jet entrance":
                score *= _ramp(mean_speed, self.SUSTAINED_THRESHOLD_MPS, 12.0)
            return float(np.clip(score, 0.0, 1.0))
        if v_q == "gust-front transient":
            return float(np.clip(_ramp(gust, self.GUST_FACTOR - 0.4, self.GUST_FACTOR + 0.8), 0.0, 1.0))
        if v_q == "calm regime":
            return float(np.clip(1.0 - _ramp(mean_speed, 0.3, self.CALM_THRESHOLD_MPS), 0.0, 1.0))
        return 0.0


def _ramp(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def _circular_mean(bearings_deg: np.ndarray) -> float:
    rad = np.radians(bearings_deg)
    s = float(np.mean(np.sin(rad))); c = float(np.mean(np.cos(rad)))
    return (np.degrees(np.arctan2(s, c)) + 360) % 360


def _circular_consistency(bearings_deg: np.ndarray) -> float:
    rad = np.radians(bearings_deg)
    return float(np.hypot(np.mean(np.sin(rad)), np.mean(np.cos(rad))))


def _sustained_speed(u: np.ndarray, v: np.ndarray, threshold: float, hours: int) -> bool:
    speed = np.hypot(u, v)
    if speed.size < hours:
        return False
    rolling = np.convolve(speed, np.ones(hours) / hours, mode="valid")
    return bool(np.any(rolling >= threshold))


def _gust_factor(u: np.ndarray, v: np.ndarray, hours: int) -> float:
    speed = np.hypot(u, v)
    if speed.size < hours:
        return 0.0
    mean = float(np.mean(speed))
    if mean <= 1e-3:
        return 0.0
    rolling_max = max(float(np.max(speed[i:i + hours])) for i in range(speed.size - hours + 1))
    return rolling_max / mean

"""k=5 Diurnal Cycle rule family. Channels T, q, r."""
from __future__ import annotations

import numpy as np

from atmozero.verifier.base import RuleFamily, WindowContext, register_rule
from atmozero.verifier.lexicons import LEXICON_DIURNAL


@register_rule
class DiurnalCycleRule(RuleFamily):
    family_id = 5
    family_name = "diurnal"
    channels = ("T", "q", "r")
    lexicon = LEXICON_DIURNAL
    tau_minus = 0.18
    tau_plus = 0.70

    NORMAL_AMPLITUDE_K = 6.0
    SUPPRESSED_AMPLITUDE_K = 2.0
    PHASE_TOLERANCE_HOURS = 2.0
    AFTERNOON_PEAK_HOUR = 14.0
    NOCTURNAL_HEAT_HOUR = 23.0
    DAYBREAK_MIN_HOUR = 6.0

    def trigger(self, ctx: WindowContext) -> int:
        T = ctx.x.get("T")
        if T is None or T.size < 24:
            return 0
        return int(_diurnal_amplitude(T) > 1.0)

    def grade(self, v_q: str, ctx: WindowContext) -> float:
        T = ctx.x.get("T")
        r = ctx.x.get("r")
        if T is None or T.size < 24:
            return 0.0
        amp = _diurnal_amplitude(T)
        peak_hour = _peak_hour(T)
        min_hour = _min_hour(T)

        if v_q == "strong diurnal cycle":
            return float(np.clip(_ramp(amp, self.NORMAL_AMPLITUDE_K, self.NORMAL_AMPLITUDE_K * 1.8), 0.0, 1.0))
        if v_q == "suppressed diurnal cycle":
            return float(np.clip(1.0 - _ramp(amp, self.SUPPRESSED_AMPLITUDE_K, self.NORMAL_AMPLITUDE_K), 0.0, 1.0))
        if v_q == "nocturnal heating":
            offset = _hour_distance(peak_hour, self.NOCTURNAL_HEAT_HOUR)
            return float(np.clip(1.0 - _ramp(offset, self.PHASE_TOLERANCE_HOURS, 8.0), 0.0, 1.0))
        if v_q == "afternoon convective peak":
            offset = _hour_distance(peak_hour, self.AFTERNOON_PEAK_HOUR)
            phase_score = 1.0 - _ramp(offset, self.PHASE_TOLERANCE_HOURS, 6.0)
            rain_assist = 0.5 if (r is not None and float(np.sum(r)) > 0.5) else 0.0
            return float(np.clip(phase_score * (0.5 + rain_assist), 0.0, 1.0))
        if v_q == "daybreak temperature minimum":
            offset = _hour_distance(min_hour, self.DAYBREAK_MIN_HOUR)
            return float(np.clip(1.0 - _ramp(offset, self.PHASE_TOLERANCE_HOURS, 6.0), 0.0, 1.0))
        return 0.0


def _diurnal_amplitude(T: np.ndarray) -> float:
    n_full_days = T.size // 24
    if n_full_days == 0:
        return float(np.ptp(T))
    return float(np.mean([np.ptp(T[d * 24:(d + 1) * 24]) for d in range(n_full_days)]))


def _peak_hour(T: np.ndarray) -> float:
    n_full_days = T.size // 24
    if n_full_days == 0:
        return float(np.argmax(T) % 24)
    return float(np.mean([int(np.argmax(T[d * 24:(d + 1) * 24])) for d in range(n_full_days)]))


def _min_hour(T: np.ndarray) -> float:
    n_full_days = T.size // 24
    if n_full_days == 0:
        return float(np.argmin(T) % 24)
    return float(np.mean([int(np.argmin(T[d * 24:(d + 1) * 24])) for d in range(n_full_days)]))


def _hour_distance(a: float, b: float) -> float:
    d = abs(a - b) % 24
    return float(min(d, 24 - d))


def _ramp(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)

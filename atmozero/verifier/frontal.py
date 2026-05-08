"""k=3 Frontal rule family. Channels T, u, v, P."""
from __future__ import annotations

import numpy as np

from atmozero.verifier.base import RuleFamily, WindowContext, register_rule
from atmozero.verifier.lexicons import LEXICON_FRONTAL


@register_rule
class FrontalRule(RuleFamily):
    family_id = 3
    family_name = "frontal"
    channels = ("T", "u", "v", "P")
    lexicon = LEXICON_FRONTAL
    tau_minus = 0.15
    tau_plus = 0.68

    DT_COLD_FRONT_K = 3.0
    DT_WARM_FRONT_K = 3.0
    WIND_SHIFT_DEG = 60.0
    P_REBOUND_HPA = 1.0

    def trigger(self, ctx: WindowContext) -> int:
        T = ctx.x.get("T"); u = ctx.x.get("u"); v = ctx.x.get("v"); P = ctx.x.get("P")
        if any(s is None for s in (T, u, v, P)) or T.size < 25:
            return 0
        for t in range(12, T.size - 12):
            dT24 = float(T[t + 12] - T[t - 12])
            wind_rot = _wind_rotation_deg(u[t - 6:t + 7], v[t - 6:t + 7])
            p_dip = _pressure_dip(P[t - 12:t + 13])
            if (abs(dT24) >= self.DT_COLD_FRONT_K
                    and wind_rot >= self.WIND_SHIFT_DEG
                    and p_dip >= self.P_REBOUND_HPA):
                return 1
        return 0

    def grade(self, v_q: str, ctx: WindowContext) -> float:
        T = ctx.x.get("T"); u = ctx.x.get("u"); v = ctx.x.get("v"); P = ctx.x.get("P")
        if any(s is None for s in (T, u, v, P)) or T.size < 25:
            return 0.0

        best_t = _locate_frontal_centre(T, u, v, P)
        dT24 = float(T[best_t + 12] - T[best_t - 12])
        wind_rot = _wind_rotation_deg(u[best_t - 6:best_t + 7], v[best_t - 6:best_t + 7])
        p_dip = _pressure_dip(P[best_t - 12:best_t + 13])

        if v_q == "cold front passage":
            score = (
                _ramp(-dT24, 1.5, self.DT_COLD_FRONT_K)
                * _ramp(wind_rot, 30.0, self.WIND_SHIFT_DEG)
                * _ramp(p_dip, 0.3, self.P_REBOUND_HPA)
            )
            return float(np.clip(score, 0.0, 1.0))
        if v_q == "warm front passage":
            score = (
                _ramp(dT24, 1.0, self.DT_WARM_FRONT_K)
                * _ramp(wind_rot, 20.0, 60.0)
                * _ramp(p_dip, 0.2, self.P_REBOUND_HPA)
            )
            return float(np.clip(score, 0.0, 1.0))
        if v_q == "occluded front":
            return float(np.clip(_two_stage_drop(T) * _ramp(p_dip, 0.6, 1.5), 0.0, 1.0))
        if v_q == "pre-frontal trough":
            p_slope = float(np.polyfit(np.arange(P.size), P, 1)[0])
            score = _ramp(-p_slope, 0.05, 0.5) * (1.0 - _ramp(abs(dT24), 1.0, 3.0))
            return float(np.clip(score, 0.0, 1.0))
        if v_q == "post-frontal ridging":
            p_slope = float(np.polyfit(np.arange(P.size), P, 1)[0])
            score = _ramp(p_slope, 0.05, 0.5) * _ramp(-dT24, 0.5, 2.5)
            return float(np.clip(score, 0.0, 1.0))
        return 0.0


def _ramp(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def _wind_rotation_deg(u: np.ndarray, v: np.ndarray) -> float:
    if u.size < 2:
        return 0.0
    bearings = np.degrees(np.arctan2(u, v)) % 360
    diffs = np.diff(bearings)
    diffs = (diffs + 180) % 360 - 180
    return float(np.abs(np.sum(diffs)))


def _pressure_dip(P: np.ndarray) -> float:
    if P.size < 3:
        return 0.0
    idx_min = int(np.argmin(P))
    if idx_min in (0, len(P) - 1):
        return 0.0
    return min(float(P[0] - P[idx_min]), float(P[-1] - P[idx_min]))


def _locate_frontal_centre(T: np.ndarray, u: np.ndarray, v: np.ndarray, P: np.ndarray) -> int:
    best_score = -np.inf
    best_t = T.size // 2
    for t in range(12, T.size - 12):
        dT24 = abs(float(T[t + 12] - T[t - 12]))
        wind_rot = _wind_rotation_deg(u[t - 6:t + 7], v[t - 6:t + 7])
        p_dip = _pressure_dip(P[t - 12:t + 13])
        score = dT24 + 0.05 * wind_rot + p_dip
        if score > best_score:
            best_score = score
            best_t = t
    return best_t


def _two_stage_drop(T: np.ndarray) -> float:
    if T.size < 24:
        return 0.0
    half = T.size // 2
    drop1 = float(T[half] - T[0])
    drop2 = float(T[-1] - T[half])
    if drop1 < -1.0 and drop2 < -1.0:
        return min(1.0, (-drop1 - drop2) / 4.0)
    return 0.0

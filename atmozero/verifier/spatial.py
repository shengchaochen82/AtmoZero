"""k=7 Spatial Coherence rule family. Channels = K nearest-neighbour windows."""
from __future__ import annotations

import numpy as np

from atmozero.verifier.base import RuleFamily, WindowContext, register_rule
from atmozero.verifier.lexicons import LEXICON_SPATIAL


@register_rule
class SpatialCoherenceRule(RuleFamily):
    family_id = 7
    family_name = "spatial_coherence"
    channels = ("T",)
    lexicon = LEXICON_SPATIAL
    tau_minus = 0.20
    tau_plus = 0.68

    REGIONAL_TOLERANCE_K = 1.0
    LOCAL_DEVIATION_K = 2.0
    COHERENT_R_MIN = 0.6

    def trigger(self, ctx: WindowContext) -> int:
        return int(ctx.neighbors is not None and len(ctx.neighbors) >= 4)

    def grade(self, v_q: str, ctx: WindowContext) -> float:
        if ctx.neighbors is None or not ctx.neighbors:
            return 0.0
        T = ctx.x.get("T")
        if T is None:
            return 0.0
        neighbour_means = []
        coherences = []
        for nb in ctx.neighbors:
            T_nb = nb.get("T")
            if T_nb is None or T_nb.size != T.size:
                continue
            neighbour_means.append(float(np.mean(T_nb)))
            try:
                r = float(np.corrcoef(T, T_nb)[0, 1])
            except Exception:
                r = 0.0
            coherences.append(r)
        if not neighbour_means:
            return 0.0
        focal_mean = float(np.mean(T))
        nb_median = float(np.median(neighbour_means))
        nb_iqr = float(np.subtract(*np.percentile(neighbour_means, [75, 25])))
        delta = focal_mean - nb_median
        coherence = float(np.mean(coherences))

        if v_q == "regional regime":
            close = 1.0 - _ramp(abs(delta), 0.5, self.REGIONAL_TOLERANCE_K * 2)
            return float(np.clip(close * _ramp(coherence, 0.4, self.COHERENT_R_MIN + 0.2), 0.0, 1.0))
        if v_q == "locally anomalous":
            outside_iqr = _ramp(abs(delta) - nb_iqr, 0.0, self.LOCAL_DEVIATION_K)
            decoupled = 1.0 - _ramp(coherence, 0.4, 0.8)
            return float(np.clip(_ramp(abs(delta), 0.5, self.LOCAL_DEVIATION_K) * 0.5
                                 + 0.5 * outside_iqr * decoupled, 0.0, 1.0))
        if v_q == "contrasted gradient":
            spread = float(np.std(neighbour_means))
            extremity = _ramp(abs(delta), 0.5, self.LOCAL_DEVIATION_K)
            return float(np.clip(_ramp(spread, 1.0, 4.0) * extremity, 0.0, 1.0))
        if v_q == "coherent advection":
            focal_drift = float(np.polyfit(np.arange(T.size), T, 1)[0])
            nb_drifts = [
                float(np.polyfit(np.arange(nb["T"].size), nb["T"], 1)[0])
                for nb in ctx.neighbors if nb.get("T") is not None and nb["T"].size == T.size
            ]
            if not nb_drifts:
                return 0.0
            drift_align = 1.0 - _ramp(abs(focal_drift - float(np.mean(nb_drifts))), 0.0, 0.05)
            return float(np.clip(_ramp(coherence, 0.5, 0.8) * drift_align, 0.0, 1.0))
        if v_q == "isolated event":
            focal_var = float(np.var(T))
            nb_var = float(np.mean([np.var(nb["T"]) for nb in ctx.neighbors if nb.get("T") is not None]))
            return float(np.clip(_ramp(focal_var - nb_var, 0.5, 4.0), 0.0, 1.0))
        return 0.0


def _ramp(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)

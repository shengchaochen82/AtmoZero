"""verl-compatible reward function for AtmoZero post-training.

verl invokes ``compute_score(data_source, solution_str, ground_truth, extra_info)``
once per rollout. We parse the typed-claim list, run the seven-family
verifier, compose ``R = sum_k w_k S_k - lambda U + mu C + nu D``, and update
the shared Proposer/cold-start state with the per-rollout verifier precision.

``extra_info`` is populated by ``scripts/prepare_verl_data.py`` with
``{window_id, x: dict, metadata: dict, neighbors: list, climatology: dict}``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from atmozero.caption.parser import parse_caption
from atmozero.proposer import WindowSampler
from atmozero.reward.composition import RewardConfig, compose_reward
from atmozero.verifier import VerifierGrader
from atmozero.verifier.base import WindowContext


_GRADER: Optional[VerifierGrader] = None
_REWARD_CFG: Optional[RewardConfig] = None
_SAMPLER: Optional[WindowSampler] = None


def _get_grader() -> VerifierGrader:
    global _GRADER
    if _GRADER is None:
        _GRADER = VerifierGrader()
    return _GRADER


def _get_reward_cfg() -> RewardConfig:
    global _REWARD_CFG
    if _REWARD_CFG is None:
        _REWARD_CFG = RewardConfig(
            lam=float(os.environ.get("ATMOZERO_LAMBDA", 2.0)),
            mu=float(os.environ.get("ATMOZERO_MU", 0.22)),
            nu=float(os.environ.get("ATMOZERO_NU", 0.06)),
        )
    return _REWARD_CFG


def _get_sampler(n_windows: int) -> Optional[WindowSampler]:
    global _SAMPLER
    state_path = os.environ.get("ATMOZERO_STATE_PATH")
    if state_path is None:
        return None
    if _SAMPLER is None:
        _SAMPLER = WindowSampler(n_windows=n_windows, state_path=state_path)
    return _SAMPLER


def _ctx_from_extra(extra_info: Dict[str, Any]) -> WindowContext:
    x = {k: np.asarray(v) for k, v in extra_info.get("x", {}).items()}
    neighbors = extra_info.get("neighbors")
    if neighbors is not None:
        neighbors = [{k: np.asarray(v) for k, v in nb.items()} for nb in neighbors]
    return WindowContext(
        x=x,
        metadata=dict(extra_info.get("metadata", {})),
        neighbors=neighbors,
        climatology=extra_info.get("climatology"),
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """Return the AtmoZero per-rollout scalar reward."""
    if extra_info is None:
        return 0.0

    grader = _get_grader()
    reward_cfg = _get_reward_cfg()

    ctx = _ctx_from_extra(extra_info)
    parsed = parse_caption(solution_str)
    claims = parsed.as_q_list()
    signals = grader.grade_window(claims, ctx)
    R = compose_reward(signals, solution_str, reward_cfg)

    if claims:
        per_claim = grader.per_claim_grades(claims, ctx)
        supported = sum(
            1 for s, (fid, _v) in zip(per_claim, claims)
            if s >= grader.rules[fid].tau_plus
        )
        precision = supported / len(claims)
    else:
        precision = 0.0

    n_windows = int(extra_info.get("n_windows", 0)) or None
    window_id = extra_info.get("window_id")
    if n_windows is not None and window_id is not None:
        sampler = _get_sampler(n_windows)
        if sampler is not None:
            sampler.record(int(window_id), precision)

    return float(R)

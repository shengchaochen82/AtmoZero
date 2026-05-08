"""Stratified 5-fold cross-fit calibration of (tau_k^-, tau_k^+) and reward weights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass
class CrossFitResult:
    thresholds: Dict[int, Tuple[float, float]]
    reward_weights: Dict[str, float]
    per_fold_thresholds: List[Dict[int, Tuple[float, float]]]


def _per_fold_thresholds(
    family_id: int,
    scores: np.ndarray,
    labels: np.ndarray,
    target_support_precision: float = 0.90,
    target_refute_precision: float = 0.90,
    target_refute_recall: float = 0.85,
) -> Tuple[float, float]:
    """Choose (tau_minus, tau_plus) for one family on one training fold."""
    if scores.size == 0:
        return 0.30, 0.70

    s_mask = labels == "S"
    r_mask = labels == "R"
    if not s_mask.any() or not r_mask.any():
        return 0.30, 0.70

    grid = np.linspace(0.05, 0.95, 91)

    best_tau_plus = 0.70
    for t in grid:
        pred = scores >= t
        if pred.sum() == 0:
            continue
        if float((pred & s_mask).sum()) / float(pred.sum()) >= target_support_precision:
            best_tau_plus = float(t)
            break

    best_tau_minus = 0.30
    for t in grid[::-1]:
        pred_neg = scores <= t
        if pred_neg.sum() == 0:
            continue
        rec = float((pred_neg & r_mask).sum()) / max(int(r_mask.sum()), 1)
        prec = float((pred_neg & r_mask).sum()) / float(pred_neg.sum())
        if rec >= target_refute_recall and prec >= target_refute_precision:
            best_tau_minus = float(t)
            break

    return best_tau_minus, best_tau_plus


def cross_fit_thresholds(
    per_family_scores: Dict[int, np.ndarray],
    per_family_labels: Dict[int, np.ndarray],
    window_ids: Dict[int, np.ndarray],
    strata: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> CrossFitResult:
    """Run a stratified 5-fold cross-fit; return per-family median thresholds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_thresholds: List[Dict[int, Tuple[float, float]]] = []
    fold_indices = list(skf.split(np.arange(len(strata)), strata))

    for train_w, _ in fold_indices:
        thresholds: Dict[int, Tuple[float, float]] = {}
        train_w_set = set(train_w.tolist())
        for fid in per_family_scores:
            scores_f = per_family_scores[fid]
            labels_f = per_family_labels[fid]
            ids_f = window_ids[fid]
            mask = np.array([wid in train_w_set for wid in ids_f])
            if mask.sum() == 0:
                thresholds[fid] = (0.30, 0.70)
                continue
            thresholds[fid] = _per_fold_thresholds(fid, scores_f[mask], labels_f[mask])
        fold_thresholds.append(thresholds)

    fids = sorted(per_family_scores.keys())
    agg_thresholds: Dict[int, Tuple[float, float]] = {}
    for fid in fids:
        tm = np.median([f[fid][0] for f in fold_thresholds])
        tp = np.median([f[fid][1] for f in fold_thresholds])
        agg_thresholds[fid] = (float(tm), float(tp))

    reward_weights = {
        "lambda": 2.0,
        "mu": 0.22,
        "nu": 0.06,
        **{f"w_{fid}": 1.0 / 7.0 for fid in fids},
    }

    return CrossFitResult(
        thresholds=agg_thresholds,
        reward_weights=reward_weights,
        per_fold_thresholds=fold_thresholds,
    )


def estimation_set_size_sweep(
    per_family_scores: Dict[int, np.ndarray],
    per_family_labels: Dict[int, np.ndarray],
    window_ids: Dict[int, np.ndarray],
    strata: np.ndarray,
    sizes: Iterable[int] = (100, 160, 200),
    seed: int = 42,
) -> Dict[int, CrossFitResult]:
    """Repeat cross-fit at varying threshold-estimation set sizes."""
    out: Dict[int, CrossFitResult] = {}
    rng = np.random.default_rng(seed)
    for n in sizes:
        if n >= len(strata):
            sub = np.arange(len(strata))
        else:
            sub = rng.choice(len(strata), size=n, replace=False)
        sub_set = set(sub.tolist())

        sub_scores: Dict[int, np.ndarray] = {}
        sub_labels: Dict[int, np.ndarray] = {}
        sub_ids: Dict[int, np.ndarray] = {}
        for fid in per_family_scores:
            keep = np.array([wid in sub_set for wid in window_ids[fid]])
            sub_scores[fid] = per_family_scores[fid][keep]
            sub_labels[fid] = per_family_labels[fid][keep]
            sub_ids[fid] = window_ids[fid][keep]

        sub_strata = strata[sub]
        out[int(n)] = cross_fit_thresholds(
            sub_scores, sub_labels, sub_ids, sub_strata, seed=seed,
        )
    return out

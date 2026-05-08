"""Calibrate per-family verifier thresholds (tau_k^-, tau_k^+) and reward weights
via stratified 5-fold cross-fit on the held-out 200-window annotated set.

Output: a JSON of ``{family_id: [tau_minus, tau_plus]}`` that the trainer and
faithfulness evaluator load via ``VerifierGrader(thresholds=...)``.

Usage::

    python scripts/calibrate_thresholds.py \\
        --scores scores.npz --labels labels.npz \\
        --window_ids window_ids.npz --strata strata.npy \\
        --output thresholds.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from atmozero.verifier.thresholds import cross_fit_thresholds, estimation_set_size_sweep


def load_npz_dict(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {int(k): np.asarray(z[k]) for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True,
                    help="npz with per-family score arrays keyed by family_id")
    ap.add_argument("--labels", required=True,
                    help="npz with per-family annotator-aggregate label arrays "
                         '("S"/"R"/"O"), keyed by family_id')
    ap.add_argument("--window_ids", required=True,
                    help="npz with per-family window-id arrays for cross-fit grouping")
    ap.add_argument("--strata", required=True,
                    help="npy of stratification labels (length = #windows)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--sensitivity", action="store_true",
                    help="also run the (100, 160, 200) estimation-set-size sweep")
    args = ap.parse_args()

    scores = load_npz_dict(args.scores)
    labels = load_npz_dict(args.labels)
    window_ids = load_npz_dict(args.window_ids)
    strata = np.load(args.strata)

    result = cross_fit_thresholds(scores, labels, window_ids, strata)
    out = {
        "thresholds": {str(fid): list(taus) for fid, taus in result.thresholds.items()},
        "reward_weights": result.reward_weights,
    }
    if args.sensitivity:
        sweep = estimation_set_size_sweep(scores, labels, window_ids, strata)
        out["sensitivity"] = {
            str(size): {str(fid): list(t) for fid, t in r.thresholds.items()}
            for size, r in sweep.items()
        }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"[calibrate_thresholds] wrote {args.output} "
          f"(5-fold cross-fit, {len(result.thresholds)} families)")


if __name__ == "__main__":
    main()

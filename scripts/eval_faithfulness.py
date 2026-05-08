"""Compute the eight caption-faithfulness metrics with bootstrap CIs.

Usage::

    python scripts/eval_faithfulness.py \\
        --captions captions.jsonl --labels labels.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atmozero.eval import eight_metric_summary, window_paired_bootstrap


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _derive_claim_terms(c: dict) -> list:
    if "claim_terms" in c and c["claim_terms"]:
        return c["claim_terms"]
    # Fall back to the value component of each parsed claim, lowercased.
    return [str(v).lower() for (_t, v) in c.get("claims", [])]


def _assemble(captions, labels):
    """Build the eight-metric payload from per-window caption + label JSONL."""
    return {
        "captions":                  [c["text"] for c in captions],
        "annot_labels":              [c["annot_labels"] for c in labels],
        "refutation_flags":          [c["refutation_flags"] for c in labels],
        "triggers":                  [c["triggers"] for c in labels],
        "family_supported":          [c["family_supported"] for c in labels],
        "claim_terms":               [_derive_claim_terms(c) for c in captions],
        "n_claims":                  [len(c["claims"]) for c in captions],
        "annotator_labels_per_claim": sum([c["annotator_labels_per_claim"] for c in labels], []),
        "nli_scores":                [c["nli_scores"] for c in labels],
        "window_ids":                [c["window_id"] for c in labels],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    captions = load_jsonl(args.captions)
    labels = load_jsonl(args.labels)
    payload = _assemble(captions, labels)

    summary = eight_metric_summary(payload)
    cis = {}
    for metric in ("Unsup", "Coverage", "NLI", "Orth"):
        cis[metric] = window_paired_bootstrap(
            lambda p, m=metric: eight_metric_summary(p)[m],
            payload,
            n_resamples=args.bootstrap,
        )

    out = {"point": summary, "ci": cis}
    text = json.dumps(out, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()

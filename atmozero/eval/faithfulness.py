"""Eight caption-faithfulness metrics."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np


def unsup_rate(
    annot_labels: List[List[str]],
    refutation_flags: List[List[bool]],
    use_verifier: bool = True,
) -> float:
    """Per-window mean of fraction-of-claims-not-supported.

    ``use_verifier=False`` returns the verifier-independent variant
    (annotator labels only).
    """
    rates = []
    for labels, refute in zip(annot_labels, refutation_flags):
        if not labels:
            continue
        if use_verifier:
            unsup = sum(1 for lbl, ref in zip(labels, refute) if lbl == "R" or ref)
        else:
            unsup = sum(1 for lbl in labels if lbl == "R")
        rates.append(unsup / len(labels))
    return float(np.mean(rates)) if rates else 0.0


def coverage_rate(
    triggers: List[Dict[int, int]],
    family_supported: List[Dict[int, bool]],
) -> float:
    """Fraction of fired families covered by a supported claim."""
    rates = []
    for g, fs in zip(triggers, family_supported):
        active = [k for k, v in g.items() if v == 1]
        if not active:
            continue
        rates.append(sum(1 for k in active if fs.get(k, False)) / len(active))
    return float(np.mean(rates)) if rates else 0.0


STOPWORDS = set("a an the and or of in on at to for with from is are was were be been by".split())


def ttr(captions: List[str], claim_terms: List[List[str]]) -> float:
    """Type-token ratio over content tokens; claim-bearing tokens promoted to weight 2."""
    ratios = []
    for cap, terms in zip(captions, claim_terms):
        toks = [t for t in cap.lower().split() if t.isalpha() and t not in STOPWORDS]
        weighted = list(toks) + list(terms)
        if not weighted:
            continue
        ratios.append(len(set(weighted)) / len(weighted))
    return float(np.mean(ratios)) if ratios else 0.0


def rule_orthogonality(family_supported: List[Dict[int, bool]], n_families: int = 7) -> float:
    """1 - mean off-diagonal cosine similarity of per-window family vectors."""
    if not family_supported:
        return 0.0
    M = np.zeros((len(family_supported), n_families))
    for w, fs in enumerate(family_supported):
        for k in range(1, n_families + 1):
            M[w, k - 1] = 1.0 if fs.get(k, False) else 0.0
    Ek = M.mean(axis=0)
    Ekk = (M[:, :, None] * M[:, None, :]).mean(axis=0)
    sim = 0.0
    pairs = 0
    for i in range(n_families):
        for j in range(i + 1, n_families):
            denom = np.sqrt(Ek[i] * Ek[j]) + 1e-9
            sim += Ekk[i, j] / denom
            pairs += 1
    return float(1.0 - sim / max(pairs, 1))


def claim_density(captions: List[str], n_claims: List[int]) -> float:
    densities = []
    for cap, n in zip(captions, n_claims):
        toks = [t for t in cap.split() if t.strip()]
        if not toks:
            continue
        densities.append(100.0 * n / len(toks))
    return float(np.mean(densities)) if densities else 0.0


def false_refutation_rate(refutation_flags: List[List[bool]], annot_labels: List[List[str]]) -> float:
    """Of all verifier-marked refutations, the fraction the annotator labels Supported."""
    num = 0
    den = 0
    for flags, labels in zip(refutation_flags, annot_labels):
        for f, lbl in zip(flags, labels):
            if f:
                den += 1
                if lbl == "S":
                    num += 1
    return float(num / den) if den else 0.0


def fleiss_kappa(annotator_labels_per_claim: List[List[str]]) -> float:
    """Fleiss kappa over the three-class label set {S, R, O}; pre-adjudication."""
    if not annotator_labels_per_claim:
        return 0.0
    cats = ("S", "R", "O")
    n = len(annotator_labels_per_claim[0])
    p_j = np.zeros(len(cats))
    P_i = []
    for labels in annotator_labels_per_claim:
        counts = Counter(labels)
        for j, c in enumerate(cats):
            p_j[j] += counts[c]
        P_i.append((sum(counts[c] ** 2 for c in cats) - n) / (n * (n - 1)))
    p_j = p_j / (len(annotator_labels_per_claim) * n)
    P_bar = float(np.mean(P_i))
    P_e = float(np.sum(p_j ** 2))
    if P_e >= 1.0 - 1e-9:
        return 1.0
    return float((P_bar - P_e) / (1.0 - P_e))


def nli_entailment_rate(scores: List[List[float]], threshold: float = 0.5) -> float:
    rates = []
    for ws in scores:
        if not ws:
            continue
        rates.append(float(sum(1 for s in ws if s >= threshold) / len(ws)))
    return float(np.mean(rates)) if rates else 0.0


def eight_metric_summary(payload: dict) -> Dict[str, float]:
    return {
        "Unsup":           unsup_rate(payload["annot_labels"], payload["refutation_flags"], use_verifier=True),
        "Unsup_HumanOnly": unsup_rate(payload["annot_labels"], payload["refutation_flags"], use_verifier=False),
        "Coverage":        coverage_rate(payload["triggers"], payload["family_supported"]),
        "TTR":             ttr(payload["captions"], payload["claim_terms"]),
        "Orth":            rule_orthogonality(payload["family_supported"]),
        "Claims":          claim_density(payload["captions"], payload["n_claims"]),
        "FalseRef":        false_refutation_rate(payload["refutation_flags"], payload["annot_labels"]),
        "Kappa":           fleiss_kappa(payload["annotator_labels_per_claim"]),
        "NLI":             nli_entailment_rate(payload["nli_scores"]),
    }

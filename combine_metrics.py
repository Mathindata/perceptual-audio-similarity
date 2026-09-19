"""
combine_metrics.py

Combines two (or more) already-computed metric delta caches into one
composite metric, by z-score normalizing each metric's deltas (so wildly
different raw scales -- MFCC's DTW cost vs HuBERT's cosine distance -- don't
let one dominate by magnitude alone) and averaging.

Motivation: listening to the mfcc-vs-hubert disagreement cases showed the two
metrics fail on systematically different, non-overlapping classes of
phonetic contrast (MFCC: coarse manner-of-articulation confusions like
nasal/fricative or fricative/stop; HuBERT: fine spectral contrasts within a
manner class, like voicing or place of sibilants). If those failure modes
are genuinely complementary rather than correlated, a combined score should
beat either metric alone.

This works directly off the *_deltas.json cache files evaluate_metrics.py
already wrote -- no audio, no GPU, no re-running the underlying metrics.

Usage:
    python combine_metrics.py \\
        --deltas-cache ./metric_cache/mfcc_<hash>_deltas.json mfcc \\
        --deltas-cache ./metric_cache/hubert_L6_<hash>_deltas.json hubert_L6

    # unequal weighting (e.g. trust hubert more) -- weight is an optional
    # third value on each --deltas-cache, defaulting to 1.0
    python combine_metrics.py \\
        --deltas-cache ./metric_cache/mfcc_<hash>_deltas.json mfcc 1.0 \\
        --deltas-cache ./metric_cache/hubert_L6_<hash>_deltas.json hubert_L6 3.0
"""

import argparse
import json
from typing import Dict, List

import numpy as np

from evaluate_metrics import (
    abx_accuracy,
    correlation_with_human,
    probit_pseudo_r2,
    load_per_listener_answers,
    print_report,
)
from perceptimatic_loader import build_triplets


def zscore_normalize(deltas: Dict[str, float]) -> Dict[str, float]:
    """
    Scale a metric's deltas by their standard deviation -- WITHOUT centering
    on the mean. delta=0 is a meaningful decision boundary here (equal
    distance to X, i.e. the metric has no opinion), not an arbitrary
    reference point, and it needs to stay at 0 for every metric being
    combined. Mean-centering would shift it: a metric's "correct" deltas
    trend positive on average (that's what a working metric does), which
    pulls its overall mean above zero, which then drags that metric's
    UNINFORMATIVE entries into artificially negative territory once
    centered -- corrupting the sign exactly where the metric had nothing to
    say. Confirmed with a synthetic check: full mean+std normalization
    pushed a 2-metric combination's accuracy to 0.31 (worse than either
    metric alone, worse than chance), even though each metric individually
    scored ~0.70. Scale-only normalization fixes this.
    """
    vals = np.array(list(deltas.values()), dtype=float)
    std = vals.std()
    if std == 0:
        std = 1.0
    return {k: v / std for k, v in deltas.items()}


def combine_deltas(deltas_list: List[Dict[str, float]], weights: List[float] = None) -> Dict[str, float]:
    """Average several metrics' z-score-normalized deltas, restricted to
    triplets every metric has a delta for."""
    if weights is None:
        weights = [1.0] * len(deltas_list)
    normalized = [zscore_normalize(d) for d in deltas_list]
    common_ids = set(normalized[0].keys())
    for d in normalized[1:]:
        common_ids &= set(d.keys())

    total_weight = sum(weights)
    combined = {}
    for tid in common_ids:
        combined[tid] = sum(w * d[tid] for w, d in zip(weights, normalized)) / total_weight
    return combined


def evaluate_combined(
    name: str,
    combined_deltas: Dict[str, float],
    human_scores: Dict[str, float],
    per_listener_answers: Dict[str, List[int]],
) -> dict:
    result = {"metric": name, "abx_accuracy": abx_accuracy(combined_deltas)}
    result.update(correlation_with_human(combined_deltas, human_scores))
    result.update(probit_pseudo_r2(combined_deltas, per_listener_answers))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deltas-cache", nargs="+", action="append", metavar="PATH NAME [WEIGHT]", required=True,
        help="a *_deltas.json path, a display name, and an optional weight (default 1.0); "
             "repeat for each metric to combine (at least 2), e.g. "
             "--deltas-cache mfcc.json mfcc 1.0 --deltas-cache hubert.json hubert_L6 3.0"
    )
    parser.add_argument("--triplets-csv", default="all_triplets.csv")
    parser.add_argument("--alignment-csv", default="all_aligned_clean_english.csv")
    parser.add_argument("--human-csv", default="human_and_models.csv")
    args = parser.parse_args()

    names, deltas_list, weights = [], [], []
    for entry in args.deltas_cache:
        if len(entry) not in (2, 3):
            parser.error(f"--deltas-cache expects PATH NAME [WEIGHT], got {len(entry)} values: {entry}")
        path, name = entry[0], entry[1]
        weight = float(entry[2]) if len(entry) == 3 else 1.0
        with open(path) as f:
            deltas_list.append(json.load(f))
        names.append(name)
        weights.append(weight)

    # Human scores + per-listener answers, no audio needed.
    triplets = build_triplets(
        triplets_csv=args.triplets_csv, alignment_csv=args.alignment_csv,
        human_csv=args.human_csv, with_audio=False,
    )
    human_scores = {t["triplet_id"]: t["human"] for t in triplets}
    per_listener_answers = load_per_listener_answers(args.human_csv)

    combined = combine_deltas(deltas_list, weights)
    combined_name = "+".join(f"{n}(w={w:g})" if w != 1.0 else n for n, w in zip(names, weights))
    print(f"Combined {len(names)} metrics ({', '.join(names)}, weights {weights}) "
          f"over {len(combined)} shared triplets.")

    results = [evaluate_combined(combined_name, combined, human_scores, per_listener_answers)]
    for name, deltas in zip(names, deltas_list):
        deltas_restricted = {tid: v for tid, v in deltas.items() if tid in combined}
        results.append(evaluate_combined(name, deltas_restricted, human_scores, per_listener_answers))

    print()
    print_report(results)


if __name__ == "__main__":
    main()

"""
evaluate_metrics.py

Evaluates one or more audio-distance metrics on the Perceptimatic English ABX
triplets, computing:

  1. ABX accuracy: fraction of triplets where the metric's delta has the
     correct sign (i.e. it judges A closer to X than B is).
  2. Correlation with mean human H: Pearson and Spearman correlation between
     the metric's per-triplet delta and the triplet's mean human score.
  3. Human agreement (Perceptimatic's own method): a probit regression of
     each individual listener's binarized answer on the metric's delta for
     that triplet, reporting McFadden's pseudo-R^2. This is the stronger
     test -- "does the metric fail on the same triplets humans find hard,"
     not just "does it get the same average direction right."

delta convention: delta = d(B, X) - d(A, X). Positive delta means the metric
judges A (TGT) closer to X than B (OTH) is -- the "correct" direction, which
matches the sign convention already used by the shipped human H and MFCC/DP
baseline columns in human_and_models.csv (confirmed: mean per-triplet
binarized_answer correlates positively with H, and the shipped MFCC delta
correlates positively with H too).

Usage:
    python evaluate_metrics.py --zrc-root /path/to/zrc2017-test-dataset \\
        --metrics mfcc logmel

Results are cached per metric to avoid recomputing DTW distances on rerun --
see --cache-dir.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from perceptimatic_loader import build_triplets


def compute_deltas(triplets: List[dict], metric) -> Dict[str, float]:
    """delta = d(B, X) - d(A, X) for every triplet, keyed by triplet_id."""
    deltas = {}
    for i, t in enumerate(triplets):
        d_ax = metric(t["A"]["wav"], t["X"]["wav"])
        d_bx = metric(t["B"]["wav"], t["X"]["wav"])
        deltas[t["triplet_id"]] = d_bx - d_ax
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(triplets)} triplets")
    return deltas


def load_per_listener_answers(human_csv: str) -> Dict[str, List[int]]:
    """triplet_id -> list of binarized_answer (-1/1) across listeners, EN only."""
    answers = defaultdict(list)
    with open(human_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["filename"]
            if fn.startswith("EN"):
                answers[fn].append(int(row["binarized_answer"]))
    return answers


def abx_accuracy(deltas: Dict[str, float]) -> float:
    return float(np.mean([d > 0 for d in deltas.values()]))


def correlation_with_human(deltas: Dict[str, float], human_scores: Dict[str, float]) -> dict:
    from scipy.stats import pearsonr, spearmanr

    ids = list(deltas.keys())
    d = np.array([deltas[i] for i in ids])
    h = np.array([human_scores[i] for i in ids])
    r_pearson, p_pearson = pearsonr(d, h)
    r_spearman, p_spearman = spearmanr(d, h)
    return {
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
    }


def probit_pseudo_r2(deltas: Dict[str, float], per_listener_answers: Dict[str, List[int]]) -> dict:
    """
    Fit y ~ delta by probit, where y is each listener's binarized answer
    (mapped from {-1, 1} to {0, 1}) and delta is broadcast to every listener
    on that triplet. Reports McFadden's pseudo-R^2 (statsmodels: prsquared)
    and the delta coefficient's sign/significance.
    """
    import statsmodels.api as sm

    y, x = [], []
    for triplet_id, answers in per_listener_answers.items():
        if triplet_id not in deltas:
            continue
        delta = deltas[triplet_id]
        for a in answers:
            y.append(1 if a > 0 else 0)
            x.append(delta)

    y = np.array(y)
    x = sm.add_constant(np.array(x))
    model = sm.Probit(y, x)
    result = model.fit(disp=0)
    return {
        "pseudo_r2": float(result.prsquared),
        "delta_coef": float(result.params[1]),
        "delta_pvalue": float(result.pvalues[1]),
        "n_listener_trials": len(y),
    }


def evaluate_metric(
    name: str,
    metric,
    triplets: List[dict],
    human_scores: Dict[str, float],
    per_listener_answers: Dict[str, List[int]],
    cache_dir: str = None,
) -> dict:
    triplet_ids = [t["triplet_id"] for t in triplets]
    required = set(triplet_ids)

    # Cache key includes the metric's config, so a code/parameter change (like
    # the n_fft fix) naturally invalidates old cache files instead of silently
    # reusing them.
    config = {k: v for k, v in vars(metric).items() if not k.startswith("_")}
    config_hash = json.dumps(config, sort_keys=True, default=str)
    import hashlib

    config_key = hashlib.sha1(config_hash.encode()).hexdigest()[:10]
    cache_path = os.path.join(cache_dir, f"{name}_{config_key}_deltas.json") if cache_dir else None

    deltas = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            deltas = json.load(f)
        cached_ok = required.issubset(deltas.keys())
        print(f"[{name}] found cache with {len(deltas)} entries "
              f"({'covers' if cached_ok else 'does NOT cover'} the requested {len(required)} triplets)")

    missing = [t for t in triplets if t["triplet_id"] not in deltas]
    if missing:
        print(f"[{name}] computing deltas for {len(missing)} triplet(s)...")
        deltas.update(compute_deltas(missing, metric))
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(deltas, f)
    else:
        print(f"[{name}] all {len(triplets)} triplets served from cache")

    deltas = {tid: deltas[tid] for tid in triplet_ids}  # restrict to the current run's set

    result = {"metric": name, "abx_accuracy": abx_accuracy(deltas)}
    result.update(correlation_with_human(deltas, human_scores))
    result.update(probit_pseudo_r2(deltas, per_listener_answers))
    return result


def print_report(results: List[dict]):
    cols = ["metric", "abx_accuracy", "pearson_r", "spearman_r", "pseudo_r2", "n_listener_trials"]
    widths = {c: max(len(c), 12) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        row = []
        for c in cols:
            v = r[c]
            row.append((f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(widths[c]))
        print("  ".join(row))


def build_metric_objs(metric_names: List[str], layers: List[int] = None) -> dict:
    """
    Turn CLI metric names into instantiated metric objects. "mfcc" and
    "logmel" are literal; anything else is treated as an embedding model
    name (wav2vec2, hubert, wavlm, or a full HF model id) passed to
    EmbeddingDistance.

    If layers is given, every embedding model name is swept across those
    layers instead of using the model's default (last) layer -- one metric
    object per (model, layer) pair, named e.g. "wav2vec2_L7".
    """
    from audio_metrics import MFCCDistance, LogMelDistance, EmbeddingDistance

    metric_objs = {}
    for m in metric_names:
        if m == "mfcc":
            metric_objs[m] = MFCCDistance()
        elif m == "logmel":
            metric_objs[m] = LogMelDistance()
        elif layers:
            for layer in layers:
                metric_objs[f"{m}_L{layer}"] = EmbeddingDistance(model=m, layer=layer)
        else:
            metric_objs[m] = EmbeddingDistance(model=m)
    return metric_objs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zrc-root", required=True, help="path to zrc2017-test-dataset")
    parser.add_argument("--triplets-csv", default="all_triplets.csv")
    parser.add_argument("--alignment-csv", default="all_aligned_clean_english.csv")
    parser.add_argument("--human-csv", default="human_and_models.csv")
    parser.add_argument(
        "--metrics", nargs="+", default=["mfcc", "logmel"],
        help="mfcc, logmel, and/or an embedding model name (wav2vec2, hubert, wavlm)"
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None,
        help="sweep embedding metrics across these hidden_states layer indices "
             "(e.g. --layers 6 7 8 9). Ignored for mfcc/logmel. If omitted, "
             "embedding metrics use their default (last) layer."
    )
    parser.add_argument("--cache-dir", default="./metric_cache")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only evaluate the first N triplets (for a quick smoke run)"
    )
    args = parser.parse_args()

    print("Loading triplets with audio...")
    triplets = build_triplets(
        triplets_csv=args.triplets_csv,
        alignment_csv=args.alignment_csv,
        human_csv=args.human_csv,
        zrc_root=args.zrc_root,
        with_audio=True,
    )
    if args.limit:
        triplets = triplets[: args.limit]
    print(f"Loaded {len(triplets)} triplets.")

    human_scores = {t["triplet_id"]: t["human"] for t in triplets}
    per_listener_answers = load_per_listener_answers(args.human_csv)

    metric_objs = build_metric_objs(args.metrics, args.layers)

    results = []
    for name, metric in metric_objs.items():
        results.append(
            evaluate_metric(name, metric, triplets, human_scores, per_listener_answers, args.cache_dir)
        )

    print()
    print_report(results)


if __name__ == "__main__":
    main()

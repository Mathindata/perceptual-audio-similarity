"""
export_triplet_audio.py

Pulls the actual A/B/X audio for specific triplets out to .wav files you can
listen to directly, selected by:

  --mode easy          highest human H -- triplets listeners found clearest
  --mode hard           lowest human H -- triplets listeners found hardest /
                        got systematically wrong
  --mode metric-error    triplets where a single metric's delta disagrees with
                        the "correct" direction (delta <= 0), sorted by how
                        confidently wrong it was. Needs --deltas-cache.
  --mode compare         triplets where TWO metrics disagree with each other
                        (one right, one wrong), ranked by how clear-cut the
                        human judgment was. Needs --deltas-cache-a/-b.

Each selected triplet gets three files:
    {triplet_id}_A_{phone}_{speaker}.wav   (TGT -- shares X's center phone)
    {triplet_id}_B_{phone}_{speaker}.wav   (OTH -- different center phone)
    {triplet_id}_X_{phone}_{speaker}.wav   (reference)
plus a summary.csv in the same folder with triplet_id, phones, human H, and
any metric delta columns, so you know what you're listening to and why it
was picked.

Usage:
    # 10 easiest and 10 hardest triplets for humans
    python export_triplet_audio.py --zrc-root <path> --mode easy --n 10
    python export_triplet_audio.py --zrc-root <path> --mode hard --n 10

    # 10 triplets where MFCC got it wrong most confidently
    python export_triplet_audio.py --zrc-root <path> --mode metric-error \\
        --deltas-cache ./metric_cache/mfcc_<hash>_deltas.json --n 10

    # 10 clearest-to-humans triplets where MFCC failed but HuBERT succeeded
    python export_triplet_audio.py --zrc-root <path> --mode compare \\
        --deltas-cache-a ./metric_cache/mfcc_<hash>_deltas.json --name-a mfcc \\
        --deltas-cache-b ./metric_cache/hubert_L6_<hash>_deltas.json --name-b hubert_L6 \\
        --focus b_wins --n 10
"""

import argparse
import csv
import json
import os
from typing import List

from perceptimatic_loader import build_triplets


def select_easy_or_hard(triplets: List[dict], n: int, mode: str) -> List[dict]:
    """mode: 'easy' -> highest human H first; 'hard' -> lowest human H first."""
    reverse = mode == "easy"
    ranked = sorted(triplets, key=lambda t: t["human"], reverse=reverse)
    return ranked[:n]


def select_metric_errors(triplets: List[dict], deltas: dict, n: int) -> List[dict]:
    """
    Triplets where the metric's delta is <= 0 (it judged B, the WRONG token,
    closer to X than A) -- sorted so the most confidently wrong ones come
    first (most negative delta).
    """
    scored = [(t, deltas[t["triplet_id"]]) for t in triplets if t["triplet_id"] in deltas]
    errors = [(t, d) for t, d in scored if d <= 0]
    errors.sort(key=lambda pair: pair[1])  # most negative (most wrong) first
    return [t for t, d in errors[:n]]


def select_metric_disagreements(
    triplets: List[dict], deltas_a: dict, deltas_b: dict, n: int, focus: str
) -> List[dict]:
    """
    Triplets where two metrics disagree on the correct direction (sign of
    delta), i.e. one calls it right and the other calls it wrong. Two raw
    delta scales usually aren't comparable (e.g. MFCC's DTW cost vs a cosine
    distance in [0,2]), so ranking uses the shared, comparable signal instead:
    human |H| -- the triplets where the human judgment was clearest are the
    most damning/illustrative examples of one metric failing where the other
    (and the humans) succeeded.

    focus:
        "b_wins" -- A (e.g. mfcc) got it wrong, B (e.g. hubert) got it right
        "a_wins" -- A got it right, B got it wrong
        "both"   -- either kind of disagreement, mixed together
    """
    candidates = []
    for t in triplets:
        tid = t["triplet_id"]
        if tid not in deltas_a or tid not in deltas_b:
            continue
        a_correct = deltas_a[tid] > 0
        b_correct = deltas_b[tid] > 0
        if a_correct == b_correct:
            continue
        if focus == "b_wins" and not (not a_correct and b_correct):
            continue
        if focus == "a_wins" and not (a_correct and not b_correct):
            continue
        candidates.append(t)

    candidates.sort(key=lambda t: abs(t["human"]), reverse=True)
    return candidates[:n]


def export_clips(selected: List[dict], out_dir: str, extra_fields: List[tuple] = None):
    """extra_fields: list of (column_name, {triplet_id: value}) pairs to add
    to summary.csv, e.g. [("mfcc_delta", mfcc_deltas), ("hubert_delta", hubert_deltas)]."""
    import soundfile as sf

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.csv")
    extra_fields = extra_fields or []

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["triplet_id", "human_H", "A_phone", "B_phone", "X_phone"]
        header += [name for name, _ in extra_fields]
        writer.writerow(header)

        for t in selected:
            tid = t["triplet_id"]
            for role in ("A", "B", "X"):
                slot = t[role]
                fname = f"{tid}_{role}_{slot['phone']}_{slot['speaker']}.wav"
                sf.write(os.path.join(out_dir, fname), slot["wav"], 16000)

            row = [tid, t["human"], t["A"]["phone"], t["B"]["phone"], t["X"]["phone"]]
            row += [values.get(tid, "") for _, values in extra_fields]
            writer.writerow(row)

    print(f"Wrote {len(selected)} triplet(s) ({len(selected) * 3} wav files) to {out_dir}")
    print(f"See {summary_path} for what each one is.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zrc-root", required=True)
    parser.add_argument("--triplets-csv", default="all_triplets.csv")
    parser.add_argument("--alignment-csv", default="all_aligned_clean_english.csv")
    parser.add_argument("--human-csv", default="human_and_models.csv")
    parser.add_argument("--mode", choices=["easy", "hard", "metric-error", "compare"], required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out-dir", default="./listen_clips")
    parser.add_argument(
        "--deltas-cache", default=None,
        help="path to a *_deltas.json file from evaluate_metrics.py's --cache-dir "
             "(required for --mode metric-error)"
    )
    parser.add_argument("--deltas-cache-a", default=None, help="metric A's deltas file (--mode compare)")
    parser.add_argument("--deltas-cache-b", default=None, help="metric B's deltas file (--mode compare)")
    parser.add_argument("--name-a", default="metric_a", help="label for metric A in summary.csv")
    parser.add_argument("--name-b", default="metric_b", help="label for metric B in summary.csv")
    parser.add_argument(
        "--focus", choices=["b_wins", "a_wins", "both"], default="b_wins",
        help="--mode compare only: which disagreement direction to select "
             "(b_wins = A wrong, B right; default, since that's usually the "
             "interesting direction -- e.g. A=mfcc, B=hubert)"
    )
    args = parser.parse_args()

    if args.mode == "metric-error" and not args.deltas_cache:
        parser.error("--mode metric-error requires --deltas-cache")
    if args.mode == "compare" and not (args.deltas_cache_a and args.deltas_cache_b):
        parser.error("--mode compare requires --deltas-cache-a and --deltas-cache-b")

    print("Loading triplets with audio...")
    triplets = build_triplets(
        triplets_csv=args.triplets_csv,
        alignment_csv=args.alignment_csv,
        human_csv=args.human_csv,
        zrc_root=args.zrc_root,
        with_audio=True,
    )
    print(f"Loaded {len(triplets)} triplets.")

    extra_fields = []
    if args.mode == "metric-error":
        with open(args.deltas_cache) as f:
            deltas = json.load(f)
        selected = select_metric_errors(triplets, deltas, args.n)
        extra_fields = [("metric_delta", deltas)]
    elif args.mode == "compare":
        with open(args.deltas_cache_a) as f:
            deltas_a = json.load(f)
        with open(args.deltas_cache_b) as f:
            deltas_b = json.load(f)
        selected = select_metric_disagreements(triplets, deltas_a, deltas_b, args.n, args.focus)
        extra_fields = [(f"{args.name_a}_delta", deltas_a), (f"{args.name_b}_delta", deltas_b)]
    else:
        selected = select_easy_or_hard(triplets, args.n, args.mode)

    export_clips(selected, args.out_dir, extra_fields=extra_fields)


if __name__ == "__main__":
    main()

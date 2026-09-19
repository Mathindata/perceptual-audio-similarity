"""
perceptimatic_loader.py

Data module for the Perceptimatic ABX benchmark (zrc2017-test-dataset, English).

Chain: triplet (TGT_item, OTH_item, X_item) -> alignment row (index -> #file, onset,
offset, #phone, speaker) -> wav on disk <ZRC_ROOT>/english/1s/<#file>.wav -> cut
[onset, offset] at 16 kHz mono.

Roles:
    A = TGT_item  (shares center phone with X)
    B = OTH_item  (different center phone from X)
    X = X_item    (the reference stimulus)

Resolved facts (confirmed against the real files):
    - English filter is the "EN" prefix on the triplet `filename` column in
      all_triplets.csv (NOT "items resolve against the English alignment" -- that
      lets in French triplets that don't get a human score).
    - 2,214 EN triplets, every one with a real human score, 983 distinct wavs.
    - Human score column is "H" in human_and_models.csv, one row per listener;
      this loader averages H per triplet filename to get a mean human delta.
    - human_and_models.csv also ships MFCC and DP baseline delta columns, useful
      to sanity-check your own MFCC implementation.

Usage:
    from perceptimatic_loader import build_triplets
    triplets = build_triplets(
        triplets_csv="all_triplets.csv",
        alignment_csv="all_aligned_clean_english.csv",
        human_csv="human_and_models.csv",
        zrc_root="/path/to/zrc2017-test-dataset",
        with_audio=True,
    )
    # triplets[0] == {
    #     "triplet_id": "EN1",
    #     "A": {"file": "2566", "onset": 0.56, "offset": 0.91, "phone": "ɑ",
    #           "speaker": "8193", "wav": <np.ndarray or None>},
    #     "B": {...},
    #     "X": {...},
    #     "human": <float, mean H across listeners>,
    # }
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional

SAMPLE_RATE = 16000


def _load_alignment(alignment_csv: str) -> Dict[str, dict]:
    """index -> {#file, onset, offset, #phone, speaker}, keyed by the `index` column."""
    align = {}
    with open(alignment_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            align[row["index"]] = {
                "file": row["#file"],
                "onset": float(row["onset"]),
                "offset": float(row["offset"]),
                "phone": row["#phone"],
                "speaker": row["speaker"],
            }
    return align


def _load_human_scores(human_csv: str) -> Dict[str, float]:
    """Mean human `H` delta per triplet filename, averaged over listeners."""
    scores = defaultdict(list)
    with open(human_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = row.get("H", "")
            if h != "":
                scores[row["filename"]].append(float(h))
    return {fn: sum(vals) / len(vals) for fn, vals in scores.items() if vals}


def _load_baseline_deltas(human_csv: str) -> Dict[str, Dict[str, float]]:
    """Mean shipped MFCC / DP baseline deltas per triplet filename (for sanity checks)."""
    deltas = defaultdict(lambda: defaultdict(list))
    with open(human_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in ("MFCC", "DP"):
                v = row.get(col, "")
                if v != "":
                    deltas[row["filename"]][col].append(float(v))
    return {
        fn: {col: sum(vals) / len(vals) for col, vals in cols.items()}
        for fn, cols in deltas.items()
    }


def _wav_path(zrc_root: str, file_id: str) -> str:
    # Essentially all referenced stimuli live in english/1s/; each #file resolves
    # to exactly one path, so a direct lookup there is correct.
    return os.path.join(zrc_root, "english", "1s", f"{file_id}.wav")


def _cut_audio(wav_path: str, onset: float, offset: float):
    """Load wav_path and cut [onset, offset] seconds, resampled to 16 kHz mono."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Files are already 16 kHz mono per the dataset spec; resample defensively
        # if that's ever not the case.
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        except ImportError:
            raise RuntimeError(
                f"{wav_path} is at {sr} Hz, not {SAMPLE_RATE} Hz, and librosa is "
                "not installed to resample it. `pip install librosa` or verify "
                "your ZRC_ROOT copy."
            )
    start = int(round(onset * SAMPLE_RATE))
    end = int(round(offset * SAMPLE_RATE))
    return audio[start:end]


def build_triplets(
    triplets_csv: str = "all_triplets.csv",
    alignment_csv: str = "all_aligned_clean_english.csv",
    human_csv: str = "human_and_models.csv",
    zrc_root: Optional[str] = None,
    with_audio: bool = False,
    include_baselines: bool = False,
) -> List[dict]:
    """
    Build the list of English Perceptimatic triplets with alignment info, optional
    audio, and mean human score.

    Args:
        triplets_csv: path to all_triplets.csv
        alignment_csv: path to all_aligned_clean_english.csv
        human_csv: path to human_and_models.csv
        zrc_root: root of the zrc2017-test-dataset folder (required if with_audio)
        with_audio: if True, cut and attach the actual audio for A/B/X (needs
            zrc_root and `pip install soundfile`)
        include_baselines: if True, attach the shipped MFCC/DP mean deltas per
            triplet under triplet["baselines"], for sanity-checking your own
            metric implementations

    Returns:
        List of dicts, one per English triplet:
        {triplet_id, A: {file, onset, offset, phone, speaker, wav}, B: {...},
         X: {...}, human: float, [baselines: {MFCC, DP}]}
    """
    if with_audio and not zrc_root:
        raise ValueError("with_audio=True requires zrc_root to be set")

    align = _load_alignment(alignment_csv)
    human_scores = _load_human_scores(human_csv)
    baselines = _load_baseline_deltas(human_csv) if include_baselines else {}

    triplets = []
    skipped_no_human = 0

    with open(triplets_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            filename = row["filename"]
            if not filename.startswith("EN"):
                continue  # French triplets have no human score; drop them here.

            if filename not in human_scores:
                skipped_no_human += 1
                continue

            entry = {"triplet_id": filename, "human": human_scores[filename]}

            for role, item_key in (("A", "TGT_item"), ("B", "OTH_item"), ("X", "X_item")):
                idx = row[item_key]
                a = align[idx]  # raises KeyError loudly if a join ever breaks
                slot = {
                    "file": a["file"],
                    "onset": a["onset"],
                    "offset": a["offset"],
                    "phone": a["phone"],
                    "speaker": a["speaker"],
                    "wav": None,
                }
                if with_audio:
                    wav_path = _wav_path(zrc_root, a["file"])
                    slot["wav"] = _cut_audio(wav_path, a["onset"], a["offset"])
                entry[role] = slot

            if include_baselines and filename in baselines:
                entry["baselines"] = baselines[filename]

            triplets.append(entry)

    if skipped_no_human:
        # Should be 0 against the real files -- surfacing this in case a different
        # copy of human_and_models.csv is ever swapped in.
        print(f"[perceptimatic_loader] warning: {skipped_no_human} EN triplets had no human score")

    return triplets


if __name__ == "__main__":
    # Quick self-check against the CSVs alone (no audio, no ZRC_ROOT needed).
    triplets = build_triplets(with_audio=False)
    print(f"Loaded {len(triplets)} English triplets (expect 2214)")
    distinct_files = {
        t[role]["file"] for t in triplets for role in ("A", "B", "X")
    }
    print(f"Distinct wav files referenced: {len(distinct_files)} (expect 983)")
    print("Sample triplet:", triplets[0])

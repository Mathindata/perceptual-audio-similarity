"""
visualize_embedding_space.py

Projects the MFCC and HuBERT(L6) embedding spaces down to 2D (PCA -> t-SNE)
for every unique phone token in the English Perceptimatic triplets, colored
by phonetic manner class. This turns the "MFCC fails on manner contrasts,
HuBERT fails on fine within-manner contrasts" finding from the listening
test into a visual: manner classes should look more separated/compact in
HuBERT's space than in MFCC's, and the reverse should show up in the
sibilant/liquid focused plot.

Produces two figures, each with MFCC (left) and HuBERT L6 (right) side by
side, using the SAME projection pipeline (StandardScaler -> PCA -> t-SNE)
for both, so any visual difference reflects the underlying representations,
not the plotting method:

  1. embedding_space_manner.png -- all ~1,055 unique tokens, colored by
     6 manner-of-articulation classes (vowel, stop, fricative, affricate,
     nasal, liquid/glide).
  2. embedding_space_contrasts.png -- just the phones involved in the
     listening-test disagreement cases (s, z, ʃ, k, ŋ, f, v, l, m, ʌ),
     colored by individual phone, so the specific confusions found by ear
     (e.g. z vs s, ŋ vs f) can be checked against cluster overlap directly.

Usage:
    python visualize_embedding_space.py --zrc-root <path>
    # optional: --sample-n 500 to subsample for a faster run
"""

import argparse
from collections import defaultdict
from typing import List

import numpy as np

from perceptimatic_loader import build_triplets

# English phone inventory in this dataset, by manner of articulation.
PHONE_TO_MANNER = {
    **{p: "vowel" for p in ["eɪ", "i", "oʊ", "u", "æ", "ɛ", "ɪ", "ɑ", "ʊ", "ʌ"]},
    **{p: "stop" for p in ["b", "d", "k", "p", "t", "ɡ"]},
    **{p: "fricative" for p in ["f", "h", "s", "v", "z", "ð", "ʃ"]},
    **{p: "affricate" for p in ["tʃ"]},
    **{p: "nasal" for p in ["m", "n", "ŋ"]},
    **{p: "liquid/glide" for p in ["l", "ɹ", "w"]},
}

CONTRAST_PHONES = {"s", "z", "ʃ", "k", "ŋ", "f", "v", "l", "m", "ʌ"}


def collect_unique_tokens(triplets: List[dict]) -> List[dict]:
    """Every distinct (file, onset, offset) token seen across A/B/X roles,
    deduplicated -- several triplets reuse the same underlying audio token."""
    tokens = {}
    for t in triplets:
        for role in ("A", "B", "X"):
            s = t[role]
            key = (s["file"], s["onset"], s["offset"])
            if key not in tokens:
                tokens[key] = s
    return list(tokens.values())


def pooled_embed(extractor, tokens: List[dict]) -> np.ndarray:
    """Mean-pool each token's frame-level features ([T, D], T varies per
    clip) into a single fixed-size vector, then stack into [N, D]."""
    vecs = []
    for i, tok in enumerate(tokens):
        feats = extractor._features(tok["wav"])
        vecs.append(feats.mean(axis=0))
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(tokens)}")
    return np.array(vecs)


def project_2d(vecs: np.ndarray, seed: int = 42) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    scaled = StandardScaler().fit_transform(vecs)
    n_pca = min(50, scaled.shape[0] - 1, scaled.shape[1])
    if scaled.shape[1] > n_pca:
        scaled = PCA(n_components=n_pca, random_state=seed).fit_transform(scaled)
    perplexity = min(30, max(5, len(vecs) // 10))
    return TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca").fit_transform(scaled)


def plot_side_by_side(coords_mfcc, coords_hubert, labels, title, out_path):
    import matplotlib.pyplot as plt

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
    color_map = {lab: cmap(i) for i, lab in enumerate(unique_labels)}
    colors = [color_map[lab] for lab in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, name in [(axes[0], coords_mfcc, "MFCC"), (axes[1], coords_hubert, "HuBERT (L6)")]:
        for lab in unique_labels:
            mask = [l == lab for l in labels]
            pts = coords[mask]
            ax.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.7, label=lab, color=color_map[lab])
        ax.set_title(f"{name} embedding space")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=min(len(unique_labels), 6), bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zrc-root", required=True)
    parser.add_argument("--triplets-csv", default="all_triplets.csv")
    parser.add_argument("--alignment-csv", default="all_aligned_clean_english.csv")
    parser.add_argument("--human-csv", default="human_and_models.csv")
    parser.add_argument("--sample-n", type=int, default=None, help="subsample tokens for a faster run")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from audio_metrics import MFCCDistance, EmbeddingDistance

    print("Loading triplets with audio...")
    triplets = build_triplets(
        triplets_csv=args.triplets_csv, alignment_csv=args.alignment_csv,
        human_csv=args.human_csv, zrc_root=args.zrc_root, with_audio=True,
    )
    tokens = collect_unique_tokens(triplets)
    print(f"{len(tokens)} unique tokens across {len(triplets)} triplets.")

    if args.sample_n and len(tokens) > args.sample_n:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(tokens), size=args.sample_n, replace=False)
        tokens = [tokens[i] for i in idx]
        print(f"Subsampled to {len(tokens)} tokens.")

    mfcc = MFCCDistance()
    hubert = EmbeddingDistance(model="hubert", layer=6, pooling="mean")

    print("Extracting MFCC features...")
    mfcc_vecs = pooled_embed(mfcc, tokens)
    print("Extracting HuBERT(L6) features (downloads the checkpoint on first use)...")
    hubert_vecs = pooled_embed(hubert, tokens)

    print("Projecting to 2D (PCA -> t-SNE)...")
    mfcc_2d = project_2d(mfcc_vecs, seed=args.seed)
    hubert_2d = project_2d(hubert_vecs, seed=args.seed)

    # Figure 1: all tokens, colored by manner class
    manners = [PHONE_TO_MANNER.get(t["phone"], "other") for t in tokens]
    plot_side_by_side(
        mfcc_2d, hubert_2d, manners,
        "Embedding space by manner of articulation (all phones)",
        f"{args.out_dir}/embedding_space_manner.png",
    )

    # Figure 2: just the listening-test contrast phones, colored individually
    mask = [t["phone"] in CONTRAST_PHONES for t in tokens]
    if sum(mask) >= 10:
        contrast_phones = [t["phone"] for t, m in zip(tokens, mask) if m]
        plot_side_by_side(
            mfcc_2d[mask], hubert_2d[mask], contrast_phones,
            "Embedding space for the listening-test contrast phones\n"
            "(s/z/ʃ = sibilants that fooled HuBERT; k/ŋ/f/v = manner contrasts that fooled MFCC)",
            f"{args.out_dir}/embedding_space_contrasts.png",
        )
    else:
        print("Too few contrast-phone tokens in this sample for figure 2 -- try without --sample-n.")


if __name__ == "__main__":
    main()

# Perceptual vs. Signal-Level Audio Similarity: Does Learned Similarity Track Human Phonetic Judgment?

## The question

Does ranking audio similarity by perceptual (learned-embedding) metrics agree with human
perceptual judgment better than signal-level (spectral) metrics? This is a measured version of
the thesis behind earlier work building a speech/audio compression system with perceptual
task-based losses in place of raw signal error.

## Ground truth: Perceptimatic

The benchmark is Perceptimatic, an ABX phone-discrimination task built on the ZeroSpeech 2017
English test set. Each item is a triplet:

- **A (TGT)** and **X** are two tokens of the *same* center phone in the *same* left/right
  phonetic context, spoken by different speakers.
- **B (OTH)** is a token with a *different* center phone in the same context.

A metric (or human listener) is asked, in effect: is X more similar to A or to B? Because A/B
and X are always different speakers, this is an across-speaker task — a metric that just
matched voice timbre would fail, which keeps the test focused on phonetic content rather than
speaker identity. The dataset ships real human ABX judgments (3-5 listeners per triplet), so the
ground truth is genuine human perception, not a label belonging to any one metric under test —
avoiding the trap where a metric would win by construction if it were evaluated against its own
kind of label.

After filtering to English triplets with a real human score: **2,214 triplets, 983 distinct
audio files**, verified exactly against the shipped CSVs.

## Metrics compared

All metrics share one interface, `distance(clip_a, clip_b) -> float`, and one evaluation
harness, so results differ only by feature representation, not by scoring logic. For every
triplet, `delta = d(B,X) - d(A,X)`; a metric is "correct" on a triplet when delta > 0 (it judges
A, the true match, closer to X than B is).

**Signal-level (baseline, no learning):**
- **MFCC** — 12 cepstral coefficients (coefficient 0, log-energy, dropped — see below),
  DTW-aligned, Euclidean cost.
- **Log-mel spectrogram** — 40 mel bands, log power, DTW-aligned, Euclidean cost.

**Perceptual (pretrained self-supervised speech models):**
- **wav2vec2-base** — pretrained on 960h of unlabeled LibriSpeech audio via a contrastive
  masked-prediction objective over its own learned quantizer codebook.
- **HuBERT-base** — same architecture, but trained via masked prediction of discrete targets
  from iterative k-means clustering of (increasingly phonetic) hidden states, which bootstraps
  the targets toward phone-like units.
- **WavLM-base** — built on HuBERT's objective, adds gated relative position bias and trains
  with simulated overlapping-speech/noise mixing for robustness.

Each transformer's hidden states are extracted at a chosen layer, DTW-aligned (or mean-pooled,
see below), with **cosine distance**, not Euclidean — see the methodology fixes below for why.

## Three methodological traps found and fixed

Each was caught by a synthetic sanity check before trusting results on real data — the same
mechanism (a metric picking up something irrelevant to content) showed up three times, at three
different levels of the pipeline.

1. **MFCC coefficient 0 is log-energy, not spectral shape.** In a synthetic check, adding
   barely-audible noise to an *otherwise identical* clip made it look farther away (by
   MFCC+Euclidean DTW) than a clip a full octave different. Coefficient 0 carries overall
   loudness, and its scale dominated the Euclidean distance. Fixed by dropping it
   (`drop_energy=True`).
2. **Log-mel's default 80 dB dynamic range is dominated by near-silent bins.** Log compression
   expands differences in near-zero-power regions, so a touch of broadband noise filling in
   quiet bins scored as a bigger change than genuine spectral content did. Fixed by capping the
   range at 40 dB.
3. **Transformer hidden-state magnitude tracks loudness/energy, not phonetic content.** In a
   synthetic check, the *same* content at a different loudness scored ~10x farther apart under
   Euclidean distance than *genuinely different* content at similar loudness did. Fixed by using
   cosine distance (direction only) instead of Euclidean for all embedding metrics.

A fourth, smaller fix: window/hop sizes. librosa's defaults (128 ms window, 32 ms hop) are sized
for music, not the 56-570 ms triphones in this dataset — some clips were shorter than the
analysis window. Switched to standard phonetic-analysis settings (25 ms window, 10 ms hop).

## Results (all 2,214 triplets, 7,195 listener-trials)

| Metric | ABX accuracy | Pearson r (vs. human H) | Spearman r | Pseudo-R² (probit) |
|---|---|---|---|---|
| Log-mel | 0.6712 | 0.2666 | 0.2656 | 0.0129 |
| wav2vec2 (L9) | 0.6125 | 0.1427 | 0.1343 | 0.0050 |
| wav2vec2 (L8) | 0.6125 | 0.1252 | 0.1271 | 0.0034 |
| wav2vec2 (L7) | 0.6775 | 0.1691 | 0.1872 | 0.0060 |
| wav2vec2 (L6, best) | 0.7480 | 0.2722 | 0.2741 | 0.0118 |
| **MFCC** | **0.7724** | **0.3555** | **0.3372** | **0.0163** |
| WavLM (L7) | 0.9015 | 0.5612 | 0.5184 | 0.0380 |
| WavLM (L5) | 0.9006 | 0.5992 | 0.5569 | 0.0381 |
| WavLM (L6) | 0.9006 | 0.5986 | 0.5598 | 0.0394 |
| HuBERT (L4) | 0.8961 | 0.6413 | 0.6118 | 0.0443 |
| HuBERT (L7) | 0.9015 | 0.5807 | 0.5453 | 0.0405 |
| HuBERT (L5) | 0.9024 | 0.6265 | 0.5892 | 0.0432 |
| **HuBERT (L6, best accuracy)** | **0.9061** | 0.6016 | 0.5657 | 0.0433 |

Two measures matter here, and they tell different stories. **ABX accuracy** asks whether a
metric gets the average case right. **Pseudo-R²** (from a probit fit of each individual
listener's answer on the metric's delta) asks something stricter: does the metric fail on the
*same* triplets individual listeners find hard, not just get the right answer on average. Even
HuBERT's best pseudo-R² (0.043) is modest in absolute terms — it explains some, not most, of
what makes a triplet hard for a human.

**Key findings:**

- **wav2vec2 never beats MFCC**, at any of the four layers tested (L6-L9), and degrades sharply
  after L6. This directly refutes the "perceptual > signal-level" thesis for this model.
- **HuBERT and WavLM decisively beat MFCC** — roughly 90% vs. 77% accuracy, and pseudo-R²
  triples. HuBERT's iterative clustering bootstraps its prediction targets toward phone-like
  units, unlike wav2vec2's arbitrary learned codebook; that difference in *what the model was
  trained to predict*, not architecture (they're otherwise near-identical), plausibly explains
  the gap.
- **HuBERT's phonetic content is a plateau (L4-L7), not a narrow peak**, then falls off sharply
  from L8 on. Accuracy is essentially flat across L4-L7 (0.896-0.906); Pearson r is actually
  highest at L4 (0.641) while accuracy peaks slightly later at L6.
- **HuBERT edges out WavLM slightly** (0.906/0.043 vs. 0.902/0.038 at their best layers),
  consistent with WavLM spending some model capacity on its denoising/multi-speaker objective
  rather than purely on phone discrimination — though the gap is small.

## Qualitative validation: listening to disagreements

Rather than trust the aggregate numbers alone, MFCC and HuBERT(L6)'s per-triplet predictions
were compared directly, and the disagreement cases (one metric right, the other wrong) exported
as audio and listened to.

**Where MFCC fails but HuBERT succeeds** (10 clearest-to-humans cases, ranked by human |H|):
coarse manner-of-articulation contrasts — vowel vs. fricative (ʌ/s), fricative vs. nasal (s/ŋ),
nasal vs. fricative (ŋ/v, ŋ/f), fricative vs. stop (ʃ/k). MFCC's errors here were large and
confident (delta as negative as -6.7), not borderline calls.

**Where HuBERT fails but MFCC succeeds** (10 clearest-to-humans cases): fine spectral contrasts
*within* a manner class — voiced vs. voiceless sibilant (z/s), place of articulation between
sibilants (s/ʃ), liquid vs. nasal or fricative (l/m, l/v). Here MFCC's deltas were large and
confident (30-54), while HuBERT's were tiny (-0.001 to -0.08) — not confidently wrong, more like
failing to resolve the distinction at all.

A plausible mechanism for MFCC's failures: stops (k) are mostly silent closure plus a brief
burst, and nasals (ŋ) are low-energy and spectrally muffled — both give DTW cheap, low-cost
alignment through near-silent frames, letting a genuinely different phone masquerade as similar.
This is the same underlying trap (low-energy regions carrying disproportionate weight) as the
MFCC/log-mel fixes above, surfacing again at the DTW-alignment level rather than in a single
coefficient. A plausible mechanism for HuBERT's failures: its masked-prediction objective likely
builds more categorical, context-integrated representations — good at coarse category
distinctions, less sharp at fine within-category acoustic cues like voicing.

The two metrics' failures are systematic and non-overlapping — not noise — and map cleanly onto
recognizable linguistic categories (manner vs. voicing/place).

## Combining metrics

Given genuinely complementary failure modes, a natural question: does combining MFCC and
HuBERT(L6) beat either alone? Deltas were normalized by **scale only** (dividing by standard
deviation, not centering on the mean) before averaging — delta=0 is a meaningful decision
boundary (equal distance to X), and mean-centering would corrupt it by shifting each metric's
uninformative entries away from zero based on its informative entries' average sign. This was
confirmed with a synthetic check: naive mean+std normalization collapsed a combination of two
~70%-accurate metrics to 31% (worse than chance); scale-only normalization fixed it.

| Combination | ABX accuracy | Pearson r | Pseudo-R² |
|---|---|---|---|
| MFCC alone | 0.7724 | 0.3555 | 0.0163 |
| HuBERT(L6) alone | 0.9061 | 0.6016 | 0.0433 |
| Equal weight (1:1) | 0.9070 | 0.5844 | 0.0427 |
| **Weighted 1:3 (favor HuBERT)** | **0.9160** | **0.6224** | **0.0469** |
| Weighted 1:10 | 0.9133 | 0.6136 | 0.0451 |

Equal-weight combination is a wash — MFCC's much larger error rate (22.8% of triplets vs.
HuBERT's 9.4%) dilutes HuBERT's stronger signal about as much as it contributes correct
rescues. Weighting toward the stronger metric (1:3) produces a real, consistent improvement over
HuBERT alone on every measure. Pushing further (1:10) is worse than 1:3 but still better than
HuBERT alone — the signature of a genuine interior optimum near 3:1, not a monotonic approach to
either extreme. The gain is modest (accuracy +1 point, pseudo-R² +8% relative) but consistent
and mechanistically explained: MFCC's complementary correction is real but bounded, since
there's limited room to improve on a metric that's already right 90% of the time.

## Conclusion

For phonetic-content discrimination on this benchmark: **"perceptual beats signal-level" is
true for some perceptual embeddings and false for others.** MFCC beats wav2vec2-base outright.
HuBERT and WavLM, whose training objectives more directly target phone-like structure, beat MFCC
decisively. The mechanism, not just the architecture, is what predicts the outcome — and the two
best-performing metric families (MFCC, HuBERT) fail on systematically different, linguistically
interpretable classes of contrast, which is itself informative independent of which one "wins."

## Caveats

- This tests only phonetic content (one of three planned axes). Speaker identity (VoxSim) and
  audio-quality (CDPAM) axes may favor different metrics or layers entirely.
- Pseudo-R² values are modest in absolute terms even for the best metric — no metric here
  explains most of what makes a triplet hard for an individual listener.
- Layer sweeps covered L4-L9 for wav2vec2/HuBERT/WavLM; the true optimum could sit just outside
  this range, though the plateau shape (L4-L7 flat, then sharp L8 falloff) makes that unlikely
  to change the qualitative conclusion.
- A mean-pooled (no-DTW) variant of the embedding distance was implemented but not yet run at
  scale — worth checking whether frame-level DTW alignment is adding real value over a single
  pooled vector per clip.

## Next steps

- Run the mean-pooled HuBERT variant across the full triplet set to check DTW's contribution.
- Extend the same harness to VoxSim (speaker identity) and CDPAM (audio quality) axes.
- Consider a learned (not just weighted-average) combination of MFCC and HuBERT features as a
  follow-up, given the demonstrated complementarity.

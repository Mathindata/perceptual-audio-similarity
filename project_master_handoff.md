# Portfolio Projects: Master Handoff

A single note to resume this work in a fresh session. It covers the audio-similarity project in
detail (that is the active build), then the state of the other projects, and the file names for
each so nothing has to be reconstructed.

These are portfolio pieces built against a Generative AI Data Scientist / Researcher job posting.
The through-line across all of them: honest evaluation, catching the trap most people miss, and
reporting what is real rather than what flatters.

---

## ACTIVE PROJECT: Perceptual vs Signal-Level Audio Similarity (Problem 3)

### The question
Does ranking audio similarity by perceptual (learned-embedding, task-based) metrics agree with
human perceptual judgment better than signal-level (spectral, L2) metrics? This is the measured
version of the thesis behind Matin's old perceptual-loss speech compression work.

### The chosen first axis: phonetic content (Perceptimatic)
An ABX phone-discrimination benchmark. Each triplet is three short triphones: A and X share the
center phone, B differs. Humans picked whether A or B sounds more like X. The dataset ships the
per-listener human results, so the human ground truth is already collected. This axis was chosen
because the ground truth is real human perception and it suits Matin's speech background.

### The core methodological guard (do not lose this)
Ground truth must be independent of the metrics under test. If you define similarity by an
attribute label and test an embedding of that same attribute, the metric wins by construction
and the result is meaningless. Perceptimatic avoids this: the human judgments are holistic
phone-discrimination choices, not the label of any single metric. Keep every claim tied to which
ground truth produced it.

### DATA MODULE: DONE and verified
The full data chain is built and tested against real metadata. Chain:
`triplet (TGT_item, OTH_item, X_item)` -> `alignment row (index -> #file, onset, offset, #phone,
speaker)` -> `wav on disk <ZRC_ROOT>/english/1s/<#file>.wav` -> `cut [onset, offset]` at 16 kHz mono.

Resolved facts (all confirmed with the real files):
- Dataset: **zrc2017 test set**, English (NOT abxLS, NOT zr2015, NOT the Task 2 dataset — all
  three were wrong turns). Downloaded via the ZeroSpeech toolbox: `zrc datasets:pull
  zrc2017-test-dataset`.
- Filenames are the `#file` number directly: `7180.wav`. Essentially all referenced stimuli live
  in the `english/1s/` subfolder (26,338 files; `10s` has 4,053, `120s` has 261). Each referenced
  `#file` resolves to exactly one path, so first-match search is correct.
- English filter is the **`EN` id prefix** on the triplet `filename`, NOT "items resolve against
  the English alignment" (that let in 46 French triplets with no human score). Correct result:
  **2,214 English triplets, every one with a real human score**, 983 distinct wavs.
- Human score column is **`H`** in `human_and_models.csv` (per-listener; the loader averages to a
  mean human delta per triplet). Confirm the column name in your copy before trusting it.
- The dataset ALSO ships `MFCC` and `DP` baseline delta columns — use these to sanity-check your
  own MFCC implementation.

Loader file: **`perceptimatic_loader.py`**. To run for real: place it beside the three CSVs, set
`ZRC_ROOT` to the `zrc2017-test-dataset` folder, `pip install soundfile`, call
`build_triplets(with_audio=True)`. It returns a list of dicts:
`{triplet_id, A/B/X: {file, onset, offset, phone, speaker, wav}, human}`.
Roles: A = TGT (same center phone as X), B = OTH (different), X = reference.

### NEXT STEP: the metrics layer (not yet built)
Build these behind ONE uniform interface so signal and perceptual metrics are interchangeable:
```
distance(clip_a, clip_b) -> float     # lower = more similar
```
- Signal-level (baseline): MFCC distance via DTW; log-mel spectral distance. (Check MFCC against
  the shipped `MFCC` column.)
- Perceptual (Matin's DNN side): DTW distance over wav2vec2 / HuBERT / WavLM frame features;
  optionally Whisper encoder features. CLAP/MERT are a poor fit for fine phonetic contrasts, so
  leave them out of this axis.

### THEN: the evaluation
For each metric and each triplet compute `delta = d(B,X) - d(A,X)`.
- ABX accuracy: fraction of triplets where `delta` has the correct sign.
- Human agreement (the real target): how well `delta` predicts the human `H`, following
  Perceptimatic's method (a probit regression of human responses on the model's delta). This asks
  "does the metric fail where humans fail," which is stronger than raw accuracy.
Report per metric; the honest possible outcome is that signal-level does fine on some axis, which
would partly refute the thesis. Report it straight.

### Design doc and diagram
Full system design: **`audio_sim_design.md`** / `.html`, with architecture diagram
`audio_sim_design.png`. Note: that design was written before the axis was fixed to Perceptimatic;
the per-section splitting it describes is not needed here (stimuli are already short triphones),
and the metric set narrows to speech-representation models as above.

### File inventory for this project
- `perceptimatic_loader.py` — the working data module (DONE).
- `all_triplets.csv` (tab-separated; `filename` is the triplet id, `EN`/`FR` prefixed).
- `all_aligned_clean_english.csv` (tab-separated; `index -> #file, onset, offset, #phone, speaker`).
- `human_and_models.csv` (comma-separated; per-listener rows, `H` human column, plus `MFCC`/`DP`
  baselines).
- `needed_zrc2017_wavs_english.csv` — the 983-file list actually referenced.
- `audio_sim_design.md` / `.html` / `audio_sim_design.png` — design and diagram.

### Future axes (after phonetic content works)
Same harness, swap the ground truth: VoxSim (speaker identity, ~70k human ratings on VoxCeleb),
CDPAM (audio quality/artifact triplets). Running across axes is what turns one number into a
finding — signal-level may win on one axis and lose badly on another.

---

## PROJECT STATE: the others

### 1. LLM fine-tuning + coding agent (worked on heavily; a real result in hand)
Fine-tuned Qwen2.5-Coder-3B (QLoRA/Unsloth, OpenHands trajectory data) to act as a coding agent,
built the agent loop, and diagnosed it rigorously. **Key finding: the partial fine-tune (~1.1 of 3
epochs, GPU quota ran out) made the model WORSE than the base model** — base produced a usable
action 80% of the time and the correct fix first try; the adapter managed 20% and rambled. This
is an honest, quantified before/after, a stronger portfolio story than a fine-tune that happened
to work.
- Open: finish training to 2-3 epochs and rerun the base-vs-adapter comparison to see if the
  adapter recovers (needs GPU). Then Phases 4 (evaluation) and 5 (verifier: raw vs verified pass
  rate).
- Files: `phase1_data_prep.ipynb`, `phase2_finetune.ipynb`, `phase3_agent_loop.ipynb`,
  `agent_core.py` (the tested ReAct loop; parser handles the OpenHands action format),
  `coding_agent_handoff.md`/`.html`, `coding_agent_interview_prep.md`/`.html`,
  `finetune_diagnostic_report.md`/`.html` (the regression finding, written up clean).

### 2. Vector search / retrieval (this is the umbrella the audio project sits under)
The audio-similarity project IS the concrete vector-search build. The broader menu (semantic
search done rigorously, dense vs sparse vs hybrid, a chunking study, ANN index tradeoffs) remains
available if a second retrieval piece is wanted later.

### 3. ANP flood-forecasting research note (substantially done; arXiv-style draft exists)
Grew from "write up the flood result" into a real experiment: Attentive Neural Process attention-
as-kernel study on Bow River flow. **Finding: the simplest explicit kernel (Laplace attention)
beats learned multi-head attention** (test NSE 0.89 vs 0.85), a matching-inductive-bias-to-problem
result. All attention types underestimate the flood peak (honest limitation, quantified).
- Open: verify the `[VERIFY]`-tagged citations before posting; add per-year error bars (only 5
  test years); regenerate the hydrograph from a full-length run.
- Files: `anp_bow_arxiv.pdf` / `.tex` (compiled 6-page draft), `anp_bow_experiments.ipynb` (runs
  on `bow_merged.csv`), figures `fig*.png`, `flood_note_outline_v2.md`/`.html` (the reframed
  scaffold around the faithfulness question, which is a SEPARATE future study — the GP-ARD
  faithfulness test was scoped but never run).

### Cross-project notes
- Compute reality: fine-tuning and the ANP full runs are GPU-bound and kept hitting free-Colab
  quota limits. The audio metrics layer is the low-compute path (pretrained inference + CPU
  scoring).
- Writing style for any drafted output: no em dashes, plain words, Matin's own voice, don't
  overuse "I"; never invent citation details — leave a field blank and flag it.

---

## Resume checklist for the audio project
- [ ] Point `perceptimatic_loader.py` at the real `zrc2017-test-dataset`, `pip install soundfile`,
      confirm `build_triplets(with_audio=True)` cuts audio (2,214 triplets).
- [ ] Build the metrics layer behind `distance(a, b)`: MFCC-DTW first (check vs shipped `MFCC`
      column), then a wav2vec2/HuBERT feature-DTW metric.
- [ ] Build evaluation: per-triplet `delta = d(B,X) - d(A,X)`; ABX sign-accuracy and probit fit to
      human `H`; per-metric table.
- [ ] Report honestly, ground-truth caveat attached.
- [ ] Later: add VoxSim (speaker) and CDPAM (quality) axes on the same harness.

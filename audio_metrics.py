"""
audio_metrics.py

Signal-level and perceptual audio-distance metrics behind one uniform interface:

    distance(clip_a, clip_b) -> float     # lower = more similar

Every metric below is a callable object with this signature, so they are
interchangeable in the evaluation harness. Clips are 1-D float32 numpy arrays
at 16 kHz mono (the format perceptimatic_loader.build_triplets returns).

Signal-level (baseline):
    MFCCDistance         -- MFCC frames, DTW-aligned, mean path cost
    LogMelDistance        -- log-mel spectrogram frames, DTW-aligned, mean path cost

Perceptual (learned embeddings):
    EmbeddingDistance     -- frame features from a pretrained speech model
                             (wav2vec2 / HuBERT / WavLM), DTW-aligned, mean path cost

All frame-based metrics share the same DTW backbone (_dtw_mean_cost), so the
only thing that differs between them is how a clip is turned into a
[T, D] sequence of frames. That keeps the comparison honest: any accuracy
difference between metrics comes from the features, not from different
alignment or aggregation logic.
"""

import numpy as np

SAMPLE_RATE = 16000


def _dtw_mean_cost(feats_a: np.ndarray, feats_b: np.ndarray, metric: str = "euclidean") -> float:
    """
    DTW-align two [T, D] frame sequences and return the mean cost along the
    optimal path -- so distance is comparable across clips of different
    lengths. metric is passed straight to librosa.sequence.dtw (euclidean,
    cosine, etc. -- anything scipy.spatial.distance.cdist supports).
    """
    import librosa

    # librosa.sequence.dtw expects [D, T] matrices.
    D, wp = librosa.sequence.dtw(X=feats_a.T, Y=feats_b.T, metric=metric)
    # D[-1, -1] is the total accumulated cost of the optimal path; wp is the
    # warping path (list of index pairs), so len(wp) is the path length.
    total_cost = D[-1, -1]
    return float(total_cost / len(wp))


class MFCCDistance:
    """
    Signal-level baseline: MFCC frames + DTW. Cross-check against the shipped
    `MFCC` baseline column in human_and_models.csv.

    Drops coefficient 0 (log-energy) by default. It carries overall loudness,
    not spectral shape, and its scale is large enough to dominate a Euclidean
    DTW cost -- in a synthetic sanity check, a tiny bit of added noise moved
    c0 enough to make two *same-pitch* clips look farther apart than two
    clips a full octave apart. Set drop_energy=False to keep it if you have a
    specific reason to.
    """

    def __init__(
        self,
        n_mfcc: int = 13,
        sr: int = SAMPLE_RATE,
        drop_energy: bool = True,
        n_fft: int = 400,
        hop_length: int = 160,
    ):
        # n_fft=400 (25 ms) / hop_length=160 (10 ms) at 16 kHz are standard
        # phonetic-analysis window sizes. librosa's own defaults (n_fft=2048,
        # hop=512 -- a 128 ms window) are sized for music, not ~100-400 ms
        # triphones: on the shortest clips in this dataset (~56 ms) that
        # window doesn't even fit, and even on a median-length clip it
        # yields only a few frames -- too coarse for DTW to actually align
        # phonetic content within the clip.
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.drop_energy = drop_energy
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _features(self, clip: np.ndarray) -> np.ndarray:
        import librosa

        n_fft = min(self.n_fft, len(clip))
        mfcc = librosa.feature.mfcc(
            y=clip, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=n_fft, hop_length=self.hop_length
        )
        if self.drop_energy:
            mfcc = mfcc[1:]
        return mfcc.T  # [T, n_mfcc(-1)]

    def __call__(self, clip_a: np.ndarray, clip_b: np.ndarray) -> float:
        return _dtw_mean_cost(self._features(clip_a), self._features(clip_b))


class LogMelDistance:
    """
    Signal-level baseline: log-mel spectrogram frames + DTW.

    Caps the dynamic range at top_db=40 rather than librosa's default 80.
    At 80 dB, bins that are near-silent get logged into huge negative values,
    so a tiny bit of broadband noise (which fills those bins in) can move
    the distance more than an octave-scale pitch/content change does -- in a
    synthetic sanity check, an almost-inaudible noise floor added to an
    otherwise identical clip looked farther away than a genuinely different
    clip. Capping the range keeps the metric anchored to the signal that's
    actually audible.
    """

    def __init__(
        self,
        n_mels: int = 40,
        sr: int = SAMPLE_RATE,
        top_db: float = 40.0,
        n_fft: int = 400,
        hop_length: int = 160,
    ):
        # Same window/hop reasoning as MFCCDistance -- see its docstring/comment.
        self.n_mels = n_mels
        self.sr = sr
        self.top_db = top_db
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _features(self, clip: np.ndarray) -> np.ndarray:
        import librosa

        n_fft = min(self.n_fft, len(clip))
        mel = librosa.feature.melspectrogram(
            y=clip, sr=self.sr, n_mels=self.n_mels, n_fft=n_fft, hop_length=self.hop_length
        )
        log_mel = librosa.power_to_db(mel, top_db=self.top_db)
        return log_mel.T  # [T, n_mels]

    def __call__(self, clip_a: np.ndarray, clip_b: np.ndarray) -> float:
        return _dtw_mean_cost(self._features(clip_a), self._features(clip_b))


def _pooled_cosine_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """Mean-pool each [T, D] sequence over time to a single D-dim vector, then
    cosine distance between the two pooled vectors -- no DTW alignment."""
    from scipy.spatial.distance import cosine

    return float(cosine(feats_a.mean(axis=0), feats_b.mean(axis=0)))


class EmbeddingDistance:
    """
    Perceptual metric: frame-level hidden states from a pretrained speech
    representation model (wav2vec2 / HuBERT / WavLM), + DTW (or mean-pooling).

    Uses cosine distance by default, not Euclidean. A transformer hidden
    state's overall magnitude tracks things like loudness/energy in the
    segment, not phonetic content -- in a synthetic check, the same content
    at a different loudness scored roughly 10x FARTHER apart under Euclidean
    than genuinely different content at similar loudness did. Cosine distance
    (which only looks at direction) gets that ordering right. This is the
    same shape of trap as MFCC's energy coefficient and log-mel's dynamic
    range -- raw signal/embedding magnitude keeps turning out to be the thing
    that swamps the actual comparison unless it's explicitly stripped out.

    Requires: pip install torch transformers
    Downloads the checkpoint from the Hugging Face Hub on first use, so this
    needs internet access and (for a full run over ~2,214 triplets) a GPU is
    strongly recommended -- CPU works but is slow.

    CLAP and MERT are deliberately left out: they're tuned for broad semantic /
    musical similarity, not the fine phonetic contrasts this benchmark tests.
    """

    _MODEL_CHOICES = {
        "wav2vec2": "facebook/wav2vec2-base",
        "hubert": "facebook/hubert-base-ls960",
        "wavlm": "microsoft/wavlm-base",
    }

    def __init__(
        self,
        model: str = "wav2vec2",
        layer: int = None,
        device: str = None,
        metric: str = "cosine",
        pooling: str = "dtw",
    ):
        """
        Args:
            model: one of "wav2vec2", "hubert", "wavlm", or a full HF model id
            layer: which hidden_states layer to use (0 = input embeddings,
                None = last layer). Middle layers often carry the most
                phonetic information for these models -- worth sweeping.
            device: "cuda" or "cpu"; auto-detected if not given
            metric: distance metric for the DTW alignment (only used when
                pooling="dtw"). "cosine" (default) or "euclidean" -- see the
                class docstring for why cosine is the safer default.
            pooling: "dtw" (default) frame-by-frame DTW alignment, or "mean"
                to average each clip's frames into one vector first and take
                cosine distance between the two pooled vectors -- cheaper,
                and a useful check on whether DTW's frame-level alignment is
                actually adding anything over a single per-clip summary.
        """
        import torch
        from transformers import AutoModel, AutoFeatureExtractor

        model_id = self._MODEL_CHOICES.get(model, model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Stored as _extractor/_model (leading underscore) rather than
        # extractor/model: evaluate_metrics.py builds its cache key from
        # vars(self), filtering out underscore-prefixed attributes. Without
        # that, the cache key would be built by stringifying the actual
        # loaded PyTorch model and HF feature extractor objects instead of
        # this config -- which is slow, and turned out to NOT even be stable
        # across runs (two runs of the supposedly identical hubert_L6 config
        # produced two different cache files). model_name keeps the plain
        # string ("hubert", "wav2vec2", ...) as the actual cache-key field.
        self._extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id, output_hidden_states=True)
        self._model.to(self.device).eval()
        self.model_name = model
        self.layer = layer
        self.metric = metric
        self.pooling = pooling
        self._torch = torch

    def _features(self, clip: np.ndarray) -> np.ndarray:
        torch = self._torch
        inputs = self._extractor(
            clip, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_values.to(self.device)
        with torch.no_grad():
            out = self._model(inputs)
        hidden_states = out.hidden_states  # tuple of [1, T, D] per layer
        layer_idx = -1 if self.layer is None else self.layer
        feats = hidden_states[layer_idx][0].cpu().numpy()  # [T, D]
        return feats

    def __call__(self, clip_a: np.ndarray, clip_b: np.ndarray) -> float:
        feats_a, feats_b = self._features(clip_a), self._features(clip_b)
        if self.pooling == "mean":
            return _pooled_cosine_distance(feats_a, feats_b)
        return _dtw_mean_cost(feats_a, feats_b, metric=self.metric)


if __name__ == "__main__":
    # Smoke test with synthetic audio (no real triplets needed) -- just checks
    # the DTW plumbing and feature extraction shapes are wired correctly.
    rng = np.random.default_rng(0)
    sr = SAMPLE_RATE
    t = np.linspace(0, 0.3, int(0.3 * sr), endpoint=False)
    clip_a = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    clip_b = (
        0.5 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(len(t))
    ).astype(np.float32)
    clip_c = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    mfcc_dist = MFCCDistance()
    d_ab = mfcc_dist(clip_a, clip_b)
    d_ac = mfcc_dist(clip_a, clip_c)
    print(f"MFCC distance, same pitch + noise: {d_ab:.4f}")
    print(f"MFCC distance, different pitch:    {d_ac:.4f}")
    assert d_ab < d_ac, "sanity check failed: near-identical clip should be closer"

    logmel_dist = LogMelDistance()
    d_ab2 = logmel_dist(clip_a, clip_b)
    d_ac2 = logmel_dist(clip_a, clip_c)
    print(f"LogMel distance, same pitch + noise: {d_ab2:.4f}")
    print(f"LogMel distance, different pitch:    {d_ac2:.4f}")
    assert d_ab2 < d_ac2, "sanity check failed: near-identical clip should be closer"

    print("Smoke test passed.")

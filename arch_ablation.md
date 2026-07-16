## Proposed Architecture — Sectioned with Ablations

### 0. Input Preprocessing
- Fixed input length: 4s (64,600 samples @ 16kHz), locked in for consistency with AASIST-L/RawNet2 literature and citability — not ablated here
  - **Parked / side quest:** shorter input length (out of scope for main results; revisit separately later if pursued, since it forfeits direct literature comparability)
- **Ablation:** pre-front-end downsampling (e.g. 16kHz → 8kHz) — tests whether 4-8kHz content is necessary; real params/compute savings if not, but literature (Li survey) flags this band as carrying unvoiced/silence-related discriminative content, so treat as a genuine hypothesis test, not a free win

### 1. Front End — Partial Polynomial Chirplet Decomposition (PPCD) + Band-Limited Resample (BLR)
- Primary contribution and main ablation focus of the project
- Per-channel learnable chirplet kernel (f1, f2, T, degree p), generated fresh each forward pass, kernel flipped before `F.conv1d` to preserve f1→f2 sweep-direction interpretability
- Per-channel adaptive FIR (windowed-sinc) bandpass filter, generated from current f1/f2, applied before band-limited resampling to prevent aliasing
- Complex demodulation (shift to baseband at channel center frequency) + lowpass + decimate; real and imaginary parts (I/Q) kept and concatenated as separate channels (not collapsed to magnitude) — preserves phase information, relevant given literature flags phase as a spoofing-relevant cue
- Shared, fixed max-BW-based decimation rate across all channels → all channel outputs sample-matched by construction
- **Ablation:** number of channels × nominal channel bandwidth (BW), tested as paired configs targeting similar total post-resample sample counts — tests resolution/coverage tradeoff at roughly fixed compute budget
- **Fixed, constrained (not ablated):** f1/f2 bounds (avoid 0 Hz and Nyquist), T range (10–100ms, physically motivated per speech literature), degree p lower bound (0.5); p upper bound TBD empirically via worst-case leakage sweep
- **Fixed, resolved via empirical sweep (not a live ablation):** guard-band margin for the adaptive BPF and resample bandwidth (BW'), determined by grid-sweeping parameter extremes and measuring worst-case spectral leakage, once p's upper bound is set

### 2. Per-Channel Normalization
- **Ablation:** none / layer norm / learned dynamic-range compression (PCEN-like, LEAF-inspired)
- Starting point: none or layer norm, given concern that PCEN-style suppression of steady-state content could discard task-relevant signal (e.g. channel/recording consistency cues) — treated as an open, testable question, not assumed

### 3. Encoder — CNN
- Standard conv stack over PPCD+BLR output, learned local/spectral-adjacent feature extraction
- Not a primary ablation target initially — conventional, proven component; revisit only if front-end ablations are inconclusive
- Init plan: inception net style (capture mult filter sizes before compression); add bottleneck(s) as necessary

### 4. Temporal Aggregation
- Simple self-attention (or lightweight Mamba, for comparison) over encoder output
- **Ablation:** self-attention vs. Mamba/SSM — secondary comparison, not the project's main contribution, informed by today's discussion that the sequence-block choice is less consequential than front-end quality
- Not over-engineered — conventional/proven components preferred here so effort stays concentrated on Section 1

### 5. Aggregation / Pooling
- **Starting default:** attentive statistics pooling (ASP)
- **Parked:** plain mean/max pooling — fallback comparison if ASP underperforms or is too costly for the param budget

### 6. Classifier Head
- Fully connected layers (1-2), dropout given class imbalance
- Not an ablation axis — kept simple by design

### 7. Loss Function
- Secondary consideration, not a primary contribution
- **Ablation:** select 3 most promising candidates from lit review (e.g. weighted BCE, OC-Softmax, focal) and test individually
- **Ablation:** combination/weighted-sum of the 3 selected losses, compared against each used individually

### 8. Data Augmentation
- **Ablation:** none / RawBoost / codec augmentation / noise+reverb (MUSAN+RIR)

### 9. Baseline Comparison
- AASIST-L, pretrained weights, evaluated through shared harness (`04_eval`) — reference point, not part of the ablation grid

---

**Ablation order:**
1. Front-end channel count × BW (paired to similar total sample counts)
2. Per-channel normalization (none / layer norm / learned PCEN-like)
3. Temporal aggregation (self-attention vs. Mamba)
4. Aggregation/pooling
5. Pre-front-end downsampling
6. Loss function (individual candidates)
7. Loss function (combination)
8. Data augmentation

**Dropped (not parked — premise no longer holds):**
- Band correlation block (Branch B) and all associated sub-ablations (mode, pairing, max_lag/chunk_size, alignment, attention-on-correlation-output) — motivating rationale (cheap/interpretable sequence-block replacement) superseded by literature evidence that front-end quality, not sequence-block design, drives the largest gains in this space
- Branch A / Branch B fusion — no longer applicable, no second branch to fuse

**Parked (not dropped, deprioritized):**
- Shorter input length (<4s) — side quest, forfeits literature comparability, separate from main results
- Plain mean/max pooling (vs. ASP)
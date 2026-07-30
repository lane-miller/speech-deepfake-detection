## Proof-of-concept plan

### POC1 — Chirplet generation exploration *(complete)*
- Single-term polynomial chirplet kernel generation (`pchirp`): instantaneous frequency, phase integration via discrete cumulative trapezoidal integration, kernel synthesis
- Constraint sanity-checking via synthetic parameter sweeps (BW, center frequency, duration, degree) and spectrogram-based visual verification
- Multi-term (Bézier) chirplet deferred pending single-term results
- Findings: degree range narrowed to ~0.5–3 (degenerate flat-then-jump behavior outside this range); BW/fc constraints need a logarithmic/ratio-based component (deferred to POC2); T kept learnable/variable per-channel

### POC2 — Chirplet constraint finalization
- Settle final BW, center-frequency, duration, and degree constraints before building anything downstream on provisional assumptions
- Resolve the log-ratio BW constraint flagged in POC1
- Output: finalized constraint ranges to carry into POC3+ and eventually the trainable parameterization layer

### POC3 — Real-speech convolution
- Using finalized constraints from POC2, generate a handful of representative chirplet kernels
- Convolve against real speech (LibriSpeech)
- Verify cross-correlation-vs-true-convolution flip behavior holds on real audio (not just synthetic impulse response)
- Inspect conv output via spectrogram
- Confirm output length matches expected convolution mode

### POC4 — BPF, spectral leakage, and I/Q
- Empirical worst-case spectral leakage sweep (across finalized f1/T/degree ranges) to size the BPF guard-band margin
- Complex demodulation
- I/Q output inspection (envelope and phase)
- One real utterance through chirplet conv → BLR
- Confirm output sample count matches `duration × BW`
- Confirm no obvious aliasing vs. expected band

### POC5 — Compute/timing sanity check
- Build realistic frontend
- Time one forward pass through the front end at target channel count
- Catch any accidentally expensive operation before committing to full training runs

### POC6 — Gradient flow check
- Toy front end + minimal encoder/head + dummy loss
- Confirm f1/f2/T/degree (or their underlying raw/reparameterized params) all receive gradients
- Confirm nothing silently detached in the custom kernel-generation code

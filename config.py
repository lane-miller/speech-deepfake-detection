import torch

# --- Audio front-end constants (shared source of truth) ---
# Chirp constraint constants
MIN_RATIO = 1.414  # sqrt(2), i.e. 1/2 octave minimum sweep
MAX_BW = 1900
MIN_BW = 200
MAX_FREQ = 7000
MIN_FREQ = 100
MAX_DUR_MS = 100
MIN_DUR_MS = 10
MAX_DEGREE = 3
MIN_DEGREE = 0.5

# FS is the target/desired input sample rate for the whole pipeline (Hz).
# Any input audio recorded at a different native sample rate is expected to
# be decimated/resampled to FS upstream, in the data-loading pipeline,
# *before* it ever reaches AudioFrontEnd. AudioFrontEnd and its submodules
# (ChirpletFilterbank, BLR) assume their input already arrives at this rate
# and perform no resampling of their own to correct for rate mismatches.
FS = 16000


# Sigmoid parameter squashing/mapping
def sigmoid_squash(theta, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(theta)
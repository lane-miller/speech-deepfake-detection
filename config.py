import torch
import os

# FS is the target/desired input sample rate for the whole pipeline (Hz).
# Any input audio recorded at a different native sample rate is expected to
# be decimated/resampled to FS upstream, in the data-loading pipeline,
# *before* it ever reaches AudioFrontEnd. AudioFrontEnd and its submodules
# (ChirpletFilterbank, BLR) assume their input already arrives at this rate
# and perform no resampling of their own to correct for rate mismatches.
FS = 16000

# --- Audio front-end constants (shared source of truth) ---
# Chirp constraint constants
MIN_RATIO = 1.414  # sqrt(2), i.e. 1/2 octave minimum sweep
MAX_BW = 1550 # assumes 2000 Hz bandwidth for BLR (safe upper bound)
MIN_BW = 200
MAX_FREQ = int(FS / 2 * 7 / 8)  # 7000 Hz used for 16kHz sample rate - maintain relation if FS changes
MIN_FREQ = 100
MAX_DUR_MS = 80
MIN_DUR_MS = 10
MAX_DEGREE = 3
MIN_DEGREE = 0.5

# --- Dataset paths ---
ASVSPOOF5_ROOT = "/Volumes/LPM03 storage/Datasets/Audio/asvspoof5"  # adjust to your actual root
ASVSPOOF5_PROTOCOL_TRAIN = os.path.join(ASVSPOOF5_ROOT, "ASVspoof5_protocols/ASVspoof5.train.tsv")
ASVSPOOF5_PROTOCOL_DEV = os.path.join(ASVSPOOF5_ROOT, "ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv")
ASVSPOOF5_PROTOCOL_EVAL = os.path.join(ASVSPOOF5_ROOT, "ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv")
ASVSPOOF5_AUDIO_ROOT = os.path.join(ASVSPOOF5_ROOT, "flac")  # or wherever the actual audio files live

# Sigmoid parameter squashing/mapping
def sigmoid_squash(theta, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(theta)
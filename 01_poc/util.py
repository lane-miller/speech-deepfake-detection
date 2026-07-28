"""
Utility functions for POC
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cti

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
FS = 16000

# pchirp - single term polynomial chirplet
def pchirp(freq1, freq2, duration_ms, degree):
    """
    """

    # Time vector
    T = duration_ms / 1000.
    t = np.arange(0., T, 1. / FS)

    # Inst freq
    f_i = freq1 + (freq2 - freq1) * (t / T) ** degree

    # Inst phase
    phi_i = 2 * np.pi * cti(f_i, t, initial=0)

    return np.sin(phi_i) * np.hanning(len(phi_i))

# Get bw and fc within constraints
def pick_bw_fc():
    """
    Selects bw and fc that automatically respect imposed constraints
    """
    # Define viable bw selection range based on constraints
    bw_lo = max(MIN_BW, (MIN_FREQ * (MIN_RATIO - 1)))
    bw_hi = min(MAX_BW, (MAX_FREQ - MIN_FREQ))

    # Select bw
    bw = pick_val(bw_lo, bw_hi)
    
    # Define viable fc selection range
    fc_lo = MIN_FREQ + bw / 2
    fc_hi = min((MAX_FREQ - bw / 2), (bw / 2 * (MIN_RATIO + 1) / (MIN_RATIO - 1)))

    # Select fc
    fc_log = pick_val(np.log(fc_lo), np.log(fc_hi))
    fc = np.exp(fc_log)
    
    return bw, fc

# Pick val from uniform distribution
def pick_val(lower_bnd, upper_bnd):
    return np.random.uniform(lower_bnd, upper_bnd)

# Get lower and upper band lims from fc and and bw
def bw_fc_to_f1f2(bw, fc):
    f1, f2 =  fc - bw / 2, fc + bw / 2
    if np.random.rand() < 0.5:
        f1, f2 = f2, f1
    return f1, f2


# Sigmoid parameter squashing/mapping
def sigmoid_squash(theta, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(theta)
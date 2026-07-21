"""
Utility functions for POC
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cti

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
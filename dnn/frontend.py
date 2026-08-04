"""
Audio front-end: chirplet filterbank + band-limited resampling (decimation-only).

Refactored from 01_poc/util.py (originally 01_poc/05_frontend.ipynb). All
shared constants (MAX_BW, MIN_BW, MAX_DEGREE, MIN_DEGREE, MAX_DUR_MS,
MIN_DUR_MS, MIN_RATIO, MIN_FREQ, MAX_FREQ, FS, sigmoid_squash) now live in
config.py, the single source of truth going forward.

config.FS is the target/desired input sample rate for the whole pipeline.
Input audio at a different native rate is expected to be decimated to
config.FS upstream, in the data-loading pipeline, before ever reaching
AudioFrontEnd. This module assumes its input already arrives at that rate
and performs no resampling to correct for rate mismatches.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.signal import firwin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MAX_BW,
    MIN_BW,
    MAX_DEGREE,
    MIN_DEGREE,
    MAX_DUR_MS,
    MIN_DUR_MS,
    MIN_RATIO,
    MIN_FREQ,
    MAX_FREQ,
    FS,
    sigmoid_squash,
)


# Define ChirpletFilterbank class
#    Note that learnable parameters theta_<param> get mapped to the actual inputs that create chirplets after passing though softplus/sigmoid for continuity

class ChirpletFilterbank(nn.Module):

    def __init__(self, n_channels, fs=FS):
        super().__init__()

        # Parameters
        self.n_channels = n_channels

        # Parameter constraints
        self.MAX_BW = MAX_BW
        self.MIN_BW = MIN_BW
        self.MAX_DEGREE = MAX_DEGREE
        self.MIN_DEGREE = MIN_DEGREE
        self.MAX_DUR_MS = MAX_DUR_MS
        self.MIN_DUR_MS = MIN_DUR_MS
        self.MIN_RATIO = MIN_RATIO
        self.MAX_FREQ = MAX_FREQ
        self.MIN_FREQ = MIN_FREQ
        self.FS = fs

        # Learnable parameter registration
        self.theta_bw = nn.Parameter(torch.zeros(n_channels))
        self.theta_fc = nn.Parameter(torch.zeros(n_channels))
        self.theta_T = nn.Parameter(torch.zeros(n_channels))
        self.theta_degree = nn.Parameter(torch.zeros(n_channels))
        self.theta_sign = nn.Parameter(torch.zeros(n_channels))

    def _get_constrained_params(self):
        """
        Apply warping so that continuous theta_* parameters respect constraints before being fed as chirplet gen inputs

        Take self.theta_* and return constrained version

        Internal method, called only by forward()
        """

        bw = sigmoid_squash(self.theta_bw, self.MIN_BW, self.MAX_BW)

        # fc's valid range depends on bw
        fc_lo = self.MIN_FREQ + bw / 2
        fc_hi = torch.min(self.MAX_FREQ - bw / 2, bw * (self.MIN_RATIO + 1) / (2 * (self.MIN_RATIO - 1)))
        fc = sigmoid_squash(self.theta_fc, fc_lo, fc_hi)

        T = sigmoid_squash(self.theta_T, self.MIN_DUR_MS, self.MAX_DUR_MS)
        degree = sigmoid_squash(self.theta_degree, self.MIN_DEGREE, self.MAX_DEGREE)

        return (bw, fc, T, degree)

    def _generate_kernels(self, bw, fc, T, degree):
        """
        Generate chirplet kernels from parameter tensors
        """

        # Compute f1, f2 from bw and fc tensors
        f1 = fc - bw / 2
        f2 = fc + bw / 2

        # Incorporate sweep direction
        w = torch.sigmoid(self.theta_sign)
        snap = (w > 0.5).float().detach() # snaps to 0 or 1 but removes grad flow
        disc = snap + (w - w.detach()) # allows for w gradient flow in backprop
        f_start = disc * f1 + (1 - disc) * f2
        f_end = disc * f2 + (1 - disc) * f1

        # Time vector (u = t / T) - right align so that .flip(-1) for conv1d give proper orientation
        Tmax = self.MAX_DUR_MS / 1000.
        L = int(Tmax * self.FS)
        L += 1 if L % 2 == 0 else 0
        t = torch.arange(L, device=bw.device)/ self.FS # keep this tensor on same device as others
        t_b = t.unsqueeze(0) # for broadcasting
        T_b = T.unsqueeze(1) / 1000 # for broadcasting
        t_shifted = t_b - (Tmax - T_b)   # shift so each channel's window starts at t_shifted=0, ending at t_shifted=T_b, right-aligned in the buffer
        mask = (t_shifted >= 0).float()
        u = torch.clamp(t_shifted / T_b, min=0.0, max=1.0)

        # Inst freq
        fsb = f_start.unsqueeze(1)
        feb = f_end.unsqueeze(1)
        degb = degree.unsqueeze(1)
        f_i = (fsb + (feb - fsb) * u ** degb) * mask

        # Inst phase - cumulative trapezoidal integration
        dt = 1.0 / self.FS
        phi_i = 2 * np.pi * dt * torch.cumsum((f_i[:, :-1] + f_i[:, 1:]) / 2, dim=1)
        phi_i = F.pad(phi_i, (1, 0))  # prepend 0, matching cumulative_trapezoid's initial=0

        # Variable length hann window per channel
        win = 0.5 * (1. - torch.cos(2 * np.pi * u)) * mask

        return torch.sin(phi_i) * win

    def forward(self, x):
        """
        Forward propagate

        x: [batch, 1, signal_len]
        """

        # Get kernel gen params
        bw, fc, T, degree = self._get_constrained_params()

        # Create kernels, reshape, and time reverse for actual conv
        k = self._generate_kernels(bw, fc, T, degree) # out: [n_channels, kernel_len]
        k = k.unsqueeze(1) # conv1d expects [out_ch, in_ch, kernel_len]
        k = k.flip(-1)

        # Conv
        pad = k.shape[-1] // 2
        y = F.conv1d(x, k, padding=pad)

        # return both y and k
        return (y, k.flip(-1).squeeze(1))


# Define BLR class (Band limited resampling)
#   No learnable params, just transforms and compresses inputs.
#
#   Decimation-only design: input audio is assumed to already be at config.FS
#   (decimated upstream in the data-loading pipeline if needed), and BLR's
#   own bandwidth reduction is pure integer decimation (down = FS // bandwidth)
#   with no interpolation/upsampling stage.

class BLR(nn.Module):

    def __init__(self, bandwidth, fs=FS):
        super().__init__()

        assert fs % bandwidth == 0, (
            f"FS ({fs}) must be evenly divisible by bandwidth ({bandwidth}); "
            "this decimation-only design requires an exact integer ratio."
        )

        self.FS = fs
        self.BW = bandwidth
        self.down = fs // bandwidth  # integer decimation factor, e.g. 16000 -> 2400 Hz

        # Define FIR LPF for decimation (no zero-stuffing, so no amplitude
        # scaling by an "up" factor is needed here)
        cutoff = 1.0 / self.down
        n_taps = 20 * self.down + 1
        filt = firwin(n_taps, cutoff, window=('kaiser', 5.0))
        self.register_buffer('fir_filter', torch.tensor(filt, dtype=torch.float32))

    def _hilbert_torch(self, x):
        """
        torch tensor compatible hilbert transform

        x: [...., signal_len]
        """
        N = x.shape[-1]
        X = torch.fft.fft(x, dim=-1) # transform along final dim

        h = torch.zeros(N, device=x.device, dtype=x.dtype) # hilbert spectrum mask
        h[0] = 1.
        if N % 2 == 0:
            h[N//2] = 1
            h[1:N//2] = 2
        else:
            h[1:(N+1)//2] = 2

        return torch.fft.ifft(X * h, dim=-1)

    def _chunked_conv1d(self, x_flat, filt, pad, chunk_size=4096):
        """
        Overlap-save chunked equivalent of F.conv1d(x_flat, filt, padding=pad).

        MPS backend does not support F.conv1d on long 1D signals in one shot
        (raises "Output channels > 65536 not supported"). A CPU-offload
        workaround was tried but is far too slow (~30s/call) to be usable.
        Instead, stay on MPS and do the "same"-padding convolution as a
        sequence of overlap-save chunks: pad the full signal once at its
        true global boundaries, then slide a window of
        (chunk_size + filter_len - 1) samples across it with stride
        chunk_size, running a *valid* (padding=0) conv1d on each window.
        Each valid conv on such a window yields exactly chunk_size output
        samples that are numerically identical to the corresponding slice
        of a single un-chunked "same"-padded conv1d, so concatenating them
        reproduces the full result with no boundary discontinuities.

        x_flat: [N, 1, L]
        filt: [1, 1, filter_len]
        pad: int, padding that would be passed to a single-call F.conv1d
             (must match filt.shape[-1] // 2 for exact same-length output)
        chunk_size: number of *output* samples produced per conv1d call.
             Keep small enough that a chunk's padded conv1d never comes close
             to the MPS "output channels > 65536" limit; tune down further
             if that error still appears on your device/build.
        """
        filter_len = filt.shape[-1]
        L_ = x_flat.shape[-1]
        x_padded = F.pad(x_flat, (pad, pad))  # zero-pad only at the true global boundaries
        chunks = []
        start = 0
        while start < L_:
            end = min(start + chunk_size, L_)
            n_out = end - start
            # Slice includes the (filter_len - 1) samples of overlap needed
            # from the preceding context so the valid conv is exact.
            in_start = start
            in_end = start + n_out + filter_len - 1
            x_chunk = x_padded[..., in_start:in_end]
            out_chunk = F.conv1d(x_chunk, filt)  # valid conv (padding=0) on the small chunk
            chunks.append(out_chunk)
            start = end
        return torch.cat(chunks, dim=-1)

    def _resample_poly_torch(self, x):
        """
        torch tensor compatible decimation-only resample (no upsampling stage).

        x: [..... signal_len], complex
        """
        # Get dims
        *batch_dim, L = x.shape

        # Split real / imag
        xr = x.real
        xi = x.imag

        # Conv1d to apply fir filter + necessary reshaping
        N = int(np.prod(batch_dim))
        xr_flat = xr.reshape(N, 1, L)
        xi_flat = xi.reshape(N, 1, L)

        filt = self.fir_filter.view(1, 1, -1)

        # MPS backend does not support F.conv1d on long 1D signals in one shot
        # (raises "Output channels > 65536 not supported"), so the filtering
        # is delegated to an overlap-save chunked conv1d helper that stays on
        # MPS. See _chunked_conv1d docstring for details.
        pad = self.fir_filter.shape[0] // 2  # matches padding used by a single-call conv1d
        xr_filt = self._chunked_conv1d(xr_flat, filt, pad)
        xi_filt = self._chunked_conv1d(xi_flat, filt, pad)

        xr_filt = xr_filt.reshape(*batch_dim, -1)
        xi_filt = xi_filt.reshape(*batch_dim, -1)

        # Decimate
        xr_dec = xr_filt[..., ::self.down]
        xi_dec = xi_filt[..., ::self.down]

        # Trim
        new_len = int(self.BW * (L / self.FS))
        assert new_len <= xr_dec.shape[-1], "Not enough samples post decimation"
        xr_dec = xr_dec[..., :new_len]
        xi_dec = xi_dec[..., :new_len]

        return torch.complex(xr_dec, xi_dec)

    def compute_fce_batch(self, kernels, threshold_db=-40):
        """
        Compute empirical band-center frequency per channel via -40dB edge detection.
        Intentionally non-differentiable: fce is a fixed empirical correction,
        not a learned quantity — used only to center demodulation.
        """
        with torch.no_grad():
            X = torch.fft.rfft(kernels, dim=-1)
            mag_db = 20*torch.log10(torch.abs(X) + 1e-12)
            mag_db -= mag_db.max(dim=-1, keepdim=True).values
            above = mag_db > threshold_db
            freqs = torch.fft.rfftfreq(kernels.shape[-1], 1/self.FS).to(kernels.device)
            idx = torch.arange(above.shape[-1], device=kernels.device)
            first_idx = torch.where(above, idx, idx.max()).min(dim=-1).values
            last_idx  = torch.where(above, idx, idx.min()).max(dim=-1).values
            f1e = freqs[first_idx]
            f2e = freqs[last_idx]
            return (f1e + f2e) / 2

    def forward(self, x, kernels):
        """
        x: [B, C, L], real, output of ChirpletFilterbank
        kernels: [C, kernel_len], chirplet kernels (for fce computation)
        """

        # Handle length internally if given input isn't already properly dimensioned
        L = x.shape[-1]
        trim_len = L - (L % self.down)
        if trim_len < L:
            x = x[..., :trim_len]
            L = trim_len

        xh = self._hilbert_torch(x)  # [B, C, L], complex

        fce = self.compute_fce_batch(kernels)  # [C]

        L = x.shape[-1]
        t = torch.arange(L, device=x.device, dtype=torch.float32) / self.FS  # [L]
        t_b = t.view(1, 1, L)
        fce_b = fce.view(1, -1, 1)  # [1, C, 1]

        xd = xh * torch.exp(-1j * 2 * torch.pi * fce_b * t_b)

        return self._resample_poly_torch(xd)


# Full AudioFrontEnd class

class AudioFrontEnd(nn.Module):

    def __init__(self, n_channels, bandwidth, fs=FS):
        super().__init__()

        self.chirplet_bank = ChirpletFilterbank(n_channels, fs=fs)
        self.blr = BLR(bandwidth, fs=fs)

    def forward(self, x):
        """
        Run the audio front end: chirplet filter bank -> band limited resampling

        x: [batch, signal_len] or [batch, 1, signal_len]

        Input is assumed to already be sampled at config.FS (or the fs override
        passed to this front end's constructor); no resampling of a mismatched
        native rate is performed here. All lengths are derived from the actual
        input tensor's shape, not from any fixed assumed duration.
        """
        # Reshape x to 3D if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)   # -> [batch, 1, signal_len]

        # Trim signal_len to multiple of the BLR decimation factor if needed
        L = x.shape[-1]
        trim_len = L - (L % self.blr.down)
        if trim_len < L:
            x = x[..., :trim_len]

        # Apply filter bank
        y, k = self.chirplet_bank(x)   # y: [B, C, L], real  k: [C, kernel_len]

        return self.blr(y, k)

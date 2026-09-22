"""DSP primitives used by the stages.

Everything here is plain ``torch``; no ``torchaudio``, to keep the dependency
surface small. Sinc resampling, FIR design and level helpers are short enough
to audit here.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Resampling (band-limited sinc, polyphase-style via conv1d)
# ----------------------------------------------------------------------------
def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _sinc_filter(
    up: int,
    down: int,
    zeros: int = 32,
    device=None,
    dtype=torch.float32,
) -> tuple[torch.Tensor, int]:
    """Design a windowed-sinc low-pass kernel for rational resampling.

    Returns the kernel and the per-side padding required. The cutoff is placed
    at the lower of the input/output Nyquist frequencies to prevent aliasing
    (when downsampling) or imaging (when upsampling).
    """
    max_rate = max(up, down)
    # Kaiser-style window via Hann here for simplicity & smooth roll-off.
    cutoff = 1.0 / max_rate  # normalised to the *upsampled* Nyquist
    half = zeros * max_rate
    n = torch.arange(-half, half + 1, device=device, dtype=dtype)
    # Ideal low-pass impulse response (normalized sinc).
    x = cutoff * n
    sinc = torch.where(
        x == 0,
        torch.ones_like(x),
        torch.sin(math.pi * x) / (math.pi * x),
    )
    window = torch.hann_window(n.numel(), periodic=False, device=device, dtype=dtype)
    kernel = sinc * window * cutoff
    return kernel, half


# Above this polyphase upsampling factor, the zero-stuffed intermediate signal
# and sinc kernel become enormous (e.g. 44100->24000 has up=80, which stuffs a
# 1 s clip to 3.5 M samples and needs a ~9.4k-tap kernel -> tens of GB). The
# internal pipeline rates (8/16/24/32/48 kHz) all reduce to up <= 4, so they
# always take the high-quality polyphase path; only exotic source rates such as
# 44.1/22.05 kHz hit the bounded interpolation fallback below.
_MAX_POLYPHASE_UP = 16


def _resample_interp(x: torch.Tensor, orig_sr: int, new_sr: int, numtaps: int = 257) -> torch.Tensor:
    """Bounded-memory resampler: anti-alias FIR + linear interpolation.

    Used as a fallback for awkward ratios (large polyphase ``up``) where the
    polyphase path would allocate tens of GB. Lower fidelity than the sinc
    polyphase resampler (correlation ~0.998 vs scipy ``resample_poly`` on a tone)
    but O(T) in memory. Anti-aliases before downsampling
    and removes imaging after upsampling.
    """
    b, c, t = x.shape
    if new_sr < orig_sr:  # downsample: band-limit to the new Nyquist first
        x = apply_fir(x, fir_lowpass(0.45 * new_sr, orig_sr, numtaps, device=x.device, dtype=x.dtype))
    target = max(1, int(round(t * new_sr / orig_sr)))
    y = F.interpolate(x.reshape(b * c, 1, t), size=target, mode="linear", align_corners=False)
    y = y.reshape(b, c, target)
    if new_sr > orig_sr:  # upsample: suppress imaging above the original Nyquist
        y = apply_fir(y, fir_lowpass(0.45 * orig_sr, new_sr, numtaps, device=y.device, dtype=y.dtype))
    return y


def resample(
    x: torch.Tensor,
    orig_sr: int,
    new_sr: int,
    zeros: int = 32,
) -> torch.Tensor:
    """Resample ``x`` ``[B, C, T]`` from ``orig_sr`` to ``new_sr``.

    Uses rational upsample-filter-downsample (polyphase) implemented with grouped
    conv1d. For non-integer ratios it reduces by the GCD first. When the reduced
    upsampling factor is very large (awkward ratios such as 44.1 kHz -> 24 kHz),
    falls back to a bounded-memory anti-aliased linear interpolation to avoid a
    multi-GB allocation.
    """
    if orig_sr == new_sr:
        return x
    g = _gcd(int(orig_sr), int(new_sr))
    up = int(new_sr) // g
    down = int(orig_sr) // g

    if up > _MAX_POLYPHASE_UP:
        return _resample_interp(x, int(orig_sr), int(new_sr))

    b, c, t = x.shape
    kernel, pad = _sinc_filter(up, down, zeros=zeros, device=x.device, dtype=x.dtype)

    # Upsample by zero-stuffing: insert (up-1) zeros between samples.
    xz = x.reshape(b * c, 1, t)
    if up > 1:
        upsampled = xz.new_zeros((b * c, 1, t * up))
        upsampled[..., ::up] = xz
    else:
        upsampled = xz

    # Low-pass filter via convolution. Scale by `up` to preserve energy.
    k = (kernel * up).view(1, 1, -1)
    filtered = F.conv1d(upsampled, k, padding=pad)

    # Downsample by taking every `down`-th sample.
    out = filtered[..., ::down]

    # Trim/pad to the ideal output length round(t * new/orig).
    target = int(math.floor(t * new_sr / orig_sr))
    if out.shape[-1] > target:
        out = out[..., :target]
    elif out.shape[-1] < target:
        out = F.pad(out, (0, target - out.shape[-1]))
    return out.reshape(b, c, target)


# ----------------------------------------------------------------------------
# FIR filtering
# ----------------------------------------------------------------------------
def fir_lowpass(
    cutoff_hz: float,
    sr: int,
    numtaps: int = 257,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Windowed-sinc low-pass FIR kernel (Hamming window)."""
    if numtaps % 2 == 0:
        numtaps += 1
    fc = cutoff_hz / (sr / 2.0)  # normalised (0..1, 1==Nyquist)
    fc = min(max(fc, 1e-4), 0.999)
    n = torch.arange(numtaps, device=device, dtype=dtype) - (numtaps - 1) / 2.0
    h = fc * torch.where(
        n == 0,
        torch.ones_like(n),
        torch.sin(math.pi * fc * n) / (math.pi * fc * n),
    )
    win = torch.hamming_window(numtaps, periodic=False, device=device, dtype=dtype)
    h = h * win
    h = h / h.sum()
    return h


def fir_highpass(
    cutoff_hz: float,
    sr: int,
    numtaps: int = 257,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Highpass via spectral inversion of a lowpass."""
    if numtaps % 2 == 0:
        numtaps += 1
    lp = fir_lowpass(cutoff_hz, sr, numtaps, device=device, dtype=dtype)
    hp = -lp
    hp[(numtaps - 1) // 2] += 1.0
    return hp


def apply_fir(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply a 1-D FIR ``kernel`` to ``x`` ``[B, C, T]`` with 'same' padding."""
    b, c, t = x.shape
    pad = (kernel.numel() - 1) // 2
    k = kernel.view(1, 1, -1).to(x.dtype)
    xr = x.reshape(b * c, 1, t)
    y = F.conv1d(xr, k, padding=pad)
    return y.reshape(b, c, t)


def bandpass(
    x: torch.Tensor,
    low_hz: Optional[float],
    high_hz: Optional[float],
    sr: int,
    numtaps: int = 257,
) -> torch.Tensor:
    """Apply a band-pass by cascading a high-pass and a low-pass FIR.

    ``low_hz=None`` skips the high-pass; ``high_hz=None`` skips the low-pass.
    The low-pass cutoff is clamped below Nyquist for safety.
    """
    out = x
    if high_hz is not None:
        hc = min(high_hz, 0.999 * sr / 2.0)
        out = apply_fir(out, fir_lowpass(hc, sr, numtaps, device=x.device, dtype=x.dtype))
    if low_hz is not None and low_hz > 0:
        out = apply_fir(out, fir_highpass(low_hz, sr, numtaps, device=x.device, dtype=x.dtype))
    return out


# ----------------------------------------------------------------------------
# Level / RMS helpers
# ----------------------------------------------------------------------------
def rms(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.mean(x**2, dim=dim) + eps)


def dbfs(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Full-scale RMS level in dB over the last dim, per [B, C]."""
    return 20.0 * torch.log10(rms(x) + eps)


def apply_gain_db(x: torch.Tensor, gain_db: float | torch.Tensor) -> torch.Tensor:
    if isinstance(gain_db, torch.Tensor):
        g = torch.pow(10.0, gain_db / 20.0)
        while g.dim() < x.dim():
            g = g.unsqueeze(-1)
    else:
        g = 10.0 ** (gain_db / 20.0)
    return x * g

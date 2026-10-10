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
# internal pipeline rates (8/16/24/32/48 kHz) all reduce to up <= 6 and take the
# zero-stuffing path; ratios such as 44.1/22.05 kHz <-> 8/16/24 kHz take
# _resample_bank, which applies the same low-pass without the zero-stuffing.
_MAX_POLYPHASE_UP = 16

# Most kernel taps _resample_bank builds for one conv1d. The ratios between
# common rates fit in one; a larger bank (both reduced factors large, such as
# 44101 -> 48000 Hz) is applied in blocks of phases.
_MAX_BANK_TAPS = 1 << 20


def _resample_bank(x: torch.Tensor, up: int, down: int, zeros: int = 32) -> torch.Tensor:
    """Resample ``x`` ``[B, C, T]`` by ``up / down`` (in lowest terms) without zero-stuffing.

    Output sample ``j`` lies at input position ``j * down / up``. It is the sum
    of the inputs within ``zeros`` zero crossings of that position, each
    weighted by the :func:`_sinc_filter` kernel (cutoff at the lower Nyquist
    frequency, Hann window, same gain) at its exact distance: the output of the
    zero-stuffing path, computed only at the output instants. The outputs with
    the same ``j % up`` (one phase) share a kernel, and the kernels of a block
    of phases are the output channels of one ``conv1d`` with stride ``down``.
    Distances come from integers, so there is no delay and no drift at any
    length. Returns ``max(1, round(T * up / down))`` samples.
    """
    b, c, t = x.shape
    hi = max(up, down)
    # The kernel reaches zeros * hi / up input samples either side of an
    # output's position, so a phase reads from k inputs before the integer part
    # of its position to k + 1 after it.
    k = int(zeros * hi // up)
    target = max(1, int(round(t * up / down)))
    frames = -(-target // up)                       # conv1d steps of `down` inputs, one output per phase each
    # cuDNN may run a float32 conv in TF32 (torch's default), which is far less
    # precise than the CPU; it never does in float64, so CUDA float32 convolves
    # in float64 and the result is cast back on assignment to y.
    work = torch.float64 if x.is_cuda and x.dtype == torch.float32 else x.dtype
    xp = F.pad(x.reshape(b * c, 1, t).to(work), (k, max(0, frames * down + k + 1 - t)))
    y = x.new_empty((b * c, frames, up))
    step = max(1, min(up, _MAX_BANK_TAPS // (down + 2 * k + 2)))   # a phase needs fewer than down + 2k + 2 taps
    for p0 in range(0, up, step):
        p = torch.arange(p0, min(up, p0 + step))   # the block's phases
        o = p * down // up                          # integer parts of their positions
        n = torch.arange(int(o[-1] - o[0]) + 2 * k + 2)
        # up * (input index - output position) for each phase and tap, exact in
        # int64; divided by hi it is the distance in zero crossings of the sinc.
        arg = ((o[0] - k + n) * up - p[:, None] * down).double() / hi
        win = torch.cos(arg * (math.pi / (2 * zeros))) ** 2
        h = torch.where(arg.abs() <= zeros, torch.sinc(arg) * win, torch.zeros_like(arg)) * (up / hi)
        o0, width = int(o[0]), (frames - 1) * down + n.numel()
        hb = h.to(device=x.device, dtype=work).unsqueeze(1)
        if b * c == 1 or step == up:
            yb = F.conv1d(xp[..., o0:o0 + width], hb, stride=down)
        else:
            # Several blocks: one conv per row on a contiguous slice, so the
            # rows are not copied out of the padded signal for every block.
            yb = torch.cat([F.conv1d(xp[r:r + 1, :, o0:o0 + width], hb, stride=down)
                            for r in range(b * c)])
        y[:, :, p0:p0 + p.numel()] = yb.transpose(1, 2)
    return y.reshape(b * c, frames * up)[:, :target].reshape(b, c, target)


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
    :func:`_resample_bank` applies the same low-pass without building the
    zero-stuffed signal. On both paths ``zeros`` is the kernel's half-width in
    zero crossings of its sinc, whose cutoff is the lower of the two Nyquist
    frequencies. An empty input gives an empty output.
    """
    if orig_sr == new_sr:
        return x
    if x.shape[-1] == 0:
        return x.new_zeros(x.shape)
    g = _gcd(int(orig_sr), int(new_sr))
    up = int(new_sr) // g
    down = int(orig_sr) // g

    if up > _MAX_POLYPHASE_UP:
        return _resample_bank(x, up, down, zeros=zeros)

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


# Output samples per iteration of linear_resize; bounds its index tensors.
_LINEAR_BLOCK = 1 << 20


def linear_resize(x: torch.Tensor, size: int) -> torch.Tensor:
    """Linearly interpolate ``x`` ``[B, C, T]`` to ``size`` samples.

    The interpolation of ``F.interpolate(x, size, mode="linear",
    align_corners=False)``: output ``j`` reads the input at position
    ``(j + 1/2) * T / size - 1/2``, clamped at 0. That position is the fraction
    ``((2j + 1) * T - size) / (2 * size)``, split into its integer part and
    remainder in int64, so it is exact at any length; only the interpolation
    weight is rounded, to the signal's dtype.
    """
    b, c, t = x.shape
    xf = x.reshape(b * c, t)
    y = xf.new_empty((b * c, size))
    den = 2 * size
    for s in range(0, size, _LINEAR_BLOCK):
        j = torch.arange(s, min(s + _LINEAR_BLOCK, size))
        num = ((2 * j + 1) * t - size).clamp_(min=0)
        i0 = num // den                             # at most t - 1
        w = ((num - i0 * den).double() / den).to(device=x.device, dtype=x.dtype)
        i1 = (i0 + 1).clamp_(max=t - 1).to(x.device)
        i0 = i0.to(x.device)
        y[:, s:s + j.numel()] = xf[:, i0] * (1 - w) + xf[:, i1] * w
    return y.reshape(b, c, size)


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

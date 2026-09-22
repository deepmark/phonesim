"""Channel band edges with a physical roll-off instead of a brick wall."""

from __future__ import annotations

import math

import torch

from phonesim.core import SimContext, Stage, resolve_range
from phonesim import dsp


def _biquad_highpass(x: torch.Tensor, fc: float, sr: int, order: int = 2) -> torch.Tensor:
    """Butterworth high-pass of even ``order`` as cascaded biquads."""
    b, c, t = x.shape
    y = x.reshape(b * c, t)
    w0 = 2.0 * math.pi * fc / sr
    for k in range(order // 2):
        # Butterworth pole angles for the cascade
        q = 1.0 / (2.0 * math.sin(math.pi * (2 * k + 1) / (2 * order)))
        alpha = math.sin(w0) / (2.0 * q)
        cw = math.cos(w0)
        b0, b1, b2 = (1 + cw) / 2, -(1 + cw), (1 + cw) / 2
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
        b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
        y = _iir2(y, b0, b1, b2, a1, a2)
    return y.reshape(b, c, t)


def _iir2(x: torch.Tensor, b0, b1, b2, a1, a2) -> torch.Tensor:
    """Direct-form-II biquad over the last axis via a scan in blocks of samples."""
    # Recurrence in Python is slow for long inputs; run it as a linear filter
    # via the impulse response truncated where it has decayed (IIR -> long FIR).
    n = x.shape[-1]
    # impulse response length: decay to -100 dB
    r = max(abs(complex(-a1 / 2, math.sqrt(max(a2 - a1 * a1 / 4, 0.0)))), 1e-6)
    length = int(min(n, max(64, math.ceil(math.log(1e-5) / math.log(min(r, 0.999999))))))
    h = torch.zeros(length, dtype=x.dtype, device=x.device)
    y1 = y2 = 0.0
    for i in range(length):
        xi = 1.0 if i == 0 else 0.0
        x1 = 1.0 if i == 1 else 0.0
        x2 = 1.0 if i == 2 else 0.0
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        h[i] = yi
        y2, y1 = y1, yi
    return torch.nn.functional.conv1d(
        torch.nn.functional.pad(x.unsqueeze(1), (length - 1, 0)), h.flip(0).view(1, 1, -1)
    ).squeeze(1)


def _rolloff_lowpass(x: torch.Tensor, f_pass: float, f_stop: float, stop_db: float, sr: int) -> torch.Tensor:
    """Low-pass with a linear-in-dB transition from ``f_pass`` (0 dB) to ``f_stop``
    (``-stop_db``), flat beyond. Realised as a zero-phase FIR from the target
    magnitude, so the slope is exact and adjustable."""
    n = 1024                                                   # design grid
    taps = 511                                                 # odd: zero phase
    f = torch.linspace(0.0, sr / 2.0, n // 2 + 1, device=x.device, dtype=x.dtype)
    slope = torch.clamp((f - f_pass) / max(f_stop - f_pass, 1.0), 0.0, 1.0)
    mag = 10.0 ** (-stop_db * slope / 20.0)
    h = torch.fft.irfft(torch.complex(mag, torch.zeros_like(mag)), n=n)
    h = torch.roll(h, n // 2)[n // 2 - taps // 2: n // 2 + taps // 2 + 1]
    h = h * torch.hann_window(taps, periodic=False, device=x.device, dtype=x.dtype)
    h = h / h.sum()
    return dsp.apply_fir(x, h)


class ChannelEdgeStage(Stage):
    """Band edges of a digital telephone channel.

    Low edge: a 2nd-order high-pass at ``low_hz`` (codec pre-processing and
    handset send masks are specified from 80-100 Hz, not the 300 Hz of the
    analogue loop). High edge: unity to ``pass_hz``, then a roll-off reaching
    ``stop_db`` at ``stop_hz`` (the channel Nyquist), flat beyond. Both edges
    may be ranges to vary per call. The roll-off FIR is zero phase; the high-pass
    is a causal biquad. Length preserving.
    """

    def __init__(
        self,
        low_hz=(80.0, 120.0),
        pass_hz=(3300.0, 3450.0),
        stop_hz=None,
        stop_db: float = 45.0,
        name=None,
    ):
        super().__init__(name=name or "ChannelEdge")
        self.low_range = resolve_range(low_hz) if low_hz is not None else None
        self.pass_range = resolve_range(pass_hz) if pass_hz is not None else None
        self.stop_hz = stop_hz
        self.stop_db = stop_db

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        sr = ctx.sample_rate
        y = x
        low = None
        if self.low_range is not None:
            low = ctx.uniform(*self.low_range)
            y = _biquad_highpass(y, low, sr, order=2)
        f_pass = None
        if self.pass_range is not None:
            f_pass = ctx.uniform(*self.pass_range)
            f_stop = self.stop_hz if self.stop_hz is not None else sr / 2.0
            y = _rolloff_lowpass(y, f_pass, f_stop, self.stop_db, sr)
        ctx.log.append(
            f"{self.name}: low {low if low else 0:.0f} Hz, pass {f_pass if f_pass else sr/2:.0f} Hz, "
            f"-{self.stop_db:.0f} dB at {self.stop_hz if self.stop_hz else sr/2:.0f} Hz @ {sr}"
        )
        return y

"""Band-limiting stage."""

from __future__ import annotations

import torch

from phonesim.core import SimContext, Stage, resolve_range
from phonesim import dsp


class BandlimitStage(Stage):
    """Band-pass the signal to a telephone-style passband.

    Defaults to the narrowband telephone channel (300-3400 Hz); ``low_hz=50,
    high_hz=7000`` gives a wideband one. The cutoffs may be ranges ``(lo, hi)``
    to randomise the band edges slightly per call.

    Notes
    -----
    Real telephony band-limiting is the *combined* effect of anti-alias filters,
    codec bandwidth, and handset transducers. This single FIR band-pass is a
    deliberate, controllable proxy for that aggregate response. It does not model
    the gentle in-band ripple of, e.g., the G.712 mask, but it reproduces the
    dominant effect: energy outside the passband is removed.
    """

    def __init__(
        self,
        low_hz=300.0,
        high_hz=3400.0,
        numtaps: int = 257,
        jitter_hz: float = 0.0,
        name=None,
    ):
        super().__init__(name=name or "Bandlimit")
        self.low_range = resolve_range(low_hz) if low_hz is not None else None
        self.high_range = resolve_range(high_hz) if high_hz is not None else None
        self.numtaps = numtaps
        self.jitter_hz = jitter_hz

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        sr = ctx.sample_rate
        low = None
        high = None
        if self.low_range is not None:
            low = ctx.uniform(*self.low_range)
            if self.jitter_hz:
                low += ctx.uniform(-self.jitter_hz, self.jitter_hz)
            low = max(0.0, low)
        if self.high_range is not None:
            high = ctx.uniform(*self.high_range)
            if self.jitter_hz:
                high += ctx.uniform(-self.jitter_hz, self.jitter_hz)
            high = min(high, 0.999 * sr / 2.0)
        y = dsp.bandpass(x, low, high, sr, numtaps=self.numtaps)
        ctx.log.append(
            f"{self.name}: passband "
            f"[{low if low else 0:.0f}, {high if high else sr/2:.0f}] Hz @ {sr}"
        )
        return y

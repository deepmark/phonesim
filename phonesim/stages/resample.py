"""Resampling stage."""

from __future__ import annotations

from typing import Optional

import torch

from phonesim.core import SimContext, Stage, warn_at_caller
from phonesim import dsp


def _whole_hz(value):
    """``value`` as an int when it is a whole number of Hz (16000, "16000",
    16000.0, a numpy integer), else unchanged, so that any other value reaches
    the mismatch warning rather than failing here."""
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return value
    return int(f) if f.is_integer() else value


class ResampleStage(Stage):
    """Resample the signal to a fixed target rate.

    Parameters
    ----------
    from_sr:
        Rate the signal should arrive at: the pipeline's input rate, or the
        ``to_sr`` of the resample before this one. The conversion always starts
        from the rate the signal arrives at (``ctx.sample_rate``), so chained
        resamplers stay consistent; when that rate differs from ``from_sr`` the
        stage warns (``FutureWarning``), and from 0.3.0 it raises
        ``ValueError``. ``None`` accepts any rate.
    to_sr:
        Output sample rate. After this stage ``ctx.sample_rate == to_sr``.
    zeros:
        Half-width of the sinc kernel in zero crossings, at any pair of rates;
        its cutoff is the lower of the input and output Nyquist frequencies.
        Higher is sharper/slower. 32 is a good default.
    """

    def __init__(self, from_sr: Optional[int], to_sr: int, zeros: int = 32, name=None):
        super().__init__(name=name or f"Resample->{to_sr}")
        self.from_sr = None if from_sr is None else _whole_hz(from_sr)
        self.to_sr = int(to_sr)
        self.zeros = zeros

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        cur = ctx.sample_rate
        if self.from_sr is not None and cur != self.from_sr:
            warn_at_caller(
                f"{self.name}: from_sr {self.from_sr} Hz disagrees with the {cur} Hz signal it receives "
                f"(the pipeline's input rate or an earlier stage's to_sr) and takes it as {cur} Hz. "
                "From 0.3.0 this raises ValueError"
            )
        if cur == self.to_sr:
            return x
        y = dsp.resample(x, cur, self.to_sr, zeros=self.zeros)
        ctx.log.append(f"{self.name}: {cur} -> {self.to_sr} Hz")
        ctx.sample_rate = self.to_sr
        return y

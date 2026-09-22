"""Resampling stage."""

from __future__ import annotations

from typing import Optional

import torch

from phonesim.core import SimContext, Stage
from phonesim import dsp


class ResampleStage(Stage):
    """Resample the signal to a fixed target rate.

    Parameters
    ----------
    from_sr:
        Expected input rate. If ``None``, the stage trusts ``ctx.sample_rate``.
        When provided it is only used as a sanity check / documentation; the
        actual conversion always uses ``ctx.sample_rate`` so that chained
        resamplers stay consistent.
    to_sr:
        Output sample rate. After this stage ``ctx.sample_rate == to_sr``.
    zeros:
        Half-width of the sinc kernel in output-Nyquist zero crossings; higher
        is sharper/slower. 32 is a good default.
    """

    def __init__(self, from_sr: Optional[int], to_sr: int, zeros: int = 32, name=None):
        super().__init__(name=name or f"Resample->{to_sr}")
        self.from_sr = from_sr
        self.to_sr = int(to_sr)
        self.zeros = zeros

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        cur = ctx.sample_rate
        if cur == self.to_sr:
            return x
        y = dsp.resample(x, cur, self.to_sr, zeros=self.zeros)
        ctx.log.append(f"{self.name}: {cur} -> {self.to_sr} Hz")
        ctx.sample_rate = self.to_sr
        return y

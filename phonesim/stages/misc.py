"""Miscellaneous transport effects: time offset and speed/clock drift."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from phonesim.core import SimContext, Stage, resolve_range


class TimeOffsetStage(Stage):
    """Prepend/trim leading samples to model a recording start offset.

    A recorded call rarely starts exactly at sample 0 relative to the clean
    reference. This shifts the signal by a random amount (positive = insert
    leading silence, negative = drop leading samples). The output length is
    preserved by trimming/padding at the end.
    """

    def __init__(self, max_ms=30.0, name=None):
        super().__init__(name=name or "TimeOffset")
        self.offset_range = resolve_range(max_ms)

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        sr = ctx.sample_rate
        # symmetric range around 0 unless a single value was given
        lo, hi = self.offset_range
        if lo == hi:
            ms = ctx.uniform(-hi, hi)
        else:
            ms = ctx.uniform(lo, hi)
        shift = int(round(ms * sr / 1000.0))
        b, c, t = x.shape
        if shift > 0:
            y = F.pad(x, (shift, 0))[..., :t]
        elif shift < 0:
            y = F.pad(x, (0, -shift))[..., -t:]
        else:
            y = x
        ctx.log.append(f"{self.name}: {ms:+.1f} ms ({shift:+d} samples)")
        return y


class SpeedDriftStage(Stage):
    """Resample by a tiny factor to model clock drift / SR mismatch.

    Sender and receiver clocks are never perfectly matched, so recorded audio is
    very slightly faster or slower than the original (typically << 1%). This
    stage resamples by ``1 + drift`` and then pads/trims to the original length,
    introducing a small, accumulating time misalignment (linear interpolation).
    """

    def __init__(self, max_drift=0.003, name=None):
        super().__init__(name=name or "SpeedDrift")
        self.drift_range = resolve_range(max_drift)

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        lo, hi = self.drift_range
        if lo == hi:
            drift = ctx.uniform(-hi, hi)
        else:
            drift = ctx.uniform(lo, hi)
        if abs(drift) < 1e-6:
            return x
        b, c, t = x.shape
        # Linear interpolation; ClockDriftStage is the windowed-sinc equivalent in ppm.
        new_len = max(2, int(round(t * (1.0 + drift))))
        y = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
        # Re-interpret at the original rate: trim/pad back to the original length.
        if y.shape[-1] >= t:
            y = y[..., :t]
        else:
            y = F.pad(y, (0, t - y.shape[-1]))
        ctx.log.append(f"{self.name}: drift {drift*100:+.3f}%")
        return y

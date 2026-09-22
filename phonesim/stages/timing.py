"""Playout timing: adaptive jitter buffer and clock drift."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from phonesim.core import SimContext, Stage, resolve_range
from phonesim.stages.packet import _conceal, _frame, _unframe

# Output samples interpolated per iteration in ClockDriftStage; bounds the
# [B*C, _BLOCK, 32] gather that would otherwise scale with the clip length.
_BLOCK = 65536


def _ola_stretch(frame: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """One frame of expansion: cross-fade the preceding frame into the current."""
    n = frame.shape[-1]
    w = torch.linspace(0.0, 1.0, n, device=frame.device, dtype=frame.dtype)
    return prev * (1.0 - w) + frame * w


class PlayoutBufferStage(Stage):
    """Adaptive jitter buffer at the receiver.

    Late packets arrive in bursts (a two-state chain, like loss) and are
    concealed. Buffer under-runs and over-runs are per-frame events at
    ``adapt_rate``: an under-run expands time by one frame (overlap-added,
    not spliced), an over-run drops one frame. Expansions and drops are
    balanced over the call, so the output stays aligned with the input on
    average, and every event is logged. Length preserving.
    """

    def __init__(self, frame_ms=20.0, late_rate=0.01, burst_probability=0.3, adapt_rate=0.002, name=None):
        super().__init__(name=name or "PlayoutBuffer")
        self.frame_ms = frame_ms
        self.late_range = resolve_range(late_rate)
        self.burst_probability = burst_probability
        self.adapt_range = resolve_range(adapt_rate)

    def _late_mask(self, nframes: int, rate: float, ctx: SimContext, device) -> torch.Tensor:
        stay = min(max(self.burst_probability, 0.0), 0.95)
        p_enter = min(rate * (1.0 - stay) / max(1e-6, 1.0 - rate), 1.0)
        mask = torch.ones(nframes, device=device)
        bad = False
        for i in range(nframes):
            u = torch.rand((), generator=ctx.generator).item()
            if bad:
                mask[i] = 0.0
                bad = u >= (1.0 - stay)
            elif u < p_enter:
                bad = True
        return mask

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        sr = ctx.sample_rate
        fl = max(1, int(sr * self.frame_ms / 1000.0))
        frames, t = _frame(x, fl)
        b, c, nframes, _ = frames.shape

        late_rate = ctx.uniform(*self.late_range)
        mask = self._late_mask(nframes, late_rate, ctx, x.device) if late_rate > 0 else torch.ones(nframes, device=x.device)
        out = _conceal(frames, mask) if late_rate > 0 else frames

        adapt = ctx.uniform(*self.adapt_range)
        events = []
        if adapt > 0:
            u = ctx.rand((nframes,), device=x.device)
            idx = torch.nonzero(u < adapt).flatten().tolist()
            pieces, i0, expanded = [], 0, 0
            for k, i in enumerate(idx):
                pieces.append(out[:, :, i0:i])
                if k % 2 == 0:                                   # under-run: expand
                    prev = out[:, :, i - 1] if i > 0 else torch.zeros_like(out[:, :, 0])
                    pieces.append(_ola_stretch(out[:, :, i], prev).unsqueeze(2))
                    pieces.append(out[:, :, i:i + 1])
                    expanded += 1
                    events.append(f"+{i}")
                else:                                            # over-run: drop
                    events.append(f"-{i}")
                i0 = i + 1
            pieces.append(out[:, :, i0:])
            out = torch.cat(pieces, dim=2)
            if out.shape[2] > nframes:
                out = out[:, :, :nframes]
            elif out.shape[2] < nframes:
                out = F.pad(out, (0, 0, 0, nframes - out.shape[2]))

        ctx.log.append(
            f"{self.name}: late_rate={late_rate:.3f}, concealed {int((mask == 0).sum().item())} frames, "
            f"adapt events {' '.join(events) if events else 'none'}"
        )
        return _unframe(out, t)


class ClockDriftStage(Stage):
    """Sender/receiver clock mismatch, in parts per million.

    Free-running telecom clocks differ by tens of ppm. The drift is realised
    with a windowed-sinc fractional resampler (flat to 0.9 Nyquist), then the
    output is padded or trimmed back to the input length. Positive drift
    means the receiver plays slower (the signal ends later).
    """

    def __init__(self, ppm=(-50.0, 50.0), name=None):
        super().__init__(name=name or "ClockDrift")
        self.ppm_range = resolve_range(ppm)

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        ppm = ctx.uniform(*self.ppm_range)
        if abs(ppm) < 0.5:
            ctx.log.append(f"{self.name}: {ppm:+.0f} ppm")
            return x
        b, c, t = x.shape
        ratio = 1.0 + ppm * 1e-6
        new_len = int(round(t * ratio))
        half = 16
        taps = torch.arange(-half + 1, half + 1, device=x.device, dtype=x.dtype)   # [32]
        xf = x.reshape(b * c, t)
        y = torch.empty(b * c, new_len, device=x.device, dtype=x.dtype)
        # Fractional delay by windowed-sinc interpolation at the new grid, one
        # block of output samples at a time; each sample sees the same kernel
        # and neighbours whatever the block boundaries.
        for s in range(0, new_len, _BLOCK):
            e = min(s + _BLOCK, new_len)
            pos = torch.arange(s, e, device=x.device, dtype=x.dtype) / ratio
            base = torch.floor(pos).long()
            frac = (pos - base.to(x.dtype))
            arg = taps.unsqueeze(0) - frac.unsqueeze(1)                              # [block, 32]
            sinc = torch.where(arg == 0, torch.ones_like(arg), torch.sin(math.pi * arg) / (math.pi * arg))
            win = 0.5 * (1.0 + torch.cos(math.pi * arg / half))
            k = sinc * win * (arg.abs() < half).to(x.dtype)
            k = k / k.sum(1, keepdim=True)
            idx = (base.unsqueeze(1) + taps.long().unsqueeze(0)).clamp(0, t - 1)      # [block, 32]
            y[:, s:e] = (xf[:, idx] * k.unsqueeze(0)).sum(-1)
        y = y[:, :t] if new_len >= t else F.pad(y, (0, t - new_len))
        ctx.log.append(f"{self.name}: {ppm:+.0f} ppm")
        return y.reshape(b, c, t)

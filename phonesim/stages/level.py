"""Send-side level control: active speech level and a peak limiter."""

from __future__ import annotations

import torch

from phonesim.core import SimContext, Stage, resolve_range
from phonesim import dsp


def active_speech_level_db(x: torch.Tensor, sr: int, margin_db: float = 15.9) -> torch.Tensor:
    """ITU-T P.56-style active speech level in dBFS, per ``[B, C]``.

    Frames within ``margin_db`` of the loudest 20 ms frame count as active;
    the level is the RMS over active frames only, so leading/trailing silence
    and pauses do not lower it.
    """
    b, c, t = x.shape
    fl = max(1, int(sr * 0.02))
    n = t // fl
    if n == 0:
        return dsp.dbfs(x)
    frames = x[..., : n * fl].reshape(b, c, n, fl)
    p = frames.pow(2).mean(-1)                                    # [B, C, n]
    p_db = 10.0 * torch.log10(p + 1e-12)
    active = p_db >= (p_db.amax(-1, keepdim=True) - margin_db)
    e = (p * active).sum(-1) / active.sum(-1).clamp(min=1)
    return 10.0 * torch.log10(e + 1e-12)


class SpeechLevelStage(Stage):
    """Set the active speech level, as a handset or platform does before encoding.

    ``target_dbov`` is the P.56 active level in dB below full scale, drawn per
    call from a range. Nominal telephony levels are around -26 dBov; the range
    is a parameter because real calls vary by 10 dB or more. Gain is capped by
    ``max_gain_db`` so silence is not amplified into noise.
    """

    def __init__(self, target_dbov=(-32.0, -22.0), max_gain_db: float = 30.0, name=None):
        super().__init__(name=name or "SpeechLevel")
        self.target_range = resolve_range(target_dbov)
        self.max_gain_db = max_gain_db

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        target = ctx.uniform(*self.target_range)
        cur = active_speech_level_db(x, ctx.sample_rate)         # [B, C]
        gain_db = (target - cur).clamp(max=self.max_gain_db)
        ctx.log.append(
            f"{self.name}: target {target:.1f} dBov, mean gain {gain_db.mean().item():+.1f} dB"
        )
        return dsp.apply_gain_db(x, gain_db)


class LimiterStage(Stage):
    """Peak limiter: identity below the threshold, soft knee above it.

    Every AGC, codec input and network sink limits peaks; a codec fed above
    full scale clips. ``ceiling_dbfs`` is the output peak.
    """

    def __init__(self, ceiling_dbfs: float = -1.0, knee_db: float = 6.0, name=None):
        super().__init__(name=name or "Limiter")
        self.ceiling = 10.0 ** (ceiling_dbfs / 20.0)
        self.knee = 10.0 ** (-knee_db / 20.0)

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        c = self.ceiling
        k = c * self.knee                                          # knee start
        a = x.abs()
        # Above the knee, compress the excess with a tanh that approaches c.
        over = torch.tanh((a - k) / (c - k)) * (c - k) + k
        y = torch.sign(x) * torch.where(a > k, over, a)
        n_over = int((a > c).sum().item())
        if n_over:
            ctx.log.append(f"{self.name}: {n_over} samples above {c:.2f} limited")
        return y

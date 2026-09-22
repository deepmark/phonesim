"""Gain, automatic-gain-control, and clipping stages."""

from __future__ import annotations

import torch

from phonesim.core import SimContext, Stage, resolve_range
from phonesim import dsp


class GainStage(Stage):
    """Apply a (possibly random) fixed gain in dB."""

    def __init__(self, gain_db=0.0, name=None):
        super().__init__(name=name or "Gain")
        self.gain_range = resolve_range(gain_db)

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        g = ctx.uniform(*self.gain_range)
        ctx.log.append(f"{self.name}: {g:+.1f} dB")
        return dsp.apply_gain_db(x, g)


class AGCStage(Stage):
    """Automatic gain control toward a target RMS level.

    Two modes:

    * ``static`` (default): compute one gain per example to bring the whole clip
      to ``target_dbfs`` (optionally with a max-gain limit). Vectorised and
      cheap; captures the dominant effect of telephony level normalisation.
    * ``dynamic``: a time-varying envelope follower with attack/release, which
      better approximates WebRTC AGC pumping but is more aggressive.

    ``dynamic`` runs a per-sample Python recurrence and is slow on long input;
    all built-in profiles use ``static``.
    """

    def __init__(
        self,
        target_dbfs=-20.0,
        max_gain_db: float = 30.0,
        mode: str = "static",
        attack_ms: float = 10.0,
        release_ms: float = 150.0,
        name=None,
    ):
        super().__init__(name=name or "AGC")
        if mode not in ("static", "dynamic"):
            raise ValueError(f"Unknown AGC mode {mode!r}; expected 'static' or 'dynamic'")
        self.target_range = resolve_range(target_dbfs)
        self.max_gain_db = max_gain_db
        self.mode = mode
        self.attack_ms = attack_ms
        self.release_ms = release_ms

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        target = ctx.uniform(*self.target_range)
        if self.mode == "static":
            cur = dsp.dbfs(x)  # [B, C]
            gain_db = (target - cur).clamp(max=self.max_gain_db)
            ctx.log.append(
                f"{self.name}(static): target {target:.1f} dBFS, "
                f"mean gain {gain_db.mean().item():+.1f} dB"
            )
            return dsp.apply_gain_db(x, gain_db)

        # dynamic envelope follower
        sr = ctx.sample_rate
        b, c, t = x.shape
        atk = float(torch.exp(torch.tensor(-1.0 / (self.attack_ms * 1e-3 * sr))))
        rel = float(torch.exp(torch.tensor(-1.0 / (self.release_ms * 1e-3 * sr))))
        target_lin = 10 ** (target / 20.0)
        env = torch.zeros(b * c, device=x.device, dtype=x.dtype)
        xr = x.reshape(b * c, t).abs()
        gains = torch.zeros_like(xr)
        e = env
        for n in range(t):
            xn = xr[:, n]
            coeff = torch.where(xn > e, torch.tensor(atk, device=x.device), torch.tensor(rel, device=x.device))
            e = coeff * e + (1 - coeff) * xn
            gains[:, n] = target_lin / (e + 1e-4)
        gains = gains.clamp(max=10 ** (self.max_gain_db / 20.0))
        y = (x.reshape(b * c, t) * gains).reshape(b, c, t)
        ctx.log.append(f"{self.name}(dynamic): target {target:.1f} dBFS")
        return y


class ClipStage(Stage):
    """Clip / soft-clip the waveform.

    ``mode="hard"`` clamps to ``+/- threshold``; ``mode="soft"`` uses ``tanh``
    saturation. ``drive_db`` boosts the
    signal into the nonlinearity before clipping, then compensates afterwards so
    the average level is roughly preserved while adding harmonic distortion.
    """

    def __init__(self, threshold=0.99, drive_db=0.0, mode="soft", name=None):
        super().__init__(name=name or "Clip")
        self.threshold_range = resolve_range(threshold)
        self.drive_range = resolve_range(drive_db)
        self.mode = mode

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        thr = ctx.uniform(*self.threshold_range)
        drive = ctx.uniform(*self.drive_range)
        xd = dsp.apply_gain_db(x, drive)
        if self.mode == "hard":
            y = xd.clamp(-thr, thr)
        else:
            y = thr * torch.tanh(xd / max(thr, 1e-6))
        y = dsp.apply_gain_db(y, -drive)
        ctx.log.append(f"{self.name}({self.mode}): thr={thr:.3f}, drive={drive:+.1f} dB")
        return y

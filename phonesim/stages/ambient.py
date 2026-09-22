"""Ambient noise entering at the microphone, before encoding."""

from __future__ import annotations

import torch

from phonesim.core import SimContext, Stage, resolve_range
from phonesim.stages.level import active_speech_level_db


class AmbientNoiseStage(Stage):
    """Add room/handset ambient noise ahead of the first encoder.

    ``snr_db`` is the ratio of active speech level to noise level, drawn per
    call. The noise has a speech-room spectrum: flat to ``knee_hz`` then
    -6 dB/octave, and nothing below ``low_hz`` (a room has no infrasound the
    channel could pass). Downstream band-limiting, coding and level control
    then shape it the way they shape speech.
    """

    def __init__(self, snr_db=(40.0, 55.0), knee_hz: float = 200.0, low_hz: float = 60.0, name=None):
        super().__init__(name=name or "Ambient")
        self.snr_range = resolve_range(snr_db)
        self.knee_hz = knee_hz
        self.low_hz = low_hz

    def _noise(self, shape, sr, ctx: SimContext, device, dtype) -> torch.Tensor:
        n = ctx.randn(shape, device=device, dtype=dtype)
        N = shape[-1]
        spec = torch.fft.rfft(n, dim=-1)
        f = torch.fft.rfftfreq(N, d=1.0 / sr).to(device)
        shape_db = -20.0 * torch.log10(torch.clamp(f / self.knee_hz, min=1.0))     # -6 dB/oct above knee
        gain = 10.0 ** (shape_db / 20.0) * (f >= self.low_hz).to(dtype)
        return torch.fft.irfft(spec * gain, n=N, dim=-1)

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        snr = ctx.uniform(*self.snr_range)
        sr = ctx.sample_rate
        noise = self._noise(x.shape, sr, ctx, x.device, x.dtype)
        asl = active_speech_level_db(x, sr)                                          # [B, C] dB
        noise_db = 20.0 * torch.log10(noise.pow(2).mean(-1).sqrt() + 1e-12)          # [B, C]
        gain_db = (asl - snr) - noise_db
        y = x + noise * (10.0 ** (gain_db / 20.0)).unsqueeze(-1)
        ctx.log.append(f"{self.name}: SNR {snr:.1f} dB re active speech")
        return y

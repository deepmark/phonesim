"""Additive-noise stage."""

from __future__ import annotations

import torch

from phonesim.core import SimContext, Stage, resolve_range
from phonesim import dsp


class NoiseStage(Stage):
    """Add background noise at a target SNR.

    ``snr_db`` may be a scalar or a ``(low, high)`` range. The noise can be white
    or shaped:

    * ``"white"``    - flat spectrum.
    * ``"pink"``     - 1/f, approximates room/ambient rumble.
    * ``"hum"``      - low-frequency mains-style hum plus white floor.

    SNR is computed per example from the signal RMS.
    """

    def __init__(self, snr_db=(25.0, 40.0), color="white", name=None):
        super().__init__(name=name or "Noise")
        self.snr_range = resolve_range(snr_db)
        self.color = color

    def _make_noise(self, shape, sr, ctx: SimContext, device, dtype) -> torch.Tensor:
        n = ctx.randn(shape, device=device, dtype=dtype)
        if self.color == "white":
            return n
        if self.color == "pink":
            # Shape white noise by 1/sqrt(f) in the frequency domain.
            N = n.shape[-1]
            spec = torch.fft.rfft(n, dim=-1)
            freqs = torch.fft.rfftfreq(N, d=1.0 / sr).to(device)
            scale = 1.0 / torch.sqrt(torch.clamp(freqs, min=1.0))
            spec = spec * scale
            out = torch.fft.irfft(spec, n=N, dim=-1)
            return out
        if self.color == "hum":
            t = torch.arange(shape[-1], device=device, dtype=dtype) / sr
            hum = torch.sin(2 * torch.pi * 50.0 * t) + 0.5 * torch.sin(2 * torch.pi * 100.0 * t)
            hum = hum.view(*([1] * (n.dim() - 1)), -1)
            return 0.7 * hum + 0.3 * n
        raise ValueError(f"Unknown noise color {self.color!r}")

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        snr = ctx.uniform(*self.snr_range)
        sr = ctx.sample_rate
        noise = self._make_noise(x.shape, sr, ctx, x.device, x.dtype)
        # Scale noise so that per-example SNR matches the target.
        sig_rms = dsp.rms(x)  # [B, C]
        noise_rms = dsp.rms(noise)
        target_noise_rms = sig_rms / (10 ** (snr / 20.0))
        scale = (target_noise_rms / (noise_rms + 1e-12)).unsqueeze(-1)
        y = x + noise * scale
        ctx.log.append(f"{self.name}({self.color}): SNR {snr:.1f} dB")
        return y

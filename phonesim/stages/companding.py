"""ITU-T G.711 mu-law / A-law: the exact 8-segment coder, table-driven."""

from __future__ import annotations

import functools

import numpy as np
import torch

from phonesim.core import SimContext, Stage


# ----------------------------------------------------------------------------
# ITU-T G.711 segmented coder (Sun g711.c algorithm)
# ----------------------------------------------------------------------------
_BIAS = 0x84
_CLIP = 8159


def _ulaw_encode(pcm: np.ndarray) -> np.ndarray:
    x = pcm.astype(np.int32) >> 2
    sign = np.where(x < 0, 0x80, 0)
    x = np.abs(x)
    x = np.minimum(x, _CLIP) + (_BIAS >> 2)
    seg = np.zeros_like(x)
    for lvl in (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF):
        seg += x > lvl
    code = np.where(seg >= 8, 0x7F, (seg << 4) | ((x >> (seg + 1)) & 0x0F))
    return (code ^ 0xFF ^ sign).astype(np.uint8)


def _ulaw_decode(code: np.ndarray) -> np.ndarray:
    u = (~code.astype(np.int32)) & 0xFF
    t = ((u & 0x0F) << 3) + _BIAS
    t <<= (u & 0x70) >> 4
    return np.where(u & 0x80, _BIAS - t, t - _BIAS).astype(np.int16)


def _alaw_encode(pcm: np.ndarray) -> np.ndarray:
    x = pcm.astype(np.int32) >> 3
    mask = np.where(x >= 0, 0xD5, 0x55)
    x = np.where(x >= 0, x, -x - 1)
    seg = np.zeros_like(x)
    for lvl in (0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF):
        seg += x > lvl
    low = np.where(seg < 2, (x >> 1) & 0x0F, (x >> seg) & 0x0F)
    code = np.where(seg >= 8, 0x7F, (seg << 4) | low) ^ mask
    return code.astype(np.uint8)


def _alaw_decode(code: np.ndarray) -> np.ndarray:
    a = code.astype(np.int32) ^ 0x55
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    t = np.where(seg == 0, t + 8, np.where(seg == 1, t + 0x108, (t + 0x108) << (seg - 1)))
    return np.where(a & 0x80, t, -t).astype(np.int16)


@functools.lru_cache(maxsize=2)
def _g711_lut(law: str) -> torch.Tensor:
    """Decoded value for every int16 input, as float in [-1, 1]."""
    ramp = np.arange(-32768, 32768, dtype=np.int16)
    if law == "mulaw":
        out = _ulaw_decode(_ulaw_encode(ramp))
    else:
        out = _alaw_decode(_alaw_encode(ramp))
    return torch.from_numpy(out.astype(np.float32) / 32768.0)


def g711_roundtrip(x: torch.Tensor, law: str) -> torch.Tensor:
    """Exact G.711 encode-decode of ``x`` in [-1, 1]."""
    lut = _g711_lut(law).to(x.device)
    idx = torch.clamp(torch.round(x * 32768.0), -32768, 32767).long() + 32768
    return lut[idx]


# Accepted spellings of the law -> the table it selects.
_LAWS = {"mulaw": "mulaw", "ulaw": "mulaw", "alaw": "alaw"}


class CompandingStage(Stage):
    """G.711 companding + 8-bit quantisation; ``law`` is ``"mulaw"`` (or ``"ulaw"``) or ``"alaw"``."""

    def __init__(self, law="mulaw", name=None):
        if law not in _LAWS:
            raise ValueError(f"law must be one of {sorted(_LAWS)}, got {law!r}")
        law = _LAWS[law]
        super().__init__(name=name or f"Compand-{law}")
        self.law = law

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        ctx.log.append(f"{self.name}: G.711")
        return g711_roundtrip(x, self.law)

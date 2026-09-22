"""Native G.711 tables against ffmpeg's pcm_mulaw and pcm_alaw.

Both decoders map every 8-bit code to the same int16 value. The encoders
agree on every int16 input except where ffmpeg's tabulated coder rounds a
sample on a decision level to the neighbouring code; those inputs are counted
exactly and each must be coded to an adjacent output level.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonesim import ffmpeg_backend as fb
from phonesim.stages import companding as C

pytestmark = pytest.mark.skipif(not fb.have_ffmpeg(), reason="needs ffmpeg")

# law -> (native encoder, native decoder, int16 inputs ffmpeg codes differently)
_LAWS = {"mulaw": (C._ulaw_encode, C._ulaw_decode, 512), "alaw": (C._alaw_encode, C._alaw_decode, 964)}
RAMP = np.arange(-32768, 32768, dtype="<i2")
CODES = np.arange(256, dtype=np.uint8)


def _ffmpeg(tmp_path, data: bytes, in_fmt: str, out_fmt: str) -> bytes:
    """Convert raw 8 kHz mono ``in_fmt`` samples to raw ``out_fmt`` samples."""
    src = tmp_path / f"in.{in_fmt}"
    src.write_bytes(data)
    return fb._run([fb.FFMPEG, "-hide_banner", "-loglevel", "error", "-f", in_fmt, "-ar", "8000", "-ac", "1",
                    "-i", str(src), "-f", out_fmt, "pipe:1"])


def _level_rank(decode) -> np.ndarray:
    """Position of each code's decoded value among the distinct output levels."""
    return np.unique(decode(CODES), return_inverse=True)[1]


@pytest.mark.parametrize("law", sorted(_LAWS))
def test_decoders_agree_on_every_code(tmp_path, law):
    _, decode, _ = _LAWS[law]
    ff = np.frombuffer(_ffmpeg(tmp_path, CODES.tobytes(), law, "s16le"), dtype="<i2")
    assert np.array_equal(ff, decode(CODES))


@pytest.mark.parametrize("law", sorted(_LAWS))
def test_encoders_differ_only_on_adjacent_levels(tmp_path, law):
    encode, decode, n_differ = _LAWS[law]
    ff = np.frombuffer(_ffmpeg(tmp_path, RAMP.tobytes(), "s16le", law), dtype=np.uint8)
    native = encode(RAMP)
    assert ff.shape == native.shape
    differ = ff != native
    assert int(differ.sum()) == n_differ
    rank = _level_rank(decode)
    assert np.all(np.abs(rank[ff[differ]] - rank[native[differ]]) == 1)

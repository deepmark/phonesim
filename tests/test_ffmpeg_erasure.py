"""Coded-domain frame erasures for AMR-NB and AMR-WB through ffmpeg.

Parsing and rewriting of the AMR storage format, the decoder's concealment
of erased frames, the decay of the error it carries into the following good
frames, and the mask rules for the other codecs. Tests that need a codec this
ffmpeg lacks skip.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from phonesim import ffmpeg_backend as fb


def _needs(codec):
    return pytest.mark.skipif(codec not in fb.available_codecs(), reason=f"needs an ffmpeg with {codec}")


_AMR = [
    pytest.param("amr_nb", "12.2k", marks=_needs("amr_nb")),
    pytest.param("amr_wb", "12.65k", marks=_needs("amr_wb")),
]
NB, WB = b"#!AMR\n", b"#!AMR-WB\n"


def _speech(sr, dur=2.0, peak=0.3):
    """A 140 Hz harmonic train (4 harmonics) under a slow 1.3 Hz amplitude modulation."""
    t = np.arange(int(sr * dur)) / sr
    x = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 5))
    x *= 0.5 * (1 + 0.8 * np.sin(2 * np.pi * 1.3 * t))
    return (peak * x / np.abs(x).max()).astype(np.float32)


def _encode(x, codec, bitrate, path):
    """Encode ``x`` at the codec's native rate into the AMR file ``path``; return its bytes."""
    sr = fb.native_sr(codec)
    wav = path.with_suffix(".wav")
    sf.write(wav, x, sr, subtype="PCM_16")
    fb._run([fb.FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav),
             "-ar", str(sr), "-ac", "1", "-c:a", fb.encoder_name(codec), "-b:a", bitrate, str(path)])
    return path.read_bytes()


def _frame(y, k, fl):
    return y[k * fl:(k + 1) * fl]


def _db(v):
    """RMS of ``v`` in dBFS (-240 for an all-zero frame)."""
    return 20 * np.log10(np.sqrt(np.mean(np.square(v, dtype=np.float64))) + 1e-12)


def _pair(codec, bitrate, erased):
    """Decode a 2 s signal with and without ``erased``; return (clean, hit, frame_len)."""
    sr = fb.native_sr(codec)
    x = _speech(sr)
    clean, _ = fb.encode_decode(x, sr, codec, bitrate=bitrate)
    hit, _ = fb.encode_decode(x, sr, codec, bitrate=bitrate, erased=erased)
    return clean, hit, sr * fb.AMR_FRAME_MS // 1000


def test_amr_frames_rejects_malformed_input():
    with pytest.raises(ValueError, match="not an AMR codec"):
        fb.amr_frames(NB + b"\x7c", "g722")
    with pytest.raises(ValueError, match="bad magic"):
        fb.amr_frames(WB + b"\x7c", "amr_nb")
    with pytest.raises(ValueError, match="unknown frame type"):
        fb.amr_frames(NB + bytes([9 << 3]) + bytes(40), "amr_nb")      # FT 9 is AMR-WB only
    with pytest.raises(ValueError, match="truncated"):
        fb.amr_frames(NB + b"\x3c" + bytes(30), "amr_nb")             # FT 7 needs 31 payload bytes
    # a mask longer than the file is truncated (the encoder pads the last frame)
    assert fb.erase_amr_frames(NB + b"\x7c\x7c", "amr_nb", [False, False, True]) == NB + b"\x7c\x7c"


@_needs("amr_nb")
def test_amr_frames_parse_and_erase(tmp_path):
    data = _encode(_speech(8000, 1.0), "amr_nb", "12.2k", tmp_path / "enc.amr")
    magic, frames = fb.amr_frames(data, "amr_nb")
    assert magic == NB and len(frames) in (50, 51)
    assert all((f[0] >> 3) & 0xF == 7 and len(f) == 32 for f in frames)
    erased = np.zeros(9, dtype=bool)                                   # shorter than the file: padded
    erased[[3, 7, 8]] = True
    out = fb.erase_amr_frames(data, "amr_nb", erased)
    magic2, frames2 = fb.amr_frames(out, "amr_nb")
    assert magic2 == magic and len(frames2) == len(frames)
    for k, (a, b) in enumerate(zip(frames, frames2)):
        assert b == (b"\x7c" if k in (3, 7, 8) else a)


@pytest.mark.parametrize("codec, bitrate", _AMR)
def test_isolated_erasures_are_concealed(codec, bitrate):
    erased = np.zeros(100, dtype=bool)
    erased[[20, 21, 30]] = True
    clean, hit, fl = _pair(codec, bitrate, erased)
    assert len(hit) == len(clean)
    # The decoder state is identical up to the first erasure.
    assert np.array_equal(_frame(hit, 10, fl), _frame(clean, 10, fl))
    floor = _db(_frame(hit, 10, fl) - _frame(clean, 10, fl))
    for k in (20, 21, 30):
        err = _db(_frame(hit, k, fl) - _frame(clean, k, fl))
        assert err > floor + 6
        assert err > _db(_frame(clean, k, fl)) - 20                    # concealment is not a near-copy
    # the first erased frame is concealed, not muted: within 12 dB of the clean level
    assert _db(_frame(hit, 20, fl)) > -45 and abs(_db(_frame(hit, 20, fl)) - _db(_frame(clean, 20, fl))) < 12


@pytest.mark.parametrize("codec, bitrate", _AMR)
def test_burst_error_decays_after_the_burst(codec, bitrate):
    erased = np.zeros(100, dtype=bool)
    erased[40:46] = True
    clean, hit, fl = _pair(codec, bitrate, erased)
    err = {k: _db(_frame(hit, k, fl) - _frame(clean, k, fl)) for k in (46, 51)}
    assert err[46] > err[51]


@_needs("g722")
def test_erasure_mask_rules_for_other_codecs():
    sr = 16000
    x = _speech(sr)
    clean, _ = fb.encode_decode(x, sr, "g722")
    with pytest.raises(ValueError, match="only available for amr_nb and amr_wb"):
        fb.encode_decode(x, sr, "g722", erased=np.array([False, True]))
    same, _ = fb.encode_decode(x, sr, "g722", erased=np.zeros(100, dtype=bool))
    assert np.array_equal(same, clean)
    short, _ = fb.encode_decode(x, sr, "g722", erased=np.zeros(97, dtype=bool))
    assert np.array_equal(short, clean)

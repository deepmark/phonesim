"""Tests for the G.711 Appendix I packet loss concealment.

A harmonic tone with one or more 20 ms frames erased; the erased input
samples are zeroed first so a pass proves the concealer never reads them.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonesim.plc import conceal


def _tone(sr: int, seconds: float = 2.0, f0: float = 150.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    x = sum(a * np.sin(2 * np.pi * f0 * (k + 1) * t) for k, a in enumerate((0.2, 0.1, 0.05)))
    return x.astype(np.float32)


def _rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _erase(x: np.ndarray, frame_len: int, frames) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mask, x_with_erased_frames_zeroed)``."""
    mask = np.zeros(-(-len(x) // frame_len), dtype=bool)
    mask[list(frames)] = True
    xin = x.copy()
    for k in frames:
        xin[k * frame_len:(k + 1) * frame_len] = 0
    return mask, xin


def test_no_erasures_is_identity():
    x = _tone(8000)
    x0 = x.copy()
    out = conceal(x, 8000, np.zeros(100, dtype=bool), 160)
    assert out is not x
    assert out.dtype == np.float32 and out.shape == x.shape
    assert np.array_equal(out, x)
    assert np.array_equal(x, x0)


def test_single_frame_tone_8k():
    sr, fl, k = 8000, 160, 50
    x = _tone(sr)
    mask, xin = _erase(x, fl, [k])
    out = conceal(xin, sr, mask, fl)
    seg = slice(k * fl, (k + 1) * fl)
    nxt = slice((k + 1) * fl, (k + 2) * fl)
    assert _corr(out[seg], x[seg]) > 0.9
    assert abs(_rms_db(out[seg]) - _rms_db(x[seg])) < 3
    assert abs(_rms_db(out[nxt]) - _rms_db(x[nxt])) < 1


def test_long_erasure_fades_to_silence():
    sr, fl, k = 8000, 160, 50
    x = _tone(sr)
    mask, xin = _erase(x, fl, range(k, k + 6))
    out = conceal(xin, sr, mask, fl)
    sub = out[k * fl:(k + 6) * fl].reshape(12, 80)
    rms = np.array([_rms_db(s) for s in sub])
    assert np.all(np.diff(rms[1:]) <= 1e-9)
    assert np.all(rms[-2:] < -60)


def test_erasure_at_edges():
    sr, fl = 8000, 160
    x = _tone(sr, 1.0)
    mask, xin = _erase(x, fl, [0])
    out = conceal(xin, sr, mask, fl)
    assert out.shape == x.shape
    assert np.all(out[:fl] == 0)
    assert np.isfinite(out).all()

    x = x[: 40 * fl + 37]
    mask, xin = _erase(x, fl, [40])
    out = conceal(xin, sr, mask, fl)
    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_single_frame_tone_16k():
    sr, fl, k = 16000, 320, 50
    x = _tone(sr)
    mask, xin = _erase(x, fl, [k])
    out = conceal(xin, sr, mask, fl)
    seg = slice(k * fl, (k + 1) * fl)
    assert _corr(out[seg], x[seg]) > 0.9


def test_very_low_level_input():
    sr, fl, k = 8000, 160, 50
    x = _tone(sr)
    x *= 1e-4 / np.abs(x).max()
    mask, xin = _erase(x, fl, [k])
    out = conceal(xin, sr, mask, fl)
    assert np.isfinite(out).all()
    assert np.abs(out[k * fl:(k + 1) * fl]).max() <= 2 * np.abs(x).max()


def test_attenuation_schedule_and_onset_blend():
    sr, fl = 8000, 160
    x = _tone(sr)
    mask, xin = _erase(x, fl, range(50, 56))                      # 120 ms erasure from sample 8000
    out = conceal(xin, sr, mask, fl)
    sub = [out[8000 + 80 * k: 8000 + 80 * (k + 1)] for k in range(12)]
    ref = _rms_db(sub[0])
    # linear fade of 0.2 per 10 ms from the second sub-frame: mean gains 0.9, 0.7, 0.5, 0.3, 0.1
    for k, g in zip(range(1, 6), (0.9, 0.7, 0.5, 0.3, 0.1)):
        assert abs((_rms_db(sub[k]) - ref) - 20 * np.log10(g)) < (1.5 if k < 5 else 3.0), k
    assert all(np.all(s == 0) for s in sub[6:])                    # silence after 60 ms
    # only the last pitch // 4 samples before the erasure are touched (pitch <= 120 -> 30 samples)
    changed = np.nonzero(out[:8000] != x[:8000])[0]
    assert len(changed) and changed.min() >= 8000 - 30


def test_frame_len_must_be_subframe_multiple():
    x = _tone(8000, 1.0)
    with pytest.raises(ValueError):
        conceal(x, 8000, np.zeros(80, dtype=bool), 100)
    with pytest.raises(ValueError):
        conceal(_tone(16000, 1.0), 16000, np.zeros(200, dtype=bool), 80)

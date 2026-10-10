"""Resampling between any two rates.

Ratios whose reduced upsampling factor is at most ``dsp._MAX_POLYPHASE_UP``
(every pair of 8/16/24/32/48 kHz) filter a zero-stuffed signal; the others
(44.1, 22.05 and 11.025 kHz against those rates) go through
``dsp._resample_bank``, which applies the same low-pass at the output instants
only. A caller at 44.1 kHz therefore gets the channel a caller at 48 kHz gets.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch

from phonesim import PhoneCallSimulator, ResampleStage, dsp
from phonesim.core import make_context

RATES = [8000, 16000, 24000, 32000, 48000]
AWKWARD = [
    (44100, 8000), (8000, 44100), (44100, 16000), (16000, 44100), (44100, 24000), (24000, 44100),
    (44100, 48000), (48000, 44100), (22050, 16000), (16000, 22050), (11025, 8000), (8000, 11025),
    (32000, 44100),
]


def _factors(a: int, b: int) -> tuple[int, int]:
    """``(up, down)`` of ``a -> b`` in lowest terms."""
    g = math.gcd(a, b)
    return b // g, a // g


def _cos(f: float, sr: int, seconds: float = 1.0) -> torch.Tensor:
    n = np.arange(int(round(seconds * sr)))
    return torch.from_numpy(np.cos(2 * np.pi * f * n / sr)).view(1, 1, -1)       # float64


def _fit(y: np.ndarray, f: float, sr: int) -> tuple[float, float, float]:
    """Gain (dB), delay (s) and residual (dB re the tone) of ``y`` against a cosine at ``f``, middle 80 %."""
    m = np.arange(len(y) // 10, len(y) - len(y) // 10)
    basis = np.stack([np.cos(2 * np.pi * f * m / sr), np.sin(2 * np.pi * f * m / sr)], 1)
    (c, s), *_ = np.linalg.lstsq(basis, y[m], rcond=None)
    resid = y[m] - basis @ np.array([c, s])
    return (20 * math.log10(math.hypot(c, s)), math.atan2(s, c) / (2 * math.pi * f),
            20 * math.log10(math.sqrt(2) * resid.std()))


def _tone(a: int, b: int, f: float) -> tuple[float, float, float]:
    return _fit(dsp.resample(_cos(f, a), a, b)[0, 0].numpy(), f, b)


@pytest.mark.parametrize("zeros", [8, 32])
def test_bank_applies_the_zero_stuffing_filter(zeros):
    x = torch.randn(2, 3, 4001, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    for a, b in itertools.permutations(RATES, 2):
        ref = dsp.resample(x, a, b, zeros=zeros)
        got = dsp._resample_bank(x, *_factors(a, b), zeros=zeros)
        n = ref.shape[-1]
        assert got.shape[-1] - n in (0, 1), (a, b)      # the bank rounds the length, the zero-stuffing path floors it
        torch.testing.assert_close(got[..., :n], ref, rtol=0, atol=1e-12, msg=f"{a} -> {b}")


@pytest.mark.parametrize("a,b", AWKWARD)
def test_awkward_ratio_is_flat_and_has_no_delay(a, b):
    ny = min(a, b) / 2
    for frac in (0.05, 0.5, 0.8, 0.9):
        gain, delay, resid = _tone(a, b, frac * ny)
        assert abs(gain) < 0.02, (frac, gain)
        assert abs(delay) < 1e-9, (frac, delay)
        assert resid < -55, (frac, resid)               # images of an upsampled tone


@pytest.mark.parametrize("low", [8000, 16000])
def test_44k_and_48k_callers_get_the_same_band_edge(low):
    for frac in (0.9, 0.95, 0.98):
        f = frac * low / 2
        assert _tone(44100, low, f)[0] == pytest.approx(_tone(48000, low, f)[0], abs=0.005)
        assert _tone(low, 44100, f)[0] == pytest.approx(_tone(low, 48000, f)[0], abs=0.005)


@pytest.mark.parametrize("a,b", [(44100, 8000), (44100, 16000), (44100, 24000), (22050, 16000), (11025, 8000)])
def test_awkward_downsampling_rejects_aliases(a, b):
    for f in [f for f in (1.1 * b / 2, 1.5 * b / 2, 0.95 * a / 2) if f < 0.97 * a / 2]:
        y = dsp.resample(_cos(f, a), a, b)[0, 0].numpy()
        y = y[len(y) // 10: -len(y) // 10]
        assert 20 * math.log10(math.sqrt(2) * y.std()) < -55, f


@pytest.mark.parametrize("a,b", AWKWARD)
def test_awkward_ratio_length_rule(a, b):
    for t in [1, 2, 3, 7, 40, 441, 442, 1000, 4410, 12345]:
        y = dsp.resample(torch.randn(1, 1, t, generator=torch.Generator().manual_seed(t)), a, b)
        assert y.shape == (1, 1, max(1, round(t * b / a))), t
        assert torch.isfinite(y).all()


@pytest.mark.parametrize("a,b", [(44100, 16000), (16000, 44100), (48000, 16000), (16000, 48000)])
def test_empty_input_gives_empty_output(a, b):
    assert dsp.resample(torch.zeros(2, 3, 0), a, b).shape == (2, 3, 0)


@pytest.mark.parametrize("a,b", [(44100, 16000), (8000, 44100)])
def test_awkward_ratio_batches_channels_and_dtypes(a, b):
    x = torch.randn(3, 2, 5000, generator=torch.Generator().manual_seed(1))
    y = dsp.resample(x, a, b)
    for i, j in itertools.product(range(3), range(2)):
        torch.testing.assert_close(y[i, j], dsp.resample(x[i:i + 1, j:j + 1], a, b)[0, 0], rtol=0, atol=1e-5)
    xt = x.transpose(0, 1).contiguous().transpose(0, 1)
    assert not xt.is_contiguous()
    torch.testing.assert_close(dsp.resample(xt, a, b), y, rtol=0, atol=0)
    y64 = dsp.resample(x.double(), a, b)
    assert y64.dtype == torch.float64
    torch.testing.assert_close(y64.float(), y, rtol=0, atol=1e-5)


def test_awkward_ratio_does_not_drift_on_long_input():
    # Output positions past 2**22 samples, where float32 rounds them to half a
    # sample: the end of a long output must equal the same stretch resampled on
    # its own, which only exact positions give.
    a, b = 16000, 44100
    up, down = _factors(a, b)
    t = (1 << 22) // up * down + 40 * down
    x = torch.rand(1, 1, t, generator=torch.Generator().manual_seed(4)) - 0.5
    y = dsp.resample(x, a, b)[0, 0]
    f0 = t // down - 30
    part = dsp.resample(x[..., f0 * down:], a, b)[0, 0]
    m = 200                                             # the stretch's own edges see zero padding
    assert f0 * up > 1 << 22
    torch.testing.assert_close(y[f0 * up + m: f0 * up + part.numel() - m], part[m:-m], rtol=0, atol=1e-6)


def test_bank_split_into_phase_blocks_gives_the_same_output(monkeypatch):
    x = torch.randn(2, 1, 9000, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    for a, b in [(44100, 16000), (16000, 44100), (44100, 8000)]:
        ref = dsp.resample(x, a, b)
        monkeypatch.setattr(dsp, "_MAX_BANK_TAPS", 5000)
        got = dsp.resample(x, a, b)
        monkeypatch.undo()
        torch.testing.assert_close(got, ref, rtol=0, atol=1e-13)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("a, b", [(44100, 48000), (44100, 8000), (44101, 48000)])
def test_awkward_ratio_on_cuda_matches_the_cpu(a, b):
    # In float32, cuDNN would run the bank's conv in TF32 by default, up to about 2e-3 off the CPU here.
    x = torch.randn(2, 3, 2 * a, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(dsp.resample(x.cuda(), a, b).cpu(), dsp.resample(x, a, b), rtol=0, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_awkward_ratio_on_cuda_with_per_op_tf32_settings(monkeypatch):
    conv = getattr(torch.backends.cudnn, "conv", None)
    if conv is None or not hasattr(conv, "fp32_precision"):
        pytest.skip("this torch has no per-op TF32 settings")
    monkeypatch.setattr(conv, "fp32_precision", "ieee")
    x = torch.randn(1, 2, 44100, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(dsp.resample(x.cuda(), 44100, 48000).cpu(), dsp.resample(x, 44100, 48000),
                               rtol=0, atol=1e-5)


def test_coprime_rates_resample_in_bounded_memory():
    # 44101 and 48000 share no factor: a single bank would hold 48000 phases of 44167 taps.
    gain, delay, _ = _fit(dsp.resample(_cos(1000.0, 44101, 0.25), 44101, 48000)[0, 0].numpy(), 1000.0, 48000)
    assert abs(gain) < 0.01 and abs(delay) < 1e-9


def test_resample_stage_zeros_reaches_awkward_ratios():
    x = torch.randn(1, 1, 8820, generator=torch.Generator().manual_seed(3))
    out = {}
    for zeros in (8, 32):
        out[zeros] = ResampleStage(44100, 16000, zeros=zeros).process(x, make_context(44100, False, 0))
        torch.testing.assert_close(out[zeros], dsp._resample_bank(x, 160, 441, zeros=zeros), rtol=0, atol=0)
    assert not torch.allclose(out[8], out[32])


def test_44k_caller_gets_the_48k_channel():
    # The same multitone at 44.1 and 48 kHz through the G.711 profile (no ffmpeg
    # needed): every tone comes out with the same gain and phase.
    tones = (300.0, 1000.0, 2000.0, 3000.0, 3300.0)
    fits = {}
    for sr in (44100, 48000):
        n = np.arange(sr)
        x = sum(0.1 * np.cos(2 * np.pi * f * n / sr) for f in tones).astype(np.float32)
        y = PhoneCallSimulator(sr, sr, profile="pstn_narrowband", randomize=False)(x, seed=0).astype(np.float64)
        fits[sr] = [_fit(y, f, sr)[:2] for f in tones]
    for (g44, d44), (g48, d48), f in zip(fits[44100], fits[48000], tones):
        assert g44 == pytest.approx(g48, abs=0.05), f
        assert d44 == pytest.approx(d48, abs=1e-6), f

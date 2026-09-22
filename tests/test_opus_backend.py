"""Tests for the in-process libopus backend."""

import numpy as np
import pytest

from phonesim import opus_backend

if not opus_backend.available():
    pytest.skip("libopus shared library not available", allow_module_level=True)

SR = 16000
FRAME = SR * 20 // 1000
BITRATE = 24000
# libopus 1.5+ exports opus_packet_has_lbrr, so a next-packet decode is counted as FEC
# only when the packet really carries an LBRR copy; 1.4 counts every one as FEC.
KNOWN = opus_backend.lbrr_probe_available()


def _multitone(seconds=2.0):
    t = np.arange(int(SR * seconds)) / SR
    x = 0.2 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.sin(2 * np.pi * 900 * t) + 0.1 * np.sin(2 * np.pi * 2000 * t)
    return x.astype(np.float32)


def _speechlike(seconds=2.0, seed=0):
    """Pulse train with a gliding pitch through three formants, under a syllabic envelope.

    SILK codes the redundant (LBRR) copy only for frames its VAD marks as
    speech, and a stationary tone stops counting as speech after ~0.6 s, so
    the loss tests need a signal shaped like this.
    """
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    phase = np.cumsum(110 + 50 * np.sin(2 * np.pi * 0.7 * t)) / SR
    pulses = np.zeros(n, np.float32)
    pulses[np.flatnonzero(np.diff(np.floor(phase)) > 0)] = 1.0
    th = np.arange(int(0.03 * SR)) / SR
    h = sum(np.exp(-th * np.pi * bw) * np.sin(2 * np.pi * f * th) for f, bw in ((500, 80), (1500, 120), (2500, 160)))
    voiced = np.convolve(pulses, h)[:n]
    voiced += 0.02 * rng.standard_normal(n) * np.std(voiced)
    x = voiced * (0.5 * (1 - np.cos(2 * np.pi * 3.5 * t)) ** 1.5 + 0.05)
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


def _corr(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _best_lag(y, x, max_lag):
    """(lag, normalised cross-correlation) maximised over ``|lag| <= max_lag``; lag > 0 means ``y`` is late."""
    best = (0, -2.0)
    for lag in range(-max_lag, max_lag + 1):
        a, b = (y[lag:], x[:len(x) - lag]) if lag >= 0 else (y[:lag], x[-lag:])
        c = _corr(a, b)
        if c > best[1]:
            best = (lag, c)
    return best


def _rms(v):
    return float(np.sqrt(np.mean(v ** 2)) + 1e-12)


def _frame(y, i):
    return y[i * FRAME:(i + 1) * FRAME]


def _loud_frames(x, count=5, gap=10):
    """Indices of the ``count`` loudest frames that are at least ``gap`` frames apart."""
    energy = (x[:len(x) // FRAME * FRAME].reshape(-1, FRAME) ** 2).sum(axis=1)
    picked = []
    for i in np.argsort(-energy):
        if all(abs(int(i) - j) >= gap for j in picked):
            picked.append(int(i))
        if len(picked) == count:
            break
    return sorted(picked)


def test_version():
    assert opus_backend.version().startswith("libopus")


def test_clean_roundtrip_is_aligned():
    x = _multitone()
    y, info = opus_backend.encode_decode(x, SR, BITRATE)
    assert y.dtype == np.float32 and len(y) == len(x)
    assert np.isfinite(y).all()
    lag, c = _best_lag(y, x, 60)
    assert abs(lag) <= 1 and c > 0.9
    assert info == {"fec": 0, "plc": 0, "lbrr_known": KNOWN}


def test_isolated_losses_fec_and_plc():
    x = _speechlike()
    lost = _loud_frames(x)
    mask = np.zeros(len(x) // FRAME, dtype=bool)
    mask[lost] = True
    clean, _ = opus_backend.encode_decode(x, SR, BITRATE)

    y, info = opus_backend.encode_decode(x, SR, BITRATE, erased=mask, fec=True, expected_loss_pct=5)
    assert info == {"fec": 5, "plc": 0, "lbrr_known": KNOWN}
    for i in lost:
        assert _corr(_frame(y, i), _frame(clean, i)) > 0.5

    y, info = opus_backend.encode_decode(x, SR, BITRATE, erased=mask, fec=False, expected_loss_pct=5)
    assert info == {"fec": 0, "plc": 5, "lbrr_known": KNOWN}
    assert np.isfinite(y).all()
    first = lost[0]
    assert abs(20 * np.log10(_rms(_frame(y, first)) / _rms(_frame(y, first - 1)))) < 6


def test_burst_uses_fec_once_and_fades_plc():
    x = _speechlike()
    start = _loud_frames(x)[1]
    mask = np.zeros(len(x) // FRAME, dtype=bool)
    mask[start:start + 4] = True
    y, info = opus_backend.encode_decode(x, SR, BITRATE, erased=mask, fec=True, expected_loss_pct=5)
    assert info == {"fec": 1, "plc": 3, "lbrr_known": KNOWN}
    assert _rms(_frame(y, start + 2)) < _rms(_frame(y, start))


def test_zero_percent_loss_adds_no_lbrr():
    """Told 0 % loss the encoder sends no LBRR, so a next-packet decode is really PLC; only 1.5+ can report that."""
    x = _speechlike()
    mask = np.zeros(len(x) // FRAME, dtype=bool)
    mask[_loud_frames(x)] = True
    _, info = opus_backend.encode_decode(x, SR, BITRATE, erased=mask, fec=True, expected_loss_pct=0)
    if KNOWN:
        assert info == {"fec": 0, "plc": 5, "lbrr_known": True}
    else:
        assert info == {"fec": 5, "plc": 0, "lbrr_known": False}


def test_probe_splits_next_packet_decodes(monkeypatch):
    """The probe decides fec vs plc per next-packet decode and never changes the audio."""
    x = _speechlike()
    mask = np.zeros(len(x) // FRAME, dtype=bool)
    mask[_loud_frames(x)] = True                          # 5 frames whose successor arrives
    mask[-1] = True                                       # no successor: PLC whatever the probe says
    kw = dict(erased=mask, fec=True, expected_loss_pct=5)
    ref, _ = opus_backend.encode_decode(x, SR, BITRATE, **kw)

    answers = []

    def alternating(packet, length):
        assert isinstance(packet, bytes) and length == len(packet) > 0
        answers.append(len(answers) % 2)
        return answers[-1]

    monkeypatch.setattr(opus_backend, "_lbrr_probe", lambda lib: alternating)
    assert opus_backend.lbrr_probe_available() is True
    y, info = opus_backend.encode_decode(x, SR, BITRATE, **kw)
    assert answers == [0, 1, 0, 1, 0]                     # asked once per next-packet decode, in order
    assert info == {"fec": 2, "plc": 4, "lbrr_known": True}
    assert np.array_equal(y, ref)

    for answer in (0, 1):
        monkeypatch.setattr(opus_backend, "_lbrr_probe", lambda lib, a=answer: lambda packet, length: a)
        y, info = opus_backend.encode_decode(x, SR, BITRATE, **kw)
        assert info == {"fec": 5 * answer, "plc": 6 - 5 * answer, "lbrr_known": True}
        assert np.array_equal(y, ref)

    monkeypatch.setattr(opus_backend, "_lbrr_probe", lambda lib: None)
    assert opus_backend.lbrr_probe_available() is False
    y, info = opus_backend.encode_decode(x, SR, BITRATE, **kw)
    assert info == {"fec": 5, "plc": 1, "lbrr_known": False}
    assert np.array_equal(y, ref)


def test_edge_frames():
    x = _multitone(0.5)
    n = len(x) // FRAME
    mask = np.zeros(n, dtype=bool)
    mask[-1] = True
    _, info = opus_backend.encode_decode(x, SR, BITRATE, erased=mask)
    assert info == {"fec": 0, "plc": 1, "lbrr_known": KNOWN}
    _, info = opus_backend.encode_decode(x, SR, BITRATE, erased=[True])
    assert info == {"fec": 0 if KNOWN else 1, "plc": 1 if KNOWN else 0, "lbrr_known": KNOWN}


def test_rejects_bad_arguments():
    x = _multitone(0.5)
    with pytest.raises(ValueError):
        opus_backend.encode_decode(x, 44100, BITRATE)
    _, info = opus_backend.encode_decode(x, SR, BITRATE, erased=np.zeros(len(x) // FRAME + 3, dtype=bool))
    assert info == {"fec": 0, "plc": 0, "lbrr_known": KNOWN}    # a longer mask is truncated
    with pytest.raises(ValueError):
        opus_backend.encode_decode(x, SR, BITRATE, frame_ms=15)


def test_absent_library(monkeypatch):
    monkeypatch.setattr(opus_backend, "_load", lambda: None)
    assert opus_backend.available() is False
    assert opus_backend.lbrr_probe_available() is False
    assert opus_backend.version() == "absent"
    with pytest.raises(RuntimeError, match="libopus"):
        opus_backend.encode_decode(_multitone(0.1), SR, BITRATE)

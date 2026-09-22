"""Test suite for phonesim.

Output rate, shape and type preservation, determinism, exact length, band
limits, each stage's physics, the ffmpeg and libopus backends, codec
erasures, profile versioning, input validation, the CLI and
packaging. Tests that need a codec this ffmpeg lacks skip.

Run with ``pytest -q``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import phonesim
from phonesim import (
    PhoneCallSimulator,
    PhoneCallPipeline,
    BandlimitStage,
    ResampleStage,
    NoiseStage,
    PacketLossStage,
    CodecStage,
    list_profiles,
    analyze_channel,
)
from phonesim import profiles as P


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
ALL_PROFILES = list_profiles()


def _sim(profile, **kw):
    """Build a simulator, skipping the test when this ffmpeg lacks a codec it needs."""
    try:
        return PhoneCallSimulator(profile=profile, **kw)
    except phonesim.CodecUnavailableError as e:  # pragma: no cover - depends on the local ffmpeg
        pytest.skip(str(e))


def _tone(freq, sr, dur=1.0, amp=0.3):
    """A pure sinusoid as a float32 numpy array."""
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _multitone(freqs, sr, dur=1.0, amp=0.2):
    sig = np.zeros(int(sr * dur), dtype=np.float32)
    for f in freqs:
        sig += _tone(f, sr, dur, amp)
    return sig


def _band_energy(x, sr, lo, hi):
    """Fraction of spectral energy in [lo, hi) Hz."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    p = np.abs(X) ** 2
    total = p.sum() + 1e-12
    mask = (f >= lo) & (f < hi)
    return float(p[mask].sum() / total)


def _no_bad_values(x):
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    return bool(torch.isfinite(t).all())


# --------------------------------------------------------------------------- #
# Output sample rate & shape
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_output_sample_rate_default_24k(profile):
    """Every profile must return audio at the configured 24 kHz output rate."""
    sr = 24000
    x = _tone(440, sr, 0.5)
    sim = _sim(profile, randomize=False)
    y = sim(x, seed=0)
    # Output length should correspond to ~0.5 s at 24 kHz (allow small codec/resample slack).
    assert abs(len(y) - sr * 0.5) < 0.1 * sr


def test_output_sample_rate_custom():
    x = _tone(440, 24000, 0.5)
    sim = PhoneCallSimulator(
        input_sample_rate=24000, output_sample_rate=16000,
        profile="pstn_narrowband", randomize=False,
    )
    y = sim(x, seed=0)
    assert abs(len(y) - 16000 * 0.5) < 0.1 * 16000


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_shape_preserved_rank1(profile):
    x = _tone(440, 24000, 0.4)
    sim = _sim(profile, randomize=False)
    y = sim(x, seed=1)
    assert np.ndim(y) == 1


def test_shape_preserved_all_ranks_and_types():
    sim = _sim("voip_opus_wideband", randomize=False)
    n = 24000 // 2

    # rank-1 numpy
    y = sim(np.zeros(n, dtype=np.float32) + 0.01, seed=2)
    assert isinstance(y, np.ndarray) and y.ndim == 1

    # rank-2 [B, T] torch
    xb = torch.zeros(3, n) + 0.01
    yb = sim(xb, seed=2)
    assert isinstance(yb, torch.Tensor) and yb.ndim == 2 and yb.shape[0] == 3

    # rank-3 [B, C, T] torch
    xc = torch.zeros(2, 1, n) + 0.01
    yc = sim(xc, seed=2)
    assert isinstance(yc, torch.Tensor) and yc.ndim == 3 and yc.shape[:2] == (2, 1)


def test_numpy_in_numpy_out_torch_in_torch_out():
    sim = _sim("pstn_narrowband", randomize=False)
    x_np = _tone(300, 24000, 0.3)
    assert isinstance(sim(x_np, seed=3), np.ndarray)
    x_t = torch.from_numpy(x_np)
    assert isinstance(sim(x_t, seed=3), torch.Tensor)


# --------------------------------------------------------------------------- #
# No NaNs / Infs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_no_nans_infs(profile):
    x = _multitone([220, 1000, 3000], 24000, 0.5)
    sim = _sim(profile, randomize=True)
    y = sim(x, seed=7)
    assert _no_bad_values(y)


def test_no_nans_on_silence_and_loud():
    for amp, name in [(0.0, "silence"), (4.0, "clipping-loud")]:
        x = _tone(500, 24000, 0.3, amp=amp)
        sim = _sim("stress_multi_transcode", randomize=True)
        y = sim(x, seed=11)
        assert _no_bad_values(y), name


# --------------------------------------------------------------------------- #
# Determinism / randomization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_deterministic_same_seed(profile):
    x = _multitone([300, 1500], 24000, 0.5)
    sim = _sim(profile, randomize=True)
    y1 = np.asarray(sim(x, seed=123)).reshape(-1)
    y2 = np.asarray(sim(x, seed=123)).reshape(-1)
    n = min(len(y1), len(y2))
    assert np.allclose(y1[:n], y2[:n], atol=1e-6)


def test_randomized_differs_across_seeds():
    x = _multitone([300, 1500], 24000, 0.5)
    sim = _sim("stress_multi_transcode", randomize=True)
    y1 = np.asarray(sim(x, seed=1)).reshape(-1)
    y2 = np.asarray(sim(x, seed=2)).reshape(-1)
    n = min(len(y1), len(y2))
    # Different seeds should yield a materially different signal.
    assert not np.allclose(y1[:n], y2[:n], atol=1e-4)


def test_randomize_false_is_stable_without_seed():
    x = _tone(440, 24000, 0.4)
    sim = _sim("voip_opus_wideband", randomize=False)
    y1 = np.asarray(sim(x)).reshape(-1)
    y2 = np.asarray(sim(x)).reshape(-1)
    n = min(len(y1), len(y2))
    assert np.allclose(y1[:n], y2[:n], atol=1e-6)


# --------------------------------------------------------------------------- #
# Bandlimiting behaviour
# --------------------------------------------------------------------------- #
def test_bandlimit_stage_reduces_high_freq():
    sr = 16000
    x = _multitone([500, 6000], sr, 1.0)
    before = _band_energy(x, sr, 4000, 8000)
    pipe = PhoneCallPipeline(
        [BandlimitStage(low_hz=300, high_hz=3400, numtaps=257)],
        input_sample_rate=sr, output_sample_rate=sr, randomize=False,
    )
    y = pipe(x, seed=0)
    after = _band_energy(np.asarray(y), sr, 4000, 8000)
    assert after < before * 0.1


def test_narrowband_profile_suppresses_above_4k():
    """pstn_narrowband should leave almost no energy above 4 kHz."""
    sr = 24000
    x = _multitone([500, 1500, 6000, 9000], sr, 1.0)
    sim = _sim("pstn_narrowband", randomize=False)
    y = np.asarray(sim(x, seed=0)).reshape(-1)
    hf = _band_energy(y, sr, 4000, sr // 2)
    assert hf < 0.02, f"high-freq energy above 4 kHz too large: {hf}"


def test_wideband_profile_suppresses_above_8k():
    """A wideband (16 kHz) path cannot carry energy above 8 kHz (Nyquist)."""
    sr = 24000
    x = _multitone([500, 3000, 6000, 10000], sr, 1.0)
    sim = _sim("voip_to_cellular_wideband", randomize=False)
    y = np.asarray(sim(x, seed=0)).reshape(-1)
    hf = _band_energy(y, sr, 8000, sr // 2)
    assert hf < 0.02, f"energy above 8 kHz too large for wideband: {hf}"


def test_wideband_keeps_more_than_narrowband():
    """Wideband path must preserve more 3-7 kHz energy than narrowband."""
    sr = 24000
    x = _multitone([1000, 5000], sr, 1.0)
    nb = np.asarray(
        _sim("pstn_narrowband", randomize=False)(x, seed=0)
    ).reshape(-1)
    wb = np.asarray(
        _sim("voip_to_cellular_wideband", randomize=False)(x, seed=0)
    ).reshape(-1)
    band_nb = _band_energy(nb, sr, 3500, 7000)
    band_wb = _band_energy(wb, sr, 3500, 7000)
    assert band_wb > band_nb


# --------------------------------------------------------------------------- #
# Real ffmpeg codec round-trip
# --------------------------------------------------------------------------- #
# Every ffmpeg codec, each marked to skip when this ffmpeg cannot run it.
FFMPEG_CODECS = [
    pytest.param(c, marks=pytest.mark.skipif(
        c not in phonesim.ffmpeg_backend.available_codecs(), reason=f"this ffmpeg lacks {c}"))
    for c in sorted(phonesim.ffmpeg_backend.CODECS)
]


@pytest.mark.parametrize("codec", FFMPEG_CODECS)
def test_ffmpeg_codec_roundtrip(codec):
    """Each ffmpeg codec round-trips to finite, length-preserved audio that
    correlates with the input at lag 0; the backend compensates the codec's
    delay, and alignment is asserted by test_ffmpeg_codecs_are_time_aligned."""
    sr = 24000
    x = _multitone([400, 1200, 2400], sr, 0.5)
    pipe = PhoneCallPipeline(
        [CodecStage(codec=codec, backend="ffmpeg")],
        input_sample_rate=sr, output_sample_rate=sr,
    )
    y = np.asarray(pipe(x, seed=0)).reshape(-1)
    assert len(y) == len(x) and _no_bad_values(y)
    r = abs(np.corrcoef(x.astype(np.float64), y.astype(np.float64))[0, 1])
    assert r > 0.7, f"{codec}: weak correlation {r:.3f}"


# --------------------------------------------------------------------------- #
# Batch processing
# --------------------------------------------------------------------------- #
def test_batch_processing():
    sim = _sim("voip_to_cellular_wideband", randomize=True)
    x = torch.randn(4, 24000 // 2) * 0.1
    y = sim(x, seed=5)
    assert y.shape[0] == 4 and y.ndim == 2
    assert _no_bad_values(y)


# --------------------------------------------------------------------------- #
# Analysis utilities
# --------------------------------------------------------------------------- #
def test_analyze_channel_basic_metrics():
    sr = 24000
    # Include genuine content above 4 kHz so the narrowband channel has HF
    # energy to remove; otherwise the clean signal's >4 kHz energy is only FFT
    # leakage and the degraded-vs-clean comparison is meaningless.
    x = _multitone([300, 1500, 3000, 6000, 9000], sr, 1.0)
    sim = _sim("pstn_narrowband", randomize=False)
    y = sim(x, seed=0)
    report = analyze_channel(x, y, sample_rate=sr, compute_pesq=False, compute_stoi=True)
    assert "snr_db" in report and np.isfinite(report["snr_db"])
    assert report["hf_energy_degraded_>4k"] < report["hf_energy_clean_>4k"]
    assert any(k.startswith("300-3400") for k in report["band_energy_degraded"])


# --------------------------------------------------------------------------- #
# Config / explicit pipeline
# --------------------------------------------------------------------------- #
_needs_amr_wb = pytest.mark.skipif(
    "amr_wb" not in phonesim.ffmpeg_backend.available_codecs(), reason="needs an AMR-WB encoder"
)


@_needs_amr_wb
def test_explicit_pipeline():
    pipe = PhoneCallPipeline(
        [
            ResampleStage(24000, 16000),
            BandlimitStage(low_hz=50, high_hz=7000),
            CodecStage(codec="amr_wb"),
            PacketLossStage(loss_rate=0.02, burst_probability=0.2),
            NoiseStage(snr_db=(25, 40)),
            ResampleStage(16000, 24000),
        ],
        input_sample_rate=24000, output_sample_rate=24000, randomize=True,
    )
    x = _tone(1000, 24000, 0.5)
    y = pipe(x, seed=0)
    assert np.ndim(y) == 1 and _no_bad_values(y)
    assert abs(len(np.asarray(y)) - 24000 * 0.5) < 0.1 * 24000


def test_return_log():
    sim = _sim("stress_multi_transcode", randomize=False)
    x = _tone(440, 24000, 0.3)
    y, log = sim(x, seed=0, return_log=True)
    assert isinstance(log, list) and len(log) > 0


# --------------------------------------------------------------------------- #
# Packet loss: the realized loss rate matches the configured loss_rate at
# every burst setting.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("loss_rate", [0.02, 0.05, 0.10])
@pytest.mark.parametrize("burst", [0.0, 0.45])
def test_packet_loss_realized_rate_matches_configured(loss_rate, burst):
    import torch
    from phonesim.core import SimContext

    stage = PacketLossStage(loss_rate=loss_rate, burst_probability=burst, frame_ms=20)
    dev = torch.device("cpu")
    lost = 0
    total = 0
    for seed in range(300):
        ctx = SimContext(sample_rate=24000, generator=torch.Generator().manual_seed(seed))
        mask = stage._sample_mask(1000, loss_rate, ctx, dev)
        lost += int((mask == 0).sum().item())
        total += mask.numel()
    realized = lost / total
    # Realized loss is within 15 % (relative) of the configured rate.
    assert abs(realized - loss_rate) <= 0.15 * loss_rate, (
        f"realized loss {realized:.4f} != configured {loss_rate} "
        f"(ratio {realized / loss_rate:.2f}x) at burst={burst}"
    )


# --------------------------------------------------------------------------- #
# Length preservation: the output has exactly the input's sample count
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile", ALL_PROFILES)
@pytest.mark.parametrize("n", [24000, 16001, 8000, 12345])
def test_simulator_preserves_exact_length(profile, n):
    sim = _sim(profile, randomize=False)
    x = _tone(440, 24000, dur=n / 24000.0)[:n]
    y = np.asarray(sim(x, seed=0)).reshape(-1)
    assert len(y) == n, f"{profile}: {n} -> {len(y)}"


def test_simulator_preserves_length_with_output_resample():
    x = _tone(440, 24000, 0.5)  # 12000 samples @ 24k
    sim = PhoneCallSimulator(
        input_sample_rate=24000, output_sample_rate=16000,
        profile="pstn_narrowband", randomize=False,
    )
    y = np.asarray(sim(x, seed=0)).reshape(-1)
    assert len(y) == 8000  # 12000 * 16000/24000, exact


# --------------------------------------------------------------------------- #
# Integer-PCM round-trip stays inside the int16 range
# --------------------------------------------------------------------------- #
def test_integer_pcm_roundtrip_no_overflow():
    sim = _sim("pstn_narrowband", randomize=False)
    # int16 input pushed toward full scale; a +1.0 internal value must not wrap.
    x = (np.iinfo(np.int16).max * np.ones(4800)).astype(np.int16)
    y = sim(x, seed=0)
    assert y.dtype == np.int16
    assert np.all(y >= np.iinfo(np.int16).min) and np.all(y <= np.iinfo(np.int16).max)
    assert np.isfinite(y.astype(np.float64)).all()


# --------------------------------------------------------------------------- #
# Config loading from a real file
# --------------------------------------------------------------------------- #
@_needs_amr_wb
def test_config_from_yaml_file(tmp_path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        "input_sr: 24000\n"
        "output_sr: 24000\n"
        "stages:\n"
        "  - {type: ResampleStage, from_sr: 24000, to_sr: 16000}\n"
        "  - {type: BandlimitStage, low_hz: 50, high_hz: 7000}\n"
        "  - {type: CodecStage, codec: amr_wb}\n"
        "  - {type: ResampleStage, from_sr: 16000, to_sr: 24000}\n"
    )
    sim = PhoneCallSimulator.from_config(str(cfg))
    x = _tone(1000, 24000, 0.4)
    y = np.asarray(sim(x, seed=0)).reshape(-1)
    assert len(y) == len(x) and _no_bad_values(y)


def test_config_from_profile_dict():
    try:
        sim = PhoneCallSimulator.from_config(
            {"profile": "voip_opus_wideband", "input_sr": 24000, "output_sr": 24000}
        )
    except phonesim.CodecUnavailableError as e:  # pragma: no cover - depends on the machine
        pytest.skip(str(e))
    y = np.asarray(sim(_tone(440, 24000, 0.3), seed=0)).reshape(-1)
    assert _no_bad_values(y)


# --------------------------------------------------------------------------- #
# Analysis: band-energy bands are well-formed at 24 kHz
# --------------------------------------------------------------------------- #
def test_band_energy_ratios_wellformed_at_24k():
    from phonesim.analysis import band_energy_ratios

    x = np.random.default_rng(0).standard_normal(24000).astype(np.float32)
    bands = band_energy_ratios(x, 24000)
    # Every band is ascending and non-empty; ratios sum to ~1.
    for key in bands:
        lo, hi = key.replace("Hz", "").split("-")
        assert float(hi) > float(lo), f"inverted band {key}"
    # Bands tile [0, Nyquist) with half-open intervals, so they sum to ~1
    # (a couple of edge/Nyquist bins fall through — well under 0.1%).
    assert abs(sum(bands.values()) - 1.0) < 1e-3


def test_analyze_rejects_batched_input():
    from phonesim.analysis import _to_np_mono

    with pytest.raises(ValueError):
        _to_np_mono(np.zeros((4, 1, 1000)))


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------- #
# Warnings, clipping, describe, parameter validation
# --------------------------------------------------------------------------- #
def test_unknown_profile_param_rejected():
    with pytest.raises(TypeError, match="bitrat") as e:
        _sim("voip_to_cellular_narrowband", profile_params={"bitrat": "4.75k"})
    assert "accepts ['bitrate']" in str(e.value)


def test_describe_shows_backend_and_bitrate():
    text = _sim("voip_to_cellular_narrowband").describe()
    assert "backend=" in text
    assert "bitrate=12.2k" in text


def test_save_audio_warns_on_clipping(tmp_path):
    from phonesim import save_audio
    x = np.zeros(8000, dtype=np.float32)
    x[100] = 1.7
    with pytest.warns(phonesim.ClippingWarning):
        save_audio(str(tmp_path / "c.wav"), x, sr=8000)
    y, _ = phonesim.load_audio(str(tmp_path / "c.wav"), sr=None)
    assert abs(y).max() <= 1.0


def test_analyze_channel_generic_metrics():
    sr = 8000
    x = _tone(440, sr, 0.5)
    rep = analyze_channel(x, x * 0.5, sample_rate=sr, compute_pesq=False, compute_stoi=False,
                          metrics={"peak": lambda a, s: float(np.abs(np.asarray(a)).max())})
    assert rep["peak_clean"] == pytest.approx(0.3, abs=1e-3)
    assert rep["peak_degraded"] == pytest.approx(0.15, abs=1e-3)


@pytest.mark.skipif(not phonesim.ffmpeg_backend.have_ffmpeg(), reason="needs ffmpeg")
def test_ffmpeg_short_decode_raises(monkeypatch):
    from phonesim import ffmpeg_backend as fb
    real_read = fb.sf.read

    def short_read(path, *a, **k):
        y, s = real_read(path, *a, **k)
        return y[: len(y) // 2], s

    monkeypatch.setattr(fb.sf, "read", short_read)
    with pytest.raises(RuntimeError, match="decoded"):
        fb.encode_decode(np.zeros(16000, dtype=np.float32), 8000, "g711_ulaw")


def test_ffmpeg_ignores_stdin(monkeypatch):
    # ffmpeg stops with status 0 on a "q" read from its controlling stdin, so
    # _run must pass -nostdin, detach stdin and bound the run. Needs no ffmpeg.
    import subprocess
    fb, calls = phonesim.ffmpeg_backend, []

    def fake_run(cmd, **kw):
        calls.append((list(cmd), kw))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(fb.subprocess, "run", fake_run)
    assert fb._run(["ffmpeg", "-i", "in.wav", "out.wav"]) == b""
    [(argv, kw)] = calls
    assert argv == ["ffmpeg", "-nostdin", "-i", "in.wav", "out.wav"]
    assert kw["stdin"] is subprocess.DEVNULL and kw["timeout"] == fb.TIMEOUT_S


# --------------------------------------------------------------------------- #
# Profile versions and the channel stages
# --------------------------------------------------------------------------- #
from phonesim.profiles import list_versions, resolve_profile
from phonesim.stages.level import active_speech_level_db
from phonesim.stages.companding import g711_roundtrip


@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_bare_name_is_the_highest_version_and_every_version_builds(profile):
    versions = list_versions(profile)
    assert versions and versions == sorted(versions)
    assert resolve_profile(profile) == f"{profile}@{versions[-1]}"
    for v in versions:
        assert _sim(f"{profile}@{v}").pipeline.name == f"{profile}@{v}"


def test_bare_name_resolves_to_the_highest_registered_version(monkeypatch):
    monkeypatch.setitem(P._REGISTRY, "pstn_narrowband@2", P.pstn_narrowband)
    monkeypatch.setitem(P._LATEST, "pstn_narrowband", 2)
    assert list_versions("pstn_narrowband") == [1, 2]
    assert resolve_profile("pstn_narrowband") == "pstn_narrowband@2"
    assert resolve_profile("pstn_narrowband@1") == "pstn_narrowband@1"
    assert P.build_profile("pstn_narrowband@1").name == "pstn_narrowband@1"


def test_unknown_version_raises():
    with pytest.raises(KeyError, match="Versions"):
        _sim("pstn_narrowband@9")


def test_g711_segmented_matches_reference_codes():
    from phonesim.stages.companding import _ulaw_encode, _alaw_encode, _ulaw_decode, _alaw_decode
    # Known ITU-T G.711 code words (Sun g711.c reference) for a few PCM values.
    pcm = np.array([0, 4, -4, 100, -100, 1000, -1000, 8000, -8000, 32767, -32768], dtype=np.int16)
    assert _ulaw_encode(pcm).tolist() == [255, 254, 126, 242, 114, 206, 78, 160, 32, 128, 0]
    assert _alaw_encode(pcm).tolist() == [213, 213, 85, 211, 83, 250, 122, 138, 10, 170, 42]
    # decode(encode(x)) is within one quantisation step of x everywhere
    ramp = np.arange(-32768, 32768, dtype=np.int16)
    for enc, dec, step in ((_ulaw_encode, _ulaw_decode, 1024), (_alaw_encode, _alaw_decode, 512)):
        err = np.abs(dec(enc(ramp)).astype(np.int32) - ramp.astype(np.int32))
        assert err.max() <= step
    y = (g711_roundtrip(torch.from_numpy(ramp.astype(np.float32) / 32768.0), "mulaw") * 32768).round()
    assert len(torch.unique(y)) == 255


@pytest.mark.parametrize("law", ["ulaw", "mulaw", "alaw"])
def test_g711_law_spellings(law):
    """``ulaw`` and ``mulaw`` select the same coder; stage and codec keep their canonical names."""
    from phonesim.core import SimContext
    canon = "alaw" if law == "alaw" else "mulaw"
    stage = phonesim.CompandingStage(law=law)
    assert (stage.law, stage.name) == (canon, f"Compand-{canon}")
    x = torch.from_numpy(_multitone([300, 900], 8000, 0.2)).view(1, 1, -1)
    assert torch.equal(stage.process(x, SimContext(8000)), phonesim.CompandingStage(law=canon).process(x, SimContext(8000)))
    sim = _sim("pstn_narrowband", profile_params={"law": law})
    assert [s.codec for s in sim.pipeline.stages if isinstance(s, CodecStage)] == ["g711_alaw" if law == "alaw" else "g711_ulaw"]
    for build in (phonesim.CompandingStage, P._g711):
        with pytest.raises(ValueError, match=r"'alaw', 'mulaw', 'ulaw'"):
            build("pcm")


def test_channel_edge_low_band_kept_and_rolloff_to_nyquist():
    from phonesim.core import SimContext
    ctx = SimContext(sample_rate=8000, randomize=False, generator=torch.Generator().manual_seed(0))
    x = torch.randn(1, 1, 8000 * 4)
    y = phonesim.ChannelEdgeStage(low_hz=100.0, pass_hz=3400.0, stop_hz=4000.0, stop_db=45.0).process(x, ctx)
    yn = y.reshape(-1).numpy(); xn = x.reshape(-1).numpy()
    e = lambda s, lo, hi: 10 * np.log10(_band_energy(s, 8000, lo, hi) + 1e-12)
    assert e(yn, 150, 300) - e(xn, 150, 300) > -3.0          # low band kept
    assert -20 > e(yn, 3800, 3950) - e(xn, 3800, 3950) > -50  # roll-off, not a wall
    assert y.shape == x.shape


def test_speech_level_and_limiter():
    from phonesim.core import SimContext
    ctx = SimContext(sample_rate=8000, randomize=False, generator=torch.Generator().manual_seed(0))
    t = torch.arange(8000 * 4) / 8000
    x = (0.02 * torch.randn(8000 * 4) * (torch.sin(2 * np.pi * 2 * t) > 0).float()).view(1, 1, -1)
    y = phonesim.SpeechLevelStage(target_dbov=-26.0).process(x, ctx)
    assert abs(active_speech_level_db(y, 8000).item() + 26.0) < 0.5
    z = phonesim.LimiterStage(ceiling_dbfs=-1.0).process(y * 20, ctx)
    assert z.abs().max() <= 10 ** (-1 / 20) + 1e-3


def test_ambient_noise_snr_re_active_speech():
    from phonesim.core import SimContext
    ctx = SimContext(sample_rate=8000, randomize=False, generator=torch.Generator().manual_seed(0))
    t = torch.arange(8000 * 8) / 8000
    x = (0.05 * torch.randn(8000 * 8) * (torch.sin(2 * np.pi * 1.5 * t) > 0).float()).view(1, 1, -1)
    y = phonesim.AmbientNoiseStage(snr_db=45.0).process(x, ctx)
    n = (y - x).reshape(-1)
    snr = active_speech_level_db(x, 8000).item() - 20 * torch.log10(n.pow(2).mean().sqrt()).item()
    assert abs(snr - 45.0) < 1.0


def test_playout_buffer_events_scale_with_duration_and_preserve_length():
    from phonesim.core import SimContext
    ctx = SimContext(sample_rate=8000, randomize=True, generator=torch.Generator().manual_seed(3))
    x = torch.randn(1, 1, 8000 * 120) * 0.1
    y = phonesim.PlayoutBufferStage(late_rate=0.0, adapt_rate=0.004).process(x, ctx)
    assert y.shape == x.shape
    events = ctx.log[-1].split("adapt events ")[1].split()
    assert 10 <= len(events) <= 40            # 6000 frames x adapt_rate 0.004 = 24 expected


def test_clock_drift_is_ppm_scale_and_flat():
    from phonesim.core import SimContext
    ctx = SimContext(sample_rate=16000, randomize=False, generator=torch.Generator().manual_seed(0))
    x = torch.randn(1, 1, 16000 * 4)
    y = phonesim.ClockDriftStage(ppm=40.0).process(x, ctx)
    assert "+40 ppm" in ctx.log[-1]
    xn, yn = x.reshape(-1).numpy(), y.reshape(-1).numpy()
    hf = 10 * np.log10(_band_energy(yn, 16000, 6000, 7200) / _band_energy(xn, 16000, 6000, 7200))
    assert abs(hf) < 0.3                       # flat to within 0.3 dB up to 7.2 kHz


@pytest.mark.parametrize("codec", FFMPEG_CODECS)
def test_ffmpeg_codecs_are_time_aligned(codec):
    """Decoded output within one sample of the input: Opus at 16 kHz, the others at their native rate."""
    from phonesim.core import SimContext
    sr = 16000 if codec == "opus" else phonesim.ffmpeg_backend.native_sr(codec)
    ctx = SimContext(sample_rate=sr, randomize=False, generator=torch.Generator().manual_seed(0))
    x = torch.from_numpy((np.random.RandomState(0).randn(sr * 2) * 0.1).astype(np.float32)).view(1, 1, -1)
    y = CodecStage(codec, backend="ffmpeg").process(x, ctx)
    a, b = x.reshape(-1).numpy(), y.reshape(-1).numpy()
    w = 150                                    # past the largest table entry (AMR-WB, 95)
    c = np.correlate(b[sr // 2: sr], a[sr // 2 - w: sr + w], "valid")
    assert abs(int(np.argmax(c)) - w) <= 1


@pytest.mark.parametrize("profile", [
    "voip_to_cellular_narrowband@1", "voip_opus_wideband@1", "pstn_narrowband@1", "stress_multi_transcode@1",
])
def test_profiles_stay_below_full_scale(profile):
    sim = _sim(profile, randomize=True)
    x = _multitone([200, 700, 1800, 3000], 24000, 3.0, amp=0.25)
    for seed in range(3):
        y = np.asarray(sim(x, seed=seed))
        assert np.abs(y).max() <= 1.0, (profile, seed, np.abs(y).max())


# --------------------------------------------------------------------------- #
# Real codecs only; batch semantics; input validation; packaging
# --------------------------------------------------------------------------- #
from pathlib import Path
from phonesim import ffmpeg_backend as _fb


def test_missing_codec_raises_at_build(monkeypatch):
    monkeypatch.setattr(_fb, "have_ffmpeg", lambda: True)
    monkeypatch.setattr(_fb, "available_codecs", lambda: {"opus": 48000})
    monkeypatch.setattr(_ob, "available", lambda: True)
    with pytest.raises(phonesim.CodecUnavailableError, match="libvo_amrwbenc.*libopencore_amrwb"):
        P.build_profile("voip_to_cellular_wideband")
    monkeypatch.setattr(_fb, "have_ffmpeg", lambda: False)
    with pytest.raises(phonesim.CodecUnavailableError, match="ffmpeg not found"):
        P.build_profile("voip_g722_wideband")


def test_default_profile_is_voip_to_cellular_narrowband():
    import inspect
    assert inspect.signature(PhoneCallSimulator).parameters["profile"].default == "voip_to_cellular_narrowband"


def test_bitrate_validation():
    assert CodecStage("amr_nb", bitrate="12200").bitrate == "12.2k"
    with pytest.raises(ValueError, match="not a mode"):
        CodecStage("amr_nb", bitrate="9k")
    with pytest.raises(ValueError, match="no bitrate"):
        CodecStage("g711_ulaw", bitrate="64k")
    with pytest.raises(ValueError, match="outside"):
        CodecStage("opus", bitrate="3k")
    with pytest.raises(ValueError, match="Unknown codec"):
        CodecStage("amr_nb_x")
    with pytest.raises(ValueError, match="Unknown codec"):
        _fb.validate_bitrate("nope", None)
    with pytest.raises(ValueError, match="cannot parse"):
        CodecStage("amr_nb", bitrate="fast")
    with pytest.raises(ValueError, match="outside"):
        CodecStage("opus", bitrate="300k")
    assert CodecStage("amr_wb").bitrate == "12.65k"        # a mode is always set, so the log names it


def test_per_example_batch_rows_are_independent_and_reproducible():
    from phonesim.simulator import row_seeds
    sim = _sim("pstn_narrowband", randomize=True)
    x = np.stack([_multitone([300, 900, 2000], 24000, 1.0) for _ in range(3)])
    y1 = np.asarray(sim(x, seed=5, per_example=True))
    y2 = np.asarray(sim(x, seed=5, per_example=True))
    assert y1.shape == x.shape and np.array_equal(y1, y2)
    assert not np.allclose(y1[0], y1[1]) and not np.allclose(y1[1], y1[2])
    seeds = row_seeds(5, 3)
    assert seeds[0] == 5 and len(set(seeds)) == 3
    for i in range(3):
        assert np.allclose(y1[i], np.asarray(sim(x[i], seed=seeds[i])), atol=1e-6)
    y6 = np.asarray(sim(x, seed=6, per_example=True))     # adjacent seeds share no row
    assert not any(np.allclose(y1[i], y6[j]) for i in range(3) for j in range(3))
    one = np.asarray(sim(x[:1], seed=5, per_example=True))  # a batch of one is the unbatched call
    assert np.allclose(one[0], np.asarray(sim(x[0], seed=5)), atol=1e-6)


def test_shared_batch_draws_parameters_once():
    sim = _sim("pstn_narrowband", randomize=True)
    x = np.stack([_multitone([300, 900, 2000], 24000, 1.0) for _ in range(3)])
    y, log = sim(x, seed=5, return_log=True)
    assert np.array_equal(np.asarray(y), np.asarray(sim(x, seed=5)))
    assert sum(line.startswith("Codec:") for line in log) == 1


def test_bad_input_rejected():
    sim = _sim("pstn_narrowband")
    with pytest.raises(ValueError, match="empty"):
        sim(np.zeros(0, dtype=np.float32))
    with pytest.raises(ValueError, match="empty"):
        sim(np.zeros((0, 1000), dtype=np.float32))
    with pytest.raises(ValueError, match="shorter"):
        sim(np.zeros(100, dtype=np.float32))
    bad = _tone(440, 24000, 0.5); bad[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sim(bad)
    bad[10] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        sim(bad)


@pytest.mark.skipif(not _fb.have_ffmpeg(), reason="needs ffmpeg")
def test_log_records_ffmpeg_version_and_decoder():
    sim = _sim("voip_to_cellular_narrowband")
    _, log = sim(_tone(500, 24000, 2.0), seed=0, return_log=True)
    codec_lines = [l for l in log if l.startswith("Codec:amr_nb")]
    assert codec_lines and "decoder libopencore_amrnb" in codec_lines[0] and "ffmpeg " in codec_lines[0]
    assert "bitrate=12.2k" in codec_lines[0]


@pytest.mark.skipif(not _fb.have_ffmpeg(), reason="needs ffmpeg")
def test_decode_runs_the_reference_decoder(monkeypatch):
    calls, real_run = [], _fb._run
    monkeypatch.setattr(_fb, "_run", lambda cmd: (calls.append(cmd), real_run(cmd))[1])
    for codec, sr in (("opus", 16000), ("amr_nb", 8000), ("amr_wb", 16000)):
        if codec not in _fb.available_codecs():
            continue
        calls.clear()
        _fb.encode_decode(np.zeros(sr, dtype=np.float32), sr, codec)
        dec = calls[-1]
        assert dec.index("-c:a") < dec.index("-i") and dec[dec.index("-c:a") + 1] == _fb.decoder_name(codec)


@pytest.mark.skipif(not _fb.have_ffmpeg(), reason="needs ffmpeg")
def test_native_g711_matches_ffmpeg_reference():
    from phonesim.core import SimContext
    ctx = SimContext(sample_rate=8000, randomize=False)
    # Put the input exactly on the int16 grid (soundfile scales PCM_16 by
    # 32768) so both paths code the same samples.
    x = np.round(_multitone([300, 1100, 2700], 8000, 1.0, amp=0.2) * 32768) / 32768
    x = torch.from_numpy(x.astype(np.float32)).view(1, 1, -1)
    a = CodecStage("g711_ulaw", backend="native").process(x, ctx)
    b = CodecStage("g711_ulaw", backend="ffmpeg").process(x, ctx)
    # ffmpeg's tabulated encoder differs only on the 512 inputs next to a
    # segment boundary (tests/test_g711.py); this signal comes no closer than
    # 62 steps to any of them, so the round trips are identical.
    assert torch.equal(a, b)


def test_license_file_present():
    text = (Path(__file__).resolve().parents[1] / "LICENSE").read_text()
    assert text.startswith("MIT License") and "DeepMark" in text


def test_describe_lists_every_stage_with_its_backend():
    sim = _sim("pstn_narrowband")
    lines = sim.describe().splitlines()
    assert lines[0] == f"Pipeline({sim.pipeline.name}) with {len(sim.pipeline.stages)} stages:"
    assert len(lines) == 1 + len(sim.pipeline.stages)
    assert all(line.startswith("  [") and line == line.rstrip() for line in lines[1:])
    assert any("backend=native" in line for line in lines)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_profile_runs_on_cuda_tensor():
    for prof in ("pstn_narrowband", "voip_to_cellular_narrowband"):
        sim = _sim(prof, randomize=True)
        x = torch.from_numpy(_multitone([300, 1200], 24000, 0.5)).cuda()
        y = sim(x, seed=0)
        assert y.device.type == "cuda" and y.shape == x.shape and torch.isfinite(y).all()


def test_cli(tmp_path):
    import json
    from phonesim import cli, save_audio
    a = tmp_path / "a.wav"
    save_audio(str(a), _tone(440, 24000, 0.5), sr=24000)
    (tmp_path / "in").mkdir()
    for name, f in (("b.wav", 300), ("c.wav", 600)):
        save_audio(str(tmp_path / "in" / name), _tone(f, 24000, 0.5), sr=24000)
    out = tmp_path / "a_deg.wav"
    cli.main(["run", "--in", str(a), "--out", str(out), "--profile", "pstn_narrowband", "--seed", "1"])
    assert out.exists()
    cli.main(["batch", "--in-dir", str(tmp_path / "in"), "--out-dir", str(tmp_path / "out"),
              "--profile", "pstn_narrowband", "--seed", "1"])
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["b.wav", "c.wav"]
    cli.main(["analyze", "--clean", str(a), "--degraded", str(out), "--json", str(tmp_path / "m.json")])
    assert "snr_db" in json.loads((tmp_path / "m.json").read_text())
    cli.main(["info"])


def test_cli_reports_missing_codec(monkeypatch, capsys):
    from phonesim import cli
    monkeypatch.setattr(_fb, "have_ffmpeg", lambda: True)
    monkeypatch.setattr(_fb, "available_codecs", lambda: {})
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--in", "x.wav", "--out", "y.wav", "--profile", "voip_g722_wideband"])
    assert e.value.code == 1 and "g722" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Codec erasures and the packetised profiles
# --------------------------------------------------------------------------- #
from phonesim import opus_backend as _ob
from phonesim.stages.packet import erasure_mask


def test_erasure_mask_rate_and_bursts():
    from phonesim.core import SimContext
    lost, total, runs = 0, 0, []
    for seed in range(200):
        ctx = SimContext(sample_rate=8000, generator=torch.Generator().manual_seed(seed))
        m = erasure_mask(500, 0.05, 0.5, ctx)
        lost += int((m == 0).sum()); total += m.numel()
        z = (m == 0).int().tolist(); k = 0
        for v in z + [0]:
            if v: k += 1
            elif k: runs.append(k); k = 0
    assert abs(lost / total - 0.05) < 0.05 * 0.15
    assert 1.8 < np.mean(runs) < 2.2                      # mean burst length 1/(1-stay) = 2


def test_codec_stage_erasure_arguments():
    with pytest.raises(ValueError, match="libopus"):
        CodecStage("opus", backend="ffmpeg", erasure_rate=0.05)
    with pytest.raises(ValueError, match="only available for opus"):
        CodecStage("amr_nb", backend="libopus")
    with pytest.raises(ValueError, match="erasure_rate"):
        CodecStage("amr_nb", erasure_rate=1.5)
    assert CodecStage("amr_nb").erasure_range == (0.0, 0.0)
    text = phonesim.Pipeline([CodecStage("g722", erasure_rate=(0.0, 0.04))], name="t").describe()
    assert "erasures=0-0.04" in text


def test_fec_is_a_libopus_setting():
    assert not hasattr(CodecStage("amr_nb"), "fec")
    with pytest.raises(ValueError, match="libopus"):
        CodecStage("amr_nb", fec=False)


@pytest.mark.skipif("amr_nb" not in _fb.available_codecs(), reason="needs AMR-NB")
def test_amr_erasures_are_concealed_by_the_decoder():
    from phonesim.core import SimContext
    x = torch.from_numpy(_multitone([140, 280, 560, 1100], 8000, 2.0, amp=0.08)).view(1, 1, -1)
    ctx = SimContext(sample_rate=8000, randomize=True, generator=torch.Generator().manual_seed(1))
    y = CodecStage("amr_nb", erasure_rate=0.1).process(x, ctx)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert "erased" in ctx.log[-1] and "decoder concealment" in ctx.log[-1]
    n = int(ctx.log[-1].split("erased ")[1].split("/")[0])
    assert 3 <= n <= 25
    ctx0 = SimContext(sample_rate=8000, randomize=True, generator=torch.Generator().manual_seed(1))
    y0 = CodecStage("amr_nb").process(x, ctx0)
    assert "erased" not in ctx0.log[-1]
    assert not torch.allclose(y, y0)


@pytest.mark.skipif(not _ob.available(), reason="needs libopus")
def test_libopus_backend_with_fec_and_plc():
    from phonesim.core import SimContext
    x = torch.from_numpy(_multitone([300, 900, 2000], 16000, 2.0)).view(1, 1, -1)
    ctx = SimContext(sample_rate=16000, randomize=True, generator=torch.Generator().manual_seed(2))
    y = CodecStage("opus", backend="libopus", bitrate="24k", erasure_rate=0.1).process(x, ctx)
    assert y.shape == x.shape and torch.isfinite(y).all()
    import re
    how = (r"(\d+) FEC, (\d+) PLC" if _ob.lbrr_probe_available()
           else r"(\d+) from the next packet, FEC or PLC not distinguishable with libopus \S+, (\d+) PLC")
    m = re.search(r"^libopus \S+, bitrate=24k, encoder told (\d+) % loss, erased (\d+)/(\d+) frames \(" + how + r"\)$",
                  ctx.log[-1].split(": ", 1)[1])
    assert m, ctx.log[-1]
    pct, n, total, fec, plc = map(int, m.groups())
    assert pct == 10 and 0 < n <= total == 100 and fec + plc == n
    ctx2 = SimContext(sample_rate=16000, randomize=True, generator=torch.Generator().manual_seed(2))
    y2 = CodecStage("opus", backend="libopus", bitrate="24k", erasure_rate=0.1).process(x, ctx2)
    assert torch.equal(y, y2)


def test_g711_appendix_i_plc_on_native_and_ffmpeg_codecs():
    from phonesim.core import SimContext
    x = torch.from_numpy(_multitone([150, 300, 450], 8000, 1.0)).view(1, 1, -1)
    ctx = SimContext(sample_rate=8000, randomize=False, generator=torch.Generator().manual_seed(0))
    y = CodecStage("g711_ulaw", erasure_rate=0.1).process(x, ctx)
    assert "G.711 App. I" in ctx.log[-1] and torch.isfinite(y).all() and y.shape == x.shape
    y0 = CodecStage("g711_ulaw").process(x, SimContext(sample_rate=8000, randomize=False))
    assert not torch.equal(y, y0) and (y - y0).abs().max() < 1.0      # concealed, not zeroed
    if "g722" in _fb.available_codecs():
        x16 = torch.from_numpy(_multitone([150, 300, 450], 16000, 1.0)).view(1, 1, -1)
        ctx = SimContext(sample_rate=16000, randomize=False, generator=torch.Generator().manual_seed(0))
        y = CodecStage("g722", erasure_rate=0.1).process(x16, ctx)
        assert "G.711 App. I" in ctx.log[-1] and torch.isfinite(y).all()


@pytest.mark.parametrize("profile, codecs, ranges", [
    ("voip_to_cellular_narrowband@1", [("opus", "libopus"), ("amr_nb", "ffmpeg")], [(0.0, 0.01), (0.0, 0.01)]),
    ("voip_to_cellular_wideband@1", [("opus", "libopus"), ("amr_wb", "ffmpeg")], [(0.0, 0.01), (0.0, 0.01)]),
    ("voip_opus_wideband@1", [("opus", "libopus")], [(0.0, 0.05)]),
    ("voip_g722_wideband@1", [("g722", "ffmpeg")], [(0.0, 0.01)]),
])
def test_packetised_profiles_erase_inside_the_codec(profile, codecs, ranges):
    sim = _sim(profile)
    assert sim.pipeline.name == profile
    names = [s.name for s in sim.pipeline.stages]
    assert "PacketLoss" not in names and "JitterBuffer" not in names
    stages = [s for s in sim.pipeline.stages if isinstance(s, CodecStage)]
    assert [(s.codec, s.backend) for s in stages] == codecs
    assert [s.erasure_range for s in stages] == ranges
    assert all(s.late_range == (0.0, 0.0) for s in sim.pipeline.stages if isinstance(s, phonesim.PlayoutBufferStage))
    y, log = sim(_multitone([200, 700, 1800], 24000, 2.0), seed=3, return_log=True)
    assert sum("erased" in l for l in log) == len(codecs) and np.isfinite(np.asarray(y)).all()


@pytest.mark.parametrize("codec, backend", [("g711_ulaw", "native"), ("g722", "ffmpeg"), ("g726", "ffmpeg")])
def test_pcm_plc_paths_accept_any_input_length(codec, backend):
    from phonesim.core import SimContext
    if backend == "ffmpeg" and codec not in _fb.available_codecs():
        pytest.skip(codec)
    stage = CodecStage(codec, backend=backend, bitrate="32k" if codec == "g726" else None, erasure_rate=0.5)
    for n in (24002, 24000 * 3 + 1, 36011, 12000 + 7):
        x = torch.from_numpy(_multitone([300, 900], 24000, n / 24000.0)[:n]).view(1, 1, -1)
        ctx = SimContext(sample_rate=24000, randomize=True, generator=torch.Generator().manual_seed(n))
        y = stage.process(x, ctx)
        assert y.shape == x.shape and torch.isfinite(y).all()


def test_codec_stage_without_erasures_draws_nothing():
    from phonesim.core import SimContext
    x = torch.from_numpy(_multitone([300, 900], 8000, 0.5)).view(1, 1, -1)
    ctx = SimContext(sample_rate=8000, randomize=True, generator=torch.Generator().manual_seed(9))
    CodecStage("g711_ulaw").process(x, ctx)
    CodecStage("g711_ulaw", erasure_rate=(0.0, 0.0)).process(x, ctx)
    assert torch.rand((), generator=ctx.generator) == torch.rand((), generator=torch.Generator().manual_seed(9))
    assert all("erased" not in line for line in ctx.log)


def test_one_erasure_mask_per_call_shared_by_the_batch():
    from phonesim.core import SimContext
    row = _multitone([300, 900, 2000], 8000, 1.0)
    x = torch.from_numpy(np.stack([row, row])).view(2, 1, -1)
    ctx = SimContext(sample_rate=8000, randomize=True, generator=torch.Generator().manual_seed(4))
    y = CodecStage("g711_ulaw", erasure_rate=0.1).process(x, ctx)
    assert torch.equal(y[0], y[1]) and len([l for l in ctx.log if "erased" in l]) == 1


def test_missing_libopus_raises_at_build(monkeypatch):
    monkeypatch.setattr(_ob, "available", lambda: False)
    with pytest.raises(phonesim.CodecUnavailableError, match="libopus"):
        P.build_profile("voip_opus_wideband")
    if {"opus", "amr_nb", "amr_wb"} <= set(_fb.available_codecs()):
        P.build_profile("stress_multi_transcode")          # Opus without erasures runs on ffmpeg


def test_config_accepts_erasure_range():
    sim = PhoneCallSimulator.from_config({
        "input_sr": 24000, "output_sr": 24000,
        "stages": [
            {"type": "ResampleStage", "from_sr": 24000, "to_sr": 8000},
            {"type": "CodecStage", "codec": "g711_ulaw", "erasure_rate": [0.01, 0.05]},
            {"type": "ResampleStage", "from_sr": 8000, "to_sr": 24000},
        ],
    })
    y = sim(_tone(440, 24000, 0.5), seed=0)
    assert np.isfinite(np.asarray(y)).all()

"""Sample-rate contracts of the library API.

An unset output rate is 24 kHz, with a ``FutureWarning`` when the input rate
differs (from 0.3.0 it is the input rate). ``save_audio`` without ``sr`` and the
analysis functions without ``sample_rate`` take 24 kHz and warn. A
``ResampleStage`` whose ``from_sr`` disagrees with the signal it receives warns
and resamples from the signal's rate. Native G.711 only, so no ffmpeg build is
needed; the chain walk builds every profile without running a codec.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import soundfile as sf
import torch

from phonesim import (
    CodecStage,
    PhoneCallPipeline,
    PhoneCallSimulator,
    ResampleStage,
    analyze_channel,
    build_profile,
    plot_channel,
    save_audio,
)
from phonesim import profiles as P
from phonesim.config import pipeline_from_config
from phonesim.core import make_context
from tests.test_versions import all_codecs_available

RATES = (8000, 16000, 24000, 32000, 44100, 48000)


def _tone(sr: int, seconds: float = 0.25) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _g711(from_sr=None) -> list:
    """Resample to 8 kHz, then native G.711."""
    return [ResampleStage(from_sr, 8000), CodecStage("g711_ulaw", backend="native")]


@pytest.fixture
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# Each builds a pipeline from 16 kHz with the output rate unset; the setting it names.
UNSET_OUTPUT = {
    "PhoneCallSimulator": ("output_sample_rate",
                           lambda: PhoneCallSimulator(input_sample_rate=16000, profile="pstn_narrowband")),
    "PhoneCallPipeline": ("output_sample_rate", lambda: PhoneCallPipeline(_g711(), input_sample_rate=16000)),
    "build_profile": ("output_sr", lambda: build_profile("pstn_narrowband", input_sr=16000)),
    "from_config": ("output_sr",
                    lambda: PhoneCallSimulator.from_config({"profile": "pstn_narrowband", "input_sr": 16000})),
}


@pytest.mark.parametrize("how", sorted(UNSET_OUTPUT))
def test_unset_output_rate_stays_24k_and_warns_when_the_input_differs(how):
    setting, build = UNSET_OUTPUT[how]
    with pytest.warns(FutureWarning, match=rf"^{setting} not set: the output is resampled to 24000 Hz, "
                                           r"not kept at the 16000 Hz input rate; from 0\.3\.0"):
        built = build()
    if how == "build_profile":
        assert built.stages[-1].to_sr == 24000
    else:
        assert built.output_sample_rate == 24000
        assert len(built(_tone(16000), seed=0)) == 24000 // 4


def test_unset_output_rate_at_24k_does_not_warn(no_warnings):
    assert PhoneCallSimulator(profile="pstn_narrowband").output_sample_rate == 24000
    assert PhoneCallPipeline(_g711()).output_sample_rate == 24000
    assert build_profile("pstn_narrowband").stages[-1].to_sr == 24000
    assert pipeline_from_config({"profile": "pstn_narrowband"})[1:] == (24000, 24000)


@pytest.mark.parametrize("sr", [8000, 44100, 48000])
def test_explicit_output_rate_does_not_warn(no_warnings, sr):
    sim = PhoneCallSimulator(input_sample_rate=sr, output_sample_rate=sr, profile="pstn_narrowband")
    assert sim.pipeline.stages[-1].to_sr == sr
    assert len(sim(_tone(sr), seed=0)) == len(_tone(sr))


def test_save_audio_without_sr_writes_24k_and_warns(tmp_path):
    path = tmp_path / "a.wav"
    with pytest.warns(FutureWarning, match="^save_audio without sr writes a 24000 Hz header"):
        save_audio(str(path), _tone(16000))
    assert sf.info(str(path)).samplerate == 24000


def test_save_audio_with_sr_does_not_warn(tmp_path, no_warnings):
    save_audio(str(tmp_path / "a.wav"), _tone(16000), sr=16000)
    assert sf.info(str(tmp_path / "a.wav")).samplerate == 16000


def test_analyze_channel_without_sample_rate_warns():
    x = _tone(16000)
    with pytest.warns(FutureWarning, match="^analyze_channel without sample_rate takes the signals to be at 24000 Hz"):
        report = analyze_channel(x, x, compute_pesq=False, compute_stoi=False)
    assert report["sample_rate"] == 24000


def test_plot_channel_without_sample_rate_warns(tmp_path):
    pytest.importorskip("matplotlib")
    x = _tone(16000)
    with pytest.warns(FutureWarning, match="^plot_channel without sample_rate takes the signals to be at 24000 Hz"):
        plot_channel(x, x, path=str(tmp_path / "p.png"))


# --------------------------------------------------------------------------- #
# ResampleStage.from_sr
# --------------------------------------------------------------------------- #
def test_from_sr_mismatch_warns_and_resamples_from_the_signal_rate():
    x = _tone(16000)
    ref = PhoneCallPipeline(_g711(), input_sample_rate=16000, output_sample_rate=16000)(x, seed=0)
    pipe = PhoneCallPipeline(_g711(24000), input_sample_rate=16000, output_sample_rate=16000)
    with pytest.warns(FutureWarning, match=r"^Resample->8000: from_sr 24000 Hz disagrees with the 16000 Hz "
                                           r"signal it receives .* From 0\.3\.0 this raises ValueError$"):
        y = pipe(x, seed=0)
    np.testing.assert_array_equal(y, ref)


def test_later_stage_from_sr_is_checked_against_the_previous_to_sr():
    pipe = PhoneCallPipeline([ResampleStage(24000, 16000), ResampleStage(8000, 24000)])
    with pytest.warns(FutureWarning, match=r"^Resample->24000: from_sr 8000 Hz disagrees with the 16000 Hz"):
        pipe(_tone(24000), seed=0)


def test_from_sr_is_checked_when_the_signal_is_already_at_to_sr():
    pipe = PhoneCallPipeline([ResampleStage(24000, 8000)], input_sample_rate=8000, output_sample_rate=8000)
    with pytest.warns(FutureWarning, match="from_sr 24000 Hz disagrees with the 8000 Hz"):
        pipe(_tone(8000), seed=0)


@pytest.mark.parametrize("sr", RATES)
def test_from_sr_none_takes_any_rate(no_warnings, sr):
    pipe = PhoneCallPipeline(_g711(), input_sample_rate=sr, output_sample_rate=sr)
    assert len(pipe(_tone(sr), seed=0)) == len(_tone(sr))


def test_config_from_sr_is_checked():
    sim = PhoneCallSimulator.from_config({"input_sr": 16000, "output_sr": 16000, "stages": [
        {"type": "ResampleStage", "from_sr": 24000, "to_sr": 8000},
        {"type": "CodecStage", "codec": "g711_ulaw", "backend": "native"},
    ]})
    with pytest.warns(FutureWarning, match="from_sr 24000 Hz disagrees with the 16000 Hz"):
        sim(_tone(16000), seed=0)


def test_from_sr_given_as_a_string_is_a_rate(no_warnings):
    # YAML loads a quoted from_sr as a string; to_sr and the config rates were always int()-ed.
    sim = PhoneCallSimulator.from_config({"input_sr": 16000, "output_sr": 16000, "stages": [
        {"type": "ResampleStage", "from_sr": "16000", "to_sr": 8000},
        {"type": "CodecStage", "codec": "g711_ulaw", "backend": "native"},
    ]})
    sim(_tone(16000), seed=0)


def test_from_sr_that_is_not_a_rate_warns_rather_than_failing():
    # 0.1.0 never read from_sr, so a value like "16k" ran; it now gets the mismatch warning.
    pipe = PhoneCallPipeline(_g711("16k"), input_sample_rate=16000, output_sample_rate=16000)
    with pytest.warns(FutureWarning, match="from_sr 16k Hz disagrees with the 16000 Hz"):
        pipe(_tone(16000), seed=0)


# Each leaves a rate unset or mismatched somewhere inside phonesim.
CALLER_WARNINGS = {
    "simulator": lambda tmp: PhoneCallSimulator(input_sample_rate=16000, profile="pstn_narrowband"),
    "from_config": lambda tmp: PhoneCallSimulator.from_config({"profile": "pstn_narrowband", "input_sr": 16000}),
    "save_audio": lambda tmp: save_audio(str(tmp / "x.wav"), np.zeros(8, np.float32)),
    "from_sr": lambda tmp: PhoneCallPipeline(_g711(24000), input_sample_rate=16000,
                                             output_sample_rate=16000)(_tone(16000), seed=0),
    "stage_called_as_module": lambda tmp: ResampleStage(24000, 8000)(
        torch.from_numpy(_tone(16000)).view(1, 1, -1), make_context(16000, False, 0)),
}


@pytest.mark.parametrize("how", sorted(CALLER_WARNINGS))
def test_warnings_name_the_callers_line(how, tmp_path):
    with pytest.warns(FutureWarning) as rec:
        CALLER_WARNINGS[how](tmp_path)
    assert [w.filename for w in rec] == [__file__] * len(rec)


VARIANTS = [(key, {}) for key in sorted(P._REGISTRY)]
VARIANTS += [("voip_to_cellular_wideband@1", {"second_transcode": True})]
VARIANTS += [("voip_opus_wideband@1", {"internal_sr": sr}) for sr in (8000, 12000, 24000, 48000)]


def test_builtin_chains_line_up_at_every_rate():
    """Each ResampleStage's from_sr is the rate the signal reaches it at, so no profile warns."""
    with all_codecs_available():
        for (key, params), input_sr, output_sr in [(v, i, o) for v in VARIANTS for i in RATES for o in RATES]:
            rate = input_sr
            for stage in build_profile(key, input_sr=input_sr, output_sr=output_sr, **params).stages:
                if isinstance(stage, ResampleStage):
                    assert stage.from_sr == rate, (key, params, input_sr, output_sr, stage.name)
                    rate = stage.to_sr
            assert rate == output_sr, (key, params, input_sr, output_sr)

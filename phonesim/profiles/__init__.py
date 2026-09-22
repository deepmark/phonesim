"""Named simulator profiles.

A *profile* is a function that, given the input/output sample rates, returns a
configured :class:`~phonesim.core.Pipeline`. Profiles are registered by name so
they can be referenced from the high-level API and from YAML/JSON configs.

* ``pstn_narrowband``             - analogue loop into a G.711 exchange, 8 kHz.
* ``pstn_g726``                   - PSTN leg over G.726 ADPCM (32 kbit/s), 8 kHz.
* ``voip_opus_wideband``          - WebRTC/VoIP over Opus with frame erasures, 16 kHz.
* ``voip_g722_wideband``          - SIP HD voice over G.722, 16 kHz.
* ``voip_to_cellular_wideband``   - VoIP -> mobile HD voice (AMR-WB), 16 kHz.
* ``voip_to_cellular_narrowband`` - VoIP -> regular mobile call (AMR-NB), 8 kHz.
* ``stress_multi_transcode``      - synthetic stress chain, not a real route.

``name@N`` fixes a profile's stage chain and parameters; a bare name resolves to
its highest registered version.

Every codec is real (ffmpeg, or libopus for Opus with erasures); a profile whose
codec this machine cannot run raises :class:`CodecUnavailableError` when built.
Frame erasures are drawn per codec hop: AMR and Opus conceal them with their own
decoders, G.722 with the G.711 Appendix I algorithm on the decoded signal.
Every profile starts and ends at the caller's sample rates (24 kHz by default).
"""

from __future__ import annotations

import inspect
from typing import Callable, Optional

from phonesim.core import CodecUnavailableError, Pipeline, Stage
from phonesim import stages as S
from phonesim import ffmpeg_backend


# Each profile builder has the signature ``fn(input_sr, output_sr, **kw) -> Pipeline``.
# "name@N" -> builder. A bare name resolves to its latest version.
_REGISTRY: dict[str, Callable[..., Pipeline]] = {}
_LATEST: dict[str, int] = {}


def register(name: str, version: int = 1):
    """Register ``fn`` as ``name@version``; the highest version is the default."""
    def deco(fn):
        _REGISTRY[f"{name}@{version}"] = fn
        _LATEST[name] = max(_LATEST.get(name, 0), version)
        return fn
    return deco


def list_profiles() -> list[str]:
    """Profile names; each resolves to its latest version."""
    return sorted(_LATEST.keys())


def list_versions(name: str) -> list[int]:
    """Registered versions of ``name`` (bare or ``name@N``), ascending."""
    base = name.split("@")[0]
    return sorted(int(k.split("@")[1]) for k in _REGISTRY if k.split("@")[0] == base)


def resolve_profile(name: str) -> str:
    """Map a bare name or ``name@N`` to its ``name@version`` key; a bare name is the highest version."""
    base, _, ver = name.partition("@")
    if base not in _LATEST:
        raise KeyError(f"Unknown profile {base!r}. Available: {list_profiles()}")
    return f"{base}@{ver or _LATEST[base]}"


def build_profile(name: str, input_sr: int = 24000, output_sr: int = 24000, **kw) -> Pipeline:
    name = resolve_profile(name)
    if name not in _REGISTRY:
        raise KeyError(f"Unknown profile version {name!r}. Versions: {list_versions(name)}")
    fn = _REGISTRY[name]
    params = inspect.signature(fn).parameters
    allowed = {n for n, p in params.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    allowed -= {"input_sr", "output_sr"}
    unknown = set(kw) - allowed
    if unknown:
        raise TypeError(
            f"profile {name!r} has no parameter(s) {sorted(unknown)}; accepts {sorted(allowed)}"
        )
    pipe = fn(input_sr=input_sr, output_sr=output_sr, **kw)
    pipe.name = name
    return pipe


def _codec(codec: str, bitrate: Optional[str] = None, backend: str = "ffmpeg", **kw) -> S.CodecStage:
    """A real codec stage, or an error saying what this machine lacks."""
    readme = 'see README, "Real codecs via ffmpeg"'
    if backend == "libopus":
        return S.CodecStage(codec, backend="libopus", bitrate=bitrate, **kw)     # raises when libopus is absent
    if not ffmpeg_backend.have_ffmpeg():
        raise CodecUnavailableError(f"ffmpeg not found on PATH (or PHONESIM_FFMPEG); {readme}")
    if codec not in ffmpeg_backend.available_codecs():
        enc, dec = ffmpeg_backend.encoder_name(codec), ffmpeg_backend.decoder_name(codec)
        hint = " AMR needs a build with libopencore-amr and libvo-amrwbenc;" if codec.startswith("amr") else ""
        raise CodecUnavailableError(
            f"{codec!r}: this ffmpeg cannot round-trip it (encoder {enc}, decoder {dec}).{hint} {readme}."
        )
    return S.CodecStage(codec, backend="ffmpeg", bitrate=bitrate, **kw)


def _opus(bitrate: str = "24k", erasure: float | tuple[float, float] = 0.0) -> S.CodecStage:
    """Opus; with erasures it runs on libopus so the decoder conceals them."""
    if erasure == 0.0:
        return _codec("opus", bitrate=bitrate)
    return _codec("opus", bitrate=bitrate, backend="libopus", erasure_rate=erasure, burst_probability=0.3)


def _amrwb(bitrate: str = "12.65k", erasure: float | tuple[float, float] = 0.0) -> S.CodecStage:
    """AMR-WB; 12.65 kbit/s is the common VoLTE/HD-voice mode."""
    return _codec("amr_wb", bitrate=bitrate, erasure_rate=erasure, burst_probability=0.3)


def _amrnb(bitrate: str = "12.2k", erasure: float | tuple[float, float] = 0.0) -> S.CodecStage:
    """AMR-NB; 12.2 kbit/s is the highest mode and the default here."""
    return _codec("amr_nb", bitrate=bitrate, erasure_rate=erasure, burst_probability=0.3)


# Accepted spellings of the G.711 law -> codec-name suffix.
_G711_LAWS = {"ulaw": "ulaw", "mulaw": "ulaw", "alaw": "alaw"}


def _g711(law: str = "ulaw") -> S.CodecStage:
    """G.711 on the native coder; ``law`` is ``"ulaw"`` (or ``"mulaw"``) or ``"alaw"``."""
    if law not in _G711_LAWS:
        raise ValueError(f"law must be one of {sorted(_G711_LAWS)}, got {law!r}")
    return S.CodecStage(f"g711_{_G711_LAWS[law]}", backend="native")


def _send_side(snr_db: tuple[float, float] = (40.0, 55.0)) -> list[Stage]:
    """Terminal/platform side before the first encoder: ambient noise, level control, limiter."""
    return [
        S.AmbientNoiseStage(snr_db=snr_db),
        S.SpeechLevelStage(target_dbov=(-32.0, -22.0)),
        S.LimiterStage(ceiling_dbfs=-1.0),
    ]


def _narrowband_edge() -> S.ChannelEdgeStage:
    # Digital mobile path: AMR-NB's 80 Hz pre-filter and the handset send mask
    # set the low edge; the high roll-off reaches the 4 kHz Nyquist.
    return S.ChannelEdgeStage(low_hz=(80.0, 120.0), pass_hz=(3300.0, 3450.0), stop_hz=4000.0, stop_db=45.0)


def _wideband_edge() -> S.ChannelEdgeStage:
    return S.ChannelEdgeStage(low_hz=(50.0, 80.0), pass_hz=(6800.0, 7100.0), stop_hz=8000.0, stop_db=45.0)


def _playout() -> S.PlayoutBufferStage:
    """Receive-side playout buffer that only adapts; late frames are the codec stage's erasures."""
    return S.PlayoutBufferStage(frame_ms=20, late_rate=0.0, burst_probability=0.3, adapt_rate=(0.0, 0.003))


# Each codec stage draws one erasure process for its hop, covering network loss
# and late arrivals (radio frame erasures on the mobile leg). AMR and Opus lose
# the frames in the coded stream and conceal them with their decoders; G.722 is
# decoded in full and the erased frames are replaced in the decoded PCM by the
# G.711 Appendix I algorithm. Managed trunks and radio legs lose up to 1 % of
# frames (the LTE conversational-voice loss target); the public Internet up to 5 %.
_TRUNK_ERASURE = (0.0, 0.01)
_INTERNET_ERASURE = (0.0, 0.05)
_RADIO_ERASURE = (0.0, 0.01)


@register("pstn_narrowband")
def pstn_narrowband(input_sr: int = 24000, output_sr: int = 24000, law: str = "ulaw") -> Pipeline:
    """Analogue loop into a G.711 exchange; the loop sets a 300 Hz low edge."""
    stages = [
        S.ResampleStage(input_sr, 8000),
        *_send_side(snr_db=(35.0, 50.0)),
        S.ChannelEdgeStage(low_hz=(250.0, 320.0), pass_hz=(3300.0, 3450.0), stop_hz=4000.0, stop_db=45.0),
        _g711(law),
        S.ClockDriftStage(ppm=(-50.0, 50.0)),
        S.ResampleStage(8000, output_sr),
    ]
    return Pipeline(stages, name="pstn_narrowband")


@register("pstn_g726")
def pstn_g726(input_sr: int = 24000, output_sr: int = 24000, bitrate: str = "32k") -> Pipeline:
    """Narrowband PSTN leg carried by G.726 ADPCM."""
    stages = [
        S.ResampleStage(input_sr, 8000),
        *_send_side(snr_db=(35.0, 50.0)),
        S.ChannelEdgeStage(low_hz=(250.0, 320.0), pass_hz=(3300.0, 3450.0), stop_hz=4000.0, stop_db=45.0),
        _codec("g726", bitrate=bitrate),
        S.ClockDriftStage(ppm=(-50.0, 50.0)),
        S.ResampleStage(8000, output_sr),
    ]
    return Pipeline(stages, name="pstn_g726")


@register("voip_opus_wideband")
def voip_opus_wideband(
    input_sr: int = 24000, output_sr: int = 24000, bitrate: str = "24k", internal_sr: int = 16000
) -> Pipeline:
    """WebRTC-style VoIP: Opus over the public Internet, erasures concealed by the decoder."""
    stages = [
        S.ResampleStage(input_sr, internal_sr),
        *_send_side(),
        S.ChannelEdgeStage(low_hz=(50.0, 80.0), pass_hz=(internal_sr * 0.42, internal_sr * 0.45), stop_hz=internal_sr / 2, stop_db=45.0),
        _opus(bitrate=bitrate, erasure=_INTERNET_ERASURE),
        _playout(),
        S.ClockDriftStage(ppm=(-50.0, 50.0)),
        S.ResampleStage(internal_sr, output_sr),
    ]
    return Pipeline(stages, name="voip_opus_wideband")


@register("voip_g722_wideband")
def voip_g722_wideband(input_sr: int = 24000, output_sr: int = 24000) -> Pipeline:
    """Wideband SIP trunk carried by G.722 (16 kHz sub-band ADPCM, 64 kbit/s); erasures concealed by G.711 Appendix I."""
    stages = [
        S.ResampleStage(input_sr, 16000),
        *_send_side(),
        _wideband_edge(),
        _codec("g722", erasure_rate=_TRUNK_ERASURE, burst_probability=0.3),
        _playout(),
        S.ClockDriftStage(ppm=(-50.0, 50.0)),
        S.ResampleStage(16000, output_sr),
    ]
    return Pipeline(stages, name="voip_g722_wideband")


@register("voip_to_cellular_wideband")
def voip_to_cellular_wideband(
    input_sr: int = 24000, output_sr: int = 24000, second_transcode: bool = False, bitrate: str = "12.65k"
) -> Pipeline:
    """Opus VoIP origination, then an AMR-WB mobile leg; ``second_transcode`` adds an AMR-NB interconnect hop."""
    stages = [
        S.ResampleStage(input_sr, 16000),
        *_send_side(),
        _opus(bitrate="24k", erasure=_TRUNK_ERASURE),
        _wideband_edge(),
        _amrwb(bitrate=bitrate, erasure=_RADIO_ERASURE),
    ]
    if second_transcode:
        stages += [S.ResampleStage(16000, 8000), _narrowband_edge(), _amrnb(), S.ResampleStage(8000, 16000)]
    stages += [_playout(), S.ClockDriftStage(ppm=(-50.0, 50.0)), S.ResampleStage(16000, output_sr)]
    return Pipeline(stages, name="voip_to_cellular_wideband")


@register("voip_to_cellular_narrowband")
def voip_to_cellular_narrowband(input_sr: int = 24000, output_sr: int = 24000, bitrate: str = "12.2k") -> Pipeline:
    """Opus VoIP origination, then one AMR-NB mobile leg; ``bitrate`` is the AMR-NB mode."""
    stages = [
        S.ResampleStage(input_sr, 16000),
        *_send_side(),
        _opus(bitrate="24k", erasure=_TRUNK_ERASURE),
        S.ResampleStage(16000, 8000),
        _narrowband_edge(),
        _amrnb(bitrate=bitrate, erasure=_RADIO_ERASURE),
        _playout(),
        S.ClockDriftStage(ppm=(-50.0, 50.0)),
        S.ResampleStage(8000, output_sr),
    ]
    return Pipeline(stages, name="voip_to_cellular_narrowband")


@register("stress_multi_transcode")
def stress_multi_transcode(input_sr: int = 24000, output_sr: int = 24000) -> Pipeline:
    """Opus -> AMR-NB -> G.711 -> AMR-WB with heavy loss, drift and offset.

    No real route carries a call this way; the parameters are synthetic.
    """
    stages = [
        S.ResampleStage(input_sr, 16000),
        S.BandlimitStage(low_hz=(50, 120), high_hz=(6500, 7200)),
        _opus(bitrate="16k"),
        S.PacketLossStage(loss_rate=(0.02, 0.10), burst_probability=0.45, frame_ms=20, conceal=True),
        # transcode down to narrowband
        S.ResampleStage(16000, 8000),
        S.BandlimitStage(low_hz=(300, 350), high_hz=(3200, 3400)),
        _amrnb(),
        _g711("alaw"),
        S.PacketLossStage(loss_rate=(0.01, 0.06), burst_probability=0.5, frame_ms=20, conceal=True),
        # back up to wideband for a final cellular leg
        S.ResampleStage(8000, 16000),
        _amrwb(),
        S.JitterBufferStage(frame_ms=20, late_rate=(0.01, 0.04), underrun_rate=0.02),
        S.AGCStage(target_dbfs=(-24, -14), mode="static", max_gain_db=35),
        S.NoiseStage(snr_db=(15, 32), color="pink"),
        S.ClipStage(threshold=(0.9, 1.0), drive_db=(0, 5), mode="soft"),
        S.SpeedDriftStage(max_drift=0.004),
        S.TimeOffsetStage(max_ms=40),
        S.ResampleStage(16000, output_sr),
    ]
    return Pipeline(stages, name="stress_multi_transcode")

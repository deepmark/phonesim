"""phonesim - a provider-free phone-call audio-degradation simulator.

Models the signal path of a telephone call: VoIP/WebRTC transport, PSTN and
cellular interconnects, real mobile voice codecs, packet loss, playout timing,
level control and background noise. Runs the real codecs (ffmpeg, libopus, the
ITU-T G.711 coder); AMR and Opus conceal erased frames with their own decoders,
G.711/G.722/G.726 with the G.711 Appendix I algorithm.

>>> from phonesim import PhoneCallSimulator, load_audio, save_audio
>>> x, sr = load_audio("input.wav", sr=24000)
>>> sim = PhoneCallSimulator(profile="voip_to_cellular_narrowband")
>>> y = sim(x, seed=1)
>>> save_audio("degraded.wav", y, sr=24000)
"""

from phonesim.core import Pipeline, SimContext, Stage
from phonesim.simulator import PhoneCallSimulator, PhoneCallPipeline
from phonesim.io_utils import load_audio, save_audio
from phonesim.profiles import (
    list_profiles,
    build_profile,
    CodecUnavailableError,
)
from phonesim.io_utils import ClippingWarning
from phonesim.analysis import analyze_channel, plot_channel
from phonesim import stages
from phonesim.stages import (
    ResampleStage,
    BandlimitStage,
    AGCStage,
    GainStage,
    ClipStage,
    NoiseStage,
    CompandingStage,
    PacketLossStage,
    JitterBufferStage,
    CodecStage,
    TimeOffsetStage,
    SpeedDriftStage,
    SpeechLevelStage,
    LimiterStage,
    ChannelEdgeStage,
    AmbientNoiseStage,
    PlayoutBufferStage,
    ClockDriftStage,
)

__version__ = "0.1.0"

__all__ = [
    "PhoneCallSimulator",
    "PhoneCallPipeline",
    "Pipeline",
    "SimContext",
    "Stage",
    "load_audio",
    "save_audio",
    "list_profiles",
    "build_profile",
    "CodecUnavailableError",
    "ClippingWarning",
    "analyze_channel",
    "plot_channel",
    "stages",
    "ResampleStage",
    "BandlimitStage",
    "AGCStage",
    "GainStage",
    "ClipStage",
    "NoiseStage",
    "CompandingStage",
    "PacketLossStage",
    "JitterBufferStage",
    "CodecStage",
    "TimeOffsetStage",
    "SpeedDriftStage",
    "SpeechLevelStage",
    "LimiterStage",
    "ChannelEdgeStage",
    "AmbientNoiseStage",
    "PlayoutBufferStage",
    "ClockDriftStage",
]

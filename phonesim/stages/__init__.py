"""Individual signal-degradation stages.

Each stage is a :class:`phonesim.core.Stage` subclass. Import them from here:

>>> from phonesim.stages import ResampleStage, BandlimitStage, CodecStage
"""

from phonesim.stages.resample import ResampleStage
from phonesim.stages.filtering import BandlimitStage
from phonesim.stages.gain import AGCStage, GainStage, ClipStage
from phonesim.stages.noise import NoiseStage
from phonesim.stages.companding import CompandingStage
from phonesim.stages.packet import PacketLossStage, JitterBufferStage
from phonesim.stages.codec import CodecStage
from phonesim.stages.misc import TimeOffsetStage, SpeedDriftStage
from phonesim.stages.level import SpeechLevelStage, LimiterStage
from phonesim.stages.channel_edge import ChannelEdgeStage
from phonesim.stages.ambient import AmbientNoiseStage
from phonesim.stages.timing import PlayoutBufferStage, ClockDriftStage

__all__ = [
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

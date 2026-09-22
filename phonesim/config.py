"""Configuration: build pipelines from YAML/JSON specs.

Two config shapes are supported:

1. Profile reference::

     {"profile": "voip_to_cellular_wideband", "input_sr": 24000, "output_sr": 24000,
      "params": {"second_transcode": false}}

2. Explicit stage list::

     {"input_sr": 24000, "output_sr": 24000,
      "stages": [
        {"type": "ResampleStage", "from_sr": 24000, "to_sr": 16000},
        {"type": "BandlimitStage", "low_hz": 50, "high_hz": 7000},
        {"type": "CodecStage", "codec": "amr_wb"},
        ...
      ]}

``input_sr`` and ``output_sr`` are the rates of the signal entering and leaving
the pipeline; both default to 24000 when the key is absent. ``params`` are
keyword arguments for the profile builder and may be omitted.

YAML is used if PyYAML is installed; JSON always works.
"""

from __future__ import annotations

import json
import os
from typing import Any

from phonesim.core import Pipeline
from phonesim import stages as S
from phonesim import profiles as P

try:  # optional
    import yaml  # type: ignore

    _HAVE_YAML = True
except Exception:  # pragma: no cover
    _HAVE_YAML = False


_STAGE_TYPES = {
    "ResampleStage": S.ResampleStage,
    "BandlimitStage": S.BandlimitStage,
    "AGCStage": S.AGCStage,
    "GainStage": S.GainStage,
    "ClipStage": S.ClipStage,
    "NoiseStage": S.NoiseStage,
    "CompandingStage": S.CompandingStage,
    "PacketLossStage": S.PacketLossStage,
    "JitterBufferStage": S.JitterBufferStage,
    "CodecStage": S.CodecStage,
    "TimeOffsetStage": S.TimeOffsetStage,
    "SpeedDriftStage": S.SpeedDriftStage,
    "SpeechLevelStage": S.SpeechLevelStage,
    "LimiterStage": S.LimiterStage,
    "ChannelEdgeStage": S.ChannelEdgeStage,
    "AmbientNoiseStage": S.AmbientNoiseStage,
    "PlayoutBufferStage": S.PlayoutBufferStage,
    "ClockDriftStage": S.ClockDriftStage,
}


_DEFAULT_SR = 24000


def stage_from_dict(spec: dict[str, Any]) -> S.Stage:
    if not isinstance(spec, dict):
        raise ValueError(f"stage spec must be a mapping with a 'type', got {spec!r}")
    spec = dict(spec)
    stype = spec.pop("type", None)
    if stype not in _STAGE_TYPES:
        raise KeyError(f"Unknown stage type {stype!r}. Known: {sorted(_STAGE_TYPES)}")
    # ResampleStage uses positional from_sr/to_sr; accept both key styles.
    if stype == "ResampleStage":
        from_sr = spec.pop("from_sr", None)
        if "to_sr" not in spec:
            raise ValueError("ResampleStage needs 'to_sr'")
        to_sr = spec.pop("to_sr")
        return S.ResampleStage(from_sr, to_sr, **spec)
    return _STAGE_TYPES[stype](**spec)


def load_config(obj: Any) -> dict:
    """Load a config from a path, file-like, JSON/YAML string, or dict."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (str, os.PathLike)) and os.path.exists(obj):
        with open(obj, "r") as f:
            text = f.read()
    elif hasattr(obj, "read"):
        text = obj.read()
    else:
        text = str(obj)
    text_stripped = text.lstrip()
    if text_stripped.startswith("{"):
        return json.loads(text)
    if _HAVE_YAML:
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(f"config is not valid YAML: {' '.join(str(e).split())}") from e
    raise RuntimeError("Config is not JSON and PyYAML is not installed for YAML support.")


def pipeline_from_config(obj: Any) -> tuple[Pipeline, int, int]:
    """Return ``(pipeline, input_sr, output_sr)`` built from a config."""
    cfg = load_config(obj)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping with 'profile' or 'stages'")
    # An empty YAML key (``input_sr:``) loads as None and takes the default.
    input_sr = int(cfg.get("input_sr") or _DEFAULT_SR)
    output_sr = int(cfg.get("output_sr") or _DEFAULT_SR)
    if "profile" in cfg:
        params = cfg.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"params must be a mapping, got {params!r}")
        pipe = P.build_profile(cfg["profile"], input_sr=input_sr, output_sr=output_sr, **params)
        return pipe, input_sr, output_sr
    if "stages" in cfg:
        specs = cfg["stages"]
        if not isinstance(specs, list):
            raise ValueError(f"stages must be a list of stage mappings, got {specs!r}")
        stages = [stage_from_dict(s) for s in specs]
        name = cfg.get("name", "from_config")
        return Pipeline(stages, name=name), input_sr, output_sr
    raise ValueError("config must contain either 'profile' or 'stages'")

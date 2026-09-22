"""Versioning contract: ``name@N`` fixes the stage chain and its parameters.

The codec implementation is logged, not versioned. Three goldens under
``tests/data/``, written by ``python tests/regen_goldens.py``:

* ``profile_fingerprints.json``: class, name and constructor attributes of
  every stage of every registered version. The chain is built with every
  codec declared available (building runs no codec), so this test runs on
  any machine.
* ``run_logs.json``: the parameter log of a seeded run on a fixed multitone,
  with the codec build strings and the measured level-control gains masked.
* ``<name@N>.npy``: the output of that run for the versions whose codec is
  the native G.711 coder, compared by SNR. Torch's float32 output differs in
  the last bit between thread counts and builds. G.711 quantises the signal
  before it, so those differences do not survive the coder; the float32
  resampler after it still differs in the last bit between thread counts
  (about 145 dB SNR on the stored multitone), hence the SNR comparison. The
  ffmpeg codecs read the signal as int16 floored by libsndfile, and the
  adaptive coders (G.726, G.722, Opus, AMR) turn a flipped bit into a
  different coded stream, so their output is not pinned.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import numpy as np
import pytest
import torch

import phonesim
from phonesim import PhoneCallSimulator, ffmpeg_backend, opus_backend
from phonesim import profiles as P
from phonesim.core import Pipeline

DATA = Path(__file__).parent / "data"
FINGERPRINTS = DATA / "profile_fingerprints.json"
RUN_LOGS = DATA / "run_logs.json"
SR = 24000
SEED = 0
# Versions with a reproducible output, and the SNR (dB) the current output
# must reach against the stored one.
OUTPUT_GOLDENS = {
    "pstn_narrowband@1": 60.0,
}
KEYS = sorted(P._REGISTRY)
REGEN = "run python tests/regen_goldens.py"

# Codec build strings, measured gains and how libopus split the erased frames
# between FEC and PLC (a property of the installed libopus) are not part of a
# version; the erasure count before the parenthesis is.
_MASKS = (
    (re.compile(r"ffmpeg [^\s,]+"), "ffmpeg *"),
    (re.compile(r"libopus [^\s,]+"), "libopus *"),
    (re.compile(r"mean gain \S+"), "mean gain *"),
    (re.compile(r"^(Codec:opus: libopus .*)\([^()]*\)$"), r"\1(*)"),
)


def multitone() -> np.ndarray:
    """1 s at 24 kHz: 300, 900 and 2000 Hz at 0.2, 0.1 and 0.1."""
    t = np.arange(SR) / SR
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((300, 0.2), (900, 0.1), (2000, 0.1)))
    return x.astype(np.float32)


def output_path(key: str) -> Path:
    return DATA / f"{key}.npy"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


@contextmanager
def all_codecs_available() -> Iterator[None]:
    """Let every codec stage build, whatever this machine has (nothing runs a codec)."""
    codecs = {c: ffmpeg_backend.native_sr(c) for c in ffmpeg_backend.CODECS}
    with mock.patch.multiple(ffmpeg_backend, available_codecs=lambda: codecs, have_ffmpeg=lambda: True), \
            mock.patch.object(opus_backend, "available", lambda: True):
        yield


def _plain(v: Any) -> Any:
    """A form JSON reproduces exactly: floats by repr, tuples as lists."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return repr(float(v))
    if isinstance(v, (tuple, list)):
        return [_plain(e) for e in v]
    if isinstance(v, dict):
        return {str(k): _plain(e) for k, e in v.items()}
    return repr(v)


def _is_bookkeeping(k: str, v: Any) -> bool:
    """Private and torch.nn.Module state; ``name`` is the fingerprint's own field."""
    return (k.startswith("_") or k in ("training", "name")
            or isinstance(v, (torch.nn.Module, torch.Tensor)) or callable(v))


def fingerprint(pipe: Pipeline) -> list:
    """``[class, name, {constructor attributes}]`` for each stage in order."""
    return [
        [type(s).__name__, s.name, {k: _plain(v) for k, v in vars(s).items() if not _is_bookkeeping(k, v)}]
        for s in pipe.stages
    ]


def mask_log(log: list[str]) -> list[str]:
    out = []
    for line in log:
        for rx, rep in _MASKS:
            line = rx.sub(rep, line)
        out.append(line)
    return out


def snr_db(ref: np.ndarray, out: np.ndarray) -> float:
    """``20 log10(rms(ref) / rms(ref - out))``; identical arrays give about 600 dB."""
    ref, out = ref.astype(np.float64), out.astype(np.float64)
    return float(20 * np.log10(np.sqrt(np.mean(ref ** 2)) / (np.sqrt(np.mean((ref - out) ** 2)) + 1e-30)))


def _sim(key: str) -> PhoneCallSimulator:
    """Build ``key``, skipping the test when this machine lacks one of its codecs."""
    try:
        return PhoneCallSimulator(profile=key)
    except phonesim.CodecUnavailableError as e:  # pragma: no cover - depends on the local ffmpeg
        pytest.skip(str(e))


@pytest.mark.parametrize("key", KEYS)
def test_stage_chain_is_pinned(key):
    with all_codecs_available():
        got = fingerprint(P.build_profile(key))
    golden = load_json(FINGERPRINTS)
    assert key in golden, f"{key} has no fingerprint; {REGEN}"
    assert got == golden[key], f"{key} changed in place; register a new version"


@pytest.mark.parametrize("key", KEYS)
def test_drawn_parameters_are_pinned(key):
    sim = _sim(key)
    golden = load_json(RUN_LOGS)
    assert key in golden, f"{key} has no run log; {REGEN}"
    _, log = sim(multitone(), seed=SEED, return_log=True)
    assert mask_log(log) == golden[key], f"{key} changed in place; register a new version"


@pytest.mark.parametrize("key, min_snr", sorted(OUTPUT_GOLDENS.items()))
def test_output_is_pinned(key, min_snr):
    sim = _sim(key)
    path = output_path(key)
    assert path.exists(), f"{key} has no output golden; {REGEN}"
    ref = np.load(path)
    out = np.asarray(sim(multitone(), seed=SEED), dtype=np.float32)
    assert out.shape == ref.shape, f"{key}: output {out.shape} vs stored {ref.shape}"
    snr = snr_db(ref, out)
    assert snr >= min_snr, f"{key}: {snr:.1f} dB SNR against the stored output, need {min_snr:g}; register a new version"

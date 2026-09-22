"""ffmpeg-based codec backend.

Encode -> decode round trips through the ``ffmpeg`` binary. Where ffmpeg has
its own decoder next to the reference one, the reference decoder is forced.

==========  ==================  ==================  ===========
codec       encoder             decoder             native rate
==========  ==================  ==================  ===========
g711_ulaw   pcm_mulaw           pcm_mulaw           8 kHz
g711_alaw   pcm_alaw            pcm_alaw            8 kHz
g722        adpcm_g722          adpcm_g722          16 kHz
g726        adpcm_g726          adpcm_g726          8 kHz (16k-40k)
opus        libopus             libopus             48 kHz
amr_nb      libopencore_amrnb   libopencore_amrnb   8 kHz
amr_wb      libvo_amrwbenc      libopencore_amrwb   16 kHz
==========  ==================  ==================  ===========

AMR needs an ffmpeg built with libopencore-amrnb, libopencore-amrwb and
libvo-amrwbenc (e.g. a static GPL build); distro builds usually have none of
them. :func:`available_codecs` probes what this build round-trips.
``PHONESIM_FFMPEG`` names the binary to use instead of the one on ``PATH``.
AMR frames can be erased in the coded stream between encode and decode
(``erased``); the OpenCORE decoder then runs its own error concealment.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf


FFMPEG = shutil.which(os.environ.get("PHONESIM_FFMPEG") or "ffmpeg")


# codec -> (encoder, container, native sample rate, takes a bitrate, decoder)
_CODEC_TABLE = {
    "g711_ulaw": ("pcm_mulaw", "wav", 8000, False, "pcm_mulaw"),
    "g711_alaw": ("pcm_alaw", "wav", 8000, False, "pcm_alaw"),
    "g722": ("adpcm_g722", "wav", 16000, False, "adpcm_g722"),
    "g726": ("adpcm_g726", "wav", 8000, True, "adpcm_g726"),
    "opus": ("libopus", "ogg", 48000, True, "libopus"),
    "amr_nb": ("libopencore_amrnb", "amr", 8000, True, "libopencore_amrnb"),
    "amr_wb": ("libvo_amrwbenc", "amr", 16000, True, "libopencore_amrwb"),
}
CODECS = frozenset(_CODEC_TABLE)

# Valid modes per codec. Opus takes any rate in its range (libopus, mono).
MODES = {
    "amr_nb": ("4.75k", "5.15k", "5.9k", "6.7k", "7.4k", "7.95k", "10.2k", "12.2k"),
    "amr_wb": ("6.6k", "8.85k", "12.65k", "14.25k", "15.85k", "18.25k", "19.85k", "23.05k", "23.85k"),
    "g726": ("16k", "24k", "32k", "40k"),
}
_OPUS_RANGE = (6_000, 256_000)
# Mode used when a stage gives none, so the log always names the mode that ran.
DEFAULT_MODE = {"amr_nb": "12.2k", "amr_wb": "12.65k", "g726": "32k", "opus": "24k"}


def _bps(text: str) -> int:
    t = text.strip().lower()
    return int(round(float(t[:-1]) * 1000)) if t.endswith("k") else int(t)


def validate_bitrate(codec: str, bitrate) -> Optional[str]:
    """Return the canonical mode string (the codec's default for ``None``), or raise ``ValueError``."""
    if codec not in _CODEC_TABLE:
        raise ValueError(f"Unknown codec {codec!r}; known: {sorted(CODECS)}")
    if bitrate is None:
        return DEFAULT_MODE.get(codec)
    if not _CODEC_TABLE[codec][3]:
        raise ValueError(f"{codec} has no bitrate setting")
    try:
        bps = _bps(str(bitrate))
    except ValueError:
        raise ValueError(f"{codec}: cannot parse bitrate {bitrate!r}; use e.g. '12.2k' or 12200") from None
    if codec in MODES:
        for m in MODES[codec]:
            if _bps(m) == bps:
                return m
        raise ValueError(f"{codec}: bitrate {bitrate!r} is not a mode; use one of {MODES[codec]}")
    if not _OPUS_RANGE[0] <= bps <= _OPUS_RANGE[1]:
        raise ValueError(f"opus: bitrate {bitrate!r} outside {_OPUS_RANGE[0]}-{_OPUS_RANGE[1]} bit/s")
    return str(bitrate)


def encoder_name(codec: str) -> str:
    return _CODEC_TABLE[codec][0]


def decoder_name(codec: str) -> str:
    return _CODEC_TABLE[codec][4]


@functools.lru_cache(maxsize=1)
def ffmpeg_version() -> str:
    """The ffmpeg version string (first token after "ffmpeg version"), or "absent"."""
    if not have_ffmpeg():
        return "absent"
    try:
        out = subprocess.run([FFMPEG, "-version"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, timeout=10).stdout.decode("utf8", "ignore")
        return out.split()[2] if out.startswith("ffmpeg version") else out.split("\n")[0][:40]
    except Exception:
        return "unknown"


# Algorithmic delay of each decoder's output relative to its input, in samples
# at the codec's native rate. Measured by cross-correlation; G.711/G.726/Opus
# (via Ogg pre-skip) come back aligned.
CODEC_DELAY = {"amr_nb": 40, "amr_wb": 95, "g722": 22}


def have_ffmpeg() -> bool:
    return FFMPEG is not None


@functools.lru_cache(maxsize=1)
def available_codecs() -> dict:
    """Return ``{codec: native_sr}`` for codecs that round-trip in this ffmpeg.

    Probes by running a 1-frame encode/decode of silence. Cached.
    """
    if not have_ffmpeg():
        return {}
    out = {}
    silence = np.zeros(2048, dtype=np.float32)
    for codec, (enc, fmt, sr, _, _dec) in _CODEC_TABLE.items():
        try:
            y = _roundtrip(silence, sr, codec)
            if y is not None and np.isfinite(y).all():
                out[codec] = sr
        except Exception:
            pass
    return out


def native_sr(codec: str) -> int:
    return _CODEC_TABLE[codec][2]


# Wall-clock budget per ffmpeg invocation. Real-time speech codecs run far
# faster than real time, so a 10-minute input finishes in seconds.
TIMEOUT_S = 120.0


def _run(cmd: list[str]) -> bytes:
    # ffmpeg reads its controlling stdin for interactive keys; a stray "q"
    # stops the encode early with exit status 0. Detach it.
    try:
        proc = subprocess.run(
            [cmd[0], "-nostdin", *cmd[1:]],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out after {TIMEOUT_S:.0f}s") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}): {proc.stderr.decode('utf8', 'ignore')[-400:]}"
        )
    return proc.stdout


# AMR storage format (RFC 4867 section 5): a magic string, then frames of one
# header byte (P FT3 FT2 FT1 FT0 Q P P) plus a payload whose size depends on
# the frame type FT. Each frame carries 20 ms; NO_DATA is the single byte 0x7C.
AMR_FRAME_MS = 20
_AMR_MAGIC = {"amr_nb": b"#!AMR\n", "amr_wb": b"#!AMR-WB\n"}
_AMR_PAYLOAD = {
    "amr_nb": {0: 12, 1: 13, 2: 15, 3: 17, 4: 19, 5: 20, 6: 26, 7: 31, 8: 5, 15: 0},
    "amr_wb": {0: 17, 1: 23, 2: 32, 3: 36, 4: 40, 5: 46, 6: 50, 7: 58, 8: 60, 9: 5, 14: 0, 15: 0},
}
_AMR_NO_DATA = b"\x7c"


def amr_frames(data: bytes, codec: str) -> tuple[bytes, list[bytes]]:
    """Split an AMR file into ``(magic, frames)``, each frame header byte plus payload.

    Raises ``ValueError`` on an unknown codec, a bad magic, an unknown frame
    type or a truncated frame.
    """
    if codec not in _AMR_MAGIC:
        raise ValueError(f"{codec!r} is not an AMR codec")
    magic, sizes = _AMR_MAGIC[codec], _AMR_PAYLOAD[codec]
    if not data.startswith(magic):
        raise ValueError(f"{codec}: bad magic {data[:len(magic)]!r}")
    frames, i = [], len(magic)
    while i < len(data):
        ft = (data[i] >> 3) & 0xF
        if ft not in sizes:
            raise ValueError(f"{codec}: unknown frame type {ft} at byte {i}")
        n = 1 + sizes[ft]
        if i + n > len(data):
            raise ValueError(f"{codec}: truncated frame at byte {i}")
        frames.append(data[i:i + n])
        i += n
    return magic, frames


def erase_amr_frames(data: bytes, codec: str, erased) -> bytes:
    """Replace the frames flagged in ``erased`` (bool array over frames) by NO_DATA.

    A mask shorter than the file is padded with ``False``, a longer one is
    truncated (the encoder pads the last frame, so counts differ by one).
    """
    magic, frames = amr_frames(data, codec)
    erased = np.asarray(erased, dtype=bool).reshape(-1)[: len(frames)]
    erased = np.pad(erased, (0, len(frames) - len(erased)))
    return magic + b"".join(_AMR_NO_DATA if e else f for f, e in zip(frames, erased))


def _roundtrip(
    mono: np.ndarray,
    sr: int,
    codec: str,
    bitrate: Optional[str] = None,
    erased=None,
) -> np.ndarray:
    """Encode then decode a mono float32 array at ``sr`` through ``codec``.

    Returns float32 audio at the codec's native sample rate (caller resamples).
    Uses temp files because several of these encoders dislike piping.
    ``erased`` (see :func:`encode_decode`) rewrites the coded file in between.
    """
    enc, fmt, native, supports_br, dec = _CODEC_TABLE[codec]
    mono = np.ascontiguousarray(mono.astype(np.float32))
    if erased is not None:
        erased = np.asarray(erased, dtype=bool).reshape(-1)
        if not erased.any():
            erased = None
        elif codec not in _AMR_MAGIC:
            raise ValueError("coded-domain erasures are only available for amr_nb and amr_wb")

    with tempfile.TemporaryDirectory() as td:
        in_wav = os.path.join(td, "in.wav")
        enc_path = os.path.join(td, f"enc.{fmt}")
        out_wav = os.path.join(td, "out.wav")
        sf.write(in_wav, mono, sr, subtype="PCM_16")

        # --- encode ---
        enc_cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", in_wav]
        # G.726 needs explicit mono 8k and a code_size; map bitrate to it.
        enc_cmd += ["-ar", str(native), "-ac", "1", "-c:a", enc]
        if supports_br and bitrate:
            enc_cmd += ["-b:a", bitrate]
        enc_cmd += [enc_path]
        _run(enc_cmd)

        if erased is not None:
            with open(enc_path, "rb") as f:
                coded = f.read()
            with open(enc_path, "wb") as f:
                f.write(erase_amr_frames(coded, codec, erased))

        # --- decode back to wav ---
        # Force the reference decoder; ffmpeg would otherwise pick its own.
        dec_cmd = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-c:a", dec,
            "-i", enc_path, "-ar", str(native), "-ac", "1", "-c:a", "pcm_s16le", out_wav,
        ]
        _run(dec_cmd)

        y, _ = sf.read(out_wav, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    # A decode that stopped early would otherwise be zero-padded by the caller
    # and pass unnoticed. Allow one codec frame of algorithmic slack.
    expected = len(mono) * native / sr
    if len(y) < expected - 0.03 * native:
        raise RuntimeError(
            f"{codec}: decoded {len(y)} samples, expected about {int(expected)}"
        )
    return y


def encode_decode(
    mono: np.ndarray,
    sr: int,
    codec: str,
    bitrate: Optional[str] = None,
    erased=None,
) -> tuple[np.ndarray, int]:
    """Public entry point. Returns ``(audio_at_native_sr, native_sr)``.

    ``erased`` optionally flags 20 ms frames of the encoder input at the codec's
    native rate (frame ``k`` covers native samples ``[k*fl, (k+1)*fl)``; the
    decoded output carries it ``CODEC_DELAY`` samples later), ``fl = native_sr
    // 50``) to erase in the coded stream; only amr_nb and amr_wb support it.
    """
    if not have_ffmpeg():
        raise RuntimeError("ffmpeg not available")
    if codec not in _CODEC_TABLE:
        raise ValueError(f"Unknown ffmpeg codec {codec!r}")
    y = _roundtrip(mono, sr, codec, bitrate=bitrate, erased=erased)
    return y, native_sr(codec)

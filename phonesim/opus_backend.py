"""In-process Opus codec through the libopus shared library.

Encode -> decode round trips call libopus directly via ``ctypes``, one packet
per frame, so packet loss can be applied between the encoder and the decoder.
Lost frames are concealed by libopus itself: with in-band FEC enabled the
decoder recovers a lost frame from the low-bitrate redundant copy (LBRR) the
encoder puts in the *next* packet; without one it runs its own PLC.

The library is looked up lazily: ``PHONESIM_LIBOPUS`` (a path), then
``ctypes.util.find_library("opus")``, then the usual platform names. Importing
this module never fails; :func:`available` reports whether libopus loaded.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import os
from typing import Callable, Optional

import numpy as np


SAMPLE_RATES = (8000, 12000, 16000, 24000, 48000)
FRAME_MS = (5, 10, 20, 40, 60)

OPUS_APPLICATION_VOIP = 2048
OPUS_SET_BITRATE = 4002
OPUS_SET_VBR = 4006
OPUS_SET_COMPLEXITY = 4010
OPUS_SET_INBAND_FEC = 4012
OPUS_SET_PACKET_LOSS_PERC = 4014
OPUS_SET_DTX = 4016
OPUS_SET_SIGNAL = 4024
OPUS_GET_LOOKAHEAD = 4027
OPUS_SIGNAL_VOICE = 3001

# Upper bound libopus documents for one packet.
_MAX_PACKET = 4000

_MISSING = (
    "libopus shared library not found; install libopus (apt: libopus0, brew: opus) "
    "or set PHONESIM_LIBOPUS"
)

_c_int, _c_void_p = ctypes.c_int, ctypes.c_void_p
# function -> (argtypes, restype). The two ``*_ctl`` calls are variadic; the
# request value (or an output pointer) is passed as an extra ctypes argument.
_SIGNATURES = {
    "opus_get_version_string": ([], ctypes.c_char_p),
    "opus_strerror": ([_c_int], ctypes.c_char_p),
    "opus_encoder_create": ([_c_int, _c_int, _c_int, ctypes.POINTER(_c_int)], _c_void_p),
    "opus_encoder_ctl": ([_c_void_p, _c_int], _c_int),
    "opus_encode": ([_c_void_p, _c_void_p, _c_int, _c_void_p, _c_int], _c_int),
    "opus_encoder_destroy": ([_c_void_p], None),
    "opus_decoder_create": ([_c_int, _c_int, ctypes.POINTER(_c_int)], _c_void_p),
    "opus_decode": ([_c_void_p, _c_void_p, _c_int, _c_void_p, _c_int, _c_int], _c_int),
    "opus_decoder_destroy": ([_c_void_p], None),
}


@functools.lru_cache(maxsize=1)
def _load() -> Optional[ctypes.CDLL]:
    """Load libopus and declare the signatures used here; ``None`` when absent."""
    override = os.environ.get("PHONESIM_LIBOPUS")
    candidates = [override] if override else [
        ctypes.util.find_library("opus"),
        "libopus.so.0",
        "libopus.0.dylib",
        "/opt/homebrew/lib/libopus.0.dylib",
        "/usr/local/lib/libopus.0.dylib",
        "opus.dll",
    ]
    for name in candidates:
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
            for fn, (argtypes, restype) in _SIGNATURES.items():
                f = getattr(lib, fn)
                f.argtypes, f.restype = argtypes, restype
        except (OSError, AttributeError):
            continue
        probe = getattr(lib, "opus_packet_has_lbrr", None)     # optional: libopus 1.5 and later
        if probe is not None:
            probe.argtypes, probe.restype = [ctypes.c_char_p, ctypes.c_int32], _c_int
        return lib
    return None


def available() -> bool:
    return _load() is not None


def version() -> str:
    """libopus's own version string (e.g. ``"libopus 1.4"``), or ``"absent"``."""
    lib = _load()
    return lib.opus_get_version_string().decode("ascii", "replace") if lib else "absent"


def _lbrr_probe(lib) -> Optional[Callable[[bytes, int], int]]:
    """``opus_packet_has_lbrr(packet, len)``, 1 when the packet carries an LBRR copy; ``None`` before libopus 1.5."""
    return getattr(lib, "opus_packet_has_lbrr", None)


def lbrr_probe_available() -> bool:
    """Whether this libopus can tell an FEC recovery from PLC (``opus_packet_has_lbrr``, libopus 1.5 and later)."""
    lib = _load()
    return lib is not None and _lbrr_probe(lib) is not None


def _check(lib, code: int, what: str) -> int:
    """Raise on a negative libopus return code; pass the value through otherwise."""
    if code < 0:
        raise RuntimeError(f"{what}: {lib.opus_strerror(code).decode('ascii', 'replace')}")
    return code


def _encode(lib, frames: np.ndarray, sr: int, bitrate_bps: int, fec: bool,
            expected_loss_pct: int) -> tuple[list[bytes], int]:
    """Encode int16 ``[n, frame]`` into one packet per frame.

    Returns ``(packets, lookahead)``; the lookahead is the encoder+decoder
    delay in samples at ``sr``.
    """
    n, frame = frames.shape
    err = _c_int()
    enc = lib.opus_encoder_create(sr, 1, OPUS_APPLICATION_VOIP, ctypes.byref(err))
    if not enc:
        _check(lib, err.value, "opus_encoder_create")
    try:
        settings = (
            (OPUS_SET_BITRATE, int(bitrate_bps)),
            (OPUS_SET_COMPLEXITY, 10),
            (OPUS_SET_SIGNAL, OPUS_SIGNAL_VOICE),
            (OPUS_SET_VBR, 1),
            (OPUS_SET_DTX, 0),
            (OPUS_SET_INBAND_FEC, 1 if fec else 0),
            (OPUS_SET_PACKET_LOSS_PERC, int(expected_loss_pct)),
        )
        for request, value in settings:
            _check(lib, lib.opus_encoder_ctl(enc, request, _c_int(value)), f"opus_encoder_ctl({request})")
        lookahead = _c_int()
        _check(lib, lib.opus_encoder_ctl(enc, OPUS_GET_LOOKAHEAD, ctypes.byref(lookahead)),
               "opus_encoder_ctl(OPUS_GET_LOOKAHEAD)")

        buf = (ctypes.c_ubyte * _MAX_PACKET)()
        packets = []
        for i in range(n):
            size = _check(lib, lib.opus_encode(enc, frames[i].ctypes.data, frame, buf, _MAX_PACKET), "opus_encode")
            packets.append(ctypes.string_at(buf, size))
    finally:
        lib.opus_encoder_destroy(enc)
    return packets, lookahead.value


def _decode(lib, packets: list[bytes], lost: np.ndarray, sr: int, frame: int,
            fec: bool) -> tuple[np.ndarray, dict]:
    """Decode packets in order, concealing the frames flagged in ``lost``.

    A lost frame whose successor arrived is, with ``fec``, decoded from that
    successor with ``decode_fec=1`` (libopus uses the LBRR copy if present and
    falls back to PLC otherwise). Any other lost frame is a NULL decode, i.e.
    plain PLC. Returns int16 ``[n * frame]`` and the ``{"fec", "plc",
    "lbrr_known"}`` counts: with the LBRR probe a next-packet decode counts as
    ``fec`` only when that packet carries an LBRR copy, otherwise as ``plc``;
    without the probe (``lbrr_known`` False) every next-packet decode counts
    as ``fec``.
    """
    n = len(packets)
    out = np.zeros(n * frame, dtype=np.int16)
    probe = _lbrr_probe(lib)
    info = {"fec": 0, "plc": 0, "lbrr_known": probe is not None}
    err = _c_int()
    dec = lib.opus_decoder_create(sr, 1, ctypes.byref(err))
    if not dec:
        _check(lib, err.value, "opus_decoder_create")
    try:
        for i in range(n):
            if not lost[i]:
                data, use_fec = packets[i], 0
            elif fec and i + 1 < n and not lost[i + 1]:
                data, use_fec = packets[i + 1], 1
                has_lbrr = probe is None or probe(data, len(data)) == 1       # no probe: assume the copy is there
                info["fec" if has_lbrr else "plc"] += 1
            else:
                data, use_fec = None, 0
                info["plc"] += 1
            pcm = out[i * frame:].ctypes.data
            _check(lib, lib.opus_decode(dec, data, len(data) if data else 0, pcm, frame, use_fec), "opus_decode")
    finally:
        lib.opus_decoder_destroy(dec)
    return out, info


def _lost_frames(erased, n: int) -> np.ndarray:
    """Bool ``[n]`` loss mask from ``erased`` (``None`` = nothing lost)."""
    if erased is None:
        return np.zeros(n, dtype=bool)
    mask = np.asarray(erased, dtype=bool).ravel()[:n]
    return np.pad(mask, (0, n - len(mask)))


def encode_decode(
    mono: np.ndarray,
    sr: int,
    bitrate_bps: int,
    erased=None,
    frame_ms: int = 20,
    fec: bool = True,
    expected_loss_pct: int = 0,
) -> tuple[np.ndarray, dict]:
    """Round-trip mono float32 audio through libopus with optional packet loss.

    ``erased`` flags lost frames (``True`` = lost) at ``frame_ms`` granularity;
    a shorter mask is padded with ``False``, a longer one truncated. The output
    has the input's length and time alignment: the encoder's lookahead is
    dropped from the front and the tail is zero-padded. Returns
    ``(audio, {"fec": n, "plc": n, "lbrr_known": bool})``: ``fec`` counts
    lost frames recovered from the next packet's LBRR copy, ``plc`` those
    libopus concealed itself. ``lbrr_known`` is True when libopus (1.5 and
    later) could check each next packet for an LBRR copy; when False, ``fec``
    counts every lost frame decoded from the next packet with ``decode_fec=1``,
    whether libopus found a copy or fell back to PLC. ``expected_loss_pct`` is
    what a receiver report would tell the encoder, in whole percent; at 0 the
    encoder adds no LBRR.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(_MISSING)
    if sr not in SAMPLE_RATES:
        raise ValueError(f"opus: sample rate {sr} not in {SAMPLE_RATES}")
    if frame_ms not in FRAME_MS:
        raise ValueError(f"opus: frame_ms {frame_ms} not in {FRAME_MS}")
    frame = sr * frame_ms // 1000

    x = np.asarray(mono, dtype=np.float32).ravel()
    n = max(1, -(-len(x) // frame))
    lost = _lost_frames(erased, n)
    pcm = np.zeros(n * frame, dtype=np.int16)
    pcm[:len(x)] = np.clip(np.rint(x * 32768.0), -32768, 32767)

    packets, lookahead = _encode(lib, pcm.reshape(n, frame), sr, bitrate_bps, fec, expected_loss_pct)
    out, info = _decode(lib, packets, lost, sr, frame, fec)

    y = out[lookahead:lookahead + len(x)].astype(np.float32) / 32768.0
    return np.pad(y, (0, len(x) - len(y))), info

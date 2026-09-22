"""Codec stage: real codecs, with the frame erasures a receiver sees.

Backends:

* ``ffmpeg``  - G.722, G.726, Opus, AMR-NB and AMR-WB through the ffmpeg binary
  (AMR and Opus decoded by their reference decoders).
* ``native``  - G.711 mu-law/A-law with the exact segmented coder, in torch.
* ``libopus`` - Opus in-process through the libopus shared library; the only
  backend that can erase Opus frames.

The stage resamples to the codec's rate, runs the codec, removes its
algorithmic delay and resamples back. With ``erasure_rate > 0`` a share of the
20 ms frames is erased. AMR and Opus lose them in the coded stream: AMR frames
become NO_DATA frames, so the decoder runs its error concealment and carries
the error into the following frames; libopus decodes a lost frame from the
next packet's in-band FEC when it carries one and otherwise runs its PLC.
G.711, G.722 and G.726 are decoded in full and the erased frames are replaced
in the decoded PCM by the ITU-T G.711 Appendix I waveform substitution.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from phonesim.core import CodecUnavailableError, SimContext, Stage, fit_length, resolve_range
from phonesim import dsp, ffmpeg_backend, opus_backend, plc
from phonesim.stages.companding import CompandingStage
from phonesim.stages.packet import erasure_mask

_NATIVE_CODECS = {"g711_ulaw", "g711_alaw"}
_DECODER_CONCEALS = {"amr_nb", "amr_wb"}
FRAME_MS = 20


def _fit_mask(erased: np.ndarray, n_samples: int, fl: int) -> np.ndarray:
    """Pad the erasure mask to the frames the decoder actually produced (resamplers round differently)."""
    need = -(-n_samples // fl)
    return np.pad(erased[:need], (0, max(0, need - len(erased))))


class CodecStage(Stage):
    """Encode and decode through a real codec.

    Parameters
    ----------
    codec:
        One of :data:`phonesim.ffmpeg_backend.CODECS`.
    backend:
        ``"ffmpeg"`` (default), ``"native"`` (G.711 only, and its default) or
        ``"libopus"`` (Opus only).
    bitrate:
        Codec mode, validated against the codec's mode table (e.g. ``"12.2k"``
        for AMR-NB). Codecs without a mode reject a bitrate.
    erasure_rate:
        Share of 20 ms frames erased (lost or too late), drawn per call;
        scalar or ``(low, high)``. AMR and Opus lose them in the coded stream
        and conceal them with their decoders; G.711, G.722 and G.726 have them
        replaced in the decoded PCM by G.711 Appendix I. Erasures come in
        bursts: a two-state chain with ``burst_probability`` = P(stay erased).
    fec:
        libopus backend only (``fec=False`` with another backend is an error):
        the encoder is told the drawn erasure rate rounded to a whole percent
        (``OPUS_SET_PACKET_LOSS_PERC`` takes integers, and a WebRTC sender
        rounds its loss estimate the same way) and adds in-band FEC
        accordingly; the decoder uses it. A rate below 0.5 % rounds to 0, at
        which the encoder adds no FEC and every erased frame falls to PLC.
        The log line reports the percent the encoder was told.
    """

    def __init__(
        self,
        codec: str,
        backend: Optional[str] = None,
        bitrate: Optional[str] = None,
        name=None,
        *,
        erasure_rate=0.0,
        burst_probability: float = 0.3,
        fec: bool = True,
    ):
        super().__init__(name=name or f"Codec:{codec}")
        if codec not in ffmpeg_backend.CODECS:
            raise ValueError(f"Unknown codec {codec!r}; known: {sorted(ffmpeg_backend.CODECS)}")
        self.codec = codec
        self.backend = backend or ("native" if codec in _NATIVE_CODECS else "ffmpeg")
        if self.backend not in ("ffmpeg", "native", "libopus"):
            raise ValueError(f"Unknown backend {self.backend!r}")
        if self.backend == "native" and codec not in _NATIVE_CODECS:
            raise ValueError(f"backend='native' is only available for G.711, not {codec!r}")
        if self.backend == "libopus" and codec != "opus":
            raise ValueError(f"backend='libopus' is only available for opus, not {codec!r}")
        if self.backend == "libopus" and not opus_backend.available():
            raise CodecUnavailableError(
                'libopus shared library not found (apt: libopus0, brew: opus, or PHONESIM_LIBOPUS); '
                'see README, "Opus erasures need libopus"'
            )
        self.bitrate = ffmpeg_backend.validate_bitrate(codec, bitrate)
        self.erasure_range = resolve_range(erasure_rate)
        if not 0.0 <= self.erasure_range[0] <= self.erasure_range[1] < 1.0:
            raise ValueError(f"erasure_rate must lie in [0, 1), got {erasure_rate!r}")
        if self.erasure_range[1] > 0 and codec == "opus" and self.backend != "libopus":
            raise ValueError("opus erasures need backend='libopus'")
        self.burst_probability = float(burst_probability)
        if self.backend == "libopus":
            self.fec = bool(fec)
        elif not fec:
            raise ValueError("fec=False needs backend='libopus'; no other backend encodes in-band FEC")
        if self.backend == "native":
            self._compander = CompandingStage(law="mulaw" if codec.endswith("ulaw") else "alaw")

    # -- rates -------------------------------------------------------------
    def _codec_sr(self, cur_sr: int) -> int:
        if self.backend == "libopus":
            return cur_sr if cur_sr in opus_backend.SAMPLE_RATES else 48000
        return ffmpeg_backend.native_sr(self.codec)

    # -- per-channel round trips (numpy at the codec's rate) ----------------
    def _ffmpeg(self, mono: np.ndarray, cur_sr: int, erased) -> np.ndarray:
        if self.codec in _DECODER_CONCEALS:
            y, _ = ffmpeg_backend.encode_decode(mono, cur_sr, self.codec, bitrate=self.bitrate, erased=erased)
        else:
            y, nsr = ffmpeg_backend.encode_decode(mono, cur_sr, self.codec, bitrate=self.bitrate)
            if erased is not None:
                y = plc.conceal(y, nsr, _fit_mask(erased, len(y), nsr * FRAME_MS // 1000), nsr * FRAME_MS // 1000)
        lead = ffmpeg_backend.CODEC_DELAY.get(self.codec, 0)
        if lead:
            y = np.concatenate([y[lead:], np.zeros(lead, dtype=y.dtype)])
        return y

    def _libopus(self, mono: np.ndarray, sr: int, erased, loss_pct: int):
        return opus_backend.encode_decode(
            mono, sr, ffmpeg_backend._bps(self.bitrate), erased=erased, frame_ms=FRAME_MS,
            fec=self.fec, expected_loss_pct=loss_pct,
        )

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        cur_sr = ctx.sample_rate
        b, c, t = x.shape
        codec_sr = self._codec_sr(cur_sr)
        fl = codec_sr * FRAME_MS // 1000
        nframes = math.ceil((t * codec_sr // cur_sr) / fl)        # frames after resampling; backends pad or truncate

        rate = ctx.uniform(*self.erasure_range)
        loss_pct = int(round(100 * rate))                          # what the encoder is told; < 0.5 % becomes 0
        erased = None
        if rate > 0:
            erased = (erasure_mask(nframes, rate, self.burst_probability, ctx) == 0).numpy()

        info = {"fec": 0, "plc": 0, "lbrr_known": False}
        if self.backend == "native":
            y = dsp.resample(x, cur_sr, codec_sr)
            saved, ctx.sample_rate = ctx.sample_rate, codec_sr
            y = self._compander.process(y, ctx)
            ctx.sample_rate = saved
            if erased is not None:
                y_np = y.detach().to(torch.float32).cpu().numpy()
                mask = _fit_mask(erased, y_np.shape[-1], fl)
                for bi in range(b):
                    for ci in range(c):
                        y_np[bi, ci] = plc.conceal(y_np[bi, ci], codec_sr, mask, fl)
                y = torch.from_numpy(y_np).to(x.device, x.dtype)
            y = fit_length(dsp.resample(y, codec_sr, cur_sr), t)
        else:
            x_cpu = x.detach().to(torch.float32).cpu()
            if self.backend == "libopus" and codec_sr != cur_sr:
                x_cpu = dsp.resample(x_cpu, cur_sr, codec_sr)
            outs = []
            for bi in range(b):
                chans = []
                for ci in range(c):
                    mono = x_cpu[bi, ci].numpy()
                    if self.backend == "libopus":
                        yi, inf = self._libopus(mono, codec_sr, erased, loss_pct)
                        if bi == 0 and ci == 0:
                            info = inf                          # the mask is shared, so every channel splits alike
                    else:
                        yi = self._ffmpeg(mono, cur_sr, erased)
                    yt = torch.from_numpy(np.ascontiguousarray(yi)).view(1, 1, -1)
                    chans.append(fit_length(dsp.resample(yt, codec_sr, cur_sr), t))
                outs.append(torch.cat(chans, dim=1))
            y = torch.cat(outs, dim=0).to(x.device, x.dtype)

        ctx.log.append(self._log_line(erased, nframes, info, loss_pct))
        return y

    def _log_line(self, erased, nframes: int, info: dict, loss_pct: int) -> str:
        if self.backend == "libopus":
            note = opus_backend.version()
        elif self.backend == "ffmpeg":
            note = f"ffmpeg {ffmpeg_backend.ffmpeg_version()}, decoder {ffmpeg_backend.decoder_name(self.codec)}"
        else:
            note = "native G.711"
        line = f"{self.name}: {note}" + (f", bitrate={self.bitrate}" if self.bitrate else "")
        if erased is not None:
            n = int(erased.sum())
            if self.backend == "libopus":
                line += f", encoder told {loss_pct} % loss"
                if info["lbrr_known"]:
                    how = f"{info['fec']} FEC, {info['plc']} PLC"
                else:
                    how = (f"{info['fec']} from the next packet, FEC or PLC not distinguishable with {note}, "
                           f"{info['plc']} PLC")
            elif self.codec in _DECODER_CONCEALS:
                how = "decoder concealment"
            else:
                how = "G.711 App. I PLC"
            line += f", erased {n}/{nframes} frames ({how})"
        return line

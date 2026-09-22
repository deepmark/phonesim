"""ITU-T G.711 Appendix I packet loss concealment on decoded PCM.

Kapilow's waveform-substitution PLC, run offline over a whole signal. An
erased frame is filled from the received history: one pitch period is taken
from its tail and repeated, the buffer is widened to two and then three
periods, and the level fades by 20% per 10 ms so a gap longer than 60 ms
ends in silence. Every transition (into the erasure, between pitch buffers,
back to received audio) is a linear overlap-add.

The standard delays its output by ``POVERLAPMAX`` samples so it can blend
the synthetic signal into the quarter period before the erasure. Here the
whole signal is available, so that blend is written into the preceding
samples in place and the output has no delay.

The constants are the standard's values at 8 kHz; every sample count is
scaled by ``sr / 8000``. Arithmetic runs in float64 on the int16 scale so
``CORRMINPOWER`` keeps its meaning.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FRAMESZ = 80           # 10 ms sub-frame
PITCH_MIN = 40         # 200 Hz
PITCH_MAX = 120        # 66 Hz
NDEC = 2               # decimation of the coarse pitch search
CORRLEN = 160          # 20 ms correlation window
CORRMINPOWER = 250.0   # energy floor of the correlation normalisation
EOVERLAPINCR = 32      # end-of-erasure overlap growth per erased sub-frame, 4 ms
ATTENFAC = 0.2         # fade per 10 ms from the second erased sub-frame on


class _Params:
    """The standard's sample counts scaled from 8 kHz to ``sr``."""

    def __init__(self, sr: int):
        def n(v: int) -> int:
            return max(1, int(round(v * sr / 8000)))

        self.framesz = n(FRAMESZ)
        self.pitch_min = n(PITCH_MIN)
        self.pitch_max = n(PITCH_MAX)
        self.pitchdiff = self.pitch_max - self.pitch_min
        self.poverlapmax = self.pitch_max // 4
        self.historylen = 3 * self.pitch_max + self.poverlapmax
        self.ndec = n(NDEC)
        self.corrlen = n(CORRLEN)
        self.corrbuflen = self.corrlen + self.pitch_max
        self.eoverlapincr = n(EOVERLAPINCR)
        self.attenincr = ATTENFAC / self.framesz


def _ola(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cross-fade ``left`` out and ``right`` in with linear ramps."""
    n = len(left)
    if n == 0:
        return right
    w = np.arange(1, n + 1) / n
    return (1.0 - w) * left + w * right


class _Concealer:
    """Erasure state machine writing synthetic sub-frames into ``out`` in place."""

    def __init__(self, p: _Params, out: np.ndarray):
        self.p = p
        self.out = out
        self.erasecnt = 0      # erased sub-frames in the current erasure
        self.pitch = 0
        self.poverlap = 0      # quarter period, the OLA length
        self.poffset = 0       # read position in the circular pitch buffer
        self.pitchblen = 0     # pitch buffer length, 1..3 periods
        self.start = 0         # pitch buffer start inside ``pitchbuf``
        self.pitchbuf = np.zeros(p.historylen)
        self.lastq = np.zeros(0)  # unblended tail of the history

    def _history(self, pos: int) -> np.ndarray:
        """The last ``historylen`` output samples before ``pos``, zero-padded at the start."""
        h = self.p.historylen
        hist = np.zeros(h)
        n = min(h, pos)
        if n:
            hist[h - n:] = self.out[pos - n:pos]
        return hist

    def _findpitch(self, hist: np.ndarray) -> int:
        """Lag in ``PITCH_MIN..PITCH_MAX`` maximising the normalised correlation with the last 20 ms."""
        p = self.p
        h = len(hist)
        target = hist[h - p.corrlen:]
        # windows[j] is the 20 ms window lagged by pitch_max - j samples
        windows = sliding_window_view(hist[h - p.corrbuflen:], p.corrlen)[: p.pitchdiff + 1]

        def score(js: np.ndarray, step: int) -> np.ndarray:
            seg = windows[js, ::step]
            energy = np.einsum("ij,ij->i", seg, seg)
            return (seg @ target[::step]) / np.sqrt(np.maximum(energy, CORRMINPOWER))

        js = np.arange(0, p.pitchdiff + 1, p.ndec)
        s = score(js, p.ndec)
        best = int(js[len(s) - 1 - np.argmax(s[::-1])])   # coarse: last maximum (the reference's >=)
        js = np.arange(max(best - p.ndec + 1, 0), min(best + p.ndec - 1, p.pitchdiff) + 1)
        best = int(js[np.argmax(score(js, 1))])           # fine: first maximum (the reference's >)
        return p.pitch_max - best

    def _add_period(self) -> None:
        """Grow the pitch buffer by one period and blend its tail into the period before its start."""
        h = len(self.pitchbuf)
        self.pitchblen += self.pitch
        self.start = s = h - self.pitchblen
        n = self.poverlap
        self.pitchbuf[h - n:] = _ola(self.lastq, self.pitchbuf[s - n:s])

    def _synth(self, n: int) -> np.ndarray:
        """Read ``n`` samples from the circular pitch buffer, advancing the read position."""
        idx = (self.poffset + np.arange(n)) % self.pitchblen
        self.poffset = (self.poffset + n) % self.pitchblen
        return self.pitchbuf[self.start + idx]

    def _attenuate(self, frame: np.ndarray) -> None:
        """Continue the linear fade: ``ATTENFAC`` per sub-frame, starting after the first one."""
        g = 1.0 - (self.erasecnt - 1) * ATTENFAC - np.arange(len(frame)) * self.p.attenincr
        frame *= np.maximum(g, 0.0)

    def erased(self, pos: int) -> None:
        """Synthesise the erased sub-frame at ``pos``."""
        sub = self.p.framesz
        if self.erasecnt == 0:
            hist = self._history(pos)
            self.pitchbuf = hist
            self.pitch = self._findpitch(hist)
            self.poverlap = self.pitch // 4
            self.lastq = hist[len(hist) - self.poverlap:].copy()
            self.poffset = 0
            self.pitchblen = 0
            self._add_period()
            # The blended tail replaces the last quarter period of received audio.
            n = min(self.poverlap, pos)
            if n:
                self.out[pos - n:pos] = self.pitchbuf[len(hist) - n:]
            frame = self._synth(sub)
        elif self.erasecnt <= 2:
            save = self.poffset
            tail = self._synth(self.poverlap)
            self.poffset = save
            while self.poffset > self.pitch:
                self.poffset -= self.pitch
            self._add_period()
            frame = self._synth(sub)
            frame[: self.poverlap] = _ola(tail, frame[: self.poverlap])
            self._attenuate(frame)
        elif self.erasecnt > 5:
            frame = np.zeros(sub)
        else:
            frame = self._synth(sub)
            self._attenuate(frame)
        self.erasecnt += 1
        self.out[pos:pos + sub] = frame

    def received(self, pos: int) -> None:
        """Blend the synthetic continuation into the first received sub-frame after an erasure."""
        p = self.p
        olen = min(self.poverlap + (self.erasecnt - 1) * p.eoverlapincr, p.framesz)
        synth = self._synth(olen) * max(1.0 - (self.erasecnt - 1) * ATTENFAC, 0.0)
        seg = self.out[pos:pos + olen]
        seg[:] = _ola(synth, seg)
        self.erasecnt = 0


def conceal(y: np.ndarray, sr: int, erased: np.ndarray, frame_len: int) -> np.ndarray:
    """Conceal erased frames of a decoded mono signal.

    ``y`` is float32 ``[T]`` at ``sr``. ``erased[k]`` is True when frame ``k``,
    samples ``[k * frame_len, (k + 1) * frame_len)``, was not received; its
    samples in ``y`` are never read. ``frame_len`` must be a multiple of the
    10 ms sub-frame. Returns a new float32 ``[T]`` array.
    """
    y = np.array(y, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError("y must be mono [T]")
    if sr <= 0:
        raise ValueError("sr must be positive")
    p = _Params(sr)
    if frame_len <= 0 or frame_len % p.framesz:
        raise ValueError(f"frame_len must be a positive multiple of {p.framesz} samples (10 ms at {sr} Hz)")
    erased = np.asarray(erased, dtype=bool).ravel()
    t = len(y)
    nframes = -(-t // frame_len)
    if len(erased) < nframes:
        raise ValueError(f"erased covers {len(erased)} frames, the signal has {nframes}")
    erased = erased[:nframes]
    if not erased.any():
        return y

    sub = p.framesz
    out = np.zeros(nframes * frame_len)
    out[:t] = y * 32768.0
    state = _Concealer(p, out)
    for k, lost in enumerate(np.repeat(erased, frame_len // sub)):
        if lost:
            state.erased(k * sub)
        elif state.erasecnt:
            state.received(k * sub)
    return (out[:t] / 32768.0).astype(np.float32)

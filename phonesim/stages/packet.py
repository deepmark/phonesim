"""Packet-level transport degradations in the decoded signal.

These stages frame the signal at an RTP-like granularity (default 20 ms) and
replace lost frames with a repeat-and-fade of the last received one. Frame
erasures at a codec hop are :class:`CodecStage`'s ``erasure_rate``: AMR and
Opus conceal them with their own decoders, G.711/G.722/G.726 with the G.711
Appendix I algorithm on the decoded PCM. :func:`erasure_mask` is the loss
process both share.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from phonesim.core import SimContext, Stage, resolve_range


def _frame(x: torch.Tensor, frame_len: int):
    """Split ``[B, C, T]`` into ``[B, C, nframes, frame_len]`` with right-pad."""
    b, c, t = x.shape
    nframes = (t + frame_len - 1) // frame_len
    pad = nframes * frame_len - t
    xp = F.pad(x, (0, pad))
    return xp.reshape(b, c, nframes, frame_len), t


def _unframe(frames: torch.Tensor, t: int) -> torch.Tensor:
    b, c, nframes, fl = frames.shape
    return frames.reshape(b, c, nframes * fl)[..., :t]


def erasure_mask(nframes: int, rate: float, stay_bad: float, ctx: SimContext, device=None) -> torch.Tensor:
    """``[nframes]`` mask, 1 = received, 0 = erased, from a two-state chain.

    A frame is erased while the chain is in the bad state. ``stay_bad`` is
    P(stay bad); P(good -> bad) is set so the stationary erasure rate equals
    ``rate``. The frame on which the chain enters the bad state is still
    received; erasures start on the next frame, so at ``stay_bad`` 0 every
    erasure is a single frame.
    """
    stay_bad = min(max(stay_bad, 0.0), 0.95)
    recover = 1.0 - stay_bad
    p_good_bad = min(max(rate * recover / max(1e-6, 1.0 - rate), 0.0), 1.0)
    mask = torch.ones(nframes, device=device)
    bad = False
    for i in range(nframes):
        u = torch.rand((), generator=ctx.generator).item()
        if bad:
            mask[i] = 0.0
            bad = u >= recover
        elif u < p_good_bad:
            bad = True
    return mask


class PacketLossStage(Stage):
    """Drop RTP frames with a Gilbert-Elliott-style bursty loss model.

    Parameters
    ----------
    loss_rate:
        Average fraction of frames lost (scalar or range).
    burst_probability:
        Probability of *staying* in the loss state once a loss begins. 0
        gives isolated single-frame losses; higher values give longer gaps.
    frame_ms:
        RTP frame size in milliseconds (commonly 20).
    conceal:
        If ``True``, hand the loss mask to the PLC reconstruction below instead
        of leaving holes. If ``False``, lost frames are zeroed (silence).

    The loss mask is sampled from ``ctx`` (reproducible).
    """

    def __init__(
        self,
        loss_rate=0.02,
        burst_probability=0.2,
        frame_ms=20.0,
        conceal=True,
        name=None,
    ):
        super().__init__(name=name or "PacketLoss")
        self.loss_range = resolve_range(loss_rate)
        self.burst_probability = burst_probability
        self.frame_ms = frame_ms
        self.conceal = conceal

    def _sample_mask(self, nframes: int, loss_rate: float, ctx: SimContext, device):
        return erasure_mask(nframes, loss_rate, self.burst_probability, ctx, device)

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        loss_rate = ctx.uniform(*self.loss_range)
        if loss_rate <= 0:
            return x
        sr = ctx.sample_rate
        fl = max(1, int(sr * self.frame_ms / 1000.0))
        frames, t = _frame(x, fl)
        b, c, nframes, _ = frames.shape
        mask = self._sample_mask(nframes, loss_rate, ctx, x.device)  # [nframes]
        m = mask.view(1, 1, nframes, 1)

        if not self.conceal:
            out = frames * m
        else:
            out = _conceal(frames, mask)
        ctx.log.append(
            f"{self.name}: loss={loss_rate:.3f}, lost {int((mask==0).sum().item())}/{nframes} "
            f"frames, conceal={self.conceal}"
        )
        return _unframe(out, t)


def _conceal(frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Repeat-and-fade: each lost frame is the last received frame at 0.5^k."""
    b, c, nframes, fl = frames.shape
    out = frames.clone()
    last_good = torch.zeros(b, c, fl, device=frames.device, dtype=frames.dtype)
    have_good = False
    consec = 0
    for i in range(nframes):
        if mask[i] > 0.5:
            last_good = frames[:, :, i, :]
            have_good = True
            consec = 0
        else:
            consec += 1
            atten = 0.5**consec
            if have_good:
                out[:, :, i, :] = last_good * atten
            else:
                out[:, :, i, :] = frames[:, :, i, :] * 0.0
    return out


class JitterBufferStage(Stage):
    """Approximate jitter-buffer behaviour: late frames and concealment.

    A real jitter buffer reorders packets and, when a packet is too late,
    treats it as lost (triggering PLC) or, when the buffer underruns, repeats
    the last frame. We model two effects:

    * ``late_rate`` fraction of frames arrive too late -> concealed (repeat-fade).
    * occasional buffer underrun -> a single frame is duplicated, introducing a
      small time stretch / glitch.
    """

    def __init__(self, frame_ms=20.0, late_rate=0.01, underrun_rate=0.005, name=None):
        super().__init__(name=name or "JitterBuffer")
        self.frame_ms = frame_ms
        self.late_range = resolve_range(late_rate)
        self.underrun_range = resolve_range(underrun_rate)

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        sr = ctx.sample_rate
        fl = max(1, int(sr * self.frame_ms / 1000.0))
        frames, t = _frame(x, fl)
        b, c, nframes, _ = frames.shape

        late_rate = ctx.uniform(*self.late_range)
        # Late frames -> treat as lost and conceal.
        mask = torch.ones(nframes, device=x.device)
        if late_rate > 0:
            u = ctx.rand((nframes,), device=x.device)
            mask = (u >= late_rate).to(x.dtype)
        out = _conceal(frames, mask)

        # Underrun: duplicate a few frames (insert repeats), then trim back to T.
        underrun_rate = ctx.uniform(*self.underrun_range)
        if underrun_rate > 0 and ctx.maybe(min(1.0, underrun_rate * nframes)):
            dup_idx = int(torch.randint(nframes, (), generator=ctx.generator).item())
            out = torch.cat([out[:, :, : dup_idx + 1], out[:, :, dup_idx:dup_idx + 1], out[:, :, dup_idx + 1:]], dim=2)
            out = out[:, :, :nframes, :]

        ctx.log.append(
            f"{self.name}: late_rate={late_rate:.3f}, concealed "
            f"{int((mask==0).sum().item())} frames"
        )
        return _unframe(out, t)

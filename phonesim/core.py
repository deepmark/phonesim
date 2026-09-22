"""Core abstractions for the phone-call simulator.

This module defines the fundamental building blocks shared across all stages:

* :class:`AudioTensor` conventions (we operate on plain ``torch.Tensor`` objects
  but standardise their shape via helpers here).
* :class:`SimContext`, which carries the per-call random state and the current
  sample rate so stages can cooperate.
* The :class:`Stage` base class and :class:`Pipeline` container.

Shape convention
----------------
Internally every stage operates on a tensor of shape ``[B, C, T]`` (batch,
channels, time). The public API accepts ``[T]``, ``[B, T]`` and ``[B, C, T]``
as well as NumPy arrays, and restores the original rank on output. See
:func:`to_internal` / :func:`from_internal`.

Sample-rate convention
----------------------
Each stage knows the sample rate of the signal it receives and the sample rate
it emits. The :class:`SimContext` tracks the "current" sample rate as the signal
flows through the pipeline so that later stages (filters, codecs) can size their
parameters correctly even when an upstream :class:`ResampleStage` changed the rate.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

ArrayLike = Union[np.ndarray, torch.Tensor]


class CodecUnavailableError(RuntimeError):
    """This machine cannot run a codec the pipeline needs."""


# ----------------------------------------------------------------------------
# Shape handling
# ----------------------------------------------------------------------------
def to_internal(x: ArrayLike) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convert an input array/tensor to the internal ``[B, C, T]`` layout.

    Returns the standardised tensor plus a ``meta`` dict that records enough
    information for :func:`from_internal` to reconstruct the original type and
    rank exactly.
    """
    meta: dict[str, Any] = {}
    if isinstance(x, np.ndarray):
        meta["was_numpy"] = True
        meta["numpy_dtype"] = x.dtype
        t = torch.from_numpy(np.ascontiguousarray(x))
    elif isinstance(x, torch.Tensor):
        meta["was_numpy"] = False
        t = x
    else:
        raise TypeError(f"Unsupported input type: {type(x)!r}")

    meta["orig_dtype"] = t.dtype
    # Promote integer PCM to float in [-1, 1) so all math is well defined.
    if not torch.is_floating_point(t):
        info = torch.iinfo(t.dtype)
        scale = float(max(abs(info.min), info.max))
        t = t.to(torch.float32) / scale
        meta["was_integer"] = True
        meta["integer_scale"] = scale
    else:
        meta["was_integer"] = False
        # Work in float32 for numerical stability; restore later if needed.
        if t.dtype != torch.float32:
            meta["upcast_from"] = t.dtype
            t = t.to(torch.float32)

    meta["orig_ndim"] = t.dim()
    meta["orig_len"] = int(t.shape[-1])
    if t.dim() == 1:  # [T] -> [1, 1, T]
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 2:  # [B, T] -> [B, 1, T]
        t = t.unsqueeze(1)
    elif t.dim() == 3:  # [B, C, T]
        pass
    else:
        raise ValueError(
            f"Audio tensor must have rank 1, 2 or 3; got rank {t.dim()}"
        )
    return t.contiguous(), meta


def fit_length(t: torch.Tensor, length: int) -> torch.Tensor:
    """Right-pad with zeros or truncate ``t`` (``[..., T]``) to exactly ``length``."""
    cur = t.shape[-1]
    if cur == length:
        return t
    if cur > length:
        return t[..., :length]
    return F.pad(t, (0, length - cur))


def from_internal(t: torch.Tensor, meta: dict[str, Any]) -> ArrayLike:
    """Inverse of :func:`to_internal`: restore original rank and type."""
    orig_ndim = meta["orig_ndim"]
    if orig_ndim == 1:
        t = t[0, 0]
    elif orig_ndim == 2:
        t = t[:, 0]
    # rank 3 stays as-is

    if meta.get("was_integer", False):
        scale = meta["integer_scale"]
        t = (t.clamp(-1.0, 1.0) * scale).round()
        # Clamp to the dtype's representable range before casting: a value of
        # +1.0 maps to +scale (e.g. 32768 for int16), which overflows the
        # positive rail and would wrap to the most-negative value on cast.
        info = torch.iinfo(meta["orig_dtype"])
        t = t.clamp(float(info.min), float(info.max))
        t = t.to(meta["orig_dtype"])
    elif "upcast_from" in meta:
        t = t.to(meta["upcast_from"])

    if meta.get("was_numpy", False):
        arr = t.detach().cpu().numpy()
        if meta.get("was_integer", False):
            arr = arr.astype(meta["numpy_dtype"])
        return arr
    return t


# ----------------------------------------------------------------------------
# Simulation context
# ----------------------------------------------------------------------------
@dataclasses.dataclass
class SimContext:
    """Per-invocation state threaded through a pipeline.

    Parameters
    ----------
    sample_rate:
        The sample rate of the signal *as it currently is*. Stages update this
        when they resample.
    randomize:
        If ``True``, stages sample their parameters from configured ranges. If
        ``False``, they use the deterministic midpoint / nominal value.
    generator:
        A ``torch.Generator`` used for *all* sampling, so a fixed seed yields a
        reproducible call. Stages must draw randomness only from here.
    log:
        Stages append human-readable strings describing the parameters they drew,
        which is invaluable for debugging "what did this call actually do".
    """

    sample_rate: int
    randomize: bool = True
    generator: Optional[torch.Generator] = None
    log: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.generator is None:
            self.generator = torch.Generator()

    # -- sampling helpers -----------------------------------------------------
    def uniform(self, low: float, high: float) -> float:
        """Draw a float in ``[low, high]`` (or return the midpoint if not randomising)."""
        if low == high:
            return float(low)
        if not self.randomize:
            return float((low + high) / 2.0)
        u = torch.rand((), generator=self.generator).item()
        return float(low + u * (high - low))

    def maybe(self, probability: float) -> bool:
        """Return ``True`` with the given probability (always ``False`` if prob<=0)."""
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        if not self.randomize:
            # Deterministic mode: treat >0.5 as "on" so presets behave predictably.
            return probability > 0.5
        return torch.rand((), generator=self.generator).item() < probability

    def randn(self, shape, device=None, dtype=torch.float32) -> torch.Tensor:
        # The seeded generator lives on the CPU (so a fixed seed reproduces the
        # exact same draw regardless of the compute device). Sample on the CPU,
        # then move to the target device. This keeps the simulator device-
        # agnostic (CPU / CUDA) without per-device generator handling.
        t = torch.randn(shape, generator=self.generator, dtype=dtype)
        return t.to(device) if device is not None else t

    def rand(self, shape, device=None, dtype=torch.float32) -> torch.Tensor:
        t = torch.rand(shape, generator=self.generator, dtype=dtype)
        return t.to(device) if device is not None else t


def make_context(
    sample_rate: int,
    randomize: bool,
    seed: Optional[int],
) -> SimContext:
    """Build a :class:`SimContext` with a seeded ``torch.Generator``.

    Seeding rules:

    * ``seed`` given            -> reproducible for that seed.
    * ``seed`` ``None`` and not randomising -> fixed default seed (0), so
      stochastic stages such as additive noise produce the same realisation on
      every call (otherwise ``randomize=False`` would still vary run to run).
    * ``seed`` ``None`` and randomising -> nondeterministic.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    elif not randomize:
        gen.manual_seed(0)
    else:
        gen.seed()
    return SimContext(sample_rate=sample_rate, randomize=randomize, generator=gen)


def resolve_range(value: Union[float, int, tuple, list]) -> tuple[float, float]:
    """Normalise a scalar-or-pair config value into a ``(low, high)`` tuple."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Range must have 2 elements, got {value!r}")
        return float(value[0]), float(value[1])
    return float(value), float(value)


# ----------------------------------------------------------------------------
# Stage / Pipeline
# ----------------------------------------------------------------------------
class Stage(torch.nn.Module):
    """Base class for a single transformation in the phone-call chain.

    Subclasses implement :meth:`process`, which receives a ``[B, C, T]`` tensor
    and the :class:`SimContext`, and returns a ``[B, C, T]`` tensor. The context's
    ``sample_rate`` should be updated by any stage that resamples.

    Attributes
    ----------
    name:
        Short identifier used in logs and reports.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name or self.__class__.__name__

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    # Stages are nn.Modules but are always driven explicitly through ``process``
    # (a Pipeline iterates stages and threads a single SimContext). ``forward``
    # requires that same context, so a stage cannot be called as a bare
    # ``stage(x)`` without one.
    def forward(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        return self.process(x, ctx)


class Pipeline(torch.nn.Module):
    """An ordered composition of :class:`Stage` objects.

    The pipeline does *not* itself resample to a target output rate; compose an
    explicit :class:`~phonesim.stages.resample.ResampleStage` as the final stage
    (the profiles do this). :class:`~phonesim.simulator.PhoneCallSimulator` wraps a
    pipeline and adds output-rate enforcement plus the public NumPy/torch API.
    """

    def __init__(self, stages: list[Stage], name: str = "pipeline"):
        super().__init__()
        self.stages = torch.nn.ModuleList(stages)
        self.name = name

    def process(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        for stage in self.stages:
            x = stage.process(x, ctx)
        return x

    def forward(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        return self.process(x, ctx)

    def describe(self) -> str:
        lines = [f"Pipeline({self.name}) with {len(self.stages)} stages:"]
        for i, s in enumerate(self.stages):
            extra = ""
            backend = getattr(s, "backend", None)
            if backend is not None:
                extra = f"  backend={backend}"
                if getattr(s, "bitrate", None):
                    extra += f" bitrate={s.bitrate}"
                lo, hi = getattr(s, "erasure_range", (0.0, 0.0))
                if hi > 0:
                    extra += f" erasures={lo:g}-{hi:g}"
            lines.append(f"  [{i}] {s.name:<22}{extra}".rstrip())
        return "\n".join(lines)

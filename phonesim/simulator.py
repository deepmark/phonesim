"""High-level simulator API.

:class:`PhoneCallSimulator` wraps a profile in a callable that accepts and
returns NumPy arrays or torch tensors of rank 1/2/3, guarantees the output
sample rate, and supports seeding and batches. :class:`PhoneCallPipeline`
gives an explicit list of stages the same conveniences.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch

from phonesim.core import (
    ArrayLike,
    Pipeline,
    SimContext,
    Stage,
    fit_length,
    from_internal,
    make_context,
    to_internal,
)
from phonesim import profiles as P
from phonesim import config as _config
from phonesim import dsp

# Shortest accepted input: one 20 ms codec frame.
_MIN_INPUT_S = 0.02


def row_seeds(seed: Optional[int], n: int) -> list[Optional[int]]:
    """Seeds of the ``n`` rows of a ``per_example`` batch run with ``seed``.

    Row 0 uses ``seed`` itself, so a batch of one matches the unbatched call;
    the other rows draw theirs from a generator seeded with ``seed``, so
    adjacent seeds do not share rows.
    """
    if seed is None:
        return [None] * n
    g = torch.Generator().manual_seed(int(seed))
    return [int(seed)] + torch.randint(0, 2**31 - 1, (max(n - 1, 0),), generator=g).tolist()


class _SimulatorBase:
    """Shared machinery for the simulator and explicit-pipeline wrappers."""

    pipeline: Pipeline
    input_sample_rate: int
    output_sample_rate: int
    randomize: bool

    def _make_ctx(self, seed: Optional[int]) -> SimContext:
        return make_context(self.input_sample_rate, self.randomize, seed)

    def _enforce_output_sr(self, x: torch.Tensor, ctx: SimContext) -> torch.Tensor:
        if ctx.sample_rate != self.output_sample_rate:
            x = dsp.resample(x, ctx.sample_rate, self.output_sample_rate)
            ctx.sample_rate = self.output_sample_rate
        return x

    def run(
        self,
        x: ArrayLike,
        seed: Optional[int] = None,
        return_log: bool = False,
        per_example: bool = False,
    ) -> Union[ArrayLike, tuple[ArrayLike, list[str]]]:
        """Process one (possibly batched) input and return the same type/rank.

        With ``per_example=True`` every row of a batch is its own call
        (independent parameter draws, loss pattern, noise), seeded by
        :func:`row_seeds`. Otherwise the whole batch shares one call.
        """
        t, meta = to_internal(x)
        if t.numel() == 0:
            raise ValueError("input is empty")
        if t.shape[-1] < _MIN_INPUT_S * self.input_sample_rate:
            raise ValueError(f"input shorter than {int(_MIN_INPUT_S * 1000)} ms")
        if not torch.isfinite(t).all():
            raise ValueError("input contains NaN or infinite samples")
        if per_example:
            rows, logs = [], []
            for i, row_seed in enumerate(row_seeds(seed, t.shape[0])):
                ctx = self._make_ctx(row_seed)
                yi = self._enforce_output_sr(self.pipeline.process(t[i:i + 1], ctx), ctx)
                rows.append(yi); logs += [f"[{i}] {line}" for line in ctx.log]
            y = torch.cat(rows, dim=0)
            ctx = self._make_ctx(seed); ctx.log = logs; ctx.sample_rate = self.output_sample_rate
        else:
            ctx = self._make_ctx(seed)
            y = self.pipeline.process(t, ctx)
            y = self._enforce_output_sr(y, ctx)
        # Restore the exact output length the caller expects: the input length
        # scaled by the output/input sample-rate ratio. Chained resampling can
        # drift this by 1-2 samples, which would misalign a sample-accurate
        # downstream consumer.
        target_len = int(round(meta["orig_len"] * self.output_sample_rate / self.input_sample_rate))
        y = fit_length(y, target_len)
        out = from_internal(y, meta)
        if return_log:
            return out, list(ctx.log)
        return out

    __call__ = run

    def describe(self) -> str:
        return self.pipeline.describe()


class PhoneCallSimulator(_SimulatorBase):
    """Simulate phone-call degradation using a named profile.

    Parameters
    ----------
    input_sample_rate, output_sample_rate:
        Rates of the signal entering and leaving the simulator. The output is
        always resampled to ``output_sample_rate`` (default 24 kHz) so it matches
        the rate the downstream consumer expects.
    profile:
        Name of a registered profile (see :func:`phonesim.profiles.list_profiles`).
    randomize:
        If ``True`` (default), each call draws fresh random parameters. If
        ``False``, deterministic nominal parameters are used.
    profile_params:
        Extra keyword arguments forwarded to the profile builder.
    seed:
        If given, a default seed used when ``run`` is called without one.

    Examples
    --------
    >>> sim = PhoneCallSimulator(profile="voip_to_cellular_narrowband")
    >>> y = sim(x)                      # x: np.ndarray or torch.Tensor at 24 kHz
    >>> y = sim(x, seed=123)            # reproducible
    """

    def __init__(
        self,
        input_sample_rate: int = 24000,
        output_sample_rate: int = 24000,
        profile: str = "voip_to_cellular_narrowband",
        randomize: bool = True,
        profile_params: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        self.input_sample_rate = int(input_sample_rate)
        self.output_sample_rate = int(output_sample_rate)
        self.profile_name = profile
        self.randomize = randomize
        self.default_seed = seed
        self.pipeline = P.build_profile(
            profile,
            input_sr=self.input_sample_rate,
            output_sr=self.output_sample_rate,
            **(profile_params or {}),
        )

    def run(self, x, seed=None, return_log=False, per_example=False):
        if seed is None:
            seed = self.default_seed
        return super().run(x, seed=seed, return_log=return_log, per_example=per_example)

    __call__ = run

    @classmethod
    def from_config(cls, cfg: Any, randomize: bool = True):
        """Build a simulator from a YAML/JSON config (path, string, or dict)."""
        pipe, in_sr, out_sr = _config.pipeline_from_config(cfg)
        obj = cls.__new__(cls)
        obj.input_sample_rate = in_sr
        obj.output_sample_rate = out_sr
        obj.profile_name = getattr(pipe, "name", "from_config")
        obj.randomize = randomize
        obj.default_seed = None
        obj.pipeline = pipe
        return obj


class PhoneCallPipeline(_SimulatorBase):
    """Explicit composition of stages with the same conveniences as the simulator.

    >>> pipe = PhoneCallPipeline([
    ...     ResampleStage(24000, 16000),
    ...     BandlimitStage(low_hz=50, high_hz=7000),
    ...     CodecStage(codec="amr_wb"),
    ...     PacketLossStage(loss_rate=0.02, burst_probability=0.2),
    ...     ResampleStage(16000, 24000),
    ... ])
    >>> y = pipe(x)
    """

    def __init__(
        self,
        stages: list[Stage],
        input_sample_rate: int = 24000,
        output_sample_rate: int = 24000,
        randomize: bool = True,
        name: str = "explicit",
    ):
        self.input_sample_rate = int(input_sample_rate)
        self.output_sample_rate = int(output_sample_rate)
        self.randomize = randomize
        self.pipeline = Pipeline(stages, name=name)

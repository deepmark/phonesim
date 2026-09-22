"""Tests for the clock-drift stage: block-invariant output, logging, length and peak memory."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import phonesim
from phonesim.core import SimContext
from phonesim.stages import timing


def _ctx(sr: int, seed: int = 0, randomize: bool = False) -> SimContext:
    return SimContext(sample_rate=sr, randomize=randomize, generator=torch.Generator().manual_seed(seed))


def _signal(*shape: int, device: str = "cpu") -> torch.Tensor:
    return (torch.randn(*shape, generator=torch.Generator().manual_seed(1)) * 0.3).to(device)


def _drift(x: torch.Tensor, ppm, ctx: SimContext) -> torch.Tensor:
    return timing.ClockDriftStage(ppm=ppm).process(x, ctx)


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("sr", [8000, 16000])
@pytest.mark.parametrize("ppm", [40.0, -37.0])
def test_blocked_drift_equals_single_block(monkeypatch, channels, sr, ppm):
    x = _signal(1, channels, int(sr * 0.7137))          # not a multiple of the block
    monkeypatch.setattr(timing, "_BLOCK", 1000)
    y_blocked = _drift(x, ppm, _ctx(sr))
    monkeypatch.setattr(timing, "_BLOCK", 10 * x.shape[-1])
    y_single = _drift(x, ppm, _ctx(sr))
    assert torch.equal(y_blocked, y_single)


@pytest.mark.parametrize("seed", [1, 3])                 # draws +26 ppm and -50 ppm
@pytest.mark.parametrize("shape", [(1, 1, 16001), (3, 2, 16001)])
def test_drift_logs_drawn_ppm_and_preserves_length(seed, shape):
    x = _signal(*shape)
    ctx = _ctx(8000, seed, randomize=True)
    y = _drift(x, (-50.0, 50.0), ctx)
    ref = _ctx(8000, seed, randomize=True)
    ppm = ref.uniform(-50.0, 50.0)                        # the stage's single draw
    assert ctx.log[-1] == f"ClockDrift: {ppm:+.0f} ppm"
    assert y.shape == x.shape and y.dtype == x.dtype and not torch.equal(y, x)
    assert torch.equal(torch.rand(4, generator=ctx.generator), torch.rand(4, generator=ref.generator))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_blocked_drift_on_cuda(monkeypatch):
    x = _signal(2, 2, 16001, device="cuda")
    monkeypatch.setattr(timing, "_BLOCK", 1000)
    y_blocked = _drift(x, -37.0, _ctx(16000))
    monkeypatch.setattr(timing, "_BLOCK", 10 * x.shape[-1])
    y_single = _drift(x, -37.0, _ctx(16000))
    assert y_blocked.device == x.device and y_blocked.shape == x.shape
    assert torch.equal(y_blocked, y_single)


_PEAK_RSS_KB = """
import torch
from phonesim.core import SimContext
from phonesim.stages.timing import ClockDriftStage

def hwm_kb():
    with open("/proc/self/status") as f:
        return next(int(line.split()[1]) for line in f if line.startswith("VmHWM:"))

x = torch.randn(1, 1, 16000 * 60, generator=torch.Generator().manual_seed(0)) * 0.3
before = hwm_kb()
ClockDriftStage(ppm=40.0).process(x, SimContext(sample_rate=16000, randomize=False))
print(hwm_kb() - before)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc/self/status")
def test_drift_peak_memory_is_bounded():
    root = str(Path(phonesim.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH")) if p)}
    out = subprocess.run([sys.executable, "-c", _PEAK_RSS_KB], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout) < 300 * 1024, f"peak RSS grew by {int(out.stdout) // 1024} MB"

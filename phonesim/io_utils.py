"""Audio load/save helpers (thin wrappers over soundfile + the sinc resampler)."""

from __future__ import annotations

from typing import Optional

import warnings

import numpy as np
import soundfile as sf
import torch

from phonesim import dsp


class ClippingWarning(RuntimeWarning):
    """Samples outside [-1, 1] were clipped on write."""


def load_audio(path: str, sr: Optional[int] = 24000, mono: bool = True):
    """Load an audio file, optionally resampling to ``sr`` and downmixing to mono.

    Returns ``(waveform_float32_numpy, sr)`` where the waveform is ``[T]`` (mono)
    or ``[C, T]`` (multichannel).
    """
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
    data = data.T  # [C, T]
    if mono and data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)
    if sr is not None and sr != file_sr:
        t = torch.from_numpy(data).unsqueeze(0)  # [1, C, T]
        t = dsp.resample(t, file_sr, sr)
        data = t.squeeze(0).numpy()
        file_sr = sr
    if mono:
        data = data[0]
    return data, file_sr


def save_audio(path: str, x, sr: int = 24000, subtype: str = "PCM_16", normalize: bool = False) -> None:
    """Save ``x`` ([T], [C, T] or torch tensor) to ``path`` at ``sr``.

    By default the signal is written as-is, with any out-of-range samples clipped
    to ``[-1, 1]`` (matching what a real fixed-point sink does). This preserves
    the absolute level so a level-sensitive consumer reads back the same signal.
    Set ``normalize=True`` to instead peak-normalise the whole clip when it
    exceeds full scale (which silently changes the level).
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        data = x
    elif x.ndim == 2:
        data = x.T  # soundfile wants [T, C]
    else:
        raise ValueError(f"save_audio expects rank 1 or 2, got {x.ndim}")
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        if normalize:
            data = data / peak
        else:
            frac = float(np.mean(np.abs(data) > 1.0))
            warnings.warn(
                f"{path}: peak {peak:.2f}, {frac:.2%} of samples clipped to full scale",
                ClippingWarning,
                stacklevel=2,
            )
            data = np.clip(data, -1.0, 1.0)
    sf.write(path, data, sr, subtype=subtype)
